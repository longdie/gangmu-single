"""Explicit, source-evaluation Responses calls with a shared campaign budget.

No provider request happens on import or configuration loading. Each request
uses a new conversation, no tools and no previous response ID. API credentials
are never retained. The shared ledger also gates the isolated extraction CLI.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import hashlib
import http.client
import json
import os
from pathlib import Path
import re
import time
import urllib.error
import urllib.request
from typing import Any

from gangmu.runtime.config import load_config
from gangmu.representation.model import digest
from .backend import CodexBackend, CodexSettings
from .budget import InvocationLedger, process_alive
from .cache import _atomic_json
from . import core_response_stream as _response_stream
from .core_response_stream import (ResponseCapture, StreamLimits, read_sse_response,
                                   read_bounded_body, validate_sse_transcript)

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / 'configs/e4-core-20261006.json'


class ConcurrentLimit(RuntimeError):
    pass


class ModelCallError(RuntimeError):
    """A retained failed request, never a successful evaluation verdict."""


def _strict_json(value):
    def invalid_constant(_):
        raise ValueError('provider JSON contains a nonfinite number')
    return json.loads(value, parse_constant=invalid_constant)


def _validated_campaign(config: dict) -> dict:
    if not isinstance(config, dict):
        raise ValueError('campaign configuration must be an object')
    if config.get('schema_version') != 'gangmu.e4.core_campaign.v1':
        raise ValueError('unknown core campaign configuration')
    budget = config['budget']
    for identifier in (config.get('campaign_id'), budget.get('budget_id')):
        if not isinstance(identifier, str) or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', identifier):
            raise ValueError('campaign identifiers must be safe file-name components')
    for name in ('max_total_invocations', 'max_observed_tokens', 'max_concurrency'):
        if type(budget[name]) is not int or budget[name] <= 0:
            raise ValueError('campaign limits must be positive integers')
    if budget['max_concurrency'] > 3:
        raise ValueError('user-authorized global concurrency is at most three')
    for reserve, total in (('heldout_reserved_invocations', 'max_total_invocations'),
                           ('heldout_reserved_observed_tokens', 'max_observed_tokens')):
        if type(budget[reserve]) is not int or not 0 <= budget[reserve] < budget[total]:
            raise ValueError('test reserve must be smaller than its total limit')
    evaluation = config['evaluation']
    if evaluation.get('response_transport', 'json') not in ('json', 'sse'):
        raise ValueError('unknown Responses transport')
    for name in ('timeout_seconds', 'max_output_tokens', 'max_attempts_per_request'):
        if type(evaluation[name]) is not int or evaluation[name] < 1:
            raise ValueError('evaluation call limits must be positive integers')
    if evaluation['max_output_tokens'] < 16:
        raise ValueError('Responses max_output_tokens must be at least sixteen')
    return config


def campaign_config(path: str | Path = CONFIG) -> dict:
    return _validated_campaign(json.loads(Path(path).read_bytes()))


class CoreCampaignLedger(InvocationLedger):
    def __init__(self, config: dict, allocation: str, *, root: Path = ROOT):
        if allocation not in ('development', 'heldout'):
            raise ValueError('unknown campaign allocation')
        self.config, self.allocation = _validated_campaign(config), allocation
        budget = config['budget']
        super().__init__(root / 'artifacts/invocation-budgets' / (budget['budget_id'] + '.json'),
                         budget['max_total_invocations'], budget['max_observed_tokens'])

    def _check_start(self, value: dict, public: dict) -> None:
        super()._check_start(value, public)
        budget = self.config['budget']
        live = [r for r in value['invocations'] if r.get('state') == 'started'
                and process_alive(r.get('owner'))]
        if len(live) >= budget['max_concurrency']:
            raise ConcurrentLimit('global campaign model-task concurrency reached')
        if self.allocation == 'development':
            tokens = sum(r.get('observed_tokens', 0) for r in value['invocations'])
            if (len(value['invocations']) >= budget['max_total_invocations'] - budget['heldout_reserved_invocations']
                    or tokens >= budget['max_observed_tokens'] - budget['heldout_reserved_observed_tokens']):
                raise RuntimeError('development start would consume registered test reserve')

    def start(self, public: dict) -> str:
        return super().start({**public, 'campaign_id': self.config['campaign_id'],
                              'campaign_allocation': self.allocation})


class CoreCampaignBackend(CodexBackend):
    """The ordinary isolated extractor, charged against the same global gate."""
    def __init__(self, project_root: Path, settings: CodexSettings, *, config: dict,
                 allocation: str = 'development'):
        super().__init__(project_root, settings)
        self.campaign, self.allocation = config, allocation

    def _invocation_ledger(self):
        budget = self.campaign['budget']
        if (self.settings.budget_id != budget['budget_id'] or
                self.settings.max_total_invocations != budget['max_total_invocations'] or
                self.settings.max_observed_tokens != budget['max_observed_tokens']):
            raise ValueError('extractor budget differs from registered campaign')
        return CoreCampaignLedger(self.campaign, self.allocation, root=self.root)


@contextmanager
def _request_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with os.fdopen(os.open(path / '.request.lock', os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600), 'a+b') as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def _output_object(response: dict) -> dict:
    if not isinstance(response, dict) or response.get('status') != 'completed':
        raise ModelCallError('provider response did not complete')
    if response.get('error') is not None or response.get('incomplete_details') is not None:
        raise ModelCallError('provider completed response contains error/incomplete details')
    output = response.get('output')
    if not isinstance(output, list):
        raise ModelCallError('provider output must be a list')
    messages, finals, legacy = [], [], []
    for item in output:
        if not isinstance(item, dict):
            raise ModelCallError('provider output item must be an object')
        if item.get('type') != 'message':
            continue
        if item.get('status') not in (None, 'completed') or not isinstance(item.get('content'), list):
            raise ModelCallError('provider message is incomplete or malformed')
        messages.append(item)
        labels = []
        for key in ('channel', 'phase'):
            raw = item.get(key)
            if raw is None:
                continue
            if raw in ('final', 'final_answer'):
                labels.append('final')
            elif raw in ('commentary', 'analysis'):
                labels.append(raw)
            else:
                raise ModelCallError('provider message has an unknown channel/phase')
        if len(set(labels)) > 1:
            raise ModelCallError('provider message channel and phase conflict')
        if item.get('role') not in (None, 'assistant'):
            raise ModelCallError('provider output message is not from the assistant')
        if labels:
            if labels[0] == 'final':
                if item.get('role') != 'assistant':
                    raise ModelCallError('provider final message must identify the assistant role')
                finals.append(item)
        else:
            legacy.append(item)
    if len(finals) > 1:
        raise ModelCallError('provider returned multiple final assistant messages')
    if finals and legacy:
        raise ModelCallError('provider returned ambiguous unclassified assistant messages')
    if finals:
        selected = finals[0]
    elif len(messages) == 1 and len(legacy) == 1:
        # Ordinary Responses messages historically omitted channel/phase.
        selected = legacy[0]
    else:
        raise ModelCallError('provider returned no unique final assistant message')
    pieces = []
    for part in selected['content']:
        if not isinstance(part, dict):
            raise ModelCallError('provider message content must be an object')
        if part.get('type') == 'refusal':
            raise ModelCallError('provider refused structured output')
        if part.get('type') == 'output_text':
            if not isinstance(part.get('text'), str):
                raise ModelCallError('provider output text must be a string')
            pieces.append(part['text'])
        else:
            raise ModelCallError('provider final message has unsupported content')
    if not pieces:
        raise ModelCallError('provider returned no structured output text')
    value = _strict_json(''.join(pieces))
    if not isinstance(value, dict):
        raise ModelCallError('provider JSON output must be an object')
    return value


def _usage(response: Any) -> tuple[list[dict], str]:
    """Malformed or missing usage is unknown, never silently known zero."""
    raw = response.get('usage') if isinstance(response, dict) else None
    if not isinstance(raw, dict) or any(type(raw.get(k)) is not int or raw[k] < 0
                                        for k in ('input_tokens', 'output_tokens')):
        return [], 'unknown_missing_or_invalid_primary_counts'
    result = {'input_tokens': raw['input_tokens'], 'output_tokens': raw['output_tokens'],
              'cached_input_tokens': 0, 'reasoning_tokens': 0}
    detail_status = 'complete'
    for container, name, output_name, maximum in (
            ('input_tokens_details', 'cached_tokens', 'cached_input_tokens', raw['input_tokens']),
            ('output_tokens_details', 'reasoning_tokens', 'reasoning_tokens', raw['output_tokens'])):
        details = raw.get(container)
        count = details.get(name) if isinstance(details, dict) else None
        if type(count) is int and 0 <= count <= maximum:
            result[output_name] = count
        else:
            detail_status = 'primary_counts_known_optional_details_unavailable'
    return [result], detail_status


class SourceResponsesClient:
    """Independent structured-output requests; source-semantic checks are external."""
    def __init__(self, *, config: dict | None = None, allocation: str = 'development',
                 root: Path = ROOT, runtime=None, transport=None,
                 retry_output_errors: bool = False):
        if type(retry_output_errors) is not bool:
            raise ValueError('Output retry policy must be an explicit boolean')
        self.retry_output_errors = retry_output_errors
        self.root = root.resolve()
        self.campaign = _validated_campaign(config) if config is not None else campaign_config()
        self.options = self.campaign['evaluation']
        self.response_transport = self.options.get('response_transport', 'json')
        runtime = runtime if runtime is not None else load_config(self.root, require_private=True)
        route = runtime.resolve_route('single_document')
        if (route.provider.alias != self.options['provider'] or route.model.name != self.options['model']
                or route.provider.protocol != 'openai_responses'):
            raise ValueError('configured route differs from the registered evaluation model')
        self.endpoint = route.provider.base_url.rstrip('/')
        if not self.endpoint.endswith('/responses'):
            self.endpoint += '/responses'
        self._key = runtime.credentials.get(route.provider.alias)
        if not self._key:
            raise ValueError('configured model credential is unavailable')
        self._redact = runtime.credentials.redact
        self._transport = transport if transport is not None else urllib.request.urlopen
        self.ledger = CoreCampaignLedger(self.campaign, allocation, root=self.root)
        self.model = {'provider': route.provider.alias, 'model': route.model.name,
            'protocol': 'openai_responses', 'effort': self.options['effort'],
            'max_output_tokens': self.options['max_output_tokens'],
            'client_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'provider_endpoint_sha256': hashlib.sha256(self.endpoint.encode()).hexdigest(),
            'provider_model_revision': 'vendor_alias_no_immutable_revision'}
        if retry_output_errors:
            self.model['technical_output_retry_policy'] = 'json_protocol_schema_within_original_cap_v1'
        # JSON keeps its historical binding shape. SSE explicitly binds its
        # parser/helper bytes and operational limits as well as client bytes.
        if self.response_transport == 'sse':
            self.model.update({'response_transport': 'sse',
                'stream_parser_version': _response_stream.VERSION,
                'stream_parser_code_sha256': hashlib.sha256(
                    Path(_response_stream.__file__).read_bytes()).hexdigest(),
                'semantic_inspection_code_sha256': hashlib.sha256(
                    Path(_response_stream.__file__).with_name('semantic_inspection.py').read_bytes()).hexdigest(),
                'stream_limits': vars(StreamLimits())})

    def _runtime_attempt_history(self, target: Path, binding: dict) -> list[Path]:
        """Refuse unpaired physical evidence before another paid attempt.

        A process can die after charging or writing a transcript but before its
        numbered receipt/result is durable. File counts alone must never reuse
        that number, overwrite its bytes or resample a possibly complete answer.
        This guard deliberately requires explicit reconciliation for that rare
        window; it does not synthesize a receipt or clear the unknown charge.
        """
        attempts = list(target.glob('attempt-*.json'))
        numbered = {}
        by_invocation = {}
        identity = digest({'body_sha256': binding['body_sha256'], 'provider': self.model['provider'],
                           'model': self.model['model'],
                           'endpoint_sha256': self.model.get('provider_endpoint_sha256')})
        rows = [row for row in self.ledger.read()['invocations']
                if row.get('request_identity') == identity or (
                    row.get('request_identity') is None
                    and row.get('run') == target.relative_to(self.root).as_posix()
                    and row.get('request_sha256') == binding['body_sha256'])]
        for path in attempts:
            if path.is_symlink() or not path.is_file():
                raise ValueError('retained runtime attempt must be a regular file')
            receipt = _strict_json(path.read_bytes())
            number = receipt.get('attempt')
            if (type(number) is not int or number < 1 or path.name != 'attempt-' + str(number) + '.json'
                    or number in numbered or receipt.get('binding') != binding
                    or receipt.get('invocation') in by_invocation):
                raise ValueError('retained runtime attempt numbering or binding differs')
            numbered[number] = path
            by_invocation[receipt.get('invocation')] = receipt
        if sorted(numbered) != list(range(1, len(numbered) + 1)):
            raise ModelCallError('orphan_response_requires_reconciliation')
        evidence = list(target.glob('transport-*.json')) + list(target.glob('response-*.json'))
        expected = {'transport-' + str(number) + '.json' for number in numbered}
        expected.update('response-' + str(number) + '.json' for number in numbered)
        if any(path.name not in expected or path.is_symlink() or not path.is_file() for path in evidence):
            raise ModelCallError('orphan_response_requires_reconciliation')
        if len(rows) != len(by_invocation) or {row['id'] for row in rows} != set(by_invocation):
            raise ModelCallError('orphan_response_requires_reconciliation')
        for row in rows:
            receipt = by_invocation[row['id']]
            if (row.get('state') != 'returned'
                    or row.get('run') != target.relative_to(self.root).as_posix()
                    or any(row.get(key) != receipt.get(key) for key in ('usage', 'error', 'returncode'))):
                raise ModelCallError('orphan_response_requires_reconciliation')
            # A complete physical answer without a committed result must not be
            # sent again. Its existing bytes remain available for reconciliation.
            if receipt.get('returncode') == 0:
                raise ModelCallError('orphan_response_requires_reconciliation')
            for prefix, key in (('transport-', 'transport_sha256'), ('response-', 'response_sha256')):
                path = target / (prefix + str(receipt['attempt']) + '.json')
                expected_digest = receipt.get(key)
                if expected_digest is not None:
                    if not path.is_file():
                        raise ModelCallError('orphan_response_requires_reconciliation')
                    if digest(_strict_json(path.read_bytes())) != expected_digest:
                        raise ValueError('retained runtime physical evidence digest differs')
                elif path.exists():
                    raise ModelCallError('orphan_response_requires_reconciliation')
        return [numbered[number] for number in sorted(numbered)]

    def recover_completed_response(self, instructions: str, payload: dict, *,
                                   output: str | Path, purpose: str,
                                   schema: dict | None = None,
                                   max_output_tokens: int | None = None) -> dict:
        """Reparse one retained completed response without another model start.

        The caller supplies the original instructions/payload/schema and
        operational model options. Historical client code is preserved in the
        original binding; the new parser is identified separately. Source and
        semantic/schema acceptance checks still belong to the host evaluator.
        Original request, failed result, response, attempt and ledger are read
        only. A separate recovery.json anchors their exact bytes and ledger row.
        """
        if not isinstance(instructions, str) or not instructions.strip() or not isinstance(payload, dict):
            raise ValueError('recovery requires original instructions and object payload')
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError('recovery purpose must be nonempty')
        cap = self.options['max_output_tokens'] if max_output_tokens is None else max_output_tokens
        if type(cap) is not int or cap < 16:
            raise ValueError('Responses max_output_tokens must be an integer at least sixteen')
        target = Path(output)
        if not target.is_absolute():
            target = self.root / target
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('recovery path cannot contain a symbolic link')
        target = target.resolve()
        if not target.is_relative_to(self.root) or target == self.root or not target.is_dir():
            raise ValueError('recovery requires an existing project-local request directory')
        expected_body = {'model': self.model['model'], 'store': False,
            'reasoning': {'effort': self.options['effort']}, 'max_output_tokens': cap,
            'input': [{'role': 'developer', 'content': instructions},
                      {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
        if self.response_transport == 'sse':
            expected_body['stream'] = True
        if schema is not None:
            expected_body['text'] = {'format': {'type': 'json_schema', 'name': 'source_evaluation',
                                              'schema': schema, 'strict': True}}
        if self._redact(expected_body) != expected_body or self._redact(purpose) != purpose:
            raise ValueError('recovery input must not contain resolved credentials')
        with _request_lock(target):
            def retained(name):
                path = target / name
                if path.is_symlink() or not path.is_file():
                    raise ValueError('recovery requires original regular retained artifacts')
                raw = path.read_bytes()
                value = _strict_json(raw)
                if not isinstance(value, dict):
                    raise ValueError('recovery retained artifact must be a JSON object')
                return value, {'path': path.relative_to(self.root).as_posix(),
                               'sha256': hashlib.sha256(raw).hexdigest()}

            request, request_ref = retained('request.json')
            result, result_ref = retained('result.json')
            binding = request.get('binding')
            if (not isinstance(binding, dict) or request.get('body') != expected_body
                    or request.get('purpose') != purpose or binding.get('body_sha256') != digest(expected_body)):
                raise ValueError('recovery original request/prompt/payload/schema binding mismatch')
            if self.response_transport == 'sse' and binding.get('request_bytes_sha256') != hashlib.sha256(
                    json.dumps(expected_body, ensure_ascii=False, allow_nan=False, sort_keys=True).encode()).hexdigest():
                raise ValueError('recovery SSE wire request binding mismatch')
            original_model = binding.get('model')
            historical_client = binding.get('client_code_sha256')
            expected_model = {**self.model, 'max_output_tokens': cap}
            if (not isinstance(original_model, dict) or not isinstance(historical_client, str)
                    or not re.fullmatch(r'[0-9a-f]{64}', historical_client)
                    or original_model.get('client_code_sha256') != historical_client
                    or {k: v for k, v in original_model.items() if k != 'client_code_sha256'} !=
                       {k: v for k, v in expected_model.items() if k != 'client_code_sha256'}):
                raise ValueError('recovery provider/model/effort/output-cap binding mismatch')
            if (result.get('status') != 'failed' or result.get('data') is not None
                    or result.get('returncode') != 1 or result.get('binding') != binding
                    or result.get('result_sha256') != digest({k: v for k, v in result.items() if k != 'result_sha256'})
                    or type(result.get('attempt')) is not int or result['attempt'] < 1):
                raise ValueError('recovery requires an intact original failed result')
            number = result['attempt']
            attempt, attempt_ref = retained('attempt-' + str(number) + '.json')
            response, response_ref = retained('response-' + str(number) + '.json')
            original_receipt = {k: v for k, v in result.items()
                                if k not in ('status', 'data', 'request_cache_hit', 'result_sha256')}
            if (attempt != original_receipt or attempt.get('response_sha256') != digest(response)
                    or type(attempt.get('http_status')) is not int or not 200 <= attempt['http_status'] < 300):
                raise ValueError('recovery original attempt/response hash or transport status mismatch')
            transport_reference = None
            if self.response_transport == 'sse':
                transcript, transport_reference = retained('transport-' + str(number) + '.json')
                if attempt.get('transport_sha256') != digest(transcript):
                    raise ValueError('recovery retained SSE transcript digest mismatch')
                validate_sse_transcript(transcript, response, self._redact)
                if transcript.get('raw_safety', {}).get('semantic_secret_redaction_changed') is not False:
                    raise ValueError('recovery must not reinterpret a redacted SSE response as original data')
            usage, usage_status = _usage(response)
            if usage != attempt.get('usage') or usage_status != attempt.get('usage_status'):
                raise ValueError('recovery retained response usage differs from original receipt')
            ledger_path = self.ledger.path
            if ledger_path.is_symlink() or not ledger_path.is_file():
                raise ValueError('recovery requires the original regular campaign ledger')
            ledger = _strict_json(ledger_path.read_bytes())
            if (not isinstance(ledger, dict) or not isinstance(ledger.get('invocations'), list)
                    or any(not isinstance(row, dict) for row in ledger['invocations'])):
                raise ValueError('recovery original campaign ledger is malformed')
            rows = [row for row in ledger['invocations'] if row.get('id') == attempt.get('invocation')]
            if len(rows) != 1:
                raise ValueError('recovery original invocation is absent or duplicated in ledger')
            row = rows[0]
            expected_row = {'state': 'returned', 'provider': self.model['provider'], 'model': self.model['model'],
                'purpose': purpose, 'backend': 'direct_responses', 'request_sha256': binding['body_sha256'],
                'run': target.relative_to(self.root).as_posix(), 'campaign_id': self.campaign['campaign_id'],
                'campaign_allocation': self.ledger.allocation, 'usage': usage, 'usage_known': bool(usage),
                'observed_tokens': sum(r['input_tokens'] + r['output_tokens'] for r in usage),
                'returncode': 1, 'error': attempt.get('error'), 'elapsed_seconds': attempt.get('elapsed_seconds')}
            if any(row.get(k) != v for k, v in expected_row.items()):
                raise ValueError('recovery original invocation/usage/model differs from ledger')
            data = _output_object(response)
            serialized = json.dumps(data, ensure_ascii=False, allow_nan=False)
            if self._redact(serialized) != serialized:
                raise ModelCallError('recovery output contains private credential-shaped content')
            recovery = {'schema_version': 'gangmu.e4.completed_response_recovery.v1', 'status': 'complete',
                'binding': binding, 'invocation': attempt['invocation'], 'attempt': number, 'data': data,
                'usage': usage, 'usage_status': usage_status, 'original_status': 'failed',
                'original_returncode': 1, 'ledger_new_invocations': 0,
                'parser_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'original_artifacts': {'request': request_ref, 'result': result_ref,
                                       'attempt': attempt_ref, 'response': response_ref},
                'ledger_reference': {'path': ledger_path.relative_to(self.root).as_posix(),
                                     'invocation_sha256': digest(row)}}
            if transport_reference is not None:
                recovery['original_artifacts']['transport'] = transport_reference
            recovery['recovery_sha256'] = digest(recovery)
            recovery_path = target / 'recovery.json'
            if recovery_path.exists() or recovery_path.is_symlink():
                previous, _ = retained('recovery.json')
                if previous != recovery:
                    raise ValueError('retained recovery artifact differs from verified original response')
            else:
                _atomic_json(recovery_path, recovery)
            return recovery

    def resume_failed_request(self, instructions: str, payload: dict, *,
                              output: str | Path, purpose: str,
                              expected_request_sha256: str,
                              schema: dict | None = None) -> dict:
        """Continue the remaining attempts of one explicitly named old request.

        Wire input, provider and all mathematical options stay unchanged. Only
        client/parser implementation bytes may migrate, recorded separately.
        Original request, results and attempts remain intact; new attempts use
        their original numbering and the same campaign ledger. This is not a
        new three-attempt task and cannot re-ask a retained complete answer.
        """
        if not isinstance(expected_request_sha256, str) or not re.fullmatch(r'[0-9a-f]{64}', expected_request_sha256):
            raise ValueError('Failed-request continuation requires its original raw request hash')
        if self.options['max_attempts_per_request'] > 3:
            raise ValueError('Historical failed-request continuation allows at most three total attempts')
        return self.call_json(instructions, payload, output=output, purpose=purpose,
            schema=schema, _resume_request_sha256=expected_request_sha256)

    def _resume_input_binding(self, target, request, body, purpose, binding, expected_hash):
        """Keep the historical wire/route identity, even for a cached retry."""
        request_path = target / 'request.json'
        old = request.get('binding', {})
        old_model, current_model = old.get('model'), binding['model']
        allowed = {'client_code_sha256', 'stream_parser_code_sha256', 'stream_parser_version'}
        if (request_path.is_symlink() or (target / 'result.json').is_symlink()
                or hashlib.sha256(request_path.read_bytes()).hexdigest() != expected_hash
                or request.get('body') != body or request.get('purpose') != purpose
                or old.get('body_sha256') != digest(body)
                or not isinstance(old_model, dict)
                or old.get('client_code_sha256') != old_model.get('client_code_sha256')
                or {k: v for k, v in old_model.items() if k not in allowed} !=
                   {k: v for k, v in current_model.items() if k not in allowed}
                or old.get('request_bytes_sha256') != binding.get('request_bytes_sha256')
                or body.get('stream') is not True):
            raise ValueError('Failed-request continuation changed original input or model routing')
        return old

    def _resume_binding(self, target, request, body, purpose, binding, expected_hash):
        """Authenticate a transport-only migration while holding the request lock."""
        old = self._resume_input_binding(target, request, body, purpose, binding, expected_hash)
        old_model = old['model']
        original_result = _strict_json((target / 'result.json').read_bytes())
        if (original_result.get('status') != 'failed' or original_result.get('data') is not None
                or original_result.get('returncode') != 1 or original_result.get('binding') != old
                or original_result.get('result_sha256') != digest({k: v for k, v in original_result.items()
                                                                  if k != 'result_sha256'})):
            raise ValueError('Failed-request continuation requires the intact original failed result')
        history, last_retryable = [], False
        for path in sorted(target.glob('attempt-*.json')):
            if path.is_symlink():
                raise ValueError('Failed-request attempt cannot be a symbolic link')
            attempt = _strict_json(path.read_bytes())
            if attempt.get('binding') != old or attempt.get('returncode') != 1:
                raise ValueError('Failed-request continuation cannot re-ask a delivered answer')
            if type(attempt.get('attempt')) is not int or path.name != 'attempt-' + str(attempt['attempt']) + '.json':
                raise ValueError('Failed-request attempt numbering changed')
            response_path = target / ('response-' + str(attempt['attempt']) + '.json')
            if response_path.is_symlink():
                raise ValueError('Failed-request response cannot be a symbolic link')
            if response_path.exists():
                response = _strict_json(response_path.read_bytes())
                if digest(response) != attempt.get('response_sha256'):
                    raise ValueError('Failed-request retained response changed')
                try:
                    _output_object(response)
                except (ModelCallError, ValueError, TypeError, KeyError):
                    pass
                else:
                    raise ValueError('Failed-request continuation found a complete retained answer')
            transcript_path = target / ('transport-' + str(attempt['attempt']) + '.json')
            if transcript_path.is_symlink():
                raise ValueError('Failed-request transport cannot be a symbolic link')
            transcript = _strict_json(transcript_path.read_bytes())
            if (digest(transcript) != attempt.get('transport_sha256')
                    or transcript.get('raw_redaction_changed') or not transcript.get('captured_bytes_complete')):
                raise ValueError('Failed-request original stream evidence changed')
            import base64
            raw = base64.b64decode(transcript['raw_base64'], validate=True)
            if hashlib.sha256(raw).hexdigest() != transcript.get('redacted_raw_sha256'):
                raise ValueError('Failed-request original stream bytes changed')
            parser = _response_stream.SemanticSSEParser(ResponseCapture('sse'))
            last_retryable = False
            try:
                parser.feed(raw)
                if transcript.get('eof_seen'):
                    parser.finish_eof()
            except (_response_stream.StreamTransientError, _response_stream.StreamTruncatedError):
                # A complete answer without a terminal frame is still not a
                # license to purchase another mathematical judgment.
                last_retryable = not parser._has_answer_output(parser.capture.response)
            except (ValueError, TypeError, KeyError):
                pass
            else:
                if parser.completed_response() is not None or parser._has_answer_output(parser.capture.response):
                    raise ValueError('Failed-request continuation found retained substantive output')
            if parser._has_answer_output(parser.capture.response):
                raise ValueError('Failed-request continuation found retained substantive output')
            history.append((path, attempt))
        numbers = sorted(a['attempt'] for _, a in history)
        if not numbers or numbers != list(range(1, len(history) + 1)):
            raise ValueError('Failed-request continuation requires consecutive original attempts')
        original_attempt = next((a for _, a in history if a['attempt'] == original_result.get('attempt')), None)
        if original_attempt != {k: v for k, v in original_result.items()
                                if k not in ('status', 'data', 'request_cache_hit', 'result_sha256')}:
            raise ValueError('Failed-request original result differs from its failed attempt')
        rows = [r for r in self.ledger.read()['invocations']
                if r.get('request_sha256') == old['body_sha256']
                and r.get('provider') == old_model['provider'] and r.get('model') == old_model['model']]
        if (len(rows) != len(history) or any(r.get('state') != 'returned' for r in rows)
                or {r['id'] for r in rows} != {a['invocation'] for _, a in history}):
            raise ValueError('Failed-request physical history differs from its shared ledger')
        by_id = {r['id']: r for r in rows}
        for _, attempt in history:
            row = by_id[attempt['invocation']]
            if (row.get('run') != target.relative_to(self.root).as_posix()
                    or row.get('error') != attempt.get('error')
                    or row.get('purpose') != purpose or row.get('returncode') != 1
                    or row.get('usage') != attempt.get('usage')
                    or row.get('request_sha256') != old['body_sha256']):
                raise ValueError('Failed-request attempt is not its actual charged request')
        if len(rows) >= self.options['max_attempts_per_request']:
            raise ModelCallError('retained_request_attempt_limit_exhausted')
        if not last_retryable:
            raise ValueError('Failed-request failure has no authenticated retryable stream error')
        execution = {'version': 'gangmu.responses.failed_request_transport_resume.v1',
            'original_request_sha256': expected_hash, 'original_binding': old,
            'executor_binding': binding, 'maximum_total_attempts': self.options['max_attempts_per_request'],
            'original_result_sha256': hashlib.sha256((target / 'result.json').read_bytes()).hexdigest(),
            'original_attempts': [{'path': p.name, 'sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
                                  for p, _ in history],
            'policy': 'same_wire_and_shared_ledger;transport_only;no_new_mathematical_task'}
        execution['sha256'] = digest(execution)
        execution_path = target / ('retry-execution-' + execution['sha256'] + '.json')
        if execution_path.exists():
            if _strict_json(execution_path.read_bytes()) != execution:
                raise ValueError('Failed-request execution registration changed')
        else:
            _atomic_json(execution_path, execution)
        return old, {'path': execution_path.relative_to(self.root).as_posix(),
                     'sha256': execution['sha256'], 'executor_binding': binding}

    def call_json(self, instructions: str, payload: dict, *, output: str | Path,
                  purpose: str, schema: dict | None = None,
                  max_output_tokens: int | None = None,
                  _resume_request_sha256: str | None = None) -> dict:
        if not isinstance(instructions, str) or not instructions.strip() or not isinstance(payload, dict):
            raise ValueError('independent evaluation requires instructions and object payload')
        if not isinstance(purpose, str) or not purpose.strip():
            raise ValueError('model call purpose must be nonempty')
        response_validator = None
        if self.retry_output_errors and schema is not None:
            # A malformed caller schema is a local configuration error and must
            # fail before a paid attempt. Only provider structural output may
            # consume the original request's remaining technical attempts.
            from jsonschema import Draft202012Validator
            Draft202012Validator.check_schema(schema)
            response_validator = Draft202012Validator(schema)
        effective_output = self.options['max_output_tokens'] if max_output_tokens is None else max_output_tokens
        if type(effective_output) is not int or effective_output < 16:
            raise ValueError('Responses max_output_tokens must be an integer at least sixteen')
        target = Path(output)
        if not target.is_absolute():
            target = self.root / target
        if any(p.is_symlink() for p in (target, *target.parents)):
            raise ValueError('model output path cannot contain a symbolic link')
        target = target.resolve()
        if not target.is_relative_to(self.root) or target == self.root:
            raise ValueError('model request output must stay inside project')
        target.mkdir(parents=True, exist_ok=True)
        body = {'model': self.model['model'], 'store': False,
            'reasoning': {'effort': self.options['effort']},
            'max_output_tokens': effective_output,
            'input': [{'role': 'developer', 'content': instructions},
                      {'role': 'user', 'content': json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
        if self.response_transport == 'sse':
            # Before request hash/retention: these are the bytes actually sent.
            body['stream'] = True
        if schema is not None:
            body['text'] = {'format': {'type': 'json_schema', 'name': 'source_evaluation',
                                     'schema': schema, 'strict': True}}
        if self._redact(body) != body or self._redact(purpose) != purpose:
            raise ValueError('evaluation input must not contain resolved credentials')
        binding = {'body_sha256': digest(body), 'model': {**self.model, 'max_output_tokens': effective_output},
                   'client_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
        wire_body = json.dumps(body, ensure_ascii=False, allow_nan=False,
                               sort_keys=self.response_transport == 'sse').encode()
        if self.response_transport == 'sse':
            binding['request_bytes_sha256'] = hashlib.sha256(wire_body).hexdigest()
        with _request_lock(target):
            request_path = target / 'request.json'
            execution = None
            if request_path.exists():
                old = json.loads(request_path.read_bytes())
                if _resume_request_sha256 is not None:
                    # A successful earlier resume is served without looking for
                    # another provider attempt, including after process restart.
                    replay_path = target / 'retry-result.json'
                    if replay_path.exists():
                        replay = _strict_json(replay_path.read_bytes())
                        if replay.get('status') == 'complete':
                            self._resume_input_binding(target, old, body, purpose, binding, _resume_request_sha256)
                            attempt = _strict_json((target / ('attempt-' + str(replay['attempt']) + '.json')).read_bytes())
                            if (replay.get('binding') != old.get('binding')
                                    or attempt != {k: v for k, v in replay.items()
                                                   if k not in ('status', 'data', 'request_cache_hit', 'result_sha256')}
                                    or replay.get('result_sha256') != digest({k: v for k, v in replay.items() if k != 'result_sha256'})):
                                raise ValueError('Retained resumed result binding changed')
                            retained = _strict_json((target / ('response-' + str(replay['attempt']) + '.json')).read_bytes())
                            transcript = _strict_json((target / ('transport-' + str(replay['attempt']) + '.json')).read_bytes())
                            validate_sse_transcript(transcript, retained, self._redact)
                            if (replay.get('transport_sha256') != digest(transcript)
                                    or replay.get('response_sha256') != digest(retained)
                                    or _output_object(retained) != replay.get('data')):
                                raise ValueError('Retained resumed answer differs from its actual response')
                            execution_ref = replay.get('transport_execution', {})
                            execution_path = self.root / execution_ref.get('path', '')
                            if not execution_path.is_relative_to(target) or execution_path.is_symlink():
                                raise ValueError('Retained retry execution reference changed')
                            registration = _strict_json(execution_path.read_bytes())
                            if (registration.get('sha256') != execution_ref.get('sha256')
                                    or registration.get('sha256') != digest({k: v for k, v in registration.items() if k != 'sha256'})
                                    or registration.get('original_binding') != old['binding']
                                    or registration.get('executor_binding') != execution_ref.get('executor_binding')
                                    or registration.get('original_request_sha256') != _resume_request_sha256
                                    or registration.get('original_result_sha256') != hashlib.sha256((target / 'result.json').read_bytes()).hexdigest()):
                                raise ValueError('Retained retry execution registration changed')
                            rows = [r for r in self.ledger.read()['invocations']
                                    if r.get('request_sha256') == old['binding']['body_sha256']
                                    and r.get('provider') == self.model['provider'] and r.get('model') == self.model['model']]
                            row = next((r for r in rows if r['id'] == replay['invocation']), {})
                            if (len(rows) != replay['attempt'] or len(rows) > self.options['max_attempts_per_request']
                                    or row.get('state') != 'returned' or row.get('returncode') != 0
                                    or row.get('run') != target.relative_to(self.root).as_posix()
                                    or row.get('purpose') != purpose or row.get('usage') != replay.get('usage')
                                    or row.get('error') != replay.get('error')):
                                raise ValueError('Retained retry differs from its actual shared ledger')
                            return {**replay, 'request_cache_hit': True}
                    binding, execution = self._resume_binding(target, old, body, purpose,
                                                              binding, _resume_request_sha256)
                elif old['binding'] != binding:
                    raise ValueError('request directory belongs to another content/model binding')
            else:
                if _resume_request_sha256 is not None:
                    raise ValueError('Failed-request continuation requires its original request')
                _atomic_json(request_path, {'binding': binding, 'purpose': purpose,
                                           'body': body, 'created_at': time.time()})
            result_path = target / ('retry-result.json' if execution is not None else 'result.json')
            if result_path.exists():
                result = json.loads(result_path.read_bytes())
                if self.retry_output_errors:
                    if (result.get('binding') != binding
                            or result.get('status') not in ('complete', 'failed')
                            or result.get('result_sha256') != digest({k: v for k, v in result.items()
                                                                    if k != 'result_sha256'})):
                        raise ValueError('retained runtime result binding or digest mismatch')
                    retained_attempt = _strict_json(
                        (target / ('attempt-' + str(result.get('attempt')) + '.json')).read_bytes())
                    if retained_attempt != {k: v for k, v in result.items()
                                            if k not in ('status', 'data', 'request_cache_hit', 'result_sha256')}:
                        raise ValueError('retained runtime result differs from its physical attempt')
                    if result['status'] == 'failed' and result.get('technical_retryable') is False:
                        raise ModelCallError('retained_request_failure_is_not_retryable')
                if result.get('status') == 'complete' and result.get('binding') == binding:
                    if result.get('result_sha256') != digest({k: v for k, v in result.items() if k != 'result_sha256'}):
                        raise ValueError('retained model result digest mismatch')
                    retained = json.loads((target / ('response-' + str(result['attempt']) + '.json')).read_bytes())
                    if self.response_transport == 'sse':
                        transcript = json.loads((target / ('transport-' + str(result['attempt']) + '.json')).read_bytes())
                        if result.get('transport_sha256') != digest(transcript):
                            raise ValueError('retained SSE transcript digest mismatch')
                        validate_sse_transcript(transcript, retained, self._redact)
                    if result.get('response_sha256') != digest(retained) or _output_object(retained) != result.get('data'):
                        raise ValueError('retained model result differs from its completed response')
                    return {**result, 'request_cache_hit': True}
            attempts = (self._runtime_attempt_history(target, binding) if self.retry_output_errors
                        else list(target.glob('attempt-*.json')))
            cap = self.options['max_attempts_per_request']
            while len(attempts) < cap:
                number = len(attempts) + 1
                start_wait = time.monotonic()
                while True:
                    try:
                        invocation = self.ledger.start({'provider': self.model['provider'],
                            'model': self.model['model'], 'purpose': purpose, 'backend': 'direct_responses',
                            'request_sha256': binding['body_sha256'],
                            'run': target.relative_to(self.root).as_posix()})
                        break
                    except ConcurrentLimit:
                        if time.monotonic() - start_wait > self.options['timeout_seconds']:
                            raise
                        time.sleep(1)
                started = time.monotonic()
                response, data, error, http_status, retryable = None, None, None, None, False
                output_phase = None
                provider_error = None
                capture = ResponseCapture('sse') if self.response_transport == 'sse' else None
                transport_sha256 = None
                retention_error = None
                try:
                    request = urllib.request.Request(self.endpoint,
                        data=wire_body,
                        headers={'Content-Type': 'application/json', 'Authorization': 'Bearer ' + self._key,
                                 'Accept': ('text/event-stream' if self.response_transport == 'sse'
                                            else 'application/json')},
                        method='POST')
                    with self._transport(request, timeout=self.options['timeout_seconds']) as stream:
                        http_status = stream.status
                        if not isinstance(http_status, int) or not 200 <= http_status < 300:
                            raise ModelCallError('provider transport returned non-success status')
                        output_phase = 'provider_protocol'
                        if self.response_transport == 'sse':
                            content_type = stream.headers.get('Content-Type', '').split(';', 1)[0].strip().lower()
                            if content_type != 'text/event-stream':
                                raw = read_bounded_body(stream, capture=capture,
                                    deadline=started + self.options['timeout_seconds'])
                                try:
                                    snapshot = _strict_json(raw)
                                    if isinstance(snapshot, dict):
                                        capture.response = snapshot
                                except (ValueError, TypeError):
                                    pass
                                raise ModelCallError('provider did not return the requested SSE content type')
                            response = read_sse_response(stream, capture=capture,
                                deadline=started + self.options['timeout_seconds'])
                        else:
                            # Original JSON path and response parser remain available.
                            response = _strict_json(stream.read())
                    data = _output_object(response)
                    output_phase = None
                    # Redact resolved secret values in the serialized data.
                    # Applying the config's key/URL sanitizer to source JSON
                    # would also erase legitimate mathematical fields named
                    # 'token' and source hyperlinks with section fragments.
                    if capture is not None:
                        _, secret_encoded = _response_stream.redact_semantic_value(data, self._redact)
                        if secret_encoded:
                            data = None
                            raise ModelCallError('provider output contains encoded private credential-shaped content')
                    serialized_data = json.dumps(data, ensure_ascii=False, allow_nan=False)
                    if self._redact(serialized_data) != serialized_data:
                        data = None
                        raise ModelCallError('provider output contains private credential-shaped content')
                    if response_validator is not None:
                        output_phase = 'provider_schema'
                        issue = next(response_validator.iter_errors(data), None)
                        if issue is not None:
                            raise ModelCallError('provider JSON output does not satisfy response schema at '
                                                 + issue.json_path)
                        output_phase = None
                except urllib.error.HTTPError as exc:
                    http_status = exc.code
                    error = 'provider_http_' + str(exc.code)
                    retryable = exc.code in (408, 429, 500, 502, 503, 504)
                    if capture is not None:
                        try:
                            raw = read_bounded_body(exc, capture=capture,
                                deadline=started + self.options['timeout_seconds'])
                            snapshot = _strict_json(raw)
                            if isinstance(snapshot, dict):
                                capture.response = snapshot
                        except (TimeoutError, ValueError, TypeError, urllib.error.URLError,
                                http.client.HTTPException, ConnectionError):
                            # Original status/retry classification remains authoritative.
                            pass
                except _response_stream.StreamTransientError as exc:
                    error, retryable = str(exc), True
                    provider_error = {'code': exc.error_code, 'type': exc.error_type,
                                      'message': exc.provider_message}
                except (TimeoutError, urllib.error.URLError, http.client.HTTPException, ConnectionError):
                    error, retryable = 'provider_transport_failure', True
                except (ModelCallError, ValueError, TypeError, KeyError) as exc:
                    error = self._redact(str(exc))
                    data = None
                    if self.retry_output_errors and output_phase in ('provider_protocol', 'provider_schema'):
                        # Valid mathematical negatives returned above are never
                        # sampled again. Refusals, credential findings and local
                        # resource bounds are also deliberately nonretryable.
                        retryable = (str(exc) != 'provider refused structured output'
                            and not (isinstance(exc, _response_stream.StreamProtocolError)
                                     and ('_limit' in str(exc) or 'control_unavailable' in str(exc))))
                        if response is not None and self._redact(response) != response:
                            retryable = False
                except BaseException as exc:
                    error = 'client_exception_' + type(exc).__name__
                    raise
                finally:
                    if capture is not None and response is None and capture.response is not None:
                        # Even a terminal failed/incomplete Response can report
                        # actual usage. No intermediate snapshot becomes usage.
                        response = capture.response
                    if capture is not None:
                        if data is None and capture.failure is None:
                            capture.failure = error or 'provider_response_not_accepted'
                        public_transport = capture.public(self._redact)
                        if ((data is not None or self.retry_output_errors)
                                and public_transport['raw_safety']['semantic_secret_redaction_changed']):
                            # A redacted provider artifact must not be promoted
                            # to mathematical success or a new success cache.
                            data, retryable = None, False
                            error = 'provider_response_contains_private_credential_shaped_content'
                        transport_sha256 = digest(public_transport)
                        try:
                            _atomic_json(target / ('transport-' + str(number) + '.json'), public_transport)
                        except OSError as exc:
                            # A paid Response is still charged with its actual
                            # usage if local transcript persistence fails. No
                            # successful result or paid retry may mask that loss.
                            retention_error = {'type': type(exc).__name__,
                                               'detail': _response_stream.redact_semantic_value(str(exc), self._redact)[0]}
                            data, retryable = None, False
                            error = 'client_transport_retention_failure_' + type(exc).__name__
                    usage, usage_status = _usage(response)
                    if capture is not None and response is not None:
                        semantic_public, _ = _response_stream.redact_semantic_value(response, self._redact)
                        public_response = self._redact(semantic_public)
                    else:
                        public_response = self._redact(response) if response is not None else None
                    receipt = {'invocation': invocation, 'attempt': number,
                        'elapsed_seconds': time.monotonic() - started, 'usage': usage,
                        'usage_status': usage_status,
                        'error': error, 'returncode': 0 if data is not None else 1,
                        'http_status': http_status, 'binding': binding,
                        **({'transport_sha256': transport_sha256} if capture is not None else {}),
                        **({'provider_error': self._redact(provider_error)} if provider_error is not None else {}),
                        **({'transport_execution': execution} if execution is not None else {}),
                        **({'transport_retention_error': retention_error} if retention_error is not None else {}),
                        'response_sha256': digest(public_response) if public_response is not None else None}
                    if self.retry_output_errors:
                        receipt['technical_retryable'] = retryable
                    self.ledger.finish(invocation, receipt)
                    attempt_path = target / ('attempt-' + str(number) + '.json')
                    _atomic_json(attempt_path, receipt)
                    if public_response is not None:
                        _atomic_json(target / ('response-' + str(number) + '.json'), public_response)
                    attempts.append(attempt_path)
                result = {**receipt, 'status': 'complete' if data is not None else 'failed',
                          'data': data, 'request_cache_hit': False}
                result['result_sha256'] = digest(result)
                _atomic_json(result_path, result)
                if data is not None:
                    return result
                if not retryable:
                    break
                if provider_error is not None and len(attempts) < cap:
                    time.sleep(min(2 ** (number - 1), 4))
            raise ModelCallError('model request failed; inspect retained redacted request outcome')
