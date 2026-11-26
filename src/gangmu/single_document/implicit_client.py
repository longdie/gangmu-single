"""Implicit-only structured calls and a separate, reservation-aware budget.

The historical HTTP/SSE parser is reused unchanged. This module never reads
paper sources; supplied frozen-node materials remain the caller's contract.
Reservations are conservative admission estimates, not a provider billing
guarantee. Missing reported usage never releases its reservation.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from gangmu.representation.model import digest
from .budget import process_alive, process_identity
from .cache import _atomic_json
from .core_client import ConcurrentLimit, SourceResponsesClient

ROOT = Path(__file__).resolve().parents[3]
VERSION = 'gangmu.e4.implicit-client.v1'
SCHEMA = 'gangmu.e4.implicit_campaign.v1'
PUBLIC_FIELDS = {'provider', 'model', 'purpose', 'backend', 'request_sha256', 'run'}
ALLOCATIONS = ('production', 'evaluation')


def _json(raw):
    def pairs(values):
        result = {}
        for key, value in values:
            if key in result:
                raise ValueError('Duplicate JSON key in implicit records')
            result[key] = value
        return result
    def constant(_):
        raise ValueError('Nonfinite JSON value in implicit records')
    return json.loads(raw, object_pairs_hook=pairs, parse_constant=constant)


def _local(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError('Implicit paths cannot traverse symbolic links')
    path = path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('Implicit artifacts must stay inside this project')
    return path


def validate_config(config):
    if not isinstance(config, dict) or config.get('schema_version') != SCHEMA:
        raise ValueError('Unknown implicit campaign configuration')
    value = copy.deepcopy(config)
    if not re.fullmatch(r'e4-implicit-[A-Za-z0-9_.-]+', value.get('campaign_id', '')):
        raise ValueError('Use a separate implicit campaign identity')
    if value.get('provider') != 'yxai' or value.get('model') != 'gpt-6.1-sol':
        raise ValueError('Implicit campaign must retain the authorized provider/model')
    budget = value.get('budget', {})
    bounds = {'max_counted_starts': 200, 'max_observed_tokens': 10_000_000,
              'max_concurrency': 3, 'max_attempts_per_task': 2}
    for name, ceiling in bounds.items():
        if type(budget.get(name)) is not int or not 1 <= budget[name] <= ceiling:
            raise ValueError('Implicit budget exceeds its finite authorization: ' + name)
    for key, default, total in (
            ('evaluation_reserved_starts', 40, 'max_counted_starts'),
            ('evaluation_reserved_tokens', 2_000_000, 'max_observed_tokens')):
        budget.setdefault(key, default)
        if type(budget[key]) is not int or not 0 < budget[key] < budget[total]:
            raise ValueError('An evaluation reserve must be positive and smaller than the total')
    value['budget'] = budget
    options = value.get('model_options', value.get('node_options', value.get('evaluation', {})))
    effort = options.get('reasoning_effort', options.get('effort', 'high'))
    if effort not in ('low', 'medium', 'high', 'xhigh'):
        raise ValueError('Unknown implicit reasoning effort')
    transport = options.get('response_transport', 'sse')
    if transport not in ('json', 'sse'):
        raise ValueError('Unknown implicit response transport')
    timeout, output = options.get('timeout_seconds', 600), options.get('max_output_tokens', 6000)
    if type(timeout) is not int or timeout <= 0 or type(output) is not int or output < 16:
        raise ValueError('Positive timeout and bounded output tokens are required')
    value['model_options'] = {'effort': effort, 'response_transport': transport,
        'timeout_seconds': timeout, 'max_output_tokens': output,
        'max_attempts_per_request': budget['max_attempts_per_task']}
    return value


class ImplicitLedger:
    """One atomic shared gate for production, independent review and retries."""
    def __init__(self, config, allocation='production', *, root=ROOT):
        self.root = Path(root).resolve()
        self.config = validate_config(config)
        if allocation not in ALLOCATIONS:
            raise ValueError('Unknown implicit allocation')
        self.allocation = allocation
        self.budget = self.config['budget']
        self.scope = {'schema_version': VERSION + '.budget',
            'campaign_id': self.config['campaign_id'], 'provider': self.config['provider'],
            'model': self.config['model'], 'budget': copy.deepcopy(self.budget),
            'reservation_policy': 'utf8_wire_bytes_plus_2maxoutput_plus4096',
            'billing_ceiling_guaranteed': False}
        self.path = _local(self.root, 'artifacts/invocation-budgets/' + self.config['campaign_id'] + '.json')
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._owned, self._owner_lock = set(), threading.Lock()

    def _validate(self, value):
        if (not isinstance(value, dict) or value.get('scope') != self.scope
                or not isinstance(value.get('invocations'), list)):
            raise ValueError('Implicit authorization/campaign cannot be replaced or reset')
        seen = set()
        for row in value['invocations']:
            if (not isinstance(row, dict)
                    or not isinstance(row.get('id'), str) or not re.fullmatch(r'[0-9a-f]{32}', row['id'])
                    or row['id'] in seen or row.get('state') not in ('started', 'returned')
                    or row.get('allocation') not in ALLOCATIONS
                    or row.get('provider') != self.config['provider']
                    or row.get('model') != self.config['model']
                    or type(row.get('reserved_tokens')) is not int or row['reserved_tokens'] <= 0
                    or type(row.get('usage_known')) is not bool
                    or type(row.get('observed_tokens')) is not int or row['observed_tokens'] < 0
                    or not isinstance(row.get('request_identity'), str)
                    or not re.fullmatch(r'[0-9a-f]{64}', row['request_identity'])):
                raise ValueError('Malformed or foreign implicit charge')
            owner = row.get('owner')
            if owner is not None and (not isinstance(owner, dict)
                    or type(owner.get('pid')) is not int or owner['pid'] <= 0
                    or not isinstance(owner.get('start_ticks'), str)
                    or not owner['start_ticks'].isdigit()):
                raise ValueError('Malformed implicit charge owner')
            if row['state'] == 'started':
                if row['usage_known'] or row['observed_tokens'] or 'usage' in row:
                    raise ValueError('Active implicit charge cannot release its reservation')
            else:
                usage = row.get('usage')
                self._validate_usage(usage)
                if (row['observed_tokens'] != sum(u['input_tokens'] + u['output_tokens'] for u in usage)
                        or row['usage_known'] and not usage):
                    raise ValueError('Implicit usage and observed charge differ')
            seen.add(row['id'])

    @staticmethod
    def _validate_usage(usage):
        if not isinstance(usage, list):
            raise ValueError('Reported usage must be an array')
        for item in usage:
            if not isinstance(item, dict) or any(type(item.get(k)) is not int or item[k] < 0
                    for k in ('input_tokens', 'output_tokens')):
                raise ValueError('Invalid reported primary usage')

    @contextmanager
    def _locked(self):
        descriptor = os.open(self.path.with_suffix('.lock'), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'a+b') as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            _local(self.root, self.path)
            value = _json(self.path.read_bytes()) if self.path.exists() else {
                'scope': self.scope, 'invocations': []}
            self._validate(value)
            yield value
            self._validate(value)
            _atomic_json(self.path, value)

    def _request(self, public):
        if not isinstance(public, dict) or set(public) != PUBLIC_FIELDS:
            raise ValueError('Caller cannot override ledger identity, ownership or usage')
        prefix = 'runs/dev/' + self.config['campaign_id'] + '/structured-call-cache/'
        run = public.get('run')
        suffix = run[len(prefix):] if isinstance(run, str) and run.startswith(prefix) else ''
        if (not re.fullmatch(r'[0-9a-f]{64}', suffix)
                or public['provider'] != self.config['provider'] or public['model'] != self.config['model']
                or public['backend'] != 'direct_responses'
                or not public['purpose'].startswith('implicit:' + self.allocation + ':')):
            raise ValueError('A registered implicit task and content-bound cache are required')
        request = _json(_local(self.root, run + '/request.json').read_bytes())
        body, binding = request['body'], request['binding']
        if (digest(body) != public['request_sha256'] or binding['body_sha256'] != public['request_sha256']
                or request.get('purpose') != public['purpose']
                or body.get('model') != self.config['model']):
            raise ValueError('Implicit request bytes/model/purpose differ from ledger admission')
        cap = body.get('max_output_tokens')
        if type(cap) is not int or cap < 16:
            raise ValueError('Implicit request requires bounded output')
        wire = json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=body.get('stream') is True).encode()
        if body.get('stream') is True and hashlib.sha256(wire).hexdigest() != binding.get('request_bytes_sha256'):
            raise ValueError('Implicit SSE physical request bytes differ')
        model = binding['model']
        if model.get('provider') != self.config['provider'] or model.get('model') != self.config['model']:
            raise ValueError('Implicit request binding has another provider/model')
        identity = digest({'body_sha256': binding['body_sha256'], 'provider': model['provider'],
            'model': model['model'], 'endpoint_sha256': model.get('provider_endpoint_sha256')})
        return identity, len(wire) + 2 * cap + 4096

    @staticmethod
    def _committed(row):
        return row['observed_tokens'] if row['usage_known'] else max(row['observed_tokens'], row['reserved_tokens'])

    def start(self, public):
        identity, reservation = self._request(public)
        with self._locked() as value:
            rows, budget = value['invocations'], self.budget
            if len(rows) >= budget['max_counted_starts']:
                raise RuntimeError('Implicit supplier-start budget exhausted')
            if sum(r['state'] == 'started' for r in rows) >= budget['max_concurrency']:
                raise ConcurrentLimit('Implicit shared concurrency reached; orphan charges remain reserved')
            if sum(r['request_identity'] == identity for r in rows) >= budget['max_attempts_per_task']:
                raise RuntimeError('Implicit content-bound request attempt cap reached')
            committed = sum(self._committed(r) for r in rows)
            if committed + reservation > budget['max_observed_tokens']:
                raise RuntimeError('Implicit active/unknown/observed token budget exhausted')
            production = [r for r in rows if r['allocation'] == 'production']
            if self.allocation == 'production' and (
                    len(production) >= budget['max_counted_starts'] - budget['evaluation_reserved_starts']
                    or sum(self._committed(r) for r in production) + reservation >
                    budget['max_observed_tokens'] - budget['evaluation_reserved_tokens']):
                raise RuntimeError('Implicit production cannot consume the independent evaluation reserve')
            identifier = uuid.uuid4().hex
            rows.append({**copy.deepcopy(public), 'id': identifier, 'state': 'started',
                'started_at': time.time(), 'owner': process_identity(os.getpid()),
                'allocation': self.allocation, 'request_identity': identity,
                'reserved_tokens': reservation, 'usage_known': False, 'observed_tokens': 0})
        with self._owner_lock:
            self._owned.add(identifier)
        return identifier

    def finish(self, identifier, result):
        with self._owner_lock:
            if identifier not in self._owned:
                raise ValueError('Only this ledger instance may finish its own charge once')
            usage = result.get('usage', [])
            self._validate_usage(usage)
            if 'usage_known' in result and type(result['usage_known']) is not bool:
                raise ValueError('Reported usage-known flag must be a boolean')
            elapsed = result.get('elapsed_seconds')
            if isinstance(elapsed, bool) or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed) or elapsed < 0:
                raise ValueError('Invalid elapsed invocation time')
            with self._locked() as value:
                row = next(r for r in value['invocations'] if r['id'] == identifier)
                if row['state'] != 'started':
                    raise ValueError('Implicit charge is already terminal')
                row.update(state='returned', finished_at=time.time(), usage=copy.deepcopy(usage),
                    usage_known=bool(usage) and result.get('usage_known', True) is True,
                    observed_tokens=sum(u['input_tokens'] + u['output_tokens'] for u in usage),
                    elapsed_seconds=elapsed, returncode=result.get('returncode'), error=result.get('error'))
            self._owned.remove(identifier)

    def snapshot(self):
        with self._locked() as value:
            rows = value['invocations']
            return {'campaign_id': self.config['campaign_id'], 'supplier_starts': len(rows),
                'invocations_started': len(rows), 'observed_tokens': sum(r.get('observed_tokens', 0) for r in rows),
                'committed_tokens': sum(self._committed(r) for r in rows),
                'active': sum(r['state'] == 'started' for r in rows),
                'active_live': sum(r['state'] == 'started' and process_alive(r.get('owner')) for r in rows),
                'dead_unfinished': sum(r['state'] == 'started' and not process_alive(r.get('owner')) for r in rows),
                'usage_unknown_invocations': sum(r['state'] == 'returned' and not r['usage_known'] for r in rows),
                'unknown_reserved_tokens': sum(r['reserved_tokens'] for r in rows
                    if r['state'] == 'returned' and not r['usage_known']),
                'active_reserved_tokens': sum(r['reserved_tokens'] for r in rows if r['state'] == 'started'),
                'remaining_starts': self.budget['max_counted_starts'] - len(rows),
                'remaining_admission_tokens': max(0, self.budget['max_observed_tokens'] - sum(self._committed(r) for r in rows)),
                'allocations': {a: {'starts': sum(r['allocation'] == a for r in rows),
                    'committed_tokens': sum(self._committed(r) for r in rows if r['allocation'] == a)} for a in ALLOCATIONS},
                'limits': copy.deepcopy(self.budget), 'billing_ceiling_guaranteed': False}


class ImplicitJudge:
    """StagedJudge-shaped API with separate production/evaluation conversations."""
    def __init__(self, config, *, root=ROOT, allocation='production', runtime=None, transport=None):
        self.config = validate_config(config)
        self.root = Path(root).resolve()
        self.ledger = ImplicitLedger(self.config, allocation, root=self.root)
        self.allocation = allocation
        b, options = self.config['budget'], self.config['model_options']
        adapted = {'schema_version': 'gangmu.e4.core_campaign.v1',
            'campaign_id': self.config['campaign_id'], 'budget': {
                'budget_id': self.config['campaign_id'], 'max_total_invocations': b['max_counted_starts'],
                'max_observed_tokens': b['max_observed_tokens'], 'max_concurrency': b['max_concurrency'],
                'heldout_reserved_invocations': 0, 'heldout_reserved_observed_tokens': 0},
            'evaluation': {'provider': self.config['provider'], 'model': self.config['model'], **options}}
        self.client = SourceResponsesClient(config=adapted, root=self.root, runtime=runtime, transport=transport)
        self.client.ledger = self.ledger
        self.client.model['implicit_client_code_sha256'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        self.cache_root = self.root / 'runs/dev' / self.config['campaign_id'] / 'structured-call-cache'
        self._receipts, self._lock = [], threading.Lock()

    def __call__(self, task_kind, instructions, payload):
        if not isinstance(task_kind, str) or not re.fullmatch(r'[A-Za-z0-9_.-]{1,128}', task_kind):
            raise ValueError('A bounded implicit structured-call purpose is required')
        purpose = 'implicit:' + self.allocation + ':' + task_kind
        key = digest({'purpose': purpose, 'instructions': instructions, 'payload': payload, 'model': self.client.model})
        receipt = self.client.call_json(instructions, payload, output=self.cache_root / key,
            purpose=purpose, schema=payload.get('response_schema'))
        with self._lock:
            self._receipts.append({'task_kind': task_kind, 'allocation': self.allocation,
                'request_key': key, 'cache_hit': receipt['request_cache_hit'],
                'invocation': receipt['invocation'], 'result_sha256': receipt['result_sha256']})
        return receipt['data']

    def receipts(self):
        with self._lock:
            return copy.deepcopy(self._receipts)
