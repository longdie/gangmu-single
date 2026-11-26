"""The Agent's explicit document-tool allowlist, with durable call accounting.

No shell, arbitrary file reader, URL fetcher, config accessor or evaluation
annotation endpoint is exposed. Authentication stays with the host backend.
Dependency-review v2 exposes actual candidate IDs and requires an explicit
accepted/excluded/unresolved disposition with an explanation for each
retrieved source/type pair. Those records remain runtime sidecars; they are
not mathematical certification or fields of the E2 graph.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import time
from typing import Any

from gangmu.representation.model import digest

from .draft import Draft, DraftError
from .source import SourceProject
from .workflow import LEGACY_VERSION, SOURCE_OPERATIONS, WorkflowGate


class DocumentTools:
    READ_OPERATIONS = {'overview', 'read', 'structure', 'search', 'labels', 'symbols',
                       'find_text', 'dependency_candidates', 'candidates', 'read_section',
                       'explicit_inventory', 'paragraph_inventory', 'reference_ledger', 'read_paragraph'}
    DRAFT_OPERATIONS = {'upsert_node', 'upsert_nodes', 'upsert_relation', 'upsert_relations',
                        'upsert_context', 'remove_node', 'remove_relation', 'add_unresolved',
                        'review_candidates', 'coverage', 'checkpoint'}
    OPERATIONS = READ_OPERATIONS | DRAFT_OPERATIONS | {'validate', 'finalize', 'finish', 'status',
                                                     'list_nodes', 'get_node', 'list_relations', 'relation_candidates',
                                                     'review_bindings', 'review_relations', 'workflow_status',
                                                     'extraction_bindings', 'review_extraction', 'target_use_fragments',
                                                     'retention_status'}

    def __init__(self, project: SourceProject, draft: Draft, max_calls: int = 160):
        if type(max_calls) is not int or not 1 <= max_calls <= 2000:
            raise ValueError('max_calls must be an integer in 1..2000')
        if project.fingerprint != draft.project.fingerprint:
            raise ValueError('document tools and draft refer to different source projects')
        self.project, self.draft, self.max_calls = project, draft, max_calls
        self.trace_path = draft.work_dir / 'tool-trace.jsonl'
        if self.trace_path.is_symlink():
            raise ValueError('tool trace cannot be a symlink')
        self.calls = 0
        if self.trace_path.exists():
            content = self.trace_path.read_bytes()
            lines = content.splitlines(keepends=True)
            valid_length = 0
            for number, line in enumerate(lines):
                try:
                    row = json.loads(line)
                except (ValueError, UnicodeError):
                    # A crash can interrupt the final append. Earlier rows are
                    # authoritative; corruption in the middle is never ignored.
                    if number != len(lines) - 1 or line.endswith(b'\n'):
                        raise ValueError('corrupt document tool trace')
                    recovery = draft.work_dir / 'tool-trace-recovery.json'
                    if recovery.is_symlink():
                        raise ValueError('trace recovery record cannot be a symlink')
                    recovery.write_text(json.dumps({'discarded_tail_sha256': digest(line.hex()),
                                                    'retained_bytes': valid_length,
                                                    'reason': 'incomplete final trace append'}) + '\n', encoding='utf-8')
                    with self.trace_path.open('r+b') as stream:
                        stream.truncate(valid_length)
                        stream.flush()
                        os.fsync(stream.fileno())
                    break
                if row.get('event') == 'started':
                    self.calls += 1
                valid_length += len(line)
        self.identity = digest(project.fingerprint)
        metadata = draft.work_dir / 'tool-session.json'
        if metadata.is_symlink():
            raise ValueError('tool session metadata cannot be a symlink')
        if metadata.exists():
            previous = json.loads(metadata.read_text(encoding='utf-8'))
            if previous.get('source_identity') != self.identity:
                raise ValueError('tool session is bound to a different source project')
        else:
            metadata.write_text(json.dumps({'source_identity': self.identity, 'max_calls': max_calls}) + '\n', encoding='utf-8')

    def _log(self, row: dict):
        if self.trace_path.is_symlink():
            raise ValueError('tool trace cannot be a symlink')
        with self.trace_path.open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _execute(self, operation: str, args: dict) -> dict:
        if operation == 'retention_status':
            from .retention import retention_status
            return retention_status(self.project, self.draft, **args)
        if operation in SOURCE_OPERATIONS.values():
            kind = next(kind for kind, name in SOURCE_OPERATIONS.items() if name == operation)
            result = getattr(self.project, operation)(**args)
            return WorkflowGate(self.project, self.draft).record_source_page(kind, args, result, self.calls)
        if operation == 'extraction_bindings':
            return WorkflowGate(self.project, self.draft).extraction_bindings(**args)
        if operation == 'target_use_fragments':
            return WorkflowGate(self.project, self.draft).target_use_fragments(**args)
        if operation == 'review_extraction':
            return WorkflowGate(self.project, self.draft).record_extraction_reviews(args, self.calls)
        if operation == 'relation_candidates':
            gate = WorkflowGate(self.project, self.draft)
            if gate.version == LEGACY_VERSION:
                from .candidates_v2 import discover
            else:
                from .candidates import discover
            result = discover(self.project, self.draft, **args)
            return gate.record_candidate_page(args, result, self.calls)
        if operation == 'review_bindings':
            return WorkflowGate(self.project, self.draft).bindings(**args)
        if operation == 'review_relations':
            return WorkflowGate(self.project, self.draft).record_reviews(args, self.calls)
        if operation == 'workflow_status':
            if args:
                raise DraftError('workflow_status takes no arguments')
            return WorkflowGate(self.project, self.draft).inspect()
        if operation in self.READ_OPERATIONS:
            if operation == 'dependency_candidates' and 'node_id' in args:
                args = dict(args)
                node_id = args.pop('node_id')
                if node_id not in self.draft.data['nodes']:
                    raise DraftError('dependency candidate target node is not registered')
                args.setdefault('spans', self.draft.data['nodes'][node_id]['spans'])
            return getattr(self.project, operation)(**args)
        if operation in self.DRAFT_OPERATIONS:
            if operation in ('upsert_node', 'upsert_nodes'):
                values = [args] if operation == 'upsert_node' else args.get('nodes', args.get('items', []))
                if isinstance(values, list) and any(isinstance(value, dict) and value.get('semantics', {}) != {} for value in values):
                    raise DraftError('E4 semantics must be {}; no verified formal-language interface is implemented')
            return self.draft.apply(operation, args)
        if operation == 'validate':
            return self.draft.validate(**args)
        if operation in ('finalize', 'finish'):
            report = self.draft.finalize(**args)
            workflow = WorkflowGate(self.project, self.draft).inspect(independent=True)
            if not workflow['complete']:
                self.draft.data['finalized'] = False
                self.draft.checkpoint()
            return {**report, 'workflow': workflow, 'draft': self.draft.summary(),
                    'ready_for_host_publication': bool(report['structurally_valid'] and workflow['complete'])}
        if operation == 'status':
            return {'draft': self.draft.summary(), 'coverage': self.draft.coverage(),
                    'workflow': WorkflowGate(self.project, self.draft).inspect(),
                    'calls_used': self.calls, 'max_calls': self.max_calls,
                    'remaining_calls': max(0, self.max_calls - self.calls)}
        if operation == 'list_nodes':
            return self.draft.list_nodes(**args)
        if operation == 'get_node':
            node_id = args.get('id')
            if node_id not in self.draft.data['nodes']:
                raise DraftError('node is not registered')
            record = copy.deepcopy(self.draft.data['nodes'][node_id])
            record['original_text'] = [self.project.resolve_span(s) for s in record['spans']]
            record['context_record'] = copy.deepcopy(self.draft.data['contexts'][record['context']])
            chain, seen, context_id = [], set(), record['context']
            while context_id is not None:
                if context_id in seen or context_id not in self.draft.data['contexts']:
                    raise DraftError('node read requires complete acyclic context ancestry')
                seen.add(context_id)
                context = copy.deepcopy(self.draft.data['contexts'][context_id])
                chain.append({'record': context,
                              'original_text': [self.project.resolve_span(span) for span in context['spans']]})
                context_id = context.get('parent')
            record['context_chain'] = chain
            return record
        if operation == 'list_relations':
            offset, limit = args.get('offset', 0), args.get('limit', 100)
            if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 150:
                raise DraftError('invalid relation pagination')
            records = list(self.draft.data['relations'].values())
            if args.get('node_id') is not None:
                records = [r for r in records if args['node_id'] in (r['source'], r['target'])]
            return {'total': len(records), 'relations': records[offset:offset + limit],
                    'next_offset': offset + limit if offset + limit < len(records) else None}
        raise DraftError('unsupported document operation')

    def call(self, operation: str, args: dict | None = None) -> dict:
        """Return bounded failures as feedback; every started call is durable.

        Recovery does not reset the tool budget. ``finish`` prepares a host
        proposal only: it cannot publish the graph or bypass E3's commit gate.
        """
        args = {} if args is None else args
        if operation not in self.OPERATIONS:
            return {'ok': False, 'error_kind': 'forbidden_operation',
                    'error': 'operation is outside the document-tool allowlist'}
        if not isinstance(args, dict):
            return {'ok': False, 'error_kind': 'invalid_arguments', 'error': 'arguments must be a JSON object'}
        if self.calls >= self.max_calls:
            return {'ok': False, 'error_kind': 'tool_budget_exhausted',
                    'error': 'configured document tool-call budget is exhausted',
                    'draft': self.draft.summary(), 'remaining_calls': 0}
        # JSON validation rejects nonfinite/nonserializable tool arguments before
        # they can reach draft storage. No tool accepts an arbitrary host path.
        try:
            arguments_digest = digest(args)
        except (ValueError, TypeError) as exc:
            return {'ok': False, 'error_kind': 'invalid_arguments', 'error': str(exc)}
        self.calls += 1
        call_id = self.calls
        start = time.monotonic()
        self._log({'event': 'started', 'call_id': call_id, 'operation': operation,
                   'arguments': args, 'arguments_sha256': arguments_digest, 'source_identity': self.identity})
        try:
            result = self._execute(operation, args)
            response = {'ok': True, 'result': result, 'call_id': call_id,
                        'remaining_calls': max(0, self.max_calls - self.calls)}
        except (ValueError, TypeError, KeyError, AttributeError, OSError) as exc:
            response = {'ok': False, 'error_kind': 'tool_failure', 'error': f'{type(exc).__name__}: {exc}',
                        'call_id': call_id, 'remaining_calls': max(0, self.max_calls - self.calls)}
            if isinstance(exc, DraftError) and exc.details is not None:
                response['error_details'] = copy.deepcopy(exc.details)
        completed = {'event': 'completed', 'call_id': call_id, 'operation': operation,
                     'response': response, 'duration_seconds': round(time.monotonic() - start, 6),
                     'draft_revision': self.draft.data['revision']}
        if (self.draft.data.get('node_content_policy') == Draft.NODE_CONTENT_POLICY
                and operation in ('upsert_node', 'upsert_nodes', 'upsert_context', 'remove_node')
                and response.get('ok') is True):
            completed['node_content_bindings'] = WorkflowGate(
                self.project, self.draft).node_content_bindings()
        self._log(completed)
        return response
