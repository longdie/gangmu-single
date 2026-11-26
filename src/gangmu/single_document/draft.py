"""Recoverable Agent draft, kept outside the accepted E2 graph snapshot.

Only source-bound accepted records are converted. A proof-target association
is recorded as ``proves`` and never treated as a proof-correctness certificate.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
import re
from typing import Any

from gangmu.representation.graph import GraphIndex
from gangmu.representation.model import (Context, Dependency, Evidence, Graph, Identity,
                                        IdentityView, Proves, Unit, digest)
from gangmu.representation.sources import SourceError, _json

from .source import SourceProject


class DraftError(ValueError):
    def __init__(self, message, *, details=None):
        super().__init__(message)
        self.details = details


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DraftError(f'{name} must be nonempty text')
    return value


def _identifier(value: Any, name: str = 'id') -> str:
    value = _text(value, name)
    if len(value) > 180 or not re.fullmatch(r'[A-Za-z0-9_.:/-]+', value):
        raise DraftError(f'{name} must be a short ASCII record identifier')
    return value


class Draft:
    VERSION = 'gangmu.e4.draft.v1'
    USE_WITNESS_BINDING = 'target_use_reference_v1'
    EXPLICIT_CONTEXT_SOURCE_POLICY = 'explicit_context_sources_v1'
    NODE_CONTENT_POLICY = 'current_node_readback_ascii_word_boundaries_v1'

    def __init__(self, project: SourceProject, work_dir: str | Path, *,
                 require_use_witnesses: bool = False, require_explicit_context_sources: bool = False,
                 require_node_content_checks: bool = False):
        if type(require_use_witnesses) is not bool:
            raise DraftError('require_use_witnesses must be boolean')
        if type(require_explicit_context_sources) is not bool:
            raise DraftError('require_explicit_context_sources must be boolean')
        if type(require_node_content_checks) is not bool:
            raise DraftError('require_node_content_checks must be boolean')
        self.project, self.work_dir = project, Path(work_dir).resolve()
        self.work_dir.mkdir(parents=True, exist_ok=True)
        self.path = self.work_dir / 'draft.json'
        if self.path.is_symlink():
            raise DraftError('draft checkpoint cannot be a symlink')
        self.data = {'schema_version': self.VERSION, 'source_fingerprint': project.fingerprint,
                     'revision': 0, 'nodes': {}, 'relations': {}, 'contexts': {},
                     'unresolved': [], 'candidate_reviews': {}, 'coverage': {}, 'finalized': False}
        if require_use_witnesses:
            self.data['use_witness_binding'] = self.USE_WITNESS_BINDING
        if require_explicit_context_sources:
            self.data['context_source_policy'] = self.EXPLICIT_CONTEXT_SOURCE_POLICY
        if require_node_content_checks:
            self.data['node_content_policy'] = self.NODE_CONTENT_POLICY
        if self.path.exists():
            self.data = _json(self.path.read_bytes())
            if self.data.get('schema_version') != self.VERSION:
                raise DraftError('unsupported draft checkpoint version')
            if self.data.get('source_fingerprint') != project.fingerprint:
                raise DraftError('draft source fingerprint differs; use a separate work directory')
            if not all(isinstance(self.data.get(k), dict) for k in ('nodes', 'relations', 'contexts', 'candidate_reviews', 'coverage')):
                raise DraftError('invalid checkpoint record collections')
        marker = self.data.get('use_witness_binding')
        if marker not in (None, self.USE_WITNESS_BINDING):
            raise DraftError('unsupported dependency use witness binding')
        if require_use_witnesses and marker != self.USE_WITNESS_BINDING:
            raise DraftError('legacy draft cannot be relabelled as witness-bound production')
        self.validate_context_source_policy(required=require_explicit_context_sources)
        self.validate_node_content_policy(required=require_node_content_checks)

    def validate_node_content_policy(self, *, required: bool | None = None) -> bool:
        if required is not None and type(required) is not bool:
            raise DraftError('require_node_content_checks must be boolean')
        if not isinstance(self.data, dict):
            raise DraftError('node content policy requires an object draft')
        marker = self.data.get('node_content_policy')
        if marker not in (None, self.NODE_CONTENT_POLICY):
            raise DraftError('unsupported node content policy')
        active = marker == self.NODE_CONTENT_POLICY
        if required is True and not active:
            raise DraftError('legacy draft cannot be relabelled with node content checks')
        if required is False and active:
            raise DraftError('node content policy cannot be disabled for a checked draft')
        if active:
            if not isinstance(self.data.get('nodes'), dict):
                raise DraftError('checked node record collection must be an object')
            for node in self.data['nodes'].values():
                if not isinstance(node, dict):
                    raise DraftError('checked node record must be an object')
                if node.get('judgment') == 'accepted':
                    self._require_word_boundaries(self._spans(node.get('spans')))
        return active

    def _require_word_boundaries(self, spans: list[dict]) -> None:
        """Reject ASCII-word cuts only; do not certify sentence completeness.

        Adjacent slices in one occurrence may legitimately divide a word in a
        paginated read. Check the boundaries of their union without changing
        stored spans. Different occurrences never fill one another's gaps.
        """
        groups = {}
        for span in spans:
            groups.setdefault((span['file'], span.get('occurrence')), []).append(
                (span['char_start'], span['char_end']))
        failures = []
        for (file, occurrence), intervals in groups.items():
            merged = []
            for start, end in sorted(intervals):
                if merged and start <= merged[-1][1]:
                    merged[-1] = (merged[-1][0], max(end, merged[-1][1]))
                else:
                    merged.append((start, end))
            text = self.project._read_text(file)
            for start, end in merged:
                for boundary, position in (('start', start), ('end', end)):
                    if (0 < position < len(text)
                            and all(c.isascii() and c.isalpha()
                                    for c in text[position - 1:position + 1])):
                        failures.append({'file': file, 'occurrence': occurrence,
                                         'boundary': boundary, 'char_position': position})
        if failures:
            raise DraftError('accepted node cuts an original ASCII word or TeX command name; '
                             'read the complete source unit and resubmit exact boundaries',
                             details={'code': 'source_word_cut', 'boundaries': failures,
                                      'mathematical_completeness': 'not_certified'})

    def validate_context_source_policy(self, *, required: bool | None = None) -> bool:
        """Validate binding mechanics only; no mathematical sufficiency is certified."""
        if required is not None and type(required) is not bool:
            raise DraftError('require_explicit_context_sources must be boolean')
        if not isinstance(self.data, dict):
            raise DraftError('context source policy requires an object draft')
        marker = self.data.get('context_source_policy')
        if marker not in (None, self.EXPLICIT_CONTEXT_SOURCE_POLICY):
            raise DraftError('unsupported explicit context source policy')
        strict = marker == self.EXPLICIT_CONTEXT_SOURCE_POLICY
        if required is True and not strict:
            raise DraftError('legacy draft cannot be relabelled with explicit context sources')
        if required is False and strict:
            raise DraftError('explicit context source policy cannot be disabled for a strict draft')
        if strict:
            if not isinstance(self.data.get('contexts'), dict):
                raise DraftError('strict context record collection must be an object')
            for context in self.data['contexts'].values():
                if not isinstance(context, dict):
                    raise DraftError('strict context record must be an object')
                spans = context.get('spans')
                if self._spans(spans) != spans:
                    raise DraftError('strict context sources differ from pinned normalized ranges')
        return strict

    def checkpoint(self) -> dict:
        target = self.path.with_suffix('.json.partial')
        if target.is_symlink():
            raise DraftError('checkpoint temporary path cannot be a symlink')
        content = json.dumps(self.data, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + '\n'
        with target.open('w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(target, self.path)
        return self.summary()

    def summary(self) -> dict:
        return {'revision': self.data['revision'], 'nodes': len(self.data['nodes']),
                'accepted_nodes': sum(n['judgment'] == 'accepted' for n in self.data['nodes'].values()),
                'relations': len(self.data['relations']),
                'accepted_relations': sum(r['judgment'] == 'accepted' for r in self.data['relations'].values()),
                'unresolved': len(self.data['unresolved']), 'finalized': self.data['finalized']}

    def _spans(self, values: Any) -> list[dict]:
        if not isinstance(values, list) or not values:
            raise DraftError('spans must be a nonempty array of original-source ranges')
        result = [self.project.normalize_span(s) for s in values]
        if len({digest(s) for s in result}) != len(result):
            raise DraftError('duplicate source spans')
        return result

    def _context(self, value: Any, spans: list[dict], default_id: str) -> str:
        if isinstance(value, str):
            if value not in self.data['contexts']:
                raise DraftError('context reference is not registered')
            return value
        if not isinstance(value, dict):
            raise DraftError('context needs a description object or registered ID')
        context = copy.deepcopy(value)
        context_id = _identifier(context.get('id', default_id), 'context id')
        strict = self.data.get('context_source_policy') == self.EXPLICIT_CONTEXT_SOURCE_POLICY
        if strict and 'spans' not in context and 'sources' not in context:
            raise DraftError('inline context requires explicit nonempty spans or sources; '
                             'alternatively reference a registered context ID')
        context_spans = self._spans(context.get('spans', context.get('sources', spans)))
        record = {'id': context_id, 'description': _text(context.get('description'), 'context description'),
                  'spans': context_spans, 'parent': context.get('parent'),
                  'assumptions': context.get('assumptions'), 'symbols': context.get('symbols')}
        for key in ('assumptions', 'symbols'):
            if record[key] is not None:
                if not isinstance(record[key], list):
                    raise DraftError(f'context {key} must be an array or null')
                record[key] = [_text(item, key) for item in record[key]]
        if record['parent'] is not None:
            _identifier(record['parent'], 'parent context')
            if record['parent'] == context_id:
                raise DraftError('context cannot contain itself')
        # E2 validates the whole ancestry at finalization. For an inline context,
        # reusing an ID with changed contents is an explicit update, not inheritance.
        self.data['contexts'][context_id] = record
        return context_id

    def _upsert_node(self, args: dict) -> dict:
        node_id = _identifier(args.get('id'))
        category = args.get('category')
        if category not in ('definition', 'statement', 'expression', 'proof'):
            raise DraftError('category must be definition, statement, expression or proof')
        spans = self._spans(args.get('spans'))
        context = self._context(args.get('context'), spans, 'ctx:' + node_id)
        judgment = args.get('judgment', 'accepted')
        if judgment not in ('accepted', 'unresolved'):
            raise DraftError('node judgment must be accepted or unresolved')
        if judgment == 'accepted' and self.data.get('node_content_policy') == self.NODE_CONTENT_POLICY:
            self._require_word_boundaries(spans)
        semantics = args.get('semantics', {})
        if not isinstance(semantics, dict):
            raise DraftError('semantics is a JSON-object extension point')
        digest(semantics)
        for key in ('field', 'subfield'):
            if args.get(key) is not None:
                _text(args[key], key)
        for existing in self.data['nodes'].values():
            if existing['id'] != node_id and existing['category'] == category and existing['spans'] == spans:
                raise DraftError('same-category source occurrence already has node ID ' + existing['id'])
        record = {'id': node_id, 'category': category, 'label': _text(args.get('label'), 'label'),
                  'spans': spans, 'context': context, 'field': args.get('field'), 'subfield': args.get('subfield'),
                  'semantics': semantics, 'judgment': judgment}
        if args.get('note') is not None:
            record['note'] = _text(args['note'], 'note')
        self.data['nodes'][node_id] = record
        return {'id': node_id, 'category': category, 'judgment': judgment}

    def _upsert_relation(self, args: dict) -> dict:
        relation_type = args.get('type')
        if relation_type not in ('statement_dependency', 'proof_dependency', 'proves', 'identity'):
            raise DraftError('unsupported E2 relation type')
        source, target = _identifier(args.get('source'), 'source'), _identifier(args.get('target'), 'target')
        if source == target:
            raise DraftError('relation endpoints must differ')
        if source not in self.data['nodes'] or target not in self.data['nodes']:
            raise DraftError('register both endpoint nodes before a relation')
        judgment = args.get('judgment', 'accepted')
        if judgment not in ('accepted', 'unresolved'):
            raise DraftError('relation judgment must be accepted or unresolved')
        if judgment == 'accepted' and relation_type in ('statement_dependency', 'proof_dependency'):
            source_category = self.data['nodes'][source]['category']
            target_category = self.data['nodes'][target]['category']
            if ((relation_type == 'proof_dependency') != (target_category == 'proof')
                    or (relation_type == 'statement_dependency' and source_category == 'proof')):
                raise DraftError(
                    'dependency direction is premise -> user: proof_dependency needs a proof TARGET; '
                    'statement_dependency needs a non-proof TARGET and non-proof SOURCE',
                    details={'constraint': 'dependency_endpoint_categories', 'source': source, 'target': target,
                        'source_category': source_category, 'target_category': target_category,
                        'submitted_type': relation_type,
                        'repair_hint': 'Read both original nodes, correct endpoints/type, and locate use_witnesses '
                                       'inside the actual target. A tool failure is not a non-use verdict.'})
        evidence = args.get('evidence_spans', [])
        if evidence:
            evidence = self._spans(evidence)
        if not isinstance(evidence, list):
            raise DraftError('evidence_spans must be an array')
        if not evidence and relation_type == 'proves':
            evidence = self.data['nodes'][source]['spans']
        if judgment == 'accepted' and not evidence:
            raise DraftError('accepted relation requires original-source use evidence')
        if judgment == 'accepted' and relation_type in ('statement_dependency', 'proof_dependency'):
            usage = self.data['nodes'][target]['spans']
            if not any(e['file'] == s['file'] and e['char_start'] < s['char_end'] and s['char_start'] < e['char_end']
                       for e in evidence for s in usage):
                raise DraftError('dependency evidence must include an original use position within the target node')
        conditions = args.get('conditions')
        if conditions is not None:
            _text(conditions, 'conditions')
        if relation_type == 'identity':
            _text(conditions, 'identity common scope')
            _text(args.get('policy'), 'identity policy')
        relation_id = _identifier(args.get('id', 'rel:' + digest([relation_type, source, target, conditions])[:20]))
        record = {'id': relation_id, 'type': relation_type, 'source': source, 'target': target,
                  'explanation': _text(args.get('explanation'), 'explanation'), 'conditions': conditions,
                  'evidence_spans': evidence, 'judgment': judgment}
        if relation_type in ('statement_dependency', 'proof_dependency'):
            try:
                witnesses = self._use_witnesses(record, args.get('use_witnesses'))
            except DraftError as exc:
                if exc.details is not None:
                    exc.details.update(source=source, target=target, relation_id=relation_id)
                raise
            if witnesses is not None:
                record['use_witnesses'] = witnesses
        if relation_type == 'identity':
            record['policy'] = args['policy']
        method = args.get('method', 'source_observation' if relation_type == 'proves' else 'agent_analysis')
        if method not in ('source_observation', 'agent_analysis'):
            raise DraftError('E4 does not issue human or formal-certificate evidence')
        if relation_type == 'identity' and method == 'source_observation':
            raise DraftError('source observation alone cannot certify identity')
        record['method'] = method
        for other in self.data['relations'].values():
            if other['id'] != relation_id and all(other.get(k) == record.get(k) for k in ('type', 'source', 'target', 'conditions', 'policy')):
                raise DraftError('duplicate directed claim already has ID ' + other['id'])
        self.data['relations'][relation_id] = record
        return {'id': relation_id, 'type': relation_type, 'judgment': judgment}

    def _use_witnesses(self, relation: dict, values: Any) -> list[dict] | None:
        """Check concrete target uses; mathematical use remains an Agent judgment.

        Only the new campaign requires this runtime sidecar. Old immutable
        drafts keep their historical contract and E2 receives no extra field.
        """
        required = self.data.get('use_witness_binding') == self.USE_WITNESS_BINDING
        if values is None and not required:
            return None
        values = [] if values is None else values
        if not isinstance(values, list) or len(values) > 30:
            raise DraftError('use_witnesses must be an array of at most 30 concrete target uses')
        if required and relation['judgment'] == 'accepted' and not values:
            raise DraftError('accepted dependency requires a concrete use_witnesses record')
        nodes = self.data['nodes']
        target, source = nodes[relation['target']], nodes[relation['source']]
        catalog = {row['item_id']: row for row in self.project._inventory().references}
        result = []
        for value in values:
            if (not isinstance(value, dict)
                    or not {'span', 'reference_id', 'reference_key'} <= set(value)
                    or set(value) - {'span', 'reference_id', 'reference_key', 'source_role'}):
                raise DraftError('use witness needs span, reference_id and reference_key only; source_role is optional')
            span = self.project.normalize_span(value['span'])
            occurrence = self._use_occurrence(span)
            if occurrence is None:
                raise DraftError('use witness requires an unambiguous source reading occurrence')
            if not any(self._use_occurrence(body) == occurrence and body['file'] == span['file']
                       and body['char_start'] <= span['char_start'] < span['char_end'] <= body['char_end']
                       for body in target['spans']):
                raise DraftError('use witness must be wholly inside the target body and reading occurrence')
            if not any(self._use_occurrence(e) == occurrence and e['file'] == span['file']
                       and e['char_start'] <= span['char_start'] < span['char_end'] <= e['char_end']
                       for e in relation['evidence_spans']):
                raise DraftError('relation evidence must preserve the complete specific use witness')
            # Reuse the existing source-clause algorithm without loading a
            # live workflow checkpoint. Retained host drafts have no work_dir;
            # this pure checker grants no page/read/review completion credit.
            from .workflow import WorkflowGate, VERSION
            gate = self.__dict__.get('_use_fragment_gate')
            if gate is None:
                gate = WorkflowGate.__new__(WorkflowGate)
                gate.project, gate.draft = self.project, self
                gate.state = {'version': VERSION, 'source_occurrence_binding': 'unambiguous_occurrence_v1'}
                self._use_fragment_gate = gate
            if span['char_end'] - span['char_start'] > 1200:
                # A single author sentence/formula can exceed the ordinary
                # bound. Permit only its exact host-derived minimal complete
                # fragment; a page or a whole multi-sentence proof is not a
                # specific use merely because it contains such a fragment.
                fragments = gate._decision_fragments(target['spans'])
                if not any(len(fragment) == 1 and
                           self._use_occurrence(fragment[0]) == occurrence and
                           all(span[key] == fragment[0][key]
                               for key in ('file', 'char_start', 'char_end'))
                           for fragment in fragments):
                    raise DraftError('use witness over 1200 characters must be an exact minimal complete original fragment',
                        details={'constraint': 'exact_minimal_long_fragment', 'target': relation['target'],
                                 'submitted_evidence_spans': [span],
                                 'candidate_fragments': [self.project.fragment_record(fragment) for fragment in fragments[:4]],
                                 'read_hint': {'operation': 'target_use_fragments',
                                               'arguments': {'node_id': relation['target'], 'offset': 0, 'limit': 20}}})
            else:
                gate._require_decision_fragment([span], target['spans'], node_id=relation['target'])
            role = value.get('source_role', source['category'])
            if role != source['category']:
                raise DraftError('source_role must match the actual source endpoint category')
            reference_id, key = value['reference_id'], value['reference_key']
            if (reference_id is None) != (key is None):
                raise DraftError('implicit use has both reference_id and reference_key null')
            if reference_id is not None:
                row = catalog.get(reference_id)
                if row is None or key not in row.get('keys', []):
                    raise DraftError('use witness references an unknown source occurrence or key')
                if not any(self._use_occurrence(ref) == occurrence and ref['file'] == span['file']
                           and span['char_start'] <= ref['char_start'] < ref['char_end'] <= span['char_end']
                           for ref in row.get('source_spans', [row['span']])):
                    raise DraftError('explicit use witness must contain its full reference occurrence')
            result.append({'span': span, 'reference_id': reference_id,
                           'reference_key': key, 'source_role': role})
        if len({digest(value) for value in result}) != len(result):
            raise DraftError('duplicate dependency use witnesses')
        return result

    def _use_occurrence(self, span: dict) -> str | None:
        candidates = {row['occurrence'] for row in self.project.source_map
                      if row['source']['file'] == span['file']
                      and row['source']['char_start'] < span['char_end']
                      and span['char_start'] < row['source']['char_end']}
        stated = span.get('occurrence')
        return (stated if stated in candidates else None) if stated is not None else (
            next(iter(candidates)) if len(candidates) == 1 else None)

    def apply(self, operation: str, args: dict) -> dict:
        if not isinstance(args, dict):
            raise DraftError('operation arguments must be a JSON object')
        before = copy.deepcopy(self.data)
        try:
            self.data['finalized'] = False
            if operation == 'upsert_node':
                result = self._upsert_node(args)
            elif operation == 'upsert_relation':
                result = self._upsert_relation(args)
            elif operation in ('upsert_nodes', 'upsert_relations'):
                key = 'nodes' if operation == 'upsert_nodes' else 'relations'
                values = args.get(key, args.get('items'))
                if not isinstance(values, list) or not 1 <= len(values) <= 100:
                    raise DraftError('batch must contain 1..100 records')
                method = self._upsert_node if key == 'nodes' else self._upsert_relation
                result = {'records': [method(value) for value in values]}
            elif operation == 'upsert_context':
                spans = self._spans(args.get('spans'))
                result = {'id': self._context(args, spans, _identifier(args.get('id')))}
            elif operation == 'remove_node':
                node_id = _identifier(args.get('id'))
                if node_id not in self.data['nodes']:
                    raise DraftError('node is not registered')
                del self.data['nodes'][node_id]
                removed = [k for k, r in self.data['relations'].items() if node_id in (r['source'], r['target'])]
                for key in removed:
                    del self.data['relations'][key]
                result = {'removed': node_id, 'removed_relations': removed}
            elif operation == 'remove_relation':
                key = _identifier(args.get('id'))
                if key not in self.data['relations']:
                    raise DraftError('relation is not registered')
                del self.data['relations'][key]
                result = {'removed': key}
            elif operation == 'add_unresolved':
                record = {'id': args.get('id', 'unresolved:' + digest(args)[:20]),
                          'reason': _text(args.get('reason'), 'unresolved reason'),
                          'spans': self._spans(args['spans']) if args.get('spans') else [],
                          'kind': args.get('kind', 'mathematical_judgment')}
                if not isinstance(record['kind'], str):
                    raise DraftError('unresolved kind must be text')
                self.data['unresolved'] = [x for x in self.data['unresolved'] if x['id'] != record['id']] + [record]
                result = record
            elif operation == 'review_candidates':
                reviews = args.get('reviews')
                if not isinstance(reviews, list) or len(reviews) > 150:
                    raise DraftError('reviews must be an array of at most 150 items')
                for review in reviews:
                    event_id = review.get('event_id')
                    if event_id not in self.project._events:
                        raise DraftError('unknown candidate event ID')
                    if review.get('outcome') not in ('accepted', 'excluded', 'unresolved'):
                        raise DraftError('candidate outcome must be accepted, excluded or unresolved')
                    _text(review.get('explanation'), 'candidate review explanation')
                    if any(n not in self.data['nodes'] for n in review.get('node_ids', [])):
                        raise DraftError('candidate review references a missing node')
                    self.data['candidate_reviews'][event_id] = copy.deepcopy(review)
                result = {'reviewed': len(reviews)}
            elif operation == 'coverage':
                section = _text(args.get('scope'), 'coverage scope')
                self.data['coverage'][section] = {'description': _text(args.get('description'), 'coverage description'),
                                                'complete': args.get('complete', False),
                                                'spans': self._spans(args['spans']) if args.get('spans') else []}
                if type(self.data['coverage'][section]['complete']) is not bool:
                    raise DraftError('coverage complete must be boolean')
                result = self.coverage()
            elif operation == 'checkpoint':
                result = self.summary()
            else:
                raise DraftError('unsupported draft operation')
            self.data['revision'] += 1
            self.checkpoint()
        except Exception:
            self.data = before
            raise
        return {**result, 'revision': self.data['revision']}

    def coverage(self) -> dict:
        candidates = [e for e in self.project.structure_data.get('environments', [])
                      if e.get('environment_kind') == 'theorem' and e.get('activity') != 'inactive'
                      and e.get('source_role') == 'document_or_bibliography']
        unreviewed = []
        for event in candidates:
            if event['id'] in self.data['candidate_reviews']:
                continue
            span = event.get('body_source') or event.get('full_source') or event['source']
            covered = any(s['file'] == span['file'] and
                          max(0, min(s['char_end'], span['char_end']) - max(s['char_start'], span['char_start']))
                          >= .8 * max(1, span['char_end'] - span['char_start'])
                          for node in self.data['nodes'].values() for s in node['spans'])
            if not covered:
                unreviewed.append({'event_id': event['id'], 'name': event.get('name'), 'span': span})
        proofs = [n['id'] for n in self.data['nodes'].values() if n['category'] == 'proof' and n['judgment'] == 'accepted']
        matched = {r['source'] for r in self.data['relations'].values() if r['type'] == 'proves' and r['judgment'] == 'accepted'}
        return {'structural_candidate_count': len(candidates), 'unreviewed_candidates': unreviewed,
                'unmatched_accepted_proofs': [p for p in proofs if p not in matched],
                'declared_scopes': self.data['coverage'], 'unresolved_count': len(self.data['unresolved']),
                'notice': 'Structural coverage is not a measurement of semantic node or dependency recall.'}

    def list_nodes(self, category: str | None = None, offset: int = 0, limit: int = 100,
                   include_text: bool = False) -> dict:
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 150:
            raise DraftError('invalid node pagination')
        records = [copy.deepcopy(n) for _, n in sorted(self.data['nodes'].items()) if category is None or n['category'] == category]
        selected = records[offset:offset + limit]
        if include_text:
            for node in selected:
                node['original_text'] = [self.project.resolve_span(s) for s in node['spans']]
        return {'total': len(records), 'nodes': selected, 'next_offset': offset + limit if offset + limit < len(records) else None}

    def to_graph(self, snapshot_id: str) -> Graph:
        self.validate_context_source_policy()
        self.validate_node_content_policy()
        self.project.assert_unchanged()
        document = self.project.document
        accepted = [n for _, n in sorted(self.data['nodes'].items()) if n['judgment'] == 'accepted']
        units = tuple(Unit(id=n['id'], category=n['category'], label=n['label'], document=document.id,
                           context=n['context'], field=n['field'], subfield=n['subfield'],
                           sources=tuple(self.project.source(s) for s in n['spans']),
                           original_text=tuple(self.project.resolve_span(s) for s in n['spans']),
                           semantics=n['semantics']) for n in accepted)
        node_ids = {n.id for n in units}
        context_ids = {n.context for n in units}
        pending = list(context_ids)
        while pending:
            context_id = pending.pop()
            if context_id not in self.data['contexts']:
                raise DraftError('missing context ' + context_id)
            parent = self.data['contexts'][context_id]['parent']
            if parent and parent not in context_ids:
                context_ids.add(parent)
                pending.append(parent)
        contexts = tuple(Context(id=c['id'], document=document.id, parent=c['parent'],
                                 sources=tuple(self.project.source(s) for s in c['spans']),
                                 description=c['description'],
                                 assumptions=None if c['assumptions'] is None else tuple(c['assumptions']),
                                 symbols=None if c['symbols'] is None else tuple(c['symbols']))
                         for key, c in sorted(self.data['contexts'].items()) if key in context_ids)
        relations, evidence_inputs, identities = [], [], []
        for _, record in sorted(self.data['relations'].items()):
            if record['judgment'] != 'accepted':
                continue
            if record['source'] not in node_ids or record['target'] not in node_ids:
                raise DraftError('accepted relation refers to an unresolved endpoint')
            if record['type'] in ('statement_dependency', 'proof_dependency'):
                normalized = self._use_witnesses(record, record.get('use_witnesses'))
                if normalized is not None and normalized != record.get('use_witnesses'):
                    raise DraftError('retained dependency use witnesses differ from current endpoints/source')
            evidence_id = 'ev:' + record['id']
            common = {'id': record['id'], 'source': record['source'], 'target': record['target'],
                      'explanation': record['explanation'], 'conditions': record['conditions'], 'evidence': (evidence_id,)}
            if record['type'] == 'identity':
                relation = Identity(**common, policy=record['policy'])
                identities.append(relation)
            elif record['type'] == 'proves':
                relation = Proves(**common)
            else:
                relation = Dependency(**common, type=record['type'])
            relations.append(relation)
            evidence_inputs.append((relation, record, evidence_id))
        # One accepted symmetric claim yields two directed records sharing the
        # same bound evidence, as required by v4; no merging of source units.
        existing = {(r.source, r.target, r.conditions, getattr(r, 'policy', None)) for r in identities}
        for relation in identities:
            if (relation.target, relation.source, relation.conditions, relation.policy) in existing:
                raise DraftError('submit each symmetric identity once; reverse direction is generated')
            relations.append(relation.model_copy(update={'id': relation.id + ':reverse',
                                                         'source': relation.target, 'target': relation.source}))
        graph = Graph(snapshot_id=_text(snapshot_id, 'snapshot_id'), documents=(document,), contexts=contexts,
                      units=units, relations=tuple(relations))
        index = GraphIndex(graph)
        evidence = []
        for relation, record, evidence_id in evidence_inputs:
            sources = tuple(self.project.source(s) for s in record['evidence_spans'])
            evidence.append(Evidence(id=evidence_id, claim_sha256=index.claim_digest(relation, sources),
                                     method=record['method'], sources=sources, explanation=record['explanation']))
        return graph.model_copy(update={'evidence': tuple(evidence)})

    def validate(self, snapshot_id: str = 'e4-draft-validation') -> dict:
        try:
            graph = self.to_graph(snapshot_id)
            report = self.project.verify_graph(graph)
        except (ValueError, TypeError, OSError) as exc:
            report = {'structurally_valid': False, 'errors': [{'record': 'draft', 'reason': str(exc)}],
                      'source_validation': 'failed_or_incomplete', 'semantic_validation': 'not_performed'}
        return {**report, 'coverage': self.coverage(), 'draft': self.summary()}

    def finalize(self, snapshot_id: str = 'e4-draft-validation') -> dict:
        report = self.validate(snapshot_id)
        if not any(n['judgment'] == 'accepted' for n in self.data['nodes'].values()):
            report['errors'].append({'record': 'draft', 'reason': 'cannot finalize an extraction with no accepted nodes'})
        if report['coverage']['unreviewed_candidates']:
            report['errors'].append({'record': 'coverage', 'reason': 'review or explicitly defer remaining structural candidates'})
        report['structurally_valid'] = not report['errors']
        self.data['finalized'] = report['structurally_valid']
        self.data['revision'] += 1
        self.checkpoint()
        report['draft'] = self.summary()
        report['ready_for_host_publication'] = self.data['finalized']
        return report
