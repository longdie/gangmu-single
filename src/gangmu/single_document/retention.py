"""Mechanical retention of exact source character ranges, never semantic credit."""
from __future__ import annotations

import copy
import hashlib

from gangmu.representation.model import digest
from .draft import DraftError

VERSION = 'gangmu.e4.source_retention.v1'


def _union(intervals):
    result = []
    for start, end in sorted(intervals):
        if result and start <= result[-1][1]:
            result[-1][1] = max(result[-1][1], end)
        else:
            result.append([start, end])
    return result


def _coverage(query, intervals):
    covered = _union(intervals)
    missing, cursor = [], query['char_start']
    for start, end in covered:
        if cursor < start:
            missing.append([cursor, start])
        cursor = max(cursor, end)
    if cursor < query['char_end']:
        missing.append([cursor, query['char_end']])
    def spans(values):
        return [{**query, 'char_start': start, 'char_end': end} for start, end in values]
    return {'complete': not missing, 'covered': spans(covered), 'uncovered': spans(missing)}


def _occurrence(project, span):
    if span.get('occurrence') is not None:
        return {'status': 'anchored', 'effective': span['occurrence'], 'candidates': [span['occurrence']]}
    candidates = {}
    for row in project.source_map:
        original = row['source']
        if original['file'] != span['file']:
            continue
        start, end = max(span['char_start'], original['char_start']), min(span['char_end'], original['char_end'])
        if start < end:
            candidates.setdefault(row['occurrence'], []).append([start, end])
    names = sorted(candidates)
    if not names:
        return {'status': 'physical_unscoped', 'effective': None, 'candidates': []}
    if len(names) == 1 and _coverage(span, candidates[names[0]])['complete']:
        return {'status': 'unique', 'effective': names[0], 'candidates': names}
    return {'status': 'ambiguous_occurrence', 'effective': None, 'candidates': names}


def retention_status(project, draft, *, spans, node_id=None):
    """Separate saved content from actual accepted-node source accessibility.

    No draft/workflow writes, reading receipts, source-copying or graph repair.
    SourceProject supplies Unicode-character half-open ranges and pinned bytes.
    """
    if not isinstance(spans, list) or not 1 <= len(spans) <= 32:
        raise DraftError('retention_status needs 1..32 original-source spans')
    queries = [project.normalize_span(span) for span in spans]
    if len({digest(span) for span in queries}) != len(queries):
        raise DraftError('retention_status query spans must be distinct')
    if node_id is not None and (not isinstance(node_id, str) or node_id not in draft.data['nodes']):
        raise DraftError('retention_status node_id must name an actual saved node')
    data = draft.data
    units, contexts, relations = data['nodes'], data['contexts'], data['relations']
    source_sha = digest(project.fingerprint)
    saved = {}
    for kind, records in (('unit', units), ('context', contexts)):
        for identifier, record in sorted(records.items()):
            actual = [project.normalize_span(span) for span in record['spans']]
            saved[(kind, identifier)] = {'kind': kind, 'id': identifier,
                'judgment': record.get('judgment'), 'record_sha256': digest(record),
                'spans': [{'span': span, 'file_sha256': project.source_files[span['file']],
                    'text_utf8_sha256': hashlib.sha256(project.resolve_span(span).encode('utf-8')).hexdigest(),
                    'occurrence': _occurrence(project, span)} for span in actual], 'consumers': []}

    def bind(source_id, consumer_id, kind, relation_id=None):
        binding = {'node_id': consumer_id, 'binding': kind, 'via_node_id': source_id,
                   'relation_id': relation_id}
        saved[('unit', source_id)]['consumers'].append({**binding, 'record_role':
            'node_body' if relation_id is None else 'dependency_source_body'})
        context_id, seen, chain = units[source_id]['context'], set(), []
        while context_id is not None:
            if context_id in seen or context_id not in contexts:
                raise DraftError('retention_status needs complete acyclic actual Context ancestry')
            seen.add(context_id)
            chain.append(context_id)
            saved[('context', context_id)]['consumers'].append({**binding, 'context_chain': list(chain),
                'record_role': 'node_context_chain' if relation_id is None else 'dependency_source_context',
                'target_scope_inheritance': 'not_inferred'})
            context_id = contexts[context_id].get('parent')

    excluded = []
    targets = sorted(units) if node_id is None else [node_id]
    for target in targets:
        if units[target]['judgment'] != 'accepted':
            continue
        bind(target, target, 'node_body_and_context')
        for identifier, relation in sorted(relations.items()):
            if relation['type'] not in ('statement_dependency', 'proof_dependency') or relation['target'] != target:
                continue
            reason = None
            if relation['judgment'] != 'accepted':
                reason = 'relation_not_accepted'
            elif relation['source'] not in units:
                reason = 'source_node_not_saved'
            elif units[relation['source']]['judgment'] != 'accepted':
                reason = 'source_node_not_accepted'
            elif (relation['source'] == target
                    or ((relation['type'] == 'proof_dependency') != (units[target]['category'] == 'proof'))
                    or (relation['type'] == 'statement_dependency' and units[relation['source']]['category'] == 'proof')):
                # Match Draft's existing endpoint constraints, including a
                # transient category edit after the relation was accepted.
                reason = 'current_dependency_endpoint_categories_invalid'
            if reason is not None:
                excluded.append({'relation_id': identifier, 'type': relation['type'],
                    'source': relation['source'], 'target': target, 'reason': reason})
                continue
            bind(relation['source'], target, 'direct_' + relation['type'], identifier)
    rows = list(saved.values())
    results = []
    for query in queries:
        occurrence = _occurrence(project, query)
        physical, stored, accessible, matches = [], [], [], []
        for row in rows:
            for item in row['spans']:
                actual = item['span']
                if query['file'] != actual['file']:
                    continue
                start, end = max(query['char_start'], actual['char_start']), min(query['char_end'], actual['char_end'])
                if start >= end:
                    continue
                interval = [start, end]
                physical.append(interval)
                compatible = (occurrence['status'] != 'ambiguous_occurrence'
                    and item['occurrence']['status'] != 'ambiguous_occurrence'
                    and occurrence['effective'] == item['occurrence']['effective'])
                matches.append({'kind': row['kind'], 'id': row['id'],
                    'record_sha256': row['record_sha256'], 'saved_span': actual,
                    'intersection': {**query, 'char_start': start, 'char_end': end},
                    'occurrence_compatible': compatible,
                    'accepted_consumer_ids': sorted({b['node_id'] for b in row['consumers']})})
                if compatible:
                    stored.append(interval)
                    if row['consumers']:
                        accessible.append(interval)
        results.append({'span': query, 'occurrence': occurrence,
            'text_utf8_sha256': hashlib.sha256(project.resolve_span(query).encode('utf-8')).hexdigest(),
            'physical_stored': _coverage(query, physical), 'stored': _coverage(query, stored),
            'accessible': _coverage(query, accessible), 'matches': matches})
    return {'schema_version': VERSION, 'source_fingerprint_sha256': source_sha,
        'draft_revision': data['revision'], 'retained_records_sha256': digest({
            'nodes': units, 'contexts': contexts, 'relations': relations}),
        'node_id': node_id, 'node_judgment': units[node_id]['judgment'] if node_id is not None else None,
        'records': copy.deepcopy(rows), 'queries': results, 'excluded_direct_dependencies': excluded,
        'qualification': ('Mechanical saved character ranges only. physical_stored does not assign a reading occurrence; '
            'stored does not imply a node consumes it. accessible requires actual accepted-node bindings and compatible '
            'occurrences; a dependency source Context is that premise\'s background, not an inherited target scope. '
            'Coverage does not certify applicability, semantic sufficiency, truth or complete source reading. '
            'Description, evidence-only spans, retrieval windows and self-reported source_review are not saved content.')}
