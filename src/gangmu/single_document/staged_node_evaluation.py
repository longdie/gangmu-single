"""Independent, source-bound node-only evaluation for the staged E4 campaign.

The injected judge starts fresh review sessions; this module performs no model
transport. Independent references fix both coverage denominators before any
prediction is inspected. Literal fidelity is separate from author math truth.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Callable

from gangmu.representation.model import Context, Unit, digest

VERSION = 'gangmu.e4.staged-node-evaluation.v1'
ROOT = Path(__file__).resolve().parents[3]
PROMPTS = ROOT / 'prompts/evaluation/staged_nodes'
AXES = ('category', 'title', 'complete_task', 'necessary_context')
THRESHOLDS = {'precision': .90, 'explicit_skeleton_coverage': .99,
              'main_statement_content_coverage': 1.0}
Judge = Callable[[str, str, dict], dict]


def _spans(record: dict) -> list[dict]:
    return record.get('source_spans', record.get('sources', []))


def _overlaps(left: dict, right: dict) -> bool:
    return any(a['file'] == b['file'] and a.get('occurrence') == b.get('occurrence')
               and a['char_start'] < b['char_end'] and b['char_start'] < a['char_end']
               for a in _spans(left) for b in _spans(right))


def _source_records(project, record: dict, *, literal: bool, document: bool) -> list[dict]:
    spans = _spans(record)
    if not isinstance(spans, list) or not spans:
        raise ValueError('missing source spans')
    texts = record.get('original_text')
    if literal and (not isinstance(texts, list) or len(texts) != len(spans)):
        raise ValueError('original_text is not paired one-to-one with source spans')
    result = []
    for index, raw in enumerate(spans):
        if document and raw.get('document') != project.document.id:
            raise ValueError('source document differs from pinned paper')
        span = project.normalize_span(raw)
        if span.get('occurrence') is None:
            occurrences = {entry['occurrence'] for entry in project.source_map
                           if entry['source']['file'] == span['file']}
            if len(occurrences) != 1:
                raise ValueError('source occurrence is absent or ambiguous')
            span = project.normalize_span(dict(span, occurrence=next(iter(occurrences))))
        text = project.resolve_span(span)
        if literal and text != texts[index]:
            raise ValueError('original_text differs from its literal original source slice')
        result.append({'source': dict(span, document=project.document.id), 'text': text})
    return result


def _reference_rows(project, rows: list[dict], name: str) -> list[dict]:
    if not isinstance(rows, list):
        raise ValueError(f'{name} must contain an independent list')
    found, normalized = set(), []
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get('id'), str) or not row['id'] or row['id'] in found:
            raise ValueError(f'{name} has an invalid or duplicate reference id')
        found.add(row['id'])
        evidence = _source_records(project, row, literal=True, document=False)
        normalized.append(dict(row, source_spans=[item['source'] for item in evidence]))
    return normalized


def _references(project, rule: dict, core: dict) -> tuple[list[dict], list[dict], list[dict]]:
    for name, reference in (('rule_reference', rule), ('core_reference', core)):
        if reference.get('paper_id') != project.paper_id:
            raise ValueError(f'{name} paper differs from pinned source')
        provenance = reference.get('provenance', {})
        if provenance.get('production_blind') is not True or provenance.get('production_predictions_read'):
            raise ValueError(f'{name} is not declared independent of predictions')
        files = reference.get('source_files', provenance.get('source_files', []))
        if not isinstance(files, list) or not files:
            raise ValueError(f'{name} has no pinned original source file hashes')
        for row in files:
            if project.source_files.get(row.get('file')) != row.get('sha256'):
                raise ValueError(f'{name} original source file hash differs')
    if core.get('source_first') is not True or core.get('independent_of_prediction') is not True:
        raise ValueError('core coverage inventory must be independent and source-first')
    if core.get('source_fingerprint') != project.fingerprint:
        raise ValueError('core_reference source fingerprint differs from pinned source')
    if rule.get('source_fingerprint', project.fingerprint) != project.fingerprint:
        raise ValueError('rule_reference source fingerprint differs from pinned source')
    all_rule = _reference_rows(project, rule.get('items'), 'rule_reference.items')
    core_rows = _reference_rows(project, core.get('core_items'), 'core_reference.core_items')
    explicit = core.get('explicit_items', {})
    qualified = explicit.get('qualified_item_ids')
    if (not isinstance(qualified, list) or not all(isinstance(key, str) for key in qualified)
            or len(qualified) != len(set(qualified))):
        raise ValueError('qualified explicit skeleton ids must be independently fixed')
    rule_ids = {row['id'] for row in all_rule}
    required_ids = {row['id'] for row in all_rule if row.get('required') is True}
    if not set(qualified) <= rule_ids or not required_ids <= set(qualified):
        raise ValueError('qualified skeleton has unknown ids or omits independently required explicit tasks')
    source_reference = explicit.get('source_reference', {})
    if source_reference:
        path = (ROOT / source_reference.get('path', '')).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError('independent skeleton source reference path is not a project-local file')
        raw = path.read_bytes()
        if (hashlib.sha256(raw).hexdigest() != source_reference.get('sha256')
                or json.loads(raw) != rule):
            raise ValueError('independent skeleton reference bytes/content differ from supplied rule reference')
    main_ids = core.get('main_statement_ids')
    roles = {row['id']: row.get('role') for row in core_rows}
    if (not isinstance(main_ids, list) or not main_ids or not all(isinstance(key, str) for key in main_ids)
            or len(main_ids) != len(set(main_ids))
            or any(roles.get(key) != 'main_statement' for key in main_ids)
            or {key for key, role in roles.items() if role == 'main_statement'} != set(main_ids)):
        raise ValueError('independent core inventory must explicitly enumerate its main statements')
    if any(row.get('required') is not True for row in core_rows):
        raise ValueError('core inventory obligations must declare required:true')
    for row in core_rows:
        contexts = row.get('context_core_ids', [])
        if (not isinstance(contexts, list) or not all(isinstance(key, str) and key in roles for key in contexts)
                or len(contexts) != len(set(contexts))):
            raise ValueError('independent core context obligations have unknown or duplicate ids')
    supplements = _reference_rows(project, rule.get('context_supplements', []), 'context_supplements')
    hints = explicit.get('qualified_item_category_hints', {})
    if not isinstance(hints, dict) or any(key not in qualified for key in hints):
        raise ValueError('independent skeleton category hints have unknown qualified ids')
    representations = core.get('qualified_item_acceptable_representations',
                               explicit.get('qualified_item_acceptable_representations', {}))
    if ('qualified_item_acceptable_representations' in core
            and 'qualified_item_acceptable_representations' in explicit
            and representations != explicit['qualified_item_acceptable_representations']):
        raise ValueError('conflicting independent skeleton representation allowances')
    if (not isinstance(representations, dict) or any(key not in qualified for key in representations)
            or any(value not in ('unit', 'unit_or_bound_context') for value in representations.values())):
        raise ValueError('independent skeleton representation allowances are invalid')
    qualified_rows = [dict(row, category_hint=hints.get(row['id'], row.get('category_hint')),
                          acceptable_representation=representations.get(row['id'], 'unit'))
                      for row in all_rule if row['id'] in set(qualified)]
    return qualified_rows, core_rows, supplements


def _batches(rows: list[dict], max_items: int, target_chars: int):
    batch, size = [], 0
    for row in rows:
        amount = len(json.dumps(row, ensure_ascii=False))
        if batch and (len(batch) >= max_items or size + amount > target_chars):
            yield batch
            batch, size = [], 0
        batch.append(row)
        size += amount
    if batch:
        yield batch


def _judgments(judge: Judge, task_kind: str, instructions: str, payload: dict,
               ids: list[str], key: str) -> tuple[dict[str, dict], dict]:
    trace = {'task_kind': task_kind, 'payload_sha256': digest(payload), 'requested_ids': ids}
    try:
        response = judge(task_kind, instructions, copy.deepcopy(payload))
    except Exception as exc:
        trace.update(status='judge_error', exception_type=type(exc).__name__)
        return {}, trace
    trace['response'] = copy.deepcopy(response)
    rows = response.get('judgments') if isinstance(response, dict) else None
    if not isinstance(rows, list):
        trace['status'] = 'invalid_response'
        return {}, trace
    counts = Counter(row.get(key) for row in rows if isinstance(row, dict) and isinstance(row.get(key), str))
    found = {row[key]: row for row in rows if isinstance(row, dict) and row.get(key) in ids
             and counts[row[key]] == 1}
    trace.update(status='completed', omitted_ids=[item for item in ids if item not in found],
                 unexpected_ids=[item for item in counts if item not in ids],
                 duplicate_ids=[item for item, count in counts.items() if count > 1])
    return found, trace


def _metric(rows: list[dict], positive: str, negative: str) -> dict:
    tally = Counter(row['status'] for row in rows)
    total = len(rows)
    lower = tally[positive] / total if total else None
    upper = (tally[positive] + tally['unknown']) / total if total else None
    return {'denominator': total, positive: tally[positive], negative: tally[negative],
            'unknown': tally['unknown'], 'value': lower if total and not tally['unknown'] else None,
            'bounds': {'lower': lower, 'upper': upper}}


def _complete_span_cover(reference: dict, records: list[dict]) -> bool:
    """All original-source bytes of every occurrence-qualified range survive."""
    for span in _spans(reference):
        intervals = sorted((part['char_start'], part['char_end']) for record in records
            for part in _spans(record) if part['file'] == span['file']
            and part.get('occurrence') == span.get('occurrence'))
        cursor = span['char_start']
        for start, end in intervals:
            if start <= cursor:
                cursor = max(cursor, end)
        if cursor < span['char_end']:
            return False
    return True


def evaluate_complete_nodes(project, node_result: dict, rule_reference: dict,
                            core_reference: dict, judge: Judge, options: dict | None = None) -> dict:
    """Review every accepted unit and fixed independent source obligations.

    Coverage input misuse raises ValueError before judging. Producer/schema,
    citation, and judge failures remain explicit failures or unknowns in the
    complete accepted-unit population. Thresholds are fixed, not configurable.
    options controls batch_max_items (default 30), batch_target_chars (default
    24000), neighborhood_chars (default 1200), and deterministic_exact_skeleton
    (default True). Exact skeleton retention also requires compatible actual
    units passing independent precision review; Context alone never certifies it.
    """
    project.assert_unchanged()
    prediction, rule, core = map(copy.deepcopy, (node_result, rule_reference, core_reference))
    skeleton, core_rows, supplements = _references(project, rule, core)
    options = options or {}
    limits = {'batch_max_items': 30, 'batch_target_chars': 24000, 'neighborhood_chars': 1200}
    if set(options) - (set(limits) | {'deterministic_exact_skeleton'}):
        raise ValueError('unknown staged node evaluation option')
    limits.update({key: value for key, value in options.items() if key in limits})
    if any(type(value) is not int or value < 1 for value in limits.values()):
        raise ValueError('evaluation batch/context sizes must be positive integers')
    deterministic = options.get('deterministic_exact_skeleton', True)
    if type(deterministic) is not bool:
        raise ValueError('deterministic_exact_skeleton must be a boolean')
    instructions = {kind: (PROMPTS / f'{kind}.md').read_text() for kind in ('precision', 'coverage')}
    provenance_errors, traces = [], []
    if prediction.get('paper_id') != project.paper_id or prediction.get('source_fingerprint') != project.fingerprint:
        provenance_errors.append('producer source identity/fingerprint differs')
    if prediction.get('documents') != [project.document.model_dump(mode='json')]:
        provenance_errors.append('producer documents differ from pinned document')
    rule_hash = prediction.get('rule_result_sha256')
    if not isinstance(rule_hash, str) or len(rule_hash) != 64 or any(c not in '0123456789abcdef' for c in rule_hash):
        provenance_errors.append('producer rule candidate result hash is absent or invalid')
    units, contexts = prediction.get('units'), prediction.get('contexts')
    if not isinstance(units, list) or not isinstance(contexts, list):
        raise ValueError('node_result must enumerate all accepted units and contexts')
    ids = Counter(row.get('id') for row in units if isinstance(row, dict) and isinstance(row.get('id'), str))
    context_ids = Counter(row.get('id') for row in contexts if isinstance(row, dict) and isinstance(row.get('id'), str))
    context_map, context_errors = {}, {}
    for index, raw in enumerate(contexts):
        key = raw.get('id') if isinstance(raw, dict) else f'context-row-{index}'
        try:
            record = Context.model_validate_json(json.dumps(raw)).model_dump(mode='json')
            if context_ids[key] != 1 or record['document'] != project.document.id:
                raise ValueError('duplicate context or wrong document')
            record['source_evidence'] = _source_records(project, record, literal=False, document=True)
            record['sources'] = [item['source'] for item in record['source_evidence']]
            context_map[key] = record
        except Exception as exc:
            context_errors[str(key)] = {'reason': 'invalid context/source citation', 'error_type': type(exc).__name__}

    def chain(key: str) -> list[dict]:
        result, seen = [], set()
        while key is not None:
            if key in seen or key not in context_map:
                raise ValueError('missing/invalid context or cyclic parent chain')
            seen.add(key)
            result.append(context_map[key])
            key = context_map[key]['parent']
        return result

    provenance = prediction.get('unit_provenance', [])
    provenance_counts = Counter(row.get('unit_id') for row in provenance
                                if isinstance(row, dict) and isinstance(row.get('unit_id'), str)) if isinstance(provenance, list) else Counter()
    round_map = {}
    for row in provenance if isinstance(provenance, list) else []:
        if not isinstance(row, dict) or not isinstance(row.get('unit_id'), str) or provenance_counts[row['unit_id']] != 1:
            continue
        value = row.get('round')
        normalized = {1: 'round1', 2: 'round2', 'round1': 'round1', 'round2': 'round2'}.get(value) if isinstance(value, (str, int)) and type(value) is not bool else None
        if normalized:
            round_map[row['unit_id']] = normalized
    if set(round_map) != set(ids) or any(value != 1 for value in provenance_counts.values()):
        provenance_errors.append('unit round provenance is missing, duplicate, invalid, or has unknown ids')
    precision_rows, valid, precision_payloads, used_contexts = [], {}, [], {}
    for index, raw in enumerate(units):
        key = raw.get('id') if isinstance(raw, dict) and isinstance(raw.get('id'), str) else None
        evaluation_id = key if isinstance(key, str) and key and ids[key] == 1 else f'accepted-row-{index}'
        row = {'unit_id': key, 'evaluation_id': evaluation_id,
               'round': round_map.get(key, 'unknown'),
               'category': raw.get('category', 'unknown') if isinstance(raw, dict) else 'unknown',
               'status': 'unsupported', 'programmatic_source_valid': False}
        try:
            record = Unit.model_validate_json(json.dumps(raw)).model_dump(mode='json')
            if ids[key] != 1 or record['document'] != project.document.id or record['semantics'] != {}:
                raise ValueError('duplicate unit, wrong document, or nonempty staged semantics')
            evidence = _source_records(project, record, literal=True, document=True)
            record['sources'] = [item['source'] for item in evidence]
            attached = chain(record['context'])
            row.update(programmatic_source_valid=True, status='unknown', reason='not reviewed')
            valid[key] = record
            used_contexts.update({item['id']: item for item in attached})
            author_tasks = [item for item in skeleton if _overlaps(record, item)]
            larger_author_tasks = [item for item in author_tasks if not _complete_span_cover(item, [record])]
            needed = []
            for item in supplements:
                owners = item.get('corresponding_item_ids', item.get('applies_to', []))
                if isinstance(owners, str):
                    owners = [owners]
                owners = list(owners) + ([item['corresponding_item_id']] if item.get('corresponding_item_id') else [])
                if any(task['id'] in owners for task in author_tasks):
                    needed.append(item)
            neighborhoods = []
            for item in evidence:
                span = item['source']
                source_text = project._read_text(span['file'])
                left = max(0, span['char_start'] - limits['neighborhood_chars'])
                right = min(len(source_text), span['char_end'] + limits['neighborhood_chars'])
                neighborhoods.append({'file': span['file'], 'occurrence': span['occurrence'],
                    'before': {'char_start': left, 'char_end': span['char_start'], 'text': source_text[left:span['char_start']]},
                    'after': {'char_start': span['char_end'], 'char_end': right, 'text': source_text[span['char_end']:right]}})
            payload = {'unit_id': key, 'unit': record, 'attached_contexts': attached,
                       'complete_author_task_ids': [item['id'] for item in author_tasks if item not in larger_author_tasks],
                       'audit_only_complete_author_tasks': larger_author_tasks,
                       'audit_only_required_context_regions': needed,
                       'audit_only_source_neighborhoods': neighborhoods}
            row['claim_sha256'] = digest({'source': project.fingerprint, 'unit': record, 'contexts': attached})
            precision_payloads.append(payload)
        except Exception as exc:
            row.update(reason='invalid E2v4 unit, literal source, or attached context', error_type=type(exc).__name__)
        precision_rows.append(row)
    precision_by_id = {row['unit_id']: row for row in precision_rows if row['programmatic_source_valid']}
    for batch in _batches(precision_payloads, limits['batch_max_items'], limits['batch_target_chars']):
        payload = {'paper_id': project.paper_id, 'source_fingerprint': project.fingerprint, 'units': batch}
        reviewed, trace = _judgments(judge, 'staged_nodes_precision', instructions['precision'], payload,
                                     [item['unit_id'] for item in batch], 'unit_id')
        traces.append(trace)
        for item in batch:
            key = item['unit_id']
            judgment = reviewed.get(key, {})
            axes = {axis: judgment.get(axis, 'unknown') for axis in AXES}
            axes = {axis: value if value in ('pass', 'fail', 'unknown') else 'unknown' for axis, value in axes.items()}
            status = judgment.get('status')
            if 'fail' in axes.values() or status == 'unsupported':
                verdict = 'unsupported'
            elif (all(value == 'pass' for value in axes.values()) and status == 'supported'
                  and isinstance(judgment.get('reason'), str) and judgment['reason'].strip()):
                verdict = 'supported'
            else:
                verdict = 'unknown'
            precision_by_id[key].update(status=verdict, axes=axes,
                                        reason=judgment.get('reason') or 'missing, duplicate, or incomplete independent judgment')

    def coverage(rows: list[dict], kind: str) -> tuple[list[dict], dict]:
        output, requests = [], []
        independent_core = {row['id']: row for row in core_rows}
        for reference in rows:
            candidates = [record for record in valid.values() if _overlaps(reference, record)]
            category = reference.get('category_hint', reference.get('category'))
            if kind == 'explicit_skeleton' and deterministic and category in ('definition', 'statement', 'expression', 'proof'):
                independently_supported = [record for record in candidates
                    if record['category'] == category and precision_by_id[record['id']]['status'] == 'supported']
                if _complete_span_cover(reference, independently_supported):
                    output.append({'reference_id': reference['id'], 'role': reference.get('role'),
                        'status': 'covered', 'unit_ids': [record['id'] for record in independently_supported],
                        'context_ids': [], 'citations_valid': True,
                        'method': 'complete_literal_spans_in_independently_supported_compatible_units',
                        'reason': 'Every pinned source span is fully retained by independently precision-supported units of the source-qualified category.'})
                    continue
            selected_contexts = {record['id']: record for record in used_contexts.values()
                                 if _overlaps(reference, record)}
            for candidate in candidates:
                selected_contexts.update({record['id']: record for record in chain(candidate['context'])})
            background = list(selected_contexts.values())
            requests.append({'reference_id': reference['id'], 'reference': reference,
                             'audit_only_required_reference_contexts': [independent_core[key]
                                 for key in reference.get('context_core_ids', [])],
                             'candidate_units': candidates, 'attached_candidate_contexts': background})
        for batch in _batches(requests, limits['batch_max_items'], limits['batch_target_chars']):
            payload = {'paper_id': project.paper_id, 'source_fingerprint': project.fingerprint,
                       'inventory_kind': kind, 'obligations': batch}
            reviewed, trace = _judgments(judge, f'staged_nodes_{kind}_coverage', instructions['coverage'], payload,
                                         [item['reference_id'] for item in batch], 'reference_id')
            traces.append(trace)
            for item in batch:
                judgment = reviewed.get(item['reference_id'], {})
                unit_keys, context_keys = judgment.get('unit_ids', []), judgment.get('context_ids', [])
                allowed_units = {row['id']: row for row in item['candidate_units']}
                allowed_contexts = {row['id']: row for row in item['attached_candidate_contexts']}
                citations_valid = (isinstance(unit_keys, list) and isinstance(context_keys, list)
                    and all(isinstance(key, str) and key in allowed_units for key in unit_keys)
                    and all(isinstance(key, str) and key in allowed_contexts for key in context_keys)
                    and len(unit_keys) == len(set(unit_keys)) and len(context_keys) == len(set(context_keys)))
                status = judgment.get('status')
                status = status if (status in ('covered', 'missing', 'unknown')
                    and isinstance(judgment.get('reason'), str) and judgment['reason'].strip()) else 'unknown'
                reference = item['reference']
                if status == 'covered':
                    category = reference.get('category_hint', reference.get('category'))
                    explicit_context_allowed = (kind == 'explicit_skeleton'
                        and reference.get('acceptable_representation') == 'unit_or_bound_context'
                        and category == 'definition')
                    requires_unit = (reference.get('role') == 'main_statement'
                        or category in ('statement', 'proof', 'expression')
                        or kind == 'explicit_skeleton' and not explicit_context_allowed)
                    expected = 'statement' if reference.get('role') == 'main_statement' else category
                    if (not citations_valid or not (unit_keys or context_keys)
                            or requires_unit and not unit_keys
                            or requires_unit and expected is not None and not any(allowed_units[key]['category'] == expected for key in unit_keys)):
                        status = 'unknown'
                output.append({'reference_id': item['reference_id'], 'role': reference.get('role'),
                               'status': status, 'unit_ids': unit_keys if citations_valid else [],
                               'context_ids': context_keys if citations_valid else [],
                               'reason': judgment.get('reason') or 'missing or invalid independent coverage judgment',
                               'citations_valid': citations_valid, 'method': 'independent_semantic_judge'})
        by_id = {row['reference_id']: row for row in output}
        output = [by_id[row['id']] for row in rows]
        return output, _metric(output, 'covered', 'missing')

    skeleton_rows, skeleton_metric = coverage(skeleton, 'explicit_skeleton')
    content_rows, content_metric = coverage(core_rows, 'main_content')
    main_ids = set(core['main_statement_ids'])
    main_rows = [row for row in content_rows if row['reference_id'] in main_ids]
    precision_metric = _metric(precision_rows, 'supported', 'unsupported')
    checks = {'nonempty_accepted_output': bool(units), 'source_and_round_provenance': not provenance_errors,
              'precision': precision_metric['bounds']['lower'] is not None and precision_metric['bounds']['lower'] >= .9,
              'explicit_skeleton_coverage': skeleton_metric['bounds']['lower'] is not None and skeleton_metric['bounds']['lower'] >= .99,
              'main_statement_content_coverage': content_metric['bounds']['lower'] == 1.0}
    strata = {}
    for dimension, labels in (('round', ('round1', 'round2', 'unknown')),
                              ('category', ('definition', 'statement', 'expression', 'proof', 'unknown'))):
        strata[dimension] = {label: _metric([row for row in precision_rows
            if (row[dimension] if row[dimension] in labels else 'unknown') == label], 'supported', 'unsupported') for label in labels}
    project.assert_unchanged()
    return {'schema_version': VERSION, 'paper_id': project.paper_id, 'status': 'passed' if all(checks.values()) else 'failed',
            'passed': all(checks.values()), 'thresholds': dict(THRESHOLDS), 'checks': checks,
            'evaluation_options': dict(limits, deterministic_exact_skeleton=deterministic),
            'bindings': {'source_fingerprint': project.fingerprint, 'source_fingerprint_sha256': digest(project.fingerprint),
                         'evaluator_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                         'node_result_sha256': digest(prediction), 'rule_result_sha256': rule_hash,
                         'rule_reference_sha256': digest(rule), 'core_reference_sha256': digest(core),
                         'prompts_sha256': {key: hashlib.sha256(value.encode()).hexdigest() for key, value in instructions.items()}},
            'precision': dict(precision_metric, judgments=precision_rows, strata=strata),
            'explicit_skeleton_coverage': dict(skeleton_metric, judgments=skeleton_rows,
                                              qualified_reference_ids=[row['id'] for row in skeleton]),
            'main_statement_content_coverage': dict(content_metric, judgments=content_rows),
            'main_statements_only_coverage': _metric(main_rows, 'covered', 'missing'),
            'programmatic_audit': {'provenance_errors': provenance_errors, 'invalid_contexts': context_errors,
                                   'source_valid_accepted_units': len(valid), 'attached_source_valid_contexts': len(used_contexts)},
            'judge_calls': traces, 'judge_call_count': len(traces),
            'limitations': {'human_expert_gold': False, 'author_mathematics_certified': False,
                            'dependency_or_dag_evaluated': False, 'whole_body_recall_claimed': False,
                            'fresh_independent_judge_sessions_required': True}}
