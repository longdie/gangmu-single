"""One source-bound completion of structurally missing review IDs.

This adapter never rejudges a uniquely identified row, including a negative or
semantic unknown. It preserves original Responses separately from any host
composition; it does not infer an intended ID from a supplier typo.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
from pathlib import Path

from gangmu.representation.model import digest

VERSION = 'gangmu.e4.staged-review-completion.v1'
TASKS = {
    'staged_nodes_precision': ('units', 'unit_id'),
    'staged_nodes_explicit_skeleton_coverage': ('obligations', 'reference_id'),
    'staged_nodes_main_content_coverage': ('obligations', 'reference_id'),
}
AXES = ('category', 'title', 'complete_task', 'necessary_context')


def _schema(kind: str, ids: list[str]) -> dict:
    key = TASKS[kind][1]
    fields = {key: {'type': 'string', 'enum': ids},
              'reason': {'type': 'string', 'minLength': 1}}
    if kind == 'staged_nodes_precision':
        fields['status'] = {'type': 'string', 'enum': ['supported', 'unsupported', 'unknown']}
        fields.update({axis: {'type': 'string', 'enum': ['pass', 'fail', 'unknown']} for axis in AXES})
    else:
        fields['status'] = {'type': 'string', 'enum': ['covered', 'missing', 'unknown']}
        fields.update({name: {'type': 'array', 'items': {'type': 'string'}}
                       for name in ('unit_ids', 'context_ids')})
    row = {'type': 'object', 'additionalProperties': False,
           'required': list(fields), 'properties': fields}
    return {'type': 'object', 'additionalProperties': False, 'required': ['judgments'],
            'properties': {'judgments': {'type': 'array', 'minItems': len(ids),
                           'maxItems': len(ids), 'items': row}}}


def _counts(rows, key):
    return Counter(row[key] for row in rows if isinstance(row, dict)
                   and isinstance(row.get(key), str))


def _valid_row(row: dict, kind: str) -> bool:
    key = TASKS[kind][1]
    fields = {key, 'status', 'reason'} | (set(AXES) if kind == 'staged_nodes_precision'
                                        else {'unit_ids', 'context_ids'})
    if set(row) != fields or not isinstance(row.get('reason'), str) or not row['reason'].strip():
        return False
    if kind == 'staged_nodes_precision':
        return (row.get('status') in ('supported', 'unsupported', 'unknown')
                and all(row.get(axis) in ('pass', 'fail', 'unknown') for axis in AXES))
    return (row.get('status') in ('covered', 'missing', 'unknown')
            and all(isinstance(row.get(name), list)
                    and all(isinstance(item, str) for item in row[name])
                    for name in ('unit_ids', 'context_ids')))


class MissingReviewCompletionJudge:
    """Wrap a synchronous judge; at most one follow-up per primary request.

    Follow-ups contain only the exact original rows whose IDs were omitted or
    duplicated, plus an ID-enumerated strict response schema. Valid-ID unknowns,
    negatives and malformed semantic axes never trigger a second opinion.
    ``traces()`` exposes actual primary/secondary responses and payload hashes.
    A changed response explicitly carries ``host_review_completion`` provenance.
    """

    def __init__(self, judge):
        if not callable(judge):
            raise ValueError('An independent review judge must be injected')
        self.judge = judge
        self._traces = []
        self.binding = {'schema_version': VERSION,
                        'adapter_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}

    def traces(self) -> list[dict]:
        return copy.deepcopy(self._traces)

    def __call__(self, kind: str, instructions: str, payload: dict) -> dict:
        if kind not in TASKS:
            return self.judge(kind, instructions, copy.deepcopy(payload))
        collection, key = TASKS[kind]
        items = payload.get(collection)
        if (not isinstance(items, list) or not items
                or any(not isinstance(row, dict) or not isinstance(row.get(key), str)
                       or not row[key] for row in items)):
            raise ValueError('Review completion requires complete requested source rows')
        ids = [row[key] for row in items]
        if len(ids) != len(set(ids)):
            raise ValueError('Review completion requested IDs must be unique')
        primary = self.judge(kind, instructions, copy.deepcopy(payload))
        trace = {**self.binding, 'task_kind': kind, 'requested_ids': ids,
                 'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest(),
                 'primary_payload_sha256': digest(payload),
                 'primary_response_sha256': digest(primary), 'primary_response': copy.deepcopy(primary),
                 'secondary_call_count': 0}
        rows = primary.get('judgments') if isinstance(primary, dict) else None
        if not isinstance(rows, list):
            trace['status'] = 'invalid_primary_shape_no_completion'
            self._traces.append(trace)
            return primary
        counts = _counts(rows, key)
        missing = [identifier for identifier in ids if counts[identifier] != 1]
        trace.update(omitted_ids=[identifier for identifier in ids if not counts[identifier]],
                     duplicate_ids=[identifier for identifier in ids if counts[identifier] > 1],
                     unexpected_ids=[identifier for identifier in counts if identifier not in ids])
        if not missing:
            trace['status'] = 'all_requested_ids_present_no_completion'
            self._traces.append(trace)
            return primary
        secondary_payload = copy.deepcopy(payload)
        secondary_payload[collection] = [copy.deepcopy(row) for row in items if row[key] in missing]
        secondary_payload['response_schema'] = _schema(kind, missing)
        secondary_kind = kind + '_missing_id_completion'
        trace.update(secondary_task_kind=secondary_kind, secondary_call_count=1,
                     secondary_requested_ids=missing,
                     secondary_payload_sha256=digest(secondary_payload),
                     original_source_row_sha256={row[key]: digest(row) for row in items if row[key] in missing})
        try:
            secondary = self.judge(secondary_kind, instructions, copy.deepcopy(secondary_payload))
            trace.update(secondary_response=copy.deepcopy(secondary), secondary_response_sha256=digest(secondary))
        except Exception as error:
            trace.update(status='secondary_error_no_repeat', secondary_error_type=type(error).__name__,
                         accepted_secondary_ids=[], remaining_missing_ids=missing)
            self._traces.append(trace)
            return {'judgments': copy.deepcopy(rows), 'host_review_completion': copy.deepcopy(trace)}
        secondary_rows = secondary.get('judgments') if isinstance(secondary, dict) else None
        secondary_rows = secondary_rows if isinstance(secondary_rows, list) else []
        secondary_counts = _counts(secondary_rows, key)
        accepted = [copy.deepcopy(row) for row in secondary_rows
                    if isinstance(row, dict) and isinstance(row.get(key), str)
                    and row[key] in missing and secondary_counts[row[key]] == 1 and _valid_row(row, kind)]
        accepted_ids = [row[key] for row in accepted]
        # Duplicate-ID primary rows are structurally unusable. Keep their exact
        # bytes in primary_response; remove them only in the explicit composition.
        composed = [copy.deepcopy(row) for row in rows if not
                    (isinstance(row, dict) and isinstance(row.get(key), str) and row[key] in accepted_ids)]
        composed.extend(accepted)
        trace.update(status='completed' if set(accepted_ids) == set(missing) else 'partial_or_unresolved_no_repeat',
                     accepted_secondary_ids=accepted_ids,
                     remaining_missing_ids=[identifier for identifier in missing if identifier not in accepted_ids],
                     secondary_duplicate_ids=[identifier for identifier, count in secondary_counts.items() if count > 1],
                     secondary_unexpected_ids=[identifier for identifier in secondary_counts if identifier not in missing],
                     composed_judgments_sha256=digest(composed))
        self._traces.append(trace)
        return {'judgments': composed, 'host_review_completion': copy.deepcopy(trace)}


def apply_precision_id_completion(original: dict, primary_payload: dict,
                                  composed_response: dict) -> dict:
    """Recompute full precision only; preserve original coverage and traces.

    ``composed_response`` must come from the adapter on the exact saved primary
    payload/response and a separately retained completion response. This helper
    starts no judge calls. The original traces remain an unchanged prefix; the
    distinct supplemental response is appended as its own trace. A completion's
    negative or semantic unknown remains negative or unknown without retry.
    """
    from .staged_node_evaluation import _metric, THRESHOLDS

    trace = composed_response.get('host_review_completion')
    if not isinstance(trace, dict) or trace.get('task_kind') != 'staged_nodes_precision':
        raise ValueError('An explicit precision ID-completion composition is required')
    if (trace.get('secondary_call_count') != 1 or trace.get('primary_payload_sha256') != digest(primary_payload)
            or original.get('thresholds') != THRESHOLDS
            or trace.get('secondary_task_kind') != 'staged_nodes_precision_missing_id_completion'
            or trace.get('instructions_sha256') != original['bindings']['prompts_sha256']['precision']):
        raise ValueError('Precision completion payload or fixed thresholds differ')
    matches = [(index, row) for index, row in enumerate(original['judge_calls'])
               if row['task_kind'] == 'staged_nodes_precision'
               and row['payload_sha256'] == digest(primary_payload)]
    if len(matches) != 1 or matches[0][1].get('response') != trace.get('primary_response'):
        raise ValueError('Completion does not bind an exact original precision response')
    index, primary_trace = matches[0]
    items = primary_payload['units']
    ids = [row['unit_id'] for row in items]
    if ids != primary_trace['requested_ids'] or trace.get('requested_ids') != ids:
        raise ValueError('Completion original requested IDs differ')
    counts = _counts(trace['primary_response']['judgments'], 'unit_id')
    eligible = [key for key in ids if counts[key] != 1]
    if trace.get('secondary_requested_ids') != eligible or not eligible:
        raise ValueError('Only structurally omitted or duplicated IDs may be completed')
    secondary_payload = copy.deepcopy(primary_payload)
    secondary_payload['units'] = [copy.deepcopy(row) for row in items if row['unit_id'] in eligible]
    secondary_payload['response_schema'] = _schema('staged_nodes_precision', eligible)
    if (trace.get('secondary_payload_sha256') != digest(secondary_payload)
            or trace.get('primary_response_sha256') != digest(trace['primary_response'])):
        raise ValueError('Completion original/full source subset binding differs')
    secondary = trace.get('secondary_response')
    secondary_rows = secondary.get('judgments') if isinstance(secondary, dict) else None
    secondary_rows = secondary_rows if isinstance(secondary_rows, list) else []
    if secondary is not None and trace.get('secondary_response_sha256') != digest(secondary):
        raise ValueError('Completion secondary response hash differs')
    secondary_counts = _counts(secondary_rows, 'unit_id')
    accepted = [copy.deepcopy(row) for row in secondary_rows if isinstance(row, dict)
                and isinstance(row.get('unit_id'), str) and row['unit_id'] in eligible
                and secondary_counts[row['unit_id']] == 1 and _valid_row(row, 'staged_nodes_precision')]
    if [row['unit_id'] for row in accepted] != trace.get('accepted_secondary_ids'):
        raise ValueError('Completion accepted IDs differ from actual secondary response')
    expected_composed = [copy.deepcopy(row) for row in trace['primary_response']['judgments'] if not
        (isinstance(row, dict) and isinstance(row.get('unit_id'), str)
         and row['unit_id'] in [new['unit_id'] for new in accepted])] + accepted
    if composed_response.get('judgments') != expected_composed:
        raise ValueError('Completion changed a protected original judgment')
    result = copy.deepcopy(original)
    precision_rows = result['precision']['judgments']
    by_id = {row['unit_id']: row for row in precision_rows}
    originals = []
    for judgment in accepted:
        key = judgment['unit_id']
        item = next(row for row in items if row['unit_id'] == key)
        claim = digest({'source': primary_payload['source_fingerprint'],
                        'unit': item['unit'], 'contexts': item['attached_contexts']})
        if (by_id[key].get('claim_sha256') != claim
                or trace['original_source_row_sha256'].get(key) != digest(item)):
            raise ValueError('Completion source/context claim differs from original accepted row')
        originals.append(copy.deepcopy(by_id[key]))
        axes = {axis: judgment[axis] for axis in AXES}
        status = ('unsupported' if 'fail' in axes.values() or judgment['status'] == 'unsupported'
                  else 'supported' if all(value == 'pass' for value in axes.values())
                  and judgment['status'] == 'supported' else 'unknown')
        by_id[key].update(status=status, axes=axes, reason=judgment['reason'])
    result['precision'].update(_metric(precision_rows, 'supported', 'unsupported'))
    for dimension, strata in result['precision']['strata'].items():
        for label in strata:
            strata[label] = _metric([row for row in precision_rows
                if (row[dimension] if row[dimension] in strata else 'unknown') == label],
                'supported', 'unsupported')
    lower = result['precision']['bounds']['lower']
    result['checks']['precision'] = lower is not None and lower >= THRESHOLDS['precision']
    result['passed'] = all(result['checks'].values())
    result['status'] = 'passed' if result['passed'] else 'failed'
    supplemental_trace = {'task_kind': trace['secondary_task_kind'],
        'payload_sha256': trace['secondary_payload_sha256'], 'requested_ids': eligible,
        'source': 'separately retained ID-completion response',
        'completes_original_judge_call_index': index}
    if secondary is None:
        supplemental_trace.update(status='judge_error', exception_type=trace.get('secondary_error_type'))
    else:
        supplemental_trace.update(status='completed', response=copy.deepcopy(secondary),
            omitted_ids=[key for key in eligible if secondary_counts[key] != 1],
            duplicate_ids=[key for key, count in secondary_counts.items() if count > 1],
            unexpected_ids=[key for key in secondary_counts if key not in eligible])
    result['judge_calls'].append(supplemental_trace)
    result['judge_call_count'] = len(result['judge_calls'])
    result['precision_id_completion'] = {
        'schema_version': VERSION, 'original_evaluation_sha256': digest(original),
        'adapter_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'original_precision_rows': originals, 'applied_unit_ids': [row['unit_id'] for row in accepted],
        'completion': copy.deepcopy(trace), 'full_precision_denominator': len(precision_rows),
        'coverage_rows_and_original_traces_unchanged': True,
        'score_method': 'Original full accepted-unit precision with one structurally missing-ID completion; original independent coverage judgments retained.'}
    return result
