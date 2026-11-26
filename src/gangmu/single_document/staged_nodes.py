"""Two-round, node-only extraction with an injected semantic judge.

No client, key, graph finalizer or dependency extractor is invoked here. Source
coordinates and original text are always supplied/replayed by the host. Proofs
remain node-stage records; this artifact is not a formal E2 graph publication.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
import re
from typing import Callable, Any

from gangmu.representation.model import Context, Unit, digest
from gangmu.representation.sources import SourceError
from .rule_nodes import RuleNodeExtractor, RULE_VERSION, SCHEMA as RULE_SCHEMA, _proof_identity

VERSION = 'staged-nodes-v3'
SCHEMA = 'gangmu.e4.staged_nodes.v1'
TASK_ROUND1 = 'staged_nodes_round1'
TASK_ROUND2 = 'staged_nodes_round2'
CATEGORIES = ('definition', 'statement', 'expression', 'proof')
PROPOSAL_KEYS = {'category', 'label', 'field', 'subfield', 'context', 'reason'}
PROMPT_ROOT = Path(__file__).resolve().parents[3] / 'prompts/single_document/staged_nodes'
Judge = Callable[[str, str, dict], dict]


@dataclass(frozen=True)
class NodeOptions:
    round1_batch_chars: int = 18000
    round1_max_items: int = 30
    round2_chunk_chars: int = 18000
    context_window_chars: int = 48000
    context_catalog_chars: int = 5000
    max_context_requests: int = 2
    context_query_results: int = 4

    @classmethod
    def parse(cls, value=None):
        if value is None:
            return cls()
        if isinstance(value, cls):
            result = value
        elif isinstance(value, dict):
            if set(value) - set(cls.__dataclass_fields__):
                raise SourceError('unknown node-only option')
            result = cls(**value)
        else:
            raise SourceError('node-only options must be a dict or NodeOptions')
        for key, number in asdict(result).items():
            if type(number) is not int or number < (0 if key == 'max_context_requests' else 1):
                raise SourceError('node-only sizes must be positive integers')
        if result.max_context_requests > 4:
            raise SourceError('at most four source-context followups are allowed per task')
        return result


def _union(intervals):
    result = []
    for a, b in sorted(intervals):
        if result and a <= result[-1][1]:
            result[-1] = (result[-1][0], max(b, result[-1][1]))
        else:
            result.append((a, b))
    return result


def _span_key(span):
    return (span['file'], span.get('occurrence'), span['char_start'], span['char_end'])


def _merge_spans(spans):
    """Preserve traversal order and repeated includes; merge only touching slices."""
    result = []
    for span in spans:
        span = dict(span)
        if (result and (span['file'], span.get('occurrence')) ==
                (result[-1]['file'], result[-1].get('occurrence')) and
                result[-1]['char_start'] <= span['char_start'] <= result[-1]['char_end']):
            result[-1]['char_end'] = max(result[-1]['char_end'], span['char_end'])
        elif not result or _span_key(span) != _span_key(result[-1]):
            result.append(span)
    return result


def _object(properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties),
            'additionalProperties': False}


def _array(item):
    return {'type': 'array', 'items': item}


def _text(nullable=False):
    return {'type': ['string', 'null'] if nullable else 'string', 'minLength': 1}


def _check_shape(value, schema):
    """Apply our small strict JSON schema locally, including fake/offline judges."""
    if 'anyOf' in schema:
        for variant in schema['anyOf']:
            try:
                _check_shape(value, variant)
                return
            except SourceError:
                pass
        raise SourceError('judge response violates its strict response schema')
    kinds = schema.get('type', [])
    kinds = [kinds] if isinstance(kinds, str) else kinds
    matches = {'object': isinstance(value, dict), 'array': isinstance(value, list),
               'string': isinstance(value, str), 'null': value is None,
               'boolean': type(value) is bool}
    if not any(matches.get(kind, False) for kind in kinds):
        raise SourceError('judge response violates its strict response schema')
    if 'enum' in schema and value not in schema['enum']:
        raise SourceError('judge response violates its strict response schema')
    if isinstance(value, dict):
        properties = schema['properties']
        if set(schema.get('required', [])) - set(value) or (
                not schema.get('additionalProperties', True) and set(value) - set(properties)):
            raise SourceError('judge response violates its strict response schema')
        for key, child in value.items():
            if key in properties:
                _check_shape(child, properties[key])
    elif isinstance(value, list):
        for child in value:
            _check_shape(child, schema['items'])
    elif isinstance(value, str) and (len(value) < schema.get('minLength', 0) or
                                   (schema.get('minLength', 0) and not value.strip())):
        raise SourceError('judge response violates its strict response schema')


def _adds_external_source(context_spans, unit_spans):
    for context in context_spans:
        bounds = _union([(span['char_start'], span['char_end']) for span in unit_spans
                         if (span['file'], span.get('occurrence')) ==
                         (context['file'], context.get('occurrence'))])
        if not any(start <= context['char_start'] < context['char_end'] <= end for start, end in bounds):
            return True
    return False


def _starts_case(literal):
    wrappers = (r'(?:\\(?:noindent|smallskip|medskip|bigskip|par|bfseries|itshape|scshape)\b\s*|'
                r'\\(?:textbf|textit|emph|textsc)\s*\{\s*|'
                r'\{\s*\\(?:bf|it|sc|bfseries|itshape|scshape)\b\s*)')
    return bool(re.match(r'\s*' + wrappers + r'*Case\b', literal, re.I))


def response_schema(task_kind, payload):
    """Strict producer shapes; selectors use host IDs/quotes, never coordinates."""
    source_ids = [r['source_id'] for r in payload['context_menu']]
    selector = _object({'source_id': {'type': 'string', 'enum': source_ids}, 'quote': _text(True)})
    context = _object({'description': _text(), 'self_contained': {'type': 'boolean'},
        'selectors': _array(selector), 'assumptions': {'type': ['array', 'null'], 'items': _text()},
        'symbols': {'type': ['array', 'null'], 'items': _text()}})
    requests = _array(_object({'source_id': _text(True), 'query': _text(True)}))
    fields = {'category': {'type': ['string', 'null'], 'enum': [*CATEGORIES, None]},
        'label': _text(True), 'field': _text(True), 'subfield': _text(True),
        'context': {'anyOf': [context, {'type': 'null'}]}, 'reason': _text()}
    if task_kind == TASK_ROUND1:
        item = _object({'candidate_id': {'type': 'string', 'enum': [r['candidate_id'] for r in payload['candidates']]},
            'decision': {'type': 'string', 'enum': ['accept', 'reject', 'context_only', 'uncertain']}, **fields})
        return _object({'decisions': _array(item), 'context_requests': requests})
    if task_kind != TASK_ROUND2:
        raise SourceError('unknown node producer task kind')
    paragraph_ids = [r['source_id'] for r in payload['paragraphs']]
    node_selector = _object({'source_id': {'type': 'string', 'enum': paragraph_ids}, 'quote': _text(True)})
    return _object({'chunk_id': {'type': 'string', 'enum': [payload['chunk_id']]},
        'decision': {'type': 'string', 'enum': ['nodes', 'context_only', 'no_supplement', 'uncertain']},
        'nodes': _array(_object({'selectors': _array(node_selector), **fields})),
        'reason': _text(), 'context_requests': requests})


class _Pipeline:
    def __init__(self, project, rule_result, judge, options):
        self.project, self.judge, self.options = project, judge, options
        project.assert_unchanged()
        if not callable(judge):
            raise SourceError('a semantic judge must be injected explicitly')
        if (not isinstance(rule_result, dict) or rule_result.get('schema_version') != RULE_SCHEMA
                or rule_result.get('rule_version') != RULE_VERSION
                or rule_result.get('source_fingerprint') != project.fingerprint
                or rule_result.get('paper_id') != project.paper_id):
            raise SourceError('rule result is not bound to the pinned source project')
        if {'gold', 'rule_reference', 'core_reference', 'precision_report', 'scores'} & set(rule_result):
            raise SourceError('producer input cannot contain evaluation references or answers')
        self.rule = copy.deepcopy(rule_result)
        self.rule_sha = digest(rule_result)
        self.candidates = self.rule.get('candidates')
        if not isinstance(self.candidates, list):
            raise SourceError('rule result must provide its full candidate list')
        self.helper = RuleNodeExtractor(project)  # read-only source mask/mapping; no rule rerun
        self.rows, self.paragraphs = {}, []
        seen = set()
        for candidate in self.candidates:
            if (not isinstance(candidate, dict) or not isinstance(candidate.get('source_spans'), list)
                    or not candidate['source_spans'] or not isinstance(candidate.get('reading_ranges'), list)
                    or not candidate['reading_ranges'] or not isinstance(candidate.get('scope_range'), dict)):
                raise SourceError('rule candidate has malformed source boundary metadata')
            identifier = candidate.get('candidate_id')
            if not isinstance(identifier, str) or not identifier or identifier in seen:
                raise SourceError('rule candidate IDs must be unique nonempty strings')
            seen.add(identifier)
            spans = [project.normalize_span(s) for s in candidate['source_spans']]
            literal = [project.resolve_span(s) for s in spans]
            if literal != candidate.get('original_text'):
                raise SourceError('rule candidate original text differs from pinned source')
            if any(not isinstance(r, dict) or set(r) != {'char_start', 'char_end'}
                   for r in [*candidate['reading_ranges'], candidate['scope_range']]):
                raise SourceError('rule candidate has malformed reading ranges')
            ranges = [(r['char_start'], r['char_end']) for r in candidate['reading_ranges']]
            if any(type(a) is not int or type(b) is not int or not 0 <= a < b <= len(project.reading)
                   for a, b in ranges):
                raise SourceError('rule candidate has invalid reading ranges')
            if self.helper._segments(ranges) != spans:
                raise SourceError('rule reading ranges and original source spans disagree')
            expected_identifier = 'rule-node:' + digest({'rule_version': RULE_VERSION,
                'source': project.fingerprint, 'spans': spans, 'family': candidate.get('structural_family')})[:24]
            if identifier != expected_identifier:
                raise SourceError('rule candidate ID no longer binds its complete original source')
            if any(ranges[index][1] > ranges[index + 1][0] for index in range(len(ranges) - 1)):
                raise SourceError('rule candidate source components are out of order or overlap')
            scope = candidate['scope_range']
            if (type(scope['char_start']) is not int or type(scope['char_end']) is not int
                    or not 0 <= scope['char_start'] < scope['char_end'] <= len(project.reading)
                    or any(not scope['char_start'] <= a < b <= scope['char_end'] for a, b in ranges)):
                raise SourceError('rule candidate has invalid complete scope range')
            self.rows[identifier] = {'source_id': identifier, 'kind': 'rule_candidate',
                'source_spans': spans, 'original_text': literal, 'reading_ranges': candidate['reading_ranges'],
                'title': candidate.get('title'), 'category_hint': candidate.get('category_hint')}
        self.protected = _union([(r['char_start'], r['char_end']) for c in self.candidates for r in c['reading_ranges']])
        self._paragraphs()
        self._background_sources()
        self.units, self.contexts, self.provenance = {}, {}, {}
        self.unit_keys, self.candidate_units = {}, {}
        self.unresolved = []
        self.round1 = {'batches': [], 'decisions': []}
        self.round2 = {'chunks': [], 'decisions': []}
        self.calls = []
        self.prompts = {TASK_ROUND1: (PROMPT_ROOT / 'round1.md').read_text(encoding='utf-8'),
                        TASK_ROUND2: (PROMPT_ROOT / 'round2.md').read_text(encoding='utf-8')}
        self.binding = {'producer_version': VERSION, 'producer_code_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'source_fingerprint_sha256': digest(project.fingerprint), 'rule_result_sha256': self.rule_sha,
            'options': asdict(options), 'prompt_sha256': {k: hashlib.sha256(v.encode()).hexdigest() for k, v in self.prompts.items()}}

    def _paragraphs(self):
        text, mask = self.project.reading, self.helper._mask
        first, last = 0, len(text)
        for event in self.project.structure_data.get('environments', []):
            if event.get('activity') == 'inactive':
                continue
            if event.get('name') == 'document':
                opening = self.helper._positions(event['source'], end=True, occurrence=event.get('file_occurrence'))
                closing = self.helper._positions(event['end_source'], occurrence=event.get('file_occurrence')) if event.get('end_source') else []
                if opening:
                    first = min(opening)
                if closing:
                    last = min(closing)
        # A nested file's endinput does not end the main document, and authors
        # may place mathematical appendices after the bibliography.
        for ending in re.finditer(r'\\(?:bye|endinput)\b', mask[first:last]):
            position = first + ending.start()
            spans = self.helper._segments([(position, first + ending.end())])
            if spans and spans[0]['file'] == self.project.main_file and spans[0].get('occurrence') == 'main':
                last = position
                break
        cuts = {first, last}
        excluded = list(self.protected)
        for event in self.project.structure_data.get('environments', []):
            if event.get('name') == 'thebibliography' and event.get('activity') != 'inactive':
                interval = self.helper._event_range(event)
                if interval:
                    excluded.append(interval)
        for section in self.project.structure_data.get('sections', []):
            if section.get('activity') == 'inactive':
                continue
            interval = self.helper._event_range(section)
            if interval and first <= interval[0] < last:
                excluded.append((interval[0], min(last, interval[1])))
        for row in self.project.source_map:
            if row['source']['file'].lower().endswith(('.sty', '.cls', '.bib', '.bbl')):
                excluded.append((row['reading_char_start'], row['reading_char_end']))
        excluded = _union(excluded)
        for a, b in excluded:
            if a < last and first < b:
                cuts.update((max(a, first), min(b, last)))
        bounds = sorted(cuts)
        self.body_range = {'char_start': first, 'char_end': last}
        self.gaps = []
        for left, right in zip(bounds, bounds[1:]):
            if any(a <= left and right <= b for a, b in excluded):
                continue
            self.gaps.append({'char_start': left, 'char_end': right})
            # Split only at blank lines outside literal braces/environments/math.
            boundaries, depth, environments, display = [left], 0, 0, 0
            token_pattern = r'\\begin\s*\{[^{}]+\}|\\end\s*\{[^{}]+\}|\\\[|\\\]|\\\(|\\\)|\\.|\$\$?|[{}]|\r?\n[ \t]*\r?\n'
            for token in re.finditer(token_pattern, mask[left:right]):
                value = token.group()
                if value.startswith('\\begin'):
                    environments += 1
                elif value.startswith('\\end'):
                    environments = max(0, environments - 1)
                elif value in ('\\[', '\\('):
                    display += 1
                elif value in ('\\]', '\\)'):
                    display = max(0, display - 1)
                elif value in ('$', '$$'):
                    display = 0 if display else 1
                elif value == '{':
                    depth += 1
                elif value == '}':
                    depth = max(0, depth - 1)
                elif '\n' in value and not (depth or environments or display):
                    boundaries.append(left + token.end())
            boundaries.append(right)
            for a, b in zip(boundaries, boundaries[1:]):
                if not mask[a:b].strip():
                    continue
                spans = self.helper._segments([(a, b)])
                if not spans:
                    continue
                literal = [self.project.resolve_span(s) for s in spans]
                if ''.join(literal) != text[a:b]:
                    raise SourceError('paragraph does not replay through pinned source mapping')
                identifier = 'staged-paragraph:' + digest({'source': self.project.fingerprint, 'spans': spans})[:24]
                before = [c for c in self.candidates if c['scope_range']['char_end'] <= a]
                after = [c for c in self.candidates if c['scope_range']['char_start'] >= b]
                row = {'source_id': identifier, 'kind': 'unheaded_paragraph', 'source_spans': spans,
                    'original_text': literal, 'reading_ranges': [{'char_start': a, 'char_end': b}],
                    'anchor_before': max(before, key=lambda c: c['scope_range']['char_end'])['candidate_id'] if before else None,
                    'anchor_after': min(after, key=lambda c: c['scope_range']['char_start'])['candidate_id'] if after else None}
                self.rows[identifier] = row
                self.paragraphs.append(row)

    def _background_sources(self):
        # Original preamble definitions can be necessary to read an author's
        # macro notation. They are Context retrieval sources, never round2 tasks.
        first = self.body_range['char_start']
        spans = [span for span in self.helper._segments([(0, first)])
                 if not span['file'].lower().endswith(('.sty', '.cls', '.bib', '.bbl'))] if first else []
        if spans:
            ranges = []
            for span in spans:
                starts = self.helper._positions(span, occurrence=span.get('occurrence'))
                ends = self.helper._positions(span, end=True, occurrence=span.get('occurrence'))
                if len(starts) != 1 or len(ends) != 1:
                    raise SourceError('preamble source has ambiguous reading occurrence')
                ranges.append({'char_start': starts[0], 'char_end': ends[0]})
            identifier = 'staged-context:' + digest({'source': self.project.fingerprint, 'spans': spans})[:24]
            self.rows[identifier] = {'source_id': identifier, 'kind': 'source_preamble', 'source_spans': spans,
                'original_text': [self.project.resolve_span(span) for span in spans],
                'reading_ranges': ranges}

    @staticmethod
    def _size(row):
        return sum(len(t) for t in row['original_text'])

    def _groups(self, rows, limit, max_items=None):
        group, size = [], 0
        for row in rows:
            amount = self._size(row)
            if group and (size + amount > limit or (max_items is not None and len(group) >= max_items)):
                yield group
                group, size = [], 0
            group.append(row)
            size += amount
        if group:
            yield group

    def _context_menu(self, targets):
        points = [(r['char_start'], r['char_end']) for t in targets for r in t['reading_ranges']]
        nearby = sorted(self.rows.values(), key=lambda r: min(abs(x['char_start'] - a) for x in r['reading_ranges'] for a, _ in points))
        result, size = [], 0
        # Initial background and local antecedents are source packets, not presumed Context.
        target_ids = {row['source_id'] for row in targets}
        for row in [*self.paragraphs[:3], *nearby]:
            if row['source_id'] in target_ids:
                continue
            if row['source_id'] in {r['source_id'] for r in result}:
                continue
            if size >= self.options.context_window_chars and result:
                break
            result.append(copy.deepcopy(row))
            size += self._size(row)
        for row in targets:
            alias = copy.deepcopy(row)
            alias.pop('original_text')
            alias['full_text_in'] = 'candidates' if row['kind'] == 'rule_candidate' else 'paragraphs'
            result.append(alias)
        return result

    def _catalog(self):
        result, amount = [], 0
        for row in self.rows.values():
            preview = ' '.join(''.join(row['original_text']).split())[:100]
            entry = {'source_id': row['source_id'], 'preview': preview, 'full_text_supplied': False}
            size = len(preview) + len(row['source_id'])
            if amount + size > self.options.context_catalog_chars:
                break
            result.append(entry)
            amount += size
        return result

    def _query(self, query):
        terms = set(re.findall(r'[\w]+', query.lower()))
        if not terms:
            return []
        ranked = []
        for row in self.rows.values():
            words = set(re.findall(r'[\w]+', ''.join(row['original_text']).lower()))
            score = len(words & terms)
            if score:
                ranked.append((score / len(terms), row['source_id']))
        return [self.rows[key] for _, key in sorted(ranked, key=lambda x: (-x[0], x[1]))[:self.options.context_query_results]]

    def _ask(self, task_kind, payload):
        payload = copy.deepcopy(payload)
        payload['producer_binding'] = self.binding
        payload['context_catalog'] = self._catalog()
        for followup in range(self.options.max_context_requests + 1):
            payload['source_context_followup'] = followup
            payload['response_schema'] = response_schema(task_kind, payload)
            instructions = self.prompts[task_kind]
            request_sha = digest({'task_kind': task_kind, 'instructions': instructions, 'payload': payload})
            try:
                response = self.judge(task_kind, instructions, copy.deepcopy(payload))
                if not isinstance(response, dict):
                    raise SourceError('judge response must be a JSON object')
                digest(response)
            except Exception as error:
                self.calls.append({'task_kind': task_kind, 'input_sha256': request_sha, 'status': 'technical_failure',
                                   'error_type': type(error).__name__, 'source_context_followup': followup})
                raise SourceError('semantic judge task failed') from None
            self.calls.append({'task_kind': task_kind, 'input_sha256': request_sha,
                'response_sha256': digest(response), 'status': 'returned', 'source_context_followup': followup})
            _check_shape(response, payload['response_schema'])
            requests = response.get('context_requests')
            if not isinstance(requests, list):
                raise SourceError('judge omitted context request disposition')
            if not requests:
                return response, {r['source_id'] for r in payload['context_menu']}
            if followup == self.options.max_context_requests:
                raise SourceError('required source context remains unresolved')
            added = []
            for request in requests:
                if not isinstance(request, dict) or set(request) != {'source_id', 'query'}:
                    raise SourceError('invalid source-context request')
                identifier, query = request['source_id'], request['query']
                if identifier is not None:
                    if identifier not in self.rows:
                        raise SourceError('unknown requested source-context ID')
                    added.append(self.rows[identifier])
                if query is not None:
                    if not isinstance(query, str) or not query.strip():
                        raise SourceError('source-context query must be nonempty text')
                    added.extend(self._query(query))
                if identifier is None and query is None:
                    raise SourceError('empty source-context request')
            existing = {r['source_id'] for r in payload['context_menu']}
            fresh = [r for r in added if r['source_id'] not in existing]
            if not fresh:
                raise SourceError('source-context request found no new source')
            for row in fresh:
                if row['source_id'] not in existing:
                    payload['context_menu'].append(copy.deepcopy(row))
                    existing.add(row['source_id'])
            payload['previous_context_requests'] = copy.deepcopy(requests)
        raise AssertionError('unreachable context followup')

    def _select(self, selectors, allowed, *, coherent=False):
        if not isinstance(selectors, list) or not selectors:
            raise SourceError('a source selection must be a nonempty array')
        spans, identifiers = [], []
        for selector in selectors:
            if not isinstance(selector, dict) or set(selector) != {'source_id', 'quote'}:
                raise SourceError('source selectors accept host ID and exact quote only')
            identifier = selector['source_id']
            if identifier not in allowed or identifier not in self.rows:
                raise SourceError('source selector was not fully read in this task')
            row, quote = self.rows[identifier], selector['quote']
            if quote is None:
                selected = copy.deepcopy(row['source_spans'])
            else:
                if not isinstance(quote, str) or not quote.strip():
                    raise SourceError('source quote must be nonempty exact text')
                literal = ''.join(row['original_text'])
                start = literal.find(quote)
                if start < 0 or literal.find(quote, start + 1) >= 0:
                    raise SourceError('source quote must occur exactly once in its selected source ID')
                end, cursor, selected = start + len(quote), 0, []
                for span, text in zip(row['source_spans'], row['original_text']):
                    a, b = max(start, cursor), min(end, cursor + len(text))
                    if a < b:
                        selected.append({**span, 'char_start': span['char_start'] + a - cursor,
                                         'char_end': span['char_start'] + b - cursor})
                    cursor += len(text)
            spans.extend(self.project.normalize_span(s) for s in selected)
            identifiers.append(identifier)
        positions = []
        for span in spans:
            starts = self.helper._positions(span, occurrence=span.get('occurrence'))
            ends = self.helper._positions(span, end=True, occurrence=span.get('occurrence'))
            if len(starts) != 1 or len(ends) != 1:
                raise SourceError('source selection has ambiguous reading occurrence')
            positions.append((starts[0], ends[0], span))
        positions.sort(key=lambda item: (item[0], item[1]))
        if coherent:
            if not any(self.helper._mask[start:end].strip() for start, end, _ in positions):
                raise SourceError('a second-round node cannot be wholly inactive, commented or literal source')
            cursor = positions[0][1]
            for start, end, _ in positions[1:]:
                if cursor < start and self.helper._mask[cursor:start].strip():
                    raise SourceError('second-round task selectors skip intervening effective source')
                cursor = max(cursor, end)
        spans = _merge_spans([span for _, _, span in positions])
        identifiers = sorted(set(identifiers), key=lambda key: self.rows[key]['reading_ranges'][0]['char_start'])
        return spans, identifiers

    def _accept(self, proposal, spans, source_ids, round_number, candidate_id=None, allowed_context=None):
        expected = PROPOSAL_KEYS | ({'candidate_id', 'decision'} if candidate_id else {'selectors'})
        if set(proposal) != expected or not isinstance(proposal.get('reason'), str) or not proposal['reason'].strip():
            raise SourceError('accepted proposal must follow the complete host response contract')
        if proposal.get('category') not in CATEGORIES or not isinstance(proposal.get('label'), str) or not proposal['label'].strip():
            raise SourceError('accepted node needs one E2 category and a nonempty label')
        context = proposal.get('context')
        if not isinstance(context, dict) or set(context) != {'description', 'self_contained', 'selectors', 'assumptions', 'symbols'}:
            raise SourceError('accepted node needs a complete source-bound Context decision')
        if type(context['self_contained']) is not bool or not isinstance(context['selectors'], list):
            raise SourceError('Context self-contained/selector decision is invalid')
        if context['selectors']:
            context_spans, context_ids = self._select(context['selectors'], allowed_context or set())
        elif context['self_contained']:
            context_spans, context_ids = spans, []
        else:
            raise SourceError('necessary background was not actually attached to the unit Context')
        if not context['self_contained'] and not _adds_external_source(context_spans, spans):
            raise SourceError('external Context decision only selected the unit own source')
        # Local objects/assumptions/symbols described by Context are supported
        # by the unit itself as well as the genuinely external antecedents.
        # Validate the external-source obligation before adding the own source.
        ordered_context = []
        for span in [*context_spans, *spans]:
            positions = self.helper._positions(span, occurrence=span.get('occurrence'))
            if len(positions) != 1:
                raise SourceError('Context source union has ambiguous reading occurrence')
            ordered_context.append((positions[0], span))
        ordered_context.sort(key=lambda item: (item[0], item[1]['char_end']))
        context_spans = _merge_spans([span for _, span in ordered_context])
        if not isinstance(context['description'], str) or not context['description'].strip():
            raise SourceError('Context description must be nonempty')
        for name in ('assumptions', 'symbols'):
            value = context[name]
            if value is not None and (not isinstance(value, list) or any(not isinstance(v, str) or not v.strip() for v in value)):
                raise SourceError('Context assumptions/symbols must be text arrays or null')
        source_key = digest({'source': self.project.fingerprint, 'spans': spans})
        identifier = 'unit:' + source_key[:24]
        source = [self.project.source(s) for s in spans]
        context_value = {'document': self.project.document.id, 'parent': None,
            'sources': [self.project.source(s).model_dump(mode='json') for s in context_spans],
            'description': context['description'], 'assumptions': context['assumptions'], 'symbols': context['symbols']}
        context_identifier = 'context:' + digest(context_value)[:24]
        context_record = Context(id=context_identifier, document=self.project.document.id, parent=None,
            sources=tuple(self.project.source(s) for s in context_spans), description=context['description'],
            assumptions=None if context['assumptions'] is None else tuple(context['assumptions']),
            symbols=None if context['symbols'] is None else tuple(context['symbols'])).model_dump(mode='json')
        unit = Unit(id=identifier, category=proposal['category'], label=proposal['label'].strip(),
            document=self.project.document.id, context=context_identifier, field=proposal.get('field'),
            subfield=proposal.get('subfield'), sources=tuple(source),
            original_text=tuple(self.project.resolve_span(s) for s in spans), semantics={}).model_dump(mode='json')
        if source_key in self.unit_keys:
            previous = self.units[self.unit_keys[source_key]]
            if any(previous[key] != unit[key] for key in ('category', 'label', 'field', 'subfield', 'context')):
                raise SourceError('duplicate source node has conflicting classification or Context')
            record = self.provenance[identifier]
            record['rounds'] = sorted(set(record['rounds']) | {round_number})
            if candidate_id:
                record['candidate_ids'] = list(dict.fromkeys([*record['candidate_ids'], candidate_id]))
            record['paragraph_ids'] = list(dict.fromkeys([*record['paragraph_ids'], *[s for s in source_ids if s.startswith('staged-paragraph:')]]))
        else:
            self.unit_keys[source_key] = identifier
            self.units[identifier], self.contexts[context_identifier] = unit, context_record
            self.provenance[identifier] = {'unit_id': identifier, 'round': round_number, 'rounds': [round_number],
                'candidate_ids': [candidate_id] if candidate_id else [],
                'paragraph_ids': [s for s in source_ids if s.startswith('staged-paragraph:')],
                'context_source_ids': context_ids, 'context_self_contained': context['self_contained'],
                'reason': proposal['reason'], 'fixed_rule_boundary': candidate_id is not None}
        if candidate_id:
            self.candidate_units[candidate_id] = identifier
        return identifier

    def _round1(self):
        rows = [self.rows[c['candidate_id']] for c in self.candidates]
        for batch in self._groups(rows, self.options.round1_batch_chars, self.options.round1_max_items):
            ids = [r['source_id'] for r in batch]
            candidates = [{**copy.deepcopy(c), 'accepted': False} for c in self.candidates if c['candidate_id'] in ids]
            payload = {'round': 1, 'batch_id': 'round1:' + digest(ids)[:24], 'candidates': candidates,
                'context_menu': self._context_menu(batch), 'source_fingerprint': self.project.fingerprint,
                'fixed_complete_source': True}
            receipt = {'batch_id': payload['batch_id'], 'candidate_ids': ids,
                'source_spans': [s for r in batch for s in r['source_spans']],
                'source_text_sha256': digest([r['original_text'] for r in batch]),
                'batch_characters': sum(self._size(r) for r in batch), 'status': 'judged'}
            try:
                response, allowed = self._ask(TASK_ROUND1, payload)
                if set(response) != {'decisions', 'context_requests'} or not isinstance(response['decisions'], list):
                    raise SourceError('invalid first-round response shape')
                decisions = response['decisions']
                if (len(decisions) != len(ids) or {d.get('candidate_id') for d in decisions if isinstance(d, dict)} != set(ids)):
                    raise SourceError('first round must decide every candidate exactly once')
            except SourceError as error:
                receipt['status'] = 'incomplete'
                self.unresolved.append({'round': 1, 'batch_id': payload['batch_id'], 'reason': str(error)})
                decisions = [{'candidate_id': identifier, 'decision': 'uncertain', 'reason': str(error)} for identifier in ids]
                allowed = set()
            self.round1['batches'].append(receipt)
            for decision in decisions:
                identifier = decision['candidate_id']
                record = {'candidate_id': identifier, 'decision': decision.get('decision'), 'reason': decision.get('reason'),
                          'unit_id': None, 'batch_id': payload['batch_id']}
                try:
                    if decision.get('decision') not in {'accept', 'reject', 'context_only', 'uncertain'}:
                        raise SourceError('invalid mathematical-purpose decision')
                    if decision['decision'] == 'accept':
                        record['unit_id'] = self._accept(decision, self.rows[identifier]['source_spans'], [identifier],
                                                       1, identifier, allowed)
                    elif decision['decision'] == 'uncertain':
                        self.unresolved.append({'round': 1, 'candidate_id': identifier, 'reason': decision.get('reason')})
                except (SourceError, ValueError) as error:
                    record.update(decision='uncertain', reason=str(error))
                    self.unresolved.append({'round': 1, 'candidate_id': identifier, 'reason': str(error)})
                self.round1['decisions'].append(record)

    def _round2(self):
        for rows in self._groups(self.paragraphs, self.options.round2_chunk_chars):
            ids = [r['source_id'] for r in rows]
            chunk_id = 'round2:' + digest(ids)[:24]
            payload = {'round': 2, 'chunk_id': chunk_id, 'paragraphs': copy.deepcopy(rows),
                'context_menu': self._context_menu(rows), 'source_fingerprint': self.project.fingerprint,
                'existing_units': [{'unit_id': u['id'], 'category': u['category'], 'label': u['label'],
                                    'sources': u['sources']} for u in self.units.values()]}
            receipt = {'chunk_id': chunk_id, 'paragraph_ids': ids,
                'source_spans': [s for r in rows for s in r['source_spans']],
                'source_text_sha256': digest([r['original_text'] for r in rows]),
                'chunk_characters': sum(self._size(r) for r in rows), 'status': 'judged'}
            record = {'chunk_id': chunk_id, 'decision': 'uncertain', 'unit_ids': [], 'reason': None}
            before = copy.deepcopy((self.units, self.contexts, self.provenance, self.unit_keys, self.candidate_units))
            try:
                response, allowed = self._ask(TASK_ROUND2, payload)
                if (set(response) != {'chunk_id', 'decision', 'nodes', 'reason', 'context_requests'}
                        or response['chunk_id'] != chunk_id or response['decision'] not in {
                            'nodes', 'context_only', 'no_supplement', 'uncertain'}
                        or not isinstance(response['nodes'], list)
                        or (bool(response['nodes']) != (response['decision'] == 'nodes'))):
                    raise SourceError('invalid second-round chunk disposition')
                record.update(decision=response['decision'], reason=response['reason'])
                if response['decision'] == 'uncertain':
                    self.unresolved.append({'round': 2, 'chunk_id': chunk_id, 'reason': response['reason']})
                for proposal in response['nodes']:
                    spans, selected_ids = self._select(proposal.get('selectors'), set(ids), coherent=True)
                    literal = ''.join(self.project.resolve_span(s) for s in spans)
                    if (_starts_case(literal) or _starts_case(''.join(self.rows[selected_ids[0]]['original_text']))):
                        raise SourceError('a proof Case is not a separate second-round task')
                    record['unit_ids'].append(self._accept(proposal, spans, selected_ids, 2, allowed_context=allowed))
                record['unit_ids'] = list(dict.fromkeys(record['unit_ids']))
            except (SourceError, ValueError) as error:
                self.units, self.contexts, self.provenance, self.unit_keys, self.candidate_units = before
                receipt['status'] = 'incomplete'
                record.update(decision='uncertain', unit_ids=[], reason=str(error))
                self.unresolved.append({'round': 2, 'chunk_id': chunk_id, 'reason': str(error)})
            self.round2['chunks'].append(receipt)
            self.round2['decisions'].append(record)

    def _proof_hints(self):
        result = []
        for candidate in self.candidates:
            proof_id = self.candidate_units.get(candidate['candidate_id'])
            if not proof_id or self.units[proof_id]['category'] != 'proof':
                continue
            hint = candidate.get('proof_attachment_hint') or {}
            heading = re.match(r'\s*\\begin\s*\{[^{}]+\}\s*\[([^\]]*)\]',
                               ''.join(candidate['original_text']))
            identity = _proof_identity(heading[1]) if heading else None
            reference_keys = list(hint.get('target_reference_keys', []))
            if identity and identity[0] == 'reference':
                reference_keys = [identity[1]]
                target_ids = [c['candidate_id'] for c in self.candidates
                              if any(identity[1] in label.get('keys', []) for label in c.get('labels', []))]
            else:
                target_ids = hint.get('target_candidate_ids', [])
            targets = [self.candidate_units[t] for t in target_ids
                       if t in self.candidate_units and self.units[self.candidate_units[t]]['category'] == 'statement']
            explicit = bool(hint.get('target_locator') or reference_keys)
            targets = sorted(set(targets))
            result.append({'proof_unit_id': proof_id, 'target_unit_ids': targets if explicit and len(targets) == 1 else [],
                'candidate_target_unit_ids': targets,
                'status': 'source_named_target_hint' if explicit and len(targets) == 1 else 'unresolved_attachment',
                'basis': 'literal_optional_heading_reference' if identity and identity[0] == 'reference' else hint.get('basis'),
                'source_spans': copy.deepcopy(candidate['source_spans']),
                'heading_title': candidate.get('title'), 'target_locator': hint.get('target_locator'),
                'target_reference_keys': reference_keys,
                'claim': 'Source attachment clue only, not a proves edge or proof-validity certificate.'})
        return result

    def run(self):
        self._round1()
        self._round2()
        self.project.assert_unchanged()
        units, contexts = list(self.units.values()), list(self.contexts.values())
        for unit in units:
            if unit['context'] not in self.contexts or unit['semantics'] != {}:
                raise SourceError('node-only E2 context binding is incomplete')
            for source, literal in zip(unit['sources'], unit['original_text']):
                if self.project.resolve_span(source) != literal:
                    raise SourceError('accepted node no longer replays its original source')
        return {'schema_version': SCHEMA, 'producer_version': VERSION,
            'paper_id': self.project.paper_id, 'source_fingerprint': copy.deepcopy(self.project.fingerprint),
            'rule_result_sha256': self.rule_sha, 'producer_binding': self.binding,
            'documents': [self.project.document.model_dump(mode='json')], 'contexts': contexts, 'units': units,
            'round1': self.round1, 'round2': self.round2,
            'unit_provenance': list(self.provenance.values()), 'proof_attachment_hints': self._proof_hints(),
            'rule_unresolved': copy.deepcopy(self.rule.get('unresolved', [])),
            'unresolved': self.unresolved, 'judgment_calls': self.calls,
            'body_reading_range': self.body_range, 'unheaded_reading_gaps': self.gaps,
            'summary': {'accepted_nodes': len(units), 'contexts': len(contexts),
                'by_category': dict(Counter(u['category'] for u in units)),
                'round1_candidates': len(self.candidates), 'round1_decisions': len(self.round1['decisions']),
                'round2_paragraphs': len(self.paragraphs), 'round2_chunks': len(self.round2['chunks']),
                'unresolved_count': len(self.unresolved), 'status': 'complete' if not self.unresolved else 'incomplete'},
            'accepted_graph': False, 'claim': 'Node-only source-bound extraction judgments; no formal graph, dependencies, proof closure or mathematical truth certification.'}


def extract_complete_nodes(project, rule_result, judge: Judge, options=None) -> dict[str, Any]:
    """Return node-stage E2-shaped records and two-round companion receipts.

    ``judge(task_kind, instructions, payload) -> dict`` is the only semantic
    boundary. The caller owns actual HTTP/ledger/cache and retry policy. The
    packet includes its strict response schema and complete producer binding.
    Inputs contain original source and rule candidates, never gold references.
    The caller supplies the accepted, pinned rule artifact; these source/ID
    replay gates do not replace the caller's verification of that artifact.
    """
    return _Pipeline(project, rule_result, judge, NodeOptions.parse(options)).run()
