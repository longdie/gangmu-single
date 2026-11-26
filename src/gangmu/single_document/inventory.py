"""Source-bound inventories for E4's two extraction rounds.

This module discovers syntax, not mathematical knowledge or dependency edges.
Every unresolved reference stays visible.  The bounded macro projection below
only substitutes literal arguments in simple author-defined reference wrappers;
it neither executes TeX nor silently treats arbitrary macro arguments as live.
"""
from __future__ import annotations

from collections import Counter
import copy
import re

from pylatexenc.latexwalker import LatexWalker, LatexWalkerEndOfStream

from gangmu.latex_structure import CITES, REFS, VERBATIM, normalize_key
from gangmu.representation.model import digest
from gangmu.representation.sources import SourceError

SCHEMA = 'gangmu.e4.source-inventory.v1'
MAX_PARAGRAPH = 6000
REFERENCE_COMMANDS = REFS | {'crefrange', 'Crefrange', 'vrefrange', 'Vrefrange',
    'refrange', 'eqrefrange', 'labelcref', 'labelcpageref', 'lcnamecref',
    'namecref', 'nameCref', 'namecrefs', 'nameCrefs'}
RANGE_COMMANDS = {'crefrange', 'Crefrange', 'vrefrange', 'Vrefrange', 'refrange', 'eqrefrange'}
RESULT_WORDS = re.compile(r'\b(theorem|lemma|proposition|corollary|definition|proof|'
    r'claim|conjecture|question|notation|remark|example|assumption)\b', re.I)
CORE_WORDS = re.compile(r'\b(theorem|lemma|proposition|corollary|definition|proof|claim|conjecture)\b', re.I)
PLAIN_HEADING = re.compile(r'(?m)^[ \t]*(?:\\noindent[ \t]*)?'
    r'((?:Theorem|Lemma|Proposition|Corollary|Definition|Proof|Claim|Conjecture)'
    r'(?:[ \t]+of[ \t]+(?:Theorem|Lemma|Proposition|Corollary))?'
    r'(?:[ \t]+\(?[A-Z]?\d+(?:\.\d+)*[a-z]?\)?)?[.:])', re.I)
NUMBERED_MENTION = re.compile(r'\b(Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|Equation)'
    r'(?:\s|~)+\(?([A-Z]?\d+(?:\.\d+)*[a-z]?)\)?', re.I)
PRINTED_NUMBERED_HEADING = re.compile(
    r'^(Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture)'
    r'(?:\s|~)+\(?([A-Z]?\d+(?:\.\d+)*[a-z]?)\)?(?=$|[\s.:({])', re.I)
# Dotted/appendix parenthesized locators are explicit candidate clues only.
# Bare single integers remain outside this bounded scanner (enumeration ambiguity).
PARENTHESIZED_LOCATOR = re.compile(r'\([ \t]*((?:[A-Z]\.\d+(?:\.\d+)*|\d+(?:\.\d+)+)[a-z]?)[ \t]*\)')
UNSAFE_MACRO = re.compile(r'\\(?:if\w*|else|fi|begin|end|input|include|catcode|csname|'
    r'expandafter|edef|xdef|def|gdef|let|futurelet|scantokens|loop|repeat)\b')
RENDER_ARGUMENT_COMMANDS = {'textbf', 'textit', 'textrm', 'textsf', 'texttt', 'emph',
    'ensuremath', 'mbox', 'hbox', 'text', 'mathrm', 'mathbf', 'mathit'}


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise SourceError(f'{name} must be an integer >= {minimum}')
    return value


def _span(value, occurrence=None):
    result = {k: value[k] for k in ('file', 'char_start', 'char_end')}
    if occurrence is not None:
        result['occurrence'] = occurrence
    elif value.get('occurrence') is not None:
        result['occurrence'] = value['occurrence']
    return result


def _overlaps(left, right):
    return left['file'] == right['file'] and max(left['char_start'], right['char_start']) < min(left['char_end'], right['char_end'])


class _Lexer:
    """Balanced literal arguments with original Unicode offsets."""
    def __init__(self, text):
        self.text = text
        self.walker = LatexWalker(text, tolerant_parsing=True)

    def token(self, pos):
        return self.walker.get_token(pos, environments=False)

    def skip(self, pos):
        while pos < len(self.text):
            if self.text[pos].isspace():
                pos += 1
            elif self.text[pos] == '%':
                stop = self.text.find('\n', pos)
                pos = len(self.text) if stop < 0 else stop + 1
            else:
                break
        return pos

    def group(self, pos, optional=False, single=False):
        pos = self.skip(pos)
        opener, closer = ('[', ']') if optional else ('{', '}')
        if pos >= len(self.text) or self.text[pos] != opener:
            if single and not optional and pos < len(self.text):
                token = self.token(pos)
                return {'value': self.text[token.pos:token.pos + token.len].rstrip(),
                        'start': token.pos, 'end': token.pos + token.len, 'balanced': True}
            return None
        start, cursor, depth, braces = pos, pos + 1, 1, 0
        while cursor < len(self.text):
            char = self.text[cursor]
            if char == '%':
                end = self.text.find('\n', cursor)
                cursor = len(self.text) if end < 0 else end + 1
                continue
            if char == '\\':
                try:
                    token = self.token(cursor)
                    cursor = token.pos + token.len
                except LatexWalkerEndOfStream:
                    break
                continue
            if optional:
                if char == '{':
                    braces += 1
                elif char == '}' and braces:
                    braces -= 1
            if not braces:
                if char == opener:
                    depth += 1
                elif char == closer:
                    depth -= 1
                    if not depth:
                        return {'value': self.text[start + 1:cursor], 'start': start,
                                'end': cursor + 1, 'balanced': True}
            cursor += 1
        return {'value': self.text[start + 1:], 'start': start, 'end': len(self.text), 'balanced': False}

    def reference(self, name, pos):
        cursor = self.skip(pos)
        if cursor < len(self.text) and self.text[cursor] == '*':
            cursor += 1
        options = []
        while (argument := self.group(cursor, optional=True)) is not None:
            options.append(argument)
            cursor = argument['end']
        arguments = []
        for _ in range(2 if name in RANGE_COMMANDS else 1):
            argument = self.group(cursor)
            if argument is None:
                break
            arguments.append(argument)
            cursor = argument['end']
        values = [options[0]['value']] if name == 'hyperref' and options else [x['value'] for x in arguments]
        raw_keys = [part.strip() for value in values for part in value.split(',')]
        status = 'supported' if len(arguments) == (2 if name in RANGE_COMMANDS else 1) and all(
            x['balanced'] for x in arguments + options) else 'unknown'
        return cursor, raw_keys, status


class SourceInventory:
    def __init__(self, project):
        self.project = project
        project.assert_unchanged()
        self.events = {r['id']: r for values in project.structure_data.values() if isinstance(values, list)
                       for r in values if isinstance(r, dict) and 'id' in r and 'source' in r}
        self.definitions = {r['id']: r for r in project.structure_data.get('definitions', [])}
        self._lexical_exclusions = {}
        self.reference_aliases = self._reference_aliases()
        # Named/format headings seed the reference scanner's exclusions.
        # The scanner never calls candidates(), so reference-driven display
        # discovery can safely follow it without a SourceProject recursion.
        self.explicit = self._explicit()
        self.references = self._references()
        self.explicit.extend(self._referenced_displays())
        self.explicit.sort(key=lambda r: (r['reading_ranges'][0]['char_start'] if r['reading_ranges'] else float('inf'),
                                         r['span']['file'], r['span']['char_start'], r['item_id']))
        self.prose = self._prose()
        self._add_boundary_advice()
        self.digests = {name: digest({'schema': SCHEMA, 'source': project.fingerprint, 'records': records})
                        for name, records in [('explicit', self.explicit), ('prose', self.prose), ('reference', self.references)]}
        project.assert_unchanged()

    def _id(self, kind, identity):
        return kind + ':' + digest({'schema': SCHEMA, 'source': self.project.fingerprint,
                                    'identity': identity})[:24]

    def _positions(self, span, end=False):
        values = []
        for item in self.project.source_map:
            original = item['source']
            coordinate = span['char_end'] if end else span['char_start']
            valid = original['char_start'] < coordinate <= original['char_end'] if end else original['char_start'] <= coordinate < original['char_end']
            if original['file'] == span['file'] and valid and (
                    span.get('occurrence') is None or item['occurrence'] == span['occurrence']):
                values.append(item['reading_char_start'] + coordinate - original['char_start'])
        return sorted(set(values))

    def _excluded(self, file):
        if file in self._lexical_exclusions:
            return self._lexical_exclusions[file]
        hidden = [e['source'] for e in self.definitions.values() if e['source']['file'] == file]
        hidden += [e['full_source'] for e in self.project.structure_data.get('environments', [])
                   if e.get('name') in VERBATIM and e.get('full_source') and e['full_source']['file'] == file]
        text, cursor = self.project._texts[file], 0
        lexer = _Lexer(text)
        while cursor < len(text):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok == 'comment':
                hidden.append({'file': file, 'char_start': token.pos, 'char_end': cursor})
            elif token.tok == 'macro' and token.arg in ('verb', 'lstinline'):
                p = cursor + int(cursor < len(text) and text[cursor] == '*')
                if p < len(text):
                    stop = text.find(text[p], p + 1)
                    end = len(text) if stop < 0 else stop + 1
                    hidden.append({'file': file, 'char_start': token.pos, 'char_end': end}); cursor = end
        self._lexical_exclusions[file] = hidden
        return hidden

    def _reference_aliases(self):
        aliases = {}
        pattern = re.compile(r'\\let\s*\\([A-Za-z@]+)\s*=?\s*\\([A-Za-z@]+)\b')
        for file, text in self.project._texts.items():
            if file.endswith(('.bib', '.bbl')):
                continue
            for match in pattern.finditer(text):
                span = {'file': file, 'char_start': match.start(), 'char_end': match.end()}
                if not any(_overlaps(span, h) for h in self._excluded(file)):
                    aliases.setdefault(match[1], []).append({'target': match[2], 'span': span})
        return aliases

    def _range(self, event):
        first = _span(event['source'], event.get('file_occurrence'))
        last = _span(event.get('end_source') or event.get('full_source') or event['source'],
                     event.get('file_occurrence'))
        starts, ends = self._positions(first), self._positions(last, end=True)
        return {'char_start': min(starts), 'char_end': min(x for x in ends if x > min(starts))} if starts and any(
            x > min(starts) for x in ends) else None

    def _segments(self, start, end):
        result = []
        for item in self.project.source_map:
            left, right = max(start, item['reading_char_start']), min(end, item['reading_char_end'])
            if left < right:
                original = item['source']
                span = {'file': original['file'], 'char_start': original['char_start'] + left - item['reading_char_start'],
                        'char_end': original['char_start'] + right - item['reading_char_start'], 'occurrence': item['occurrence']}
                result.append(span)
        return result

    def _record(self, kind, event, **extra):
        occurrence = event.get('file_occurrence')
        canonical = _span(event.get('full_source') or event['source'], occurrence)
        reading = self._range(event)
        segments = self._segments(**{'start': reading['char_start'], 'end': reading['char_end']}) if reading else [canonical]
        return {'item_id': self._id(kind, {'span': canonical, 'basis': extra.get('basis')}),
                'span': canonical, 'source_spans': segments, 'original_text': [self.project.resolve_span(s) for s in segments],
                'reading_ranges': [reading] if reading else [], 'event_id': event.get('id'),
                'activity': event.get('activity', 'unknown'), 'section_ids': event.get('section_ids', []), **extra}

    def _environment_title(self, event, seen=None):
        seen = set() if seen is None else seen
        definition = self.definitions.get(event.get('definition_id'))
        title = str((definition or {}).get('body') or event.get('name', '')).strip()
        if (definition or {}).get('theorem_like'):
            return title
        alias = re.fullmatch(r'\s*\\begin\{([A-Za-z*]+)\}(?:\[[^\]]*\])?\s*', title)
        if alias and alias[1] not in seen:
            seen.add(alias[1])
            target = next((d for d in self.definitions.values() if d.get('definition_kind') == 'environment'
                           and d.get('name') == alias[1]), None)
            return self._environment_title({'name': alias[1], 'definition_id': (target or {}).get('id')}, seen)
        return event.get('name', '').removesuffix('*')

    @staticmethod
    def _category(title):
        if re.search(r'\b(proof|demonstration)\b', title, re.I):
            return 'proof'
        if re.search(r'\b(theorem|lemma|proposition|corollary)\b', title, re.I):
            return 'statement'
        if re.search(r'\bdefinition\b', title, re.I):
            return 'definition'
        return 'statement'

    def _explicit(self):
        records = []
        for event in self.project.structure_data.get('environments', []):
            if event.get('source_role') != 'document_or_bibliography':
                continue
            title = self._environment_title(event)
            if event.get('environment_kind') != 'theorem' and not RESULT_WORDS.search(title):
                continue
            records.append(self._record('explicit', event, basis='named_environment', name=event['name'],
                title=title, category_hint=self._category(title), eligible_hint='core' if CORE_WORDS.search(title) else 'requires_semantic_review',
                boundary_status='complete_static_span' if event.get('closed') else 'unresolved',
                parent_environment=event.get('parent_environment')))
        headings = list(self.project.structure_data.get('format_cues', []))
        known = {(e['source']['file'], e['source']['char_start']) for e in headings}
        for file, text in self.project._texts.items():
            if file.endswith(('.sty', '.cls', '.bib', '.bbl')):
                continue
            for match in PLAIN_HEADING.finditer(text):
                span = {'file': file, 'char_start': match.start(1), 'char_end': match.end(1)}
                if (file, match.start(1)) not in known and not any(_overlaps(span, h) for h in self._excluded(file)):
                    for occurrence in self._occurrences(span):
                        headings.append({'source': span, 'file_occurrence': occurrence,
                                         'raw_content': match[1], 'activity': 'requires_activity_review',
                                         'heading_basis': 'plain_mathematical_heading'})
            # Formatting arguments are not theorem environments.  Only the
            # heading is certain; an Agent must delimit its actual body.
            lexer = _Lexer(text)
            cursor = 0
            while cursor < len(text):
                try:
                    token = lexer.token(cursor)
                except LatexWalkerEndOfStream:
                    break
                cursor = token.pos + token.len
                if token.tok != 'macro' or token.arg not in {'textbf', 'textsc', 'emph'}:
                    continue
                argument = lexer.group(cursor)
                if argument:
                    cursor = argument['end']
                    clean = re.sub(r'\\(?:bf|it|rm|sc)\b|\s+', ' ', argument['value']).strip()
                    span = {'file': file, 'char_start': token.pos, 'char_end': cursor}
                    if RESULT_WORDS.match(clean) and len(clean) <= 300 and (file, token.pos) not in known and not any(
                            _overlaps(span, h) for h in self._excluded(file)):
                        for occurrence in self._occurrences(span):
                            headings.append({'source': span, 'file_occurrence': occurrence,
                                             'raw_content': argument['value'], 'activity': 'unknown'})
        for event in headings:
            title = event.get('raw_content', '')
            clean = re.sub(r'\\(?:bf|it|rm|sc)\b|\s+', ' ', title).strip()
            if not RESULT_WORDS.match(clean):
                continue
            # A heading that merely names a theorem in ordinary emphasis is
            # retained for review, never automatically counted as a node.
            records.append(self._record('explicit', event, basis=event.get('heading_basis', 'formatted_mathematical_heading'), title=title,
                category_hint=self._category(title), eligible_hint='requires_semantic_review',
                boundary_status='heading_only_agent_must_determine_body'))
        records.sort(key=lambda r: (r['reading_ranges'][0]['char_start'] if r['reading_ranges'] else float('inf'),
                                    r['span']['file'], r['span']['char_start'], r['item_id']))
        return records

    def _referenced_displays(self):
        """Offer full-ledger display hints, without resolving manual counters.

        Exact syntactic owners and an ambiguous manual-locator reading pool
        remain distinct. Supplemental hints do not cut surrounding prose or
        certify independently reusable mathematical objects.
        """
        references = [r for r in self.references if r['activity'] != 'inactive'
                      and r['reading_ranges'] and r['resolution'] != 'external']
        owners = {}
        for reference in references:
            for target in reference['targets']:
                for owner in target['math_owners']:
                    if owner['category_hint'] == 'expression':
                        owners.setdefault(owner['event_id'], []).append(reference)
        manual = [r for r in references if r['basis'] in (
            'handwritten_numbered_reference', 'handwritten_parenthesized_reference')
            and r.get('locator', {}).get('kind') == 'equation']
        legacy_references = {(r['source']['file'], r['source']['char_start'], r.get('file_occurrence'), key)
            for r in self.project.structure_data.get('references', [])
            if r.get('activity') == 'active' and r.get('source_role') == 'document_or_bibliography'
            for key in r.get('keys', [])}
        numbered = {'equation', 'align', 'alignat', 'gather', 'multline', 'eqnarray'}
        records = []
        for event in self.project.structure_data.get('environments', []):
            span = event.get('full_source')
            if (event.get('environment_kind') != 'math' or not span
                    or event.get('activity') == 'inactive'
                    or event.get('source_role') != 'document_or_bibliography'
                    or span['file'].endswith(('.sty', '.cls', '.bib', '.bbl'))
                    or self._range(event) is None):
                continue
            if any(parent.get('name') == 'thebibliography' and parent.get('full_source')
                   and parent.get('file_occurrence') == event.get('file_occurrence')
                   and parent['full_source']['file'] == span['file']
                   and parent['full_source']['char_start'] <= span['char_start']
                   and span['char_end'] <= parent['full_source']['char_end']
                   for parent in self.project.structure_data.get('environments', [])):
                continue
            labels = [key for key, rows in self.project._labels.items() for label in rows
                      if _overlaps(label['source'], span)
                      and label.get('file_occurrence') == event.get('file_occurrence')]
            direct = owners.get(event['id'], [])
            # Do not guess which source counter a dotted manual number means.
            # Only numberable/tagged display environments form its reading
            # pool. The ledger's original UNKNOWN target remains unchanged.
            tagged = bool(re.search(r'\\tag\*?\s*\{', self.project.resolve_span(
                _span(span, event.get('file_occurrence')))))
            possible_manual = manual if event.get('name') in numbered or tagged else []
            if not direct and not possible_manual:
                continue
            # A pure manual pool is never a legacy cited display. Even an
            # unrelated old key must not turn its reading hint into a prose cut.
            legacy = any((r['span']['file'], r['span']['char_start'], r['span'].get('occurrence'), key)
                         in legacy_references for r in direct for key in r['keys'] if key in labels)
            extra = {}
            if not legacy:
                extra = {'candidate_only': True, 'preserve_surrounding_prose': True,
                         'reference_item_ids': sorted({r['item_id'] for r in direct or possible_manual}),
                         'reference_endpoint_status': 'full_ledger_syntactic_owner_hint' if direct
                             else 'unresolved_manual_locator_reading_pool'}
            records.append(self._record('explicit', event,
                basis='referenced_display_environment' if direct else 'handwritten_locator_display_candidate',
                name=event['name'], title=event['name'], category_hint='expression', labels=labels,
                eligible_hint='requires_semantic_review', parent_environment=event.get('parent_environment'),
                boundary_status='complete_static_span' if event.get('closed') else 'unresolved', **extra))
        return records

    def _prose(self):
        mapping = self.project.source_map
        if not mapping:
            return []
        document = next((e for e in self.project.structure_data.get('environments', []) if e.get('name') == 'document'), None)
        doc_range = self._range(document) if document else None
        first = doc_range['char_start'] if doc_range else 0
        last = doc_range['char_end'] if doc_range else len(self.project.reading)
        if document:
            begins = self._positions(_span(document['source'], document.get('file_occurrence')), end=True)
            closes = self._positions(_span(document['end_source'], document.get('file_occurrence'))) if document.get('end_source') else []
            first = min(begins) if begins else first
            last = min(closes) if closes else last
        mask, cuts = [], {first, last}
        for record in self.explicit:
            if record.get('preserve_surrounding_prose'):
                continue
            for interval in record['reading_ranges']:
                left, right = max(first, interval['char_start']), min(last, interval['char_end'])
                if left < right:
                    cuts.update((left, right))
                    # A notation/remark/example region may combine ambient
                    # setup with a genuine unheaded construction. Keep that
                    # interior available to round 2 after the whole environment
                    # has been adjudicated; syntax alone does not make it one
                    # mathematical node. Core blocks and independent displays
                    # remain protected against automatic duplicate splitting.
                    if (record['boundary_status'] == 'complete_static_span'
                            and (record.get('eligible_hint') == 'core'
                                 or record.get('category_hint') == 'expression')):
                        mask.append((left, right))
        sections = []
        for event in self.project.structure_data.get('sections', []):
            interval = self._range(event)
            if interval and first <= interval['char_start'] < last:
                cuts.update((interval['char_start'], min(last, interval['char_end'])))
                sections.append((interval['char_start'], event))
        # Support code and bibliography are not second-round body paragraphs.
        for item in mapping:
            if item['source']['file'].endswith(('.sty', '.cls', '.bib', '.bbl')):
                left, right = max(first, item['reading_char_start']), min(last, item['reading_char_end'])
                if left < right:
                    cuts.update((left, right)); mask.append((left, right))
        records = []
        boundaries = sorted(cuts)
        for left, right in zip(boundaries, boundaries[1:]):
            if any(a <= left and right <= b for a, b in mask):
                continue
            text = self.project.reading[left:right]
            paragraphs = [left] + [left + m.end() for m in re.finditer(r'\r?\n\s*\r?\n', text)] + [right]
            for begin, finish in zip(paragraphs, paragraphs[1:]):
                for start in range(begin, finish, MAX_PARAGRAPH):
                    end = min(finish, start + MAX_PARAGRAPH)
                    if not self.project.reading[start:end].strip():
                        continue
                    spans = self._segments(start, end)
                    if not spans:
                        continue
                    preceding = [r for r in self.explicit if r['reading_ranges'] and r['reading_ranges'][0]['char_end'] <= start]
                    following = [r for r in self.explicit if r['reading_ranges'] and r['reading_ranges'][0]['char_start'] >= end]
                    section = next((e for p, e in reversed(sections) if p <= start), None)
                    records.append({'item_id': self._id('prose', {'source_spans': spans}), 'span': spans[0],
                        'source_spans': spans, 'original_text': [self.project.resolve_span(s) for s in spans],
                        'reading_ranges': [{'char_start': start, 'char_end': end}],
                        'anchor_before': preceding[-1]['item_id'] if preceding else None,
                        'anchor_after': following[0]['item_id'] if following else None,
                        'section_ids': [section['id']] if section else [], 'activity': 'requires_source_review',
                        'boundary_status': 'complete_reading_fragment',
                        'boundary_advice': {'candidate_only': True,
                            'complete_paragraph_spans': self._segments(begin, finish),
                            'complete_paragraph_range': {'char_start': begin, 'char_end': finish}},
                        'notice': 'Segmentation is a reading aid; search the whole document for actual use and context.'})
        return records

    def _sentence_ranges(self, start, end):
        """Conservative original-reading sentence anchors, not semantic units.

        Keep punctuation inside math, braces and parentheses intact, including
        decimal numbers and '(resp. ...)'. Unfinished fragments remain data.
        """
        text = self.project.reading[start:end]
        lexer, cursor, first = _Lexer(text), 0, 0
        braces, parentheses, math = 0, 0, None
        result = []
        while cursor < len(text):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok in ('mathmode_inline', 'mathmode_display', 'mathmode_delimiter'):
                if math is None:
                    math = token.arg
                elif token.arg == math or (math, token.arg) in (('\\(', '\\)'), ('\\[', '\\]')):
                    math = None
            elif token.tok == 'brace_open':
                braces += 1
            elif token.tok == 'brace_close':
                braces = max(0, braces - 1)
            elif token.tok == 'char' and math is None:
                if token.arg in '([':
                    parentheses += 1
                elif token.arg in ')]':
                    parentheses = max(0, parentheses - 1)
                elif (token.arg in '.!?' and not braces and not parentheses
                        and (cursor == len(text) or text[cursor].isspace())
                        and not (token.arg == '.' and token.pos and text[token.pos - 1].isdigit()
                                 and cursor < len(text) and text[cursor].isdigit())):
                    result.append((start + first, start + cursor)); first = cursor
        if first < len(text) and text[first:].strip():
            result.append((start + first, end))
        return [(a + len(self.project.reading[a:b]) - len(self.project.reading[a:b].lstrip()),
                 b - len(self.project.reading[a:b]) + len(self.project.reading[a:b].rstrip()))
                for a, b in result if self.project.reading[a:b].strip()]

    def _content_advice(self, start, end, lists):
        """Suggest useful complete source sentences, never accept or classify facts."""
        candidates = []
        for left, right in self._sentence_ranges(start, end):
            if any(a < right and left < b for _, a, b in lists):
                continue  # Complete lists stay available as compound regions.
            text = self.project.reading[left:right]
            statement = re.search(r'\b(?:if\b[\s\S]*?\bthen|conversely|if and only if|'
                r'one[- ]to[- ]one correspondence|it follows that|there (?:is|exists|are))\b', text, re.I)
            declaration = re.search(r'\b(?:we (?:define|call|denote|write)|is called|are called|'
                                    r'is defined|are defined)\b', text, re.I)
            construction = re.search(r'\b(?:consist\w*|closure|assigns?|sends?|produces?|'
                                     r'set of|subcategory|subcategories)\b', text, re.I)
            category = 'statement' if statement else 'definition' if declaration and construction else None
            if category is None:
                continue
            if candidates and candidates[-1]['category_hint'] == category:
                previous = candidates[-1]['reading_range']
                if not self.project.reading[previous['char_end']:left].strip():
                    previous['char_end'] = right
                    candidates[-1]['source_spans'] = self._segments(previous['char_start'], right)
                    continue
            candidates.append({'basis': 'possible_independent_assertion' if category == 'statement'
                else 'possible_reusable_construction', 'category_hint': category,
                'boundary_status': 'source_sentence_group_requires_semantic_review',
                'reading_range': {'char_start': left, 'char_end': right},
                'source_spans': self._segments(left, right), 'context_required': True})
        return candidates

    def _add_boundary_advice(self):
        """Read-only anchors prevent a syntax heading from implying body coverage.

        Advice preserves all existing source ranges. Closed list/environment
        endpoints are syntax facts; usefulness, independent identity and the
        correct node boundary remain explicit Agent decisions.
        """
        events = self.project.structure_data.get('environments', [])
        lists = []
        for event in events:
            interval = self._range(event)
            if event.get('name') in ('itemize', 'enumerate', 'description') and event.get('closed') and interval:
                lists.append((event, interval['char_start'], interval['char_end']))
        for record in self.prose:
            advice = record['boundary_advice']; interval = advice['complete_paragraph_range']
            advice['content_candidates'] = self._content_advice(interval['char_start'], interval['char_end'], lists)
            advice['closed_list_regions'] = [{'event_id': event['id'], 'name': event['name'],
                'source_spans': self._segments(left, right)} for event, left, right in lists
                if left < interval['char_end'] and interval['char_start'] < right]
        sections = [interval['char_start'] for event in self.project.structure_data.get('sections', [])
                    if (interval := self._range(event)) is not None]
        for record in self.explicit:
            advice = {'candidate_only': True, 'complete_regions': [], 'following_prose_anchors': [],
                      'content_candidates': []}
            record['boundary_advice'] = advice
            if not record['reading_ranges']:
                continue
            interval = record['reading_ranges'][0]; first, end = interval['char_start'], interval['char_end']
            boundary = min([len(self.project.reading)] + [r['reading_ranges'][0]['char_start'] for r in self.explicit
                if r['reading_ranges'] and not r.get('preserve_surrounding_prose')
                and r['reading_ranges'][0]['char_start'] >= end
                and r['item_id'] != record['item_id']] + [p for p in sections if p >= end])
            if record['boundary_status'] == 'complete_static_span':
                advice['complete_regions'].append({'basis': 'closed_environment',
                    'boundary_status': 'complete_static_span', 'category_hint': record['category_hint'],
                    'source_spans': record['source_spans']})
            elif record['boundary_status'] == 'heading_only_agent_must_determine_body':
                adjacent = [(event, left, right) for event, left, right in lists if end <= left < right <= boundary
                    and not re.search(r'\r?\n\s*\r?\n', self.project.reading[end:left].strip())]
                if adjacent:
                    event, _, end = min(adjacent, key=lambda item: item[1])
                    advice['complete_regions'].append({'basis': 'heading_to_closed_list', 'event_id': event['id'],
                        'boundary_status': 'closed_list_boundary_requires_semantic_review',
                        'category_hint': record['category_hint'], 'source_spans': self._segments(first, end)})
            event = self.events.get(record.get('event_id'), {})
            body_first, body_end = first, interval['char_end']
            if record['boundary_status'] == 'complete_static_span' and event.get('end_source'):
                openings = self._positions(_span(event['source'], event.get('file_occurrence')), end=True)
                closings = self._positions(_span(event['end_source'], event.get('file_occurrence')))
                body_first = next((p for p in openings if first <= p <= body_end), body_first)
                body_end = next((p for p in closings if body_first <= p <= body_end), body_end)
            if record.get('eligible_hint') != 'core':
                advice['content_candidates'] = self._content_advice(body_first, body_end, lists)
            for prose in self.prose:
                bounds = prose['boundary_advice']['complete_paragraph_range']
                if not end <= bounds['char_start'] < bounds['char_end'] <= boundary:
                    continue
                if prose['boundary_advice']['content_candidates']:
                    advice['following_prose_anchors'].append({'item_id': prose['item_id'],
                        'source_spans': prose['boundary_advice']['complete_paragraph_spans'],
                        'content_candidates': prose['boundary_advice']['content_candidates']})
                    break

    def _occurrences(self, span):
        return sorted({item['occurrence'] for item in self.project.source_map if _overlaps(item['source'], span)}) or [None]

    def _targets(self, keys, dynamic=False):
        records = []
        for key in keys:
            key_dynamic = bool(re.search(r'[\\#]', key))
            labels = self.project._labels.get(key, []) if not key_dynamic else []
            candidates, spans, event_ids = [], [], []
            for label in labels:
                label_span = _span(label['source'], label.get('file_occurrence'))
                containers = [e for e in self.project.structure_data.get('environments', [])
                              if e.get('full_source') and _overlaps(e['full_source'], label_span)
                              and e.get('file_occurrence') == label.get('file_occurrence')]
                # Explicit label owner is first, then nested and outer owners.
                containers.sort(key=lambda e: (e['id'] != label.get('environment_id'),
                                               e['full_source']['char_end'] - e['full_source']['char_start']))
                mathematical = [e for e in containers if e.get('environment_kind') in ('theorem', 'math') or RESULT_WORDS.search(self._environment_title(e))]
                ownership_basis = {e['id']: 'lexical_environment_containment' for e in mathematical}
                if not mathematical:
                    # Some authors place a label just after the closing marker.
                    # Preserve the adjacent mathematical owner as a candidate;
                    # LaTeX counter semantics and actual ownership still need review.
                    preceding = [e for e in self.project.structure_data.get('environments', [])
                        if e.get('full_source') and e['full_source']['file'] == label_span['file']
                        and e.get('file_occurrence') == label.get('file_occurrence')
                        and e['full_source']['char_end'] <= label_span['char_start']
                        and (e.get('environment_kind') in ('theorem', 'math') or RESULT_WORDS.search(self._environment_title(e)))]
                    if preceding:
                        nearest = max(preceding, key=lambda e: e['full_source']['char_end'])
                        between = self.project._texts[label_span['file']][nearest['full_source']['char_end']:label_span['char_start']]
                        if not re.sub(r'\\label\s*\{[^{}]*\}|\s+', '', between):
                            mathematical = [nearest]; ownership_basis[nearest['id']] = 'adjacent_post_environment_label_requires_review'
                selected = mathematical or ([self.events[label['environment_id']]] if label.get('environment_id') in self.events else [])
                if selected:
                    for event in selected:
                        candidate_span = _span(event.get('full_source') or event['source'], event.get('file_occurrence'))
                        spans.append(candidate_span); event_ids.append(event['id'])
                candidates.append({'label_span': label_span, 'label_event_id': label['id'],
                    'owner_event_id': label.get('environment_id'), 'mathematical_event_ids': [e['id'] for e in mathematical],
                    'math_owners': [{'event_id': e['id'], 'category_hint': 'expression' if e.get('environment_kind') == 'math' else self._category(self._environment_title(e)),
                        'source_spans': self._record('owner', e)['source_spans'],
                        'boundary_status': 'complete_static_span' if e.get('closed') else 'unresolved',
                        'ownership_basis': ownership_basis[e['id']]} for e in mathematical]})
            resolution = 'dynamic' if key_dynamic else ('unresolved' if not labels else 'ambiguous' if len(labels) > 1 else
                         'resolved' if any(c['mathematical_event_ids'] for c in candidates) else 'non_mathematical_or_unclassified')
            records.append({'key': key, 'resolution': resolution, 'label_spans': [c['label_span'] for c in candidates],
                            'target_spans': spans, 'target_event_ids': event_ids, 'label_candidates': candidates,
                            'math_owners': [owner for c in candidates for owner in c['math_owners']],
                            'owner_source_spans': [s for c in candidates for owner in c['math_owners'] for s in owner['source_spans']]})
        return records

    def _reference_record(self, event, keys, syntax_status, **extra):
        record = self._record('reference', event, basis=extra.pop('basis', 'literal_reference'),
                              command=event.get('command', event.get('name')), keys=keys,
                              syntax_status=syntax_status, **extra)
        dynamic = any(re.search(r'[\\#]', key) for key in keys)
        record['targets'] = self._targets(keys, dynamic)
        record['resolution'] = 'dynamic' if dynamic else ('unknown' if not keys or syntax_status != 'supported' else
            'ambiguous' if any(t['resolution'] == 'ambiguous' for t in record['targets']) else
            'unresolved' if any(t['resolution'] == 'unresolved' for t in record['targets']) else 'resolved')
        record['item_id'] = self._id('reference', {'span': record['span'], 'basis': record['basis'],
            'keys': keys, 'projection_index': record.get('projection_index'), 'macro_chain': record.get('macro_chain')})
        record['relation_semantics'] = 'syntactic_reference_not_mathematical_dependency'
        return record

    def _numbered_heading_targets(self, key, kind, number):
        """Literal printed headings are anchors, never inferred TeX counters.

        Preserve all occurrences and activity hints. The Agent must still read
        the complete mathematical body, accept its node and decide actual use.
        No heading-to-next-heading range is manufactured as an owner body.
        """
        anchors = []
        for record in self.explicit:
            if record.get('boundary_status') != 'heading_only_agent_must_determine_body':
                continue
            title = re.sub(r'\\(?:bf|it|rm|sc)\b|\s+', ' ', record.get('title', '')).strip()
            match = PRINTED_NUMBERED_HEADING.match(title)
            if match and match[1].lower() == kind and match[2] == number:
                anchors.append({'item_id': record['item_id'], 'event_id': record.get('event_id'),
                    'kind': kind, 'number': number, 'category_hint': record.get('category_hint'),
                    'source_spans': record['source_spans'], 'activity': record['activity'],
                    'boundary_status': record['boundary_status']})
        return [{'key': key, 'resolution': 'heading_candidates_require_review' if anchors else 'unresolved',
                 'label_spans': [], 'target_spans': [], 'target_event_ids': [],
                 'label_candidates': [], 'math_owners': [], 'owner_source_spans': [],
                 'heading_anchors': anchors}]

    def _external_record(self, event, keys, **extra):
        syntax_status = extra.pop('syntax_status', 'supported')
        record = self._reference_record(event, keys, syntax_status, **extra)
        record['resolution'] = 'external' if keys and syntax_status == 'supported' else 'unknown'
        record['targets'] = [{'key': key, 'resolution': 'external', 'label_spans': [], 'target_spans': [],
                             'target_event_ids': [], 'label_candidates': [], 'math_owners': [], 'owner_source_spans': []} for key in keys]
        return record

    @staticmethod
    def _literal_math_ranges(text, excluded=()):
        lexer, cursor, math, ranges = _Lexer(text), 0, None, []
        excluded, index = sorted(excluded), 0
        while cursor < len(text):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            while index < len(excluded) and excluded[index][1] <= token.pos:
                index += 1
            if index < len(excluded) and excluded[index][0] <= token.pos < excluded[index][1]:
                cursor = max(cursor, excluded[index][1]); continue
            if token.tok == 'macro' and token.arg in ('verb', 'lstinline'):
                first = cursor + int(cursor < len(text) and text[cursor] == '*')
                if first < len(text):
                    last = text.find(text[first], first + 1)
                    cursor = len(text) if last < 0 else last + 1
                continue
            if token.tok in ('mathmode_inline', 'mathmode_display', 'mathmode_delimiter'):
                if math is None:
                    math = (token.arg, token.pos)
                elif token.arg == math[0] or (math[0], token.arg) in (('\\(', '\\)'), ('\\[', '\\]')):
                    ranges.append((math[1], cursor)); math = None
        if math is not None:
            ranges.append((math[1], len(text)))
        return ranges

    def _bibliographic_brackets(self, existing):
        """Bounded literal printed citations tied to declared bibliography items.

        Printed labels take precedence over repeated internal bibitem keys.
        This resolves a bibliographic source declaration, never a mathematical
        document/unit, and retains all ambiguous declaration candidates.
        """
        displayed, keyed = {}, {}
        for item in self.project.structure_data.get('bibliography_items', []):
            label, key = item.get('display_label'), item.get('key')
            if isinstance(label, str) and label.strip():
                displayed.setdefault(label.strip(), []).append(item)
            if isinstance(key, str) and key.strip():
                keyed.setdefault(key.strip(), []).append(item)
        if not displayed and not keyed:
            return []
        document = next((e for e in self.project.structure_data.get('environments', []) if e.get('name') == 'document'), None)
        interval = self._range(document) if document else None
        lower, upper = (interval['char_start'], interval['char_end']) if interval else (0, len(self.project.reading))
        if document:
            lower = min(self._positions(_span(document['source'], document.get('file_occurrence')), end=True) or [lower])
            upper = min(self._positions(_span(document['end_source'], document.get('file_occurrence'))) or [upper]) if document.get('end_source') else upper
        reading_definitions = [(bounds['char_start'], bounds['char_end']) for event in self.definitions.values()
                               if (bounds := self._range(event)) is not None]
        reading_hidden = self._literal_math_ranges(self.project.reading, reading_definitions)
        reading_hidden += [(bounds['char_start'], bounds['char_end'])
            for event in self.project.structure_data.get('environments', [])
            if (event.get('name') == 'thebibliography' or event.get('environment_kind') == 'math')
            and (bounds := self._range(event)) is not None]
        records = []
        for file, text in self.project._texts.items():
            if file.endswith(('.sty', '.cls', '.bib', '.bbl')):
                continue
            hidden = list(self._excluded(file)) + [r['span'] for r in existing if r['span']['file'] == file]
            hidden += [e['full_source'] for e in self.project.structure_data.get('environments', [])
                if e.get('full_source') and e['full_source']['file'] == file
                and (e.get('name') == 'thebibliography' or e.get('environment_kind') == 'math')]
            hidden += [{'file': file, 'char_start': left, 'char_end': right}
                       for left, right in self._literal_math_ranges(text,
                           [(row['char_start'], row['char_end']) for row in hidden])]
            for match in re.finditer(r'\[([^\[\]\r\n]{1,240})\]', text):
                span = {'file': file, 'char_start': match.start(), 'char_end': match.end()}
                if any(_overlaps(span, h) for h in hidden):
                    continue
                parts = [part.strip() for part in match[1].split(',')]
                keys, options, candidates = [], [], {}
                for part in parts:
                    matches = displayed.get(part) or keyed.get(part) or []
                    if matches or re.fullmatch(r'[A-Za-z0-9_.:-]+', part):
                        keys.append(part)
                        candidates[part] = [{'item_id': item['id'], 'key': item.get('key'),
                            'display_label': item.get('display_label'), 'activity': item.get('activity', 'unknown'),
                            'matched_as': 'display_label' if part in displayed else 'bibliographic_key',
                            'source_span': _span(item['source'], item.get('file_occurrence'))} for item in matches]
                    elif re.match(r'(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|Section|Equation|pp?\.|§)\b', part, re.I):
                        options.append(part)
                    else:
                        break
                else:
                    if not any(candidates.values()):
                        continue
                    for occurrence in self._occurrences(span):
                        anchored = _span(span, occurrence)
                        if not any(lower <= pos < upper and not any(a <= pos < b for a, b in reading_hidden)
                                   for pos in self._positions(anchored)):
                            continue
                        record = self._external_record({'source': span, 'file_occurrence': occurrence,
                            'command': 'literal_bibliographic_bracket', 'activity': 'requires_activity_review'}, keys,
                            syntax_status='supported' if all(candidates.values()) else 'unknown',
                            basis='handwritten_bibliographic_reference', raw_bracket=match[0], locator_options=options,
                            identified_external_mathematical_document=False)
                        for target in record['targets']:
                            target['bibliography_candidates'] = candidates[target['key']]
                            target['bibliographic_identity_ambiguous'] = len(candidates[target['key']]) > 1
                            if not candidates[target['key']]:
                                target['resolution'] = 'unknown'
                        records.append(record)
        return records

    def _project_macro(self, definition, values, seen=None):
        seen = [] if seen is None else seen
        if definition['id'] in seen or len(seen) >= 8:
            return None
        body = definition.get('body') or ''
        if definition.get('activity') != 'active' or definition.get('delimited_parameters') or definition.get('xparse_signature') or UNSAFE_MACRO.search(body):
            return None
        if re.search(r'##|#[^1-9]', body):
            return None
        if any(int(n) > len(values) for n in re.findall(r'#([1-9])', body)):
            return None
        expanded = re.sub(r'#([1-9])', lambda m: values[int(m[1]) - 1], body)
        lexer, cursor, found = _Lexer(expanded), 0, []
        while cursor < len(expanded):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok != 'macro':
                continue
            if token.arg in REFERENCE_COMMANDS:
                cursor, keys, status = lexer.reference(token.arg, cursor)
                found.append({'command': token.arg, 'keys': [normalize_key(k) for k in keys],
                              'syntax_status': status, 'macro_chain': seen + [definition['id']]})
            else:
                definitions = [d for d in self.definitions.values() if d.get('name') == token.arg and
                               d.get('definition_kind') == 'macro']
                if len(definitions) > 1:
                    return None
                nested = definitions[0] if definitions else None
                if nested:
                    args, cursor, valid = self._macro_arguments(lexer, cursor, nested)
                    projected = self._project_macro(nested, args, seen + [definition['id']]) if valid else None
                    if projected:
                        found.extend(projected)
                    elif nested.get('contains_structure') or 'ref' in str(nested.get('body', '')).lower():
                        return None
                elif token.arg not in RENDER_ARGUMENT_COMMANDS | {'protect', 'relax'}:
                    argument = lexer.group(cursor)
                    if argument and re.search(r'\\(?:' + '|'.join(sorted(REFERENCE_COMMANDS)) + r')\b', argument['value']):
                        return None
        return found

    @staticmethod
    def _macro_arguments(lexer, cursor, definition):
        values, valid = [], True
        count = definition.get('argument_count', 0)
        if definition.get('optional_default') is not None:
            argument = lexer.group(cursor, optional=True)
            values.append(argument['value'] if argument else definition['optional_default'])
            if argument:
                cursor = argument['end']; valid &= argument['balanced']
            count -= 1
        for _ in range(max(0, count)):
            argument = lexer.group(cursor, single=True)
            if argument is None:
                valid = False; break
            values.append(argument['value']); cursor = argument['end']; valid &= argument['balanced']
        return values, cursor, valid

    def _references(self):
        records, seen = [], set()
        existing = {(e['source']['file'], e['source']['char_start'], e.get('file_occurrence')): e
                    for e in self.project.structure_data.get('references', [])}
        for file, text in self.project._texts.items():
            if file.endswith(('.sty', '.cls', '.bib', '.bbl')):
                continue
            lexer, cursor, conditions = _Lexer(text), 0, []
            excluded = [e['source'] for e in self.definitions.values() if e['source']['file'] == file]
            excluded += [e['full_source'] for e in self.project.structure_data.get('environments', [])
                         if e.get('name') in VERBATIM and e.get('full_source') and e['full_source']['file'] == file]
            calls = {}
            for event in self.project.structure_data.get('macro_calls', []):
                if event['source']['file'] == file:
                    calls.setdefault(event['source']['char_start'], []).append(event)
            while cursor < len(text):
                exclusion = next((s for s in excluded if s['char_start'] <= cursor < s['char_end']), None)
                if exclusion:
                    cursor = exclusion['char_end']; continue
                try:
                    token = lexer.token(cursor)
                except LatexWalkerEndOfStream:
                    break
                cursor = token.pos + token.len
                if any(s['char_start'] <= token.pos < s['char_end'] for s in excluded) or token.tok != 'macro':
                    continue
                if token.arg == 'let':
                    first = lexer.group(cursor, single=True)
                    if first:
                        cursor = lexer.skip(first['end'])
                        if cursor < len(text) and text[cursor] == '=':
                            cursor += 1
                        second = lexer.group(cursor, single=True)
                        if second:
                            cursor = second['end']
                    continue
                if token.arg in {'iftrue', 'iffalse'} or token.arg.startswith('if'):
                    conditions.append(True if token.arg == 'iftrue' else False if token.arg == 'iffalse' else None)
                    continue
                if token.arg == 'else' and conditions:
                    conditions[-1] = not conditions[-1] if conditions[-1] is not None else None
                    continue
                if token.arg == 'fi' and conditions:
                    conditions.pop(); continue
                activity = 'inactive' if False in conditions else 'conditional' if None in conditions else 'active'
                if token.arg in {'verb', 'lstinline'}:
                    p = cursor + int(cursor < len(text) and text[cursor] == '*')
                    if p < len(text):
                        stop = text.find(text[p], p + 1)
                        cursor = len(text) if stop < 0 else stop + 1
                    continue
                if token.arg in REFERENCE_COMMANDS:
                    end, raw_keys, status = lexer.reference(token.arg, cursor)
                    cursor = end
                    for occurrence in self._occurrences({'file': file, 'char_start': token.pos, 'char_end': end}):
                        event = existing.get((file, token.pos, occurrence), {'source': {'file': file, 'char_start': token.pos,
                            'char_end': end}, 'file_occurrence': occurrence, 'command': token.arg,
                            'activity': 'unmapped' if occurrence is None else activity})
                        event = {**event, 'source': {'file': file, 'char_start': token.pos, 'char_end': end}}
                        records.append(self._reference_record(event, [normalize_key(k) for k in raw_keys], status))
                        seen.add((file, token.pos, occurrence))
                    continue
                aliases = self.reference_aliases.get(token.arg, [])
                if aliases:
                    alias = aliases[0]
                    target = alias['target']
                    # Only a unique literal alias to a supported reference is
                    # adapted.  Group/conditional scope is not reconstructed.
                    # If its scope cannot be established, retain unknown.
                    alias_event = next((e for e in self.project.structure_data.get('unresolved', [])
                        if e.get('command') == 'let' and _overlaps(e['source'], alias['span'])), {})
                    declared_positions = self._positions(alias['span'])
                    called_positions = self._positions({'file': file, 'char_start': token.pos, 'char_end': cursor})
                    safe = len(aliases) == 1 and target in REFERENCE_COMMANDS and alias_event.get('activity') == 'active' and (
                        alias_event.get('scope') == ['document'] and not alias_event.get('condition_ids') and
                        declared_positions and called_positions and max(declared_positions) < min(called_positions) and
                        not any(d.get('name') == token.arg for d in self.definitions.values()))
                    end, keys, status = lexer.reference(target if target in REFERENCE_COMMANDS else token.arg, cursor)
                    cursor = end
                    for occurrence in self._occurrences({'file': file, 'char_start': token.pos, 'char_end': end}):
                        event = {'source': {'file': file, 'char_start': token.pos, 'char_end': end},
                                 'file_occurrence': occurrence, 'command': token.arg,
                                 'activity': 'unmapped' if occurrence is None else activity}
                        records.append(self._reference_record(event, [normalize_key(k) for k in keys],
                            status if safe else 'unknown', basis='literal_reference_alias', projected_command=target,
                            definition_spans=[alias['span']],
                            unresolved_reason=None if safe else 'Alias scope/redefinition or target semantics require review.'))
                    continue
                if token.arg == 'href':
                    url = lexer.group(cursor)
                    display = lexer.group(url['end']) if url else None
                    end = display['end'] if display else url['end'] if url else cursor
                    event = {'source': {'file': file, 'char_start': token.pos, 'char_end': end},
                             'command': token.arg, 'activity': activity}
                    records.append(self._external_record(event, [url['value']] if url else [], basis='external_hyperlink',
                        syntax_status='supported' if url and display and url['balanced'] and display['balanced'] else 'unknown'))
                    cursor = end
                    continue
                found_calls = calls.get(token.pos)
                if found_calls:
                    end = cursor
                    for call in found_calls:
                        definition = self.definitions.get(call.get('definition_id'))
                        if definition is None:
                            continue
                        values, end, valid = self._macro_arguments(lexer, cursor, definition)
                        projected = self._project_macro(definition, values) if valid else None
                        if projected:
                            for index, projection in enumerate(projected):
                                record = self._reference_record({**call, 'command': token.arg}, projection['keys'],
                                    projection['syntax_status'], basis='bounded_literal_macro_projection',
                                    projected_command=projection['command'], macro_chain=projection['macro_chain'], projection_index=index,
                                    definition_spans=[_span(self.definitions[i]['source']) for i in projection['macro_chain']])
                                records.append(record)
                        elif 'ref' in str(definition.get('body', '')).lower() or any(re.search(r'\\(?:' + '|'.join(sorted(REFERENCE_COMMANDS)) + r')\b', value) for value in values):
                            records.append(self._reference_record(call, [], 'unknown', basis='unexpanded_macro_reference',
                                unresolved_reason='Macro execution/argument liveness cannot be certified by the bounded literal adapter.',
                                definition_spans=[_span(definition['source'])]))
                    cursor = max(cursor, end)
                    continue
                if re.search(r'(?:ref|hyperlink)', token.arg, re.I):
                    # Unknown custom reference commands are explicit obligations.
                    end, raw_keys, _ = lexer.reference(token.arg, cursor)
                    if end > cursor:
                        cursor = end
                    event = {'source': {'file': file, 'char_start': token.pos, 'char_end': cursor},
                             'command': token.arg, 'activity': activity}
                    records.append(self._reference_record(event, [normalize_key(k) for k in raw_keys], 'unknown',
                        basis='unknown_reference_command', unresolved_reason='Unknown command semantics; keys are literal hints only.'))
        # E1 can find references inside titles/known wrappers not reached by the
        # supplemental scanner.  Preserve each occurrence exactly once.
        for identity, event in existing.items():
            if identity not in seen:
                records.append(self._reference_record(event, event.get('keys', []), 'supported'))
        for event in self.project.structure_data.get('citations', []):
            if event.get('source_role') == 'document_or_bibliography' and not event['source']['file'].endswith(('.bib', '.bbl')):
                records.append(self._external_record(event, event.get('keys', []), basis='bibliographic_reference',
                                                     locator_options=event.get('options', [])))
        records.extend(self._bibliographic_brackets(records))
        # Literal numbered locators are explicit source clues, but automatic
        # theorem counters are not recomputed here.  Retain their exact text as
        # unresolved locators for Agent endpoint resolution, including forward
        # mentions.  Self headings, comments and reference-command arguments
        # are not duplicate reference events.
        for file, text in self.project._texts.items():
            if file.endswith(('.sty', '.cls', '.bib', '.bbl')):
                continue
            hidden = list(self._excluded(file))
            hidden += [r['span'] for r in self.explicit if r['basis'] in ('formatted_mathematical_heading', 'plain_mathematical_heading')]
            hidden += [r['span'] for r in records if r['span']['file'] == file]
            for match in NUMBERED_MENTION.finditer(text):
                span = {'file': file, 'char_start': match.start(), 'char_end': match.end()}
                if any(_overlaps(span, h) for h in hidden):
                    continue
                key = 'text:' + match[1].lower() + ':' + match[2]
                for occurrence in self._occurrences(span):
                    event = {'source': span, 'file_occurrence': occurrence, 'command': 'literal_numbered_locator',
                             'activity': 'unmapped' if occurrence is None else 'requires_activity_review'}
                    record = self._reference_record(event, [key], 'unknown', basis='handwritten_numbered_reference',
                        locator={'kind': match[1].lower(), 'number': match[2]},
                        unresolved_reason='Literal source locator found; automatic numbering/endpoint needs review.')
                    record['targets'] = self._numbered_heading_targets(key, match[1].lower(), match[2])
                    records.append(record)
            # References such as 'Y in (1.1)' or 'the triangle (2.2)' do not
            # contain a result noun or a ref command. Preserve exact source,
            # each include occurrence and an UNKNOWN endpoint/type obligation.
            # Exclude existing command/named references and explicit tags;
            # discovering this string alone never creates a mathematical edge.
            parenthesized_hidden = hidden + [r['span'] for r in records if r['span']['file'] == file]
            parenthesized_hidden += [{'file': file, 'char_start': m.start(), 'char_end': m.end()}
                for m in re.finditer(r'\\tag\*?\s*\{[^{}]*\}', text)]
            for match in PARENTHESIZED_LOCATOR.finditer(text):
                span = {'file': file, 'char_start': match.start(), 'char_end': match.end()}
                if any(_overlaps(span, h) for h in parenthesized_hidden):
                    continue
                key = 'text:equation:' + match[1]
                for occurrence in self._occurrences(span):
                    event = {'source': span, 'file_occurrence': occurrence,
                             'command': 'literal_parenthesized_numbered_locator',
                             'activity': 'unmapped' if occurrence is None else 'requires_activity_review'}
                    records.append(self._reference_record(event, [key], 'unknown',
                        basis='handwritten_parenthesized_reference',
                        locator={'kind': 'equation', 'number': match[1], 'kind_status': 'candidate'},
                        unresolved_reason='Parenthesized dotted locator found; whether this is an equation reference and its actual endpoint require source review.'))
        unique = {r['item_id']: r for r in records}
        result = list(unique.values())
        result.sort(key=lambda r: (r['reading_ranges'][0]['char_start'] if r['reading_ranges'] else float('inf'),
                                  r['span']['file'], r['span']['char_start'], r['item_id']))
        return result

    def page(self, kind, offset=0, limit=50, preview_chars=600, item_id=None):
        records = {'explicit': self.explicit, 'prose': self.prose, 'reference': self.references}[kind]
        if item_id is not None:
            records = [r for r in records if r['item_id'] == item_id]
            if not records:
                raise SourceError('unknown inventory item ID')
        start, count = _integer(offset, 'offset'), min(_integer(limit, 'limit', 1), 100)
        width = min(_integer(preview_chars, 'preview_chars'), 24000)
        candidates = records[start:start + count]
        # The prose page itself can supply complete reading receipts.  Bound
        # response text, not inventory coverage; next_offset uses actual count.
        if kind == 'prose':
            selected_records, characters = [], 0
            for record in candidates:
                size = sum(len(t) for t in record['original_text'])
                if selected_records and characters + size > 24000:
                    break
                selected_records.append(record); characters += size
            candidates = selected_records
        selected = copy.deepcopy(candidates)
        # Source-bound original text is always available through read_paragraph
        # or spans; truncation is explicit and never changes inventory digests.
        for record, original in zip(selected, candidates):
            text = ''.join(original['original_text'])
            record['original_text'] = [part[:width] for part in original['original_text']] if width else []
            record['text_truncated'] = any(len(t) > width for t in original['original_text']) if width else bool(text)
            record['text_complete'] = not record['text_truncated']
            record['preview'] = text[:width]
        return {'schema': SCHEMA, 'kind': kind, 'total': len(records), 'offset': start,
                'records': selected, 'next_offset': start + len(selected) if start + len(selected) < len(records) else None,
                'inventory_sha256': self.digests[kind],
                'notice': 'Syntax and source coverage only; not accepted knowledge, actual-use decisions, or mathematical correctness.'}

    def read_paragraph(self, item_id, offset=0, max_chars=24000):
        record = next((r for r in self.prose if r['item_id'] == item_id), None)
        if record is None:
            raise SourceError('unknown paragraph inventory ID')
        first, last = record['reading_ranges'][0]['char_start'], record['reading_ranges'][0]['char_end']
        start = first + _integer(offset, 'offset')
        if start > last:
            raise SourceError('paragraph offset exceeds its range')
        end = min(last, start + min(_integer(max_chars, 'max_chars', 1), 24000))
        spans = self._segments(start, end)
        return {'item_id': item_id, 'offset': offset, 'total_characters': last - first,
                'segments': [{'span': s, 'text': self.project.resolve_span(s)} for s in spans],
                'next_offset': end - first if end < last else None, 'inventory_sha256': self.digests['prose']}

    def summary(self):
        return {'schema': SCHEMA, 'counts': {'explicit': len(self.explicit), 'prose': len(self.prose),
                                            'reference': len(self.references)},
                'inventory_sha256': self.digests,
                'reference_resolution_counts': dict(Counter(r['resolution'] for r in self.references)),
                'limits': ['No general TeX execution or arbitrary macro expansion.',
                           'Unknown activity, title boundaries and macro semantics require explicit review.',
                           'Handwritten numbered/ named references are not exhaustively recognized by this syntax inventory.']}
