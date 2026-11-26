"""Pinned, read-only source access with format-specific discovery indexes.

The structure index is a discovery aid. It is never converted to accepted
mathematical knowledge here, and neither TeX nor author commands are executed.
"""
from __future__ import annotations

import bisect
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from gangmu.latex_structure import StructureExtractor
from gangmu.representation.graph import validate
from gangmu.representation.model import Document, Graph, Source, digest
from gangmu.representation.sources import E1Corpus, SourceError, _decode, _file, _relative

from .retrieval import BM25Index, PARAMETERS as BM25_PARAMETERS, ignored_macros, tokenize

TEXT_SUFFIXES = {'.tex', '.sty', '.cls', '.bib', '.bbl', '.ltx', '.def', '', '.md', '.markdown'}
MAX_READ = 24000


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _int(value: Any, name: str, low: int = 0) -> int:
    if type(value) is not int or value < low:
        raise SourceError(f'{name} must be an integer >= {low}')
    return value


class SourceProject:
    """A document, its original source inventory, and its syntactic indexes.

    Reads recheck the pinned digest. Public tool responses expose relative
    author file names only, never private configuration or host paths.
    """

    @classmethod
    def from_artifact(cls, artifact_dir: str | Path, paper_id: str | None = None):
        directory = Path(artifact_dir).resolve()
        if not (directory / 'artifact.json').is_file():
            if (directory.parent / 'artifact.json').is_file():
                paper_id = paper_id or directory.name
                directory = directory.parent
            else:
                raise SourceError('expected E1 corpus artifact root or one paper directory')
        corpus = E1Corpus(directory)
        if paper_id is None:
            if len(corpus.papers) != 1:
                raise SourceError('paper_id is required for a multi-document artifact')
            paper_id = next(iter(corpus.papers))
        row = corpus._paper(paper_id)
        structure = corpus.record(paper_id, 'preprocess/structure.json')
        quality = corpus.record(paper_id, 'preprocess/quality.json')
        mapping = corpus.record(paper_id, 'preprocess/source_map.json')
        markdown = structure.get('schema_version') == 'gangmu.markdown-structure.v1'
        reading_name = 'reading.md' if markdown else 'reading.tex'
        reading = corpus.read(f'{paper_id}/preprocess/{reading_name}').decode('utf-8')
        if structure.get('paper_id') != paper_id or structure.get('arxiv_id') != row['arxiv_id']:
            raise SourceError('preprocessing identity differs from the selected paper')
        if structure.get('schema_version') not in {'gangmu.latex-structure.v1', 'gangmu.markdown-structure.v1'}:
            raise SourceError('unsupported preprocessing schema')
        files = {path[len(paper_id + '/source/'):]: value for path, value in corpus.files.items()
                 if path.startswith(paper_id + '/source/')}
        if not files:
            raise SourceError('source inventory is empty')
        self = cls.__new__(cls)
        self.source_root = directory / paper_id / 'source'
        self.paper_id, self.arxiv_id = paper_id, row['arxiv_id']
        self.main_file = structure['main_file']
        self.corpus = corpus
        self.artifact_version, self.artifact_tree_sha256 = corpus.version, corpus.tree_sha256
        self.source_files = files
        self.preprocess_sha256 = digest({name: corpus.files[f'{paper_id}/preprocess/{name}']
                                        for name in ('structure.json', 'quality.json', 'source_map.json', reading_name)})
        self.structure_data, self.reading, self.source_map, self.quality = structure, reading, mapping, quality
        self._initialize()
        return self

    @classmethod
    def from_source(cls, source_dir: str | Path, main_file: str, paper_id: str,
                    arxiv_id: str):
        root = Path(source_dir).resolve()
        _relative(main_file)
        if not paper_id or not arxiv_id:
            raise SourceError('explicit document and version identifiers are required')
        files = {}
        for path in sorted(root.rglob('*')):
            if path.is_symlink():
                raise SourceError('symlinks are not accepted in source projects')
            if path.is_file():
                if path.stat().st_size > 30_000_000:
                    raise SourceError('source project file exceeds 30MB limit')
                files[path.relative_to(root).as_posix()] = _sha(path.read_bytes())
        _file(root, main_file)
        if Path(main_file).suffix.lower() in {'.md', '.markdown'}:
            from gangmu.markdown_structure import extract
            structure, reading, mapping, quality = extract(root, main_file, paper_id=paper_id, arxiv_id=arxiv_id)
        else:
            structure, reading, mapping, quality = StructureExtractor(
                root, main_file, paper_id=paper_id, arxiv_id=arxiv_id).extract()
        self = cls.__new__(cls)
        self.source_root, self.paper_id, self.arxiv_id, self.main_file = root, paper_id, arxiv_id, main_file
        self.corpus, self.source_files = None, files
        self.artifact_tree_sha256 = digest(files)
        self.artifact_version = 'e4-source-' + self.artifact_tree_sha256[:16]
        self.structure_data, self.reading, self.source_map, self.quality = structure, reading, mapping, quality
        # The absolute source-root field is historical metadata, not semantic input.
        portable_structure = {k: v for k, v in structure.items() if k != 'source_root'}
        self.preprocess_sha256 = digest({'structure': portable_structure, 'map': mapping,
                                        'reading_sha256': _sha(reading.encode()), 'quality': quality})
        self._initialize()
        return self

    def _initialize(self):
        self.source_format = self.structure_data.get('source_format', 'latex')
        self.document = Document(id='doc:' + self.paper_id, paper_id=self.paper_id,
                                 arxiv_id=self.arxiv_id, artifact_version=self.artifact_version,
                                 artifact_tree_sha256=self.artifact_tree_sha256)
        self.fingerprint = {'paper_id': self.paper_id, 'arxiv_id': self.arxiv_id,
                            'main_file': self.main_file, 'source_files': self.source_files,
                            'artifact_version': self.artifact_version,
                            'artifact_tree_sha256': self.artifact_tree_sha256,
                            'preprocess_sha256': self.preprocess_sha256}
        if self.source_format == 'markdown':
            self.fingerprint['source_format'] = 'markdown'
        self._texts, self._line_starts = {}, {}
        for relative in self.source_files:
            _relative(relative)
            if Path(relative).suffix.lower() in TEXT_SUFFIXES:
                text = self._read_text(relative)
                self._texts[relative] = text
                self._line_starts[relative] = [0] + [i + 1 for i, c in enumerate(text) if c == '\n']
        if self.main_file not in self._texts:
            raise SourceError('main file is not a readable text source')
        self._events = {record['id']: record for value in self.structure_data.values()
                        if isinstance(value, list) for record in value
                        if isinstance(record, dict) and 'id' in record and 'source' in record}
        self._labels = {}
        for event in self.structure_data.get('labels', []):
            for key in event.get('keys', []):
                self._labels.setdefault(key, []).append(event)
        # Paragraph chunks are exhaustive over eligible original text, including
        # files missed by static include traversal. They are retrieval units only.
        self._chunks = []
        self._retrieval_ignored_macros = ignored_macros(self.structure_data)
        for file, text in self._texts.items():
            if Path(file).suffix.lower() in {'.bib', '.bbl'}:
                continue
            starts = [0] + [m.end() for m in re.finditer(r'\n\s*\n', text)]
            starts.append(len(text))
            for left, right in zip(starts, starts[1:]):
                for start in range(left, right, 1800):
                    end = min(start + 1800, right)
                    if text[start:end].strip():
                        self._chunks.append({'span': {'file': file, 'char_start': start, 'char_end': end},
                                             'tokens': Counter(self._tokens(text[start:end]))})
        self._df = Counter(t for c in self._chunks for t in c['tokens'])
        self._search_index = BM25Index([chunk['tokens'] for chunk in self._chunks],
                                      k1=BM25_PARAMETERS['k1'], b=BM25_PARAMETERS['b'])

    def _tokens(self, text: str) -> list[str]:
        return tokenize(text, ignored=self._retrieval_ignored_macros)

    def _read_text(self, file: str) -> str:
        _relative(file)
        if file not in self.source_files or Path(file).suffix.lower() not in TEXT_SUFFIXES:
            raise SourceError('file is outside the allowed source inventory')
        raw = (self.corpus.read(f'{self.paper_id}/source/{file}') if self.corpus else
               _file(self.source_root, file).read_bytes())
        if _sha(raw) != self.source_files[file]:
            raise SourceError('source file changed after project was pinned')
        return _decode(raw)[0]

    def assert_unchanged(self):
        actual = set()
        for path in self.source_root.rglob('*'):
            if path.is_symlink():
                raise SourceError('symlink appeared in the pinned source project')
            if path.is_file():
                actual.add(path.relative_to(self.source_root).as_posix())
        if actual != set(self.source_files):
            raise SourceError('source project file inventory changed after pinning')
        for file, expected in self.source_files.items():
            raw = (self.corpus.read(f'{self.paper_id}/source/{file}') if self.corpus else
                   _file(self.source_root, file).read_bytes())
            if _sha(raw) != expected:
                raise SourceError('source inventory changed after project was pinned')
        return {'unchanged': True, 'files': len(self.source_files)}

    def normalize_span(self, span: dict) -> dict:
        if not isinstance(span, dict):
            raise SourceError('span must be an object')
        file = span.get('file')
        text = self._read_text(file)
        start = _int(span.get('char_start', span.get('start')), 'char_start')
        end = _int(span.get('char_end', span.get('end')), 'char_end')
        if not 0 <= start < end <= len(text):
            raise SourceError('source span must be nonempty and within the original file')
        result = {'file': file, 'char_start': start, 'char_end': end}
        if span.get('occurrence') is not None:
            occurrence = span['occurrence']
            if not isinstance(occurrence, str) or not occurrence:
                raise SourceError('occurrence must be a nonempty string')
            if self.corpus:
                self.corpus.anchor(self.paper_id, result, occurrence=occurrence)
            else:
                intervals = sorted((s['source']['char_start'], s['source']['char_end']) for s in self.source_map
                                   if s['occurrence'] == occurrence and s['source']['file'] == file)
                cursor = start
                for a, b in intervals:
                    if a <= cursor <= b:
                        cursor = max(cursor, b)
                if cursor < end:
                    raise SourceError('source span is outside the specified reading occurrence')
            result['occurrence'] = occurrence
        return result

    def resolve_span(self, span: dict) -> str:
        checked = self.normalize_span(span)
        return self._read_text(checked['file'])[checked['char_start']:checked['char_end']]

    def source(self, span: dict) -> Source:
        return Source(document=self.document.id, **self.normalize_span(span))

    def overview(self) -> dict:
        return {'paper_id': self.paper_id, 'arxiv_id': self.arxiv_id, 'main_file': self.main_file,
                'source_version': self.artifact_version, 'source_tree_sha256': self.artifact_tree_sha256,
                'files': [{'file': f, 'characters': len(t), 'lines': len(self._line_starts[f]),
                           'sha256': self.source_files[f]} for f, t in self._texts.items()],
                'structure_counts': {k: len(v) for k, v in self.structure_data.items() if isinstance(v, list)},
                'quality': {k: self.quality.get(k) for k in ('status', 'unresolved_by_reason',
                             'unclosed_environments', 'recorder_untraversed_author_inputs')},
                'notice': 'Structure and reference links are candidates, not accepted nodes or dependencies.'}

    def _inventory(self):
        # Import lazily so legacy read-only retrieval does not build the full
        # two-round inventories, and retain one immutable source-bound instance.
        from gangmu.single_document.inventory import SourceInventory
        if not hasattr(self, '_source_inventory'):
            self._source_inventory = SourceInventory(self)
        self.assert_unchanged()
        return self._source_inventory

    def explicit_inventory(self, offset: int = 0, limit: int = 50,
                           preview_chars: int = 600, item_id: str | None = None) -> dict:
        return self._inventory().page('explicit', offset, limit, preview_chars, item_id)

    def paragraph_inventory(self, offset: int = 0, limit: int = 50,
                            preview_chars: int = 6000, item_id: str | None = None) -> dict:
        return self._inventory().page('prose', offset, limit, preview_chars, item_id)

    def reference_ledger(self, offset: int = 0, limit: int = 50,
                         preview_chars: int = 600, item_id: str | None = None) -> dict:
        return self._inventory().page('reference', offset, limit, preview_chars, item_id)

    def read_paragraph(self, item_id: str, offset: int = 0, max_chars: int = MAX_READ) -> dict:
        return self._inventory().read_paragraph(item_id, offset, max_chars)

    @staticmethod
    def _pure_mathematical_title(text: str) -> bool:
        """Recognize a whole syntactic title, never remove mathematical prose.

        A formatted inventory candidate may contain an entire assertion. Only
        these narrow wrappers and a full title match permit a body boundary.
        """
        text = text.strip()
        wrapped = re.fullmatch(r"\{\s*\\(?:bf|it|rm|sc)\s+([^{}]+)\}", text)
        if wrapped is None:
            wrapped = re.fullmatch(r"\\(?:textbf|textsc|emph)\s*\{([^{}]+)\}", text)
        if wrapped is not None:
            text = wrapped[1].strip()
        name = r"(?:Theorem|Lemma|Proposition|Corollary|Definition(?:-Proposition)?|Proof|Claim|Conjecture|Question|Notation|Remark|Example|Assumption)"
        number = r"(?:[A-Z]?\d+(?:\.\d+)*[a-z]?|[A-Z])"
        return re.fullmatch(name + r"(?:\s+of\s+" + name + r")?(?:\s+\(?" + number + r"\)?)?[.:]?",
                            text, re.I) is not None

    def _decision_header_ranges(self):
        """Host-inventoried pure headings/openings in reading coordinates.

        Plain/formatted mentions in a sentence cannot create a new sentence
        boundary. Environment arguments are removed only when they are one
        complete, plain title; assumptions and arbitrary macros stay intact.
        """
        inventory = self._inventory()
        if hasattr(self, '_decision_headers'):
            return self._decision_headers
        headers = []
        for record in inventory.explicit:
            basis = record.get('basis')
            if basis in ('formatted_mathematical_heading', 'plain_mathematical_heading'):
                spans = record.get('source_spans', [record['span']])
                if len(spans) != 1:
                    continue
                literal = self.resolve_span(spans[0])
                if not self._pure_mathematical_title(literal):
                    continue
                wrapper = re.match(r'(?:\{\s*\\|\\)(textbf|textsc|emph|bf|it|rm|sc)\b', literal.lstrip())
                if wrapper is not None and any(value.get('name', '').lstrip('\\') == wrapper[1]
                        for value in self.structure_data.get('definitions', [])):
                    continue
                span = spans[0]
                text = self._texts[span['file']]
                prefix = text[:span['char_start']]
                breaks = list(re.finditer(r'\r?\n[ \t]*\r?\n', prefix))
                paragraph_prefix = prefix[breaks[-1].end() if breaks else 0:]
                if not re.fullmatch(r'\s*(?:\\noindent\s*)?', paragraph_prefix):
                    continue
                # An overridden command is not a known layout-only prefix.
                if '\\noindent' in paragraph_prefix and any(
                        value.get('name') == 'noindent' for value in self.structure_data.get('definitions', [])):
                    continue
                headers.extend((value['char_start'], value['char_end'])
                               for value in record.get('reading_ranges', []))
            elif basis == 'named_environment':
                event = inventory.events.get(record.get('event_id'), {})
                opening = event.get('source')
                if not opening:
                    continue
                arguments = event.get('arguments', [])
                if arguments and (len(arguments) != 1 or not self._pure_mathematical_title(arguments[0].get('value', ''))):
                    continue
                literal = self.resolve_span(opening)
                pattern = r'\\begin\s*\{' + re.escape(event.get('name', '')) + r'\}'
                if arguments:
                    # No nested macros/braces or a guessed unclosed option.
                    pattern += r'\s*\[[^\[\]{}\\]*\]'
                if re.fullmatch(pattern, literal) is None:
                    continue
                occurrence = record['span'].get('occurrence')
                anchored = dict(opening, occurrence=occurrence) if occurrence is not None else opening
                starts, ends = inventory._positions(anchored), inventory._positions(anchored, end=True)
                for first in starts:
                    last = next((end for end in ends if end > first), None)
                    if last is not None:
                        headers.append((first, last))
        self._decision_headers = tuple(sorted(set(headers)))
        return self._decision_headers

    def _decision_closing_ranges(self):
        """Locate actual theorem/proof endings, never guessed source macros."""
        if hasattr(self, '_decision_closings'):
            return self._decision_closings
        inventory, closings = self._inventory(), []
        for event in self.structure_data.get('environments', []):
            ending = event.get('end_source')
            if (not event.get('closed') or not ending
                    or event.get('environment_kind') not in ('theorem', 'proof')
                    or event.get('activity') != 'active'):
                continue
            if re.fullmatch(r'\\end\s*\{' + re.escape(event['name']) + r'\}',
                            self.resolve_span(ending)) is None:
                continue
            anchored = dict(ending, occurrence=event['file_occurrence'])
            starts = inventory._positions(anchored)
            ends = inventory._positions(anchored, end=True)
            for first in starts:
                last = next((end for end in ends if end > first), None)
                if last is not None:
                    closings.append((first, last))
        self._decision_closings = tuple(sorted(set(closings)))
        return self._decision_closings

    def decision_fragments(self, spans: list[dict], *, reference: bool = False) -> list[list[dict]]:
        """Exact complete reading fragments; locations are not semantic approval.

        Preserve original coordinates and reading occurrences. Only host-known
        pure titles/openings can separate a body from its syntax header. The
        original sentence scanner, math/braces and coverage rules stay intact.
        """
        inventory = self._inventory()
        headers, intervals = self._decision_header_ranges(), []
        closings = self._decision_closing_ranges()
        for raw in spans:
            span = self.normalize_span(raw)
            occurrences = {row['occurrence'] for row in self.source_map
                if row['source']['file'] == span['file']
                and row['source']['char_start'] < span['char_end']
                and span['char_start'] < row['source']['char_end']}
            occurrence = span.get('occurrence')
            if occurrence is None and len(occurrences) == 1:
                occurrence = next(iter(occurrences))
            if occurrence not in occurrences:
                continue
            anchored = dict(span, occurrence=occurrence)
            starts, ends = inventory._positions(anchored), inventory._positions(anchored, end=True)
            for first in starts:
                last = next((end for end in ends if end > first), None)
                if last is None:
                    continue
                if reference:
                    # A reference in a pure title names source syntax, such as
                    # the theorem a proof is attributed to. Preserve the whole
                    # host-known title for that review; a bare name is not a
                    # complete fragment. Target mathematical uses still follow
                    # the body-only path below.
                    for a, b in headers:
                        if a <= first < last <= b:
                            segments = inventory._segments(a, b)
                            if segments not in intervals:
                                intervals.append(segments)
                    separators = list(re.finditer(r'\r?\n[ \t]*\r?\n', self.reading))
                    lower = max([0] + [m.end() for m in separators if m.end() <= first])
                    upper = min([len(self.reading)] + [m.start() for m in separators if m.start() >= last])
                else:
                    lower, upper = first, last
                # A standalone inventory heading can still be reviewed as
                # source syntax. Separate it only when this scope has a body.
                separators = [(a, b) for a, b in headers if lower <= a < b < upper
                              and self.reading[b:upper].strip()]
                cursor, bodies = lower, []
                for a, b in separators:
                    if a < cursor:
                        continue
                    before = self.reading[cursor:a].strip()
                    layout_prefix = re.fullmatch(r'\\noindent\s*', before) is not None
                    if before and not layout_prefix and not before.endswith(('.', '!', '?')):
                        # Do not turn an unfinished preceding assumption into
                        # a new complete witness merely by inserting a title.
                        continue
                    if cursor < a:
                        bodies.append((cursor, a))
                    cursor = b
                bodies.append((cursor, upper))
                for left, right in bodies:
                    for a, b in inventory._sentence_ranges(left, right):
                        # Sentence punctuation can leave the previous theorem's
                        # closing command attached to the next proof. Remove
                        # only leading, inventoried endings and whitespace;
                        # never drop a preceding assumption or mathematical text.
                        for start, end in closings:
                            if a <= start < end <= b and not self.reading[a:start].strip():
                                a = end
                                while a < b and self.reading[a].isspace():
                                    a += 1
                        if reference and not (a < last and first < b):
                            continue
                        literal = self.reading[a:b]
                        stripped = re.sub(r'\\(?:begin|end|label)\s*\{[^{}]*\}', '', literal).strip()
                        if not stripped or re.fullmatch(r'\$*\\[A-Za-z@]+\$*', stripped):
                            continue
                        segments = inventory._segments(a, b)
                        if segments not in intervals:
                            intervals.append(segments)
        return intervals

    def fragment_record(self, spans: list[dict]) -> dict:
        """One complete fragment, with explicit actual-text and position keys."""
        source_spans = [self.normalize_span(span) for span in spans]
        return {'fragment_id': 'fragment:' + digest({'source': self.fingerprint, 'spans': source_spans})[:24],
                'source_spans': source_spans,
                'original_text': [self.resolve_span(span) for span in source_spans],
                'span': dict(source_spans[0]) if len(source_spans) == 1 else None}

    def read(self, file: str | None = None, char_start: int | None = None,
             char_end: int | None = None, line_start: int | None = None,
             line_end: int | None = None, event_id: str | None = None,
             max_chars: int = MAX_READ) -> dict:
        limit = min(_int(max_chars, 'max_chars', 1), MAX_READ)
        if event_id is not None:
            event = self._events.get(event_id)
            if not event:
                raise SourceError('unknown structure event')
            span = event.get('full_source') or event['source']
            file, char_start, char_end = span['file'], span['char_start'], span['char_end']
        file = file or self.main_file
        text = self._read_text(file)
        if line_start is not None:
            starts = self._line_starts[file]
            first = _int(line_start, 'line_start', 1)
            last = _int(line_end if line_end is not None else min(first + 49, len(starts)), 'line_end', 1)
            if not first <= last <= len(starts):
                raise SourceError('line range outside source file')
            char_start, char_end = starts[first - 1], starts[last] if last < len(starts) else len(text)
        start = 0 if char_start is None else _int(char_start, 'char_start')
        requested_end = min(len(text), start + limit) if char_end is None else _int(char_end, 'char_end')
        if not 0 <= start <= requested_end <= len(text):
            raise SourceError('character range outside source file')
        end = min(requested_end, start + limit)
        return {'span': {'file': file, 'char_start': start, 'char_end': end},
                'text': text[start:end], 'truncated': end < requested_end,
                'next_char': end if end < requested_end else None,
                'line_start': bisect.bisect_right(self._line_starts[file], start),
                'line_end': bisect.bisect_right(self._line_starts[file], end)}

    def structure(self, kind: str = 'environments', offset: int = 0, limit: int = 40,
                  event_ids: list[str] | None = None, file: str | None = None,
                  preview_chars: int = 500) -> dict:
        allowed = {'sections', 'environments', 'math_spans', 'format_cues', 'labels', 'references',
                   'citations', 'definitions', 'includes', 'macro_calls', 'conditions', 'unresolved'}
        if kind not in allowed:
            raise SourceError('unsupported structure index')
        records = self.structure_data.get(kind, [])
        if event_ids is not None:
            records = [r for r in records if r['id'] in event_ids]
        if file is not None:
            self._read_text(file)
            records = [r for r in records if r['source']['file'] == file]
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 100)
        preview = min(_int(preview_chars, 'preview_chars'), 1200)
        selected = []
        for original in records[start:start + count]:
            # Source-root/host paths are never returned; these records contain
            # only author-relative sources and static extraction metadata.
            record = dict(original)
            span = record.get('full_source') or record['source']
            if preview:
                record['preview'] = self.resolve_span(span)[:preview]
            selected.append(record)
        return {'kind': kind, 'total': len(records), 'offset': start, 'records': selected,
                'next_offset': start + len(selected) if start + len(selected) < len(records) else None}

    def labels(self, key: str | None = None, offset: int = 0, limit: int = 50) -> dict:
        if key is None:
            return self.structure('labels', offset=offset, limit=limit)
        return {'key': key, 'records': self._labels.get(key, []),
                'ambiguous': len(self._labels.get(key, [])) > 1}

    def candidates(self, kind: str = 'all', offset: int = 0, limit: int = 40,
                   preview_chars: int = 600) -> dict:
        """Broad discovery inventory, never an automatic acceptance rule.

        Formal environments, bold result cues, prose definitions and referenced
        display expressions remain separate candidate families. In particular,
        an arbitrary display formula is not promoted to an expression node.
        """
        if kind not in ('all', 'formal', 'informal', 'expression'):
            raise SourceError('candidate kind must be all, formal, informal or expression')
        from .inventory import SourceInventory
        records = []
        definitions = {e['id']: e for e in self.structure_data.get('definitions', [])}
        if kind in ('all', 'formal'):
            for event in self.structure_data.get('environments', []):
                if event.get('environment_kind') != 'theorem' or event.get('activity') == 'inactive':
                    continue
                if event.get('source_role') != 'document_or_bibliography':
                    continue
                title = definitions.get(event.get('definition_id'), {}).get('body', event['name'])
                category = SourceInventory._category(str(title))
                records.append({'candidate_id': event['id'], 'event_id': event['id'],
                                'basis': 'named_environment', 'name': event['name'], 'category_hint': category,
                                'span': {**(event.get('full_source') or event['source']),
                                         'occurrence': event.get('file_occurrence')},
                                'section_ids': event.get('section_ids', []), 'activity': event['activity'],
                                'boundary_status': 'complete_static_span' if event.get('closed') and event.get('full_source') else 'unresolved'})
        if kind in ('all', 'informal'):
            for event in self.structure_data.get('format_cues', []):
                if event.get('activity') != 'inactive' and re.search(
                        r'\b(theorem|lemma|proposition|corollary|definition|proof|claim)\b', event.get('raw_content', ''), re.I):
                    records.append({'candidate_id': event['id'], 'event_id': event['id'],
                                    'basis': 'formatted_mathematical_heading',
                                    'span': {**event['source'], 'occurrence': event.get('file_occurrence')},
                                    'boundary_status': 'heading_only_agent_must_determine_body',
                                    'category_hint': SourceInventory._category(event['raw_content'])})
            pattern = re.compile(r'\b(?:is called|are called|we (?:define|write|denote)|by definition|is defined|are defined)\b', re.I)
            statement_pattern = re.compile(r'\b(?:there (?:is|exists)|there are|if and only if|'
                                           r'one[- ]to[- ]one correspondence|it follows that|'
                                           r'we (?:prove|show)|the following result)\b', re.I)
            for chunk in self._chunks:
                raw = self.resolve_span(chunk['span'])
                occurrences = {item['occurrence'] for item in self.source_map
                    if item['source']['file'] == chunk['span']['file']
                    and item['source']['char_start'] < chunk['span']['char_end']
                    and chunk['span']['char_start'] < item['source']['char_end']} or {None}
                anchored = []
                for occurrence in sorted(occurrences, key=lambda value: value or ''):
                    proposed = {**chunk['span'], **({'occurrence': occurrence} if occurrence is not None else {})}
                    try:
                        anchored.append(self.normalize_span(proposed))
                    except SourceError:
                        # Physical chunks can cross an include directive or
                        # unmapped region. Preserve the source clue without
                        # inventing a single effective reading occurrence.
                        continue
                for span in anchored or [chunk['span']]:
                    if pattern.search(raw):
                        records.append({'candidate_id': 'prose:' + digest(span)[:16],
                                        'basis': 'prose_definition_cue', 'span': span,
                                        'category_hint': 'definition', 'boundary_status': 'paragraph_agent_must_delimit'})
                    if statement_pattern.search(raw):
                        records.append({'candidate_id': 'prose-result:' + digest(span)[:16],
                                        'basis': 'prose_result_cue', 'span': span,
                                        'category_hint': 'statement', 'boundary_status': 'paragraph_agent_must_delimit'})
        if kind in ('all', 'expression'):
            # Share the full-ledger display hints with the two-round inventory.
            # SourceInventory seeds headings before scanning references and
            # never calls this method, so initialization is acyclic.
            for record in self._inventory().explicit:
                if record.get('category_hint') != 'expression':
                    continue
                event_id = record['event_id']
                records.append({'candidate_id': event_id, 'event_id': event_id,
                    'basis': 'independently_referenced_display'
                        if record['basis'] == 'referenced_display_environment' else record['basis'],
                    'span': dict(record['span']), 'category_hint': 'expression',
                    'labels': list(record.get('labels', [])), 'boundary_status': record['boundary_status'],
                    **{key: list(record[key]) if isinstance(record[key], list) else record[key]
                       for key in ('candidate_only', 'preserve_surrounding_prose',
                        'reference_item_ids', 'reference_endpoint_status') if key in record}})
        records.sort(key=lambda x: (x['span']['file'], x['span']['char_start'], x['candidate_id']))
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 100)
        selected = records[start:start + count]
        width = min(_int(preview_chars, 'preview_chars'), 1500)
        if width:
            selected = [{**r, 'preview': self.resolve_span(r['span'])[:width]} for r in selected]
        return {'kind': kind, 'total': len(records), 'records': selected,
                'next_offset': start + len(selected) if start + len(selected) < len(records) else None,
                'notice': 'Hints can overlap, be irrelevant or omit knowledge. Agent judgment and full reading remain required.'}

    def read_section(self, section_id: str, offset: int = 0, max_chars: int = MAX_READ) -> dict:
        """Read a section in E1 traversal order with original segment mappings."""
        sections = self.structure_data.get('sections', [])
        section = next((e for e in sections if e['id'] == section_id), None)
        if section is None:
            raise SourceError('unknown section ID')
        levels = {'part': 0, 'chapter': 1, 'section': 2, 'subsection': 3,
                  'subsubsection': 4, 'paragraph': 5, 'subparagraph': 6}
        level = levels.get(section['command'], 2)
        def position(event):
            span = event['source']
            values = [s['reading_char_start'] + span['char_start'] - s['source']['char_start']
                      for s in self.source_map if s['source']['file'] == span['file']
                      and s['source']['char_start'] <= span['char_start'] < s['source']['char_end']
                      and s['occurrence'] == event.get('file_occurrence', 'main')]
            return min(values) if values else None
        first = position(section)
        if first is None:
            raise SourceError('section has no mapped reading occurrence')
        bounds = [p for e in sections if levels.get(e['command'], 2) <= level
                  and (p := position(e)) is not None and p > first]
        last = min(bounds, default=len(self.reading))
        start = first + _int(offset, 'offset')
        if start > last:
            raise SourceError('section reading offset is outside its range')
        end = min(last, start + min(_int(max_chars, 'max_chars', 1), MAX_READ))
        segments = []
        for mapping in self.source_map:
            a, b = max(start, mapping['reading_char_start']), min(end, mapping['reading_char_end'])
            if a < b:
                original = mapping['source']
                span = {'file': original['file'],
                        'char_start': original['char_start'] + a - mapping['reading_char_start'],
                        'char_end': original['char_start'] + b - mapping['reading_char_start'],
                        'occurrence': mapping['occurrence']}
                segments.append({'span': span, 'text': self.resolve_span(span)})
        return {'section_id': section_id, 'title': section.get('title'), 'offset': offset,
                'total_reading_characters': last - first, 'segments': segments,
                'next_offset': end - first if end < last else None,
                'notice': 'Text is segmented by original files; use the returned spans for source binding.'}

    def symbols(self, name: str | None = None, offset: int = 0, limit: int = 50,
                pattern: str | None = None) -> dict:
        records = self.structure_data.get('definitions', [])
        if name is not None:
            clean = name.lstrip('\\')
            records = [r for r in records if r.get('name') == clean or clean in r.get('names', [])]
        if pattern is not None:
            if not isinstance(pattern, str) or not pattern:
                raise SourceError('symbol pattern must be nonempty literal text')
            clean = pattern.lstrip('\\').lower()
            records = [r for r in records if clean in r.get('name', '').lower()]
        return {'total': len(records), 'records': records[_int(offset, 'offset'):offset + min(_int(limit, 'limit', 1), 100)],
                'notice': 'TeX macro definitions are syntax, not automatically mathematical definitions.'}

    def find_text(self, text: str, file: str | None = None, offset: int = 0,
                  limit: int = 20, case_sensitive: bool = True) -> dict:
        if not isinstance(text, str) or not text or len(text) > 4000:
            raise SourceError('text must contain 1..4000 characters')
        files = [file] if file else list(self._texts)
        matches = []
        for name in files:
            original = self._read_text(name)
            haystack, needle = (original, text) if case_sensitive else (original.lower(), text.lower())
            cursor = 0
            while True:
                position = haystack.find(needle, cursor)
                if position < 0:
                    break
                matches.append({'file': name, 'char_start': position, 'char_end': position + len(text)})
                cursor = position + max(1, len(text))
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 100)
        return {'total': len(matches), 'matches': matches[start:start + count],
                'next_offset': start + count if start + count < len(matches) else None}

    def search(self, query: str, offset: int = 0, limit: int = 12,
               file: str | None = None, max_chars: int = 1200,
               preview_chars: int | None = None) -> dict:
        if not isinstance(query, str) or not query.strip() or len(query) > 4000:
            raise SourceError('query must contain 1..4000 characters')
        if file is not None:
            self._read_text(file)
        tokens = self._tokens(query)
        eligible = ({index for index, chunk in enumerate(self._chunks) if chunk['span']['file'] == file}
                    if file is not None else None)
        scored = [(hit['score'], self._chunks[hit['index']]['span'])
                  for hit in self._search_index.search(tokens, eligible=eligible)]
        scored.sort(key=lambda x: (-x[0], x[1]['file'], x[1]['char_start']))
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 50)
        width = min(_int(preview_chars if preview_chars is not None else max_chars, 'max_chars', 1), 1800)
        return {'query': query, 'total_candidates': len(scored), 'offset': start,
                'results': [{'score': round(score, 6), 'span': span,
                             'text': self.resolve_span(span)[:width]} for score, span in scored[start:start + count]],
                'next_offset': start + count if start + count < len(scored) else None,
                'method': 'full-source LaTeX-aware BM25; paginate to expand beyond top-k',
                'parameters': dict(BM25_PARAMETERS),
                'notice': 'Similarity is not evidence of a mathematical dependency.'}

    def dependency_candidates(self, spans: list[dict], query: str | None = None,
                              offset: int = 0, limit: int = 30) -> dict:
        checked = [self.normalize_span(s) for s in spans]
        links = []
        references = []
        for reference in self._inventory().references:
            ref = reference['span']
            if not any(s['file'] == ref['file'] and s['char_start'] <= ref['char_start'] < s['char_end'] and
                       (s.get('occurrence') is None or ref.get('occurrence') == s['occurrence']) for s in checked):
                continue
            references.append(reference)
            targets = reference['targets'] or [{'key': None, 'resolution': 'unknown', 'label_spans': [],
                'target_event_ids': [], 'math_owners': [], 'owner_source_spans': []}]
            for target in targets:
                # Missing/nonmathematical/unknown targets remain visible, rather
                # than disappearing before the Agent can classify the reference.
                for label_span in target['label_spans'] or [None]:
                    links.append({'reference_id': reference.get('event_id') or reference['item_id'],
                                  'ledger_item_id': reference['item_id'], 'label_key': target['key'],
                                  'use_span': ref, 'label_span': label_span,
                                  'resolution': target['resolution'], 'syntax_status': reference['syntax_status'],
                                  'activity': reference['activity'], 'target_event_ids': target['target_event_ids'],
                                  'math_owners': target.get('math_owners', []),
                                  'owner_source_spans': target.get('owner_source_spans', []),
                                  'distance': self.distance(ref, label_span) if label_span else None,
                                  'method': 'explicit_label_candidate' if label_span else 'unresolved_explicit_reference'})
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 100)
        result = {'explicit_total': len(links), 'reference_total': len(references),
                  'unresolved_total': sum(r['resolution'] in ('unresolved', 'unknown', 'dynamic', 'ambiguous') for r in references),
                  'explicit': links[start:start + count],
                  'next_offset': start + count if start + count < len(links) else None,
                  'notice': 'Navigation/comparison/negation references require separate judgment.'}
        if query:
            result['retrieval'] = self.search(query, offset=offset, limit=min(limit, 20))
        return result

    def dependency_candidates_legacy(self, spans: list[dict], query: str | None = None,
                                     offset: int = 0, limit: int = 30) -> dict:
        """Historical v2 replay only; preserves the earlier lossy inventory.

        New extraction uses dependency_candidates/reference_ledger.  Retained
        old workflow assets must be checked against their own original policy.
        """
        checked = [self.normalize_span(s) for s in spans]
        links = []
        for reference in self.structure_data.get('references', []):
            ref = reference['source']
            if any(s['file'] == ref['file'] and s['char_start'] <= ref['char_start'] < s['char_end'] for s in checked):
                for key in reference.get('keys', []):
                    for label in self._labels.get(key, []):
                        target = label['source']
                        links.append({'reference_id': reference['id'], 'label_key': key,
                                      'use_span': ref, 'label_span': target,
                                      'target_event_ids': [event['id'] for event in self.structure_data.get('environments', [])
                                                          if event.get('full_source') and event['full_source']['file'] == target['file']
                                                          and event['full_source']['char_start'] <= target['char_start'] < event['full_source']['char_end']],
                                      'distance': self.distance(ref, target), 'method': 'explicit_label_candidate'})
        start, count = _int(offset, 'offset'), min(_int(limit, 'limit', 1), 100)
        result = {'explicit_total': len(links), 'explicit': links[start:start + count],
                  'next_offset': start + count if start + count < len(links) else None,
                  'notice': 'Navigation/comparison/negation references require separate judgment.'}
        if query:
            result['retrieval'] = self.search(query, offset=offset, limit=min(limit, 20))
        return result

    def distance(self, left: dict, right: dict) -> dict:
        def positions(span):
            return [item['reading_char_start'] + span['char_start'] - item['source']['char_start']
                    for item in self.source_map if item['source']['file'] == span['file']
                    and item['source']['char_start'] <= span['char_start'] < item['source']['char_end']
                    and (span.get('occurrence') is None or item['occurrence'] == span['occurrence'])]
        a, b = positions(left), positions(right)
        signed = min((y - x for x in a for y in b), key=abs) if a and b else None
        return {'cross_file': left['file'] != right['file'], 'reading_char_distance': abs(signed) if signed is not None else None,
                'referenced_target_is_forward': signed > 0 if signed is not None else None,
                'same_file_char_distance': abs(left['char_start'] - right['char_start']) if left['file'] == right['file'] else None}

    def verify_graph(self, graph: Graph) -> dict:
        report = validate(graph, self.corpus)
        try:
            self.assert_unchanged()
            if graph.documents != (self.document,):
                raise SourceError('graph must contain exactly the selected pinned document')
            for record in graph.records():
                for source in getattr(record, 'sources', ()):
                    if source.document != self.document.id:
                        raise SourceError('source belongs to a different document')
                    self.normalize_span(source.model_dump(exclude={'document'}, exclude_none=True))
                if hasattr(record, 'original_text'):
                    for source, original in zip(record.sources, record.original_text):
                        if self.resolve_span(source.model_dump(exclude={'document'}, exclude_none=True)) != original:
                            raise SourceError('original_text differs from pinned original source')
        except (SourceError, ValueError, OSError) as exc:
            report.errors.append({'record': 'project', 'reason': str(exc), 'category': 'source'})
            report.source_validation = 'failed_or_incomplete'
        else:
            report.source_validation = 'verified' if not any(e.get('category') == 'source' for e in report.errors) else 'failed_or_incomplete'
        return report.to_dict()
