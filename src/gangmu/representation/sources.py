"""Read immutable E1 snapshots without interpreting TeX as mathematical knowledge.

The snapshot manifest is an integrity boundary, not a signature or a correctness
certificate. Callers pin ``expected_tree`` when consuming an accepted artifact.
All text reads preserve original line endings and explicitly retain unknown
author encodings. Historical absolute E1 paths are never followed.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
from dataclasses import asdict, dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterator


class SourceError(ValueError):
    """Invalid, modified, or unresolvable source material."""


def sha256(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _hash(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def _integer(value: Any, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or '\\' in value or '\x00' in value:
        raise SourceError('source path must be a nonempty POSIX relative path')
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in ('', '.', '..') for part in value.split('/')):
        raise SourceError('absolute or noncanonical source path')
    return path


def _file(root: Path, relative: str) -> Path:
    path = root
    for part in _relative(relative).parts:
        path = path / part
        if path.is_symlink():
            raise SourceError(f'symlink in snapshot path: {relative}')
    if not path.is_file() or not path.resolve().is_relative_to(root.resolve()):
        raise SourceError(f'missing or escaping snapshot file: {relative}')
    return path


def _json(content: bytes) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise SourceError(f'duplicate JSON key: {key}')
            result[key] = value
        return result

    try:
        def invalid_constant(value):
            raise SourceError(f'nonfinite JSON constant: {value}')
        return json.loads(content, object_pairs_hook=pairs, parse_constant=invalid_constant)
    except (UnicodeError, json.JSONDecodeError) as exc:
        raise SourceError('invalid JSON record') from exc


def _decode(content: bytes) -> tuple[str, str, str]:
    try:
        return content.decode('utf-8'), 'utf-8', 'strict_utf8'
    except UnicodeDecodeError:
        return content.decode('latin-1'), 'latin-1', 'author_encoding_unresolved_byte_transport'


def _position(text: str, offset: int) -> tuple[int, int]:
    return text.count('\n', 0, offset) + 1, offset - text.rfind('\n', 0, offset)


@dataclass(frozen=True)
class SourceAnchor:
    """Half-open byte and character span, qualified by a pinned source snapshot.

    ``occurrence`` identifies a reading occurrence, not a permanent unit identity.
    This record makes no claim about semantic interpretation or theoremhood.
    """

    paper_id: str
    arxiv_id: str
    artifact_version: str
    artifact_tree_sha256: str
    file: str
    file_sha256: str
    char_start: int
    char_end: int
    byte_start: int
    byte_end: int
    line_start: int
    column_start: int
    line_end: int
    column_end: int
    slice_sha256: str
    encoding_status: str
    occurrence: str | None = None

    def __post_init__(self):
        _relative(self.file)
        for field in ('paper_id', 'arxiv_id', 'artifact_version'):
            if not isinstance(getattr(self, field), str) or not getattr(self, field):
                raise SourceError(f'invalid {field}')
        for field in ('artifact_tree_sha256', 'file_sha256', 'slice_sha256'):
            if not _hash(getattr(self, field)):
                raise SourceError(f'invalid {field}')
        for field in ('char_start', 'char_end', 'byte_start', 'byte_end'):
            if not _integer(getattr(self, field)):
                raise SourceError(f'invalid {field}')
        for field in ('line_start', 'line_end', 'column_start', 'column_end'):
            if not _integer(getattr(self, field), 1):
                raise SourceError(f'invalid {field}')
        if self.char_end < self.char_start or self.byte_end < self.byte_start:
            raise SourceError('reversed source span')
        if (self.line_end, self.column_end) < (self.line_start, self.column_start):
            raise SourceError('reversed source position')
        if self.encoding_status not in ('strict_utf8', 'author_encoding_unresolved_byte_transport'):
            raise SourceError('unknown encoding status')
        if self.occurrence is not None and (not isinstance(self.occurrence, str) or not self.occurrence):
            raise SourceError('invalid reading occurrence')

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class E1Corpus:
    """Read and verify the existing E1 artifact; never modify or recompile it."""

    def __init__(self, root: Path | str, *, expected_tree: str | None = None):
        self.root = Path(root).resolve()
        self.manifest = _json(_file(self.root, 'artifact.json').read_bytes())
        if not isinstance(self.manifest, dict) or self.manifest.get('schema_version') != 'e1-preprocessed-artifact-v1':
            raise SourceError('unsupported E1 artifact schema')
        self.files = self.manifest.get('files')
        if not isinstance(self.files, dict) or not self.files:
            raise SourceError('missing registered file inventory')
        for path, digest in self.files.items():
            _relative(path)
            if not _hash(digest):
                raise SourceError('invalid registered file digest')
        self.tree_sha256 = sha256(json.dumps(self.files, sort_keys=True).encode())
        if self.tree_sha256 != self.manifest.get('tree_sha256'):
            raise SourceError('registered inventory tree digest mismatch')
        if expected_tree is not None and self.tree_sha256 != expected_tree:
            raise SourceError('artifact does not match the requested pinned tree')
        self.version = self.manifest.get('version')
        if not isinstance(self.version, str) or not self.version:
            raise SourceError('missing artifact version')
        self.papers: dict[str, dict[str, str]] = {}
        for row in csv.DictReader(io.StringIO(self.read('corpus.csv').decode('utf-8'))):
            pid = row.get('paper_id', '')
            if not pid or '/' in pid or '\\' in pid or pid in self.papers:
                raise SourceError('invalid or duplicate paper identity')
            _relative(pid)
            if not row.get('arxiv_id'):
                raise SourceError('missing pinned document version')
            self.papers[pid] = row
        if not self.papers:
            raise SourceError('empty corpus manifest')

    def read(self, relative: str) -> bytes:
        """Recheck a file before using its bytes, including after earlier verification."""
        if relative not in self.files:
            raise SourceError(f'unregistered snapshot file: {relative}')
        content = _file(self.root, relative).read_bytes()
        if sha256(content) != self.files[relative]:
            raise SourceError(f'snapshot file digest mismatch: {relative}')
        return content

    def record(self, paper_id: str, relative: str) -> Any:
        self._paper(paper_id)
        _relative(relative)
        return _json(self.read(f'{paper_id}/{relative}'))

    def _paper(self, paper_id: str) -> dict[str, str]:
        try:
            return self.papers[paper_id]
        except KeyError as exc:
            raise SourceError(f'unknown corpus paper: {paper_id}') from exc

    def verify(self) -> dict[str, Any]:
        actual = set()
        for path in self.root.rglob('*'):
            if path.is_symlink():
                raise SourceError(f'symlink in artifact: {path.relative_to(self.root)}')
            if path.is_file() and path != self.root / 'artifact.json':
                actual.add(path.relative_to(self.root).as_posix())
        if actual != set(self.files):
            raise SourceError('actual artifact file inventory differs from registration')
        for path in self.files:
            self.read(path)
        for pid, row in self.papers.items():
            inventory = self.record(pid, 'inventory.json')
            structure = self.record(pid, 'preprocess/structure.json')
            for record in (inventory, structure):
                if record['paper_id'] != pid or record['arxiv_id'] != row['arxiv_id']:
                    raise SourceError('paper identity mismatch between E1 records')
            if structure.get('schema_version') not in {'gangmu.latex-structure.v1', 'gangmu.markdown-structure.v1'}:
                raise SourceError('unsupported E1 structure schema')
            for item in inventory['files']:
                raw = self.read(f"{pid}/source/{item['path']}")
                if len(raw) != item['size_bytes'] or sha256(raw) != item['sha256']:
                    raise SourceError('original source inventory mismatch')
        return {'papers': len(self.papers), 'registered_files': len(self.files),
                'artifact_version': self.version, 'tree_sha256': self.tree_sha256}

    def anchor(self, paper_id: str, span: dict[str, Any], *, occurrence: str | None = None) -> SourceAnchor:
        row = self._paper(paper_id)
        if not isinstance(span, dict) or not {'file', 'char_start', 'char_end'} <= span.keys():
            raise SourceError('source span needs file and character bounds')
        raw = self.read(f"{paper_id}/source/{span['file']}")
        text, encoding, status = _decode(raw)
        start, end = span['char_start'], span['char_end']
        if not _integer(start) or not _integer(end) or not 0 <= start <= end <= len(text):
            raise SourceError('character range outside the original source')
        if occurrence is not None:
            mapping = self.record(paper_id, 'preprocess/source_map.json')
            intervals = sorted((item['source']['char_start'], item['source']['char_end'])
                               for item in mapping if item['occurrence'] == occurrence
                               and item['source']['file'] == span['file'])
            cursor = start
            found = False
            for left, right in intervals:
                if left <= cursor <= right:
                    cursor = max(cursor, right)
                    found = True
                if found and cursor >= end:
                    break
            if not found or cursor < end:
                raise SourceError('span does not belong to the named reading occurrence')
        byte_start, byte_end = len(text[:start].encode(encoding)), len(text[:end].encode(encoding))
        line_start, column_start = _position(text, start)
        line_end, column_end = _position(text, end)
        expected = {'byte_start': byte_start, 'byte_end': byte_end, 'line_start': line_start,
                    'column_start': column_start, 'line_end': line_end, 'column_end': column_end}
        for key, value in expected.items():
            if key in span and (not _integer(span[key]) or span[key] != value):
                raise SourceError(f'{key} disagrees with original bytes/characters')
        return SourceAnchor(paper_id=paper_id, arxiv_id=row['arxiv_id'], artifact_version=self.version,
                            artifact_tree_sha256=self.tree_sha256, file=span['file'], file_sha256=sha256(raw),
                            char_start=start, char_end=end, **expected,
                            slice_sha256=sha256(raw[byte_start:byte_end]), encoding_status=status,
                            occurrence=occurrence)

    def lines(self, paper_id: str, file: str, first: int, last: int) -> SourceAnchor:
        """Create an anchor from inclusive display line numbers; bytes stay authoritative."""
        raw = self.read(f'{paper_id}/source/{file}')
        text, _, _ = _decode(raw)
        starts = [0] + [i + 1 for i, char in enumerate(text) if char == '\n']
        if not _integer(first, 1) or not _integer(last, 1) or not first <= last <= len(starts):
            raise SourceError('line range outside original source')
        return self.anchor(paper_id, {'file': file, 'char_start': starts[first - 1],
                                     'char_end': starts[last] if last < len(starts) else len(text)})

    def resolve(self, anchor: SourceAnchor) -> str:
        """Resolve only this exact snapshot, including document version and slice hash."""
        checked = self.anchor(anchor.paper_id, anchor.to_dict(), occurrence=anchor.occurrence)
        if checked != anchor:
            raise SourceError('source anchor has stale or incompatible identity/hash fields')
        raw = self.read(f'{anchor.paper_id}/source/{anchor.file}')
        text, _, _ = _decode(raw)
        return text[anchor.char_start:anchor.char_end]

    def candidates(self, paper_id: str) -> Iterator[dict[str, Any]]:
        """Yield theorem-shaped structures; these are not accepted knowledge units.

        No source scopes are inferred, no macros expanded, and no mathematical
        dependence follows from citations or a theorem-like environment.
        """
        structure = self.record(paper_id, 'preprocess/structure.json')
        if structure.get('source_format') == 'markdown':
            for block in structure['blocks']:
                if block.get('formal_hint'):
                    yield {'event_id': block['id'], 'source_format': 'markdown',
                           'activity': 'active', 'scope_status': 'syntactic_only',
                           'boundary_status': 'candidate_block_not_mathematical_boundary',
                           'source': self.anchor(paper_id, block['source'], occurrence='markdown:main').to_dict(),
                           'semantic_status': 'not_extracted'}
            return
        for event in structure['environments']:
            if event['environment_kind'] != 'theorem' or event['source_role'] == 'author_package':
                continue
            span = event.get('full_source')
            yield {'event_id': event['id'], 'environment_name': event['name'],
                   'activity': event['activity'], 'condition_ids': event['condition_ids'],
                   'section_ids': event['section_ids'], 'scope_status': 'syntactic_only',
                   'boundary_status': 'complete_static_span' if span else 'unresolved_boundary',
                   'source': self.anchor(paper_id, span or event['source'],
                                         occurrence=event['file_occurrence']).to_dict(),
                   'semantic_status': 'not_extracted'}
