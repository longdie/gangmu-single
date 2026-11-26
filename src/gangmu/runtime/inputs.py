"""Register LaTeX or Markdown as immutable, source-preserving corpus snapshots.

Registration performs the existing static preprocessing, never compilation or
mathematical acceptance. A source directory is copied in full and never edited.
Use one snapshot per document for appendable knowledge bases: the E1 source
identity binds the entire registered corpus, including every member document.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import shutil
import stat
import tempfile
from typing import Iterable

import pylatexenc

from gangmu import latex_structure, markdown_structure
from gangmu.representation.model import digest
from gangmu.representation.sources import E1Corpus, SourceError, _file, _json, _relative

MAX_FILE_BYTES = 30_000_000
MAX_TOTAL_BYTES = 268_435_456
MAX_FILES = 10_000


@dataclass(frozen=True)
class UserSource:
    source_dir: Path | str
    paper_id: str
    version: str
    main_file: str = 'main.tex'
    title: str = ''
    source_format: str = 'latex'


@dataclass(frozen=True)
class RegisteredCorpus:
    artifact_dir: Path
    fingerprint: str
    paper_ids: tuple[str, ...]
    reused: bool

    def corpus(self) -> E1Corpus:
        return E1Corpus(self.artifact_dir)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n',
                    encoding='utf-8')


def _inventory(root: Path) -> list[dict]:
    """List all supplied files; no guessed include closure or hidden exclusions."""
    records = []
    total = 0
    for path in sorted(root.rglob('*')):
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            raise SourceError('symlinks are not accepted in user source directories')
        if stat.S_ISDIR(mode):
            continue
        if not stat.S_ISREG(mode):
            raise SourceError('special files are not accepted in user source directories')
        relative = path.relative_to(root).as_posix()
        _relative(relative)
        size = path.stat().st_size
        total += size
        if size > MAX_FILE_BYTES or total > MAX_TOTAL_BYTES or len(records) >= MAX_FILES:
            raise SourceError('user source exceeds the file count or byte limits')
        # Compare metadata after reading, then re-inventory after copying. This
        # catches ordinary source edits rather than mixing two input versions.
        raw = _file(root, relative).read_bytes()
        if len(raw) != size:
            raise SourceError('source changed while preparing its inventory')
        records.append({'path': relative, 'size_bytes': len(raw),
                        'sha256': _sha(raw), 'suffix': path.suffix.lower()})
    if not records:
        raise SourceError('user source directory contains no files')
    return records


def _producer() -> dict:
    return {'registration_code_sha256': _sha(Path(__file__).read_bytes()),
            'preprocessing_code_sha256': _sha(Path(latex_structure.__file__).read_bytes()),
            'markdown_preprocessing_code_sha256': _sha(Path(markdown_structure.__file__).read_bytes()),
            'pylatexenc_version': pylatexenc.__version__}


def _validate(source: UserSource, output_root: Path) -> tuple[Path, dict]:
    if not isinstance(source, UserSource):
        raise SourceError('sources must be UserSource records')
    if not isinstance(source.paper_id, str) or '/' in source.paper_id or '\\' in source.paper_id:
        raise SourceError('paper_id must be one nonempty directory name')
    _relative(source.paper_id)
    if not isinstance(source.version, str) or not source.version.strip():
        raise SourceError('an explicit nonempty document version is required')
    if any(ord(c) < 32 for c in source.version + source.paper_id):
        raise SourceError('document identifiers must not contain control characters')
    if not isinstance(source.title, str):
        raise SourceError('document title must be a string')
    _relative(source.main_file)
    root = Path(source.source_dir).expanduser().resolve()
    if not root.is_dir():
        raise SourceError('user source directory does not exist')
    if output_root.is_relative_to(root) or root.is_relative_to(output_root):
        raise SourceError('source and registered artifact directories must not overlap')
    inventory = _inventory(root)
    files = {item['path']: item['sha256'] for item in inventory}
    if source.main_file not in files:
        raise SourceError('main file is absent from the user source inventory')
    suffixes = {'latex': {'.tex', '.ltx', ''}, 'markdown': markdown_structure.SUFFIXES}
    if source.source_format not in suffixes or Path(source.main_file).suffix.lower() not in suffixes[source.source_format]:
        raise SourceError('main file suffix must match the explicit source format')
    return root, {'paper_id': source.paper_id, 'arxiv_id': source.version,
                  'main_file': source.main_file, 'title': source.title,
                  'source_files': files, 'source_tree_sha256': digest(files),
                  'inventory': inventory, 'source_format': source.source_format}


@contextmanager
def _registration_lock(parent: Path):
    lock = parent / '.register.lock'
    descriptor = os.open(lock, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(descriptor, fcntl.LOCK_UN)
        os.close(descriptor)


def _safe_output_root(project_root: Path | str, namespace=None) -> Path:
    project = Path(project_root).resolve()
    if not project.is_dir():
        raise SourceError('project root must be an existing directory')
    current = project
    from gangmu.runtime.layout import identifier
    names = ('artifacts', identifier(namespace), 'preprocessed', 'user') if namespace else ('artifacts', 'preprocessed', 'user')
    for name in names:
        current = current / name
        if current.is_symlink():
            raise SourceError('artifact output directory must not contain symlinks')
    return current


def register_source(project_root: Path | str, source: UserSource, *, namespace=None) -> RegisteredCorpus:
    """Register one document; this is the stable choice for incremental intake."""
    return register_sources(project_root, [source], namespace=namespace)


def register_sources(project_root: Path | str,
                     sources: Iterable[UserSource], *, namespace=None) -> RegisteredCorpus:
    """Register an explicit corpus of any positive size without paid services.

    Repeating identical bytes and producer returns the verified existing
    snapshot. Changing bytes or the main file under the same paper/version is
    rejected: callers must assign a new version explicitly. Source paths are
    excluded from identity, so relocating identical input is harmless.
    """
    parent = _safe_output_root(project_root, namespace)
    prepared = [_validate(source, parent) for source in sources]
    if not prepared:
        raise SourceError('at least one user source is required')
    prepared.sort(key=lambda item: item[1]['paper_id'])
    papers = [item[1] for item in prepared]
    if len({paper['paper_id'] for paper in papers}) != len(papers):
        raise SourceError('duplicate paper_id in the supplied corpus')
    identity = {'schema_version': 'gangmu.user-source-registration.v1',
                'producer': _producer(), 'papers': papers}
    fingerprint = digest(identity)
    parent.mkdir(parents=True, exist_ok=True)
    destination = parent / fingerprint
    index_path = parent / 'identity-index.json'
    with _registration_lock(parent):
        if index_path.is_symlink():
            raise SourceError('source identity index must not be a symlink')
        index = _json(index_path.read_bytes()) if index_path.exists() else {}
        if not isinstance(index, dict):
            raise SourceError('invalid source identity index')
        bindings = {}
        for paper in papers:
            key = digest([paper['paper_id'], paper['arxiv_id']] +
                         (['markdown'] if paper['source_format'] == 'markdown' else []))
            binding = {'paper_id': paper['paper_id'], 'version': paper['arxiv_id'],
                       'main_file': paper['main_file'],
                       'source_tree_sha256': paper['source_tree_sha256']}
            if key not in index:
                # A previous process may have published its immutable snapshot
                # immediately before interruption prevented the index write.
                for record in parent.glob('*/registration.json'):
                    if record.is_symlink() or record.parent.is_symlink():
                        raise SourceError('registered snapshot must not contain symlinks')
                    registered = _json(record.read_bytes())
                    for existing in registered.get('papers', []):
                        if ((existing['paper_id'], existing['arxiv_id'], existing.get('source_format', 'latex')) ==
                                (paper['paper_id'], paper['arxiv_id'], paper['source_format'])):
                            E1Corpus(record.parent).verify()
                            recovered = {'paper_id': existing['paper_id'], 'version': existing['arxiv_id'],
                                         'main_file': existing['main_file'],
                                         'source_tree_sha256': existing['source_tree_sha256']}
                            if recovered != binding:
                                raise SourceError('paper/version already binds different source bytes or main file; '
                                                  'choose a new document version')
            if key in index and index[key] != binding:
                raise SourceError('paper/version already binds different source bytes or main file; '
                                  'choose a new document version')
            bindings[key] = binding
        if destination.exists() or destination.is_symlink():
            if destination.is_symlink():
                raise SourceError('registered snapshot must not be a symlink')
            corpus = E1Corpus(destination)
            corpus.verify()
            if corpus.read('registration.json') != _json_bytes(identity):
                raise SourceError('existing registration differs from its content identity')
            for original, paper in prepared:
                if _inventory(original) != paper['inventory']:
                    raise SourceError('source changed while its snapshot was being verified')
            reused = True
        else:
            temporary = Path(tempfile.mkdtemp(prefix='.register-', dir=parent))
            try:
                for original, paper in prepared:
                    _create_paper(temporary, original, paper)
                for original, paper in prepared:
                    if _inventory(original) != paper['inventory']:
                        raise SourceError('source changed while its snapshot was being copied')
                _write_corpus(temporary, identity, fingerprint)
                E1Corpus(temporary).verify()
                temporary.rename(destination)
                reused = False
            finally:
                if temporary.exists():
                    shutil.rmtree(temporary)
        # Atomic replacement also repairs a missing index entry after an
        # interruption between snapshot publication and index publication.
        updated = {**index, **bindings}
        if updated != index:
            with tempfile.NamedTemporaryFile(dir=parent, prefix='.identity-', delete=False) as handle:
                staged_index = Path(handle.name)
            try:
                _write_json(staged_index, updated)
                staged_index.replace(index_path)
            finally:
                staged_index.unlink(missing_ok=True)
    return RegisteredCorpus(destination, fingerprint, tuple(p['paper_id'] for p in papers), reused)


def _json_bytes(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n').encode('utf-8')


def _create_paper(temporary: Path, original: Path, paper: dict) -> None:
    base = temporary / paper['paper_id']
    source_root = base / 'source'
    source_root.mkdir(parents=True)
    for item in paper['inventory']:
        target = source_root / item['path']
        target.parent.mkdir(parents=True, exist_ok=True)
        original_file = _file(original, item['path'])
        raw = original_file.read_bytes()
        if _sha(raw) != item['sha256']:
            raise SourceError('source changed while copying')
        target.write_bytes(raw)
    _write_json(base / 'inventory.json', {
        'schema_version': 1, 'paper_id': paper['paper_id'], 'arxiv_id': paper['arxiv_id'],
        'source_format': 'local_directory', 'document_format': paper['source_format'],
        'source_tree_sha256': paper['source_tree_sha256'],
        'file_count': len(paper['inventory']),
        'total_bytes': sum(item['size_bytes'] for item in paper['inventory']),
        'main_file': paper['main_file'], 'files': paper['inventory']})
    processor = markdown_structure if paper['source_format'] == 'markdown' else latex_structure
    quality = processor.preprocess(source_root, paper['main_file'], base / 'preprocess',
        paper_id=paper['paper_id'], arxiv_id=paper['arxiv_id'])
    if quality['source_roundtrip_failures']:
        raise SourceError('preprocessing could not preserve original source spans')
    # SourceProject resolves registered files beneath the corpus, so never
    # retain the transient absolute staging directory as the source locator.
    structure_path = base / 'preprocess' / 'structure.json'
    structure = json.loads(structure_path.read_text(encoding='utf-8'))
    structure['source_root'] = f"{paper['paper_id']}/source"
    _write_json(structure_path, structure)


def _write_corpus(temporary: Path, identity: dict, fingerprint: str) -> None:
    table = io.StringIO(newline='')
    writer = csv.DictWriter(table, fieldnames=['paper_id', 'arxiv_id', 'main_file', 'title'],
                            lineterminator='\n')
    writer.writeheader()
    writer.writerows({key: paper[key] for key in writer.fieldnames} for paper in identity['papers'])
    (temporary / 'corpus.csv').write_text(table.getvalue(), encoding='utf-8', newline='')
    _write_json(temporary / 'registration.json', identity)
    _write_json(temporary / 'preprocessing.json', {
        'schema_version': 'gangmu.user-source-preprocessing.v1',
        'scope': 'source_preprocessing_only', 'compilation_performed': False,
        'semantic_extraction_performed': False, 'ocr_performed': False,
        'papers': [{key: paper[key] for key in ('paper_id', 'arxiv_id', 'main_file', 'source_tree_sha256', 'source_format')}
                   for paper in identity['papers']],
        'limits': ['Static preprocessing preserves unresolved TeX and source diagnostics.',
                   'No compilation, PDF review, mathematical acceptance, or dependency judgment is performed.']})
    files = {path.relative_to(temporary).as_posix(): _sha(path.read_bytes())
             for path in sorted(temporary.rglob('*')) if path.is_file()}
    _write_json(temporary / 'artifact.json', {
        'schema_version': 'e1-preprocessed-artifact-v1',
        'version': 'user-source-' + fingerprint,
        'registration_sha256': fingerprint,
        'files': files, 'tree_sha256': _sha(json.dumps(files, sort_keys=True).encode()),
        'source_resolution': 'Original files resolve under <paper_id>/source within this snapshot.',
        'limits': 'Source registration and static preprocessing only; not a compilation or semantic certificate.'})
