"""Immutable six-file MathIR snapshots with a checked manifest.

Each collection is an ordinary JSON array.  The manifest checks storage
integrity, not mathematical truth or authenticity.  Loading is deliberately
in-memory; this format does not claim streaming or database performance.
"""
from __future__ import annotations

import ctypes
import errno
import hashlib
import json
import math
import os
import re
import shutil
import stat
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import Graph


COLLECTIONS = ('documents', 'contexts', 'units', 'relations', 'evidence', 'identity_views')
SCHEMA_VERSION = 'gangmu.mathir.v4'
STORAGE_VERSION = 'gangmu.mathir.bundle.v1'
_FILES = {name: f'{name}.json' for name in COLLECTIONS}
_MANIFEST_FIELDS = {'storage_version', 'schema_version', 'snapshot_id', 'files', 'graph_sha256'}


def _fail(message: str):
    # graph.py may expose these functions too; keep runtime imports acyclic.
    from .graph import GraphError
    raise GraphError(message)


def _checked_path(directory: str | Path) -> Path:
    path = Path(directory)
    if '..' in path.parts:
        _fail('bundle path must not contain parent traversal')
    path = path.absolute()
    for component in (*reversed(path.parents), path):
        if component.is_symlink():
            _fail(f'symlink in bundle path: {component}')
    return path


def _hash(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _is_hash(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r'[0-9a-f]{64}', value) is not None


def _parse(content: bytes, name: str):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                _fail(f'{name}: duplicate JSON key: {key}')
            result[key] = value
        return result

    def constant(value):
        _fail(f'{name}: nonfinite JSON number: {value}')

    def number(value):
        result = float(value)
        if not math.isfinite(result):
            constant(value)
        return result

    try:
        return json.loads(content.decode('utf-8'), object_pairs_hook=pairs,
                          parse_constant=constant, parse_float=number)
    except (UnicodeError, json.JSONDecodeError) as exc:
        _fail(f'{name}: invalid UTF-8 JSON: {exc}')


def _encoded(value: object, *, canonical: bool = False) -> bytes:
    options = {'sort_keys': True, 'separators': (',', ':')} if canonical else {'indent': 2}
    return (json.dumps(value, ensure_ascii=False, allow_nan=False, **options)
            + ('' if canonical else '\n')).encode('utf-8')


def _manifest(data: object) -> dict:
    if not isinstance(data, dict) or set(data) != _MANIFEST_FIELDS:
        _fail('manifest.json must have exactly the documented manifest fields')
    if data['storage_version'] != STORAGE_VERSION:
        _fail(f'unsupported bundle storage_version: {data["storage_version"]!r}')
    if data['schema_version'] != SCHEMA_VERSION:
        _fail(f'unsupported bundle schema_version: {data["schema_version"]!r}')
    if not isinstance(data['snapshot_id'], str) or not data['snapshot_id']:
        _fail('manifest.json needs a nonempty snapshot_id')
    if not _is_hash(data['graph_sha256']):
        _fail('manifest.json has an invalid graph_sha256')
    files = data['files']
    if not isinstance(files, dict) or set(files) != set(COLLECTIONS):
        _fail('manifest.json must list exactly the six fixed collections')
    for collection, filename in _FILES.items():
        entry = files[collection]
        if not isinstance(entry, dict) or set(entry) != {'file', 'sha256', 'count'}:
            _fail(f'manifest.json: invalid metadata fields for {collection}')
        if entry['file'] != filename:
            _fail(f'manifest.json: {collection} must use fixed file {filename}')
        if not _is_hash(entry['sha256']):
            _fail(f'manifest.json: invalid SHA-256 for {collection}')
        if type(entry['count']) is not int or entry['count'] < 0:
            _fail(f'manifest.json: invalid record count for {collection}')
    return data


def _read_file(directory_fd: int, name: str) -> bytes:
    # O_NONBLOCK also prevents a substituted FIFO from hanging the reader.
    flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
    fd = os.open(name, flags, dir_fd=directory_fd)
    try:
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            _fail(f'bundle entry is not a regular file: {name}')
        with os.fdopen(fd, 'rb', closefd=False) as stream:
            return stream.read()
    finally:
        os.close(fd)


def load_bundle(directory: str | Path) -> Graph:
    """Read and validate one complete v4 snapshot; never change input files."""
    from .graph import GraphError, validate
    from .model import Graph

    try:
        root = _checked_path(directory)
        directory_fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            expected = set(_FILES.values()) | {'manifest.json'}
            actual = set(os.listdir(directory_fd))
            if actual != expected:
                _fail(f'bundle file set mismatch: missing={sorted(expected - actual)}, '
                      f'unexpected={sorted(actual - expected)}')
            for name in expected:
                if stat.S_ISLNK(os.stat(name, dir_fd=directory_fd, follow_symlinks=False).st_mode):
                    _fail(f'symlink in bundle: {name}')
            manifest = _manifest(_parse(_read_file(directory_fd, 'manifest.json'), 'manifest.json'))
            data = {'schema_version': manifest['schema_version'], 'snapshot_id': manifest['snapshot_id']}
            for collection, filename in _FILES.items():
                content = _read_file(directory_fd, filename)
                entry = manifest['files'][collection]
                if _hash(content) != entry['sha256']:
                    _fail(f'bundle byte digest mismatch: {filename}')
                records = _parse(content, filename)
                if not isinstance(records, list) or any(not isinstance(record, dict) for record in records):
                    _fail(f'{filename} must contain a JSON array of records')
                if len(records) != entry['count']:
                    _fail(f'bundle record count mismatch: {filename}')
                data[collection] = records
        finally:
            os.close(directory_fd)
        if _hash(_encoded(data, canonical=True)) != manifest['graph_sha256']:
            _fail('bundle canonical graph digest mismatch')
        graph = Graph.model_validate_json(_encoded(data))
        report = validate(graph)
        if report.errors:
            _fail(f'invalid bundle graph: {json.dumps(report.errors, ensure_ascii=False)}')
        return graph
    except GraphError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise GraphError(f'cannot load bundle: {exc}') from exc


def _publish(staging: Path, target: Path) -> None:
    """Linux atomic directory rename that also rejects a concurrent target."""
    # Checking exists() followed by os.rename() can overwrite an empty directory
    # created between the two calls. RENAME_NOREPLACE closes that race.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = getattr(libc, 'renameat2', None)
    if rename is None:
        _fail('atomic bundle publication requires Linux renameat2(RENAME_NOREPLACE)')
    rename.argtypes = (ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint)
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(staging), -100, os.fsencode(target), 1) != 0:
        code = ctypes.get_errno()
        if code == errno.EEXIST:
            _fail(f'bundle directory already exists; use a new version directory: {target}')
        if code in (errno.ENOSYS, errno.EINVAL, errno.EOPNOTSUPP):
            _fail('filesystem does not support atomic bundle publication with '
                  'renameat2(RENAME_NOREPLACE)')
        raise OSError(code, os.strerror(code), os.fspath(target))


def save_bundle(graph: Graph, directory: str | Path) -> None:
    """Publish six arrays and a manifest atomically, refusing any existing path.

    A sibling staging directory is removed on failure. No existing snapshot is
    overwritten; callers select a fresh version directory for each publication.
    """
    from .graph import GraphError, validate

    staging = None
    try:
        target = _checked_path(directory)
        if target.exists():
            _fail(f'bundle directory already exists; use a new version directory: {target}')
        # Pydantic can serialize bypassed NaN/Infinity values as null. Inspect
        # its JSON-mode data with strict JSON encoding before that lossy step.
        data = graph.model_dump(mode='json')
        canonical = _encoded(data, canonical=True)
        report = validate(graph)
        if report.errors:
            _fail(f'invalid bundle graph: {json.dumps(report.errors, ensure_ascii=False)}')
        if set(data) != {'schema_version', 'snapshot_id', *COLLECTIONS}:
            _fail('graph fields do not match the six-collection bundle schema')
        if data['schema_version'] != SCHEMA_VERSION:
            _fail(f'unsupported bundle schema_version: {data["schema_version"]!r}')
        manifest = {
            'storage_version': STORAGE_VERSION,
            'schema_version': data['schema_version'],
            'snapshot_id': data['snapshot_id'],
            'graph_sha256': _hash(canonical),
            'files': {},
        }
        target.parent.mkdir(parents=True, exist_ok=True)
        _checked_path(target)  # Recheck newly created parent components.
        staging = Path(tempfile.mkdtemp(prefix=f'.{target.name}.', dir=target.parent))
        for collection, filename in _FILES.items():
            content = _encoded(data[collection])
            with (staging / filename).open('xb') as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            manifest['files'][collection] = {
                'file': filename, 'sha256': _hash(content), 'count': len(data[collection]),
            }
        with (staging / 'manifest.json').open('xb') as stream:
            stream.write(_encoded(manifest))
            stream.flush()
            os.fsync(stream.fileno())
        # Verify the actual staged bytes and cross-record references before
        # making this version visible at its permanent directory name.
        load_bundle(staging)
        directory_fd = os.open(staging, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
        _checked_path(target)
        _publish(staging, target)
        staging = None
    except GraphError:
        raise
    except (OSError, ValueError, TypeError) as exc:
        raise GraphError(f'cannot save bundle: {exc}') from exc
    finally:
        if staging is not None:
            shutil.rmtree(staging)
