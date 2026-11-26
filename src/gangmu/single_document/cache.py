"""Result-bound extraction reuse and recoverable single-host process locking.

The cache deduplicates model work; it is not a source of mathematical authority.
Every hit verifies its retained artifacts. E3 current support is checked separately.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import tempfile
import time
from typing import Any, Mapping

from gangmu.runtime.protocol import canonical, clone, sha

VERSION = 'gangmu.single_document.cache.v1'
_HASH = re.compile(r'[0-9a-f]{64}\Z')
_NAME = re.compile(r'[A-Za-z][A-Za-z0-9_.-]{0,127}\Z')
_SECRET_KEYS = frozenset({'api_key', 'authorization', 'access_token', 'refresh_token',
                          'password', 'secret', 'credential', 'credentials'})


def _public(value: Any) -> Any:
    """Refuse credential-shaped keys rather than silently putting them on disk."""
    if isinstance(value, Mapping):
        result = {}
        for key, item in value.items():
            if not isinstance(key, str) or key.lower() in _SECRET_KEYS:
                raise ValueError('cache metadata must contain public string keys only')
            result[key] = _public(item)
        return result
    if isinstance(value, (tuple, list)):
        return [_public(item) for item in value]
    return clone(value)


def _relative(value: str) -> str:
    if not isinstance(value, str) or '\\' in value or '\x00' in value:
        raise ValueError('fingerprint file names must be normalized relative paths')
    path = Path(value)
    if (not isinstance(value, str) or not value or path.is_absolute()
            or '..' in path.parts or value != path.as_posix()):
        raise ValueError('fingerprint file names must be normalized relative paths')
    return value


def _digests(values: Mapping[str, str]) -> dict[str, str]:
    result = {}
    for name, digest in sorted(values.items()):
        if not isinstance(digest, str) or not _HASH.fullmatch(digest):
            raise ValueError('fingerprint file values must be SHA-256 hashes')
        result[_relative(name)] = digest
    return result


@dataclass(frozen=True)
class Fingerprint:
    key: str
    manifest: dict


def stable_fingerprint(*, source_files: Mapping[str, str], preprocessing: Any,
                       code_files: Mapping[str, str], prompt_files: Mapping[str, str],
                       schema_version: str, provider: str, actual_model: str,
                       result_parameters: Mapping[str, Any],
                       agent_version: str = 'single-document-v1') -> Fingerprint:
    """Pin all result-affecting inputs, without Git HEAD, credentials or run paths.

    ``actual_model`` must come from the selected backend. When a provider exposes
    only a model alias, that limitation belongs in result_parameters/records;
    this function cannot discover a hidden vendor revision.
    """
    if not all(isinstance(item, str) and item.strip() for item in
               (schema_version, provider, actual_model, agent_version)):
        raise ValueError('schema, provider, actual model and agent version are required')
    if not source_files or not code_files or not prompt_files:
        raise ValueError('source, implementation and prompt hashes cannot be empty')
    manifest = _public({
        'cache_version': VERSION, 'agent_version': agent_version,
        'source_files': _digests(source_files), 'preprocessing': preprocessing,
        'code_files': _digests(code_files), 'prompt_files': _digests(prompt_files),
        'schema_version': schema_version, 'provider': provider,
        'actual_model': actual_model, 'result_parameters': result_parameters,
    })
    return Fingerprint(sha(manifest), manifest)


def file_digests(root: str | Path, files: list[str] | tuple[str, ...] | None = None) -> dict[str, str]:
    """Hash actual regular files, never traverse a symbolic link."""
    root = Path(root).resolve(strict=True)
    selected = [Path(name) for name in files] if files is not None else sorted(
        path.relative_to(root) for path in root.rglob('*') if path.is_file() or path.is_symlink())
    result = {}
    for relative in selected:
        name = _relative(relative.as_posix())
        path = root / relative
        if any(component.is_symlink() for component in (path, *path.parents)) or not path.is_file():
            raise ValueError('cache inputs must be regular files without symbolic links')
        result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix='.' + path.name + '-', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'wb') as stream:
            stream.write(canonical(value) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(name, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(name):
            os.unlink(name)


class CacheSession:
    """One fingerprint owner. Only access inside ExtractionCache.session()."""

    def __init__(self, cache: 'ExtractionCache', fingerprint: Fingerprint):
        self.cache, self.fingerprint = cache, fingerprint
        self.directory = cache.root / fingerprint.key
        if self.directory.is_symlink():
            raise ValueError('cache entry must not be a symbolic link')
        self.directory.mkdir(exist_ok=True)
        self.record_path = self.directory / 'record.json'
        if self.record_path.is_symlink():
            raise ValueError('cache record must not be a symbolic link')
        self.record = self._read_record()
        self.hit = False
        self.recovered = False
        self.integrity_error: str | None = None
        if self.record and self.record.get('status') == 'complete':
            try:
                self._verify_complete()
            except (OSError, ValueError, KeyError, TypeError) as exc:
                self.integrity_error = type(exc).__name__
                self._event('corrupt_artifact', {'error_type': self.integrity_error})
            else:
                self.hit = True
                self._event('cache_hit', {})
        if not self.hit:
            previous = self.record or {}
            self.recovered = previous.get('status') == 'running'
            self.record = {
                'version': VERSION, 'key': fingerprint.key, 'fingerprint': fingerprint.manifest,
                'status': 'running', 'pid': os.getpid(), 'started_at': time.time(),
                'attempt': previous.get('attempt', 0) + 1,
                'recovered_interrupted_owner': self.recovered,
                'previous_status': previous.get('status'),
                'checkpoint': previous.get('checkpoint'),
            }
            _atomic_json(self.record_path, self.record)
            self._event('attempt_started', {'attempt': self.record['attempt'],
                                          'recovered': self.recovered})

    def _read_record(self) -> dict | None:
        if not self.record_path.exists():
            return None
        try:
            value = json.loads(self.record_path.read_bytes())
            if (value.get('version') != VERSION or value.get('key') != self.fingerprint.key
                    or sha(value.get('fingerprint')) != self.fingerprint.key):
                raise ValueError('cache record fingerprint mismatch')
            return value
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            # An invalid record cannot authorize a hit. Keep its bytes for audit.
            old_hash = hashlib.sha256(self.record_path.read_bytes()).hexdigest()
            destination = self.directory / ('corrupt-record-' + old_hash + '.json')
            if not destination.exists():
                shutil.copyfile(self.record_path, destination)
            self._event('corrupt_record', {'sha256': old_hash, 'error_type': type(exc).__name__})
            return None

    def _event(self, kind: str, details: dict) -> None:
        with (self.directory / 'events.jsonl').open('ab') as stream:
            stream.write(canonical({'time': time.time(), 'kind': kind, **details}) + b'\n')
            stream.flush()
            os.fsync(stream.fileno())

    def _verify_complete(self) -> None:
        artifacts = self.record['artifacts']
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError('complete extraction cache must retain artifacts')
        for name, item in artifacts.items():
            if not _NAME.fullmatch(name) or item['kind'] not in ('file', 'directory'):
                raise ValueError('invalid cache artifact record')
            target = self.directory / _relative(item['path'])
            if item['kind'] == 'file':
                if (not target.is_file() or target.is_symlink()
                        or hashlib.sha256(target.read_bytes()).hexdigest() != item['sha256']):
                    raise ValueError('cached file missing or changed')
            elif file_digests(target) != item['files']:
                raise ValueError('cached artifact tree missing or changed')
        if sha(self.record['result']) != self.record['result_sha256']:
            raise ValueError('cached result metadata changed')

    @property
    def result(self) -> dict:
        if not self.hit:
            raise ValueError('result is available only after a complete verified cache hit')
        self._verify_complete()
        return clone(self.record['result'])

    def artifact_path(self, name: str) -> Path:
        if not self.hit:
            raise ValueError('artifact path is available only after a complete verified hit')
        self._verify_complete()
        return self.directory / self.record['artifacts'][name]['path']

    @property
    def checkpoint(self) -> dict | None:
        item = self.record.get('checkpoint')
        if item is None:
            return None
        try:
            path = self.directory / _relative(item['path'])
            if any(component.is_symlink() for component in (path, *path.parents)):
                raise ValueError('symbolic-link checkpoint')
            raw = path.read_bytes()
            if hashlib.sha256(raw).hexdigest() != item['sha256']:
                raise ValueError('checkpoint bytes changed')
            value = json.loads(raw)
            if not isinstance(value, dict):
                raise ValueError('checkpoint must contain an object')
            return value
        except (OSError, ValueError, KeyError, TypeError):
            self._event('checkpoint_corrupt', {})
            return None

    def save_checkpoint(self, value: Mapping[str, Any]) -> None:
        if self.hit or self.record['status'] != 'running':
            raise ValueError('only a running cache owner can update checkpoints')
        public = _public(value)
        digest = sha(public)
        relative = 'checkpoints/' + digest + '.json'
        path = self.directory / relative
        if not path.exists():
            _atomic_json(path, public)
        self.record['checkpoint'] = {'path': relative, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}
        _atomic_json(self.record_path, self.record)
        self._event('checkpoint_saved', {'content_sha256': digest})

    def complete(self, *, artifacts: Mapping[str, str | Path], result: Mapping[str, Any]) -> dict:
        if self.hit or self.record['status'] != 'running' or not artifacts:
            raise ValueError('only running extraction can complete with retained artifacts')
        public = _public(result)
        kept = {}
        retained = self.directory / ('artifacts-attempt-' + str(self.record['attempt']))
        retained.mkdir(exist_ok=False)
        for name, original in artifacts.items():
            if not isinstance(name, str) or not _NAME.fullmatch(name):
                raise ValueError('cache artifact names must be bounded identifiers')
            original = Path(original).absolute()
            if any(path.is_symlink() for path in (original, *original.parents)):
                raise ValueError('cannot retain symbolic-link cache artifacts')
            target = retained / name
            if original.is_dir():
                before = file_digests(original)
                shutil.copytree(original, target)
                after = file_digests(target)
                if before != after or file_digests(original) != before:
                    raise ValueError('artifact changed during cache retention')
                kept[name] = {'kind': 'directory', 'path': target.relative_to(self.directory).as_posix(), 'files': after}
            elif original.is_file():
                content = original.read_bytes()
                target.write_bytes(content)
                kept[name] = {'kind': 'file', 'path': target.relative_to(self.directory).as_posix(),
                              'sha256': hashlib.sha256(content).hexdigest()}
            else:
                raise ValueError('cache artifacts must exist as regular files or directories')
        self.record.update(status='complete', completed_at=time.time(), artifacts=kept,
                           result=public, result_sha256=sha(public))
        _atomic_json(self.record_path, self.record)
        self._verify_complete()
        self.hit = True
        self._event('completed', {'artifact_names': sorted(kept)})
        return clone(self.record)

    def fail(self, reason: str, *, details: Mapping[str, Any] | None = None) -> None:
        if self.hit or self.record['status'] != 'running':
            return
        # The caller supplies a public diagnostic code, never an exception dump.
        if not isinstance(reason, str) or not _NAME.fullmatch(reason):
            raise ValueError('cache failure must be a public bounded diagnostic code')
        self.record.update(status='failed', failed_at=time.time(), reason=reason,
                           failure_details=_public(details or {}))
        _atomic_json(self.record_path, self.record)
        self._event('failed', {'reason': reason})


class ExtractionCache:
    def __init__(self, root: str | Path):
        self.root = Path(root).absolute()
        if any(path.is_symlink() for path in (self.root, *self.root.parents)):
            raise ValueError('cache root must not contain symbolic links')
        self.root.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def session(self, fingerprint: Fingerprint, *, lock_timeout: float | None = None):
        if not isinstance(fingerprint, Fingerprint) or sha(fingerprint.manifest) != fingerprint.key:
            raise ValueError('invalid extraction fingerprint')
        lock_path = self.root / (fingerprint.key + '.lock')
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, 'a+b') as lock:
            if lock_timeout is None:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            else:
                if lock_timeout < 0:
                    raise ValueError('lock timeout must be nonnegative')
                until = time.monotonic() + lock_timeout
                while True:
                    try:
                        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                        break
                    except BlockingIOError:
                        if time.monotonic() >= until:
                            raise TimeoutError('another process owns this extraction fingerprint')
                        time.sleep(min(0.05, max(0, until - time.monotonic())))
            session = CacheSession(self, fingerprint)
            try:
                yield session
            except BaseException as exc:
                session.fail('owner_exception', details={'exception_type': type(exc).__name__})
                raise
            finally:
                if not session.hit and session.record['status'] == 'running':
                    session.fail('incomplete_attempt')
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
