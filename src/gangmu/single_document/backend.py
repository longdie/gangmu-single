"""Isolated, credential-redacted noninteractive Codex process adapter.

The document agent has no host shell, arbitrary file or web tool. The Codex
JavaScript tool host can compose source-bound MCP calls without direct host
file/network access. Authentication files
belong to the host adapter and are never exposed through that server.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import threading
import subprocess
import sys
import time
import uuid
from typing import Any

from gangmu.runtime.config import RuntimeConfig
from .budget import InvocationLedger, process_identity
from .cache import _atomic_json


class BackendError(RuntimeError):
    def __init__(self, message: str, result: dict[str, Any]):
        super().__init__(message)
        self.result = result


@dataclass(frozen=True)
class CodexSettings:
    provider: str
    base_url: str
    model: str
    api_key: str = field(repr=False)
    executable: str = "codex"
    effort: str = "high"
    timeout_seconds: int = 3600
    max_tool_calls: int = 250
    budget_id: str = "single-document-default"
    max_total_invocations: int = 36
    max_observed_tokens: int = 36000000

    def __post_init__(self):
        if not re.fullmatch(r'[A-Za-z0-9_.-]{1,100}', self.budget_id) or self.budget_id in ('.', '..'):
            raise ValueError("budget_id must be a short public identifier")
        if type(self.timeout_seconds) is not int or not 1 <= self.timeout_seconds <= 7200:
            raise ValueError("timeout_seconds must be an integer in 1..7200")
        if type(self.max_tool_calls) is not int or not 1 <= self.max_tool_calls <= 2000:
            raise ValueError("max_tool_calls must be an integer in 1..2000")

    @classmethod
    def from_config(cls, config: RuntimeConfig, **options: Any) -> "CodexSettings":
        route = config.resolve_route("single_document")
        if route.provider.protocol != "openai_responses":
            raise ValueError("Codex document backend requires the responses protocol")
        key = config.credentials.get(route.provider.alias)
        if not key:
            raise ValueError("Document route credential is not configured")
        return cls(provider=route.provider.alias, base_url=route.provider.base_url,
                   model=route.model.name, api_key=key, **options)

    def public(self) -> dict[str, Any]:
        return {"provider": self.provider, "model": self.model, "protocol": "openai_responses",
                "effort": self.effort, "timeout_seconds": self.timeout_seconds,
                "max_tool_calls": self.max_tool_calls}


def _toml(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@contextmanager
def _host_cancellation():
    """Own only ordinary main-thread handlers for one backend lifecycle."""
    previous = {}
    state = {'quiet': False, 'interrupted': False, 'pending': False}

    def interrupt(signum, frame):
        if state['quiet']:
            if not state['interrupted']:
                state['pending'] = True
            return
        state['interrupted'] = True
        raise KeyboardInterrupt('isolated document host cancellation')

    def quiet(*, interrupted=False):
        state['quiet'] = True
        state['interrupted'] |= interrupted

    try:
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGINT, signal.SIGTERM):
                handler = signal.getsignal(signum)
                ordinary = handler in (signal.SIG_DFL, signal.SIG_IGN) or (
                    signum == signal.SIGINT and handler is signal.default_int_handler)
                if ordinary:
                    previous[signum] = handler
                    signal.signal(signum, interrupt)
        yield quiet, tuple(previous)
    finally:
        state['quiet'] = True
        for signum, handler in previous.items():
            if signal.getsignal(signum) is interrupt:
                signal.signal(signum, handler)
        if state['pending']:
            raise KeyboardInterrupt('isolated document host cancellation after cleanup')


class CodexBackend:
    def __init__(self, project_root: Path, settings: CodexSettings):
        self.root = project_root.resolve()
        self.settings = settings

    def _redact(self, text: str) -> str:
        return text.replace(self.settings.api_key, "[REDACTED]")

    def _invocation_ledger(self):
        return InvocationLedger(self.root / "artifacts/invocation-budgets" / (self.settings.budget_id + '.json'),
                                self.settings.max_total_invocations, self.settings.max_observed_tokens)

    def _config(self, instructions: str, session: Path, worker_root: Path | None = None) -> str:
        settings = self.settings
        worker_root = worker_root or self.root
        lines = [f"model = {_toml(settings.model)}", 'model_provider = "document_provider"',
                 f"model_reasoning_effort = {_toml(settings.effort)}",
                 f"developer_instructions = {_toml(instructions)}",
                 'approval_policy = "never"', 'sandbox_mode = "read-only"',
                 'web_search = "disabled"', 'project_doc_max_bytes = 0',
                 '', '[features]']
        lines.append('code_mode_host = true')
        for name in ("shell_tool", "unified_exec", "multi_agent", "apps",
                     "browser_use", "browser_use_external", "computer_use", "image_generation",
                     "view_image", "plugins", "workspace_dependencies", "sleep_tool",
                     "skill_search", "tool_suggest"):
            lines.append(f"{name} = false")
        lines += ['', '[agents]', 'max_threads = 1', 'max_depth = 0',
                  '', '[model_providers.document_provider]', 'name = "document_provider"',
                  f"base_url = {_toml(settings.base_url.rstrip('/'))}", 'wire_api = "responses"',
                  'requires_openai_auth = true', 'request_max_retries = 1', 'stream_max_retries = 1',
                  '', '[mcp_servers.document]', f"command = {_toml(sys.executable)}",
                  f"args = {_toml(['-B', str(worker_root / 'scripts/e4_tool_launcher.py'), '--session', str(session)])}",
                  'startup_timeout_sec = 30', 'tool_timeout_sec = 120',
                  'enabled = true', 'enabled_tools = ["document"]',
                  'default_tools_approval_mode = "approve"']
        return '\n'.join(lines) + '\n'

    @staticmethod
    def _stop_and_drain(process):
        # The group can outlive its supervisor. Drain even after poll() has
        # observed an exit, and stop any descendants holding the output pipes.
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            return process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            return process.communicate()

    def _response(self, invocation, instructions, started, process, stdout, final, error, *, timed_out=False):
        events = []
        for line in stdout.splitlines():
            try:
                value = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                events.append(value)
        usage = [value['usage'] for value in events
                 if value.get('type') == 'turn.completed' and isinstance(value.get('usage'), dict)]
        completed = any(value.get('type') == 'turn.completed' for value in events)
        content = None
        if final.exists():
            try:
                content = json.loads(self._redact(final.read_text()))
            except (json.JSONDecodeError, OSError):
                error = error or 'invalid_final_output'
        complete_response = bool(process and process.returncode == 0 and completed and not timed_out)
        return {'invocation': invocation, 'backend': 'codex', 'settings': self.settings.public(),
                'elapsed_seconds': time.monotonic() - started,
                'returncode': process.returncode if process else None, 'usage': usage,
                'usage_known': bool(usage) and complete_response,
                'usage_completeness': 'complete_cli_response' if complete_response else 'observations_not_total_cost',
                'completed': complete_response, 'final': content, 'error': error,
                'instructions_sha256': hashlib.sha256(instructions.encode()).hexdigest()}

    def _retain_response(self, output_dir, ledger, ledger_id, result, stdout, stderr, *, recovering=False):
        # Recovery attempts each independent record even if one destination
        # fails, so a log-write failure cannot prevent terminal accounting.
        actions = [
            ('events.jsonl', lambda: (output_dir / 'events.jsonl').write_text(stdout)),
            ('stderr.txt', lambda: (output_dir / 'stderr.txt').write_text(stderr)),
            ('process.json', lambda: _atomic_json(output_dir / 'process.json', {'state': 'returned', 'process': None})),
        ]
        if ledger_id is not None:
            actions += [('invocation_ledger', lambda: ledger.finish(ledger_id, result)),
                        ('run_budget', lambda: result.update(run_budget=ledger.snapshot()))]
        failures = []
        for name, action in actions:
            try:
                action()
            except Exception as exc:
                if not recovering:
                    raise
                failures.append(name + ':' + type(exc).__name__)
        if failures:
            result['retention_errors'] = failures
        try:
            _atomic_json(output_dir / 'result.json', result)
        except Exception as exc:
            if not recovering:
                raise
            result.setdefault('retention_errors', []).append('result.json:' + type(exc).__name__)

    def run(self, *, instructions: str, payload: dict[str, Any], session_spec: dict[str, Any],
            output_dir: Path, worker_root: Path | None = None) -> dict[str, Any]:
        output_dir = output_dir.resolve()
        if not output_dir.is_relative_to(self.root):
            raise ValueError("Backend output must stay in this project")
        if worker_root is not None:
            worker_root = worker_root.resolve(strict=True)
            if not worker_root.is_relative_to(self.root):
                raise ValueError("Worker implementation must remain inside project")
        invocation = uuid.uuid4().hex
        private = self.root / '.work' / 'e4-codex' / invocation
        home = private / 'codex-home'
        workspace = private / 'workspace'
        spec = private / 'session.json'
        config = home / 'config.toml'
        auth = home / 'auth.json'
        final = private / 'last-message.json'
        schema = private / 'output-schema.json'
        started = time.monotonic()
        stdout = stderr = ''
        process = None
        ledger = ledger_id = None
        error = None
        timed_out = False
        result = None
        supervisor = (worker_root or self.root) / 'scripts/e4_process_supervisor.py'
        with _host_cancellation() as (quiet, owned_signals):
            try:
                output_dir.mkdir(parents=True, exist_ok=True)
                ledger = self._invocation_ledger()
                # Defer owned cancellation signals across the durable charge
                # and ID assignment, so even a prelaunch cancellation can be
                # reconciled to its exact terminal ledger row.
                old_mask = signal.pthread_sigmask(signal.SIG_BLOCK, owned_signals) if owned_signals else None
                try:
                    ledger_id = ledger.start({'provider': self.settings.provider, 'model': self.settings.model,
                                              'run': str(output_dir.relative_to(self.root))})
                finally:
                    if old_mask is not None:
                        signal.pthread_sigmask(signal.SIG_SETMASK, old_mask)
                private.mkdir(parents=True, mode=0o700)
                home.mkdir(mode=0o700)
                workspace.mkdir()
                spec.write_text(json.dumps(session_spec, ensure_ascii=False)); spec.chmod(0o600)
                config.write_text(self._config(instructions, spec, worker_root)); config.chmod(0o600)
                auth.write_text(json.dumps({'OPENAI_API_KEY': self.settings.api_key})); auth.chmod(0o600)
                schema.write_text(json.dumps({'type': 'object', 'properties': {
                    'status': {'type': 'string', 'enum': ['complete', 'incomplete']},
                    'summary': {'type': 'string'}}, 'required': ['status', 'summary'], 'additionalProperties': False}))
                env = os.environ.copy()
                for name in tuple(env):
                    if any(word in name.upper() for word in ('API_KEY', 'TOKEN', 'SECRET', 'PASSWORD', 'CREDENTIAL', 'AUTH')):
                        env.pop(name, None)
                env.update({'CODEX_HOME': str(home), 'NO_COLOR': '1', 'PYTHONDONTWRITEBYTECODE': '1'})
                command = [self.settings.executable, 'exec', '--strict-config', '--json', '--ephemeral',
                           '--skip-git-repo-check', '--ignore-rules', '--model', self.settings.model,
                           '--sandbox', 'read-only', '--output-schema', str(schema),
                           '--output-last-message', str(final), '--color', 'never', '--cd', str(workspace), '-']
                owner = process_identity(os.getpid())
                process = subprocess.Popen([sys.executable, '-B', str(supervisor),
                                           '--owner-pid', str(owner['pid']), '--owner-start', owner['start_ticks'],
                                           '--', *command], cwd=workspace, env=env, text=True,
                                           stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                           start_new_session=True)
                _atomic_json(output_dir / 'process.json', {'state': 'running', 'process': process_identity(process.pid)})
                try:
                    stdout, stderr = process.communicate(json.dumps(payload, ensure_ascii=False),
                                                         timeout=self.settings.timeout_seconds)
                except subprocess.TimeoutExpired:
                    error, timed_out = 'timeout', True
                    stdout, stderr = self._stop_and_drain(process)
                stdout, stderr = self._redact(stdout), self._redact(stderr)
                result = self._response(invocation, instructions, started, process, stdout, final, error, timed_out=timed_out)
                self._retain_response(output_dir, ledger, ledger_id, result, stdout, stderr)
            except BaseException as exc:
                quiet(interrupted=True)
                drain_error = None
                if process is not None:
                    try:
                        stdout, stderr = self._stop_and_drain(process)
                    except Exception as failure:
                        drain_error = type(failure).__name__
                stdout, stderr = self._redact(stdout), self._redact(stderr)
                error = ('process_start_failed' if process is None else 'host_io_failed') if isinstance(exc, OSError) else (
                    'host_interrupted_' + type(exc).__name__)
                result = self._response(invocation, instructions, started, process, stdout, final, error, timed_out=timed_out)
                if drain_error:
                    result['drain_error'] = drain_error
                    result['usage_known'] = False
                self._retain_response(output_dir, ledger, ledger_id, result, stdout, stderr, recovering=True)
                if not (isinstance(exc, OSError) and process is None):
                    raise
            finally:
                # A second ordinary cancellation cannot interrupt credential
                # deletion. A first cancellation here is raised after cleanup.
                quiet()
                try:
                    auth.unlink(missing_ok=True)
                finally:
                    shutil.rmtree(private, ignore_errors=True)
        if result['error'] or not process or process.returncode != 0 or not result['completed'] or not result['final']:
            raise BackendError('Codex document invocation did not complete; see redacted run record',result)
        return result
