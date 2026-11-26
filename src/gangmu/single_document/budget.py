"""Durable invocation and observed-token budgets, shared across documents.

The CLI does not expose a hard per-request token ceiling. The observed-token
limit prevents subsequent starts; an in-flight request can overshoot it and
failed/unreported usage is explicitly unknown. Start/time/tool bounds are hard.
"""
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time
import uuid

from .cache import _atomic_json


def process_identity(pid: int) -> dict | None:
    try:
        raw = Path(f"/proc/{pid}/stat").read_text()
        fields = raw[raw.rfind(")") + 2:].split()
        if fields[0] == "Z":
            return None
        return {"pid": pid, "start_ticks": fields[19]}
    except (OSError, IndexError):
        return None


def process_alive(identity: dict | None) -> bool:
    return bool(identity and process_identity(identity["pid"]) == identity)


class InvocationLedger:
    def __init__(self, path: Path, max_invocations: int, max_observed_tokens: int):
        self.path = path
        self.max_invocations = max_invocations
        self.max_observed_tokens = max_observed_tokens
        if min(max_invocations, max_observed_tokens) <= 0:
            raise ValueError("Run invocation/token limits must be positive")
        path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _locked(self):
        descriptor = os.open(self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            if self.path.is_symlink():
                raise ValueError("Invocation ledger must not be a symbolic link")
            value = json.loads(self.path.read_bytes()) if self.path.exists() else {
                "version": "gangmu.e4.invocation_budget.v1", "invocations": []}
            yield value
            _atomic_json(self.path, value)

    def _check_start(self, value: dict, public: dict) -> None:
        """Check start limits under the ledger lock; host campaigns may extend."""
        tokens = sum(int(row.get("observed_tokens", 0)) for row in value["invocations"])
        if len(value["invocations"]) >= self.max_invocations or tokens >= self.max_observed_tokens:
            raise RuntimeError("Configured run invocation/observed-token budget exhausted")

    def start(self, public: dict) -> str:
        with self._locked() as value:
            self._check_start(value, public)
            identifier = uuid.uuid4().hex
            value["invocations"].append({"id": identifier, "state": "started",
                "started_at": time.time(), "owner": process_identity(os.getpid()), **public})
            value["limits"] = {"max_invocations": self.max_invocations,
                               "max_observed_tokens": self.max_observed_tokens,
                               "token_ceiling": "observed_after_response_not_hard_per_request"}
            return identifier

    def finish(self, identifier: str, result: dict) -> None:
        with self._locked() as value:
            row = next(row for row in value["invocations"] if row["id"] == identifier)
            usage = result.get("usage", [])
            row.update(state="returned", finished_at=time.time(),
                       usage_known=bool(usage) and result.get("usage_known", True) is True,
                       usage=usage,
                       observed_tokens=sum(int(item.get("input_tokens", 0)) + int(item.get("output_tokens", 0))
                                           for item in usage), elapsed_seconds=result["elapsed_seconds"],
                       error=result.get("error"), returncode=result.get("returncode"))

    def snapshot(self) -> dict:
        with self._locked() as value:
            rows = value["invocations"]
            return {"invocations_started": len(rows), "observed_tokens": sum(r.get("observed_tokens", 0) for r in rows),
                    "cached_input_tokens": sum(u.get("cached_input_tokens", 0) for r in rows for u in r.get("usage", [])),
                    "uncached_input_tokens": sum(max(0, u.get("input_tokens", 0) - u.get("cached_input_tokens", 0))
                                                 for r in rows for u in r.get("usage", [])),
                    "output_tokens": sum(u.get("output_tokens", 0) for r in rows for u in r.get("usage", [])),
                    "usage_unknown_invocations": sum(not r.get("usage_known", False) for r in rows),
                    "elapsed_invocation_seconds": sum(r.get("elapsed_seconds", 0) for r in rows),
                    "limits": value.get("limits", {})}
