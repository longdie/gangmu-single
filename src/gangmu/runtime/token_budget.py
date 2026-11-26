"""Versioned E5 token pools and the shared E4 input-production resource gate.

This is an admission budget, not a claim that a third-party tokenizer or output
limit guarantees the final invoice. Allocation names are audit labels only.
Unknown usage retains its reservation; imported historical bytes stay read-only.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid

from gangmu.representation.model import digest
from gangmu.single_document.budget import process_alive, process_identity
from gangmu.single_document.cache import _atomic_json
from gangmu.single_document.core_client import ConcurrentLimit
from gangmu.single_document.implicit_client import ImplicitLedger, _json, _local

MODE = "token_pool_v2"
VERSION = "gangmu.e5.token_pool.v2"
FRESH_MODE = "token_pool_v3"
FRESH_VERSION = "gangmu.e5.token_pool.v3"
FRESH_BUDGET_ID = "e5-semantic-reconstruction-quality-20261010"
FIDELITY_MODE = "token_pool_v4"
FIDELITY_VERSION = "gangmu.e5.token_pool.v4"
FIDELITY_BUDGET_ID = "e5-v1-input-fidelity-20261011"
POOL_MODES = (MODE, FRESH_MODE, FIDELITY_MODE)
TOTAL_CEILING = 20_000_000
SHARED_BUDGET_ID = "e5-stage1-total-20261010"
ALLOCATIONS = ("production", "evaluation")


class TokenBudgetLimit(RuntimeError):
    """One actual request cannot be safely admitted into the remaining pool."""


def validate_pool_budget(budget):
    allowed = {"mode", "budget_id", "max_total_tokens", "max_concurrency", "input_token_encoding",
               "input_token_multiplier", "input_token_overhead", "output_limit_enforced"}
    mode = budget.get("mode") if isinstance(budget, dict) else None
    if mode == MODE:
        allowed |= {"historical_ledger", "historical_ledger_sha256"}
    if not isinstance(budget, dict) or set(budget) != allowed or mode not in POOL_MODES:
        raise ValueError("Token pool requires its explicit contract and no call/allocation quotas")
    identity = {MODE: SHARED_BUDGET_ID, FRESH_MODE: FRESH_BUDGET_ID,
                FIDELITY_MODE: FIDELITY_BUDGET_ID}[mode]
    if budget.get("budget_id") != identity:
        raise ValueError("All strategies in a budget authorization must share its registered pool identity")
    if (type(budget["max_total_tokens"]) is not int
            or not 1 <= budget["max_total_tokens"] <= TOTAL_CEILING
            or type(budget["max_concurrency"]) is not int
            or not 1 <= budget["max_concurrency"] <= 3):
        raise ValueError("Token pool exceeds the E5 token or concurrency authorization")
    if mode == MODE and (not isinstance(budget["historical_ledger"], str)
            or not re.fullmatch(r"[0-9a-f]{64}", budget["historical_ledger_sha256"])):
        raise ValueError("Token pool requires a content-bound historical ledger")
    if budget["input_token_encoding"] not in ("o200k_base", "cl100k_base"):
        raise ValueError("Token pool requires a registered input tokenizer")
    multiplier = budget["input_token_multiplier"]
    if (isinstance(multiplier, bool) or not isinstance(multiplier, (int, float))
            or not math.isfinite(multiplier) or not 1 <= multiplier <= 2
            or type(budget["input_token_overhead"]) is not int
            or not 0 <= budget["input_token_overhead"] <= 256
            or type(budget["output_limit_enforced"]) is not bool):
        raise ValueError("Invalid token estimation or provider output-limit contract")
    return copy.deepcopy(budget)


def estimate_request_tokens(body, budget):
    """Count the actual serialized prompt with an explicit, disclosed tokenizer."""
    try:
        import tiktoken
    except ImportError as exc:
        raise ValueError("E5 token admission requires the configured tiktoken tokenizer") from exc
    encoder = tiktoken.get_encoding(budget["input_token_encoding"])
    cap = body.get("max_output_tokens")
    if type(cap) is not int or cap < 16:
        raise ValueError("E5 token admission requires bounded output")
    # The output cap includes reasoning tokens in the Responses contract. Cached
    # input is still input; neither caching nor optional usage details discount it.
    content = {k: v for k, v in body.items() if k not in ("stream", "store", "max_output_tokens")}
    import json
    encoded = json.dumps(content, ensure_ascii=False, allow_nan=False, sort_keys=True)
    counted = len(encoder.encode(encoded, disallowed_special=()))
    estimated_input = math.ceil(counted * budget["input_token_multiplier"]) + budget["input_token_overhead"]
    return {"input_tokens_counted": counted, "input_tokens_reserved": estimated_input,
            "output_tokens_reserved": cap, "reserved_tokens": estimated_input + cap,
            "input_token_encoding": budget["input_token_encoding"],
            "input_token_multiplier": budget["input_token_multiplier"],
            "input_token_overhead": budget["input_token_overhead"],
            "output_limit_enforced": budget["output_limit_enforced"],
            "billing_ceiling_guaranteed": False}


class TokenPoolLedger(ImplicitLedger):
    """Atomic, unpartitioned pool shared across E5 strategy campaign versions."""

    def __init__(self, config, allocation="production", *, root, request_namespace="semantic"):
        self.root = Path(root).resolve()
        self.config = copy.deepcopy(config)
        self.budget = validate_pool_budget(config["budget"])
        self.fresh = self.budget["mode"] in (FRESH_MODE, FIDELITY_MODE)
        self.request_namespace = request_namespace
        if request_namespace not in (("semantic", "topology")
                if self.budget["mode"] == FRESH_MODE else ("semantic",)):
            raise ValueError("This token-pool contract does not authorize the request namespace")
        campaign_pattern = r"e4-topology-[A-Za-z0-9_.-]+" if request_namespace == "topology" else r"e5-stage1-[A-Za-z0-9_.-]+"
        if self.budget["mode"] == FIDELITY_MODE:
            campaign_pattern = r"e5-stage1-input-fidelity-[A-Za-z0-9_.-]+"
        if not re.fullmatch(campaign_pattern, config.get("campaign_id", "")):
            raise ValueError("Token-pool campaign differs from its resource namespace")
        if allocation not in ALLOCATIONS:
            raise ValueError("Unknown E5 allocation audit label")
        self.allocation = allocation
        self.history_path = None if self.fresh else _local(self.root, self.budget["historical_ledger"])
        self.path = _local(self.root, "artifacts/invocation-budgets/" + self.budget["budget_id"] + ".json")
        self.authorization_path = self.path.with_suffix(".authorization.json")
        if self.path == self.history_path:
            raise ValueError("Token pool must not overwrite its historical ledger")
        pool_version = {MODE: VERSION, FRESH_MODE: FRESH_VERSION,
                        FIDELITY_MODE: FIDELITY_VERSION}[self.budget["mode"]]
        self.scope = {"schema_version": pool_version, "budget_id": self.budget["budget_id"],
                      "provider": self.config["provider"], "model": self.config["model"],
                      "budget": copy.deepcopy(self.budget), "billing_ceiling_guaranteed": False,
                      "reservation_policy": "explicit_tokenizer_input_estimate_plus_bounded_output"}
        if self.fresh:
            self.scope["initialization_policy"] = "empty_no_history_import"
        self._owned, self._owner_lock = set(), threading.Lock()
        # Eagerly verify history, without creating/replacing either ledger.
        if not self.fresh:
            self._history()

    def _history(self):
        if self.fresh:
            return {"invocations": []}
        _local(self.root, self.history_path)
        raw = self.history_path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != self.budget["historical_ledger_sha256"]:
            raise ValueError("E5 historical ledger bytes differ from the registered import")
        history = _json(raw)
        if not isinstance(history, dict) or not isinstance(history.get("scope"), dict):
            raise ValueError("Malformed historical E5 ledger")
        validator = object.__new__(ImplicitLedger)
        validator.config, validator.scope = self.config, history.get("scope")
        ImplicitLedger._validate(validator, history)
        scope = history["scope"]
        if (scope.get("schema_version") != "gangmu.e5.semantic_client.v1.budget"
                or scope.get("campaign_id") != "e5-stage1-20261010"
                or scope.get("provider") != self.config["provider"]
                or scope.get("model") != self.config["model"]):
            raise ValueError("Token pool history must be the authorized E5 stage-one ledger")
        if any(row["state"] == "started" for row in history["invocations"]):
            raise ValueError("Historical ledger must be quiescent before token-pool migration")
        return history

    def _validate(self, value):
        ImplicitLedger._validate(self, value)
        history = self._history()
        rows = history["invocations"]
        if self.fresh:
            if "history_import" not in value or value["history_import"] is not None:
                raise ValueError("The new quality pool cannot import historical consumption")
        elif (value.get("history_import") != {
                "path": self.history_path.relative_to(self.root).as_posix(),
                "sha256": self.budget["historical_ledger_sha256"], "supplier_starts": len(rows)}
                or value["invocations"][:len(rows)] != rows):
            raise ValueError("Imported E5 history cannot be replaced, deleted or relabeled")
        for row in value["invocations"][len(rows):]:
            namespace = row.get("request_namespace", "semantic")
            pattern = r"e4-topology-[A-Za-z0-9_.-]+" if self.fresh and namespace == "topology" else r"e5-stage1-[A-Za-z0-9_.-]+"
            if self.budget["mode"] == FIDELITY_MODE:
                pattern = r"e5-stage1-input-fidelity-[A-Za-z0-9_.-]+"
            if (not re.fullmatch(pattern, row.get("campaign_id", ""))
                    or not isinstance(row.get("token_estimate"), dict)
                    or row["token_estimate"].get("reserved_tokens") != row["reserved_tokens"]
                    or type(row.get("supplier_total_tokens", 0)) is not int
                    or row.get("supplier_total_tokens", 0) < 0
                    or type(row.get("total_usage_known", False)) is not bool
                    or row.get("total_usage_known", False) and row["state"] != "returned"):
                raise ValueError("New E5 charges require campaign and token-estimate bindings")
            namespaces = ("semantic", "topology") if self.budget["mode"] == FRESH_MODE else ("semantic",)
            if self.fresh and (row.get("budget_id") != self.budget["budget_id"]
                    or namespace not in namespaces
                    or not isinstance(row.get("purpose"), str)
                    or not row["purpose"].startswith(namespace + ":" + row["allocation"] + ":")
                    or not isinstance(row.get("run"), str)
                    or re.fullmatch("runs/dev/" + re.escape(row["campaign_id"])
                        + r"/structured-call-cache/[0-9a-f]{64}", row["run"]) is None):
                raise ValueError("New quality charges require their actual shared budget and request namespace")

    def read(self):
        """Read and validate retained accounting without initializing or rewriting it."""
        _local(self.root, self.path)
        if self.fresh:
            _local(self.root, self.authorization_path)
            if self.path.exists() != self.authorization_path.exists():
                raise ValueError("The initialized quality pool or its immutable authorization is missing; no reset is allowed")
            if self.authorization_path.exists() and _json(self.authorization_path.read_bytes()) != self._authorization():
                raise ValueError("The quality pool authorization cannot be changed or replaced")
        if self.path.exists():
            value = _json(self.path.read_bytes())
        elif self.fresh:
            value = {"scope": self.scope, "history_import": None, "invocations": []}
        else:
            history = self._history()
            value = {"scope": self.scope, "history_import": {
                "path": self.history_path.relative_to(self.root).as_posix(),
                "sha256": self.budget["historical_ledger_sha256"],
                "supplier_starts": len(history["invocations"])},
                "invocations": copy.deepcopy(history["invocations"])}
        self._validate(value)
        return value

    def _authorization(self):
        return {"version": "gangmu.e5.token_pool_authorization.v1", "scope": self.scope,
                "initialization_policy": "one_empty_pool_no_reinitialization"}

    def initialize(self):
        """Explicitly register this authorization once; existing fees never reset."""
        if not self.fresh:
            raise ValueError("Historical token pools cannot be initialized or replaced")
        with self._locked(initialize=True):
            pass
        return self.snapshot()

    def validate_receipt(self, invocation, *, request_sha256=None, run=None, allocation=None):
        """Verify one retained supplier charge read-only; never create a ledger."""
        value = self.read()
        row = next((item for item in value["invocations"] if item["id"] == invocation), None)
        if row is None or row["state"] != "returned":
            raise ValueError("Supplier evidence requires a terminal invocation in the bound pool")
        for field, expected in (("request_sha256", request_sha256), ("run", run), ("allocation", allocation)):
            if expected is not None and row.get(field) != expected:
                raise ValueError("Supplier receipt differs from its actual ledger: " + field)
        return copy.deepcopy(row)

    @contextmanager
    def _locked(self, *, initialize=False):
        if not self.fresh:
            raise ValueError("The completed cumulative E5 pool is read-only; use the new quality authorization")
        _local(self.root, self.path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        descriptor = os.open(self.path.with_suffix(".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            if not initialize and not self.path.exists():
                raise ValueError("Explicitly initialize the new quality authorization before any supplier start")
            value = self.read()
            self._validate(value)
            yield value
            self._validate(value)
            _atomic_json(self.path, value)
            if not self.authorization_path.exists():
                _atomic_json(self.authorization_path, self._authorization())

    def _request(self, public):
        expected = {"provider", "model", "purpose", "backend", "request_sha256", "run"}
        if not isinstance(public, dict) or set(public) != expected:
            raise ValueError("Caller cannot override E5 token charges")
        prefix = "runs/dev/" + self.config["campaign_id"] + "/structured-call-cache/"
        run = public.get("run")
        suffix = run[len(prefix):] if isinstance(run, str) and run.startswith(prefix) else ""
        if (not re.fullmatch(r"[0-9a-f]{64}", suffix)
                or public["provider"] != self.config["provider"]
                or public["model"] != self.config["model"]
                or public["backend"] != "direct_responses"
                or not isinstance(public["purpose"], str)
                or not public["purpose"].startswith(self.request_namespace + ":" + self.allocation + ":")):
            raise ValueError("E5 requires a registered content-bound task")
        request = _json(_local(self.root, run + "/request.json").read_bytes())
        body, binding = request["body"], request["binding"]
        if (digest(body) != public["request_sha256"]
                or binding["body_sha256"] != public["request_sha256"]
                or request.get("purpose") != public["purpose"]
                or body.get("model") != self.config["model"]):
            raise ValueError("E5 request differs from token-pool admission")
        import json
        wire = json.dumps(body, ensure_ascii=False, allow_nan=False,
                          sort_keys=body.get("stream") is True).encode()
        if (body.get("stream") is True
                and hashlib.sha256(wire).hexdigest() != binding.get("request_bytes_sha256")):
            raise ValueError("E5 physical request bytes differ")
        model = binding["model"]
        if model.get("provider") != self.config["provider"] or model.get("model") != self.config["model"]:
            raise ValueError("E5 request has a foreign provider/model binding")
        identity = digest({"body_sha256": binding["body_sha256"], "provider": model["provider"],
                           "model": model["model"], "endpoint_sha256": model.get("provider_endpoint_sha256")})
        return identity, estimate_request_tokens(body, self.budget)

    def start(self, public):
        if not self.fresh:
            raise ValueError("The completed cumulative E5 pool is read-only; no new supplier starts")
        identity, estimate = self._request(public)
        reservation = estimate["reserved_tokens"]
        with self._locked() as value:
            rows = value["invocations"]
            if sum(row["state"] == "started" for row in rows) >= self.budget["max_concurrency"]:
                raise ConcurrentLimit("E5 token pool concurrency reached; unfinished charges remain reserved")
            committed = sum(self._committed(row) for row in rows)
            if committed + reservation > self.budget["max_total_tokens"]:
                raise TokenBudgetLimit("E5 remaining token budget " + str(max(0, self.budget["max_total_tokens"] - committed))
                                       + " is below this request reservation " + str(reservation))
            identifier = uuid.uuid4().hex
            rows.append({**copy.deepcopy(public), "id": identifier, "state": "started",
                         "started_at": time.time(), "owner": process_identity(os.getpid()),
                         "allocation": self.allocation, "campaign_id": self.config["campaign_id"],
                         "request_identity": identity, "reserved_tokens": reservation,
                         "token_estimate": estimate, "usage_known": False, "observed_tokens": 0})
            if self.fresh:
                rows[-1].update(budget_id=self.budget["budget_id"], request_namespace=self.request_namespace)
        with self._owner_lock:
            self._owned.add(identifier)
        return identifier

    @staticmethod
    def _committed(row):
        actual = max(row["observed_tokens"], row.get("supplier_total_tokens", 0))
        if row["usage_known"] or row.get("total_usage_known", False):
            return actual
        return max(actual, row["reserved_tokens"])

    def record_retained_total(self, receipt):
        """Bind an actual vendor total to retained bytes, without changing them.

        Standard input/output totals already count cached and reasoning tokens.
        A reported larger total must also be charged. A terminal total alone can
        settle usage even when optional primary breakdowns were unavailable.
        """
        if not self.fresh:
            # Old response replay must not settle or otherwise rewrite old fees.
            return
        with self._locked() as value:
            history_count = 0
            row = next((row for row in value["invocations"][history_count:]
                        if row["id"] == receipt.get("invocation")), None)
            if row is None or row["state"] != "returned":
                raise ValueError("Reported supplier total must belong to a returned new invocation")
            attempt = receipt.get("attempt")
            if type(attempt) is not int or attempt < 1:
                raise ValueError("Retained supplier total needs a physical attempt identity")
            target = _local(self.root, row["run"])
            attempt_path = _local(self.root, target / ("attempt-" + str(attempt) + ".json"))
            retained = _json(attempt_path.read_bytes())
            response_path = _local(self.root, target / ("response-" + str(attempt) + ".json"))
            if not response_path.exists():
                return
            response = _json(response_path.read_bytes())
            if (retained.get("invocation") != row["id"]
                    or retained.get("usage") != row["usage"]
                    or retained.get("response_sha256") != digest(response)
                    or receipt.get("response_sha256") != retained.get("response_sha256")
                    or retained.get("binding", {}).get("body_sha256") != row["request_sha256"]):
                raise ValueError("Reported supplier total differs from retained invocation bytes")
            if not isinstance(response, dict):
                return
            usage = response.get("usage") if isinstance(response, dict) else None
            total = usage.get("total_tokens") if isinstance(usage, dict) else None
            if (response.get("status") not in ("completed", "incomplete", "failed")
                    or type(total) is not int or total < 0):
                return
            row.update(supplier_total_tokens=total, total_usage_known=True,
                       supplier_total_response_sha256=digest(response))

    def close_dead_unfinished(self, identifier):
        """Release only an orphan's concurrency slot; retain all token liability."""
        with self._locked() as value:
            history_count = 0
            row = next((row for row in value["invocations"][history_count:] if row["id"] == identifier), None)
            if row is None or row["state"] != "started" or process_alive(row.get("owner")):
                raise ValueError("Only a verified dead new E5 invocation can close as unknown")
            row.update(state="returned", finished_at=time.time(), usage=[], usage_known=False,
                       observed_tokens=0, elapsed_seconds=0, returncode=1,
                       error="dead_unfinished_usage_unknown")

    def snapshot(self):
        value = self.read()
        rows = value["invocations"]
        committed = sum(self._committed(row) for row in rows)
        historical = 0 if self.fresh else value["history_import"]["supplier_starts"]
        return {"campaign_id": self.config["campaign_id"], "budget_id": self.budget["budget_id"],
                    "supplier_starts": len(rows), "invocations_started": len(rows),
                    "historical_starts": historical, "new_supplier_starts": len(rows) - historical,
                    "observed_tokens": sum(max(row["observed_tokens"], row.get("supplier_total_tokens", 0)) for row in rows),
                    "committed_tokens": committed, "active": sum(row["state"] == "started" for row in rows),
                    "active_live": sum(row["state"] == "started" and process_alive(row.get("owner")) for row in rows),
                    "dead_unfinished": sum(row["state"] == "started" and not process_alive(row.get("owner")) for row in rows),
                    "usage_unknown_invocations": sum(row["state"] == "returned" and not row["usage_known"] and not row.get("total_usage_known", False) for row in rows),
                    "unknown_reserved_tokens": sum(row["reserved_tokens"] for row in rows if row["state"] == "returned" and not row["usage_known"] and not row.get("total_usage_known", False)),
                    "active_reserved_tokens": sum(row["reserved_tokens"] for row in rows if row["state"] == "started"),
                    "remaining_starts": None,
                    "remaining_admission_tokens": max(0, self.budget["max_total_tokens"] - committed),
                    "over_reservation_invocations": sum((row["usage_known"] or row.get("total_usage_known", False)) and self._committed(row) > row["reserved_tokens"] for row in rows),
                    "allocations": {a: {"starts": sum(row["allocation"] == a for row in rows),
                                           "committed_tokens": sum(self._committed(row) for row in rows if row["allocation"] == a)} for a in ALLOCATIONS},
                    "limits": copy.deepcopy(self.budget), "billing_ceiling_guaranteed": False}
