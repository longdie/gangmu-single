"""Explicit, resumable runtime sessions for ordinary document processing.

Sessions use the existing Responses transport and completed-response verifier.
They own fresh, user-selected budgets and never import historical task grants.
Constructing a judge does not initialize a budget or contact a provider.
"""
from __future__ import annotations

import copy
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time
import uuid

from gangmu.representation.model import digest
from gangmu.runtime.config import load_config
from gangmu.runtime.layout import identifier, storage_path, storage_relative
from gangmu.runtime.admission import acquire_physical_slot
from gangmu.runtime.token_budget import TokenBudgetLimit, TokenPoolLedger, estimate_request_tokens
from gangmu.single_document.budget import process_identity
from gangmu.single_document.core_client import (
    ConcurrentLimit,
    ModelCallError,
    SourceResponsesClient,
    _output_object,
)
from gangmu.single_document.core_response_stream import validate_sse_transcript
from gangmu.single_document.implicit_client import ImplicitLedger, _json, _local

VERSION = "gangmu.runtime.session.v1"
PROFILES = {"semantic": ("high", 12000), "demands": ("medium", 8000),
            "dependency": ("high", 16000)}
_SAFE_ID = r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}"


def estimate_phase_two_tokens(body, budget):
    estimate = estimate_request_tokens(body, budget)
    estimate["base_reserved_tokens"] = estimate["reserved_tokens"]
    estimate["reserved_tokens"] = max(
        estimate["reserved_tokens"],
        7 * estimate["input_tokens_counted"] + 32 + max(20000, body["max_output_tokens"]),
    )
    estimate["reservation_guard"] = "max(base,7*counted_input+32+max(20000,max_output))"
    return estimate


def make_session_config(session_id, *, max_total_tokens, provider="yxai",
                        model="gpt-6.1-sol", max_concurrency=3,
                        response_transport="sse", timeout_seconds=600, project=None):
    """Describe one explicit spending ceiling; this function grants no calls."""
    value = {"schema_version": VERSION, "session_id": session_id,
        "campaign_id": session_id, "provider": provider, "model": model,
        "stage": "retrieval", "retrieval": {"dense_top_k": 20,
            "auxiliary_enabled": True, "reranker_enabled": False,
            "auxiliary": {"method": "bm25_math_v1", "threshold": 0.55,
                "limit": 3, "k1": 1.2, "b": 0.75}},
        "budget": {"mode": VERSION, "budget_id": "session-" + str(session_id),
            "max_total_tokens": max_total_tokens, "max_concurrency": max_concurrency,
            "input_token_encoding": "o200k_base", "input_token_multiplier": 1.25,
            "input_token_overhead": 32, "output_limit_enforced": False},
        "model_options": {"effort": "high", "max_output_tokens": 12000,
            "max_attempts_per_request": 3, "response_transport": response_transport,
            "timeout_seconds": timeout_seconds}}
    if project is not None:
        value["project"] = identifier(project)
    return validate_session_config(value)


def validate_session_config(config):
    value = copy.deepcopy(config)
    if (not isinstance(value, dict) or value.get("schema_version") != VERSION
            or not isinstance(value.get("session_id"), str)
            or not re.fullmatch(_SAFE_ID, value["session_id"])
            or value.get("campaign_id") != value["session_id"]
            or any(not isinstance(value.get(key), str) or not value[key].strip()
                   for key in ("provider", "model"))):
        raise ValueError("A safe independent runtime session identifier and model are required")
    if "project" in value:
        identifier(value["project"])
    budget, options = value.get("budget", {}), value.get("model_options", {})
    expected = {"mode", "budget_id", "max_total_tokens", "max_concurrency",
        "input_token_encoding", "input_token_multiplier", "input_token_overhead",
        "output_limit_enforced"}
    if (set(budget) != expected or budget.get("mode") != VERSION
            or budget.get("budget_id") != "session-" + value["session_id"]
            or type(budget.get("max_total_tokens")) is not int or budget["max_total_tokens"] < 1
            or type(budget.get("max_concurrency")) is not int or not 1 <= budget["max_concurrency"] <= 3
            or budget.get("input_token_encoding") != "o200k_base"
            or budget.get("input_token_multiplier") != 1.25
            or budget.get("input_token_overhead") != 32
            or budget.get("output_limit_enforced") is not False):
        raise ValueError("Session requires an explicit positive token ceiling and at most three workers")
    if (set(options) != {"effort", "max_output_tokens", "max_attempts_per_request",
                         "response_transport", "timeout_seconds"}
            or options.get("effort") not in {"low", "medium", "high", "xhigh"}
            or type(options.get("max_output_tokens")) is not int
            or not 16 <= options["max_output_tokens"] <= 65536
            or type(options.get("max_attempts_per_request")) is not int
            or options["max_attempts_per_request"] != 3
            or options.get("response_transport") not in {"json", "sse"}
            or type(options.get("timeout_seconds")) is not int
            or not 1 <= options["timeout_seconds"] <= 600):
        raise ValueError("Session requires bounded transport options and three technical attempts")
    return value


class SessionTokenLedger(TokenPoolLedger):
    """One atomic pool for extraction, demand, dependency and semantic calls."""

    def __init__(self, config, allocation="production", *, root):
        self.root, self.config = Path(root).resolve(), validate_session_config(config)
        if allocation not in ("production", "evaluation"):
            raise ValueError("Unknown runtime allocation")
        self.allocation, self.budget = allocation, copy.deepcopy(self.config["budget"])
        self.pool_id = self.budget["budget_id"]
        self.path = storage_path(self.root, "artifacts/invocation-budgets/" + self.pool_id + ".json", self.config.get("project"))
        self.authorization_path = self.path.with_suffix(".authorization.json")
        self.fresh, self.history_path = True, None
        self.request_namespace = "integration"
        self.scope = {"schema_version": VERSION, "session_id": self.config["session_id"],
            "budget_id": self.pool_id, "provider": self.config["provider"],
            "model": self.config["model"], "budget": copy.deepcopy(self.budget),
            "initialization_policy": "empty_no_history_import",
            "reservation_policy": "max(base,7*counted_input+32+max(20000,max_output))",
            "billing_ceiling_guaranteed": False}
        if self.config.get("project") is not None:
            self.scope["project"] = self.config["project"]
        self._owned, self._owner_lock = set(), threading.Lock()
        self._physical_slots = {}

    def _authorization(self):
        return {"version": VERSION + ".authorization", "scope": self.scope,
                "initialization_policy": "one_empty_pool_no_reinitialization"}

    # Explicitly bypass the historical additive-grant mixin. Its original
    # authentication remains available to original ledgers, untouched.
    read = TokenPoolLedger.read
    _locked = TokenPoolLedger._locked
    snapshot = TokenPoolLedger.snapshot

    def additive_authorization(self, *args, **kwargs):
        raise ValueError("A session ceiling is immutable; choose a separate explicit session")

    apply_additive_authorization = additive_authorization

    def _validate(self, value):
        ImplicitLedger._validate(self, value)
        if set(value) != {"scope", "history_import", "invocations"} or value.get("history_import") is not None:
            raise ValueError("Runtime sessions cannot import or rewrite historical charges")
        counts = {}
        for row in value["invocations"]:
            namespace = row.get("request_namespace")
            if (row.get("campaign_id") != self.config["session_id"]
                    or row.get("budget_id") != self.pool_id
                    or namespace not in {"topology", "integration"}
                    or not row.get("purpose", "").startswith(namespace + ":" + row["allocation"] + ":")
                    or row.get("token_estimate", {}).get("reserved_tokens") != row["reserved_tokens"]
                    or type(row.get("supplier_total_tokens", 0)) is not int
                    or row.get("supplier_total_tokens", 0) < 0
                    or type(row.get("total_usage_known", False)) is not bool
                    or row.get("total_usage_known") and row["state"] != "returned"):
                raise ValueError("Runtime charge is outside its bound session")
            public = {key: row[key] for key in ("provider", "model", "purpose", "backend", "request_sha256", "run")}
            identity, _ = self._request(public, allocation=row["allocation"], with_estimate=False)
            if identity != row["request_identity"]:
                raise ValueError("Runtime request changed after its physical start")
            counts[identity] = counts.get(identity, 0) + 1
        if any(count > 3 for count in counts.values()):
            raise ValueError("Runtime request exceeded three cumulative technical attempts")

    def _request(self, public, *, allocation=None, with_estimate=True):
        if not isinstance(public, dict) or set(public) != {"provider", "model", "purpose", "backend", "request_sha256", "run"}:
            raise ValueError("Caller cannot override session charges")
        prefix = storage_relative("runs/" + self.config["session_id"] + "/structured-call-cache", self.config.get("project")) + "/"
        run, purpose = public.get("run", ""), public.get("purpose", "")
        namespace = purpose.split(":", 1)[0]
        if (not isinstance(run, str) or not run.startswith(prefix)
                or not re.fullmatch(r"[0-9a-f]{64}", run[len(prefix):])
                or namespace not in {"topology", "integration"}
                or not purpose.startswith(namespace + ":" + (allocation or self.allocation) + ":")
                or public["provider"] != self.config["provider"] or public["model"] != self.config["model"]
                or public["backend"] != "direct_responses"):
            raise ValueError("Runtime request has foreign routing, purpose or cache identity")
        request = _json(_local(self.root, run + "/request.json").read_bytes())
        body, binding = request["body"], request["binding"]
        model = binding["model"]
        if (digest(body) != public["request_sha256"] or binding["body_sha256"] != public["request_sha256"]
                or request.get("purpose") != purpose or body.get("model") != self.config["model"]
                or model.get("provider") != self.config["provider"] or model.get("model") != self.config["model"]
                or model.get("runtime_session_id") != self.config["session_id"]
                or model.get("runtime_session_version") != VERSION
                or model.get("runtime_project") != self.config.get("project")):
            raise ValueError("Runtime actual request/model binding differs")
        wire = json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=body.get("stream") is True).encode()
        if body.get("stream") is True and hashlib.sha256(wire).hexdigest() != binding.get("request_bytes_sha256"):
            raise ValueError("Runtime SSE physical bytes differ")
        identity = digest({"body_sha256": binding["body_sha256"], "provider": model["provider"],
            "model": model["model"], "endpoint_sha256": model.get("provider_endpoint_sha256")})
        return identity, estimate_phase_two_tokens(body, self.budget) if with_estimate else None

    def start(self, public):
        identity, estimate = self._request(public)
        slot = acquire_physical_slot(self.root, self.config["provider"])
        try:
            with self._locked() as value:
                rows = value["invocations"]
                same = [row for row in rows if row["request_identity"] == identity]
                if any(row["state"] == "returned" and row.get("returncode") == 0 for row in same):
                    raise ValueError("An existing runtime answer cannot be requested again")
                if len(same) >= 3:
                    raise ModelCallError("retained_request_attempt_limit_exhausted")
                if any(row["state"] == "started" for row in same):
                    raise ConcurrentLimit("An identical runtime request is in progress")
                if sum(row["state"] == "started" for row in rows) >= self.budget["max_concurrency"]:
                    raise ConcurrentLimit("Runtime shared concurrency reached")
                reservation = estimate["reserved_tokens"]
                committed = sum(self._committed(row) for row in rows)
                if committed + reservation > self.budget["max_total_tokens"]:
                    raise TokenBudgetLimit("Session remaining tokens are below this request reservation")
                identifier = uuid.uuid4().hex
                rows.append({**copy.deepcopy(public), "id": identifier, "state": "started",
                    "started_at": time.time(), "owner": process_identity(os.getpid()),
                    "allocation": self.allocation, "campaign_id": self.config["session_id"],
                    "request_identity": identity, "reserved_tokens": reservation,
                    "token_estimate": estimate, "usage_known": False, "observed_tokens": 0,
                    "budget_id": self.pool_id, "request_namespace": public["purpose"].split(":", 1)[0]})
        except BaseException:
            slot.release()
            raise
        with self._owner_lock:
            self._owned.add(identifier)
            self._physical_slots[identifier] = slot
        return identifier

    def finish(self, identifier, result):
        # Commit the actual/unknown charge before releasing physical admission.
        # A settlement error retains the slot until corrected or process exit.
        super().finish(identifier, result)
        with self._owner_lock:
            slot = self._physical_slots.pop(identifier)
        slot.release()

    def validate_receipt(self, invocation_or_receipt, *, task_kind=None,
                         request_sha256=None, run=None, allocation=None,
                         instructions=None, payload=None, data=None):
        """Authenticate one retained response against its exact request bytes."""
        receipt = invocation_or_receipt if isinstance(invocation_or_receipt, dict) else {}
        invocation = receipt.get("invocation") if receipt else invocation_or_receipt
        run = run or receipt.get("run")
        request_sha256 = request_sha256 or receipt.get("request_sha256")
        row = TokenPoolLedger.validate_receipt(
            self,
            invocation,
            request_sha256=request_sha256,
            run=run,
            allocation=allocation,
        )
        directory = _local(self.root, row["run"])
        request = _json((directory / "request.json").read_bytes())
        result = _json((directory / "result.json").read_bytes())
        if (
            result.get("status") != "complete"
            or result.get("invocation") != invocation
            or digest({k: v for k, v in result.items() if k != "result_sha256"}) != result.get("result_sha256")
            or result.get("binding") != request.get("binding")
            or digest(request["body"]) != row["request_sha256"]
            or request.get("purpose") != row["purpose"]
        ):
            raise ValueError("Runtime result is not the intact completed bound request")
        if receipt.get("result_sha256") and receipt["result_sha256"] != result["result_sha256"]:
            raise ValueError("Supplied result digest differs")
        if task_kind and row["purpose"] != row["request_namespace"] + ":" + row["allocation"] + ":" + task_kind:
            raise ValueError("Runtime task differs from its authenticated receipt")
        body_input = request["body"].get("input", [])
        if instructions is not None and (not body_input or body_input[0].get("content") != instructions):
            raise ValueError("Runtime instructions differ from the retained request")
        if payload is not None and (len(body_input) != 2 or _json(body_input[1]["content"]) != payload):
            raise ValueError("Runtime payload differs from the retained request")
        if data is not None and digest(data) != digest(result.get("data")):
            raise ValueError("Runtime answer differs from the retained result")
        attempt = result.get("attempt")
        physical = _json((directory / ("attempt-" + str(attempt) + ".json")).read_bytes())
        response = _json((directory / ("response-" + str(attempt) + ".json")).read_bytes())
        if (
            physical.get("invocation") != invocation
            or physical.get("usage") != row.get("usage")
            or physical.get("error") is not None
            or physical.get("returncode") != 0
            or physical.get("binding") != request["binding"]
            or physical.get("response_sha256") != digest(response)
            or result.get("response_sha256") != physical.get("response_sha256")
            or _output_object(response) != result.get("data")
        ):
            raise ValueError("Runtime physical response identity differs")
        if request["body"].get("stream") is True:
            transcript = _json((directory / ("transport-" + str(attempt) + ".json")).read_bytes())
            if (
                result.get("transport_sha256") != digest(transcript)
                or physical.get("transport_sha256") != digest(transcript)
            ):
                raise ValueError("Runtime SSE transcript identity differs")
            validate_sse_transcript(transcript, response, lambda value: value)
        return {
            **copy.deepcopy(row),
            "receipt": {
                **copy.deepcopy(receipt),
                "run": row["run"],
                "invocation": invocation,
                "result_sha256": result["result_sha256"],
                "request_sha256": row["request_sha256"],
            },
        }


class SessionJudge:
    """Production E5/E7 call protocol with an independent shared session."""

    namespace = "integration"

    def __init__(self, config, *, root, profile="semantic", allocation="production",
                 runtime=None, transport=None, stage_options=None):
        self.root, self.config = Path(root).resolve(), validate_session_config(config)
        if profile not in PROFILES:
            raise ValueError("Unknown session production profile")
        self.profile, self.allocation = profile, allocation
        options = copy.deepcopy(self.config["model_options"])
        options.update(zip(("effort", "max_output_tokens"), PROFILES[profile]))
        if stage_options is not None:
            stage = copy.deepcopy(stage_options)
            if "reasoning_effort" in stage:
                if "effort" in stage and stage["effort"] != stage["reasoning_effort"]:
                    raise ValueError("Stage effort aliases disagree")
                stage["effort"] = stage.pop("reasoning_effort")
            if (stage.pop("provider", self.config["provider"]) != self.config["provider"]
                    or stage.pop("model", self.config["model"]) != self.config["model"]):
                raise ValueError("Stage options cannot change the session provider/model")
            if set(stage) - set(options):
                raise ValueError("Unknown session stage transport option")
            options.update(stage)
        self.config["model_options"] = options
        validate_session_config(self.config)
        self.stage_options = copy.deepcopy(options)
        self.ledger = SessionTokenLedger(self.config, allocation, root=self.root)
        adapted = {"schema_version": "gangmu.e4.core_campaign.v1", "campaign_id": self.config["session_id"],
            "budget": {"budget_id": self.ledger.pool_id, "max_total_invocations": 1,
                "max_observed_tokens": self.ledger.budget["max_total_tokens"],
                "max_concurrency": self.ledger.budget["max_concurrency"],
                "heldout_reserved_invocations": 0, "heldout_reserved_observed_tokens": 0},
            "evaluation": {"provider": self.config["provider"], "model": self.config["model"], **options}}
        self.runtime = runtime if runtime is not None else load_config(self.root, require_private=True)
        self.transport = transport
        self.client = SourceResponsesClient(config=adapted, root=self.root, runtime=self.runtime,
            transport=transport, retry_output_errors=True)
        self.client.ledger = self.ledger
        self.client.model.update(runtime_session_id=self.config["session_id"],
            runtime_session_version=VERSION,
            runtime_session_code_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        if self.config.get("project") is not None:
            self.client.model["runtime_project"] = self.config["project"]
        self.cache_root = storage_path(self.root, "runs/" + self.config["session_id"] + "/structured-call-cache", self.config.get("project"))
        self._receipts, self._lock = [], threading.Lock()
        self._receipt_context, self._admission_recovery = threading.local(), threading.local()

    def receipts(self):
        return copy.deepcopy(getattr(self._receipt_context, "receipts", []))

    def all_receipts(self):
        with self._lock:
            return copy.deepcopy(self._receipts)

    @contextmanager
    def admission_recovery_context(self):
        previous = getattr(self._admission_recovery, "enabled", False)
        try:
            yield lambda: setattr(self._admission_recovery, "enabled", True)
        finally:
            self._admission_recovery.enabled = previous

    def __call__(self, task_kind, instructions, payload, *, schema=None):
        if not isinstance(task_kind, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", task_kind):
            raise ValueError("A bounded runtime task name is required")
        payload = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        if self.namespace == "topology":
            schema = payload.get("response_schema") if schema is None else schema
        purpose = self.namespace + ":" + self.allocation + ":" + task_kind
        key = self._key(purpose, instructions, payload, schema)
        target = self.cache_root / key
        if (getattr(self._admission_recovery, "enabled", False)
                and not self.unstarted_request(task_kind, instructions, payload, schema=schema)):
            raise ValueError("An earlier physical request cannot be retried as an unstarted admission task")
        try:
            result = self.client.call_json(instructions, payload, output=target, purpose=purpose, schema=schema)
        finally:
            for path in sorted(target.glob("attempt-*.json")):
                self.ledger.record_retained_total(_json(path.read_bytes()))
        request = _json((target / "request.json").read_bytes())
        receipt = {"task_kind": task_kind, "allocation": self.allocation, "request_key": key,
            "run": target.relative_to(self.root).as_posix(), "cache_hit": result["request_cache_hit"],
            "invocation": result["invocation"], "result_sha256": result["result_sha256"],
            "request_sha256": request["binding"]["body_sha256"]}
        self.ledger.validate_receipt(receipt, task_kind=task_kind,
            instructions=instructions, payload=payload, data=result["data"])
        with self._lock:
            self._receipts.append(receipt)
        if not hasattr(self._receipt_context, "receipts"):
            self._receipt_context.receipts = []
        self._receipt_context.receipts.append(receipt)
        return result["data"]

    request = __call__

    def _key(self, purpose, instructions, payload, schema):
        material = {"purpose": purpose, "instructions": instructions, "payload": payload,
                    "model": self.client.model}
        if self.namespace != "topology":
            material["schema"] = schema
        return digest(material)

    def unstarted_request(self, task_kind, instructions, payload, *, schema=None):
        payload = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        purpose = self.namespace + ":" + self.allocation + ":" + task_kind
        target = self.cache_root / self._key(purpose, instructions, payload, schema)
        run = target.relative_to(self.root).as_posix()
        return not any(row["run"] == run for row in self.ledger.read()["invocations"])

    def _body(self, instructions, payload, schema=None):
        options = self.config["model_options"]
        body = {"model": self.config["model"], "store": False,
            "reasoning": {"effort": options["effort"]}, "max_output_tokens": options["max_output_tokens"],
            "input": [{"role": "developer", "content": instructions},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
        if options["response_transport"] == "sse":
            body["stream"] = True
        if schema is not None:
            body["text"] = {"format": {"type": "json_schema", "name": "source_evaluation",
                                      "schema": schema, "strict": True}}
        return body

    def estimate_tokens(self, instructions, payload, *, schema=None):
        normalized = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        return estimate_phase_two_tokens(self._body(instructions, normalized, schema), self.config["budget"])

    def fork(self, stage_options=None, allocation=None):
        return type(self)(self.config, root=self.root, profile=self.profile,
            allocation=allocation or self.allocation, runtime=self.runtime, transport=self.transport,
            stage_options=stage_options or self.stage_options)


class SessionTopologyJudge(SessionJudge):
    """TopologyJudge protocol, including its exact request-key convention."""

    namespace = "topology"

    def receipts(self):
        # E4 writes a stage's complete administrative inventory after its
        # workers join. E5 uses worker-local slicing instead (the base API).
        return self.all_receipts()
