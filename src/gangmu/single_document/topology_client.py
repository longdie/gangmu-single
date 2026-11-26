"""Separate topology campaign; reuse the frozen reservation-aware HTTP gate.

The legacy gate accepts an e4-implicit identity. A distinct ledger_id is used
as a documented adapter, never the prior implicit campaign or its remainder.
No call happens on import, loading or snapshot. Stage forks share one ledger.
"""
from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
import re

from gangmu.representation.model import digest
from .implicit_client import ImplicitJudge, ImplicitLedger, ROOT, PUBLIC_FIELDS, _json, _local, validate_config

VERSION = "gangmu.e4.topology-client.v2"
LEGACY_VERSION = "gangmu.e4.topology-client.v1"
POLICY_VERSION = "gangmu.e4.topology-single-pass.v2"


def validate_runtime_policy(config):
    """Only the current single-pass policy may execute new finalization work.

    Archived v1 configurations remain readable through :func:`load_config`;
    they are not an implicit request to restore their old runtime score gate.
    """
    if (config.get("schema_version") != "gangmu.e4.topology_campaign.v2"
            or config.get("runtime_policy") != POLICY_VERSION
            or any(type(config.get(key)) is not int or config[key] != 1 for key in (
                "max_node_expansion_rounds", "max_document_isolation_rounds", "max_reextractions_per_document"))
            or any(config.get(key) is not False for key in (
                "repair_relation_evaluation", "runtime_precision_gate",
                "whole_output_runtime_evaluation", "formal_publication", "recall_acceptance"))
            or "v2" not in config.get("campaign_id", "")
            or "v2" not in config.get("ledger_id", "")):
        raise ValueError("Finalization requires the v2 one-pass policy without runtime precision evaluation")
    options = config.get("expanded_retrieval", {})
    if not isinstance(options, dict) or any(isinstance(value, list) for value in options.values()):
        raise ValueError("Single-pass retrieval uses one fixed option set, not a multi-round schedule")
    inheritance = config.get("budget_inheritance")
    if inheritance is not None:
        if not isinstance(inheritance, dict) or inheritance.get("policy_change_grants_new_budget") is not False:
            raise ValueError("Changing finalization policy cannot grant a new supplier budget")
        for historical, limit, authorized in (
                ("historical_counted_starts", "max_counted_starts", "combined_max_counted_starts"),
                ("historical_observed_tokens", "max_observed_tokens", "combined_max_observed_tokens")):
            prior, current, total = inheritance.get(historical), config.get("budget", {}).get(limit), inheritance.get(authorized)
            if (type(prior) is not int or prior < 0 or type(current) is not int
                    or type(total) is not int or prior + current > total
                    or total > (200 if limit == "max_counted_starts" else 10_000_000)):
                raise ValueError("Inherited topology budget exceeds the original combined authorization")
    return config


def adapted_config(config, stage_options=None):
    if config.get("schema_version") not in {"gangmu.e4.topology_campaign.v1", "gangmu.e4.topology_campaign.v2"}:
        raise ValueError("Unknown topology campaign")
    if config.get("schema_version") == "gangmu.e4.topology_campaign.v2":
        validate_runtime_policy(config)
    campaign, ledger = config.get("campaign_id", ""), config.get("ledger_id", "")
    if (not re.fullmatch(r"e4-topology-[A-Za-z0-9_.-]+", campaign)
            or not re.fullmatch(r"e4-implicit-topology-[A-Za-z0-9_.-]+", ledger)):
        raise ValueError("Topology requires a separate campaign and shared gate identity")
    options = copy.deepcopy(stage_options or config["model_options"])
    if options.get("provider", config["provider"]) != config["provider"] or options.get("model", config["model"]) != config["model"]:
        raise ValueError("Stage fork changes authorized topology provider/model")
    for key in ("provider", "model"):
        options.pop(key, None)
    result = {"schema_version": "gangmu.e4.implicit_campaign.v1", "campaign_id": ledger,
              "provider": config["provider"], "model": config["model"],
              "budget": copy.deepcopy(config["budget"]), "model_options": options}
    validate_config(result)
    return result


class TopologyLedger(ImplicitLedger):
    """Same finite accounting algorithm, topology request namespace and scope."""
    def __init__(self, config, allocation="production", *, root=ROOT):
        self.topology_config = copy.deepcopy(config)
        inherited = config.get("budget_inheritance")
        if inherited is not None:
            pointer = inherited.get("from_ledger")
            if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
                raise ValueError("Inherited supplier usage requires an immutable ledger binding")
            source = _local(Path(root).resolve(), pointer["path"])
            content = source.read_bytes()
            if hashlib.sha256(content).hexdigest() != pointer["sha256"]:
                raise ValueError("Historical supplier ledger changed after budget inheritance")
            prior = _json(content)
            rows = prior.get("invocations", [])
            if (prior.get("scope", {}).get("campaign_id") != inherited.get("from_ledger_id")
                    or len(rows) != inherited["historical_counted_starts"]
                    or sum(row.get("observed_tokens", 0) for row in rows) != inherited["historical_observed_tokens"]
                    or any(row.get("state") not in {"returned", "failed"} or not row.get("usage_known") for row in rows)):
                raise ValueError("Inherited budget must bind terminal known historical supplier usage")
        super().__init__(adapted_config(config), allocation, root=root)
        version = VERSION if config.get("schema_version") == "gangmu.e4.topology_campaign.v2" else LEGACY_VERSION
        self.scope.update(schema_version=version + ".budget",
                          topology_campaign_id=config["campaign_id"])

    def _request(self, public):
        if not isinstance(public, dict) or set(public) != PUBLIC_FIELDS:
            raise ValueError("Caller cannot override topology ownership or budget")
        prefix = "runs/dev/" + self.topology_config["campaign_id"] + "/structured-call-cache/"
        run = public.get("run")
        suffix = run[len(prefix):] if isinstance(run, str) and run.startswith(prefix) else ""
        if (not re.fullmatch(r"[0-9a-f]{64}", suffix)
                or public["provider"] != self.config["provider"] or public["model"] != self.config["model"]
                or public["backend"] != "direct_responses"
                or not public["purpose"].startswith("topology:" + self.allocation + ":")):
            raise ValueError("A registered topology task and content-bound cache are required")
        request = _json(_local(self.root, run + "/request.json").read_bytes())
        body, binding = request["body"], request["binding"]
        if (digest(body) != public["request_sha256"] or binding["body_sha256"] != public["request_sha256"]
                or request.get("purpose") != public["purpose"] or body.get("model") != self.config["model"]):
            raise ValueError("Topology actual bytes/model/purpose differ")
        cap = body.get("max_output_tokens")
        if type(cap) is not int or cap < 16:
            raise ValueError("Topology output must have a finite token cap")
        wire = json.dumps(body, ensure_ascii=False, allow_nan=False, sort_keys=body.get("stream") is True).encode()
        if body.get("stream") is True and hashlib.sha256(wire).hexdigest() != binding.get("request_bytes_sha256"):
            raise ValueError("Topology SSE physical bytes differ")
        model = binding["model"]
        if (model.get("provider") != self.config["provider"] or model.get("model") != self.config["model"]
                or model.get("topology_client_code_sha256") != hashlib.sha256(Path(__file__).read_bytes()).hexdigest()):
            raise ValueError("Topology actual model or client differs")
        identity = digest({"body_sha256": binding["body_sha256"], "provider": model["provider"],
            "model": model["model"], "endpoint_sha256": model.get("provider_endpoint_sha256")})
        return identity, len(wire) + 2 * cap + 4096


class TopologyJudge(ImplicitJudge):
    def __init__(self, config, *, root=ROOT, allocation="production", stage_options=None,
                 runtime=None, transport=None):
        validate_runtime_policy(config)
        self.topology_config = copy.deepcopy(config)
        self.stage_options = copy.deepcopy(stage_options or config["model_options"])
        self.runtime, self.transport = runtime, transport
        super().__init__(adapted_config(config, stage_options), root=root,
                         allocation=allocation, runtime=runtime, transport=transport)
        self.ledger = TopologyLedger(config, allocation, root=root)
        self.client.ledger = self.ledger
        self.client.model["topology_client_code_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if config.get("schema_version") == "gangmu.e4.topology_campaign.v2":
            self.client.model.update(topology_client_version=VERSION,
                                     topology_runtime_policy=POLICY_VERSION)
        self.cache_root = self.root / "runs/dev" / config["campaign_id"] / "structured-call-cache"

    def __call__(self, task_kind, instructions, payload):
        if not isinstance(task_kind, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", task_kind):
            raise ValueError("Invalid topology structured-call purpose")
        purpose = "topology:" + self.allocation + ":" + task_kind
        key = digest({"purpose": purpose, "instructions": instructions,
                      "payload": payload, "model": self.client.model})
        receipt = self.client.call_json(instructions, payload, output=self.cache_root / key,
                                       purpose=purpose, schema=payload.get("response_schema"))
        with self._lock:
            self._receipts.append({"task_kind": task_kind, "allocation": self.allocation,
                "request_key": key, "cache_hit": receipt["request_cache_hit"],
                "invocation": receipt["invocation"], "result_sha256": receipt["result_sha256"]})
        return receipt["data"]

    def fork(self, stage_options=None, allocation=None):
        return TopologyJudge(self.topology_config, root=self.root,
            allocation=allocation or self.allocation,
            stage_options=stage_options or self.stage_options,
            runtime=self.runtime, transport=self.transport)


def load_config(path):
    config = json.loads(Path(path).read_bytes())
    adapted_config(config)
    if config.get("schema_version") == "gangmu.e4.topology_campaign.v2":
        return validate_runtime_policy(config)
    if (config.get("max_node_expansion_rounds") != 3
            or config.get("max_document_isolation_rounds") != 3
            or config.get("max_reextractions_per_document") != 1
            or config.get("formal_publication") is not False
            or config.get("recall_acceptance") is not False):
        raise ValueError("Topology authorization/policy differs")
    return config
