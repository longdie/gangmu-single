"""Discharge exact source-scan obligations with independently retained reviews.

This validates bindings, not the mathematical truth of a review. Raw discovery
and its blockers are retained. Missing or changed evidence cannot authorize a
node-stage launch, including when a zero-addition rule input equals the base.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from .rule_nodes_named_heading_project_v2 import _value_sha256

VERSION = "gangmu.rule.scan-source-review.v3"
ROOT = Path(__file__).resolve().parents[3]
_OBLIGATION_FIELDS = ("input_expansion_obligations", "activity_obligations",
                      "qed_proof_boundary_obligations", "factory_scope_obligations")


def scan_obligations(metadata):
    rows = []
    if "prior_metadata" in metadata:
        rows.extend(scan_obligations(metadata["prior_metadata"]))
    for key in _OBLIGATION_FIELDS:
        values = metadata.get(key, [])
        if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
            raise ValueError("Malformed source-scan obligations")
        rows.extend(copy.deepcopy(values))
    return rows


def _evidence(binding):
    if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
        raise ValueError("Review evidence needs an exact local path and hash")
    if not isinstance(binding["path"], str) or not binding["path"]:
        raise ValueError("Review evidence needs a nonempty path")
    path = ROOT / binding["path"]
    resolved = path.resolve()
    if (not resolved.is_relative_to(ROOT) or not resolved.is_file()
            or any(p.is_symlink() for p in (path, *path.parents))):
        raise ValueError("Review evidence must be an existing project-local nonsymlink file")
    if hashlib.sha256(resolved.read_bytes()).hexdigest() != binding["sha256"]:
        raise ValueError("Independent source-review evidence bytes changed")


def resolve_scan_review(project, wrapper, packet=None):
    """Return a reviewed envelope; never silently turn a raw blocker into ready.

    The receipt is an explicitly trusted independent source-review claim. It
    must bind the entire raw wrapper and exactly one decision per obligation.
    No evaluation references or review explanation are inserted in rule_input.
    """
    metadata = wrapper["metadata"]
    if metadata["source_fingerprint"] != project.fingerprint:
        raise ValueError("Scan review source fingerprint differs")
    obligations = scan_obligations(metadata)
    hashes = [_value_sha256(x) for x in obligations]
    if (len(hashes) != len(set(hashes)) or type(metadata.get("blocking_scan_obligation_count")) is not int
            or metadata["blocking_scan_obligation_count"] != len(obligations)
            or metadata.get("scan_ready_for_discovery_acceptance") is not (not obligations)):
        raise ValueError("Raw scan obligations and readiness disagree")
    if packet is None:
        decisions = []
    else:
        if (packet.get("schema_version") != VERSION or packet.get("paper_id") != project.paper_id
                or packet.get("source_fingerprint") != project.fingerprint
                or packet.get("raw_wrapper_value_sha256") != _value_sha256(wrapper)
                or packet.get("independent_source_review") is not True
                or not isinstance(packet.get("decisions"), list)):
            raise ValueError("Independent review must bind these exact raw discovery bytes and source")
        decisions = packet["decisions"]
        decision_hashes = [x.get("obligation_value_sha256") for x in decisions]
        if len(decision_hashes) != len(set(decision_hashes)) or set(decision_hashes) != set(hashes):
            raise ValueError("Source review must address every exact obligation once, with no unused decisions")
        by_hash = dict(zip(hashes, obligations))
        for decision in decisions:
            item = by_hash[decision["obligation_value_sha256"]]
            disposition = decision.get("disposition")
            if item["kind"] == "unheaded_qed_candidate_full_boundary_review":
                allowed = disposition == "complete_original_proof_boundary"
            elif item["kind"] == "named_heading_input_expansion_unresolved":
                allowed = (disposition == "non_task_support_input" and item.get("status") == "system_input_expected"
                           and item.get("activity") == "active" and not item.get("conditional_scope"))
            else:
                allowed = False
            if (not allowed or not isinstance(decision.get("reason"), str) or not decision["reason"].strip()
                    or _value_sha256(decision.get("source_spans")) != _value_sha256(item["source_spans"])
                    or not isinstance(decision.get("evidence"), list) or not decision["evidence"]):
                raise ValueError("Unsupported or unbound source obligation disposition")
            for span in decision["source_spans"]:
                project.resolve_span(span)
            for evidence in decision["evidence"]:
                _evidence(evidence)
    ready = not obligations or bool(packet)
    rule_input = copy.deepcopy(wrapper["rule_input"])
    if ready and obligations:
        # Preserve the original queue in raw_rule_wrapper, and record every
        # discharged entry separately. Do not carry settled structural review
        # obligations as if they were missing source or mathematical failures.
        pending = rule_input.get("unresolved", [])
        discharged = [x for x in pending if _value_sha256(x) in set(hashes)]
        if discharged:
            rule_input["unresolved"] = [x for x in pending if _value_sha256(x) not in set(hashes)]
            rule_input["summary"]["unresolved_count"] = len(rule_input["unresolved"])
            rule_input["source_scan_resolution_binding"] = {
                "schema_version": VERSION, "raw_rule_input_value_sha256": _value_sha256(wrapper["rule_input"]),
                "source_review_value_sha256": _value_sha256(packet), "discharged_obligation_value_sha256": hashes}
    project.assert_unchanged()
    return {"schema_version": VERSION, "rule_input": rule_input,
        "source_review_packet": copy.deepcopy(packet),
        "raw_rule_wrapper": copy.deepcopy(wrapper),
        "metadata": {"schema_version": VERSION, "source_fingerprint": copy.deepcopy(project.fingerprint),
            "raw_wrapper_value_sha256": _value_sha256(wrapper),
            "source_review_value_sha256": _value_sha256(packet) if packet is not None else None,
            "raw_blocking_scan_obligation_count": len(obligations),
            "resolved_scan_obligation_count": len(decisions),
            "blocking_scan_obligation_count": 0 if ready else len(obligations),
            "scan_ready_for_discovery_acceptance": ready,
            "evidence_review_is_a_trusted_source_claim_not_a_mathematical_proof": True,
            "merged_rule_result_value_sha256": _value_sha256(rule_input),
            "supplier_starts": 0, "TeX_executions": 0}}
