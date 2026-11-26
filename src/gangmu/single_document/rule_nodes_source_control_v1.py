"""Explicit source-control envelope over the unchanged historical task-v6 result.

The old complete raw wrapper is retained. Independent named-v3 scans can settle
only an exactly bound, resolved/active/traversed input whose old lexical scope
consisted solely of standard iff tokens. No mathematical candidate is changed,
no historical module is replaced, and this module does not authorize a host call.
"""
from __future__ import annotations

import copy
from pathlib import Path

from . import rule_nodes_named_heading_project_v2 as prior_named
from . import rule_nodes_named_heading_project_v3 as named
from . import rule_nodes_task_project_v6 as historical_task
from .rule_nodes_handwritten_v6 import extract_handwritten_rules
from .rule_scan_review_v3 import resolve_scan_review, scan_obligations

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.rule.source-control.v1"
_FIELDS = ("input_expansion_obligations", "activity_obligations",
           "qed_proof_boundary_obligations", "factory_scope_obligations")
_INPUT_IDENTITY = ("include_id", "command", "status", "activity", "condition_ids",
    "source_spans", "reading_ranges", "resolved_file", "target_occurrence_traversed_in_source_map")
# Bind the complete historical task composition and its source/mask support.
# A future host must independently pin these records and the new entry modules.
_DEPENDENCY_PATHS = (
    "src/gangmu/single_document/rule_nodes.py",
    "src/gangmu/single_document/inventory.py",
    "src/gangmu/single_document/rule_nodes_named_heading_project_v2.py",
    "src/gangmu/single_document/rule_nodes_named_headings_v2.py",
    "src/gangmu/single_document/rule_nodes_qed_proofs_v3.py",
    "src/gangmu/single_document/rule_nodes_project_supplement_v3.py",
    "src/gangmu/single_document/rule_nodes_qed_proofs_v4.py",
    "src/gangmu/single_document/rule_nodes_task_project_v6.py",
    "src/gangmu/single_document/rule_nodes_handwritten_v5.py",
    "src/gangmu/single_document/rule_nodes_handwritten_v6.py",
    "src/gangmu/single_document/rule_nodes_numbered_wrappers_v5.py",
    "src/gangmu/single_document/rule_scan_review_v3.py",
    "src/gangmu/single_document/source.py",
    "src/gangmu/single_document/retrieval.py",
    "src/gangmu/latex_structure.py",
    "src/gangmu/representation/model.py",
    "src/gangmu/representation/graph.py",
    "src/gangmu/representation/sources.py",
)
_value_sha256 = prior_named._value_sha256


def program_bindings():
    """Return repository-relative actual code bindings for independent admission."""
    named._verify_pinned_dependencies()
    return {
        "source_control_adapter": named._file_binding(__file__),
        "named_heading_adapter": named._file_binding(named.__file__),
        "conditional_helper": named._file_binding(ROOT / "src/gangmu/single_document/rule_source_conditional_v1.py"),
        "historical_task_dependencies": [named._file_binding(ROOT / path) for path in _DEPENDENCY_PATHS],
    }


def _groups(metadata):
    groups = {key: [] for key in _FIELDS}
    if "prior_metadata" in metadata:
        prior = _groups(metadata["prior_metadata"])
        for key in _FIELDS:
            groups[key].extend(prior[key])
    for key in _FIELDS:
        rows = metadata.get(key, [])
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise ValueError("Malformed source-control scan obligations")
        groups[key].extend(copy.deepcopy(rows))
    return groups


def _identity(record):
    if any(key not in record for key in _INPUT_IDENTITY):
        raise ValueError("Input review needs the complete original instruction identity")
    return {key: copy.deepcopy(record[key]) for key in _INPUT_IDENTITY}


def _eligible_iff_input(item):
    scopes = item.get("conditional_scope")
    return (item.get("kind") == "named_heading_input_expansion_unresolved"
        and item.get("status") == "resolved" and item.get("activity") == "active"
        and item.get("target_occurrence_traversed_in_source_map") is True
        and item.get("instruction_not_blanked_by_adapter") is True
        and isinstance(scopes, list) and bool(scopes)
        and all(isinstance(scope, dict) and scope.get("command") == "iff"
                and scope.get("literal_known") is False for scope in scopes)
        and isinstance(item.get("reasons"), list)
        and set(item["reasons"]) == {"input_activity_unresolved"})


def _blanking_evidence(item, comparisons, project, base_mask):
    if not _eligible_iff_input(item):
        return None
    ranges = item.get("reading_ranges", [])
    if (not ranges or not named.source_conditional_module.standard_iff_is_unshadowed(
            project, base_mask, ranges[0]["char_start"])):
        return None
    identity = _identity(item)
    evidence = []
    for comparison in comparisons:
        matches = [row for row in comparison["new_blanked_input_instructions"]
                   if _identity(row) == identity]
        if (len(matches) != 1 or matches[0].get("conditional_scope")
                or matches[0].get("instruction_only_blanked_in_local_scan_mask") is not True):
            return None
        evidence.append({"context": comparison["context"],
            "new_named_wrapper_value_sha256": comparison["new_named_wrapper_value_sha256"],
            "blanked_instruction": copy.deepcopy(matches[0])})
    return evidence


def prepare_source_control(project):
    """Rebuild a new envelope while retaining the complete historical raw result.

    Candidate changes fail compatibility rather than inheriting old acceptance.
    Other old obligations and newly detected named obligations remain pending.
    The result is source-only input for a future explicitly bound admission path.
    """
    bindings = program_bindings()
    raw = historical_task.supplement_project_rule_result(project)
    raw_groups = _groups(raw["metadata"])
    raw_obligations = scan_obligations(raw["metadata"])
    if (raw["metadata"]["source_fingerprint"] != project.fingerprint
            or raw["metadata"]["blocking_scan_obligation_count"] != len(raw_obligations)
            or len({_value_sha256(row) for row in raw_obligations}) != len(raw_obligations)):
        raise ValueError("Historical raw source-control obligations disagree")
    handwritten, _ = extract_handwritten_rules(project)
    base_mask = named.base_rule_module.RuleNodeExtractor(project)._mask
    comparisons = []
    newly_detected = {key: [] for key in _FIELDS}
    for context, base_result in (("legacy-base", None), ("handwritten-v6-base", handwritten)):
        old = prior_named.supplement_project_rule_result(project, base_result)
        new = named.supplement_project_rule_result(project, base_result)
        if old["rule_input"]["candidates"] != new["rule_input"]["candidates"]:
            raise ValueError("Named-v3 changes candidate values; historical acceptance cannot be reused")
        old_groups = _groups(old["metadata"])
        new_groups = _groups(new["metadata"])
        for key in _FIELDS:
            previous_hashes = {_value_sha256(row) for row in old_groups[key]}
            newly_detected[key].extend(copy.deepcopy(row) for row in new_groups[key]
                                       if _value_sha256(row) not in previous_hashes)
        comparisons.append({"context": context,
            "old_named_wrapper_value_sha256": _value_sha256(old),
            "new_named_wrapper_value_sha256": _value_sha256(new),
            "candidate_values_unchanged": True,
            "candidate_values_sha256": _value_sha256(new["rule_input"]["candidates"]),
            "old_named_code_bindings": copy.deepcopy(old["metadata"]["code_bindings"]),
            "new_named_code_bindings": copy.deepcopy(new["metadata"]["code_bindings"]),
            "old_input_expansion_obligations": copy.deepcopy(old_groups["input_expansion_obligations"]),
            "new_input_expansion_obligations": copy.deepcopy(new_groups["input_expansion_obligations"]),
            "new_blanked_input_instructions": copy.deepcopy(new["metadata"]["blanked_input_instructions"])})
    pending = {key: [] for key in _FIELDS}
    decisions = []
    for key in _FIELDS:
        for item in raw_groups[key]:
            evidence = (_blanking_evidence(item, comparisons, project, base_mask)
                        if key == "input_expansion_obligations" else None)
            if evidence is None:
                pending[key].append(copy.deepcopy(item))
                continue
            for span in item["source_spans"]:
                project.resolve_span(span)
            decisions.append({"disposition": "resolved_active_traversed_standard_iff_scope",
                "obligation_value_sha256": _value_sha256(item),
                "original_obligation": copy.deepcopy(item),
                "source_spans": copy.deepcopy(item["source_spans"]),
                "source_fingerprint": copy.deepcopy(project.fingerprint),
                "source_control_version": VERSION, "evidence": evidence})
    pending_hashes = {_value_sha256(row) for rows in pending.values() for row in rows}
    added_count = 0
    for key in _FIELDS:
        for item in newly_detected[key]:
            value_hash = _value_sha256(item)
            if value_hash not in pending_hashes:
                pending[key].append(copy.deepcopy(item))
                pending_hashes.add(value_hash)
                added_count += 1
    metadata = {"schema_version": VERSION,
        "actual_algorithm": "unchanged-task-v6-plus-compatible-named-v3-source-control",
        "source_fingerprint": copy.deepcopy(project.fingerprint),
        "program_bindings": bindings,
        "raw_wrapper_value_sha256": _value_sha256(raw),
        "raw_blocking_scan_obligation_count": len(raw_obligations),
        "resolved_standard_iff_input_count": len(decisions),
        "newly_detected_scan_obligation_count": added_count,
        "blocking_scan_obligation_count": len(pending_hashes),
        "scan_ready_for_discovery_acceptance": not pending_hashes,
        "historical_rule_input_values_unchanged": True,
        "historical_candidate_values_unchanged": True,
        "merged_rule_result_value_sha256": _value_sha256(raw["rule_input"]),
        "source_control_decisions_value_sha256": _value_sha256(decisions),
        "supplier_starts": 0, "TeX_executions": 0,
        "current_frozen_host_admission_authorized": False,
        "historical_stage_acceptance_reissued": False,
        **pending}
    project.assert_unchanged()
    if bindings != program_bindings():
        raise ValueError("Source-control code bytes changed during preparation")
    envelope = {"schema_version": VERSION,
        "rule_input": copy.deepcopy(raw["rule_input"]),
        "raw_rule_wrapper": copy.deepcopy(raw),
        "named_scan_comparisons": comparisons,
        "source_control_decisions": decisions, "metadata": metadata}
    if len(scan_obligations(metadata)) != len(pending_hashes):
        raise ValueError("Source-control pending obligation identities are not unique")
    return envelope


def validate_source_control(project, envelope):
    """Require the complete supplied source-control envelope to replay exactly."""
    prepared = prepare_source_control(project)
    if envelope != prepared:
        raise ValueError("Supplied source-control envelope differs from complete source-only replay")
    return prepared


def resolve_source_control_review(project, envelope, packet=None):
    """Reuse bound v3 reviews for remaining obligations after exact new replay.

    This is an explicit new entry; the old host never calls it. The returned v3
    review envelope retains this source-control envelope as its raw wrapper.
    """
    prepared = validate_source_control(project, envelope)
    return resolve_scan_review(project, prepared, packet)
