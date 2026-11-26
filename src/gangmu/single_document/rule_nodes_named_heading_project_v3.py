"""Explicit named-heading v3 adapter with bound source-only conditional guards.

Only the two named-heading activity guards differ from the historical v2
algorithm. Its unchanged auxiliary functions, heading discovery and source-mapped
candidate contract are reused explicitly. The historical composition remains
unchanged; a caller must select this separately named adapter.
"""
from __future__ import annotations

import collections
import copy
import hashlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ADAPTER_VERSION = "gangmu.rule.named-heading-project-supplement.v3"
_PINNED_DEPENDENCIES = {
    "prior_named_heading_project_adapter": (
        "src/gangmu/single_document/rule_nodes_named_heading_project_v2.py",
        "872cb8685ecbdb4f01ab8e344ea5c0b2987c1f383b2039423ce5d128f36060dc",
    ),
    "source_conditional_helper": (
        "src/gangmu/single_document/rule_source_conditional_v1.py",
        "648cc4688a1eeed1a3c7bb287e71d3d56fd23f0e36f9c1f0a79acbacdb8f0b06",
    ),
    "e1_command_definition_constants": (
        "src/gangmu/latex_structure.py",
        "5fbdf709b47110dc9780ef05372b556686ef5754ee005c46750a65080faa32f2",
    ),
}


def _file_binding(path: str | Path) -> dict[str, str]:
    """Bind actual repository code bytes without accepting redirected paths."""
    p = Path(path)
    resolved = p.resolve()
    if p.is_symlink() or resolved != p.absolute() or not resolved.is_relative_to(ROOT):
        raise ValueError("Adapter code dependencies must be local nonsymlink repository files")
    if not resolved.is_file():
        raise ValueError("Adapter code dependency is not a regular file")
    return {
        "path": str(resolved.relative_to(ROOT)),
        "sha256": hashlib.sha256(resolved.read_bytes()).hexdigest(),
    }


def _verify_pinned_dependencies() -> dict[str, dict[str, str]]:
    bindings = {}
    for name, (relative_path, expected_sha256) in _PINNED_DEPENDENCIES.items():
        binding = _file_binding(ROOT / relative_path)
        if binding["sha256"] != expected_sha256:
            raise ValueError(f"Named-heading v3 dependency hash mismatch: {relative_path}")
        bindings[name] = binding
    return bindings


# Check the code bytes before importing the reused algorithms, then check them
# again at the public entry point so a changed file cannot acquire a fresh binding.
_verify_pinned_dependencies()

from gangmu.single_document import rule_nodes_named_heading_project_v2 as prior_adapter_module
from gangmu.single_document import rule_source_conditional_v1 as source_conditional_module

base_rule_module = prior_adapter_module.base_rule_module
heading_module = prior_adapter_module.heading_module
_instruction_reading_ranges = prior_adapter_module._instruction_reading_ranges
_value_sha256 = prior_adapter_module._value_sha256
source_conditional_scope = source_conditional_module.source_conditional_scope


def _project_scan_mask(project, base_mask: str) -> tuple[str, list[dict[str, Any]], list[dict[str, Any]]]:
    """Blank only resolved, actually traversed input/include instruction ranges.

    The local mask never replaces project.reading or the frozen extractor mask.
    Resolution alone is insufficient: the target file and event occurrence must
    be present immediately after the invocation in the actual source map.
    Unexpanded/conditional inputs remain visible and are blocking review items.
    This operation neither inserts paragraph breaks nor certifies TeX activity.
    """
    chars = list(base_mask)
    blanked = []
    obligations = []
    for event in project.structure_data.get("includes", []):
        if event.get("command") not in {"input", "include"}:
            continue
        ranges = _instruction_reading_ranges(project, event)
        targets = [mapping for mapping in project.source_map if mapping["occurrence"] == event["id"] and mapping["source"]["file"] == event.get("resolved_file")]
        traversed = bool(ranges and targets and min(mapping["reading_char_start"] for mapping in targets) == ranges[-1]["char_end"] and any(mapping["reading_char_end"] > mapping["reading_char_start"] for mapping in targets))
        conditional_scope = source_conditional_scope(project, base_mask, ranges[0]["char_start"]) if ranges else []
        allowed = event.get("status") == "resolved" and event.get("activity") == "active" and traversed and not conditional_scope
        record = {
            "include_id": event["id"],
            "command": event["command"],
            "status": event.get("status"),
            "activity": event.get("activity"),
            "condition_ids": copy.deepcopy(event.get("condition_ids", [])),
            "source_spans": [{**copy.deepcopy(event["source"]), "occurrence": event["file_occurrence"]}],
            "reading_ranges": ranges,
            "resolved_file": event.get("resolved_file"),
            "target_occurrence_traversed_in_source_map": traversed,
            "conditional_scope": conditional_scope,
        }
        if allowed:
            for span in ranges:
                chars[span["char_start"]:span["char_end"]] = " " * (span["char_end"] - span["char_start"])
            blanked.append({**record, "instruction_only_blanked_in_local_scan_mask": True})
        else:
            reasons = []
            if event.get("status") != "resolved":
                reasons.append("input_not_resolved_and_expanded")
            if event.get("activity") != "active" or conditional_scope:
                reasons.append("input_activity_unresolved")
            if not ranges:
                reasons.append("instruction_mapping_not_complete")
            if not traversed:
                reasons.append("target_occurrence_not_traversed")
            obligations.append({
                **record,
                "kind": "named_heading_input_expansion_unresolved",
                "reasons": reasons,
                "instruction_not_blanked_by_adapter": True,
                "blocking_until_source_expansion_and_activity_review": True,
                "obligation": "Independent source expansion/activity decision before discovery/node-stage acceptance; an unresolved input may hide additional task structure.",
            })
    return "".join(chars), blanked, obligations


def supplement_project_rule_result(project, base_rule_result: dict[str, Any] | None = None) -> dict[str, Any]:
    """Return the v3 metadata wrapper and an exact compatible rule_input.

    With zero additions, rule_input is deep-value identical to the complete base,
    not merely identical candidates. With additions, old candidates keep every
    value and ID and a real supplement binding changes the rule-result hash.
    The inner v1 schema/rule_version describes the legacy representation contract;
    actual base/new algorithm versions and code hashes are explicit in metadata.
    """
    fixed_bindings = _verify_pinned_dependencies()
    extractor = base_rule_module.RuleNodeExtractor(project)
    base = extractor.extract() if base_rule_result is None else base_rule_result
    # _candidate appends to its extractor's candidate list. Copy the complete base
    # first so that source mapping cannot silently mutate caller-owned base data.
    original_base = copy.deepcopy(base)
    if original_base.get("paper_id") != project.paper_id:
        raise ValueError("Base rule result belongs to another paper")
    if original_base.get("source_fingerprint") != project.fingerprint:
        raise ValueError("Base rule result must bind the exact pinned SourceProject")
    blocked = [(r["char_start"], r["char_end"]) for c in original_base["candidates"] for r in c.get("reading_ranges", [])]
    scan_mask, blanked_input_instructions, input_obligations = _project_scan_mask(project, extractor._mask)
    discovered = heading_module.discover_named_heading_candidates(
        project.reading,
        file="__project_reading__",
        occurrence="__project_reading__",
        excluded_ranges=blocked,
        activity_mask=scan_mask,
    )
    additions = []
    activity_obligations = []
    for candidate in discovered:
        kind = candidate.author_type.casefold()
        status = "conjectural" if kind == "conjecture" else "question" if kind in {"question", "problem"} else "author_task"
        conditional_scope = source_conditional_scope(project, extractor._mask, candidate.char_start)
        activity = "requires_activity_review" if conditional_scope else "active"
        row = extractor._candidate(
            candidate.char_start,
            candidate.char_end,
            title=candidate.printed_title,
            family="handwritten_named_mathematical_heading",
            boundary_status="complete_named_heading_candidate",
            basis=f"{heading_module.SCHEMA_VERSION}:{candidate.boundary_basis}",
            activity=activity,
            category_hint="definition" if kind == "definition" else "statement",
            author_status=status,
        )
        if row is None or not row.get("source_spans"):
            raise ValueError("Named heading could not be mapped to exact original source")
        if row["accepted"] is not False or row["activity"] != activity:
            raise ValueError("Supplement must retain its actual lexical activity status and remain unaccepted")
        if row["reading_ranges"] != [{"char_start": candidate.char_start, "char_end": candidate.char_end}]:
            raise ValueError("Mapped candidate must retain its complete discovery reading range")
        if len(row["source_spans"]) != len(row["original_text"]):
            raise ValueError("Original source slices must pair one-to-one with source spans")
        for span, literal in zip(row["source_spans"], row["original_text"]):
            if project.resolve_span(span) != literal:
                raise ValueError("Supplement candidate original-source slice mismatch")
        additions.append(row)
        if conditional_scope:
            activity_obligations.append({
                "kind": "named_heading_activity_scope_unresolved",
                "candidate_id": row["candidate_id"],
                "reading_ranges": copy.deepcopy(row["reading_ranges"]),
                "source_spans": copy.deepcopy(row["source_spans"]),
                "activity": "requires_activity_review",
                "conditional_scope": conditional_scope,
                "blocking_until_source_activity_review": True,
                "obligation": "Independent original-source activity decision before discovery/node-stage acceptance; accepted=false is a separate semantic status and cannot satisfy this obligation.",
            })
    code = {
        "base_rule_representation_and_mask": _file_binding(base_rule_module.__file__),
        "new_named_heading_algorithm": _file_binding(heading_module.__file__),
        "new_project_adapter": _file_binding(__file__),
        **fixed_bindings,
    }
    source_control = {
        "helper_version": source_conditional_module.VERSION,
        "historical_adapter_version": prior_adapter_module.ADAPTER_VERSION,
        "code_bindings": copy.deepcopy(fixed_bindings),
        "scope": "Only the two named-heading input/title conditional guards use the separately bound source-only helper; the historical composition is unchanged.",
        "standard_iff_exception": "Ordinary logical iff is exempt only when source declarations, assignments, deferred code and dynamic control leave it unshadowed; all other unresolved conditional scopes remain blocking.",
        "original_reading_and_base_mask_unchanged": True,
        "TeX_executions": 0,
    }
    metadata = {
        "schema_version": ADAPTER_VERSION,
        "actual_algorithm": "explicit-base-plus-named-heading-supplement-v3-source-conditional-v1",
        "base_rule_algorithm_version": original_base["rule_version"],
        "new_heading_algorithm_version": heading_module.SCHEMA_VERSION,
        "legacy_inner_schema_scope": "Representation/ID compatibility contract only; this does not label the new composite algorithm as the old extractor.",
        "code_bindings": code,
        "source_control": source_control,
        "source_fingerprint": copy.deepcopy(project.fingerprint),
        "base_rule_result_value_sha256": _value_sha256(original_base),
        "supplement_candidates_value_sha256": _value_sha256(additions),
        "added_candidate_ids": [c["candidate_id"] for c in additions],
        "supplement_count": len(additions),
        "candidate_variant": "named-heading-v2-source-mapped" if additions else "unchanged-base-mathematical-input",
        "discovery_scope": "Every complete named mathematical task candidate counts, including unproved conjectures/questions; semantic acceptance remains false.",
        "source_mapping_basis": "Existing _candidate source-map and v1 ID representation; old extractor aligned _mask screens inactive/macro bodies and old reading_ranges prevent duplicate tasks.",
        "activity_basis": "Known false branches and macro bodies use the existing non-injecting aligned mask. The bound source-only helper exempts ordinary unshadowed iff; visible unknown conditional scopes remain requires_activity_review. Active is assigned only to new titles outside unresolved conditional scopes. No TeX evaluation or broad activity certification is claimed.",
        "input_instruction_scan_policy": "Only resolved, active input/include events with complete exact instruction mapping and actual target occurrence traversal are blanked in a separate local aligned scan mask. Original reading/source and frozen base mask retain invocation text. This creates no paragraph boundary; inline and line-wrapped prose remain excluded.",
        "local_scan_mask_sha256": hashlib.sha256(scan_mask.encode("utf-8")).hexdigest(),
        "blanked_input_instructions": blanked_input_instructions,
        "input_expansion_obligations": input_obligations,
        "unresolved_input_expansion_count": len(input_obligations),
        "activity_obligations": activity_obligations,
        "unresolved_activity_candidate_count": len(activity_obligations),
        "blocking_scan_obligation_count": len(input_obligations) + len(activity_obligations),
        "scan_ready_for_discovery_acceptance": not (input_obligations or activity_obligations),
        "supplier_starts": 0,
        "TeX_executions": 0,
        "cache_reuse_authorized": False,
        "needs_semantic_boundary_review": True,
    }
    merged = copy.deepcopy(original_base)
    if additions:
        merged["candidates"].extend(additions)
        merged["summary"]["candidate_count"] = len(merged["candidates"])
        merged["summary"]["by_family"] = dict(sorted(collections.Counter(c["structural_family"] for c in merged["candidates"]).items()))
        merged["summary"]["by_category_hint"] = dict(sorted(collections.Counter(c.get("category_hint") or "unknown" for c in merged["candidates"]).items()))
        if activity_obligations or input_obligations:
            merged["unresolved"].extend(copy.deepcopy(activity_obligations + input_obligations))
            merged["summary"]["unresolved_count"] = len(merged["unresolved"])
        merged["named_heading_supplement_binding"] = {
            "schema_version": ADAPTER_VERSION,
            "new_heading_algorithm_version": heading_module.SCHEMA_VERSION,
            "new_algorithm_code": code["new_named_heading_algorithm"],
            "project_adapter_code": code["new_project_adapter"],
            "source_control": copy.deepcopy(source_control),
            "source_fingerprint": copy.deepcopy(project.fingerprint),
            "base_rule_result_value_sha256": metadata["base_rule_result_value_sha256"],
            "supplement_candidates_value_sha256": metadata["supplement_candidates_value_sha256"],
            "added_candidate_ids": metadata["added_candidate_ids"],
            "activity_obligations": activity_obligations,
            "input_instruction_scan_policy": metadata["input_instruction_scan_policy"],
            "local_scan_mask_sha256": metadata["local_scan_mask_sha256"],
            "blanked_input_instructions": blanked_input_instructions,
            "input_expansion_obligations": input_obligations,
        }
    if _value_sha256(merged["candidates"][:len(original_base["candidates"])]) != _value_sha256(original_base["candidates"]):
        raise AssertionError("Supplement changed a preexisting mathematical candidate value")
    no_additions_equal = _value_sha256(merged) == _value_sha256(original_base) if not additions else None
    if not additions and no_additions_equal is not True:
        raise AssertionError("A no-addition project must retain the entire base rule input exactly")
    metadata["merged_rule_result_value_sha256"] = _value_sha256(merged)
    metadata["existing_candidate_prefix_values_unchanged"] = True
    metadata["no_additions_entire_rule_input_values_equal_base"] = no_additions_equal
    return {"schema_version": ADAPTER_VERSION, "rule_input": merged, "supplement_candidates": additions, "metadata": metadata}
