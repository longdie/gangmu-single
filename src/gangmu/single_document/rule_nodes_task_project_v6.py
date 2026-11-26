"""Compose isolated handwritten v6 with signed task supplements and wrappers.

Historical v4 discovery is retained, including its failed boundaries. Source
reviews remain exact independently supplied claims; this composition cannot
discharge them or authorize a later-stage model call.
"""
from __future__ import annotations

import collections
import copy

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from . import rule_nodes_qed_proofs_v4 as previous
from .rule_nodes_handwritten_v6 import extract_handwritten_rules
from .rule_nodes_numbered_wrappers_v5 import discover_numbered_task_wrappers
from .rule_scan_review_v3 import scan_obligations

VERSION = "gangmu.rule.task-project-supplement.v6"


def supplement_project_rule_result(project):
    historical = previous.supplement_project_rule_result(project)
    handwritten, handwritten_metadata = extract_handwritten_rules(project)
    prior = previous.supplement_project_rule_result(project, handwritten)
    merged = copy.deepcopy(prior["rule_input"])
    numbered = discover_numbered_task_wrappers(project, merged["candidates"])
    merged["candidates"].extend(copy.deepcopy(numbered["candidates"]))
    section_containers = []
    for section in list(merged["candidates"]):
        if section["structural_family"] not in {"literal_mathematical_section_task_v5", "mathematical_section_task", "numbered_example_section_task_v6"}:
            continue
        scope = section["scope_range"]
        children = [c for c in merged["candidates"] if c is not section
                    and c.get("category_hint") in {"statement", "definition", "proof", "expression"}
                    and scope["char_start"] < c["scope_range"]["char_start"]
                    and c["scope_range"]["char_end"] <= scope["char_end"]]
        if not children:
            continue
        spans = section["heading_source_spans"]
        merged["structural_cues"].append({"structural_family": "mathematical_section_container",
            "title": section["section_title"], "source_spans": copy.deepcopy(spans),
            "original_text": [project.resolve_span(s) for s in spans],
            "reason": "organizes_independently_supplemented_mathematical_tasks_v6",
            "child_candidate_ids": [c["candidate_id"] for c in children], "accepted": False})
        section_containers.append(copy.deepcopy(section))
        merged["candidates"].remove(section)
    resolved_ids = set(numbered["resolved_prior_structural_issue_event_ids"])
    resolved = [copy.deepcopy(x) for x in merged["unresolved"]
        if x.get("reason") == "unsupported_structural_wrapper" and x.get("event_id") in resolved_ids]
    merged["unresolved"] = [x for x in merged["unresolved"] if x not in resolved]
    merged["unresolved"].extend(copy.deepcopy(numbered["obligations"]))
    if numbered["candidates"]:
        # Newly recognized literal tasks can be the true structural neighbors of
        # existing proofs. These are only attachment hints, never proves edges.
        extractor = base.RuleNodeExtractor(project)
        extractor.candidates = merged["candidates"]
        for candidate in extractor.candidates:
            candidate["parent_candidate_id"] = None
        extractor._nesting_and_attachment()
    merged["summary"].update(candidate_count=len(merged["candidates"]),
        unresolved_count=len(merged["unresolved"]),
        by_family=dict(sorted(collections.Counter(c["structural_family"] for c in merged["candidates"]).items())),
        by_category_hint=dict(sorted(collections.Counter(str(c.get("category_hint")) for c in merged["candidates"]).items())))
    # Preserve the entire historical inner object when there was no change in
    # its actual source tasks. Metadata still binds the new discovery program.
    if merged == historical["rule_input"]:
        merged = copy.deepcopy(historical["rule_input"])
    metadata = {"schema_version": VERSION,
        "actual_algorithm": "handwritten-v6-plus-signed-v2-v3-v4-plus-literal-numbered-wrapper-v5",
        "code": named._file_binding(__file__), "source_fingerprint": copy.deepcopy(project.fingerprint),
        "handwritten_program": copy.deepcopy(handwritten_metadata),
        "numbered_wrapper_program": {k: copy.deepcopy(v) for k, v in numbered.items() if k != "candidates"},
        "prior_metadata": copy.deepcopy(prior["metadata"]),
        "activity_obligations": copy.deepcopy(handwritten_metadata["activity_obligations"]),
        "factory_scope_obligations": copy.deepcopy(numbered["obligations"]),
        "resolved_prior_structural_issues": resolved,
        "mathematical_sections_reclassified_as_containers": section_containers,
        "historical_v4_wrapper_value_sha256": named._value_sha256(historical),
        "prior_v4_wrapper_with_v6_base_value_sha256": named._value_sha256(prior),
        "historical_inner_unchanged": merged == historical["rule_input"],
        "historical_candidate_ids_removed": sorted({c["candidate_id"] for c in historical["rule_input"]["candidates"]}
            - {c["candidate_id"] for c in merged["candidates"]}),
        "candidate_ids_added": sorted({c["candidate_id"] for c in merged["candidates"]}
            - {c["candidate_id"] for c in historical["rule_input"]["candidates"]}),
        "merged_rule_result_value_sha256": named._value_sha256(merged),
        "supplier_starts": 0, "TeX_executions": 0}
    obligations = scan_obligations(metadata)
    if len({named._value_sha256(x) for x in obligations}) != len(obligations):
        raise ValueError("Duplicate inherited or new source scan obligations")
    metadata["blocking_scan_obligation_count"] = len(obligations)
    metadata["scan_ready_for_discovery_acceptance"] = not obligations
    for candidate in merged["candidates"]:
        if len(candidate["source_spans"]) != len(candidate["original_text"]) or any(
                project.resolve_span(s) != t for s, t in zip(candidate["source_spans"], candidate["original_text"])):
            raise ValueError("Composed tasks must retain exact source components")
    project.assert_unchanged()
    return {"schema_version": VERSION, "rule_input": merged,
        "historical_v4_wrapper": historical, "prior_v4_wrapper_with_v6_base": prior,
        "numbered_wrapper_candidates": copy.deepcopy(numbered["candidates"]), "metadata": metadata}
