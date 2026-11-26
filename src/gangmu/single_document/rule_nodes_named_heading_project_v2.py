"""Explicit base-plus-v2 supplement adapter to the existing rule representation.

The frozen extractor is called only for its base result, aligned lexical mask and
source-mapped v1 candidate-construction contract. The new algorithm lives in the
separate named-heading helper and is openly bound in new artifact metadata.
No supplier, node host, budget/cache mutation or E2-field expansion occurs here.
"""
from __future__ import annotations
import collections
import copy
import hashlib
import json
from pathlib import Path
import re
from typing import Any

from gangmu.single_document import rule_nodes as base_rule_module
from gangmu.single_document import rule_nodes_named_headings_v2 as heading_module

ADAPTER_VERSION = "gangmu.rule.named-heading-project-supplement.v2"
_CONDITIONAL_TOKEN = re.compile(r"\\(?:if[A-Za-z@]*|else|or|fi)\b")


def _unresolved_condition_scope(mask: str, position: int) -> list[dict[str, Any]]:
    """Conservatively track visible TeX conditional scopes; never evaluate TeX.

    Literal true/false branches are screened by the caller's existing mask. Other
    if* conditions remain source-review obligations in either branch. An unmatched
    else/or/fi also prevents an unqualified active assertion for the following
    title. This is a lexical activity obligation, not semantic task acceptance.
    """
    stack: list[dict[str, Any]] = []
    unmatched = []
    for match in _CONDITIONAL_TOKEN.finditer(mask, 0, position):
        command = match.group(0)[1:]
        if command.startswith("if"):
            # A \newif declaration names a switch but does not open a branch.
            if re.search(r"\\newif\s*$", mask[max(0, match.start() - 30):match.start()]):
                continue
            stack.append({"command": command, "char_start": match.start(), "literal_known": command in {"iftrue", "iffalse"}, "branch": "then"})
        elif command in {"else", "or"}:
            if stack:
                stack[-1]["branch"] = command
            else:
                unmatched.append({"command": command, "char_start": match.start(), "kind": "unmatched_conditional_branch"})
        elif stack:
            stack.pop()
        else:
            unmatched.append({"command": command, "char_start": match.start(), "kind": "unmatched_conditional_end"})
    return unmatched + [s for s in stack if not s["literal_known"]]


def _value_sha256(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _file_binding(path: str) -> dict[str, str]:
    p = Path(path).resolve()
    root = Path(__file__).resolve().parents[3]
    label = str(p.relative_to(root)) if p.is_relative_to(root) else str(p)
    return {"path": label, "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}


def _instruction_reading_ranges(project, event: dict[str, Any]) -> list[dict[str, int]]:
    """Map one include instruction using its physical file and occurrence."""
    source = event["source"]
    start, end = source["char_start"], source["char_end"]
    pieces = []
    for mapping in project.source_map:
        mapped_source = mapping["source"]
        if mapped_source["file"] != source["file"] or mapping["occurrence"] != event["file_occurrence"]:
            continue
        a = max(start, mapped_source["char_start"])
        b = min(end, mapped_source["char_end"])
        if a >= b:
            continue
        if mapping["reading_char_end"] - mapping["reading_char_start"] != mapped_source["char_end"] - mapped_source["char_start"]:
            raise ValueError("SourceProject mapping must preserve original character alignment")
        reading_start = mapping["reading_char_start"] + a - mapped_source["char_start"]
        pieces.append((a, b, reading_start, reading_start + b - a))
    pieces.sort()
    cursor = start
    previous_reading_end = None
    for a, b, reading_start, reading_end in pieces:
        if a != cursor or (previous_reading_end is not None and reading_start != previous_reading_end):
            return []
        cursor = b
        previous_reading_end = reading_end
    if cursor != end or not pieces:
        return []
    ranges = [{"char_start": pieces[0][2], "char_end": pieces[-1][3]}]
    literal = project.resolve_span({**source, "occurrence": event["file_occurrence"]})
    if project.reading[ranges[0]["char_start"]:ranges[0]["char_end"]] != literal:
        raise ValueError("Input instruction mapping does not retain the exact original source")
    return ranges


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
        conditional_scope = _unresolved_condition_scope(base_mask, ranges[0]["char_start"]) if ranges else []
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
    """Return a metadata wrapper and the exact compatible rule_input for a project.

    With zero additions, rule_input is deep-value identical to the complete base,
    not merely identical candidates. With additions, old candidates keep every
    value and ID and a real supplement binding changes the rule-result hash.
    The inner v1 schema/rule_version describes the legacy representation contract;
    actual base/new algorithm versions and code hashes are explicit in metadata.
    """
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
        conditional_scope = _unresolved_condition_scope(extractor._mask, candidate.char_start)
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
    }
    metadata = {
        "schema_version": ADAPTER_VERSION,
        "actual_algorithm": "explicit-base-plus-named-heading-supplement-v2",
        "base_rule_algorithm_version": original_base["rule_version"],
        "new_heading_algorithm_version": heading_module.SCHEMA_VERSION,
        "legacy_inner_schema_scope": "Representation/ID compatibility contract only; this does not label the new composite algorithm as the old extractor.",
        "code_bindings": code,
        "source_fingerprint": copy.deepcopy(project.fingerprint),
        "base_rule_result_value_sha256": _value_sha256(original_base),
        "supplement_candidates_value_sha256": _value_sha256(additions),
        "added_candidate_ids": [c["candidate_id"] for c in additions],
        "supplement_count": len(additions),
        "candidate_variant": "named-heading-v2-source-mapped" if additions else "unchanged-base-mathematical-input",
        "discovery_scope": "Every complete named mathematical task candidate counts, including unproved conjectures/questions; semantic acceptance remains false.",
        "source_mapping_basis": "Existing _candidate source-map and v1 ID representation; old extractor aligned _mask screens inactive/macro bodies and old reading_ranges prevent duplicate tasks.",
        "activity_basis": "Known false branches and macro bodies use the existing non-injecting aligned mask. Visible unknown if* conditional scopes remain requires_activity_review; active is assigned only to new titles outside unresolved conditional scopes. No TeX evaluation or broad activity certification is claimed.",
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
