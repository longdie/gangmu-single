"""Explicit composition of frozen rules, named headings and bounded factories.

This statically recognizes one literal two-argument repeated-task environment
factory. It never executes TeX or treats an arbitrary name prefix as a theorem.
Original issues are preserved when a supported declaration discharges them.
"""
from __future__ import annotations

import collections
import copy
import re

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from . import rule_nodes_named_headings_v2 as headings
from . import rule_nodes_qed_proofs_v3 as qed

VERSION = "gangmu.rule.project-task-supplement.v3"
_FACTORY = re.compile(
    r"\s*\\newenvironment\s*\{([A-Za-z]+)#1\}\s*\[1\]\s*\{\s*"
    r"\\def\s*\\([A-Za-z@]+)\s*\{#2\s*\\ref\s*\{##1\}\}\s*"
    r"\\begin\s*\{([A-Za-z@]+)\}\s*\}\s*\{\s*\\end\s*\{\3\}\s*\}\s*")


def _declared_control_name(project, definition):
    # The E1 lexical name may stop before @ outside makeatletter. Such an
    # ambiguous literal assignment must still block an internal carrier claim.
    literal = project.resolve_span({**definition["source"], "occurrence": definition["file_occurrence"]})
    match = re.match(r"\\(?:def|gdef|edef|xdef)\s*\\([A-Za-z@]+)|"
                     r"\\(?:newcommand|renewcommand|providecommand)\*?\s*\{?\s*\\([A-Za-z@]+)", literal)
    return next((x for x in match.groups() if x), None) if match else definition.get("name")


def _recognized_factories(project):
    definitions = {d["id"]: d for d in project.structure_data.get("definitions", [])}
    results = []
    for call in project.structure_data.get("macro_calls", []):
        definition = definitions.get(call.get("definition_id"), {})
        if (definition.get("definition_kind") != "macro" or definition.get("argument_count") != 2
                or definition.get("optional_default") is not None):
            continue
        match = _FACTORY.fullmatch(headings._mask_noncontent(definition.get("body") or ""))
        if not match:
            continue
        prefix, title_macro, inner_name = match.groups()
        carriers = [d for d in definitions.values() if d.get("name") == inner_name
                    and d.get("definition_kind") == "environment"
                    and d.get("theorem_like") is True and (d.get("body") or "").strip() == "\\" + title_macro]
        args = call.get("arguments", [])
        if len(carriers) != 1 or len(args) != 2:
            continue
        name_arg, title_arg = [a.get("value", "") for a in args]
        if not re.fullmatch(r"[A-Za-z]+", name_arg):
            continue
        results.append({"name": prefix + name_arg, "title": title_arg,
                        "literal_mathematical_title": bool(re.fullmatch(r"[A-Za-z][A-Za-z -]{0,79}", title_arg)
                                                           and base._task_title_match(title_arg)),
                        "factory_call": call, "factory_definition": definition, "carrier_definition": carriers[0]})
    return results


def supplement_project_rule_result(project, base_rule_result=None):
    prior = qed.supplement_project_rule_result(project, base_rule_result)
    incoming = copy.deepcopy(prior["rule_input"])
    extractor = base.RuleNodeExtractor(project)
    additions, obligations, declarations = [], [], []
    factories = _recognized_factories(project)
    for factory in factories:
        call, definition, carrier = [factory[k] for k in ("factory_call", "factory_definition", "carrier_definition")]
        declaration_ranges = [extractor._event_range(x) for x in (call, definition, carrier)]
        active_declaration = all(x.get("activity") == "active" and not x.get("condition_ids")
                                and x.get("scope") == ["document"]
                                for x in (call, definition, carrier)) and all(
            r and not qed.source_conditional_scope(project, extractor._mask, r[0]) for r in declaration_ranges)
        declarations.append({
            "environment_name": factory["name"], "printed_title": factory["title"],
            "factory_call_id": call["id"], "factory_definition_id": definition["id"],
            "carrier_definition_id": carrier["id"], "activity_resolved": active_declaration,
            "literal_mathematical_title": factory["literal_mathematical_title"],
            "source_spans": [{**copy.deepcopy(x["source"]), "occurrence": x["file_occurrence"]} for x in (call, definition, carrier)],
        })
        for event in project.structure_data.get("environments", []):
            if event.get("name") != factory["name"] or event.get("activity") == "inactive":
                continue
            interval = extractor._event_range(event)
            if interval and any(r["char_start"] == interval[0] and interval[1] == r["char_end"]
                                for c in incoming["candidates"] for r in c["reading_ranges"]):
                continue
            # Declarations after an occurrence and local/ambiguous definitions
            # cannot be projected into an earlier or unrelated occurrence.
            call_range, definition_range, carrier_range = declaration_ranges
            safe_order = bool(interval and all(r and r[1] <= interval[0] for r in (call_range, definition_range, carrier_range)))
            later_calls = [extractor._event_range(other["factory_call"]) for other in factories
                           if other["name"] == factory["name"] and other["factory_call"]["id"] != call["id"]]
            if interval and call_range and any(r and call_range[1] < r[1] <= interval[0] for r in later_calls):
                continue
            scope = qed.source_conditional_scope(project, extractor._mask, interval[0]) if interval else []
            shadow_from = {name: call_range[1] if call_range else None
                           for name in (factory["name"], "end" + factory["name"])}
            shadow_from.update({name: carrier_range[1] if carrier_range else None
                                for name in (carrier["name"], "end" + carrier["name"])})
            shadow_definitions = [d for d in project.structure_data.get("definitions", [])
                                  if (name := _declared_control_name(project, d)) in shadow_from and shadow_from[name] is not None
                                  and d.get("activity") != "inactive"
                                  and (r := extractor._event_range(d)) and interval
                                  and shadow_from[name] <= r[0] < interval[0]]
            visible_assignment = bool(interval and any(
                begin is not None and re.search(r"\\(?:let|futurelet)\s*\\" + re.escape(name) + r"\b",
                                               extractor._mask[begin:interval[0]])
                for name, begin in shadow_from.items()))
            if (not interval or not event.get("closed") or not active_declaration or not safe_order
                    or not factory["literal_mathematical_title"] or shadow_definitions or visible_assignment
                    or event.get("activity") != "active" or scope or event.get("definition_id") is not None):
                obligations.append({"kind": "repeated_task_factory_source_scope_unresolved",
                    "event_id": event["id"], "factory_call_id": call["id"],
                    "source_spans": [{**copy.deepcopy(event["source"]), "occurrence": event["file_occurrence"]}],
                    "blocking_until_source_scope_review": True,
                    "obligation": "Establish literal declaration order, activity and complete task boundary before accepting the repeated task."})
                continue
            row = extractor._candidate(*interval, title=factory["title"],
                family="literal_parameterized_repeated_task_environment",
                boundary_status="complete_closed_environment", basis=VERSION + ":exact_literal_two_argument_factory",
                event=event, activity="active", environment_name=event["name"],
                factory_call_id=call["id"], factory_definition_id=definition["id"],
                carrier_definition_id=carrier["id"], factory_source_spans=copy.deepcopy(declarations[-1]["source_spans"]))
            additions.append(row)
    resolved = [copy.deepcopy(x) for x in incoming["unresolved"]
                if x.get("reason") == "unsupported_structural_wrapper"
                and any(d["factory_call_id"] == x.get("event_id") and d["activity_resolved"] for d in declarations)]
    merged = copy.deepcopy(incoming)
    if additions:
        merged["candidates"].extend(additions)
        merged["unresolved"] = [x for x in merged["unresolved"] if x not in resolved] + copy.deepcopy(obligations)
        merged["summary"]["candidate_count"] = len(merged["candidates"])
        merged["summary"]["unresolved_count"] = len(merged["unresolved"])
        merged["summary"]["by_family"] = dict(sorted(collections.Counter(c["structural_family"] for c in merged["candidates"]).items()))
        merged["summary"]["by_category_hint"] = dict(sorted(collections.Counter(c.get("category_hint") or "unknown" for c in merged["candidates"]).items()))
        merged["task_factory_supplement_binding"] = {
            "schema_version": VERSION, "code": named._file_binding(__file__),
            "source_fingerprint": copy.deepcopy(project.fingerprint),
            "prior_rule_input_value_sha256": named._value_sha256(incoming),
            "declarations": declarations, "resolved_prior_structural_issues": resolved,
            "added_candidate_ids": [x["candidate_id"] for x in additions],
            "supplement_candidates_value_sha256": named._value_sha256(additions),
        }
    metadata = {"schema_version": VERSION,
        "actual_algorithm": "base-plus-named-heading-v2-plus-reviewed-qed-v3-plus-literal-factory-v3",
        "code": named._file_binding(__file__), "source_fingerprint": copy.deepcopy(project.fingerprint),
        "prior_metadata": copy.deepcopy(prior["metadata"]), "declarations": declarations,
        "resolved_prior_structural_issues": resolved, "factory_candidates_added": len(additions),
        "factory_scope_obligations": obligations,
        "blocking_scan_obligation_count": prior["metadata"]["blocking_scan_obligation_count"] + len(obligations),
        "scan_ready_for_discovery_acceptance": prior["metadata"]["scan_ready_for_discovery_acceptance"] and not obligations,
        "no_factory_changes_entire_rule_input_values_equal_prior": named._value_sha256(merged) == named._value_sha256(incoming) if not additions else None,
        "merged_rule_result_value_sha256": named._value_sha256(merged), "supplier_starts": 0, "TeX_executions": 0}
    project.assert_unchanged()
    return {"schema_version": VERSION, "rule_input": merged,
            "supplement_candidates": prior["supplement_candidates"] + additions, "metadata": metadata}
