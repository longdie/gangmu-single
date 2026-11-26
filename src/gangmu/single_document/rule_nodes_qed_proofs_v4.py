"""Propose explicitly introduced, unheaded author proofs ending with QED.

This extends the signed v3 result without changing its code or saved outputs.
Multi-paragraph arguments are structural candidates, never certified proofs:
each still needs the exact independent full-boundary source review. Historical
unresolved obligations are archived verbatim beside their replacements.
"""
from __future__ import annotations

import collections
import copy
import re

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from . import rule_nodes_project_supplement_v3 as previous
from . import rule_nodes_qed_proofs_v3 as qed
from .rule_scan_review_v3 import scan_obligations

VERSION = "gangmu.rule.qed-project-supplement.v4"
_PROOF_INTRO = re.compile(
    r"^(?:Now\s+(?:we\s+are\s+ready\s+to\s+prove|we\s+prove)|"
    r"Before\s+proving\s+(?:Theorem|Lemma|Proposition|Corollary|Claim)\b"
    r"[^\n]*(?:\r?\n[^\n]*){0,2}?\b(?:finish|complete)\s+the\s+proof\s+of)\b", re.I)
_SHORT_START = re.compile(r"^(?:Now|As|Then|Thus|Hence|Therefore|Consequently|It\s+follows)\b", re.I)
_SCOPE_BARRIER = re.compile(r"\\(?:section|subsection|subsubsection|paragraph|subparagraph|chapter|part|bibliography|bye)\*?\b")


def _body_activity_and_environments_resolved(project, mask, start, end):
    body = mask[start:end]
    iff_overridden = (any(d.get("name") == "iff" for d in project.structure_data.get("definitions", []))
                      or bool(re.search(r"\\(?:let|futurelet)\s*\\iff\b", mask[:end])))
    if iff_overridden and named._unresolved_condition_scope(mask, start):
        return False
    if not iff_overridden:
        body = re.sub(r"\\iff\b", lambda m: " " * len(m.group(0)), body)
    # A condition opened and closed inside the argument is not represented by
    # scope-at-start alone. Keep it unresolved rather than allowing a proof-only
    # boundary receipt to consume an activity or syntax obligation.
    if re.search(r"\\(?:if[A-Za-z]*|else|or|fi)\b", body):
        return False
    if re.search(r"\\(?:let|futurelet)\s*\\(?:qed|qedhere)\b", mask[:end]):
        return False
    stack = []
    commands = list(re.finditer(r"\\(?:begin|end)\b", body))
    for command in commands:
        env = re.match(r"\\(begin|end)\s*\{([A-Za-z@*][A-Za-z0-9@*_-]*)\}", body[command.start():])
        if not env:
            return False
        action, name = env.groups()
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
    return not stack


def _replace_obligations(metadata, replacements):
    result = copy.deepcopy(metadata)
    if "prior_metadata" in result:
        result["prior_metadata"] = _replace_obligations(result["prior_metadata"], replacements)
    if "qed_proof_boundary_obligations" in result:
        result["qed_proof_boundary_obligations"] = [
            copy.deepcopy(replacements.get(named._value_sha256(x), x))
            for x in result["qed_proof_boundary_obligations"]]
    return result


def supplement_project_rule_result(project, base_rule_result=None):
    prior = previous.supplement_project_rule_result(project, base_rule_result)
    incoming = copy.deepcopy(prior["rule_input"])
    extractor = base.RuleNodeExtractor(project)
    mask, _, _ = named._project_scan_mask(project, extractor._mask)
    replacements, additions, transitions = {}, [], []
    for obligation in scan_obligations(prior["metadata"]):
        if obligation["kind"] != "unheaded_qed_proof_boundary_unresolved":
            continue
        ranges = obligation.get("reading_ranges", [])
        if len(ranges) != 1 or obligation.get("conditional_scope"):
            continue
        start, end = ranges[0]["char_start"], ranges[0]["char_end"]
        body = mask[start:end]
        marker = re.search(r"\\(?:qed|qedhere)\b$", body)
        if marker and re.search(r"\\(?:qed|qedhere)\b", body[:marker.start()]):
            continue
        if not marker or _SCOPE_BARRIER.search(body) or not qed._top_level_qed(
                extractor, start, re.compile(r"\\(?:qed|qedhere)\b").search(mask, end - len(marker.group(0)), end)):
            continue
        if any(d.get("name") in {"qed", "qedhere"} for d in project.structure_data.get("definitions", [])):
            continue
        if not _body_activity_and_environments_resolved(project, mask, start, end):
            continue
        if "no_independent_paragraph_after_previous_task" in obligation["reasons"] or "no_preceding_complete_mathematical_task" in obligation["reasons"]:
            continue
        intro = bool(_PROOF_INTRO.match(body))
        short = bool(_SHORT_START.match(body) and qed._RESULT.search(body) and qed._INFERENCE.search(body)
                     and re.search(r"\\(?:ref|eqref|cref|Cref|autoref)\s*\{", body)
                     and not re.search(r"\r?\n[ \t]*\r?\n", body))
        if not (intro or short):
            continue
        row = extractor._candidate(start, end, title="Proof (unheaded author argument)",
            family="unheaded_qed_author_proof", boundary_status="unheaded_qed_requires_full_boundary_review",
            basis=VERSION + ":explicit_proof_intro_or_short_result_consequence_then_balanced_qed",
            activity="active", category_hint="proof", author_status="author_supplied_proof_unverified")
        if row["source_spans"] != obligation["source_spans"]:
            raise AssertionError("Proof proposal changed the original diagnostic source boundary")
        for span, literal in zip(row["source_spans"], row["original_text"]):
            if project.resolve_span(span) != literal:
                raise AssertionError("Author proof must preserve exact original text")
        decision = {"kind": "unheaded_qed_candidate_full_boundary_review", "candidate_id": row["candidate_id"],
            "source_spans": copy.deepcopy(row["source_spans"]), "reading_ranges": copy.deepcopy(row["reading_ranges"]),
            "blocking_until_source_boundary_and_activity_review": True,
            "obligation": "Independently establish the full author proof start, terminal QED and activity; a nearby closed task or proof announcement alone does not certify the complete boundary."}
        original_hash = named._value_sha256(obligation)
        replacements[original_hash] = decision
        additions.append(row)
        transitions.append({"original_obligation": copy.deepcopy(obligation),
                            "original_obligation_value_sha256": original_hash,
                            "replacement_obligation": copy.deepcopy(decision), "candidate_id": row["candidate_id"]})
    merged = copy.deepcopy(incoming)
    if additions:
        merged["candidates"].extend(additions)
        merged["unresolved"] = [copy.deepcopy(replacements.get(named._value_sha256(x), x)) for x in merged["unresolved"]]
        merged["summary"]["candidate_count"] = len(merged["candidates"])
        merged["summary"]["unresolved_count"] = len(merged["unresolved"])
        merged["summary"]["by_family"] = dict(sorted(collections.Counter(c["structural_family"] for c in merged["candidates"]).items()))
        merged["summary"]["by_category_hint"] = dict(sorted(collections.Counter(c.get("category_hint") or "unknown" for c in merged["candidates"]).items()))
        merged["qed_author_proof_supplement_binding"] = {"schema_version": VERSION,
            "code": named._file_binding(__file__), "source_fingerprint": copy.deepcopy(project.fingerprint),
            "prior_rule_input_value_sha256": named._value_sha256(incoming),
            "added_candidate_ids": [r["candidate_id"] for r in additions],
            "supplement_candidates_value_sha256": named._value_sha256(additions)}
    metadata = {"schema_version": VERSION, "actual_algorithm": "signed-v3-plus-explicit-unheaded-author-proof-v4",
        "code": named._file_binding(__file__), "source_fingerprint": copy.deepcopy(project.fingerprint),
        "prior_metadata": _replace_obligations(prior["metadata"], replacements),
        "prior_wrapper_value_sha256": named._value_sha256(prior), "proof_obligation_transitions": transitions,
        "unheaded_author_proof_candidates_added": len(additions),
        "blocking_scan_obligation_count": prior["metadata"]["blocking_scan_obligation_count"],
        "scan_ready_for_discovery_acceptance": prior["metadata"]["scan_ready_for_discovery_acceptance"],
        "no_additions_entire_rule_input_equal_prior": named._value_sha256(merged) == named._value_sha256(incoming) if not additions else None,
        "merged_rule_result_value_sha256": named._value_sha256(merged), "supplier_starts": 0, "TeX_executions": 0}
    if named._value_sha256(merged["candidates"][:len(incoming["candidates"])]) != named._value_sha256(incoming["candidates"]):
        raise AssertionError("Unheaded proof supplement altered previous candidates")
    project.assert_unchanged()
    return {"schema_version": VERSION, "rule_input": merged, "prior_raw_wrapper": prior,
            "supplement_candidates": prior["supplement_candidates"] + additions, "metadata": metadata}
