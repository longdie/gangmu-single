"""Source-mapped short unheaded QED proofs, without altering frozen rules.

Only a single consequence paragraph immediately after a complete mathematical
task is proposed. Even this paragraph requires an independent full-boundary
review: a closed nested lemma does not establish an enclosing proof's start.
Other uncovered QED markers stay explicit boundary/activity obligations.
"""
from __future__ import annotations

import collections
import copy
import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from . import rule_nodes_named_headings_v2 as headings

VERSION = "gangmu.rule.qed-project-supplement.v3"
_QED = re.compile(r"\\(?:qed|qedhere)\b")
_START = re.compile(r"(?:As\b|Then\b|Thus\b|Hence\b|Therefore\b|Consequently\b|It\s+follows\b)", re.I)
_RESULT = re.compile(r"\b(?:Theorem|Lemma|Proposition|Corollary|Claim)\b", re.I)
_INFERENCE = re.compile(r"\b(?:implies|follows|follow|obtained|consequence|yields|deduce|deduced)\b", re.I)
_BARRIER = re.compile(r"\\(?:begin|end|section\*?|subsection\*?|subsubsection\*?|paragraph\*?|bibliography|bye)\b")


def source_conditional_scope(project, mask, position):
    """The standard math command iff is not a TeX conditional opener.

    Keep it unresolved if the author defines/overrides that command. All other
    visible if* tokens retain the conservative existing conditional review.
    """
    if not any(d.get("name") == "iff" for d in project.structure_data.get("definitions", [])):
        mask = re.sub(r"\\iff\b", lambda m: " " * len(m.group(0)), mask)
    return named._unresolved_condition_scope(mask, position)


def _top_level_qed(extractor, start: int, marker: re.Match) -> bool:
    """Check delimiter closure using the existing aligned lexical reader."""
    cursor, depth, math = start, 0, None
    while cursor <= marker.start():
        try:
            token = extractor.lexer.token(cursor)
        except LatexWalkerEndOfStream:
            return False
        cursor = token.pos + token.len
        if token.pos == marker.start():
            return token.tok == "macro" and token.arg in {"qed", "qedhere"} and depth == 0 and math is None
        if token.pos > marker.start():
            return False
        if token.tok == "brace_open":
            depth += 1
        elif token.tok == "brace_close":
            depth -= 1
            if depth < 0:
                return False
        elif token.tok in {"mathmode_inline", "mathmode_display", "mathmode_delimiter"}:
            if math is None:
                math = token.arg
            elif token.arg == math or (math, token.arg) in ((r"\(", r"\)"), (r"\[", r"\]")):
                math = None
            else:
                return False
    return False


def supplement_project_rule_result(project, base_rule_result=None):
    """Compose v2 named-title discovery with conservative QED discovery.

    With no new QED candidate, the entire v2 rule_input is retained verbatim.
    All raw scan obligations remain visible in the outer metadata. They must
    be independently discharged before accepting discovery or starting nodes.
    """
    prior = named.supplement_project_rule_result(project, base_rule_result)
    incoming = copy.deepcopy(prior["rule_input"])
    extractor = base.RuleNodeExtractor(project)
    mask, _, _ = named._project_scan_mask(project, extractor._mask)
    lower, upper = headings._active_bounds(mask)
    ranges = sorted(((r["char_start"], r["char_end"], c)
                    for c in incoming["candidates"] for r in c["reading_ranges"]),
                    key=lambda row: (row[0], row[1]))
    additions, obligations = [], []
    for marker in _QED.finditer(mask, lower, upper):
        if any(a <= marker.start() < b for a, b, _ in ranges):
            continue
        preceding = [(b, c) for a, b, c in ranges if b <= marker.start()]
        previous_end, previous = max(preceding, key=lambda row: row[0]) if preceding else (lower, None)
        gap = mask[previous_end:marker.start()]
        start = previous_end + len(gap) - len(gap.lstrip())
        body = mask[start:marker.start()]
        scope = source_conditional_scope(project, extractor._mask, start)
        reasons = []
        if previous is None or previous.get("boundary_status", "").startswith("unresolved"):
            reasons.append("no_preceding_complete_mathematical_task")
        if not re.match(r"[ \t]*(?:\r?\n[ \t]*){2}", gap):
            reasons.append("no_independent_paragraph_after_previous_task")
        if re.search(r"\r?\n[ \t]*\r?\n", body):
            reasons.append("multiple_unheaded_paragraphs_require_full_proof_boundary_review")
        if _BARRIER.search(body):
            reasons.append("intervening_task_or_scope_marker")
        if not (_START.match(body) and _RESULT.search(body) and _INFERENCE.search(body)
                and re.search(r"\\(?:ref|eqref|cref|Cref|autoref)\s*\{", body)):
            reasons.append("not_an_explicit_short_result_consequence")
        if not _top_level_qed(extractor, start, marker):
            reasons.append("qed_not_at_balanced_top_level")
        if scope:
            reasons.append("source_activity_scope_unresolved")
        if reasons:
            obligations.append({
                "kind": "unheaded_qed_proof_boundary_unresolved", "reasons": reasons,
                "source_spans": extractor._segments([(start, marker.end())]),
                "reading_ranges": [{"char_start": start, "char_end": marker.end()}],
                "conditional_scope": scope, "blocking_until_source_boundary_and_activity_review": True,
                "obligation": "Review the full author proof boundary; the last paragraph alone is not a complete proof by assumption.",
            })
            continue
        row = extractor._candidate(start, marker.end(), title="Proof (unheaded QED consequence)",
            family="unheaded_qed_consequence", boundary_status="short_qed_consequence_requires_full_boundary_review",
            basis=VERSION + ":complete_task_then_single_consequence_paragraph_then_top_level_qed",
            activity="active", category_hint="proof", author_status="author_supplied_proof_unverified")
        for span, literal in zip(row["source_spans"], row["original_text"]):
            if project.resolve_span(span) != literal:
                raise ValueError("Unheaded proof does not retain exact original source")
        additions.append(row)
        ranges.extend((r["char_start"], r["char_end"], row) for r in row["reading_ranges"])
        obligations.append({
            "kind": "unheaded_qed_candidate_full_boundary_review", "candidate_id": row["candidate_id"],
            "source_spans": copy.deepcopy(row["source_spans"]),
            "reading_ranges": copy.deepcopy(row["reading_ranges"]),
            "blocking_until_source_boundary_and_activity_review": True,
            "obligation": "Independently establish the full author proof, including possible prose before a nested mathematical task; an implication word does not certify a complete proof boundary.",
        })
    code = named._file_binding(__file__)
    merged = copy.deepcopy(incoming)
    if additions:
        merged["candidates"].extend(additions)
        merged["summary"]["candidate_count"] = len(merged["candidates"])
        merged["summary"]["by_family"] = dict(sorted(collections.Counter(c["structural_family"] for c in merged["candidates"]).items()))
        merged["summary"]["by_category_hint"] = dict(sorted(collections.Counter(c.get("category_hint") or "unknown" for c in merged["candidates"]).items()))
        merged["qed_proof_supplement_binding"] = {
            "schema_version": VERSION, "code": code,
            "prior_rule_input_value_sha256": named._value_sha256(incoming),
            "source_fingerprint": copy.deepcopy(project.fingerprint),
            "added_candidate_ids": [r["candidate_id"] for r in additions],
            "supplement_candidates_value_sha256": named._value_sha256(additions),
        }
        if obligations:
            merged["unresolved"].extend(copy.deepcopy(obligations))
            merged["summary"]["unresolved_count"] = len(merged["unresolved"])
    if named._value_sha256(merged["candidates"][:len(incoming["candidates"])]) != named._value_sha256(incoming["candidates"]):
        raise AssertionError("QED supplement changed a previous candidate")
    metadata = {
        "schema_version": VERSION, "actual_algorithm": "base-plus-named-heading-v2-plus-short-qed-v3",
        "code": code, "source_fingerprint": copy.deepcopy(project.fingerprint),
        "prior_metadata": copy.deepcopy(prior["metadata"]),
        "prior_rule_input_value_sha256": named._value_sha256(incoming),
        "qed_proof_supplement_count": len(additions),
        "qed_proof_boundary_obligations": obligations,
        "blocking_scan_obligation_count": prior["metadata"]["blocking_scan_obligation_count"] + len(obligations),
        "scan_ready_for_discovery_acceptance": prior["metadata"]["scan_ready_for_discovery_acceptance"] and not obligations,
        "no_qed_additions_entire_rule_input_values_equal_prior": named._value_sha256(merged) == named._value_sha256(incoming) if not additions else None,
        "merged_rule_result_value_sha256": named._value_sha256(merged),
        "supplier_starts": 0, "TeX_executions": 0,
    }
    project.assert_unchanged()
    return {"schema_version": VERSION, "rule_input": merged,
            "supplement_candidates": prior["supplement_candidates"] + additions, "metadata": metadata}
