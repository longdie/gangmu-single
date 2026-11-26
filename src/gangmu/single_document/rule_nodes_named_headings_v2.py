"""Isolated original-TeX named-heading candidate supplement, explicit v2.

This module does not import or alter the frozen rule extractor. It discovers
standalone named/numbered mathematical titles followed by substantive content;
semantic acceptance and ambiguous handwritten-body boundaries remain review work.
Every returned candidate counts for discovery, including questions/conjectures.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass
import hashlib
import re
from typing import Any, Mapping, Sequence

SCHEMA_VERSION = "gangmu.rule.named-heading-supplement.v2"
_TYPES = r"Theorem|Proposition|Lemma|Corollary|Definition|Fact|Axiom|Assumption|Condition|Conjecture|Question|Problem|Example|Remark"
_TYPE = re.compile(rf"\b({_TYPES})\b", re.I)
_ENV = re.compile(r"\\(begin|end)\s*\{([^{}]+)\}")
_DECL = re.compile(r"\\newtheorem\*?\s*\{([^{}]+)\}\s*(?:\[[^\]\r\n]+\]\s*)?\{([^{}]+)\}")
_WRAPPERS = re.compile(r"\\(?:textbf|textit|textsc|emph|underline|paragraph\*?|subparagraph\*?)\s*\{")
_FONT = re.compile(r"\{\s*\\(?:bf|it|sc|bfseries|itshape|scshape)\b\s*")
_LAYOUT = re.compile(r"\\(?:noindent|medskip|smallskip|bigskip|par)\b\s*")
_BREAK = re.compile(r"\\(?:medskip|smallskip|bigskip|par|section\*?|subsection\*?|subsubsection\*?|paragraph\*?|subparagraph\*?|vspace\*?|end|begin)\b")
_BODY_START = re.compile(r"(?:If|Assume|Let|Suppose|Every|Any|For|There|No|A|An|The|We\s+define|Does)\b", re.I)


@dataclass(frozen=True)
class NamedHeadingCandidate:
    file: str
    occurrence: str
    char_start: int
    char_end: int
    title_end: int
    body_start: int
    printed_title: str
    author_type: str
    boundary_basis: str
    original_text: str
    activity_screened: bool = False

    def as_dict(self) -> dict[str, Any]:
        identity = f"{self.file}\0{self.occurrence}\0{self.char_start}\0{self.char_end}"
        digest = hashlib.sha256(identity.encode("utf-8")).hexdigest()[:20]
        kind = self.author_type.casefold()
        status = "conjectural" if kind == "conjecture" else "question" if kind in {"question", "problem"} else "author_task"
        return {
            "id": f"named-heading-v2:{digest}",
            "candidate_version": SCHEMA_VERSION,
            "structural_family": "standalone_named_mathematical_heading",
            "is_closed_environment": False,
            "printed_title": self.printed_title,
            "author_type": self.author_type,
            "author_status": status,
            "category_hint": "definition" if kind == "definition" else "statement",
            "discovery_required": True,
            "required": status not in {"conjectural", "question"},
            "required_flag_scope": "Semantic required-task slice only; false never exempts candidate discovery.",
            "source_spans": [{"file": self.file, "occurrence": self.occurrence, "char_start": self.char_start, "char_end": self.char_end}],
            "original_text": [self.original_text],
            "title_char_end": self.title_end,
            "body_char_start": self.body_start,
            "boundary_basis": self.boundary_basis,
            "needs_semantic_boundary_review": True,
            "activity": "caller_mask_screened_candidate" if self.activity_screened else "needs_activity_review",
        }


def _mask_noncontent(text: str) -> str:
    chars = list(text)
    for match in re.finditer(r"%[^\r\n]*", text):
        p = match.start() - 1
        slashes = 0
        while p >= 0 and text[p] == "\\":
            slashes += 1
            p -= 1
        if slashes % 2 == 0:
            chars[match.start():match.end()] = " " * (match.end() - match.start())
    masked = "".join(chars)
    for match in re.finditer(r"\\begin\s*\{(comment|verbatim\*?|lstlisting)\}.*?\\end\s*\{\1\}", masked, re.S):
        chars[match.start():match.end()] = [c if c in "\r\n" else " " for c in masked[match.start():match.end()]]
    return "".join(chars)


def _group_end(text: str, opening: int, upper: int) -> int | None:
    depth = 0
    for pos in range(opening, upper):
        c = text[pos]
        if c not in "{}":
            continue
        backslashes = 0
        before = pos - 1
        while before >= 0 and text[before] == "\\":
            backslashes += 1
            before -= 1
        if backslashes % 2:
            continue
        depth += 1 if c == "{" else -1
        if depth == 0:
            return pos + 1
    return None


def _active_bounds(masked: str) -> tuple[int, int]:
    begin = re.search(r"\\begin\s*\{document\}", masked)
    lower = begin.end() if begin else 0
    end = re.search(r"\\end\s*\{document\}|\\begin\s*\{thebibliography\}|\\bibliography\s*\{", masked[lower:])
    return lower, lower + end.start() if end else len(masked)


def _declared_task_ranges(masked: str, lower: int, upper: int) -> list[tuple[int, int]]:
    names = {"proof"}
    names.update(m.group(1) for m in _DECL.finditer(masked) if _TYPE.search(m.group(2)))
    names.update({"equation", "equation*", "align", "align*", "gather", "gather*", "tikzpicture"})
    stack: list[tuple[str, int]] = []
    spans = []
    for match in _ENV.finditer(masked, lower, upper):
        action, name = match.groups()
        if action == "begin":
            stack.append((name, match.start()))
        elif stack and stack[-1][0] == name:
            _, start = stack.pop()
            if name in names:
                spans.append((start, match.end()))
    # An unclosed known environment must not make its internal prose a new task.
    spans.extend((start, upper) for name, start in stack if name in names)
    return spans


def _normal_title(raw: str) -> tuple[str, str] | None:
    title = re.sub(r"\s+", " ", raw).strip().rstrip(".: ")
    if not title or len(title) > 180 or re.search(r"\\(?:ref|eqref|cite|begin|end|label)\b|[{}$=!?]", title):
        return None
    # Internal custom commands need later review; do not strip unknown macros into
    # an apparently certain title. Ordinary named TeX dash notation is retained.
    if "\\" in title:
        return None
    matches = list(_TYPE.finditer(title))
    if len(matches) != 1:
        return None
    match = matches[0]
    prefix = title[:match.start()].strip()
    suffix = title[match.end():].strip()
    if prefix and not suffix:
        # Named task type at the end: The Auslander--Reiten Conjecture, Poincare
        # Theorem. Reject ordinary sentence/citation lead-ins even if emphasized.
        name = re.sub(r"^The\s+", "", prefix, flags=re.I)
        if not re.fullmatch(r"[A-Z][A-Za-z0-9'’\-– ]{0,140}", name):
            return None
        if re.search(r"\b(?:by|see|using|from|recall|proof|statement|according|of|our|this)\b", name, re.I):
            return None
    elif not prefix:
        # Type first requires an actual number/name; a generic heading such as
        # 'Remark' or an organizational 'Theorem' alone is outside this supplement.
        if not suffix or not re.fullmatch(r"(?:\d+(?:\.\d+)*[A-Za-z]?|\(?[A-Z][A-Za-z0-9'’\-– .]*\)?)", suffix):
            return None
    else:
        return None
    return title, match.group(1).title()


def _strip_citation_suffix(tail: str) -> bool:
    # This is suffix validation, not arbitrary TeX expansion. Braces may contain
    # citation keys/options but no surrounding prose or task-body text.
    rest = re.sub(r"\\(?:cite[a-zA-Z]*|label)(?:\[[^\]]*\])*\s*\{[^{}]*\}", "", tail)
    return not rest.strip(" \t\r\n().:;")


def _parse_heading(masked: str, start: int, line_end: int) -> tuple[int, int, str, str] | None:
    pos = start
    while pos < line_end and masked[pos].isspace():
        pos += 1
    while True:
        layout = _LAYOUT.match(masked, pos, line_end)
        if not layout:
            break
        pos = layout.end()
    title_start = pos
    opening = _WRAPPERS.match(masked, pos, line_end)
    font = _FONT.match(masked, pos, line_end)
    if opening or font:
        match = opening or font
        brace = masked.find("{", pos, match.end())
        end = _group_end(masked, brace, line_end)
        if end is None:
            return None
        content_start = match.end()
        parsed = _normal_title(masked[content_start:end - 1])
        if parsed is None or not _strip_citation_suffix(masked[end:line_end]):
            return None
        return title_start, line_end, parsed[0], parsed[1]
    raw = masked[pos:line_end]
    citation = re.search(r"\s*(?:\(\s*)?\\cite[a-zA-Z]*(?:\[[^\]]*\])*\s*\{", raw)
    title_raw = raw[:citation.start()] if citation else raw
    if citation and not _strip_citation_suffix(raw[citation.start():]):
        return None
    parsed = _normal_title(title_raw)
    return (title_start, line_end, parsed[0], parsed[1]) if parsed else None


def _substantive_body(body: str) -> bool:
    visible = body.strip()
    while True:
        wrapper = _WRAPPERS.match(visible)
        font = _FONT.match(visible)
        if not wrapper and not font:
            break
        visible = visible[(wrapper or font).end():].lstrip()
    words = re.findall(r"[A-Za-z]{2,}", visible)
    if len("".join(words)) < 12:
        return False
    return bool(_BODY_START.match(visible) or visible.startswith(("$", "\\[", "\\(")))


def _body_end(masked: str, start: int, upper: int) -> tuple[int, str] | None:
    pos = start
    while pos < upper and masked[pos].isspace():
        pos += 1
    if pos == upper or _BREAK.match(masked, pos):
        return None
    wrapper = _WRAPPERS.match(masked, pos, upper) or _FONT.match(masked, pos, upper)
    if wrapper:
        brace = masked.find("{", pos, wrapper.end())
        wrapped_end = _group_end(masked, brace, upper)
        if wrapped_end is None:
            return None
        end = wrapped_end
        while end < upper and masked[end] in ".; ":
            end += 1
        # A following substantive tail in the same paragraph belongs to this body.
        if end >= upper or masked[end] in "\r\n":
            return end, "balanced_outer_statement_wrapper"
    depth = 0
    p = pos
    while p < upper:
        if masked[p] == "{" and (p == 0 or masked[p - 1] != "\\"):
            depth += 1
        elif masked[p] == "}" and (p == 0 or masked[p - 1] != "\\"):
            depth -= 1
        if depth == 0 and (_BREAK.match(masked, p) or re.match(r"\r?\n[ \t]*\r?\n", masked[p:])):
            break
        p += 1
    while p > pos and masked[p - 1].isspace():
        p -= 1
    return (p, "conservative_immediate_paragraph") if p > pos else None


def discover_named_heading_candidates(
    original_text: str,
    *,
    file: str,
    occurrence: str = "main",
    excluded_ranges: Sequence[tuple[int, int]] = (),
    activity_mask: str | None = None,
) -> list[NamedHeadingCandidate]:
    """Return source-grounded supplement candidates; never execute or expand TeX.

    Coordinates are Unicode offsets in the caller's original UTF-8 byte decode.
    Candidate boundary review is explicit for every result. Existing declared
    mathematical tasks and caller-provided occupied ranges suppress duplicates.
    A project adapter may blank proven resolved/traversed input instructions in
    its aligned mask. This does not create a paragraph/layout boundary: visible
    inline or line-wrapped prose keeps the same exclusion as single-file text.
    """
    if activity_mask is not None:
        if len(activity_mask) != len(original_text):
            raise ValueError("Caller mask must preserve the complete reading character alignment")
        if any(masked_char != source_char and masked_char != " " for source_char, masked_char in zip(original_text, activity_mask)):
            raise ValueError("An activity mask may only blank original characters with spaces; it cannot inject text")
    masked = _mask_noncontent(original_text if activity_mask is None else activity_mask)
    lower, upper = _active_bounds(masked)
    blocked = _declared_task_ranges(masked, lower, upper) + list(excluded_ranges)
    results = []
    line_starts = [0] + [m.end() for m in re.finditer("\n", masked)]
    for start in line_starts:
        if not lower <= start < upper or any(a <= start < b for a, b in blocked):
            continue
        end = masked.find("\n", start, upper)
        end = upper if end < 0 else end
        parsed = _parse_heading(masked, start, end)
        if parsed is None:
            continue
        title_start, title_end, title, kind = parsed
        # A standalone line must also start a paragraph, or explicitly declare a
        # paragraph/layout boundary. This rejects line-wrapped inline emphasis.
        previous = masked[lower:start].rstrip(" \t\r\n")
        previous_line = masked[masked.rfind("\n", lower, max(lower, start - 1)) + 1:start].strip()
        prefix_boundary = bool(re.match(r"\s*(?:\\(?:noindent|medskip|smallskip|bigskip|paragraph\*?|subparagraph\*?))\b", masked[start:end]))
        blank_before = not previous_line
        layout_before = bool(re.fullmatch(r"(?:\\(?:medskip|smallskip|bigskip|noindent|par)\s*)+", previous_line))
        if previous and not (blank_before or layout_before or prefix_boundary):
            continue
        body_start = title_end
        while body_start < upper and masked[body_start].isspace():
            body_start += 1
        body_line_end = masked.find("\n", body_start, upper)
        body_line_end = upper if body_line_end < 0 else body_line_end
        if _parse_heading(masked, body_start, body_line_end) is not None:
            # A new title is not the preceding title's statement body.
            continue
        found = _body_end(masked, body_start, upper)
        if found is None:
            continue
        body_end, basis = found
        if any(title_start < b and body_end > a for a, b in blocked):
            continue
        if not _substantive_body(masked[body_start:body_end]):
            continue
        results.append(NamedHeadingCandidate(file, occurrence, title_start, body_end, title_end, body_start, title, kind, basis, original_text[title_start:body_end], activity_mask is not None))
    return results


def merge_named_heading_supplement(
    existing_candidates: Sequence[Mapping[str, Any]],
    supplement: Sequence[NamedHeadingCandidate],
) -> dict[str, Any]:
    """Append new candidates under v2; existing objects retain exact deep values.

    Adapter input uses source_spans with physical file/occurrence/ranges. This
    function is not wired into the old extractor or cache/request registration.
    """
    old = copy.deepcopy(list(existing_candidates))
    merged = copy.deepcopy(old)
    occupied = [s for c in old for s in c.get("source_spans", [])]
    additions = []
    for candidate in supplement:
        if any(s.get("file") == candidate.file and s.get("occurrence", "main") == candidate.occurrence and s["char_start"] < candidate.char_end and s["char_end"] > candidate.char_start for s in occupied):
            continue
        row = candidate.as_dict()
        merged.append(row)
        additions.append(row)
        occupied.extend(row["source_spans"])
    assert merged[:len(old)] == old
    return {"schema_version": SCHEMA_VERSION, "candidates": merged, "supplement": additions, "supplement_count": len(additions), "existing_candidates_prefix_unchanged": True, "no_new_candidates_exact_value_equal": merged == old if not additions else None}
