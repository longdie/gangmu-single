"""Literal numbered example sections and explicitly titled warning tasks.

This extends the frozen v5 extractor. Unheaded ordinary prose definitions are
not converted into structural tasks here. Source activity remains checked;
neither the author source nor prior producers are changed.
"""
from __future__ import annotations

import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from .rule_nodes_handwritten_v5 import HandwrittenRuleNodeExtractor as PreviousExtractor

VERSION = "gangmu.rule.handwritten-boundaries.v6"
_NUMBERED_EXAMPLE = re.compile(r"Examples?\s+(?:[A-Z]?\d+(?:\.\d+)*[a-z]?|[IVXLCDM]+)[.:]?", re.I)
_WARNING = re.compile(r"(?m)^[ \t]*(?:\\noindent[ \t]*)?(?P<title>Warning|Caution)[ \t]*:")
_NEXT_TASK = re.compile(r"(?im)^[ \t]*(?:However[, \t]+)?we\s+have\s+the\s+following\s*[.:]?\s*$")


class HandwrittenRuleNodeExtractor(PreviousExtractor):
    def __init__(self, project):
        super().__init__(project)
        self._v6_heading_cache = None

    def _section_tasks(self):
        super()._section_tasks()
        sections = self._sections()
        for index, section in enumerate(sections):
            title = base._clean_title(section["event"].get("title") or "")
            if (not _NUMBERED_EXAMPLE.fullmatch(title) or self._inside_formal(section["start"])
                    or not self._balanced_body(0, section["start"], allow_document=True)):
                continue
            event = section["event"]
            spans = self._segments([(section["start"], section["title_end"])])
            reasons = []
            if event.get("activity") != "active" or named._unresolved_condition_scope(
                    self._condition_mask(self._mask, section["title_end"]), section["start"]):
                reasons.append("section_task_activity_unresolved")
            command = event.get("command")
            if (any(d.get("name", "").lstrip("\\") == command
                    for d in self.project.structure_data.get("definitions", []))
                    or re.search(r"\\(?:let|futurelet)\s*\\" + re.escape(command) + r"\b",
                                 self._mask[:section["title_end"]])):
                reasons.append("author_section_command_override")
            stop = min((s["start"] for s in sections[index + 1:] if s["level"] <= section["level"]),
                       default=len(self.text))
            stop = min(stop, min((p for p in self._scope_barriers() if p > section["start"]),
                                 default=len(self.text)))
            body = self._condition_mask(self._mask[section["title_end"]:stop], stop,
                                        offset=section["title_end"])
            if re.search(r"\\if(?!true\b|false\b)[A-Za-z]*\b", body):
                reasons.append("conditional_syntax_inside_section_task")
            if reasons:
                obligation = {"kind": "numbered_example_section_source_scope_unresolved",
                              "source_spans": spans, "reasons": reasons}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            children = [s for s in sections[index + 1:] if s["start"] < stop]
            formal = [e for e in self.events if e["kind"] == "formal"
                      and section["title_end"] <= e["start"] < stop]
            handwritten = [h for h in self._headings() if h["family"] not in {"case_marker", "barrier"}
                           and section["title_end"] <= h["start"] < stop]
            if children or formal or handwritten:
                self.structural_cues.append({"structural_family": "mathematical_section_container",
                    "title": title, "source_spans": spans,
                    "original_text": [self.project.resolve_span(s) for s in spans],
                    "reason": "numbered_example_organizes_independently_headed_tasks", "accepted": False})
                continue
            end = self._trim_end(section["start"], stop)
            if not self._mask[section["title_end"]:end].strip():
                continue
            candidate = self._candidate(section["start"], end, title=title,
                family="numbered_example_section_task_v6", boundary_status="complete_section_scope",
                basis=VERSION + ":literal_numbered_example_and_section_scope", event=event,
                section_title=title, heading_source_spans=spans)
            candidate["eligibility_hint"] = "requires_semantic_review"
            self.events.append({"kind": "section_task", "start": section["start"], "end": end,
                                "candidate": candidate})

    def _headings(self):
        if self._v6_heading_cache is not None:
            return [dict(h) for h in self._v6_heading_cache]
        headings = super()._headings()
        seen = {h["start"] for h in headings}
        scan, _, _ = named._project_scan_mask(self.project, self._mask)
        for match in _WARNING.finditer(scan):
            start = match.start("title")
            if (start in seen or not self._at_heading_position(start) or self._inside_formal(start)
                    or not self._balanced_body(0, start, allow_document=True)):
                continue
            noindent = scan.find("\\noindent", match.start(), start)
            if noindent >= 0:
                start = noindent
            if named._unresolved_condition_scope(self._condition_mask(scan, match.end()), start):
                obligation = {"kind": "warning_heading_activity_unresolved",
                              "source_spans": self._segments([(start, match.end())]),
                              "reasons": ["unresolved_heading_activity"]}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            headings.append({"start": start, "title_end": match.end(), "title": match["title"] + ":",
                             "category": "statement", "family": "plain_warning_heading_v6"})
            seen.add(start)
        self._v6_heading_cache = sorted(headings, key=lambda h: (h["start"], h["family"]))
        return [dict(h) for h in self._v6_heading_cache]

    def _balanced_body(self, start, end, *, allow_document=False):
        cursor, depth, math, envs = start, 0, [], []
        while cursor < end:
            try:
                token = self.lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            if token.pos >= end:
                break
            cursor = token.pos + token.len
            if cursor > end:
                return False
            if token.tok == "brace_open":
                depth += 1
            elif token.tok == "brace_close":
                if depth == 0:
                    return False
                depth -= 1
            elif token.tok in ("mathmode_inline", "mathmode_display", "mathmode_delimiter"):
                if math and (math[-1] == token.arg or (math[-1], token.arg) in
                             ((r"\(", r"\)"), (r"\[", r"\]"))):
                    math.pop()
                elif token.arg in ("$", "$$", r"\(", r"\["):
                    # Source TeX often puts inline mathematics inside a text
                    # argument of a display. Track the nested delimiters rather
                    # than poisoning all later document-scope titles.
                    math.append(token.arg)
                else:
                    return False
            elif token.tok == "macro" and token.arg in ("begin", "end"):
                group = self.lexer.group(cursor)
                if not group or not group["balanced"] or group["end"] > end:
                    return False
                if token.arg == "begin":
                    envs.append(group["value"])
                elif not envs or envs.pop() != group["value"]:
                    return False
                cursor = group["end"]
        return depth == 0 and not math and (not envs or allow_document and envs == ["document"])

    def _scan_body_end(self, heading, stop, ignored=None):
        end, basis = super()._scan_body_end(heading, stop, ignored)
        if heading["family"] != "plain_warning_heading_v6":
            return end, basis
        for match in _NEXT_TASK.finditer(self._mask, heading["title_end"], end):
            before = self._mask[heading["title_end"]:match.start()].rstrip()
            if before.endswith((".", "!", "?")) and self._balanced_body(heading["title_end"], match.start()):
                end, basis = match.start(), "v6_completed_warning_then_explicit_next_task_introduction"
                break
        body = self._condition_mask(self._mask[heading["title_end"]:end], end,
                                    offset=heading["title_end"])
        if re.search(r"\\if(?!true\b|false\b)[A-Za-z]*\b", body):
            obligation = {"kind": "warning_body_activity_unresolved",
                          "source_spans": self._segments([(heading["start"], end)]),
                          "reasons": ["conditional_syntax_inside_warning_task"]}
            if obligation not in self.v5_obligations:
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
            basis = "unresolved_warning_body_activity"
        return end, basis

    def _candidate(self, *args, **kwargs):
        candidate = super()._candidate(*args, **kwargs)
        if candidate["structural_family"] == "plain_warning_heading_v6":
            candidate["category_hint"] = "statement"
            candidate["eligibility_hint"] = "requires_semantic_review"
        return candidate


def extract_handwritten_rules(project):
    extractor = HandwrittenRuleNodeExtractor(project)
    rules = extractor.extract()
    return rules, {"schema_version": VERSION, "code": named._file_binding(__file__),
        "source_fingerprint": project.fingerprint, "activity_obligations": extractor.v5_obligations,
        "recovered_literal_sections": extractor.v5_recovered_sections,
        "supplier_starts": 0, "TeX_executions": 0}
