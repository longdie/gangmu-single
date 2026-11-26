"""Isolated handwritten-task boundaries; older producers remain unchanged.

Typography discovers tasks, not their mathematical validity. Paragraph breaks
alone do not end tasks: only balanced, completed prose followed by an explicit
navigation or exposition cue does. Cases remain within an enclosing proof.
No paper identifiers, reference coordinates or evaluation lists are read.
"""
from __future__ import annotations

import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named

VERSION = "gangmu.rule.handwritten-boundaries.v5"
_PARAGRAPH = re.compile(r"\r?\n[ \t]*\r?\n")
_NAVIGATION = re.compile(
    r"^(?:In\s+(?:Section|Chapter|Appendix)\b|"
    r"(?:In\s+the\s+following|Finally|Next|Below|Here)[,\s]+we\s+"
    r"(?:give|present|show|discuss|illustrate)\b|"
    r"The\s+following\s+(?:definition|theorem|lemma|proposition|corollary|"
    r"example|question|result|construction)\b|"
    r"(?:As\s+an?\s+(?:application|illustration)|Motivated\s+by)\b|"
    r"Although\s+we\s+(?:are|were)\s+(?:convinced|unable)\b)", re.I)
_EXPOSITION = re.compile(
    r"^(?:(?:A|An)\s+(?:typical|simple|concrete|instructive)\s+example\b|"
    r"Applying\s+(?:(?:the|this|that)\s+)?(?:above|preceding|previous)\s+"
    r"(?:result|theorem|proposition|lemma)\b)", re.I)
_FIND_WAYS = re.compile(r"^By\b[\s\S]*?\bwe\s+(?:find|describe)\s+"
                        r"(?:ways|methods|limitations)\b", re.I)
_NEXT_ASSERTION = re.compile(r"^By\b[\s\S]*?\b(?:immediately\s+)?we\s+"
                            r"(?:have|obtain|get)\s*[.:]?\s*$", re.I)


class HandwrittenRuleNodeExtractor(base.RuleNodeExtractor):
    """Share the established environment and provenance machinery explicitly."""

    def __init__(self, project):
        super().__init__(project)
        self.v5_obligations = []
        self._v5_heading_cache = None
        self._v5_section_cache = None
        self.v5_recovered_sections = []

    def _sections(self):
        if self._v5_section_cache is not None:
            return self._v5_section_cache
        sections = super()._sections()
        seen = {s["start"] for s in sections}
        scan, _, _ = named._project_scan_mask(self.project, self._mask)
        lexer = base._Lexer(scan)
        levels = {name: index for index, name in enumerate(("part", "chapter", "section", "subsection",
                                                           "subsubsection", "paragraph", "subparagraph"))}
        for match in re.finditer(r"(?m)^[ \t]*\\(part|chapter|section|subsection|subsubsection|paragraph|subparagraph)\b", scan):
            start = scan.find("\\", match.start(), match.end())
            if start in seen or self._inside_formal(start):
                continue
            cursor = lexer.skip(match.end())
            starred = scan[cursor:cursor + 1] == "*"
            cursor += int(starred)
            optional = lexer.group(cursor, optional=True)
            if optional:
                if not optional["balanced"]:
                    continue
                cursor = optional["end"]
            group = lexer.group(cursor)
            if not group or not group["balanced"]:
                continue
            spans = self._segments([(start, group["end"])])
            reasons = []
            if len(spans) != 1:
                reasons.append("section_command_crosses_source_segments")
            if named._unresolved_condition_scope(self._condition_mask(scan, group["end"]), start):
                reasons.append("section_activity_unresolved")
            if any(d.get("name", "").lstrip("\\") == match[1]
                   for d in self.project.structure_data.get("definitions", [])) or re.search(
                    r"\\(?:let|futurelet)\s*\\" + match[1] + r"\b", self._mask[:group["end"]]):
                reasons.append("author_section_command_override")
            if reasons:
                obligation = {"kind": "literal_section_source_scope_unresolved", "source_spans": spans, "reasons": reasons}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            event = {"id": "v5-section:" + base.digest({"source": self.project.fingerprint, "spans": spans})[:24],
                "source": spans[0], "file_occurrence": spans[0]["occurrence"], "command": match[1],
                "starred": starred, "title": group["value"], "activity": "active",
                "source_role": "literal_source_section_recovered_v5", "condition_ids": []}
            row = {"event": event, "start": start, "title_end": group["end"], "level": levels[match[1]]}
            self.v5_recovered_sections.append(row)
            sections.append(row)
            seen.add(start)
        self._v5_section_cache = sorted(sections, key=lambda s: s["start"])
        return self._v5_section_cache

    def _section_tasks(self):
        super()._section_tasks()
        sections = self._sections()
        for index, section in enumerate(sections):
            title = base._clean_title(section["event"].get("title") or "")
            if not re.fullmatch(r"(?:Examples?|Notations?|Conventions?(?:\s+and\s+notations?)?|Setting)[.:]?", title, re.I):
                continue
            if self._inside_formal(section["start"]):
                continue
            reasons = []
            event = section["event"]
            if event.get("activity") != "active" or named._unresolved_condition_scope(
                    self._condition_mask(self._mask, section["title_end"]), section["start"]):
                reasons.append("section_task_activity_unresolved")
            command = event.get("command")
            if any(d.get("name", "").lstrip("\\") == command for d in self.project.structure_data.get("definitions", [])) or re.search(
                    r"\\(?:let|futurelet)\s*\\" + re.escape(command) + r"\b", self._mask[:section["title_end"]]):
                reasons.append("author_section_command_override")
            if reasons:
                obligation = {"kind": "mathematical_section_task_source_scope_unresolved",
                    "source_spans": self._segments([(section["start"], section["title_end"])]), "reasons": reasons}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            stop = min((s["start"] for s in sections[index + 1:] if s["level"] <= section["level"]), default=len(self.text))
            stop = min(stop, min((p for p in self._scope_barriers() if p > section["start"]), default=len(self.text)))
            body = self._condition_mask(self._mask[section["title_end"]:stop], stop, offset=section["title_end"])
            if re.search(r"\\if(?!true\b|false\b)[A-Za-z]*\b", body):
                obligation = {"kind": "mathematical_section_task_body_activity_unresolved",
                    "source_spans": self._segments([(section["start"], stop)]),
                    "reasons": ["conditional_syntax_inside_section_task"]}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            children = [s for s in sections[index + 1:] if s["start"] < stop]
            formal = [e for e in self.events if e["kind"] == "formal" and section["title_end"] <= e["start"] < stop]
            handwritten_children = [h for h in self._headings() if h["family"] not in {"case_marker", "barrier"}
                                    and section["title_end"] <= h["start"] < stop]
            if children or formal or handwritten_children:
                spans = self._segments([(section["start"], section["title_end"])])
                self.structural_cues.append({"structural_family": "mathematical_section_container", "title": title,
                    "source_spans": spans, "original_text": [self.project.resolve_span(s) for s in spans],
                    "reason": "organizes_independently_headed_mathematical_tasks", "accepted": False})
                continue
            end = self._trim_end(section["start"], stop)
            if not self._mask[section["title_end"]:end].strip():
                continue
            candidate = self._candidate(section["start"], end, title=title,
                family="literal_mathematical_section_task_v5", boundary_status="complete_section_scope",
                basis=VERSION + ":literal_task_section_title_and_scope", event=section["event"],
                section_title=title, heading_source_spans=self._segments([(section["start"], section["title_end"])]))
            candidate["eligibility_hint"] = "requires_semantic_review"
            self.events.append({"kind": "section_task", "start": section["start"], "end": end, "candidate": candidate})

    def _trim_end(self, start, end):
        # Remove only full trailing comment lines, not arbitrary masked syntax.
        # Literal inactive branches must retain both if/fi delimiters; their
        # mask alone cannot identify a balanced physical source boundary.
        while end > start:
            while end > start and self.text[end - 1].isspace():
                end -= 1
            line_start = self.text.rfind("\n", start, end) + 1
            if line_start < start:
                line_start = start
            if (re.match(r"[ \t]*%", self.text[line_start:end])
                    and not self._mask[line_start:end].strip()):
                end = line_start
                continue
            break
        return super()._trim_end(start, end)

    def _condition_mask(self, mask, position, *, offset=0):
        """Ignore proven literal font aliases, never arbitrary if* commands.

        Only one active, document-scope, zero-argument declaration whose whole
        body is a standard math font applied to one letter qualifies. Uses before
        that declaration and names with visible rebinding remain unresolved.
        This changes an aligned scan mask, never source content or activity data.
        """
        overridden = any(d.get("name", "").lstrip("\\") == "iff"
                         for d in self.project.structure_data.get("definitions", []))
        overridden = overridden or bool(re.search(r"\\(?:let|futurelet)\s*\\iff\b", self._mask[:position]))
        overridden = overridden or bool(re.search(r"\\csname\s*iff\s*\\endcsname\b", self._mask[:position]))
        safe = {} if overridden else {"iff": 0}
        definitions = self.project.structure_data.get("definitions", [])
        for definition in definitions:
            name = definition.get("name", "").lstrip("\\")
            if (not re.fullmatch(r"if[A-Za-z]+", name) or name in base.PRIMITIVE_CONDITIONALS
                    or sum(d.get("name", "").lstrip("\\") == name for d in definitions) != 1
                    or definition.get("definition_kind") != "macro"
                    or definition.get("command") not in {"newcommand", "def"}
                    or definition.get("activity") != "active" or definition.get("condition_ids")
                    or definition.get("scope") != ["document"]
                    or definition.get("argument_count") != 0 or definition.get("optional_default") is not None
                    or not re.fullmatch(r"\s*\\math(?:frak|cal|rm|sf|it|bf|bb|scr)\s*\{[A-Za-z]\}\s*",
                                        definition.get("body") or "")):
                continue
            interval = self._event_range(definition)
            if interval is None or interval[1] > position:
                continue
            declaration_mask = self._mask if overridden else re.sub(r"\\iff\b", lambda m: " " * len(m[0]), self._mask)
            if named._unresolved_condition_scope(declaration_mask, interval[0]):
                continue
            visible = self._mask[:position]
            font = re.search(r"\\(math[A-Za-z]+)", definition["body"])[1]
            if (any(d.get("name", "").lstrip("\\") == font for d in definitions)
                    or any(re.search(r"\\(?:let|futurelet)\s*\\" + re.escape(control) + r"\b", visible)
                           or re.search(r"\\csname\s*" + re.escape(control) + r"\s*\\endcsname\b", visible)
                           for control in (name, font))):
                continue
            safe[name] = interval[1]
        def replace(match):
            ready = safe.get(match[0][1:])
            return " " * len(match[0]) if ready is not None and offset + match.start() >= ready else match[0]
        return re.sub(r"\\if[A-Za-z]*\b", replace, mask)

    def _headings(self):
        if self._v5_heading_cache is not None:
            return [dict(h) for h in self._v5_heading_cache]
        headings = super()._headings()
        for heading in headings:
            if heading.get("category") is None or heading["family"] in {"barrier", "case_marker"}:
                continue
            line_start = self._mask.rfind("\n", 0, heading["start"]) + 1
            prefix = self._mask[line_start:heading["start"]]
            noindent = re.search(r"\\noindent\s*$", prefix)
            if noindent:
                heading["start"] = line_start + noindent.start()
        for section in self.v5_recovered_sections:
            headings.append({"start": section["start"], "title_end": section["title_end"], "family": "barrier"})
        seen = {h["start"] for h in headings}
        scan, _, _ = named._project_scan_mask(self.project, self._mask)
        lexer = base._Lexer(scan)
        cursor = 0
        while cursor < len(scan):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok != "brace_open":
                continue
            group = lexer.group(token.pos)
            if not group or not group["balanced"] or not re.match(r"\s*\\em\b", group["value"]):
                continue
            # This scans the formatting group itself, including an assertion
            # wholly printed inside it. Its contents are never rewritten.
            title = base._clean_title(re.sub(r"^\s*\\em\b", "", group["value"], count=1))
            if not base._task_title_match(title) or not self._at_heading_position(token.pos) or self._inside_formal(token.pos):
                continue
            start = token.pos
            line_start = scan.rfind("\n", 0, start) + 1
            prefix = scan[line_start:start]
            noindent = re.fullmatch(r"\s*(\\noindent\s*)", prefix)
            if noindent:
                start = line_start + noindent.start(1)
            if start in seen:
                continue
            reasons = []
            if any(d.get("name", "").lstrip("\\") == "em"
                   for d in self.project.structure_data.get("definitions", [])) or re.search(r"\\(?:let|futurelet)\s*\\em\b", self._mask[:group["end"]]):
                reasons.append("author_em_override")
            if named._unresolved_condition_scope(self._condition_mask(scan, group["end"]), token.pos):
                reasons.append("unresolved_heading_activity")
            conditional_body = self._condition_mask(group["value"], group["end"], offset=group["start"] + 1)
            if re.search(r"\\(?:if[A-Za-z]*|else|or|fi)\b", conditional_body):
                reasons.append("conditional_inside_formatting_group")
            if base.category(title) == "proof" and any(re.search(r"\\" + re.escape(q) + r"\b", group["value"]) for q in base.QED):
                if any(d.get("name", "").lstrip("\\") in base.QED
                       for d in self.project.structure_data.get("definitions", [])) or any(
                    re.search(r"\\(?:let|futurelet)\s*\\" + re.escape(q) + r"\b", self._mask[:group["end"]]) for q in base.QED):
                    reasons.append("author_proof_end_marker_override")
            if reasons:
                obligation = {"kind": "handwritten_heading_activity_unresolved",
                    "source_spans": self._segments([(start, group["end"])]), "reasons": reasons}
                self.v5_obligations.append(obligation)
                self.unresolved.append(obligation)
                continue
            headings.append({"start": start, "title_end": group["end"], "title": title,
                "category": base.category(title), "family": "em_group_heading_v5",
                "format_group_start": token.pos})
            seen.add(start)
            cursor = group["end"]
        self._v5_heading_cache = sorted(headings, key=lambda h: (h["start"], h["family"]))
        return [dict(h) for h in self._v5_heading_cache]

    def _top_level_paragraphs(self, start, stop, ignored):
        """Yield real paragraph gaps outside groups, math and environments."""
        cursor, depth, math, envs = start, 0, None, []
        while cursor < stop:
            covered = next((b for a, b in ignored or [] if a <= cursor < b), None)
            if covered is not None:
                cursor = covered
                continue
            try:
                token = self.lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            if token.pos >= stop:
                break
            if not depth and math is None and not envs:
                gap_end = token.pos + token.len if token.tok == "char" and not token.arg.strip() else token.pos
                for match in _PARAGRAPH.finditer(self._mask, cursor, min(stop, gap_end)):
                    yield match.start(), match.end()
            covered = next((b for a, b in ignored or [] if a <= token.pos < b), None)
            if covered is not None:
                cursor = covered
                continue
            cursor = token.pos + token.len
            if token.tok == "brace_open":
                depth += 1
            elif token.tok == "brace_close":
                if depth == 0:
                    return
                depth -= 1
            elif token.tok in ("mathmode_inline", "mathmode_display", "mathmode_delimiter"):
                if math is None:
                    math = token.arg
                elif math == token.arg or (math, token.arg) in ((r"\(", r"\)"), (r"\[", r"\]")):
                    math = None
            elif token.tok == "macro" and token.arg in ("begin", "end"):
                group = self.lexer.group(cursor)
                if group and group["balanced"]:
                    if token.arg == "begin":
                        envs.append(group["value"])
                    elif envs and envs[-1] == group["value"]:
                        envs.pop()
                    else:
                        return
                    cursor = group["end"]

    def _discourse_end(self, heading, end, ignored):
        for split, next_start in self._top_level_paragraphs(heading["title_end"], end, ignored):
            before = self._mask[heading["title_end"]:split].rstrip()
            if not re.search(r"(?:[.!?][}\s]*|\$\$|\\\]|\\end\s*\{(?:equation\*?|align\*?|gather\*?)\})$", before):
                continue
            after = self._mask[next_start:end].lstrip()
            after = re.sub(r"^\\noindent\b\s*", "", after)
            next_paragraph = _PARAGRAPH.split(after, maxsplit=1)[0].strip()
            cue = _NAVIGATION.match(after)
            if heading["category"] != "proof":
                cue = cue or _EXPOSITION.match(after) or _FIND_WAYS.match(next_paragraph)
                if heading["category"] == "definition" and re.search(r"\bis\s+called\b", before, re.I):
                    cue = cue or re.match(r"In\s+fact[,\s]+by\s+(?:\[|\\cite\b)", after, re.I)
                if re.match(r"Remarks?\b", heading["title"], re.I):
                    cue = cue or re.match(r"Associated\s+(?:to|with)\b[^\n]*\bthere\s+(?:is|are)\b", next_paragraph, re.I)
            # A complete argument may be followed by a source introduction to
            # the next displayed task; it is not another step of that proof.
            if (_NEXT_ASSERTION.fullmatch(after.strip()) and any(h["start"] == end
                    and h.get("category") in {"statement", "definition", "proof"} for h in self._headings())):
                cue = True
            if cue:
                return split
        return None

    def _scan_body_end(self, heading, stop, ignored=None):
        if heading["category"] == "proof":
            if heading.get("format_group_start") is not None:
                group_heading = dict(heading, title_end=heading["format_group_start"])
                group_end, group_basis = base.RuleNodeExtractor._scan_body_end(
                    self, group_heading, heading["title_end"])
                if group_basis == "explicit_qed_marker" and group_end == heading["title_end"]:
                    return group_end, "v5_explicit_terminal_qed_inside_heading_group"
            headings = self._headings()
            is_case_stop = any(h["start"] == stop and h["family"] == "case_marker" for h in headings)
            active_outer = any(c["category_hint"] == "proof" and
                c["scope_range"]["char_start"] < heading["start"] < c["scope_range"]["char_end"]
                for c in self.candidates)
            # While the earlier named outer proof is being assembled, its
            # candidate does not exist yet. An explicitly same-target resumed
            # proof is the envelope; the intervening lemma proof ends at Case.
            pending_outer = False
            for outer in headings:
                if outer["family"] != "explicit_proof_introduction" or outer["start"] >= heading["start"]:
                    continue
                barrier = min((h["start"] for h in headings if h["family"] == "barrier"
                               and h["start"] > outer["start"]), default=len(self.text))
                first_resume = next((tail for tail in headings if tail.get("category") == "proof"
                    and tail["family"] != "explicit_proof_introduction" and outer["start"] < tail["start"] < barrier
                    and base._locator(tail.get("title", "")) == outer["target"]), None)
                if first_resume is None or first_resume["start"] <= heading["start"]:
                    continue
                first_task = min([h["start"] for h in headings if h["family"] not in {"case_marker", "barrier"}
                    and outer["start"] < h["start"] < barrier] + [e["start"] for e in self.events
                    if e["kind"] == "formal" and outer["start"] < e["start"] < barrier], default=barrier)
                # Inspect only the outer's first component, stopping before a
                # nested task. A nested proof's QED cannot close this envelope.
                _, outer_basis = base.RuleNodeExtractor._scan_body_end(self, outer, first_task)
                if outer_basis != "explicit_qed_marker":
                    pending_outer = True
                    break
            if is_case_stop and not active_outer and not pending_outer:
                # The old nested-task fallback used the first Case as a task
                # stop even when no enclosing proof existed. Keep all Cases.
                stop = min((h["start"] for h in headings if h["start"] > heading["start"]
                            and h["family"] != "case_marker"), default=len(self.text))
        if heading.get("family") == "em_group_heading_v5":
            task = base._task_title_match(heading["title"])
            suffix = heading["title"][task.end():] if task else ""
            after = self._mask[heading["title_end"]:stop]
            separated = (not after.strip() or re.match(r"[ \t]*(?:\r?\n[ \t]*){2}|"
                r"\s*\\(?:par|bigskip|medskip|smallskip|vspace)\b", after))
            if separated and re.match(r"\s*(?:(?:[A-Z]?\d+(?:\.\d+)*|[A-Z])\s*)?[.:]\s+\S", suffix, re.I):
                return heading["title_end"], "v5_assertion_wholly_inside_formatting_group"
        end, basis = super()._scan_body_end(heading, stop, ignored)
        # Explicit QED and closed plain environments have their own stronger
        # boundary. Ordinary paragraph narration cannot override those ends.
        if basis not in {"next_explicit_task_or_section", "document_end", "top_level_statement_layout_separator"}:
            return end, basis
        discourse = self._discourse_end(heading, end, ignored)
        return (discourse, "v5_completed_task_then_explicit_discourse_transition") if discourse is not None else (end, basis)


def extract_handwritten_rules(project):
    extractor = HandwrittenRuleNodeExtractor(project)
    rules = extractor.extract()
    return rules, {"schema_version": VERSION, "code": named._file_binding(__file__),
        "source_fingerprint": project.fingerprint, "activity_obligations": extractor.v5_obligations,
        "recovered_literal_sections": extractor.v5_recovered_sections,
        "supplier_starts": 0, "TeX_executions": 0}
