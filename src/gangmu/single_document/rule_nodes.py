"""Versioned, deterministic explicit mathematical-region extraction.

This is a source inventory, not an accepted graph or a TeX interpreter.  Closed
environments and bounded literal author wrappers keep their original bytes;
handwritten headings are delimited by explicit layout/task markers.  Ambiguous
activity and boundaries remain visible rather than being decided by a model.
The historical E1 index and E4 baseline are deliberately not modified.
"""
from __future__ import annotations

from collections import Counter
import re
from typing import Any

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from gangmu.latex_structure import PRIMITIVE_CONDITIONALS, VERBATIM
from gangmu.representation.model import digest
from gangmu.representation.sources import SourceError

from .inventory import _Lexer

SCHEMA = "gangmu.e4.rule-nodes.v1"
RULE_VERSION = "explicit-regions-v1"
KINDS = ("theorem", "lemma", "proposition", "corollary", "definition", "proof",
         "demonstration", "claim", "conjecture", "question", "problem", "fact",
         "remark", "example", "assumption", "axiom", "condition", "notation",
         "convention", "setting", "exercise", "target", "need")
KIND_RE = re.compile(r"^(?:" + "|".join(KINDS) + r")(?:s)?\b", re.I)
NAMED_TASK_RE = re.compile(r"^(?:(?:A|An|The)\s+)?"
    r"(?:[A-Za-z][A-Za-z-]*\s+){0,4}"
    r"(?P<task>algorithms?|procedures?|constructions?|criteri(?:on|a))\b", re.I)
LAYOUT = {"medskip", "bigskip", "smallskip", "vspace", "vskip", "goodbreak",
          "pagebreak", "newpage", "clearpage"}
QED = {"qed", "qedhere", "square", "blacksquare", "Box", "endproof", "enddemo"}
UNSAFE_BODY = re.compile(r"\\(?:if\w*|else|fi|input|include|catcode|csname|expandafter|"
                         r"edef|xdef|def|gdef|let|futurelet|scantokens|loop|repeat)\b")
PROOF_INTRO = re.compile(r"(?im)^[ \t]*For\s+the\s+proof\s+of\s+"
                         r"((?:Theorem|Lemma|Proposition|Corollary|Claim)\s+[A-Z]?\d+(?:\.\d+)*)"
                         r"\s*[,.:]")
PROOF_ANNOUNCEMENT = re.compile(r"(?im)^[ \t]*(?:(?:In\s+the\s+following|Here|Below)[, \t]+)?"
    r"We\s+(?:give|provide|present|sketch)\s+(?:a\s+|an\s+|the\s+)?"
    r"(?:(?:short|direct|brief|alternative|new|elementary)\s+)*(?:proof|demonstration)\b")
PROOF_SECTION = re.compile(r"^(?:(?:A|An|The)\s+)?"
    r"(?:(?:short|direct|brief|alternative|new|elementary)\s+)*(?:proof|demonstration)\b", re.I)
NONMATHEMATICAL_TITLE = re.compile(r"^(?:acknowledg(?:e)?ments?|references?|bibliography|"
                                  r"funding|author\s+contributions?|data\s+availability)\b", re.I)
PLAIN_HEAD = re.compile(r"(?im)^[ \t]*(?:\\noindent[ \t]*)?"
                        r"((?:Proof|Demonstration)(?:\s+of\s+(?:Theorem|Lemma|Proposition|"
                        r"Corollary|Claim)(?:\s+[A-Z]?\d+(?:\.\d+)*)?)?[.:](?!\d)|"
                        r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|"
                        r"Question|Remark|Example|Assumption|Notation)"
                        r"(?:\s+[A-Z]?\d+(?:\.\d+)*)?[.:](?!\d))")


def _clean_title(value: str) -> str:
    value = value.strip()
    while value.startswith("{"):
        outer = _Lexer(value).group(0)
        if not outer or not outer["balanced"] or outer["end"] != len(value):
            break
        value = outer["value"].strip()
    value = re.sub(r"\\(?:bf|bfseries|it|itshape|rm|sc|gross)\b", " ", value)
    value = re.sub(r"\\(?:textbf|textit|emph|textrm|textsc)\s*\{([^{}]*)\}", r"\1", value)
    # Only true whole-title groups are removed.  A trailing locator such as
    # \ref{1} keeps its own balanced argument rather than losing its brace.
    return " ".join(value.strip().split())


def _task_title_match(title: str):
    """Recognize nominal task headings, not prose mentioning a procedure.

    Names before the task noun are lexical descriptors, never a special paper
    identifier. Source layout still decides whether this is a heading and the
    next stage decides whether the complete task is useful mathematical knowledge.
    """
    ordinary = KIND_RE.match(title)
    if ordinary:
        return ordinary
    named = NAMED_TASK_RE.match(title)
    if not named:
        return None
    prefix = title[:named.start("task")]
    suffix = title[named.end():].lstrip()
    if re.search(r"\b(?:we|they|you|he|she|it|our|their|using|use|uses|"
                 r"used|mention|mentions|proof|prove|proves)\b", prefix, re.I):
        return None
    if re.match(r"(?:is|are|was|were|has|have|uses?|provides?|appears?|"
                r"shows?|proves?)\b", suffix, re.I):
        return None
    if suffix and suffix[0] not in ".:([{$\\" and not re.match(
            r"(?:[A-Z]?\d+(?:\.\d+)*\b|[A-Z]\b|(?:for|of|on|with)\b)", suffix, re.I):
        return None
    return named


def category(title: str) -> str | None:
    title = _clean_title(title)
    if re.match(r"(?:proof|demonstration)\b", title, re.I):
        return "proof"
    if re.match(r"definitions?\b", title, re.I) and not re.search(
            r"definition\s*[-/]\s*(?:proposition|lemma|theorem)", title, re.I):
        return "definition"
    if _task_title_match(title):
        return "statement"
    return None


def _locator(title: str) -> str | None:
    match = re.search(r"\b(theorem|lemma|proposition|corollary|definition|claim)"
                      r"(?:\s|~)+([A-Z]?\d+(?:\.\d+)*[a-z]?)\b", title, re.I)
    return match[1].lower() + ":" + match[2].lower() if match else None


def _proof_identity(title: str) -> tuple[str, str] | None:
    """Read a proof's explicitly named target, never a later incidental ref."""
    title = _clean_title(title)
    prefix = re.match(r"^(?:End\s+of\s+(?:the\s+)?)?"
        r"(?:(?:A|An|The)\s+)?(?:(?:short|direct|brief|alternative|new|elementary)\s+)*"
        r"(?:proof|demonstration)\s+of\s+", title, re.I)
    if prefix is None:
        return None
    target = title[prefix.end():]
    reference = re.match(r"(?:(?:Theorem|Lemma|Proposition|Corollary|Claim)\s*[~\s]*)?"
        r"\\(?:ref|autoref|cref|Cref|nameref)\*?\s*\{([^{}\\,]+)\}", target)
    if reference and reference[1].strip():
        return "reference", reference[1].strip()
    locator = _locator(target)
    return ("locator", locator) if locator and re.match(
        r"(?:Theorem|Lemma|Proposition|Corollary|Claim)\b", target, re.I) else None


def _literal_environment_heading(body: str) -> str | None:
    """Read a literal printed heading without evaluating author TeX code.

    An author may declare ``prf`` with ``\\noindent {\\bf Proof.}\\rm``
    instead of wrapping ``\\begin{proof}``. Only bounded formatting prefixes
    and groups are traversed; ordinary mentions and dynamic TeX are not titles.
    """
    if UNSAFE_BODY.search(body):
        return None
    value = body.lstrip()
    layout = re.compile(r"^\\(?:noindent|par|smallskip|medskip|bigskip|"
                        r"bf|bfseries|it|itshape|rm|rmfamily|normalfont)\b\s*")
    formatting = re.compile(r"^\\(?:textbf|textit|emph|textrm|textsc)\b\s*")
    for _ in range(32):
        command = layout.match(value)
        if command:
            value = value[command.end():].lstrip()
            continue
        command = formatting.match(value)
        group_start = command.end() if command else 0
        if value[group_start:group_start + 1] == "{":
            group = _Lexer(value).group(group_start)
            if not group or not group["balanced"]:
                return None
            value = group["value"].lstrip()
            continue
        break
    match = _task_title_match(value)
    if not match:
        return None
    suffix = value[match.end():].lstrip()
    if not suffix or suffix.startswith((".", ":", "(", "[")):
        return match.group(0)
    if re.match(r"[A-Z]?\d+(?:\.\d+)*(?:\b|[.:])", suffix):
        return match.group(0) + " " + suffix.splitlines()[0]
    if match.group(0).lower() in {"proof", "demonstration"} and re.match(
            r"of\s+(?:Theorem|Lemma|Proposition|Corollary|Claim)\b", suffix, re.I):
        return match.group(0) + " " + suffix.splitlines()[0]
    return None


class RuleNodeExtractor:
    """Produce source-bound candidates without knowledge or relation decisions."""

    def __init__(self, project):
        project.assert_unchanged()
        self.project = project
        self.text = project.reading
        self.definitions = {d["id"]: d for d in project.structure_data.get("definitions", [])}
        self.unresolved: list[dict] = []
        self.candidates: list[dict] = []
        self.structural_cues: list[dict] = []
        self.events: list[dict] = []
        self._range_cache: dict[str, tuple[int, int] | None] = {}
        self._mask = self._lexical_mask()
        self.lexer = _Lexer(self._mask)

    def _positions(self, span, *, end=False, occurrence=None):
        coordinate = span["char_end" if end else "char_start"]
        return sorted({row["reading_char_start"] + coordinate - row["source"]["char_start"]
                       for row in self.project.source_map
                       if row["source"]["file"] == span["file"]
                       and (occurrence is None or row["occurrence"] == occurrence)
                       and (row["source"]["char_start"] < coordinate <= row["source"]["char_end"]
                            if end else row["source"]["char_start"] <= coordinate < row["source"]["char_end"])})

    def _event_range(self, event):
        if event.get("id") in self._range_cache:
            return self._range_cache[event["id"]]
        source = event["source"]
        ending = event.get("end_source") or event.get("full_source") or source
        occurrence = event.get("file_occurrence")
        starts = self._positions(source, occurrence=occurrence)
        ends = self._positions(ending, end=True,
                               occurrence=occurrence if ending["file"] == source["file"] else None)
        pair = next(((a, min(b for b in ends if b > a)) for a in starts if any(b > a for b in ends)), None)
        if event.get("id"):
            self._range_cache[event["id"]] = pair
        return pair

    def _project_source_span(self, span):
        """Project every occurrence of a file-local region, including definitions."""
        intervals = []
        for row in self.project.source_map:
            src = row["source"]
            if src["file"] != span["file"]:
                continue
            a, b = max(src["char_start"], span["char_start"]), min(src["char_end"], span["char_end"])
            if a < b:
                intervals.append((row["reading_char_start"] + a - src["char_start"],
                                  row["reading_char_start"] + b - src["char_start"]))
        return intervals

    def _lexical_mask(self):
        """Hide inert syntax but preserve every source coordinate and newline."""
        chars = list(self.text)
        hidden = []
        for kind in ("definitions", "deferred_code"):
            for event in self.project.structure_data.get(kind, []):
                hidden.extend(self._project_source_span(event["source"]))
        for event in self.project.structure_data.get("environments", []):
            if event.get("name") in VERBATIM or event.get("name") == "thebibliography":
                interval = self._event_range(event)
                if interval:
                    hidden.append(interval)
        for a, b in hidden:
            for i in range(a, b):
                if chars[i] not in "\r\n":
                    chars[i] = " "
        # A token walk suppresses comments, literal inactive branches and inline
        # verbatim.  Definition bodies have already been blanked, so an unused
        # macro's if/fi tokens cannot alter the document's activity state.
        visible = "".join(chars)
        lexer, cursor, conditions = _Lexer(visible), 0, []
        condition_names = PRIMITIVE_CONDITIONALS | {d.get("name", "").lstrip("\\")
            for d in self.definitions.values() if d.get("definition_kind") == "conditional"}
        while cursor < len(self.text):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if False in conditions:
                hidden.append((token.pos, cursor))
            if token.tok == "comment":
                hidden.append((token.pos, cursor))
            elif token.tok == "macro" and token.arg in ("verb", "lstinline"):
                a = cursor + int(cursor < len(self.text) and self.text[cursor] == "*")
                if a < len(self.text):
                    b = self.text.find(self.text[a], a + 1)
                    cursor = len(self.text) if b < 0 else b + 1
                    hidden.append((token.pos, cursor))
            elif token.tok == "macro" and token.arg in condition_names:
                conditions.append(True if token.arg == "iftrue" else False if token.arg == "iffalse" else None)
            elif token.tok == "macro" and token.arg in ("else", "or") and conditions:
                conditions[-1] = not conditions[-1] if conditions[-1] is not None and token.arg == "else" else None
            elif token.tok == "macro" and token.arg == "fi" and conditions:
                conditions.pop()
        for a, b in hidden:
            for i in range(a, b):
                if chars[i] not in "\r\n":
                    chars[i] = " "
        return "".join(chars)

    def _segments(self, intervals):
        result = []
        for start, end in intervals:
            for row in self.project.source_map:
                a, b = max(start, row["reading_char_start"]), min(end, row["reading_char_end"])
                if a >= b:
                    continue
                src = row["source"]
                result.append({"file": src["file"],
                               "char_start": src["char_start"] + a - row["reading_char_start"],
                               "char_end": src["char_start"] + b - row["reading_char_start"],
                               "occurrence": row["occurrence"]})
        return result

    def _issue(self, reason, *, interval=None, event=None, **extra):
        spans = self._segments([interval]) if interval else ([event["source"]] if event else [])
        record = {"reason": reason, "source_spans": spans, **extra}
        if event:
            record["event_id"] = event.get("id")
        self.unresolved.append(record)

    def _candidate(self, start, end, *, title, family, boundary_status, basis,
                   event=None, intervals=None, activity=None, **extra):
        if not 0 <= start < end <= len(self.text):
            raise SourceError("rule candidate interval is outside pinned reading")
        intervals = intervals or [(start, end)]
        if any(not start <= a < b <= end for a, b in intervals) or any(
                intervals[i][1] > intervals[i + 1][0] for i in range(len(intervals) - 1)):
            raise SourceError("rule candidate source components are outside or overlap its reading scope")
        spans = self._segments(intervals)
        original = [self.project.resolve_span(span) for span in spans]
        expected = "".join(self.text[a:b] for a, b in intervals)
        if "".join(original) != expected:
            raise SourceError("rule candidate does not round-trip through source map")
        labels = []
        for label in self.project.structure_data.get("labels", []):
            loc = self._event_range(label)
            if loc and any(a <= loc[0] and loc[1] <= b for a, b in intervals):
                labels.append({"keys": label.get("keys", []), "source_spans": self._segments([loc]),
                               "ownership": "contained_syntax_only"})
        candidate = {"candidate_id": "rule-node:" + digest({"rule_version": RULE_VERSION,
                        "source": self.project.fingerprint, "spans": spans, "family": family})[:24],
                     "structural_family": family, "title": _clean_title(title),
                     "category_hint": category(title), "source_spans": spans, "original_text": original,
                     "reading_ranges": [{"char_start": a, "char_end": b} for a, b in intervals],
                     "scope_range": {"char_start": start, "char_end": end},
                     "activity": activity or (event or {}).get("activity", "requires_activity_review"),
                     "boundary_status": boundary_status, "boundary_basis": basis,
                     "eligibility_hint": "core" if re.match(r"(?:theorem|lemma|proposition|corollary|"
                           r"definition|proof|claim|fact)\b", _clean_title(title), re.I)
                           else "requires_semantic_review",
                     "labels": labels, "parent_candidate_id": None, "proof_attachment_hint": None,
                     "event_id": (event or {}).get("id"), "accepted": False, **extra}
        if candidate["category_hint"] == "proof":
            target = _locator(_clean_title(title))
            named_kind = re.search(r"\b(?:proof|demonstration)\s+of\s+"
                                   r"(theorem|lemma|proposition|corollary|claim)\b", title, re.I)
            candidate["proof_attachment_hint"] = {"basis": "explicit_heading_locator" if target else
                        "explicit_heading_kind" if named_kind else "adjacent_task_requires_semantic_review",
                        "target_locator": target, "target_kind": named_kind[1].lower() if named_kind else None,
                        "target_candidate_ids": [], "status": "candidate_only"}
        self.candidates.append(candidate)
        return candidate

    def _environment_title(self, event, seen=None):
        seen = set() if seen is None else seen
        name = event.get("name", "")
        if name in seen:
            return name
        seen.add(name)
        definition = self.definitions.get(event.get("definition_id"))
        if definition and definition.get("theorem_like"):
            return definition.get("body") or name
        if definition:
            body = definition.get("body", "") or ""
            # Literal newenvironment aliases may wrap formatting/optional args.
            match = re.search(r"\\begin\s*\{([^{}]+)\}", body)
            if match and not UNSAFE_BODY.search(body):
                target = next((d for d in self.definitions.values()
                               if d.get("definition_kind") == "environment" and d.get("name") == match[1]), None)
                return self._environment_title({"name": match[1], "definition_id": (target or {}).get("id")}, seen)
            if definition.get("definition_kind") == "environment":
                heading = _literal_environment_heading(body)
                if heading and not UNSAFE_BODY.search(definition.get("end_body") or ""):
                    return heading
        return name.removesuffix("*")

    def _formal(self):
        for event in self.project.structure_data.get("environments", []):
            if event.get("source_role") != "document_or_bibliography" or event.get("activity") == "inactive":
                continue
            title = self._environment_title(event)
            if event.get("environment_kind") != "theorem" and not _task_title_match(_clean_title(title)):
                definition = self.definitions.get(event.get("definition_id"), {})
                targets = re.findall(r"\\begin\s*\{([^{}]+)\}", definition.get("body") or "")
                if any(_task_title_match(_clean_title(self._environment_title({"name": target,
                    "definition_id": next((d["id"] for d in self.definitions.values()
                        if d.get("definition_kind") == "environment" and d["name"] == target), None)})))
                    for target in targets):
                    self._issue("unsupported_mathematical_environment_alias", event=event,
                                interval=self._event_range(event), name=event.get("name"),
                                definition_id=event.get("definition_id"))
                continue
            interval = self._event_range(event)
            if not interval:
                self._issue("unmapped_mathematical_environment", event=event, title=title)
                continue
            if not event.get("closed"):
                self._issue("unclosed_mathematical_environment", event=event, interval=interval, title=title)
                continue
            optional_title = " ".join(str(arg.get("value", arg.get("text", ""))) for arg in event.get("arguments", []))
            candidate = self._candidate(*interval, title=title + (" " + optional_title if optional_title else ""),
                    family="formal_environment", boundary_status="complete_closed_environment",
                    basis="declared_environment_start_end", event=event, environment_name=event["name"],
                    definition_id=event.get("definition_id"))
            self.events.append({"kind": "formal", "start": interval[0], "end": interval[1],
                                "candidate": candidate, "event": event})

    def _wrappers(self):
        """Use only literal structural templates, never arbitrary expansion."""
        paired = []
        for call in self.project.structure_data.get("macro_calls", []):
            if call.get("activity") == "inactive" or call.get("source_role") != "document_or_bibliography":
                continue
            definition = self.definitions.get(call.get("definition_id"), {})
            body = definition.get("body") or ""
            if definition.get("argument_traversal_adapter") == "single_evaluation_savebox_v1":
                # E1 already traverses its literal source argument once.  This
                # bounded layout wrapper cannot create a mathematical task.
                continue
            if not definition.get("contains_structure"):
                continue
            interval = self._event_range(call)
            if not interval:
                self._issue("unmapped_structural_wrapper", event=call)
                continue
            if UNSAFE_BODY.search(body) or definition.get("delimited_parameters") or definition.get("xparse_signature"):
                self._issue("unsupported_structural_wrapper", interval=interval, event=call, name=call["name"])
                continue
            begin = re.fullmatch(r"\s*\\begin\s*\{([^{}]+)\}\s*(?:\[[^\]]*\])?\s*", body)
            end = re.fullmatch(r"\s*\\end\s*\{([^{}]+)\}\s*", body)
            whole = re.fullmatch(r"\s*\\begin\s*\{([^{}]+)\}\s*(.*?)\s*\\end\s*\{\1\}\s*", body, re.S)
            target = (begin or end or whole)[1] if (begin or end or whole) else None
            title = self._environment_title({"name": target, "definition_id": next(
                (d["id"] for d in self.definitions.values() if d.get("definition_kind") == "environment"
                 and d["name"] == target), None)}) if target else ""
            if not _task_title_match(_clean_title(title)):
                # Nonmathematical wrappers are outside node extraction, but a
                # potential theorem in an unbounded template remains visible.
                if re.search(r"(?:theorem|lemma|proof|proposition|definition)", body, re.I):
                    self._issue("unbounded_mathematical_wrapper", interval=interval, event=call, name=call["name"])
                continue
            if whole:
                if re.sub(r"#[1-9]|\s", "", whole[2]):
                    self._issue("wrapper_contains_unprojected_literal_content", interval=interval, event=call,
                                name=call["name"])
                    continue
                candidate = self._candidate(*interval, title=title, family="literal_macro_block",
                    boundary_status="complete_literal_wrapper", basis="bounded_balanced_environment_template",
                    event=call, macro_name=call["name"], definition_id=definition["id"],
                    macro_definition_source=definition["source"],
                    body_argument_sources=call.get("arguments", []))
                self.events.append({"kind": "formal", "start": interval[0], "end": interval[1], "candidate": candidate})
            elif begin or end:
                paired.append({"kind": "begin" if begin else "end", "name": target, "title": title,
                               "start": interval[0], "end": interval[1], "call": call, "definition": definition})
            else:
                self._issue("unbounded_mathematical_wrapper", interval=interval, event=call, name=call["name"])
        stack = []
        for marker in sorted(paired, key=lambda m: m["start"]):
            if marker["kind"] == "begin":
                stack.append(marker)
            elif stack and stack[-1]["name"] == marker["name"]:
                opening = stack.pop()
                candidate = self._candidate(opening["start"], marker["end"], title=opening["title"],
                    family="literal_macro_environment", boundary_status="complete_literal_wrapper",
                    basis="paired_literal_environment_macros", event=opening["call"],
                    wrapper_definition_ids=[opening["definition"]["id"], marker["definition"]["id"]])
                self.events.append({"kind": "formal", "start": opening["start"], "end": marker["end"], "candidate": candidate})
            else:
                self._issue("unmatched_mathematical_wrapper_end", interval=(marker["start"], marker["end"]),
                            name=marker["call"]["name"])
        for opening in stack:
            self._issue("unclosed_mathematical_wrapper", interval=(opening["start"], opening["end"]),
                        name=opening["call"]["name"])

    def _inside_formal(self, start):
        return any(e["start"] <= start < e["end"] for e in self.events if e["kind"] == "formal")

    def _scope_barriers(self):
        """Explicit document/appendix/bibliography boundaries in pinned reading."""
        result = []
        for event in self.project.structure_data.get("environments", []):
            if event.get("name") == "thebibliography":
                interval = self._event_range(event)
                if interval:
                    result.append(interval[0])
            elif event.get("name") in ("appendix", "document") and event.get("end_source"):
                ends = self._positions(event["end_source"], occurrence=event.get("file_occurrence"))
                result.extend(ends)
        for event in self.project.structure_data.get("includes", []):
            if event.get("command") == "bibliography":
                interval = self._event_range(event)
                if interval:
                    result.append(interval[0])
        result.extend(m.start() for m in re.finditer(
            r"\\(?:end\s*\{(?:document|appendix)\}|begin\s*\{thebibliography\}|"
            r"(?:appendix|bye|endinput)\b)", self._mask))
        return sorted(set(result))

    def _sections(self):
        """Active source headings with their explicit nesting levels."""
        sections = []
        levels = {name: index for index, name in enumerate(("part", "chapter", "section", "subsection",
                                                           "subsubsection", "paragraph", "subparagraph"))}
        for event in self.project.structure_data.get("sections", []):
            interval = self._event_range(event)
            if interval and event.get("activity") != "inactive":
                sections.append({"event": event, "start": interval[0], "title_end": interval[1],
                                 "level": levels.get(event.get("command"), 99)})
        sections.sort(key=lambda section: section["start"])
        return sections

    def _section_tasks(self):
        """Keep explicitly named unheaded tasks; preserve organizers as cues."""
        sections = self._sections()
        for index, section in enumerate(sections):
            event = section["event"]
            title = _clean_title(event.get("title") or "")
            hinted = "definition" if re.match(r"Definitions?\b", title, re.I) else \
                     "proof" if PROOF_SECTION.match(title) else None
            if hinted is None or self._inside_formal(section["start"]):
                continue
            stop = min((other["start"] for other in sections[index + 1:]
                        if other["level"] <= section["level"]), default=len(self.text))
            stop = min(stop, min((a for a in self._scope_barriers() if a > section["start"]), default=len(self.text)))
            child_sections = [other for other in sections[index + 1:] if other["start"] < stop]
            formal_tasks = [e for e in self.events if e["kind"] == "formal"
                            and section["title_end"] <= e["start"] < stop]
            if child_sections or formal_tasks:
                # This title organizes separately declared mathematical tasks.
                # It remains discoverable without duplicating their whole proof
                # content as another automatically qualified mathematical node.
                spans = self._segments([(section["start"], section["title_end"])])
                self.structural_cues.append({"structural_family": "mathematical_section_container",
                    "title": title, "category_hint": hinted, "source_spans": spans,
                    "original_text": [self.project.resolve_span(span) for span in spans],
                    "scope_range": {"char_start": section["start"], "char_end": stop},
                    "event_id": event.get("id"), "activity": event.get("activity", "unknown"),
                    "reason": "organizes_independently_headed_mathematical_tasks",
                    "child_candidate_ids": [e["candidate"]["candidate_id"] for e in formal_tasks],
                    "accepted": False})
                continue
            end = self._trim_end(section["start"], stop)
            if end <= section["title_end"]:
                self._issue("empty_mathematical_section_task", interval=(section["start"], section["title_end"]),
                            title=title)
                continue
            normalized = title if hinted == "definition" or re.match(r"(?:Proof|Demonstration)\b", title, re.I) else "Proof: " + title
            candidate = self._candidate(section["start"], end, title=normalized,
                family="mathematical_section_task", boundary_status="complete_section_scope",
                basis="explicit_mathematical_section_title_and_scope", event=event,
                section_title=title, heading_source_spans=self._segments([(section["start"], section["title_end"])]))
            candidate["eligibility_hint"] = "requires_semantic_review"
            self.events.append({"kind": "section_task", "start": section["start"], "end": end,
                                "candidate": candidate})

    def _at_heading_position(self, start):
        line = self._mask[self._mask.rfind("\n", 0, start) + 1:start]
        line = re.sub(r"\\(?:noindent|medskip|bigskip|smallskip|goodbreak|par)\b", "", line)
        return not line.strip()

    def _headings(self):
        headings, seen = [], set()

        def add(start, end, title, family, **extra):
            clean = _clean_title(title)
            if start in seen or not (_task_title_match(clean) or re.match(r"Case\b", clean, re.I)
                    or NONMATHEMATICAL_TITLE.match(clean)) or self._inside_formal(start):
                return
            if not KIND_RE.match(clean) and NAMED_TASK_RE.match(clean):
                # An emphasized sentence is not automatically a named task.
                # New nominal titles need their own terminal heading marker.
                if not (clean.endswith((".", ":")) or
                        re.match(r"[ \t]*[.:]", self._mask[end:])):
                    return
            if not self._at_heading_position(start):
                return
            line_start = self._mask.rfind("\n", 0, start) + 1
            prefix = self._mask[line_start:start]
            noindent = re.fullmatch(r"\s*(\\noindent\s*)", prefix)
            if noindent:
                start = line_start + noindent.start(1)
            seen.add(start)
            headings.append({"start": start, "title_end": end, "title": clean,
                             "category": category(clean),
                             "family": "case_marker" if re.match(r"Case\b", clean, re.I) else
                                       "barrier" if NONMATHEMATICAL_TITLE.match(clean) else family, **extra})

        cursor = 0
        while cursor < len(self._mask):
            try:
                token = self.lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok == "brace_open":
                group = self.lexer.group(token.pos)
                if group and group["balanced"] and re.match(r"\s*\\(?:bf|bfseries|it|itshape|gross)\b", group["value"]):
                    add(token.pos, group["end"], group["value"], "formatted_heading")
            elif token.tok == "macro" and token.arg in ("textbf", "textit", "emph", "textsc"):
                group = self.lexer.group(cursor)
                if group and group["balanced"]:
                    add(token.pos, group["end"], group["value"], "formatted_heading")
                    cursor = group["end"]
            elif token.tok == "macro" and token.arg in ("proclaim", "demo"):
                group = self.lexer.group(cursor)
                if token.arg == "demo" and group and group["balanced"]:
                    add(token.pos, group["end"], "Proof " + group["value"], "ams_plain_environment",
                        ending_command="enddemo")
                    cursor = group["end"]
                elif token.arg == "proclaim":
                    stop = self._mask.find(".", cursor)
                    if stop >= 0 and stop - cursor < 160:
                        add(token.pos, stop + 1, self._mask[cursor:stop + 1], "ams_plain_environment",
                            ending_command="endproclaim")
        for match in PLAIN_HEAD.finditer(self._mask):
            start = match.start(1)
            if not match[1][0].islower():
                add(start, match.end(1), match[1], "plain_heading")
        for match in re.finditer(r"(?im)^[ \t]*(Case\s+[A-Z]?\d+(?:\.\d+)*[.:](?!\d))", self._mask):
            if match[1][0].isupper():
                add(match.start(1), match.end(1), match[1], "case_marker")
        for match in PROOF_INTRO.finditer(self._mask):
            if not self._inside_formal(match.start()):
                headings.append({"start": match.start(), "title_end": match.end(),
                                 "title": "Proof of " + match[1], "category": "proof",
                                 "family": "explicit_proof_introduction", "target": _locator(match[1])})
        for match in PROOF_ANNOUNCEMENT.finditer(self._mask):
            if self._inside_formal(match.start()) or any(e["kind"] == "section_task" and
                    e["candidate"]["category_hint"] == "proof" and e["start"] <= match.start() < e["end"]
                    for e in self.events):
                continue
            line_end = self._mask.find("\n", match.end())
            line_end = len(self._mask) if line_end < 0 else line_end
            remainder = self._mask[match.end():line_end]
            if re.match(r"\s*(?:later\b|elsewhere\b|in\s+(?:Section|Appendix|Chapter)\b)", remainder, re.I):
                self.structural_cues.append({"structural_family": "proof_navigation_announcement",
                    "source_spans": self._segments([(match.start(), line_end)]),
                    "reason": "announces_proof_in_another_scope", "accepted": False})
                continue
            following = min((a for a in self._scope_barriers() if a > match.start()), default=len(self.text))
            formal = min((e["start"] for e in self.events if e["kind"] == "formal"
                          and e["start"] > match.start()), default=len(self.text))
            if not self._mask[line_end:min(following, formal)].strip():
                # A bare announcement immediately followed by a formal proof
                # is a discovery cue, not a duplicate proof region.
                self.structural_cues.append({"structural_family": "proof_navigation_announcement",
                    "source_spans": self._segments([(match.start(), line_end)]),
                    "reason": "no_separate_unheaded_proof_body", "accepted": False})
                continue
            headings.append({"start": match.start(), "title_end": match.end(),
                             "title": "Proof (author announcement)", "category": "proof",
                             "family": "direct_proof_announcement"})
        # Section/end-document barriers cannot become mathematical candidates.
        for event in self.project.structure_data.get("sections", []):
            interval = self._event_range(event)
            if interval:
                headings.append({"start": interval[0], "title_end": interval[1], "family": "barrier"})
        for event in self.project.structure_data.get("format_cues", []):
            title = _clean_title(event.get("raw_content", ""))
            if re.match(r"\d+[.]\s|case\b", title, re.I) or NONMATHEMATICAL_TITLE.match(title):
                interval = self._event_range(event)
                if interval and interval[0] not in seen:
                    headings.append({"start": interval[0], "title_end": interval[1],
                                     "title": title,
                                     "family": "case_marker" if re.match(r"case\b", title, re.I) else "barrier"})
        for token in re.finditer(r"\\(?:end\s*\{document\}|(?:bye|endinput)\b)", self._mask):
            headings.append({"start": token.start(), "title_end": token.end(), "family": "barrier"})
        for position in self._scope_barriers():
            headings.append({"start": position, "title_end": position, "family": "barrier"})
        return sorted(headings, key=lambda h: (h["start"], h["family"]))

    def _scan_body_end(self, heading, stop, ignored=None):
        """Top-level explicit task markers, with balanced groups/math preserved."""
        if heading.get("ending_command"):
            endmark = re.search(r"\\" + heading["ending_command"] + r"\b", self._mask[heading["title_end"]:stop])
            if endmark:
                return heading["title_end"] + endmark.end(), "explicit_plain_environment_end"
            return stop, "unresolved_missing_plain_environment_end"
        if heading["category"] == "proof":
            # Every handwritten-proof entry path shares the same boundary.
            # Explicit nested-task subtraction may cross those known tasks;
            # otherwise the next formal mathematical region is independent.
            stop = min(stop, min((e["start"] for e in self.events if e["kind"] == "formal"
                and heading["title_end"] <= e["start"] < stop
                and not any(a <= e["start"] < b for a, b in ignored or [])), default=stop))
        cursor, depth, math, envs, substantive = heading["title_end"], 0, None, [], False
        embedded = re.match(r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim)"
                            r"(?:\s+[A-Z]?\d+(?:\.\d+)*)?[.:]\s+\S", heading["title"], re.I)
        if not embedded and NAMED_TASK_RE.match(heading["title"]):
            task = _task_title_match(heading["title"])
            if task:
                # A nominal heading may itself contain the complete algorithm
                # assertion, as with a complete theorem formatting group.
                # Match an immediate task-number/punctuation separator rather
                # than a colon inside a mathematical title parameter.
                embedded = re.match(r"\s*(?:(?:[A-Z]?\d+(?:\.\d+)*|[A-Z])\s*)?"
                                    r"[.:]\s+\S", heading["title"][task.end():], re.I)
        if embedded:
            # A formatting group may hold the complete assertion, not only its
            # label.  Do not replace that author text with a following paragraph.
            after = self._mask[cursor:stop]
            if re.match(r"[ \t]*(?:\r?\n[ \t]*){2}|\s*\\(?:par|bigskip|medskip|smallskip|vspace)\b", after):
                return cursor, "assertion_inside_heading_group"
            substantive = True
        while cursor < stop:
            covered = next((b for a, b in ignored or [] if a <= cursor < b), None)
            if covered is not None:
                cursor = covered
                continue
            try:
                token = self.lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            # The lexer may advance over whitespace before returning a token.
            # Do not enter an excluded environment and then leave a phantom
            # open-environment state when its remaining source is skipped.
            covered = next((b for a, b in ignored or [] if a <= token.pos < b), None)
            if covered is not None:
                cursor = covered
                continue
            cursor = token.pos + token.len
            if token.tok == "brace_open":
                depth += 1
            elif token.tok == "brace_close":
                depth = max(0, depth - 1)
            elif token.tok in ("mathmode_inline", "mathmode_display", "mathmode_delimiter"):
                if math is None:
                    math = token.arg
                elif token.arg == math or (math, token.arg) in ((r"\(", r"\)"), (r"\[", r"\]")):
                    math = None
                substantive = True
            elif token.tok == "macro":
                name = token.arg
                if name in ("begin", "end"):
                    group = self.lexer.group(cursor)
                    if group and group["balanced"]:
                        if name == "begin":
                            envs.append(group["value"])
                        elif envs and envs[-1] == group["value"]:
                            envs.pop()
                        cursor = group["end"]
                if heading["category"] == "proof" and name in QED and not envs:
                    # Preserve the closing math delimiter after a printed box.
                    finish = cursor
                    if math:
                        suffix = re.match(r"[ \t]*(?:\$\$?|\\\)|\\\])", self._mask[finish:stop])
                        if suffix:
                            finish += suffix.end()
                    remaining = depth
                    while remaining:
                        closer = re.match(r"[ \t\r\n]*\}", self._mask[finish:stop])
                        if closer is None:
                            break
                        finish += closer.end()
                        remaining -= 1
                    if remaining:
                        return stop, "unresolved_qed_inside_unclosed_group"
                    return finish, "explicit_qed_marker"
                if heading["category"] != "proof" and name in LAYOUT and substantive and not depth and not math and not envs:
                    # A multipart statement may explicitly resume after spacing.
                    nextpos = cursor
                    if name in ("vspace", "vskip"):
                        arg = self.lexer.group(nextpos)
                        if arg:
                            nextpos = arg["end"]
                    following = self._mask[nextpos:stop].lstrip()
                    if re.match(r"(?:\([a-zivx]+\)|\\item)\s", following, re.I):
                        continue
                    if heading["category"] == "definition":
                        # Definitions frequently close a list, add vertical
                        # space, then finish by naming the just-defined family.
                        # Keep such explicit naming tails.  This bounded lexical
                        # adapter never decides which ordinary prose is useful.
                        block = re.split(r"\\(?:vspace|bigskip|medskip)\b", following, maxsplit=1)[0]
                        if (re.match(r"(?:If\b|Conversely\b|We\s+(?:denote|write|call)\b|"
                                     r"The\s+notation\b)", following, re.I)
                                and re.search(r"\b(?:we\s+(?:denote|write|call)|is\s+denoted)\b", block, re.I)):
                            continue
                    return token.pos, "top_level_statement_layout_separator"
            elif token.tok == "char" and token.arg.strip():
                substantive = True
        return stop, "next_explicit_task_or_section" if stop < len(self.text) else "document_end"

    def _trim_end(self, start, end):
        while end > start and self.text[end - 1].isspace():
            end -= 1
        # Drop only trailing layout commands/comments; mathematical braces are
        # never trimmed and source text inside the retained region is untouched.
        while True:
            tail = self._mask[start:end]
            match = re.search(r"(?:\\(?:bigskip|medskip|smallskip|goodbreak|par)\b|"
                              r"\\vspace\*?\s*\{[^{}]*\})[ \t\r\n]*$", tail)
            if not match:
                break
            end = start + match.start()
            while end > start and self.text[end - 1].isspace():
                end -= 1
        return end

    def _trim_start(self, start, end):
        """Remove only a detached component's leading whitespace/layout."""
        while start < end and self.text[start].isspace():
            start += 1
        while True:
            match = re.match(r"(?:\\(?:bigskip|medskip|smallskip|goodbreak|par)\b|"
                             r"\\vspace\*?\s*\{[^{}]*\})[ \t\r\n]*", self._mask[start:end])
            if not match:
                break
            start += match.end()
        return start

    def _example_intro_start(self, heading, formal_start):
        """Find an independent illustration paragraph immediately before Example.

        This is a bounded source-layout adapter.  It is only used at the next
        formal Example boundary and does not truncate proof Cases or arbitrary
        occurrences of the word 'example' inside mathematical reasoning.
        """
        example = next((e for e in self.events if e["kind"] == "formal" and
                        e["start"] == formal_start and re.match(r"Examples?\b",
                        e["candidate"]["title"], re.I)), None)
        if example is None:
            return None
        end = formal_start
        while end > heading["title_end"] and self._mask[end - 1].isspace():
            end -= 1
        breaks = list(re.finditer(r"\r?\n[ \t]*\r?\n", self._mask[heading["title_end"]:end]))
        start = heading["title_end"] + breaks[-1].end() if breaks else heading["title_end"]
        # Authors also place the independent introduction on the immediately
        # adjacent source line without a blank paragraph separator.
        starts = [start] + [start + m.end() for m in re.finditer(r"\r?\n", self._mask[start:end])]
        for beginning in starts:
            paragraph = self._mask[beginning:end].strip()
            if (re.match(r"(?:\\noindent\s*)?(?:We\s+(?:illustrate|give|present)\b|"
                         r"The\s+following\s+example\b|As\s+an\s+example\b)", paragraph, re.I)
                    and re.search(r"\bexamples?\b", paragraph, re.I)
                    and not re.search(r"\\(?:qed|qedhere|square|Box)\b", paragraph)
                    and not re.search(r"(?im)^[ \t]*(?:Case|Proof|Theorem|Lemma)\b", paragraph)):
                return beginning
        return None

    def _initial_proof_section(self, heading):
        """A proof section whose first body task is this explicit heading."""
        sections = [s for s in self._sections() if s["title_end"] <= heading["start"]]
        if not sections:
            return None
        section = sections[-1]
        target = _proof_identity(section["event"].get("title") or "")
        gap = self._mask[section["title_end"]:heading["start"]]
        gap = re.sub(r"\\label\s*\{[^{}\\]+\}|"
                     r"\\(?:noindent|par|smallskip|medskip|bigskip|goodbreak)\b", "", gap)
        return section if target and not gap.strip() else None

    def _compose_proof_section(self, event):
        """Keep a same-task section title as a cue rather than a duplicate node."""
        candidate = event["candidate"]
        spans = candidate["heading_source_spans"]
        self.structural_cues.append({"structural_family": "mathematical_section_container",
            "title": candidate["section_title"], "category_hint": "proof", "source_spans": spans,
            "original_text": [self.project.resolve_span(span) for span in spans],
            "scope_range": candidate["scope_range"], "event_id": candidate.get("event_id"),
            "activity": candidate["activity"], "accepted": False,
            "reason": "same_target_section_composed_with_explicit_proof_body"})
        self.candidates.remove(candidate)
        self.events.remove(event)

    def _proof_heading_targets(self, heading):
        """Bound an anonymous adjacent proof by its preceding task's own label.

        Labels owned by an equation or another nested environment do not name
        the enclosing theorem.  Without a direct label, numbered headings can
        still supply a literal locator; missing identity never licenses scope
        expansion.
        """
        explicit = _proof_identity(heading["title"])
        if explicit:
            return {explicit}
        if re.search(r"\bof\b", heading["title"], re.I):
            return set()
        section = self._initial_proof_section(heading)
        if section:
            return {_proof_identity(section["event"].get("title") or "")}
        previous = [e for e in self.events if e["end"] <= heading["start"]
                    and e["candidate"]["category_hint"] in ("statement", "definition")]
        if not previous:
            return set()
        adjacent = max(previous, key=lambda e: e["end"])
        gap = self._mask[adjacent["end"]:heading["start"]]
        gap = re.sub(r"\\(?:noindent|par|smallskip|medskip|bigskip|goodbreak)\b", "", gap)
        if gap.strip():
            return set()
        targets = set()
        locator = _locator(adjacent["candidate"]["title"])
        if locator:
            targets.add(("locator", locator))
        event_id = adjacent["candidate"].get("event_id")
        for label in self.project.structure_data.get("labels", []):
            interval = self._event_range(label)
            if (event_id and label.get("environment_id") == event_id and interval
                    and adjacent["start"] <= interval[0] < interval[1] <= adjacent["end"]
                    and not label.get("dynamic") and label.get("activity") != "inactive"):
                targets.update(("reference", key) for key in label.get("keys", []))
        return targets

    def _continued_section_proof(self, heading, headings):
        """Recognize an enclosing proof explicitly continued by same-task sections.

        The very next section must name the same task.  Each later sibling or
        ancestor section must also name it; an ordinary topic, different proof,
        appendix, bibliography or completed outer QED prevents continuation.
        Independently declared inner tasks keep their own complete source and
        are subtracted from the enclosing proof's source components.
        """
        targets = self._proof_heading_targets(heading)
        if not targets:
            return None
        sections = [s for s in self._sections() if s["start"] > heading["start"]
                    and not self._inside_formal(s["start"])]
        if not sections or _proof_identity(sections[0]["event"].get("title") or "") not in targets:
            return None
        first = sections[0]
        section_starts = {s["start"] for s in sections}
        hardstop = min([a for a in self._scope_barriers() if a > heading["start"]] + [h["start"] for h in headings
            if h["family"] == "barrier" and h["start"] not in section_starts
            and h["start"] > heading["start"]
            and NONMATHEMATICAL_TITLE.match(h.get("title", ""))], default=len(self.text))
        if first["start"] >= hardstop:
            return None
        level, stop = first["level"], hardstop
        continuation_sections = [first]
        for section in sections[1:]:
            if section["start"] >= stop:
                break
            same = _proof_identity(section["event"].get("title") or "") in targets
            if section["level"] <= level:
                if not same:
                    stop = section["start"]
                    break
                level = section["level"]
                continuation_sections.append(section)
        formal = [(e["start"], e["end"]) for e in self.events if e["kind"] == "formal"
                  and heading["start"] < e["start"] < stop]
        excluded = list(formal)
        # An explicitly headed inner unclosed task is still independent.  Its
        # next heading/formal region bounds it, while Case returns to the outer
        # proof.  This mirrors the ordinary handwritten-task boundary rule.
        for nested in headings:
            if not heading["start"] < nested["start"] < stop or nested["family"] in ("barrier", "case_marker"):
                continue
            nstop = min((h["start"] for h in headings if h["start"] > nested["start"]), default=stop)
            nstop = min(nstop, min((a for a, _ in formal if a > nested["start"]), default=stop), stop)
            nend, _ = self._scan_body_end(nested, nstop)
            if nend > nested["start"]:
                excluded.append((nested["start"], self._trim_end(nested["start"], nend)))
        # A descendant Definition/other proof section denotes an independent
        # task.  Matching continuation sections are retained as component
        # markers rather than duplicated as automatically qualified nodes.
        merged = []
        for event in self.events:
            if event["kind"] != "section_task" or event["start"] >= stop:
                continue
            initial_container = event["start"] < heading["start"] < event["end"]
            if event["start"] < first["start"] and not initial_container:
                continue
            if _proof_identity(event["candidate"].get("section_title") or "") in targets:
                merged.append(event)
            elif not initial_container:
                excluded.append((event["start"], event["end"]))
        _, initial_basis = self._scan_body_end(heading, first["start"], ignored=excluded)
        if initial_basis == "explicit_qed_marker":
            return None
        end, basis = self._scan_body_end(heading, stop, ignored=excluded)
        for event in merged:
            if event["start"] < end:
                self._compose_proof_section(event)
        basis = ("unresolved_same_target_proof_sections_" + basis.removeprefix("unresolved_")
                 if basis.startswith("unresolved_") else "same_target_proof_sections_" + basis)
        return end, basis, excluded, [
            s["event"].get("id") for s in continuation_sections if s["start"] < end], targets

    def _handwritten(self):
        headings = self._headings()
        nonbarrier = [h for h in headings if h["family"] not in ("barrier", "case_marker")]
        for index, heading in enumerate(nonbarrier):
            nexttask = min((h["start"] for h in headings if h["start"] > heading["start"]), default=len(self.text))
            formalstart = min((e["start"] for e in self.events if e["kind"] == "formal"
                              and e["start"] > heading["start"]), default=len(self.text))
            stop = min(nexttask, formalstart)
            excluded = []
            continuation = self._continued_section_proof(heading, headings) if heading["category"] == "proof" else None
            if continuation:
                end, basis, excluded, section_ids, targets = continuation
                heading["continuation_section_ids"] = section_ids
            elif heading["family"] == "explicit_proof_introduction":
                # Author explicitly introduces proof preparatory cases and later
                # resumes the same named task.  Keep this outer task contiguous
                # in reading scope but exclude independently headed subproofs.
                matching = next((h for h in nonbarrier[index + 1:] if h["category"] == "proof"
                    and h["family"] != "explicit_proof_introduction"
                    and _locator(h["title"]) == heading["target"]), None)
                barrier = min((h["start"] for h in headings if h["family"] == "barrier"
                               and h["start"] > heading["start"]), default=len(self.text))
                if matching and matching["start"] < barrier:
                    tailstop = min((h["start"] for h in headings if h["start"] > matching["start"]),
                                   default=len(self.text))
                    tailstop = min(tailstop, min((e["start"] for e in self.events if e["kind"] == "formal"
                                   and e["start"] > matching["start"]), default=len(self.text)))
                    stop, endbasis = self._scan_body_end(matching, tailstop)
                    heading["merged_tail_start"] = matching["start"]
                    for nested in nonbarrier[index + 1:]:
                        if nested["start"] >= matching["start"]:
                            break
                        nstop = min((h["start"] for h in headings if h["start"] > nested["start"]),
                                    default=matching["start"])
                        nend, _ = self._scan_body_end(nested, min(nstop, matching["start"]))
                        if nested["category"] in ("statement", "definition", "proof"):
                            excluded.append((nested["start"], self._trim_end(nested["start"], nend)))
                    end, basis = stop, "explicit_named_proof_components_" + endbasis
                else:
                    end, basis = self._scan_body_end(heading, stop)
            elif heading["category"] == "proof":
                # Handwritten outer proofs may introduce an independently
                # headed lemma, prove it, and explicitly resume their Cases.
                # The inner QED closes the inner task, not the enclosing proof.
                barrier = min((h["start"] for h in headings if h["family"] == "barrier"
                               and h["start"] > heading["start"]), default=len(self.text))
                following = [h for h in headings if heading["start"] < h["start"] < barrier]
                following.extend({"start": e["start"], "title_end": e["end"], "end": e["end"],
                                  "title": e["candidate"]["title"],
                                  "category": e["candidate"]["category_hint"],
                                  "family": "formal_task"}
                                 for e in self.events if e["kind"] == "formal"
                                 and heading["start"] < e["start"] < barrier)
                following.sort(key=lambda h: h["start"])
                tasks = [h for h in following if h["family"] != "case_marker"]
                active_outer = any(c["category_hint"] == "proof" and
                    c["scope_range"]["char_start"] < heading["start"] < c["scope_range"]["char_end"]
                    for c in self.candidates)
                if active_outer:
                    # In an already identified enclosing proof, the next Case
                    # explicitly returns to that enclosing task.  An unmarked
                    # inner proof (TAU024 Lemma 2) need not have its own QED.
                    end, basis = self._scan_body_end(heading, stop)
                elif tasks and re.match(r"(?:Lemma|Claim|Definition)\b", tasks[0].get("title", ""), re.I):
                    nested = tasks[0]
                    inner = next((h for h in tasks[1:] if h.get("category") == "proof"), None)
                    resume = next((h for h in following if h["family"] == "case_marker"
                                   and inner and h["start"] > inner["start"]), None)
                    completed_end, completed_basis = self._scan_body_end(heading, nested["start"])
                    if completed_basis == "explicit_qed_marker":
                        end, basis = completed_end, completed_basis
                    elif inner and resume:
                        nested_end = nested["end"] if nested["family"] == "formal_task" else self._scan_body_end(nested, inner["start"])[0]
                        inner_end = inner["end"] if inner["family"] == "formal_task" else self._scan_body_end(inner, resume["start"])[0]
                        excluded = [(nested["start"], self._trim_end(nested["start"], nested_end)),
                                    (inner["start"], self._trim_end(inner["start"], inner_end))]
                        next_main = min((h["start"] for h in following if h["start"] > resume["start"]
                            and h["family"] != "case_marker"), default=barrier)
                        end, endbasis = self._scan_body_end(heading, next_main, ignored=excluded)
                        basis = "outer_proof_resumes_cases_" + endbasis
                    else:
                        end, basis = self._scan_body_end(heading, stop)
                else:
                    # Case headings are internal proof structure, not new tasks.
                    next_main = min((h["start"] for h in following if h["family"] != "case_marker"),
                                    default=barrier)
                    end, basis = self._scan_body_end(heading, next_main)
            else:
                end, basis = self._scan_body_end(heading, stop)
            if heading["category"] == "proof":
                intro = self._example_intro_start(heading, end)
                if intro is not None:
                    end, basis = intro, "next_formal_example_excluding_independent_illustration_intro"
            # A named tail already retained by its preparatory outer proof is
            # a component marker, not a second duplicate proof node.
            if any(c.get("merged_tail_start") == heading["start"] for c in self.candidates):
                continue
            if continuation:
                # Within the author's explicitly continued proof envelope,
                # layout tokens belong to the original source just as Cases
                # do.  Only empty whitespace around detached components may
                # be discarded, not the author's intervening source commands.
                while end > heading["start"] and self.text[end - 1].isspace():
                    end -= 1
            else:
                end = self._trim_end(heading["start"], end)
            if end <= heading["start"]:
                self._issue("empty_handwritten_task", interval=(heading["start"], heading["title_end"]))
                continue
            intervals, first = [], heading["start"]
            excluded = [(max(heading["start"], a), min(end, b)) for a, b in excluded
                        if max(heading["start"], a) < min(end, b)]
            for a, b in sorted(excluded):
                if first < a:
                    intervals.append((first, a))
                first = max(first, b)
            if first < end:
                intervals.append((first, end))
            if excluded:
                components = []
                for a, b in intervals:
                    if continuation:
                        while a < b and self.text[a].isspace():
                            a += 1
                        while a < b and self.text[b - 1].isspace():
                            b -= 1
                    else:
                        a = self._trim_start(a, b)
                        b = self._trim_end(a, b)
                    if a < b:
                        components.append((a, b))
                intervals = components
            status = "unresolved" if basis.startswith("unresolved") else "complete_explicit_task_markers"
            candidate = self._candidate(heading["start"], end, title=heading["title"], family=heading["family"],
                    boundary_status=status, basis=basis, intervals=intervals,
                    heading_source_spans=self._segments([(heading["start"], heading["title_end"])]),
                    **({"merged_tail_start": heading["merged_tail_start"]} if "merged_tail_start" in heading else {}),
                    **({"continuation_section_ids": heading["continuation_section_ids"]}
                       if "continuation_section_ids" in heading else {}))
            if continuation:
                hint = candidate["proof_attachment_hint"]
                hint["basis"] = "explicit_same_target_proof_sections"
                hint["target_reference_keys"] = sorted(value for kind, value in targets if kind == "reference")
                hint["target_locator"] = next((value for kind, value in targets if kind == "locator"), None)
                if hint["target_locator"] is None and hint["target_reference_keys"]:
                    hint["target_locator"] = "reference:" + hint["target_reference_keys"][0]
            if heading["category"] == "proof":
                section = self._initial_proof_section(heading)
                if section and _proof_identity(section["event"].get("title") or "") in self._proof_heading_targets(heading):
                    for event in list(self.events):
                        if event["kind"] == "section_task" and event["candidate"].get("event_id") == section["event"].get("id"):
                            self._compose_proof_section(event)
            self.events.append({"kind": "handwritten", "start": heading["start"], "end": end, "candidate": candidate})
            if status == "unresolved":
                self._issue(basis, interval=(heading["start"], end))

    def _nesting_and_attachment(self):
        direct_labels = {}
        for label in self.project.structure_data.get("labels", []):
            owner = label.get("environment_id")
            if owner and not label.get("dynamic") and label.get("activity") != "inactive":
                direct_labels.setdefault(owner, set()).update(label.get("keys", []))
        for candidate in self.candidates:
            scope = candidate["scope_range"]
            containers = [c for c in self.candidates if c is not candidate and
                c["scope_range"]["char_start"] <= scope["char_start"] and
                scope["char_end"] <= c["scope_range"]["char_end"] and c["scope_range"] != scope]
            if containers:
                candidate["parent_candidate_id"] = min(containers, key=lambda c:
                    c["scope_range"]["char_end"] - c["scope_range"]["char_start"])["candidate_id"]
            hint = candidate.get("proof_attachment_hint")
            if hint is None:
                continue
            if hint.get("target_reference_keys"):
                keys = set(hint["target_reference_keys"])
                targets = [c for c in self.candidates if c["category_hint"] != "proof"
                           and keys.intersection(direct_labels.get(c.get("event_id"), set()))]
            elif hint["target_locator"]:
                targets = [c for c in self.candidates if c["category_hint"] != "proof"
                           and _locator(c["title"]) == hint["target_locator"]]
            elif hint.get("target_kind"):
                targets = [c for c in self.candidates if c["category_hint"] != "proof"
                           and re.match(re.escape(hint["target_kind"]) + r"\b", c["title"], re.I)]
            else:
                previous = [c for c in self.candidates if c["category_hint"] in ("statement", "definition")
                            and c["scope_range"]["char_end"] <= scope["char_start"]]
                targets = [max(previous, key=lambda c: c["scope_range"]["char_end"])] if previous else []
            hint["target_candidate_ids"] = [c["candidate_id"] for c in targets]

    def extract(self) -> dict[str, Any]:
        self._formal()
        self._wrappers()
        self._section_tasks()
        self._handwritten()
        self.candidates.sort(key=lambda c: (c["scope_range"]["char_start"], c["scope_range"]["char_end"], c["candidate_id"]))
        self._nesting_and_attachment()
        mapped = {row["source"]["file"] for row in self.project.source_map}
        for filename, text in self.project._texts.items():
            if filename not in mapped and not filename.endswith((".sty", ".cls", ".bib", ".bbl")) and text.strip():
                self.unresolved.append({"reason": "untraversed_author_source", "source_spans": [
                    {"file": filename, "char_start": 0, "char_end": len(text)}],
                    "activity": "unmapped_requires_source_review"})
        self.project.assert_unchanged()
        return {"schema_version": SCHEMA, "rule_version": RULE_VERSION,
                "paper_id": self.project.paper_id, "source_fingerprint": self.project.fingerprint,
                "candidates": self.candidates, "unresolved": self.unresolved,
                "structural_cues": self.structural_cues,
                "summary": {"candidate_count": len(self.candidates),
                    "by_family": dict(Counter(c["structural_family"] for c in self.candidates)),
                    "by_category_hint": dict(Counter(str(c["category_hint"]) for c in self.candidates)),
                    "unresolved_count": len(self.unresolved)},
                "paid_model_calls": 0, "accepted_graph": False,
                "claim": "Explicit source regions only; no mathematical acceptance or dependency edges."}


def extract_rule_nodes(project):
    return RuleNodeExtractor(project).extract()
