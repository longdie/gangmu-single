"""Versioned source-owner fixes, preserving the frozen v1 scanner verbatim.

Only candidate source envelopes change. No Unit, dependency, TeX counter or
mathematical truth is inferred. The bounded repeat-environment grammar is the
same literal newreptheorem factory supported by v1 reference discovery.
"""
from __future__ import annotations
import copy
import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from gangmu.representation.model import digest
from gangmu.latex_structure import PRIMITIVE_CONDITIONALS
from . import explicit_reference_inventory as frozen
from . import inventory as historical

VERSION = "gangmu.e4.explicit-reference-inventory.v2"
LITERAL_RESULT_TITLE = re.compile(r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|"
    r"Assumption|Remark|Example|Notation|Question)", re.I)


class ExplicitReferenceInventoryV2(frozen.ExplicitReferenceInventory):
    def __init__(self, project):
        super().__init__(project)
        self.digests["reference"] = digest({"version": VERSION,
            "source": project.fingerprint, "records": self.references})

    def _repeat_result_owners(self):
        if hasattr(self, "_v2_repeat_owners"):
            return self._v2_repeat_owners
        template = (r"\newenvironment{rep#1}[1]{\def\rep@title{#2\ref{##1}}"
                    r"\begin{rep@theorem}}{\end{rep@theorem}}")
        factories = {definition["id"]: definition for definition in self.definitions.values()
            if re.sub(r"\s+", "", frozen._without_comments(definition.get("body") or "")) == template
            and definition.get("argument_count") == 2 and definition.get("optional_default") is None
            and definition.get("activity") == "active" and definition.get("scope") == ["document"]}
        declarations = {}
        for call in self.project.structure_data.get("macro_calls", []):
            definition = factories.get(call.get("definition_id"))
            if not definition or call.get("activity") != "active" or call.get("condition_ids"):
                continue
            source = call["source"]
            lexer = frozen.LiteralReferenceLexer(self.project._texts[source["file"]])
            token = lexer.token(source["char_start"])
            values, _, valid = self._macro_arguments(lexer, token.pos + token.len, definition)
            if (valid and len(values) == 2 and re.fullmatch(r"[A-Za-z0-9_-]+", values[0])
                    and LITERAL_RESULT_TITLE.fullmatch(values[1].strip())):
                declarations.setdefault("rep" + values[0], []).append({
                    "definition": definition, "call": call, "title": values[1].strip()})
        owners = []
        for environment in self.project.structure_data.get("environments", []):
            candidates = declarations.get(environment.get("name"), [])
            if (len(candidates) != 1 or not environment.get("closed") or not environment.get("full_source")
                    or environment.get("activity") != "active" or environment.get("condition_ids")):
                continue
            declared = candidates[0]
            position, declaration_position = self._positions(environment["source"]), self._positions(declared["call"]["source"])
            if not position or not declaration_position or max(declaration_position) >= min(position):
                continue
            span = historical._span(environment["full_source"], environment.get("file_occurrence"))
            owners.append({"event_id": environment["id"], "category_hint": self._category(declared["title"]),
                "source_spans": self._record("owner", environment)["source_spans"],
                "boundary_status": "complete_static_span", "ownership_basis": "bounded_literal_repeat_result_factory_requires_review",
                "factory_title": declared["title"], "environment_name": environment["name"],
                "full_source": span, "activity": environment.get("activity", "unknown"),
                "factory_definition_spans": [historical._span(declared["definition"]["source"]),
                                             historical._span(declared["call"]["source"])]})
        self._v2_repeat_owners = owners
        return owners

    def _body_render_boundaries(self, file):
        if not hasattr(self, "_v2_render_boundaries"):
            self._v2_render_boundaries = {}
        if file in self._v2_render_boundaries:
            return self._v2_render_boundaries[file]
        text, cursor, boundaries = self.project._texts[file], 0, []
        hidden = sorted(self._excluded(file) + [event["source"] for event in
            self.project.structure_data.get("deferred_code", []) if event["source"]["file"] == file],
            key=lambda row: row["char_start"])
        lexer = historical._Lexer(text)
        conditions = []
        custom_conditions = {definition.get("name", "").lstrip("\\") for definition in self.definitions.values()
            if definition.get("definition_kind") == "conditional"}
        condition_commands = PRIMITIVE_CONDITIONALS | custom_conditions | {
            "ifpdf", "ifPDFTeX", "ifXeTeX", "ifLuaTeX", "ifxetex", "ifluatex"}
        while cursor < len(text):
            excluded = next((row for row in hidden if row["char_start"] <= cursor < row["char_end"]), None)
            if excluded is not None:
                cursor = excluded["char_end"]
                continue
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if any(row["char_start"] <= token.pos < row["char_end"] for row in hidden):
                continue
            if token.tok == "macro":
                if token.arg in condition_commands or token.arg.startswith("if@"):
                    value = True if token.arg == "iftrue" else False if token.arg == "iffalse" else None
                    conditions.append({"value": value, "branch": value, "else_seen": False})
                elif token.arg == "else" and conditions:
                    frame = conditions[-1]
                    frame["branch"] = (not frame["value"] if frame["value"] is not None
                        and not frame["else_seen"] else None)
                    frame["else_seen"] = True
                elif token.arg == "or" and conditions:
                    conditions[-1]["branch"] = None
                elif token.arg == "fi" and conditions:
                    conditions.pop()
                elif token.arg in {"bibliography", "printbibliography"} and all(
                        frame["branch"] is True for frame in conditions):
                    boundaries.append({"file": file, "char_start": token.pos,
                        "char_end": cursor, "basis": "literal_bibliography_rendering_command"})
        for event in self.project.structure_data.get("environments", []):
            if event.get("name") == "document" and event.get("closed") and event.get("end_source", {}).get("file") == file:
                boundaries.append({**historical._span(event["end_source"], event.get("file_occurrence")),
                                   "basis": "literal_document_closing_start"})
        self._v2_render_boundaries[file] = boundaries
        return boundaries

    def _targets(self, keys, dynamic=False):
        rows = super()._targets(keys, dynamic)
        for target in rows:
            new_owners = []
            for owner in self._repeat_result_owners():
                span = owner["full_source"]
                if any(label["file"] == span["file"] and label.get("occurrence") == span.get("occurrence")
                       and span["char_start"] <= label["char_start"] < label["char_end"] <= span["char_end"]
                       for label in target.get("label_spans", [])):
                    new_owners.append(copy.deepcopy(owner))
            if new_owners:
                for candidate in target.get("label_candidates", []):
                    label = candidate["label_span"]
                    for owner in new_owners:
                        span = owner["full_source"]
                        if (label["file"] == span["file"] and label.get("occurrence") == span.get("occurrence")
                                and span["char_start"] <= label["char_start"] < label["char_end"] <= span["char_end"]
                                and owner["event_id"] not in candidate["mathematical_event_ids"]):
                            candidate["mathematical_event_ids"].append(owner["event_id"])
                            candidate["math_owners"].append(copy.deepcopy(owner))
                target["math_owners"].extend(owner for owner in new_owners if not any(
                    old["event_id"] == owner["event_id"] for old in target["math_owners"]))
                target["owner_source_spans"] = [span for owner in target["math_owners"] for span in owner["source_spans"]]
                target["target_spans"] = copy.deepcopy(target["owner_source_spans"])
                target["target_event_ids"] = [owner["event_id"] for owner in target["math_owners"]]
                if target["resolution"] == "non_mathematical_or_unclassified":
                    target["resolution"] = "mathematical_owner_candidates_require_review"
                target["candidate_adapter_version"] = VERSION
            candidates = target.get("source_section_candidates", [])
            changed = False
            for candidate in candidates:
                for span in candidate["source_spans"]:
                    boundaries = [boundary for boundary in self._body_render_boundaries(span["file"])
                        if (boundary.get("occurrence") is None or boundary.get("occurrence") == span.get("occurrence"))
                        and span["char_start"] < boundary["char_start"] < span["char_end"]]
                    if boundaries:
                        boundary = min(boundaries, key=lambda row: row["char_start"])
                        span["char_end"] = boundary["char_start"]
                        candidate["body_end_boundary"] = copy.deepcopy(boundary)
                        changed = True
            if changed:
                if not target["math_owners"]:
                    target["target_spans"] = [copy.deepcopy(span) for candidate in candidates for span in candidate["source_spans"]]
                target["candidate_adapter_version"] = VERSION
        return rows


def scan_references(project):
    """Full v2 ledger; unchanged occurrences keep their v1 ID and raw fields."""
    return copy.deepcopy(ExplicitReferenceInventoryV2(project).references)
