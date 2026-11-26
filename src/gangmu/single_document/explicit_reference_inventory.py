"""Literal reference scanning for the explicit-only E4 continuation.

Historical E1 and node inventories remain unchanged. This adapter preserves a
single ref/eqref key containing commas, parses protected cleveref key lists,
and rebuilds the same source label owners under those exact literal keys.
No TeX is executed, no mathematical dependency is inferred and no node is made.
"""
from __future__ import annotations

import copy
import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from gangmu.latex_structure import normalize_key
from gangmu.representation.model import digest
from . import inventory as historical

VERSION = "gangmu.e4.explicit-reference-inventory.v1"
LIST_COMMANDS = {"cref", "Cref", "cpageref", "Cpageref", "labelcref",
    "labelcpageref", "namecrefs", "nameCrefs"}
RESULT_NAMES = r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|Equation)"
COMPOSITE_LOCATOR = re.compile(r"\b(" + RESULT_NAMES
    + r"(?:(?:\s|~)*(?:-|/|and|&)(?:\s|~)*" + RESULT_NAMES + r")+)"
    + r"(?:\s|~)+\(?([A-Z]?\d+(?:\.\d+)*[a-z]?)\)?(?![0-9A-Za-z]|\.[0-9])", re.I)
SECTION_LEVELS = {name: level for level, name in enumerate(
    ("part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"))}


def _without_comments(value: str) -> str:
    """Remove TeX comments before interpreting keys, retaining escaped tokens.

    A comment consumes its line ending; leading whitespace on the next physical
    line is ignored by TeX. Escaped percent signs remain macro-bearing keys and
    therefore keep the existing dynamic/unknown treatment.
    """
    result, pos = [], 0
    while pos < len(value):
        if value[pos] == "\\":
            result.append(value[pos:pos + 2]); pos += 2
        elif value[pos] == "%":
            end = value.find("\n", pos)
            pos = len(value) if end < 0 else end + 1
            while pos < len(value) and value[pos] in " \t\r":
                pos += 1
        else:
            result.append(value[pos]); pos += 1
    return "".join(result)


def _list_keys(value: str) -> list[str]:
    """Split only top-level commas; remove balanced protection around each key."""
    value = _without_comments(value)
    parts, start, depth, pos = [], 0, 0, 0
    while pos < len(value):
        if value[pos] == "\\":
            pos += 2
            continue
        if value[pos] == "{":
            depth += 1
        elif value[pos] == "}":
            depth -= 1
        elif value[pos] == "," and depth == 0:
            parts.append(value[start:pos]); start = pos + 1
        pos += 1
    parts.append(value[start:])
    result = []
    for part in parts:
        key = part.strip()
        while key.startswith("{"):
            group = historical._Lexer(key).group(0)
            if not group or not group["balanced"] or group["end"] != len(key):
                break
            key = group["value"].strip()
        result.append(normalize_key(key))
    return result


class LiteralReferenceLexer(historical._Lexer):
    def reference(self, name, pos):
        cursor = self.skip(pos)
        if cursor < len(self.text) and self.text[cursor] == "*":
            cursor += 1
        options = []
        while (argument := self.group(cursor, optional=True)) is not None:
            options.append(argument); cursor = argument["end"]
        arguments = []
        for _ in range(2 if name in historical.RANGE_COMMANDS else 1):
            argument = self.group(cursor)
            if argument is None:
                break
            arguments.append(argument); cursor = argument["end"]
        values = ([options[0]["value"]] if name == "hyperref" and options
                  else [x["value"] for x in arguments])
        keys = [key for value in values for key in
                (_list_keys(value) if name in LIST_COMMANDS else [normalize_key(_without_comments(value))])]
        expected = 2 if name in historical.RANGE_COMMANDS else 1
        status = ("supported" if len(arguments) == expected and all(
            x["balanced"] for x in arguments + options) else "unknown")
        return cursor, keys, status


class ExplicitReferenceInventory(historical.SourceInventory):
    def __init__(self, project):
        # Correct discovery metadata on a private view, never the pinned source
        # or the historical project passed by the caller.
        view = copy.copy(project)
        view.structure_data = {**project.structure_data,
            "labels": copy.deepcopy(project.structure_data.get("labels", []))}
        view._labels = {}
        for label in view.structure_data["labels"]:
            source = label["source"]
            lexer = LiteralReferenceLexer(project._texts[source["file"]])
            token = lexer.token(source["char_start"])
            group = lexer.group(token.pos + token.len)
            if token.arg == "label" and group and group["balanced"]:
                value = _without_comments(group["value"])
                label["keys"] = [normalize_key(value)]
                label["dynamic"] = bool(re.search(r"[\\#]", value))
            for key in label.get("keys", []):
                view._labels.setdefault(key, []).append(label)
        super().__init__(view)
        self.digests["reference"] = digest({"version": VERSION,
            "source": project.fingerprint, "records": self.references})

    def _targets(self, keys, dynamic=False):
        rows = super()._targets(keys, dynamic)
        sections = self.project.structure_data.get("sections", [])
        for target in rows:
            if target.get("math_owners") or target["resolution"] in {"dynamic", "unresolved"}:
                continue
            candidates = []
            for label in target.get("label_spans", []):
                text = self.project._texts[label["file"]]
                preceding = [section for section in sections
                    if section["source"]["file"] == label["file"]
                    and section.get("file_occurrence") == label.get("occurrence")
                    and section["source"]["char_end"] <= label["char_start"]]
                if not preceding:
                    continue
                section = max(preceding, key=lambda row: row["source"]["char_start"])
                between = text[section["source"]["char_end"]:label["char_start"]]
                if re.sub(r"\\label\s*\{[^{}]*\}|\s+", "", _without_comments(between)):
                    continue
                level = SECTION_LEVELS.get(section["command"])
                if level is None:
                    continue
                following = [row["source"]["char_start"] for row in sections
                    if row["source"]["file"] == label["file"]
                    and row.get("file_occurrence") == label.get("occurrence")
                    and row["source"]["char_start"] > section["source"]["char_start"]
                    and SECTION_LEVELS.get(row["command"], -1) <= level]
                ends = [row["full_source"]["char_start"] for row in self.project.structure_data.get("environments", [])
                    if row.get("name") == "thebibliography" and row.get("full_source")
                    and row["full_source"]["file"] == label["file"]
                    and row["full_source"]["char_start"] > section["source"]["char_start"]]
                document = [row["full_source"]["char_end"] for row in self.project.structure_data.get("environments", [])
                    if row.get("name") == "document" and row.get("full_source")
                    and row["full_source"]["file"] == label["file"]
                    and row["full_source"]["char_end"] > section["source"]["char_start"]]
                end = min(following + ends + document + [len(text)])
                span = {**label, "char_start": section["source"]["char_start"], "char_end": end}
                candidates.append({"event_id": section["id"], "command": section["command"],
                    "title": section["title"], "heading_span": historical._span(section["source"], section.get("file_occurrence")),
                    "label_span": copy.deepcopy(label), "source_spans": [span], "category_hint": None,
                    "ownership_basis": "literal_adjacent_section_label_source_envelope_requires_review",
                    "mathematical_status": "not_determined_by_syntax"})
            if candidates:
                # A source section is an endpoint-reading envelope, not a
                # mathematical Unit/owner. Organizational headings remain
                # non-mathematical or unclassified; label prefixes decide none.
                target["source_section_candidates"] = candidates
                target["target_spans"] = [span for row in candidates for span in row["source_spans"]]
                target["target_event_ids"] = [row["event_id"] for row in candidates]
        return rows

    def _references(self):
        rows = super()._references()
        composites = {file: {match.end(): match for match in COMPOSITE_LOCATOR.finditer(text)}
                      for file, text in self.project._texts.items()}
        for index, row in enumerate(rows):
            if row["basis"] != "handwritten_numbered_reference":
                continue
            source = row["span"]
            match = composites[source["file"]].get(source["char_end"])
            if match is None or not match.start() < source["char_start"]:
                continue
            kinds = [value.lower() for value in re.findall(RESULT_NAMES, match[1], re.I)]
            expanded = {**source, "char_start": match.start()}
            event = {"source": expanded, "file_occurrence": source.get("occurrence"),
                "command": row["command"], "activity": row["activity"]}
            repaired = self._reference_record(event, row["keys"], row["syntax_status"], basis=row["basis"],
                locator={**row["locator"], "kinds": kinds}, unresolved_reason=row["unresolved_reason"])
            repaired["targets"] = copy.deepcopy(row["targets"])
            repaired["prior_item_id"] = row["item_id"]
            repaired["complete_composite_locator_anchor"] = True
            rows[index] = repaired
        for row in rows:
            if row["basis"] not in {"literal_reference", "literal_reference_alias"}:
                continue
            source = row["span"]
            lexer = LiteralReferenceLexer(self.project._texts[source["file"]])
            token = lexer.token(source["char_start"])
            command = row.get("projected_command", row["command"])
            if command not in historical.REFERENCE_COMMANDS:
                continue
            end, keys, status = lexer.reference(command, token.pos + token.len)
            if end != source["char_end"]:
                raise ValueError("Literal scanner changed original reference boundaries")
            if row["basis"] == "literal_reference_alias" and row["syntax_status"] != "supported":
                status = "unknown"
            row["prior_item_id"] = row["item_id"]
            row["keys"], row["syntax_status"] = keys, status
            dynamic = any(re.search(r"[\\#]", key) for key in keys)
            row["targets"] = self._targets(keys, dynamic)
            row["resolution"] = ("dynamic" if dynamic else
                "unknown" if not keys or status != "supported" else
                "ambiguous" if any(t["resolution"] == "ambiguous" for t in row["targets"]) else
                "unresolved" if any(t["resolution"] == "unresolved" for t in row["targets"]) else "resolved")
            row["item_id"] = "reference:" + digest({"version": VERSION,
                "source": self.project.fingerprint, "span": row["span"],
                "basis": row["basis"], "keys": keys})[:24]
            row["literal_key_adapter"] = VERSION
        rows.extend(self._factory_environment_references())
        rows.extend(self._case_locators(rows))
        rows.sort(key=lambda r: (r["reading_ranges"][0]["char_start"] if r["reading_ranges"] else float("inf"),
                                 r["span"]["file"], r["span"]["char_start"], r["item_id"]))
        return rows

    def _case_locators(self, rows):
        """Retain intra-proof numbered case citations as syntax, without nodes.

        Whether a case is already contained in the user's complete proof is a
        later ownership/use decision. Discovering it never creates a self edge.
        """
        token = r"(?:\d+|[IVXLCDM]+|[A-Z])\*{0,2}(?:(?:\s|~)*\([a-z]\))?(?![A-Za-z0-9])"
        pattern = re.compile(r"\b(?i:Cases?)(?:\s|~)+(" + token
            + r"(?:(?:\s|~)*(?:and|,|--?|–|to)(?:\s|~)*" + token + r")*)")
        added = []
        for file, text in self.project._texts.items():
            if file.endswith((".sty", ".cls", ".bib", ".bbl")):
                continue
            hidden = list(self._excluded(file)) + [r["span"] for r in rows if r["span"]["file"] == file]
            for match in pattern.finditer(text):
                source = {"file": file, "char_start": match.start(), "char_end": match.end()}
                if any(historical._overlaps(source, span) for span in hidden):
                    continue
                numbers = [re.sub(r"\s|~", "", item.group(0)) for item in re.finditer(token, match[1])]
                for occurrence in self._occurrences(source):
                    event = {"source": source, "file_occurrence": occurrence,
                        "command": "literal_case_locator", "activity": "requires_activity_review"}
                    added.append(self._reference_record(event, ["text:case:" + n for n in numbers], "unknown",
                        basis="handwritten_case_reference", locator={"kind": "case", "numbers": numbers},
                        unresolved_reason="Numbered proof case locator; headings, navigation and intra-unit uses require ownership review, not a new Case node."))
        return added

    def _factory_environment_references(self):
        """Recognize a literal repeat-result heading factory without running it.

        The bounded grammar below is the actual common newreptheorem factory.
        Its title's ref argument is used at an environment begin, not at the
        preamble factory declaration. Other factory programs stay unknown.
        """
        template = (r"\newenvironment{rep#1}[1]{\def\rep@title{#2\ref{##1}}"
                    r"\begin{rep@theorem}}{\end{rep@theorem}}")
        factories = {}
        for definition in self.definitions.values():
            body = re.sub(r"\s+", "", _without_comments(definition.get("body") or ""))
            if (body == template and definition.get("argument_count") == 2
                    and definition.get("optional_default") is None
                    and definition.get("activity") == "active"
                    and definition.get("scope") == ["document"]):
                factories[definition["id"]] = definition
        declarations = {}
        for call in self.project.structure_data.get("macro_calls", []):
            definition = factories.get(call.get("definition_id"))
            if not definition or call.get("activity") != "active" or call.get("condition_ids"):
                continue
            source = call["source"]
            lexer = LiteralReferenceLexer(self.project._texts[source["file"]])
            token = lexer.token(source["char_start"])
            values, end, valid = self._macro_arguments(lexer, token.pos + token.len, definition)
            if valid and len(values) == 2 and re.fullmatch(r"[A-Za-z0-9_-]+", values[0]):
                name = "rep" + values[0]
                declarations.setdefault(name, []).append({"definition": definition, "call": call})
        rows = []
        for environment in self.project.structure_data.get("environments", []):
            matches = declarations.get(environment.get("name"), [])
            if len(matches) != 1:
                continue
            declaration = matches[0]
            position = self._positions(environment["source"])
            declared_position = self._positions(declaration["call"]["source"])
            if not position or not declared_position or max(declared_position) >= min(position):
                continue
            source = environment["source"]
            lexer = LiteralReferenceLexer(self.project._texts[source["file"]])
            group = lexer.group(source["char_end"])
            if not group:
                continue
            key = normalize_key(_without_comments(group["value"]))
            event = {**environment, "source": {**source, "char_end": group["end"]}, "command": "begin"}
            rows.append(self._reference_record(event, [key], "supported" if group["balanced"] else "unknown",
                basis="literal_factory_environment_reference", projected_command="ref",
                environment_name=environment["name"],
                definition_spans=[historical._span(declaration["definition"]["source"]),
                                  historical._span(declaration["call"]["source"])],
                projection_scope="Literal heading reference only; not a dependency or environment execution."))
        return rows

    def _project_macro(self, definition, values, seen=None):
        # Same bounded wrapper grammar as the historical scanner; only literal
        # reference argument parsing differs. Unsafe execution stays unknown.
        seen = [] if seen is None else seen
        if definition["id"] in seen or len(seen) >= 8:
            return None
        body = definition.get("body") or ""
        if (definition.get("activity") != "active" or definition.get("delimited_parameters")
                or definition.get("xparse_signature") or historical.UNSAFE_MACRO.search(body)
                or re.search(r"##|#[^1-9]", body)
                or any(int(n) > len(values) for n in re.findall(r"#([1-9])", body))):
            return None
        expanded = re.sub(r"#([1-9])", lambda m: values[int(m[1]) - 1], body)
        lexer, cursor, found = LiteralReferenceLexer(expanded), 0, []
        while cursor < len(expanded):
            try:
                token = lexer.token(cursor)
            except LatexWalkerEndOfStream:
                break
            cursor = token.pos + token.len
            if token.tok != "macro":
                continue
            if token.arg in historical.REFERENCE_COMMANDS:
                cursor, keys, status = lexer.reference(token.arg, cursor)
                found.append({"command": token.arg, "keys": keys,
                    "syntax_status": status, "macro_chain": seen + [definition["id"]]})
            else:
                definitions = [d for d in self.definitions.values() if d.get("name") == token.arg
                               and d.get("definition_kind") == "macro"]
                if len(definitions) > 1:
                    return None
                nested = definitions[0] if definitions else None
                if nested:
                    args, cursor, valid = self._macro_arguments(lexer, cursor, nested)
                    projected = self._project_macro(nested, args, seen + [definition["id"]]) if valid else None
                    if projected:
                        found.extend(projected)
                    elif nested.get("contains_structure") or "ref" in str(nested.get("body", "")).lower():
                        return None
                elif token.arg not in historical.RENDER_ARGUMENT_COMMANDS | {"protect", "relax"}:
                    argument = lexer.group(cursor)
                    if argument and re.search(r"\\(?:" + "|".join(sorted(historical.REFERENCE_COMMANDS))
                            + r")\b", argument["value"]):
                        return None
        return found


def scan_references(project):
    """Return the full ledger, including external, unresolved and literal clues."""
    return copy.deepcopy(ExplicitReferenceInventory(project).references)
