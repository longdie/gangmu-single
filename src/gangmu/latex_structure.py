"""Conservative, source-preserving LaTeX structure extraction (not a TeX engine).

Uses pylatexenc's token lexer and explicit balanced argument/scope handling.
Offsets always refer to immutable author sources, not compiled/repaired copies.
"""
from __future__ import annotations

import bisect
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

import pylatexenc
from pylatexenc.latexwalker import LatexWalker, LatexWalkerEndOfStream


SCHEMA = "gangmu.latex-structure.v1"
SECTIONS = {"part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"}
REFS = {"ref", "eqref", "pageref", "autoref", "nameref", "cref", "Cref", "vref", "Vref", "hyperref", "thref", "thnameref", "fullref", "cpageref", "Cpageref"}
CITES = {"cite", "citep", "citet", "citealt", "citealp", "citeauthor", "citeyear", "parencite", "textcite", "autocite", "nocite", "Cite", "cites", "footcite"}
VERBATIM = {"verbatim", "verbatim*", "Verbatim", "BVerbatim", "lstlisting", "minted", "comment", "filecontents", "filecontents*"}
MATH = {"equation", "equation*", "align", "align*", "alignat", "alignat*", "gather", "gather*", "multline", "multline*", "displaymath", "math", "eqnarray", "eqnarray*", "split"}
THEOREMS = {"theorem", "lemma", "proposition", "corollary", "definition", "remark", "example", "conjecture", "claim", "proof"}
COMMAND_DEFS = {"newcommand", "renewcommand", "providecommand", "DeclareRobustCommand"}
ENV_DEFS = {"newenvironment", "renewenvironment"}
XDEFS = {"NewDocumentCommand", "RenewDocumentCommand", "ProvideDocumentCommand", "DeclareDocumentCommand"}
STRUCTURAL_BODY = re.compile(r"\\(?:begin|end|label|ref|cite|input|include|section|subsection|chapter)\b")
PRIMITIVE_CONDITIONALS = {"if", "ifx", "ifcat", "ifnum", "ifdim", "ifodd", "ifvmode", "ifhmode", "ifmmode", "ifinner", "ifvoid", "ifhbox", "ifvbox", "ifeof", "iftrue", "iffalse", "ifcase", "ifdefined", "ifcsname", "iffontchar"}


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_key(value: str) -> str:
    # TeX turns a normal line ending/indentation into interword whitespace.
    # Raw text is retained separately; macros in keys are not expanded.
    return " ".join(value.split())


def boxed_argument_adapter(body, argc):
    """Recognize the corpus's single-evaluation savebox wrapper, not arbitrary macros.

    Only traverse its original argument; do not manufacture the wrapper's layout
    environments. Standard LaTeX sbox/settowidth/usebox semantics are assumed.
    """
    if argc != 1 or not body:
        return False
    # This is a bounded compatibility rule after the definition body has already
    # been delimited by the token/group parser. Comments have no matching weight.
    clean = re.sub(r"(?<!\\)%[^\n]*(?:\n|$)", "", body)
    pattern = (r"\s*\\sbox\\([A-Za-z@]+)\s*\{#1\}\s*"
               r"\\settowidth\\([A-Za-z@]+)\s*\{\\usebox\{\\\1\}\}\s*"
               r"\\begin\{minipage\}\{\\\2\}\\usebox\{\\\1\}\\end\{minipage\}\s*")
    return bool(re.fullmatch(pattern, clean))


class Source:
    def __init__(self, path: Path, root: Path):
        self.path = path
        self.relative = path.relative_to(root).as_posix()
        self.raw = path.read_bytes()
        try:
            self.text = self.raw.decode("utf-8")
            self.encoding = "utf-8"
            self.encoding_status = "strict_utf8"
        except UnicodeDecodeError:
            self.text = self.raw.decode("latin-1")
            self.encoding = "latin-1"
            self.encoding_status = "byte_preserving_fallback_author_encoding_unresolved"
        self.hash = sha256(self.raw)
        self.lines = [0] + [i + 1 for i, c in enumerate(self.text) if c == "\n"]
        self.bytes = [0]
        for c in self.text:
            self.bytes.append(self.bytes[-1] + len(c.encode(self.encoding)))
        self.walker = LatexWalker(self.text, tolerant_parsing=True)

    def undecoded_spans(self):
        if self.encoding_status == "strict_utf8":
            return []
        return [{"source": self.span(m.start(), m.end()), "raw_hex": self.raw[m.start():m.end()].hex()}
                for m in re.finditer(r"[^\x00-\x7f]+", self.text)]

    def span(self, start: int, end: int) -> dict:
        a = bisect.bisect_right(self.lines, start)
        b = bisect.bisect_right(self.lines, end)
        return {"file": self.relative, "char_start": start, "char_end": end,
                "byte_start": self.bytes[start], "byte_end": self.bytes[end],
                "line_start": a, "column_start": start - self.lines[a - 1] + 1,
                "line_end": b, "column_end": end - self.lines[b - 1] + 1}

    def token(self, pos: int):
        return self.walker.get_token(pos, environments=False)

    def skip(self, pos: int) -> int:
        s = self.text
        while pos < len(s):
            if s[pos].isspace():
                pos += 1
            elif s[pos] == "%":
                end = s.find("\n", pos)
                pos = len(s) if end < 0 else end + 1
            else:
                return pos
        return pos

    def group(self, pos: int, opening="{", closing="}"):
        """Read one balanced literal group, ignoring escaped delimiters/comments."""
        pos = self.skip(pos)
        if pos >= len(self.text) or self.text[pos] != opening:
            return None
        start, i, depth, brace_depth = pos, pos + 1, 1, 0
        while i < len(self.text):
            c = self.text[i]
            if c == "%":
                e = self.text.find("\n", i)
                i = len(self.text) if e < 0 else e + 1
                continue
            if c == "\\":
                try:
                    t = self.token(i)
                    i = t.pos + t.len
                    if t.tok == "macro" and t.arg in {"verb", "lstinline"}:
                        if i < len(self.text) and self.text[i] == "*":
                            i += 1
                        if i < len(self.text):
                            end = self.text.find(self.text[i], i + 1)
                            i = len(self.text) if end < 0 else end + 1
                    continue
                except LatexWalkerEndOfStream:
                    break
            if opening == "[":
                if c == "{":
                    brace_depth += 1
                elif c == "}" and brace_depth:
                    brace_depth -= 1
            if brace_depth == 0:
                if c == opening:
                    depth += 1
                elif c == closing:
                    depth -= 1
                    if not depth:
                        return {"value": self.text[start + 1:i], "start": start, "end": i + 1, "balanced": True}
            i += 1
        return {"value": self.text[start + 1:], "start": start, "end": len(self.text), "balanced": False}

    def argument(self, pos: int, optional=False, single=False):
        g = self.group(pos, "[", "]") if optional else self.group(pos)
        if g or optional or not single:
            return g
        pos = self.skip(pos)
        if pos >= len(self.text):
            return None
        t = self.token(pos)
        return {"value": self.text[t.pos:t.pos + t.len].rstrip(), "start": t.pos,
                "end": t.pos + t.len, "balanced": True, "single_token": True}


class StructureExtractor:
    def __init__(self, root, main, *, paper_id=None, arxiv_id=None, recorder=None, max_expanded_chars=40_000_000):
        self.root = Path(root).resolve()
        self.main = self._safe(self.root / main)
        self.paper_id, self.arxiv_id = paper_id, arxiv_id
        self.recorder = Path(recorder) if recorder else None
        self.sources = {}
        self.records = {k: [] for k in ["includes", "sections", "environments", "math_spans", "format_cues", "labels", "references", "citations", "bibliography_items", "definitions", "deferred_code", "macro_calls", "conditions", "unresolved"]}
        self.scopes = [{"id": "document", "macros": {}, "envs": {}}]
        self.scope_counter = self.event_counter = 0
        self.env_stack = []
        self.conditions = []
        self.custom_conditions = set()
        self.unknown_macros = {}
        self.file_stack = []
        self.occurrence_stack = []
        self.reading = []
        self.source_map = []
        self.reading_length = 0
        self.max_expanded_chars = max_expanded_chars
        self.includeonly = None
        self.section_context = []
        self.section_levels = []
        self.global_next = False
        self.makeatletter = False
        self.math_stack = []

    def _safe(self, path):
        path = Path(path).resolve()
        if not path.is_relative_to(self.root):
            raise ValueError(f"source path leaves root: {path}")
        return path

    def source(self, path):
        path = self._safe(path)
        key = path.relative_to(self.root).as_posix()
        if key not in self.sources:
            self.sources[key] = Source(path, self.root)
        return self.sources[key]

    def state(self):
        if any(c["branch_active"] is False for c in self.conditions):
            return "inactive"
        if any(c["branch_active"] is None for c in self.conditions):
            return "conditional"
        return "active"

    def emit(self, kind, src, start, end, **data):
        self.event_counter += 1
        rec = {"id": f"e{self.event_counter}", "source": src.span(start, end),
               "source_role": "author_support_code" if src.path.suffix in {".sty", ".cls"} else "document_or_bibliography",
               "activity": self.state(), "scope": [x["id"] for x in self.scopes],
               "condition_ids": [c["id"] for c in self.conditions],
               "file_occurrence": self.occurrence_stack[-1] if self.occurrence_stack else "main", **data}
        self.records[kind].append(rec)
        return rec

    def issue(self, src, start, end, reason, **data):
        return self.emit("unresolved", src, start, end, reason=reason, **data)

    def push_scope(self, kind, identifier):
        self.scope_counter += 1
        self.scopes.append({"id": f"{kind}:{identifier}:{self.scope_counter}", "macros": {}, "envs": {}})

    def lookup(self, name, kind="macros"):
        for scope in reversed(self.scopes):
            if name in scope[kind]:
                return scope[kind][name]
        return None

    def define(self, record, name, kind="macros", globally=False):
        if self.state() == "active":
            self.scopes[0 if globally else -1][kind][name] = record

    def append(self, src, start, end, occurrence):
        if end <= start:
            return
        text = src.text[start:end]
        if self.reading_length + len(text) > self.max_expanded_chars:
            raise ValueError("expanded source exceeds configured limit")
        self.source_map.append({"reading_char_start": self.reading_length,
                                "reading_char_end": self.reading_length + len(text),
                                "source": src.span(start, end), "occurrence": occurrence})
        self.reading.append(text)
        self.reading_length += len(text)

    def read_arg(self, src, pos, optional=False, single=False):
        arg = src.argument(pos, optional=optional, single=single)
        if arg and arg.get("single_token") and self.makeatletter and arg["end"] < len(src.text) and (src.text[arg["end"]] == "@" or arg["value"].startswith("\\@")):
            end = arg["end"]
            while end < len(src.text) and (src.text[end].isalpha() or src.text[end] == "@"):
                end += 1
            arg["value"], arg["end"] = src.text[arg["start"]:end], end
        if arg and not arg["balanced"]:
            self.issue(src, arg["start"], arg["end"], "unbalanced_argument")
        return arg

    def args(self, src, pos, n=1, optional=False):
        out = []
        if optional:
            a = self.read_arg(src, pos, optional=True)
            if a:
                out.append(a)
                pos = a["end"]
        for _ in range(n):
            a = self.read_arg(src, pos, single=True)
            if not a:
                break
            out.append(a)
            pos = a["end"]
        return out, pos

    def arg_values(self, src, args):
        return [{"value": a["value"], "source": src.span(a["start"], a["end"])} for a in args]

    def title_references(self, src, argument, owner_id, role):
        """Known LaTeX headings render their title; retain literal refs within it.

        No expansion of user commands inside titles is attempted. Their arguments
        are skipped and any hidden reference is explicitly marked as unresolved.
        """
        p, end = argument["start"] + 1, argument["end"] - 1
        while p < end:
            try:
                token = src.token(p)
            except LatexWalkerEndOfStream:
                break
            p = token.pos + token.len
            if token.tok != "macro":
                continue
            name = token.arg
            if name in REFS | CITES:
                options = []
                while True:
                    optional = self.read_arg(src, p, optional=True)
                    if not optional:
                        break
                    options.append(optional)
                    p = optional["end"]
                arg = self.read_arg(src, p)
                if not arg or arg["end"] > argument["end"]:
                    continue
                p = arg["end"]
                value = options[0]["value"] if name == "hyperref" and options else arg["value"]
                raw_keys = [x.strip() for x in value.split(",")]
                self.emit("citations" if name in CITES else "references", src, token.pos, p,
                          command=name, keys=[normalize_key(x) for x in raw_keys], raw_keys=raw_keys,
                          dynamic=bool(re.search(r"[\\#]", value)), options=self.arg_values(src, options),
                          environment_id=owner_id if role == "environment_title" else None,
                          section_ids=list(self.section_context), container_id=owner_id, argument_role=role)
            elif self.lookup(name):
                definition = self.lookup(name)
                aa, q = self.args(src, p, n=definition.get("argument_count", 0))
                if any(STRUCTURAL_BODY.search(x["value"]) for x in aa):
                    self.issue(src, token.pos, q, "title_macro_argument_not_expanded", name=name, container_id=owner_id)
                p = q

    def definition(self, src, name, start, pos):
        s = src.text
        p = src.skip(pos)
        star = p < len(s) and s[p] == "*"
        if star:
            p += 1
        a = self.read_arg(src, p, single=True)
        if not a:
            self.issue(src, start, p, "definition_missing_name", command=name)
            return p
        if a.get("single_token") and self.makeatletter and a["end"] < len(s) and (s[a["end"]] == "@" or a["value"].startswith("\\@")):
            end = a["end"]
            while end < len(s) and (s[end].isalpha() or s[end] == "@"):
                end += 1
            a["value"] = s[a["start"]:end]
            a["end"] = end
        identifier = a["value"].strip().lstrip("\\")
        p, argc, default, body, ending, signature = a["end"], 0, None, None, None, None
        dynamic_name = identifier == "csname"
        if dynamic_name:
            stop = s.find("\\endcsname", p)
            if stop >= 0:
                identifier = "\\csname" + s[p:stop] + "\\endcsname"
                p = stop + len("\\endcsname")
        env = name in ENV_DEFS or name == "newtheorem"
        if name == "newtheorem":
            shared = self.read_arg(src, p, optional=True)
            if shared:
                p = shared["end"]
            title = self.read_arg(src, p)
            if title:
                p, body = title["end"], title["value"]
            within = self.read_arg(src, p, optional=True)
            if within:
                p = within["end"]
            extra = {"shared_counter": shared["value"] if shared else None,
                     "within_counter": within["value"] if within else None, "theorem_like": True}
        elif name in {"def", "gdef", "edef", "xdef"}:
            # Primitive definitions have an undelimited/delimited parameter text.
            param_start = p
            while p < len(s):
                t = src.token(p)
                if t.tok == "brace_open":
                    break
                p = t.pos + t.len
            signature = s[param_start:p]
            argc = max([int(x) for x in re.findall(r"#([1-9])", signature)] or [0])
            b = self.read_arg(src, p)
            if b:
                body, p = b["value"], b["end"]
            extra = {"parameter_text": signature, "delimited_parameters": bool(re.sub(r"#[1-9]|\s", "", signature or "")), "expansion_mode": name}
        elif name in XDEFS:
            sig = self.read_arg(src, p)
            if sig:
                signature, p = sig["value"], sig["end"]
            b = self.read_arg(src, p)
            if b:
                body, p = b["value"], b["end"]
            extra = {"xparse_signature": signature, "signature_interpreted": False}
        elif name == "DeclareMathOperator":
            b = self.read_arg(src, p)
            if b:
                body, p = b["value"], b["end"]
            extra = {}
        else:
            count = self.read_arg(src, p, optional=True)
            if count:
                p = count["end"]
                try:
                    argc = int(count["value"])
                except ValueError:
                    self.issue(src, count["start"], count["end"], "dynamic_parameter_count")
                default_arg = self.read_arg(src, p, optional=True)
                if default_arg:
                    default, p = default_arg["value"], default_arg["end"]
            b = self.read_arg(src, p, single=True)
            if b:
                body, p = b["value"], b["end"]
            if env:
                e = self.read_arg(src, p, single=True)
                if e:
                    ending, p = e["value"], e["end"]
            extra = {}
        rec = self.emit("definitions", src, start, p, command=name, name=identifier,
                        definition_kind="environment" if env else "macro", argument_count=argc,
                        optional_default=default, body=body, end_body=ending, starred=star,
                        contains_structure=bool(STRUCTURAL_BODY.search((body or "") + (ending or ""))),
                        scope_semantics="conservative_lexical", **extra)
        if boxed_argument_adapter(body, argc):
            rec["argument_traversal_adapter"] = "single_evaluation_savebox_v1"
        valid_name = env or bool(re.fullmatch(r"[A-Za-z@]+|[^A-Za-z]", identifier))
        if dynamic_name:
            self.issue(src, start, p, "dynamic_definition_name", definition_id=rec["id"])
        elif not valid_name:
            self.issue(src, start, p, "invalid_literal_control_sequence_name", definition_id=rec["id"], name=identifier)
        else:
            self.define(rec, identifier, "envs" if env else "macros", globally=name in {"gdef", "xdef"} or self.global_next)
        self.global_next = False
        if body is None:
            self.issue(src, start, p, "definition_missing_body", definition_id=rec["id"])
        return p

    def resolve(self, raw, current, suffix=".tex", directory=None):
        raw = raw.strip().strip('"')
        if not raw or any(c in raw for c in "\\#{}$~"):
            return None, "dynamic_path"
        candidates = []
        bases = [self.root, current.path.parent]
        if directory:
            if any(c in directory for c in "\\#{}$~"):
                return None, "dynamic_directory"
            bases = [self.root / directory, current.path.parent / directory]
        for base in bases:
            p = base / raw
            for c in [p, p.with_suffix(suffix) if not p.suffix else p]:
                try:
                    c = self._safe(c)
                except ValueError:
                    continue
                if c.is_file() and c not in candidates:
                    candidates.append(c)
        return (candidates[0], "literal_root_first") if candidates else (None, "not_found")

    def include(self, src, name, start, pos):
        directory = None
        optional, p = self.args(src, pos, n=0, optional=True)
        raw_args, p = self.args(src, p, n=2 if name in {"import", "subimport", "inputfrom", "includefrom"} else 1)
        if not raw_args:
            self.issue(src, start, p, "input_missing_path", command=name)
            return p, []
        if len(raw_args) == 2:
            directory = raw_args[0]["value"]
        raw = raw_args[-1]["value"]
        # Primitive unbraced input consumes a filename, rather than one token.
        if name in {"input", "include"} and raw_args[-1].get("single_token") and not raw.startswith("\\"):
            q = raw_args[-1]["start"]
            while p < len(src.text) and not src.text[p].isspace() and src.text[p] not in "%{}":
                p += 1
            raw = src.text[q:p]
        package = name in {"usepackage", "RequirePackage", "documentclass", "LoadClass"}
        suffix = ".sty" if name in {"usepackage", "RequirePackage"} else ".cls" if name in {"documentclass", "LoadClass"} else ".bbl" if name == "bibliography" else ".tex"
        paths = []
        names = raw.split(",") if package else [raw]
        if name == "bibliography":
            names = [self.main.stem]
        for filename in names:
            path, basis = self.resolve(filename, src, suffix, directory)
            status = "resolved" if path else "external_package" if package and basis == "not_found" else basis
            if package and not filename.strip():
                status = "empty_package_list"
            if name == "input" and not path and filename.strip() in {"amstex", "pictex", "xy", "epsf", "amssym.def", "amssym.tex"}:
                status = "system_input_expected"
            if name == "include" and self.includeonly is not None and filename.removesuffix(".tex") not in self.includeonly:
                status = "excluded_by_includeonly"
            if self.state() == "conditional":
                status = "conditional_not_expanded"
            rec = self.emit("includes", src, start, p, command=name, requested=filename, directory=directory,
                            resolved_file=path.relative_to(self.root).as_posix() if path else None,
                            status=status, resolution_basis=basis,
                            role="author_package" if package else "bibliography" if name == "bibliography" else "document_input")
            if status == "resolved":
                if path in self.file_stack:
                    rec["status"] = "cycle_not_expanded"
                    self.issue(src, start, p, "input_cycle", include_id=rec["id"])
                else:
                    paths.append((path, rec["id"]))
            elif status not in {"external_package", "excluded_by_includeonly", "empty_package_list", "system_input_expected"}:
                self.issue(src, start, p, "unexpanded_input", include_id=rec["id"], status=status)
        return p, paths

    def process_file(self, path, occurrence="main"):
        src = self.source(path)
        package_mode = path.suffix in {".sty", ".cls"}
        saved_makeatletter = self.makeatletter
        saved_scopes = list(self.scopes)
        saved_conditions = list(self.conditions)
        if package_mode:
            # LaTeX loads classes and packages with @ a letter, then restores it.
            self.makeatletter = True
        self.file_stack.append(path)
        self.occurrence_stack.append(occurrence)
        s, p, chunk = src.text, 0, 0
        initial_condition_depth = len(self.conditions)
        while p < len(s):
            try:
                t = src.token(p)
            except LatexWalkerEndOfStream:
                break
            except Exception as exc:
                self.issue(src, p, min(p + 1, len(s)), "token_error", error=str(exc))
                p += 1
                continue
            start, p = t.pos, t.pos + t.len
            if t.tok == "comment":
                continue
            if t.tok in {"mathmode_inline", "mathmode_display"} and self.state() != "inactive":
                if t.arg == "$$" and self.math_stack and self.math_stack[-1]["closing"] == "$":
                    t.arg, p = "$", start + 1
                closing = {"$": "$", "$$": "$$", "\\(": "\\)", "\\[": "\\]"}
                if self.math_stack and self.math_stack[-1]["closing"] == t.arg:
                    entry = self.math_stack.pop()
                    rec = entry["record"]
                    rec["end_source"], rec["closed"] = src.span(start, p), True
                    if rec["source"]["file"] == src.relative:
                        rec["full_source"] = src.span(rec["source"]["char_start"], p)
                elif t.arg in closing:
                    rec = self.emit("math_spans", src, start, p, delimiter=t.arg, math_kind=t.tok, closed=False)
                    self.math_stack.append({"closing": closing[t.arg], "record": rec})
                else:
                    self.issue(src, start, p, "unmatched_math_delimiter", delimiter=t.arg)
                continue
            if t.tok in {"brace_open", "brace_close"}:
                if self.state() != "inactive":
                    if t.tok == "brace_open":
                        q = src.skip(p)
                        if s.startswith("\\bf ", q) or s.startswith("\\bf\n", q) or s.startswith("\\gross ", q):
                            g = src.group(start)
                            if g and len(g["value"]) < 500:
                                self.emit("format_cues", src, start, g["end"], cue="bold_group",
                                          raw_content=g["value"], interpretation="formatting_only_not_validated_section_or_claim")
                        self.push_scope("group", f"{src.relative}:{start}")
                    elif len(self.scopes) > 1 and self.scopes[-1]["id"].startswith("group:"):
                        self.scopes.pop()
                    else:
                        self.issue(src, start, p, "unmatched_group_close")
                continue
            if t.tok != "macro":
                continue
            name = t.arg
            if self.makeatletter and p < len(s) and (s[p] == "@" or name == "@"):
                q = p
                while p < len(s) and (s[p].isalpha() or s[p] == "@"):
                    p += 1
                name += s[q:p]
            if name in {"makeatletter", "makeatother"}:
                self.makeatletter = name == "makeatletter"
                continue
            if name == "newif":
                a = self.read_arg(src, p, single=True)
                if a:
                    p = a["end"]
                    if self.makeatletter and p < len(s) and s[p] == "@":
                        while p < len(s) and (s[p].isalpha() or s[p] == "@"):
                            p += 1
                        a["value"] = s[a["start"]:p]
                    self.custom_conditions.add(a["value"].strip().lstrip("\\"))
                    self.emit("definitions", src, start, p, command=name, name=a["value"], definition_kind="conditional", evaluation="unknown")
                continue
            if name in PRIMITIVE_CONDITIONALS or name in self.custom_conditions or name.startswith("if@") or name in {"ifpdf", "ifPDFTeX", "ifXeTeX", "ifLuaTeX", "ifxetex", "ifluatex"}:
                value = True if name == "iftrue" else False if name == "iffalse" else None
                rec = self.emit("conditions", src, start, p, command=name, evaluated=value)
                self.conditions.append({"id": rec["id"], "value": value, "branch_active": value, "command": name})
                continue
            if name in {"else", "or"}:
                if self.conditions:
                    c = self.conditions[-1]
                    c["branch_active"] = not c["value"] if c["value"] is not None and name == "else" else None
                else:
                    self.issue(src, start, p, "conditional_branch_without_if", command=name)
                continue
            if name == "fi":
                if self.conditions:
                    self.conditions.pop()
                else:
                    self.issue(src, start, p, "fi_without_if")
                continue
            if self.state() == "inactive":
                continue
            if name in {"DeclareOption", "AtBeginDocument", "AtEndDocument", "AtBeginPackage", "AtEndPackage", "AtBeginEnvironment", "AtEndEnvironment", "pgfkeys", "pgfqkeys", "tikzset", "tcbset", "lstset", "xpatchcmd", "patchcmd", "pretocmd", "apptocmd", "appto", "preto", "@ifundefined", "@ifpackageloaded", "ifcsdef", "ifdef", "ifundef"}:
                count = 2 if name in {"DeclareOption", "AtBeginEnvironment", "AtEndEnvironment", "pgfqkeys"} else 1
                if name in {"xpatchcmd", "patchcmd"}:
                    count = 5
                elif name in {"pretocmd", "apptocmd"}:
                    count = 4
                elif name in {"appto", "preto"}:
                    count = 2
                elif name in {"@ifundefined", "@ifpackageloaded", "ifcsdef", "ifdef", "ifundef"}:
                    count = 3
                q = src.skip(p)
                if q < len(s) and s[q] == "*":
                    p = q + 1
                    if name == "DeclareOption":
                        count = 1
                aa, p = self.args(src, p, n=count)
                self.emit("deferred_code", src, start, p, command=name, arguments=self.arg_values(src, aa),
                          semantics="registered_code_or_key_configuration_not_executed_here")
                continue
            if name in {"verb", "lstinline"}:
                q = p + (p < len(s) and s[p] == "*")
                if q < len(s):
                    e = s.find(s[q], q + 1)
                    if e >= 0:
                        p = e + 1
                    else:
                        self.issue(src, start, len(s), "unclosed_inline_verbatim")
                        p = len(s)
                continue
            if name in COMMAND_DEFS | ENV_DEFS | XDEFS | {"newtheorem", "DeclareMathOperator", "def", "gdef", "edef", "xdef"}:
                p = self.definition(src, name, start, p)
                continue
            if name == "global":
                self.global_next = True
                continue
            if name in {"begingroup", "bgroup"}:
                self.push_scope("explicit_group", f"{src.relative}:{start}")
                continue
            if name in {"endgroup", "egroup"}:
                if len(self.scopes) > 1:
                    self.scopes.pop()
                else:
                    self.issue(src, start, p, "unmatched_explicit_group")
                continue
            if name in {"catcode", "csname", "let", "futurelet", "expandafter", "scantokens", "loop", "repeat"}:
                self.issue(src, start, p, "tex_execution_required", command=name)
                if name in {"let", "futurelet"}:
                    # Avoid treating alias targets (e.g. \let\oldsection\section) as executions.
                    for _ in range(2 if name == "let" else 3):
                        p = src.skip(p)
                        if p < len(s) and s[p] == "=":
                            p = src.skip(p + 1)
                        a = self.read_arg(src, p, single=True)
                        if a:
                            p = a["end"]
                continue
            if name == "includeonly":
                a = self.read_arg(src, p)
                if a:
                    p = a["end"]
                    if "\\" not in a["value"] and self.state() == "active":
                        self.includeonly = {x.strip().removesuffix(".tex") for x in a["value"].split(",")}
                    else:
                        self.issue(src, start, p, "dynamic_includeonly")
                continue
            if name in {"input", "include", "subfile", "import", "subimport", "inputfrom", "includefrom", "usepackage", "RequirePackage", "documentclass", "LoadClass", "bibliography"}:
                p, paths = self.include(src, name, start, p)
                if paths:
                    self.append(src, chunk, p, occurrence)
                    for child, event in paths:
                        self.process_file(child, event)
                    chunk = p
                continue
            if name in {"begin", "end"}:
                a = self.read_arg(src, p)
                if not a:
                    self.issue(src, start, p, "environment_missing_name")
                    continue
                env, p = a["value"].strip(), a["end"]
                if name == "begin":
                    definition = self.lookup(env, "envs")
                    extra_args = []
                    opt = self.read_arg(src, p, optional=True)
                    if opt:
                        extra_args.append(opt)
                        p = opt["end"]
                    for _ in range((definition or {}).get("argument_count", 0) - int((definition or {}).get("optional_default") is not None)):
                        x = self.read_arg(src, p, single=True)
                        if x:
                            extra_args.append(x)
                            p = x["end"]
                    kind = "verbatim" if env in VERBATIM else "math" if env in MATH else "theorem" if (definition or {}).get("theorem_like") or env in THEOREMS else "other"
                    rec = self.emit("environments", src, start, p, name=env, environment_kind=kind,
                                    definition_id=definition["id"] if definition else None, arguments=self.arg_values(src, extra_args),
                                    parent_environment=self.env_stack[-1]["record"]["id"] if self.env_stack else None,
                                    section_ids=list(self.section_context), closed=False)
                    if kind == "theorem":
                        for title_argument in extra_args:
                            self.title_references(src, title_argument, rec["id"], "environment_title")
                    if env in VERBATIM:
                        end_marker = "\\end{" + env + "}"
                        e = s.find(end_marker, p)
                        if e < 0:
                            self.issue(src, start, len(s), "unclosed_verbatim", environment=env)
                            p = len(s)
                        else:
                            rec["body_source"] = src.span(p, e)
                            p = e + len(end_marker)
                            rec["full_source"], rec["closed"] = src.span(start, p), True
                        continue
                    self.push_scope("environment", rec["id"])
                    self.env_stack.append({"name": env, "record": rec, "body_start": p, "file": src.relative,
                                           "scope_depth": len(self.scopes)})
                else:
                    matching = next((i for i in range(len(self.env_stack) - 1, -1, -1) if self.env_stack[i]["name"] == env), None)
                    if matching is None:
                        self.issue(src, start, p, "unmatched_environment_end", environment=env)
                    else:
                        for unclosed in self.env_stack[matching + 1:]:
                            self.issue(src, start, p, "environment_closed_out_of_order", environment=unclosed["name"])
                        entry = self.env_stack[matching]
                        rec = entry["record"]
                        rec["end_source"], rec["closed"] = src.span(start, p), True
                        if entry["file"] == src.relative:
                            rec["body_source"] = src.span(entry["body_start"], start)
                            rec["full_source"] = src.span(rec["source"]["char_start"], p)
                        self.env_stack = self.env_stack[:matching]
                        self.scopes = self.scopes[:entry["scope_depth"] - 1]
                    if env == "document":
                        self.append(src, chunk, p, occurrence)
                        chunk = p
                        if p < len(s) and s[p:].strip():
                            self.issue(src, p, len(s), "trailing_source_after_end_document")
                        break
                continue
            if name in SECTIONS:
                q = src.skip(p)
                starred = q < len(s) and s[q] == "*"
                if starred:
                    p = q + 1
                args, p = self.args(src, p, n=1, optional=True)
                rec = self.emit("sections", src, start, p, command=name, starred=starred,
                                title=args[-1]["value"] if args else None,
                                short_title=args[0]["value"] if len(args) > 1 else None)
                level = ["part", "chapter", "section", "subsection", "subsubsection", "paragraph", "subparagraph"].index(name)
                self.section_levels = [(old_level, old_id) for old_level, old_id in self.section_levels if old_level < level]
                self.section_levels.append((level, rec["id"]))
                self.section_context = [old_id for _, old_id in self.section_levels]
                for title_argument in args:
                    self.title_references(src, title_argument, rec["id"], "section_title")
                continue
            if name == "label" or name in REFS | CITES:
                q = src.skip(p)
                if q < len(s) and s[q] == "*":
                    p = q + 1
                options = []
                while True:
                    a = self.read_arg(src, p, optional=True)
                    if not a:
                        break
                    options.append(a)
                    p = a["end"]
                args, p = self.args(src, p)
                if name == "hyperref" and options:
                    value = options[0]["value"]
                else:
                    value = args[0]["value"] if args else None
                raw_keys = [x.strip() for x in value.split(",")] if value is not None else []
                keys = [normalize_key(x) for x in raw_keys]
                self.emit("labels" if name == "label" else "citations" if name in CITES else "references", src, start, p,
                          command=name, keys=keys, raw_keys=raw_keys, dynamic=bool(value and re.search(r"[\\#]", value)),
                          options=self.arg_values(src, options),
                          environment_id=self.env_stack[-1]["record"]["id"] if self.env_stack else None,
                          section_ids=list(self.section_context))
                continue
            if name == "bibitem":
                aa, p = self.args(src, p, n=1, optional=True)
                self.emit("bibliography_items", src, start, p, key=normalize_key(aa[-1]["value"]) if aa else None,
                          display_label=aa[0]["value"] if len(aa) > 1 else None,
                          semantic_link_status="bibliographic_key_only_not_identified_external_document")
                continue
            if name in {"endinput", "bye"}:
                self.append(src, chunk, p, occurrence)
                chunk = p
                break
            definition = self.lookup(name)
            if definition:
                argc = definition.get("argument_count", 0)
                args = []
                q = p
                if definition.get("optional_default") is not None:
                    a = self.read_arg(src, q, optional=True)
                    if a:
                        args.append(a)
                        q = a["end"]
                    argc -= 1
                for _ in range(max(0, argc)):
                    a = self.read_arg(src, q, single=True)
                    if a:
                        args.append(a)
                        q = a["end"]
                rec = self.emit("macro_calls", src, start, q, name=name, definition_id=definition["id"],
                                arguments=self.arg_values(src, args), expanded=False)
                if definition.get("contains_structure") or definition.get("delimited_parameters") or definition.get("xparse_signature"):
                    self.issue(src, start, q, "custom_macro_requires_expansion", macro_call_id=rec["id"], name=name)
                adapter = definition.get("argument_traversal_adapter")
                if adapter:
                    rec["argument_traversal_adapter"] = adapter
                if not adapter and any(STRUCTURAL_BODY.search(a["value"]) for a in args):
                    self.issue(src, start, q, "macro_argument_structure_not_expanded", macro_call_id=rec["id"], name=name)
                # A macro can ignore, duplicate or reorder its arguments. Their
                # contents are not automatically live structure; retain the exact
                # arguments and make required expansion explicit instead.
                if not adapter:
                    p = q
            else:
                info = self.unknown_macros.setdefault(name, {"count": 0, "first_source": src.span(start, p)})
                info["count"] += 1
        else:
            p = len(s)
        self.append(src, chunk, p, occurrence)
        self.file_stack.pop()
        self.occurrence_stack.pop()
        if package_mode:
            if len(self.scopes) != len(saved_scopes) or len(self.conditions) != len(saved_conditions):
                self.issue(src, 0, 0, "package_scope_or_condition_uncertain_restored", package=src.relative)
            self.scopes = saved_scopes
            self.conditions = saved_conditions
            self.makeatletter = saved_makeatletter

    def recorder_inputs(self):
        result = {"available": bool(self.recorder and self.recorder.is_file()),
                  "note": "TeX recorder inputs are independent execution evidence, not reading order or mathematical dependencies.",
                  "author_inputs": [], "not_in_static_traversal": []}
        if not result["available"]:
            return result
        raw = self.recorder.read_bytes()
        result["sha256"] = sha256(raw)
        # Match project files by the longest unambiguous suffix; build paths differ
        # from immutable source paths. A basename alone is not accepted if ambiguous.
        all_files = [p.relative_to(self.root).as_posix() for p in self.root.rglob("*") if p.is_file()]
        matches = set()
        for line in raw.decode("utf-8", errors="replace").splitlines():
            if not line.startswith("INPUT "):
                continue
            value = line[6:].replace("\\", "/")
            if value.startswith("/") and "/work/" not in value and not value.startswith(str(self.root) + "/"):
                continue
            candidates = [x for x in all_files if value == x or value.endswith("/" + x)]
            if candidates:
                longest = max(candidates, key=len)
                matches.add(longest)
        result["author_inputs"] = sorted(matches)
        result["not_in_static_traversal"] = sorted(x for x in matches if x not in self.sources and Path(x).suffix in {".tex", ".sty", ".cls", ".bbl"})
        return result

    def extract(self):
        self.process_file(self.main)
        for item in self.env_stack:
            rec = item["record"]
            rec["closure_status"] = "unclosed_or_macro_generated_end"
        labels = {}
        for label in self.records["labels"]:
            for key in label["keys"]:
                if not label["dynamic"]:
                    labels.setdefault(key, []).append(label["id"])
        for ref in self.records["references"]:
            ref["candidate_targets"] = {key: labels.get(key, []) for key in ref["keys"]}
            ref["relation_semantics"] = "syntactic_reference_not_mathematical_dependency"
        label_by_id = {rec["id"]: rec for rec in self.records["labels"]}
        distances = []
        for ref in self.records["references"]:
            for target_ids in ref["candidate_targets"].values():
                for target in target_ids:
                    dest = label_by_id[target]
                    distances.append({"reference_id": ref["id"], "label_id": target,
                                      "cross_file": ref["source"]["file"] != dest["source"]["file"],
                                      "line_distance": abs(ref["source"]["line_start"] - dest["source"]["line_start"]) if ref["source"]["file"] == dest["source"]["file"] else None})
        bib = {}
        for item in self.records["bibliography_items"]:
            if item["key"]:
                bib.setdefault(item["key"], []).append(item["id"])
        for cite in self.records["citations"]:
            cite["candidate_bibliography_items"] = {key: bib.get(key, []) for key in cite["keys"]}
        inventory = []
        for path in sorted(self.root.rglob("*")):
            if path.is_file():
                data = path.read_bytes()
                relative = path.relative_to(self.root).as_posix()
                inventory.append({"path": relative, "bytes": len(data), "sha256": sha256(data),
                                  "statically_traversed": relative in self.sources})
        structure = {"schema_version": SCHEMA, "paper_id": self.paper_id, "arxiv_id": self.arxiv_id,
                     "source_root": str(self.root), "main_file": self.main.relative_to(self.root).as_posix(),
                     "parser": {"pylatexenc": pylatexenc.__version__, "mode": "conservative_token_structure_no_macro_execution"},
                     "author_file_inventory": inventory,
                     "files": [{"path": s.relative, "sha256": s.hash, "bytes": len(s.raw), "characters": len(s.text),
                                "encoding": s.encoding, "encoding_status": s.encoding_status,
                                "undecoded_non_ascii_spans": s.undecoded_spans()} for s in self.sources.values()], **self.records,
                     "unmodeled_macro_names": self.unknown_macros, "recorder": self.recorder_inputs()}
        text = "".join(self.reading)
        failures = []
        for segment in self.source_map:
            a = segment["source"]
            src = self.sources[a["file"]]
            original = src.raw[a["byte_start"]:a["byte_end"]].decode(src.encoding)
            if original != text[segment["reading_char_start"]:segment["reading_char_end"]]:
                failures.append(a)
        for values in self.records.values():
            for rec in values:
                a = rec["source"]
                src = self.sources[a["file"]]
                if src.text[a["char_start"]:a["char_end"]] != src.raw[a["byte_start"]:a["byte_end"]].decode(src.encoding):
                    failures.append(a)
        quality = {"schema_version": SCHEMA, "status": "ok_with_explicit_limits" if not failures else "source_map_failed",
                   "counts": {key: len(value) for key, value in self.records.items()},
                   "files_traversed": len(self.sources), "reading_characters": len(text),
                   "source_segments": len(self.source_map), "source_roundtrip_failures": failures,
                   "unresolved_by_reason": dict(Counter(x["reason"] for x in self.records["unresolved"])),
                   "unresolved_by_source_role": dict(Counter(x["source_role"] for x in self.records["unresolved"])),
                   "unclosed_environments": [x["record"]["id"] for x in self.env_stack],
                   "unclosed_conditionals": [x["id"] for x in self.conditions],
                   "unclosed_math_spans": [x["record"]["id"] for x in self.math_stack],
                   "duplicate_labels": {key: ids for key, ids in labels.items() if len(ids) > 1},
                   "unresolved_reference_keys": sorted({k for ref in self.records["references"] for k, ids in ref["candidate_targets"].items() if not ids}),
                   "unresolved_citation_keys": sorted({k for cite in self.records["citations"] for k, ids in cite["candidate_bibliography_items"].items() if not ids and k != "*"}),
                   "reference_distance_examples": sorted(distances, key=lambda x: x["line_distance"] or 0, reverse=True)[:10],
                   "cross_file_reference_candidates": sum(x["cross_file"] for x in distances),
                   "unmodeled_macros_distinct": len(self.unknown_macros),
                   "encoding_unresolved_files": [s.relative for s in self.sources.values() if s.encoding_status != "strict_utf8"],
                   "recorder_untraversed_author_inputs": structure["recorder"]["not_in_static_traversal"],
                   "semantic_extraction_performed": False,
                   "limits": ["No general TeX macro expansion, catcode execution, or unknown conditional evaluation.",
                              "Unknown package commands are preserved in source; structural wrappers may need explicit adaptation.",
                              "Label resolution is a candidate syntax link, not a mathematical or proof dependency."]}
        return structure, text, self.source_map, quality


def preprocess(root, main, output, **kwargs):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    structure, text, source_map, quality = StructureExtractor(root, main, **kwargs).extract()
    for name, data in [("structure.json", structure), ("source_map.json", source_map), ("quality.json", quality)]:
        (output / name).write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output / "reading.tex").write_text(text, encoding="utf-8", newline="")
    return quality
