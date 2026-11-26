"""Source-only conditional scopes with an explicit standard-iff exception.

This helper is not used by the historical frozen node strategy. Integrating it
requires a new source-control binding and regression review. Original text and
the aligned lexical mask are never rewritten in the source artifact.
"""
from __future__ import annotations

import re

from gangmu.latex_structure import COMMAND_DEFS, XDEFS

VERSION = "gangmu.rule.source-conditional-scope.v1"
_IFF = re.compile(r"\\iff(?![A-Za-z@])")
_CONDITIONAL = re.compile(r"\\(?:if[A-Za-z@]*|else|or|fi)(?![A-Za-z@])")
_ASSIGN_IFF = re.compile(r"\\(?:let|futurelet)\s*\\iff(?![A-Za-z@])")
_DYNAMIC = re.compile(r"\\(?:catcode|csname|expandafter|scantokens|edef|xdef)(?![A-Za-z@])")
_MUTATION_COMMANDS = COMMAND_DEFS | XDEFS | {
    "let", "futurelet", "def", "gdef", "edef", "xdef", "newif", "DeclareMathOperator"}
_MUTATION = re.compile(r"\\(?:" + "|".join(sorted(_MUTATION_COMMANDS)) + r")(?![A-Za-z@])")
_TARGET = re.compile(r"\s*\*?\s*(?:\{\s*)?\\([A-Za-z@]+)")
_PROTECTED = _MUTATION_COMMANDS | {
    "iff", "catcode", "csname", "expandafter", "scantokens"}


def _command_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    return value.strip().strip("{}").strip().lstrip("\\")


def _could_mutate_iff(body: str) -> bool:
    if _DYNAMIC.search(body):
        return True
    for command in _MUTATION.finditer(body):
        target = _TARGET.match(body, command.end())
        if target is None or target.group(1) in _PROTECTED:
            return True
    return False


def standard_iff_is_unshadowed(project, mask: str | None = None,
                              position: int | None = None) -> bool:
    """Do not exempt iff when the source declares, aliases or uses it as a switch.

    Conditional declarations in E1 retain backslash-prefixed names, unlike some
    macro definitions. Conditions are checked as well as declarations so a
    parsed iff switch cannot lose its activity obligation through this helper.
    """
    structure = project.structure_data
    if (
        any(_command_name(row.get("name")) in _PROTECTED
            for row in structure.get("definitions", []))
        or any(_command_name(row.get("command")) == "iff"
               for row in structure.get("conditions", []))
    ):
        return False
    # E1 intentionally reports primitive assignments as unresolved operations,
    # not macro definitions. Without a visible mask their target is unknown.
    assignments = [row for row in structure.get("unresolved", [])
                   if row.get("command") in {"let", "futurelet"}
                   and row.get("activity") != "inactive"]
    if mask is None:
        return not assignments
    visible = mask[:position] if position is not None else mask
    if _ASSIGN_IFF.search(visible) or _DYNAMIC.search(visible):
        return False
    # An unexpanded author macro may perform an assignment when called. Retain
    # the obligation even for an unused suspicious definition rather than
    # claiming to evaluate its control effects.
    for row in structure.get("definitions", []):
        for key in ("body", "end_body"):
            body = row.get(key)
            if isinstance(body, str) and _could_mutate_iff(body):
                return False
    for row in structure.get("deferred_code", []):
        for argument in row.get("arguments", []):
            body = argument.get("value")
            if isinstance(body, str) and _could_mutate_iff(body):
                return False
    return True


def source_conditional_scope(project, mask: str, position: int) -> list[dict]:
    """Ignore the ordinary logical operator; preserve every other scope check."""
    if standard_iff_is_unshadowed(project, mask, position):
        mask = _IFF.sub(lambda match: " " * len(match.group(0)), mask)
    stack: list[dict] = []
    unmatched: list[dict] = []
    for match in _CONDITIONAL.finditer(mask, 0, position):
        command = match.group(0)[1:]
        if command.startswith("if"):
            if re.search(r"\\newif\s*$", mask[max(0, match.start() - 30):match.start()]):
                continue
            stack.append({"command": command, "char_start": match.start(),
                          "literal_known": command in {"iftrue", "iffalse"},
                          "branch": "then"})
        elif command in {"else", "or"}:
            if stack:
                stack[-1]["branch"] = command
            else:
                unmatched.append({"command": command, "char_start": match.start(),
                                  "kind": "unmatched_conditional_branch"})
        elif stack:
            stack.pop()
        else:
            unmatched.append({"command": command, "char_start": match.start(),
                              "kind": "unmatched_conditional_end"})
    return unmatched + [scope for scope in stack if not scope["literal_known"]]
