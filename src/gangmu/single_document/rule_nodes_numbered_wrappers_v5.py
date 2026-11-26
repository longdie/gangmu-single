"""Discover literal theorem wrappers that only replace a displayed number.

This helper returns additions and diagnostics; it never edits the incoming rule
result, runs TeX, expands author macros or certifies mathematical content. The
only supported template is a one-argument newenvironment whose opening changes
``\\the<carrier>`` to ``#1`` and enters a literal, numbered newtheorem carrier.
Declarations, calls and closing delimiters retain their original source bytes.
"""
from __future__ import annotations

import copy
import re

from pylatexenc.latexwalker import LatexWalkerEndOfStream

from . import rule_nodes as base
from . import rule_nodes_named_heading_project_v2 as named
from . import rule_nodes_named_headings_v2 as headings
from .rule_nodes_project_supplement_v3 import _declared_control_name

VERSION = "gangmu.rule.numbered-task-wrappers.v5"
_NAME = r"[A-Za-z]+"
_OPEN = re.compile(
    r"\s*\\renewcommand\s*(?:\\the(?P<plain>" + _NAME + r")\b|"
    r"\{\s*\\the(?P<braced>" + _NAME + r")\s*\})"
    r"\s*\{\s*#1\s*\}\s*(?P<entry>\\" + _NAME + r"\b|"
    r"\\begin\s*\{\s*" + _NAME + r"\s*\})\s*")
_MUTATION = re.compile(r"\\(?:def|gdef|edef|xdef|newcommand|renewcommand|providecommand|"
                       r"newenvironment|renewenvironment|newtheorem|let|futurelet)\b")
_DYNAMIC = re.compile(r"\\(?:catcode|csname|expandafter|scantokens)\b")
_ENV_TOKEN = re.compile(r"\\(?:begin|end)\b")


def _template(definition):
    body = headings._mask_noncontent(definition.get("body") or "")
    ending = headings._mask_noncontent(definition.get("end_body") or "")
    match = _OPEN.fullmatch(body)
    if not match:
        return None
    carrier = match["plain"] or match["braced"]
    entry = match["entry"]
    if entry.startswith("\\begin"):
        if not re.fullmatch(r"\\begin\s*\{\s*" + re.escape(carrier) + r"\s*\}", entry):
            return None
        expected_end = r"\s*\\end\s*\{\s*" + re.escape(carrier) + r"\s*\}\s*"
        invocation = "literal_begin_end"
    else:
        if entry != "\\" + carrier:
            return None
        expected_end = r"\s*\\end" + re.escape(carrier) + r"\b\s*"
        invocation = "literal_control_pair"
    if not re.fullmatch(expected_end, ending):
        return None
    return {"carrier_name": carrier, "carrier_invocation": invocation}


def _possible_carriers(definition, carrier_names):
    """Keep unsupported near-templates visible instead of silently dropping them."""
    body = headings._mask_noncontent(definition.get("body") or "")
    counter_names = set(re.findall(r"\\the(" + _NAME + r")\b", body))
    invoked = {name for name in carrier_names if re.search(
        r"\\" + re.escape(name) + r"\b|\\begin\s*\{\s*" + re.escape(name) + r"\s*\}", body)}
    return sorted(counter_names | (invoked if _MUTATION.search(body) else set()))


def _unindexed_wrapper_hints(project, extractor, definitions, carrier_names):
    """Bounded literal hints only; an omitted/inert declaration cannot add a task.

    E1 intentionally omits declarations in literal false branches. Preserve an
    unresolved active use of the same literal wrapper name even in that case.
    A hint inside an unused macro is likewise never an active declaration.
    """
    raw = headings._mask_noncontent(project.reading)
    lexer = base._Lexer(raw)
    indexed = {r[0] for d in definitions if (r := extractor._event_range(d))}
    hints = []
    for command in re.finditer(r"\\(?:newenvironment|renewenvironment)\b", raw):
        if command.start() in indexed:
            continue
        pos = lexer.skip(command.end())
        if raw[pos:pos + 1] == "*":
            pos += 1
        name = lexer.group(pos)
        if not name or not name["balanced"] or not re.fullmatch(_NAME, name["value"].strip()):
            continue
        pos = name["end"]
        for _ in range(2):
            option = lexer.group(pos, optional=True)
            if not option:
                break
            if not option["balanced"]:
                break
            pos = option["end"]
        opening = lexer.group(pos)
        if not opening or not opening["balanced"]:
            continue
        ending = lexer.group(opening["end"])
        if not ending or not ending["balanced"]:
            continue
        literal = {"body": opening["value"], "end_body": ending["value"]}
        template = _template(literal)
        possible = [template["carrier_name"]] if template else _possible_carriers(literal, carrier_names)
        if possible:
            hints.append({"environment_name": name["value"].strip(), "wrapper_definition_id": None,
                "carrier_names": possible, "literal_template_supported": template is not None,
                "declaration_global_active": False, "unindexed_source_hint_not_active_declaration": True,
                "source_spans": extractor._segments([(command.start(), ending["end"])])})
    return hints


def _condition_scope(project, mask, position):
    # Unlike v3's helper, account for primitive assignments of iff as well.
    overridden = any(d.get("name") == "iff" for d in project.structure_data.get("definitions", []))
    overridden |= bool(re.search(r"\\(?:let|futurelet)\s*\\iff\b", mask[:position]))
    if not overridden:
        mask = re.sub(r"\\iff\b", lambda m: " " * len(m.group(0)), mask)
    return named._unresolved_condition_scope(mask, position)


def _global_active(project, extractor, declaration):
    interval = extractor._event_range(declaration)
    return bool(interval and declaration.get("activity") == "active"
                and declaration.get("scope") == ["document"]
                and not declaration.get("condition_ids")
                and not _condition_scope(project, extractor._mask, interval[0]))


def _closed_environment(mask, interval, wrapper_name):
    """The E1 closed flag alone can include out-of-order nested environments."""
    stack = []
    body = mask[interval[0]:interval[1]]
    tokens = list(_ENV_TOKEN.finditer(body))
    for command in tokens:
        match = re.match(r"\\(begin|end)\s*\{\s*([A-Za-z@*][A-Za-z0-9@*_-]*)\s*\}",
                         body[command.start():])
        if not match:
            return False
        action, name = match.groups()
        if not stack and (action != "begin" or name != wrapper_name or command.start() != 0):
            return False
        if action == "begin":
            stack.append(name)
        elif not stack or stack.pop() != name:
            return False
        if not stack and command.start() + match.end() != len(body):
            return False
    if not tokens or stack:
        return False
    lexer, cursor, depth, math = base._Lexer(body), 0, 0, None
    while cursor < len(body):
        try:
            token = lexer.token(cursor)
        except LatexWalkerEndOfStream:
            break
        cursor = token.pos + token.len
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
    return depth == 0 and math is None


def _shadow_reasons(project, extractor, declaration, carrier, event_end):
    names = {declaration["name"], "end" + declaration["name"], carrier["name"],
             "end" + carrier["name"], "the" + carrier["name"],
             "renewcommand", "newenvironment", "newtheorem", "begin", "end"}
    allowed_ids = {declaration["id"], carrier["id"]}
    reasons = []
    for other in project.structure_data.get("definitions", []):
        interval = extractor._event_range(other)
        if (other.get("id") not in allowed_ids and other.get("activity") != "inactive"
                and interval and interval[0] < event_end
                and _declared_control_name(project, other) in names):
            reasons.append("protected_control_redefined_or_shadowed")
    visible = extractor._mask[:event_end]
    if any(re.search(r"\\(?:let|futurelet)\s*\\" + re.escape(name) + r"\b", visible)
           for name in names):
        reasons.append("protected_control_primitive_assignment")
    if _DYNAMIC.search(visible):
        reasons.append("dynamic_control_or_catcode_effect_unresolved")
    # No expansion is performed. A visible invocation of a macro that contains
    # a relevant assignment is itself enough to retain a source-review blocker.
    definitions = extractor.definitions
    for call in project.structure_data.get("macro_calls", []):
        interval = extractor._event_range(call)
        body = headings._mask_noncontent((definitions.get(call.get("definition_id")) or {}).get("body") or "")
        if (call.get("activity") != "inactive" and interval and interval[0] < event_end
                and _MUTATION.search(body) and any(re.search(r"\\" + re.escape(name) + r"\b", body) for name in names)):
            reasons.append("unexpanded_macro_may_shadow_protected_control")
    return sorted(set(reasons))


def discover_numbered_task_wrappers(project, existing_candidates):
    """Return source-preserving additions and blockers for strict literal wrappers.

    ``resolved_prior_structural_issue_event_ids`` contains only safely supported
    closed use events. The caller must preserve and archive any prior issue it
    chooses to replace; this helper does not change readiness or remove issues.
    """
    project.assert_unchanged()
    if not isinstance(existing_candidates, list) or any(not isinstance(c, dict) for c in existing_candidates):
        raise ValueError("Existing candidates must be a list of source-bound candidate objects")
    incoming_hash = named._value_sha256(existing_candidates)
    extractor = base.RuleNodeExtractor(project)
    definitions = list(project.structure_data.get("definitions", []))
    carrier_names = {d.get("name") for d in definitions if d.get("theorem_like") is True}
    wrapper_definitions = []
    declarations = []
    for definition in definitions:
        if definition.get("definition_kind") != "environment" or definition.get("theorem_like"):
            continue
        template = _template(definition)
        possible = [template["carrier_name"]] if template else _possible_carriers(definition, carrier_names)
        if not possible:
            continue
        declaration = {"environment_name": definition["name"], "wrapper_definition_id": definition["id"],
            "carrier_names": possible, "literal_template_supported": template is not None,
            "declaration_global_active": _global_active(project, extractor, definition),
            "source_spans": [{**copy.deepcopy(definition["source"]), "occurrence": definition["file_occurrence"]}]}
        declarations.append(declaration)
        wrapper_definitions.append((definition, template, declaration))
    additions, obligations, resolved, covered = [], [], [], []
    wrappers_by_name = {}
    for item in wrapper_definitions:
        wrappers_by_name.setdefault(item[0]["name"], []).append(item)
    hints = _unindexed_wrapper_hints(project, extractor, definitions, carrier_names)
    declarations.extend(hints)
    hinted_names = {h["environment_name"] for h in hints}
    for event in project.structure_data.get("environments", []):
        if event.get("activity") == "inactive":
            continue
        if event.get("name") not in wrappers_by_name:
            if event.get("name") in hinted_names:
                interval = extractor._event_range(event)
                obligations.append({"kind": "numbered_task_wrapper_source_scope_unresolved", "event_id": event["id"],
                    "wrapper_definition_id": None, "use_definition_id": event.get("definition_id"),
                    "source_spans": extractor._segments([interval]) if interval else [
                        {**copy.deepcopy(event["source"]), "occurrence": event["file_occurrence"]}],
                    "reading_ranges": [{"char_start": interval[0], "char_end": interval[1]}] if interval else [],
                    "reasons": ["wrapper_declaration_not_indexed_active_definition"],
                    "blocking_until_source_scope_review": True,
                    "obligation": "The literal source hint is not an indexed active wrapper declaration; establish the declaration and use scope without assuming macro execution."})
            continue
        options = wrappers_by_name[event["name"]]
        selected = next((x for x in options if x[0]["id"] == event.get("definition_id")), None)
        # A later declaration must not manufacture an earlier task. Still return
        # a blocker for this relevant source event when no selected declaration exists.
        definition, template, declaration = selected or options[0]
        interval = extractor._event_range(event)
        reasons = []
        if not selected:
            reasons.append("use_definition_id_does_not_match_wrapper")
        if not template:
            reasons.append("wrapper_is_not_exact_counter_renaming_template")
        if (definition.get("command") != "newenvironment" or definition.get("starred") is not False
                or type(definition.get("argument_count")) is not int or definition.get("argument_count") != 1
                or definition.get("optional_default") is not None
                or not re.fullmatch(_NAME, definition.get("name") or "")):
            reasons.append("wrapper_declaration_form_unsupported")
        if not declaration["declaration_global_active"]:
            reasons.append("wrapper_declaration_not_global_unconditional_active")
        if not interval or event.get("closed") is not True:
            reasons.append("use_not_complete_closed_mapped_environment")
        if event.get("activity") != "active" or event.get("condition_ids"):
            reasons.append("use_activity_unresolved")
        if event.get("source_role") != "document_or_bibliography":
            reasons.append("use_not_document_source")
        args = event.get("arguments", [])
        if (len(args) != 1 or not re.fullmatch(r"[A-Za-z0-9]+(?:[.:-][A-Za-z0-9]+)*",
                                             (args[0].get("value") or "").strip())):
            reasons.append("displayed_number_argument_not_literal")
        carrier = None
        if template:
            carriers = [d for d in definitions if d.get("name") == template["carrier_name"]
                        and d.get("definition_kind") == "environment" and d.get("theorem_like") is True
                        and d.get("activity") != "inactive"]
            before_use = [d for d in carriers if (r := extractor._event_range(d)) and interval and r[1] <= interval[0]]
            if len(before_use) == 1:
                carrier = before_use[0]
            else:
                reasons.append("carrier_declaration_missing_late_or_ambiguous")
        if carrier:
            title = (carrier.get("body") or "").strip()
            if (carrier.get("command") != "newtheorem" or carrier.get("starred") is not False
                    or not re.fullmatch(r"[A-Za-z][A-Za-z \t-]{0,79}", title)
                    or base.category(title) not in {"statement", "definition"}):
                reasons.append("carrier_not_literal_numbered_mathematical_type")
            if not _global_active(project, extractor, carrier):
                reasons.append("carrier_declaration_not_global_unconditional_active")
            wrapper_range = extractor._event_range(definition)
            carrier_range = extractor._event_range(carrier)
            if not (interval and wrapper_range and carrier_range
                    and carrier_range[1] <= wrapper_range[0] and wrapper_range[1] <= interval[0]):
                reasons.append("carrier_wrapper_use_order_unresolved")
            if interval:
                reasons.extend(_shadow_reasons(project, extractor, definition, carrier, interval[1]))
        if interval:
            if _condition_scope(project, extractor._mask, interval[0]):
                reasons.append("use_visible_conditional_scope_unresolved")
            # Unknown conditional syntax inside the body can change whether its
            # end marker executes. Literal activity metadata at begin is insufficient.
            body = extractor._mask[interval[0]:interval[1]]
            iff_overridden = any(d.get("name") == "iff" for d in definitions) or bool(
                re.search(r"\\(?:let|futurelet)\s*\\iff\b", extractor._mask[:interval[1]]))
            if not iff_overridden:
                body = re.sub(r"\\iff\b", lambda m: " " * len(m.group(0)), body)
            if re.search(r"\\(?:if[A-Za-z]*|else|or|fi)\b", body):
                reasons.append("conditional_syntax_within_use_unresolved")
            if not _closed_environment(extractor._mask, interval, event["name"]):
                reasons.append("literal_environment_nesting_or_end_unresolved")
        if reasons:
            obligations.append({"kind": "numbered_task_wrapper_source_scope_unresolved", "event_id": event["id"],
                "wrapper_definition_id": definition["id"], "use_definition_id": event.get("definition_id"),
                "source_spans": extractor._segments([interval]) if interval else [
                    {**copy.deepcopy(event["source"]), "occurrence": event["file_occurrence"]}],
                "reading_ranges": [{"char_start": interval[0], "char_end": interval[1]}] if interval else [],
                "reasons": sorted(set(reasons)), "blocking_until_source_scope_review": True,
                "obligation": "Establish the exact literal wrapper, carrier, declaration order, activity, absence of control overrides and complete source boundary."})
            continue
        resolved.append(event["id"])
        if any(c.get("scope_range") == {"char_start": interval[0], "char_end": interval[1]}
               and c.get("reading_ranges") == [{"char_start": interval[0], "char_end": interval[1]}]
               for c in existing_candidates):
            covered.append(event["id"])
            continue
        row = extractor._candidate(*interval, title=title,
            family="literal_numbered_task_wrapper_environment", boundary_status="complete_closed_environment",
            basis=VERSION + ":exact_literal_display_number_only_wrapper", event=event, activity="active",
            environment_name=event["name"], definition_id=definition["id"], carrier_definition_id=carrier["id"],
            carrier_environment_name=carrier["name"], displayed_number_arguments=copy.deepcopy(event.get("arguments", [])),
            wrapper_declaration_source_spans=copy.deepcopy(declaration["source_spans"]),
            carrier_declaration_source_spans=[{**copy.deepcopy(carrier["source"]), "occurrence": carrier["file_occurrence"]}])
        additions.append(row)
    if named._value_sha256(existing_candidates) != incoming_hash:
        raise AssertionError("Numbered wrapper discovery changed incoming candidates")
    project.assert_unchanged()
    return {"schema_version": VERSION, "candidates": additions, "obligations": obligations,
        "declarations": declarations, "resolved_prior_structural_issue_event_ids": resolved,
        "already_covered_event_ids": covered, "source_fingerprint": copy.deepcopy(project.fingerprint),
        "code": named._file_binding(__file__), "incoming_candidates_value_sha256": incoming_hash,
        "supplier_starts": 0, "TeX_executions": 0, "arbitrary_macro_expansions": 0}
