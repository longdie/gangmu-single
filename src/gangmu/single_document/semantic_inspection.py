"""Bounded resolved-secret inspection of JSON mathematical responses.

Container shape and JSON/Unicode encoding have independent finite budgets.
All keys, strings and nested encodings remain inspected.  Changed or omitted
data is never accepted as a mathematical response.  No credentials are loaded
here: callers supply their resolved-secret redactor.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
import math
import re
from typing import Any, Callable


@dataclass(frozen=True)
class InspectionLimits:
    max_container_depth: int = 256
    max_encoding_depth: int = 32
    max_nodes: int = 100_000
    max_inspected_utf8_bytes: int = 64 * 1024 * 1024
    max_json_text_utf8_bytes: int = 16 * 1024 * 1024

    def __post_init__(self):
        for item in vars(self).values():
            if type(item) is not int or item <= 0:
                raise ValueError("inspection limits must be positive integers")


class _CannotInspect(Exception):
    """Carries only a fixed public code, never input material."""


_UNICODE_ESCAPE = re.compile(r"\\+u([0-9a-fA-F]{4})")


def decode_unicode_runs(value: str) -> str:
    """The legacy Unicode-escape semantics, with linear slash-run scanning.

    Applying a greedy ``\\+u`` regexp to a long nonmatching slash run can
    repeatedly backtrack. Each source character is consumed at most once here.
    """
    pieces = []
    position = copied = 0
    hexdigits = "0123456789abcdefABCDEF"
    while position < len(value):
        start = value.find("\\", position)
        if start < 0:
            break
        end = start + 1
        while end < len(value) and value[end] == "\\":
            end += 1
        digits = value[end + 1:end + 5]
        if value[end:end + 1] == "u" and len(digits) == 4 and all(char in hexdigits for char in digits):
            pieces.extend((value[copied:start], chr(int(digits, 16))))
            position = copied = end + 5
        else:
            position = end
    if not pieces:
        return value
    pieces.append(value[copied:])
    return "".join(pieces)


class _Inspection:
    def __init__(self, redact: Callable[[Any], Any], limits: InspectionLimits):
        self.redact = redact
        self.limits = limits
        self.nodes = 0
        self.utf8_bytes = 0
        self.max_container_seen = 0
        self.max_encoding_seen = 0
        self.json_decodes = 0
        self.unicode_decodes = 0
        self.active_containers: set[int] = set()

    def charge_text(self, text: str) -> int:
        # Reject cheaply before allocating an obviously oversized UTF-8 copy.
        remaining = self.limits.max_inspected_utf8_bytes - self.utf8_bytes
        if len(text) > remaining:
            raise _CannotInspect("byte_limit")
        try:
            size = len(text.encode("utf-8"))
        except UnicodeError:
            raise _CannotInspect("invalid_unicode") from None
        self.utf8_bytes += size
        if self.utf8_bytes > self.limits.max_inspected_utf8_bytes:
            raise _CannotInspect("byte_limit")
        return size

    def literal(self, value: str) -> tuple[str, bool]:
        # Quoting preserves ordinary mathematical URLs and token-named fields.
        # It does not suppress matching an actual resolved secret value.
        framed = '"' + value + '"'
        self.charge_text(framed)
        try:
            result = self.redact(framed)
        except Exception:
            raise _CannotInspect("redactor_failure") from None
        if not isinstance(result, str):
            raise _CannotInspect("redactor_nontext")
        changed = result != framed
        if changed:
            self.charge_text(result)
        if not result.startswith('"') or not result.endswith('"'):
            return "[REDACTED]", True
        return result[1:-1], changed

    def json_preflight(self, value: str, container_depth: int, byte_size: int):
        """Bound JSON allocation before loads, using quote-aware lexical work.

        Every valid JSON value/key contributes at least one lexical token; both
        decoding and subsequent inspection remain charged independently. This
        scanner is a resource preflight, not a JSON-validity replacement.
        """
        if byte_size > self.limits.max_json_text_utf8_bytes:
            raise _CannotInspect("json_text_byte_limit")
        available = self.limits.max_nodes - self.nodes
        tokens = nesting = 0
        quoted = escaped = primitive = False
        for char in value:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
                continue
            if char == '"':
                quoted, primitive = True, False
                tokens += 1
            elif char in "[{":
                nesting += 1
                tokens += 1
                primitive = False
                if container_depth + nesting > self.limits.max_container_depth:
                    raise _CannotInspect("container_depth_limit")
            elif char in "]}":
                nesting -= 1
                primitive = False
            elif char.isspace() or char in ",:":
                primitive = False
            elif not primitive:
                primitive = True
                tokens += 1
            if tokens > available:
                raise _CannotInspect("json_node_preflight_limit")

    def serialize(self, value: Any) -> str:
        try:
            encoded = json.dumps(value, ensure_ascii=False, allow_nan=False)
        except (ValueError, TypeError, RecursionError, UnicodeError):
            raise _CannotInspect("serialization_failure") from None
        self.charge_text(encoded)
        return encoded

    def visit(self, value: Any, container_depth: int, encoding_depth: int):
        self.nodes += 1
        if self.nodes > self.limits.max_nodes:
            raise _CannotInspect("node_limit")
        if encoding_depth > self.limits.max_encoding_depth:
            raise _CannotInspect("encoding_depth_limit")
        self.max_encoding_seen = max(self.max_encoding_seen, encoding_depth)
        if isinstance(value, (dict, list)):
            child_depth = container_depth + 1
            self.max_container_seen = max(self.max_container_seen, child_depth)
            if child_depth > self.limits.max_container_depth:
                raise _CannotInspect("container_depth_limit")
            identity = id(value)
            if identity in self.active_containers:
                raise _CannotInspect("cyclic_container")
            self.active_containers.add(identity)
            try:
                changed = False
                if isinstance(value, dict):
                    output = {}
                    for key, item in value.items():
                        if not isinstance(key, str):
                            raise _CannotInspect("nonstring_key")
                        safe_key, key_changed = self.visit(key, child_depth, encoding_depth)
                        safe_item, item_changed = self.visit(item, child_depth, encoding_depth)
                        if safe_key in output:
                            raise _CannotInspect("redacted_key_collision")
                        output[safe_key] = safe_item
                        changed |= key_changed or item_changed
                else:
                    output = []
                    for item in value:
                        safe_item, item_changed = self.visit(item, child_depth, encoding_depth)
                        output.append(safe_item)
                        changed |= item_changed
                return output, changed
            finally:
                self.active_containers.remove(identity)
        if not isinstance(value, str):
            if not (value is None or isinstance(value, (bool, int))
                    or isinstance(value, float) and math.isfinite(value)):
                raise _CannotInspect("non_json_scalar")
            if isinstance(value, int) and value.bit_length() > 4 * (
                    self.limits.max_inspected_utf8_bytes - self.utf8_bytes):
                raise _CannotInspect("byte_limit")
            try:
                scalar_text = json.dumps(value, allow_nan=False)
            except (ValueError, TypeError):
                raise _CannotInspect("scalar_serialization_failure") from None
            self.charge_text(scalar_text)
            return value, False

        byte_size = self.charge_text(value)
        safe, changed = self.literal(value)
        # Continue inspecting even after a literal secret was found: a string
        # can contain both a plain secret and another encoded occurrence.
        inspect_size = self.charge_text(safe) if changed else byte_size
        if safe.lstrip().startswith(("{", "[", '"')):
            self.json_preflight(safe, container_depth, inspect_size)
            try:
                def no_constant(_):
                    raise ValueError("nonfinite JSON")
                def unique_pairs(pairs):
                    output = {}
                    for key, item in pairs:
                        if key in output:
                            raise _CannotInspect("json_duplicate_key")
                        output[key] = item
                    return output
                decoded = json.loads(safe, parse_constant=no_constant, object_pairs_hook=unique_pairs)
            except (ValueError, TypeError):
                decoded = None
            except RecursionError:
                raise _CannotInspect("json_parser_recursion") from None
            if isinstance(decoded, (dict, list, str)):
                self.json_decodes += 1
                safe_decoded, inner_changed = self.visit(decoded, container_depth, encoding_depth + 1)
                if inner_changed:
                    safe = self.serialize(safe_decoded)
                    changed = True
        candidate = decode_unicode_runs(safe)
        if candidate != safe:
            self.unicode_decodes += 1
            safe_candidate, inner_changed = self.visit(candidate, container_depth, encoding_depth + 1)
            if inner_changed:
                safe, changed = safe_candidate, True
        # Exact original spelling is retained if all inspection paths were safe.
        return safe if changed else value, changed


def inspect_bounded(value: Any, redact: Callable[[Any], Any], *, limits=None, initial_depth=0):
    state = _Inspection(redact, limits or InspectionLimits())
    reason = None
    try:
        if type(initial_depth) is not int or initial_depth < 0:
            raise _CannotInspect("invalid_initial_depth")
        # Historical explicit depth conservatively reserves both budgets.
        result, changed = state.visit(value, initial_depth, initial_depth)
    except _CannotInspect as exc:
        reason = str(exc)
        result, changed = f"[OMITTED: secret inspection {reason}]", True
    except (RecursionError, MemoryError):
        reason = "runtime_resource_limit"
        result, changed = "[OMITTED: secret inspection runtime_resource_limit]", True
    report = {
        "reason": reason,
        "nodes": state.nodes,
        "inspected_utf8_bytes": state.utf8_bytes,
        "max_container_depth": state.max_container_seen,
        "max_encoding_depth": state.max_encoding_seen,
        "json_decodes": state.json_decodes,
        "unicode_decodes": state.unicode_decodes,
        "limits": vars(state.limits),
    }
    return result, changed, report


def redact_semantic_value_bounded(value: Any, redact: Callable[[Any], Any], *, limits=None, initial_depth=0):
    result, changed, _ = inspect_bounded(value, redact, limits=limits, initial_depth=initial_depth)
    return result, changed
