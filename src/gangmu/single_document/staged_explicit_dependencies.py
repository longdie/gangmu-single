"""Explicit-reference decisions over immutable, already extracted nodes.

Syntax discovers a ledger, never a dependency.  Every occurrence/key remains
accounted for, including external, ambiguous and missing endpoints.  No lexical
retrieval, node extraction, source rewriting or semantic scoring occurs here.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
import re

from gangmu.representation.model import Evidence, Graph, Source, digest
from gangmu.representation.graph import GraphIndex
from .candidates import _reference_sources
from .inventory import SourceInventory
from . import explicit_reference_inventory as explicit_inventory

VERSION = "gangmu.e4.staged-explicit-dependencies.v1"
ROOT = Path(__file__).resolve().parents[3]
PROMPT = ROOT / "prompts/single_document/staged_explicit_dependencies/decide_v1.md"
TYPES = ("statement_dependency", "proof_dependency", "proves")
DISPOSITIONS = ("accepted", "rejected", "unknown")


def file_record(path):
    path = Path(path).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Explicit-dependency inputs must stay in the project")
    return {"path": str(path.relative_to(ROOT)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _span(row):
    return {key: row[key] for key in ("file", "char_start", "char_end", "occurrence")
            if row.get(key) is not None}


def _intersects(a, b):
    return (a["file"] == b["file"] and
            (a.get("occurrence") is None or b.get("occurrence") is None or a["occurrence"] == b["occurrence"])
            and max(a["char_start"], b["char_start"]) < min(a["char_end"], b["char_end"]))


def validate_nodes(project, nodes):
    """Retain every Unit/Context and check their exact original source bytes."""
    if (nodes.get("paper_id") != project.paper_id or nodes.get("source_fingerprint") != project.fingerprint
            or nodes.get("documents") != [project.document.model_dump(mode="json")]):
        raise ValueError("Frozen node result belongs to a different pinned source")
    units = nodes.get("units", [])
    contexts = nodes.get("contexts", [])
    if not units or len({unit["id"] for unit in units}) != len(units):
        raise ValueError("Frozen Units must be nonempty and uniquely identified")
    if len({context["id"] for context in contexts}) != len(contexts):
        raise ValueError("Frozen Context IDs must be unique")
    graph = Graph.model_validate_json(json.dumps({"snapshot_id": "explicit-frozen-node-validation",
                  "schema_version": "gangmu.mathir.v4", "documents": nodes["documents"], "contexts": contexts,
                  "units": units, "relations": [], "evidence": [], "identity_views": []}))
    by_context = {context["id"]: context for context in contexts}
    for unit in units:
        if unit["context"] not in by_context:
            raise ValueError("Frozen Unit has no existing Context")
        if unit["original_text"] != [project.resolve_span(_span(span)) for span in unit["sources"]]:
            raise ValueError("Frozen Unit original text differs from pinned author source")
    for context in contexts:
        for span in context["sources"]:
            project.resolve_span(_span(span))
        seen, cursor = set(), context
        while cursor is not None:
            if cursor["id"] in seen:
                raise ValueError("Frozen Context ancestry is cyclic")
            seen.add(cursor["id"])
            parent = cursor.get("parent")
            if parent is not None and parent not in by_context:
                raise ValueError("Frozen Context parent is missing")
            cursor = by_context.get(parent)
    project.assert_unchanged()
    return graph


def _node_adapter(unit):
    return {**unit, "spans": [_span(source) for source in unit["sources"]]}


def _literal_locator_candidates(project, units, reference):
    """Source literal headings/tags only; never recompute automatic counters."""
    locator = reference.get("locator") or {}
    number, kind = locator.get("number"), locator.get("kind")
    if not number:
        return []
    number_re = re.escape(number)
    heading_kinds = r"(?:Theorem|Lemma|Proposition|Corollary|Definition|Claim|Conjecture|Equation)"
    header = re.compile(r"\b(" + heading_kinds + r"(?:\s*(?:-|/|and|&)\s*" + heading_kinds + r")*)"
                        r"(?:\s|~)+\(?" + number_re + r"\)?(?=$|[\s.:({])", re.I)
    tag = re.compile(r"\\tag\*?\s*\{\s*" + number_re + r"\s*\}"
                     r"|\\(?:eqno|leqno)\s*\(?\s*" + number_re + r"\s*\)?(?=$|[^0-9A-Za-z.])")
    candidates = []
    for unit in units:
        if unit["category"] == "proof":
            continue
        for span in unit["spans"]:
            text = project.resolve_span(span)
            # The heading belongs at the beginning; a mention inside a theorem
            # or a Context cannot rename that node as the referenced endpoint.
            prefix = re.sub(r"\\(?:noindent|par|smallskip|medskip|bigskip)\b|[{}]", " ", text[:500])
            match = header.search(prefix)
            if (match is not None and kind in {value.lower() for value in re.findall(heading_kinds, match[1], re.I)}
                    and not re.sub(r"\\(?:textbf|textit|textsc|emph|bf|it|sc)\b|\s+", "", prefix[:match.start()])):
                candidates.append({"node_id": unit["id"], "basis": "literal_author_numbered_heading",
                                   "matched_source_spans": [span]})
                break
            if kind == "equation" and tag.search(text):
                candidates.append({"node_id": unit["id"], "basis": "literal_author_equation_tag",
                                   "matched_source_spans": [span]})
                break
    return candidates


def _context_sources(unit, contexts):
    result, seen, cursor = [], set(), unit["context"]
    while cursor is not None:
        if cursor in seen or cursor not in contexts:
            raise ValueError("Frozen Context ancestry changed")
        seen.add(cursor)
        record = contexts[cursor]
        result.extend({"context_id": cursor, "span": _span(span)} for span in record["sources"])
        cursor = record.get("parent")
    return result


def build_scan(project, nodes, *, references=None, locator_bindings=None):
    validate_nodes(project, nodes)
    rows = copy.deepcopy(explicit_inventory.scan_references(project) if references is None else references)
    # A caller may supply a separately reviewed scanner, but its entire ledger
    # is retained and content-bound; no subset is accepted as the full ledger.
    if len({row["item_id"] for row in rows}) != len(rows):
        raise ValueError("Reference ledger IDs are not unique")
    units = [_node_adapter(unit) for unit in nodes["units"]]
    by_unit = {unit["id"]: unit for unit in units}
    contexts = {context["id"]: context for context in nodes["contexts"]}
    locator_candidates, locator_records = [], []
    for bound in locator_bindings or []:
        if set(bound) != {"artifact", "data"}:
            raise ValueError("Compiled locator input needs exact retained artifact and parsed data")
        artifact = bound["artifact"]
        path = ROOT / artifact["path"]
        if file_record(path) != artifact or json.loads(path.read_bytes()) != bound["data"]:
            raise ValueError("Compiled locator artifact bytes differ")
        value = bound["data"]
        node_binding = value.get("node_result_binding", {})
        if set(node_binding) != {"path", "sha256"}:
            raise ValueError("Compiled numbering must bind the exact source node result")
        source_nodes_path = ROOT / node_binding["path"]
        if (value.get("paper_id") != project.paper_id or value.get("source_fingerprint") != project.fingerprint
                or value.get("includes_dependency_labels_or_relation_obligations") is not False
                or value.get("reextracts_nodes") is not False or value.get("supplier_calls") != 0
                or file_record(source_nodes_path) != node_binding
                or json.loads(source_nodes_path.read_bytes()) != nodes):
            raise ValueError("Locator input must bind the same complete frozen node/source and contain no relation gold")
        source_identity = value.get("source_byte_identity", {})
        if (source_identity.get("exact_bytes_equal") is not True
                or source_identity.get("source_sha256") != source_identity.get("compiled_source_sha256")):
            raise ValueError("Compiled locator requires unmodified byte-identical author source")
        original = ROOT / source_identity["original_source_path"]
        compiled = ROOT / source_identity["compiled_source_path"]
        if (file_record(original)["sha256"] != source_identity["source_sha256"]
                or file_record(compiled)["sha256"] != source_identity["compiled_source_sha256"]
                or original.read_bytes() != compiled.read_bytes()):
            raise ValueError("Original and compiled source bytes no longer match")
        for name in ("compiled_pdf_binding", "pdf_text_binding"):
            material = value[name]
            if file_record(ROOT / material["path"]) != {key: material[key] for key in ("path", "sha256")}:
                raise ValueError("Retained compiled PDF/layout bytes differ")
        if value["compiled_pdf_binding"].get("source_patches") != []:
            raise ValueError("Patched-source compilation cannot resolve a frozen source locator")
        locator_records.append(copy.deepcopy(artifact))
        for candidate in value.get("candidate_locators", []):
            identifier = candidate.get("unit_id")
            anchors = candidate.get("source_anchor_spans", [])
            if (identifier not in by_unit or not isinstance(candidate.get("kind"), str)
                    or not isinstance(candidate.get("number"), str) or not anchors):
                raise ValueError("Every compiled locator needs an existing fixed unit and original-source anchors")
            normalized = [project.normalize_span(_span(anchor)) for anchor in anchors]
            if any(not _covered(anchor, by_unit[identifier]["spans"]) for anchor in normalized):
                raise ValueError("Compiled locator anchor is outside the selected complete unit body")
            originals = candidate.get("source_anchor_original_text", candidate.get("raw_text"))
            if isinstance(originals, str):
                originals = [originals]
            if originals != [project.resolve_span(anchor) for anchor in normalized]:
                raise ValueError("Compiled locator anchor raw text differs from pinned author source")
            locator_candidates.append({**copy.deepcopy(candidate), "source_anchor_spans": normalized,
                                       "artifact_binding": copy.deepcopy(artifact)})
    tasks, exclusions = [], []
    for reference in rows:
        use = reference["span"]
        users = []
        for unit in units:
            body = [span for span in unit["spans"] if _intersects(span, use)]
            inherited = [row for row in _context_sources(unit, contexts) if _intersects(row["span"], use)]
            if body or inherited:
                users.append({"node_id": unit["id"], "body_occurrence": bool(body),
                              "context_ids": sorted({row["context_id"] for row in inherited})})
        keys = list(dict.fromkeys(reference.get("keys", []))) or [None]
        for key in keys:
            task_reference = copy.deepcopy(reference)
            if (task_reference.get("locator", {}).get("kind") == "case"
                    and isinstance(key, str) and key.startswith("text:case:")):
                task_reference["locator"]["number"] = key[len("text:case:"):]
            endpoints = [endpoint for endpoint in reference.get("targets", []) if endpoint.get("key") == key]
            sources = {}
            for endpoint in endpoints:
                for unit, basis, owner_ids in _reference_sources(units, endpoint):
                    sources.setdefault(unit["id"], {"node_id": unit["id"], "bases": [], "owner_ids": []})
                    sources[unit["id"]]["bases"].append(basis)
                    sources[unit["id"]]["owner_ids"].extend(owner_ids)
            for candidate in _literal_locator_candidates(project, units, task_reference):
                sources.setdefault(candidate["node_id"], {"node_id": candidate["node_id"], "bases": [], "owner_ids": []})
                sources[candidate["node_id"]]["bases"].append(candidate["basis"])
            locator = task_reference.get("locator") or {}
            if locator.get("kind") == "case":
                for unit in units:
                    if unit["category"] != "proof":
                        continue
                    anchors = [row["span"] for row in rows if key in row.get("keys", [])
                        and row.get("locator", {}).get("kind") == "case"
                        and any(_intersects(span, row["span"]) for span in unit["spans"])]
                    if anchors:
                        sources.setdefault(unit["id"], {"node_id": unit["id"], "bases": [], "owner_ids": []})
                        sources[unit["id"]]["bases"].append("literal_case_locator_carried_by_existing_proof_requires_review")
                        sources[unit["id"]]["case_locator_source_spans"] = anchors
            for candidate in locator_candidates:
                if (candidate["kind"], candidate["number"]) == (locator.get("kind"), locator.get("number")):
                    identifier = candidate["unit_id"]
                    sources.setdefault(identifier, {"node_id": identifier, "bases": [], "owner_ids": []})
                    sources[identifier]["bases"].append("same_source_compiled_numbering_anchor_candidate_only")
                    sources[identifier].setdefault("compiled_locator_candidates", []).append(copy.deepcopy(candidate))
            proof_sources = []
            for hint in nodes.get("proof_attachment_hints", []):
                for statement_id in hint.get("candidate_target_unit_ids", []):
                    if statement_id in sources and hint.get("proof_unit_id") in by_unit:
                        proof_sources.append({"node_id": hint["proof_unit_id"], "statement_id": statement_id,
                                              "basis": "frozen_attachment_hint_requires_proves_review"})
            task = {"task_id": "explicit-task:" + digest({"source": project.fingerprint,
                    "reference_id": reference["item_id"], "key": key})[:24], "kind": "reference",
                    "reference_id": reference["item_id"], "reference_key": key,
                    "reference": task_reference, "referenced_endpoints": endpoints,
                    "premise_candidates": sorted(sources.values(), key=lambda row: row["node_id"]),
                    "proof_premise_candidates": proof_sources, "user_candidates": users,
                    "endpoint_status": "located_candidates" if sources else "missing_premise_endpoint",
                    "user_status": "located_candidates" if users else "missing_user_endpoint"}
            if endpoints and all(endpoint.get("resolution") == "external" for endpoint in endpoints):
                exclusions.append({**task, "disposition": "external_endpoint",
                    "reason": "External locator has no single-document mathematical endpoint; actual external use remains unjudged."})
            else:
                tasks.append(task)
    for hint in nodes.get("proof_attachment_hints", []):
        proof_id = hint.get("proof_unit_id")
        targets = [identifier for identifier in hint.get("candidate_target_unit_ids", []) if identifier in by_unit]
        if proof_id not in by_unit or by_unit[proof_id]["category"] != "proof":
            raise ValueError("Frozen proof attachment hint has a missing/nonproof endpoint")
        tasks.append({"task_id": "proves-task:" + digest({"nodes": digest(nodes), "hint": hint})[:24],
            "kind": "proves", "reference_id": None, "reference_key": None, "reference": None,
            "referenced_endpoints": [], "premise_candidates": [{"node_id": proof_id, "bases": [hint["basis"]], "owner_ids": []}],
            "proof_premise_candidates": [], "user_candidates": [{"node_id": identifier, "body_occurrence": True,
                "context_ids": []} for identifier in targets], "attachment_hint": copy.deepcopy(hint),
            "endpoint_status": "located_candidates", "user_status": "located_candidates" if targets else "missing_user_endpoint"})
    expected_pairs = [(row["item_id"], key) for row in rows for key in list(dict.fromkeys(row.get("keys", []))) or [None]]
    actual_pairs = [(task["reference_id"], task["reference_key"]) for task in tasks + exclusions if task["kind"] == "reference"]
    if Counter(expected_pairs) != Counter(actual_pairs):
        raise AssertionError("Every reference occurrence/key must have exactly one retained task")
    result = {"schema_version": VERSION + ".scan", "paper_id": project.paper_id,
              "source_fingerprint": project.fingerprint, "nodes_value_sha256": digest(nodes),
              "reference_ledger": rows, "reference_ledger_value_sha256": digest(rows),
              "scanner_bindings": {"explicit_scanner": file_record(explicit_inventory.__file__),
                                   "historical_inventory": file_record(ROOT / "src/gangmu/single_document/inventory.py")},
              "compiled_locator_artifacts": locator_records,
              "compiled_locator_candidates": locator_candidates,
              "tasks": tasks, "structural_exclusions": exclusions,
              "counts": {"references": len(rows), "reference_keys": len(expected_pairs),
                         "model_reference_tasks": sum(task["kind"] == "reference" for task in tasks),
                         "proves_tasks": sum(task["kind"] == "proves" for task in tasks),
                         "external_keys": len(exclusions), "all_keys_retained_once": True},
              "implicit_retrieval": False, "node_extraction": False, "quality_accepted": False}
    result["scan_value_sha256"] = digest(result)
    return result


def _materials(project, nodes, task):
    units = {unit["id"]: unit for unit in nodes["units"]}
    contexts = {context["id"]: context for context in nodes["contexts"]}
    source_menu, node_menu = {}, {}

    def add_source(span, role):
        normalized = project.normalize_span(_span(span))
        identifier = "source:" + digest({"source": project.fingerprint, "span": normalized})[:24]
        if identifier not in source_menu:
            source_menu[identifier] = {"source_id": identifier, "span": normalized,
                "text": project.resolve_span(normalized), "roles": []}
        if role not in source_menu[identifier]["roles"]:
            source_menu[identifier]["roles"].append(role)
        return identifier

    ids = sorted({row["node_id"] for name in ("premise_candidates", "proof_premise_candidates", "user_candidates")
                  for row in task[name]})
    for identifier in ids:
        unit = units[identifier]
        records = []
        cursor, seen = unit["context"], set()
        while cursor is not None:
            if cursor in seen:
                raise ValueError("Context cycle in frozen source material")
            seen.add(cursor)
            context = contexts[cursor]
            records.append({**{key: context.get(key) for key in ("id", "parent", "description", "symbols", "assumptions")},
                            "source_ids": [add_source(span, "context:" + cursor) for span in context["sources"]]})
            cursor = context.get("parent")
        node_menu[identifier] = {"id": identifier, "category": unit["category"], "label": unit["label"],
            "source_ids": [add_source(span, "node:" + identifier) for span in unit["sources"]], "contexts": records}
    reference = task.get("reference")
    if reference:
        use = reference["span"]
        add_source(use, "exact_reference_occurrence")
        fragments = project.decision_fragments([use], reference=True)
        if fragments:
            for fragment in fragments:
                for span in fragment:
                    add_source(span, "complete_reference_use_clause")
        else:
            # A literal locator can be organizational markup. Keep a full
            # paragraph/window so rejection is judged against actual source.
            text = project._read_text(use["file"])
            left = text.rfind("\n\n", 0, use["char_start"])
            right = text.find("\n\n", use["char_end"])
            add_source({**use, "char_start": max(0, left + 2), "char_end": len(text) if right < 0 else right},
                       "complete_reference_paragraph")
        for span in reference.get("definition_spans", []):
            add_source(span, "reference_wrapper_definition")
        for endpoint in task["referenced_endpoints"]:
            for span in endpoint.get("owner_source_spans", []) + endpoint.get("label_spans", []):
                add_source(span, "locator_endpoint_author_source")
            for candidate in endpoint.get("source_section_candidates", []):
                for span in candidate["source_spans"]:
                    add_source(span, "unclassified_source_section_envelope_requires_review")
    return {"task": copy.deepcopy(task), "node_menu": list(node_menu.values()), "source_menu": list(source_menu.values()),
            "frozen_node_index": [{"id": unit["id"], "category": unit["category"], "label": unit["label"]}
                                  for unit in nodes["units"]]}


def response_schema(task_ids):
    nullable = lambda kind: {"anyOf": [{"type": kind}, {"type": "null"}]}
    evidence = {"type": "object", "additionalProperties": False, "required": ["source_id", "quote"],
                "properties": {"source_id": {"type": "string"}, "quote": nullable("string")}}
    use = {"type": "object", "additionalProperties": False,
           "required": ["source", "target", "type", "explanation", "evidence"], "properties": {
               "source": {"type": "string"}, "target": {"type": "string"}, "type": {"type": "string", "enum": list(TYPES)},
               "explanation": {"type": "string"}, "evidence": {"type": "array", "items": evidence}}}
    source_request = {"type": "object", "additionalProperties": False, "required": ["kind", "value"],
        "properties": {"kind": {"type": "string", "enum": ["node_original_source", "original_source_phrase"]},
                       "value": {"type": "string"}}}
    row = {"type": "object", "additionalProperties": False,
           "required": ["task_id", "disposition", "reason", "uses", "evidence", "source_requests"], "properties": {
               "task_id": {"type": "string", "enum": task_ids}, "disposition": {"type": "string", "enum": list(DISPOSITIONS)},
               "reason": {"type": "string"}, "uses": {"type": "array", "items": use},
               "evidence": {"type": "array", "items": evidence},
               "source_requests": {"type": "array", "items": source_request, "maxItems": 2}}}
    return {"type": "object", "additionalProperties": False, "required": ["packet_id", "decisions"],
            "properties": {"packet_id": {"type": "string"}, "decisions": {"type": "array", "items": row,
                "minItems": len(task_ids), "maxItems": len(task_ids)}}}


def _packet(materials, binding):
    ids = [row["task"]["task_id"] for row in materials]
    result = {"schema_version": VERSION + ".request", "binding": copy.deepcopy(binding),
              "expected_task_ids": ids, "items": copy.deepcopy(materials), "response_schema": response_schema(ids)}
    result["packet_id"] = "explicit-packet:" + digest(result)[:32]
    return result


def actual_wire_bytes(instructions, payload, *, model="gpt-6.1-sol", reasoning_effort="high",
                      max_output_tokens=12000, response_transport="sse"):
    """Match the retained Responses client's complete API body, both schemas."""
    body = {"model": model, "store": False, "reasoning": {"effort": reasoning_effort},
            "max_output_tokens": max_output_tokens,
            "input": [{"role": "developer", "content": instructions},
                      {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}]}
    if response_transport == "sse":
        body["stream"] = True
    if payload.get("response_schema") is not None:
        body["text"] = {"format": {"type": "json_schema", "name": "source_evaluation",
                                 "schema": payload["response_schema"], "strict": True}}
    return json.dumps(body, ensure_ascii=False, allow_nan=False,
                      sort_keys=response_transport == "sse").encode("utf-8")


def prepare_packets(project, nodes, scan, *, max_wire_bytes=160000, max_items=12, wire_options=None):
    if type(max_wire_bytes) is not int or max_wire_bytes < 10000 or type(max_items) is not int or max_items < 1:
        raise ValueError("Positive explicit packet bounds required")
    validate_nodes(project, nodes)
    if scan.get("nodes_value_sha256") != digest(nodes) or scan.get("source_fingerprint") != project.fingerprint:
        raise ValueError("Explicit scan is not bound to the frozen Units/Contexts/source")
    wire_options = copy.deepcopy(wire_options or {})
    binding = {"source": project.fingerprint, "nodes_value_sha256": digest(nodes), "wire_options": wire_options,
               "scan_value_sha256": scan["scan_value_sha256"], "program": file_record(__file__), "prompt": file_record(PROMPT)}
    packets, current, oversize = [], [], []

    def wire_bytes(values):
        packet = _packet(values, binding)
        return len(actual_wire_bytes(PROMPT.read_text(encoding="utf-8"), packet, **wire_options))

    for task in scan["tasks"]:
        material = _materials(project, nodes, task)
        if wire_bytes([material]) > max_wire_bytes:
            oversize.append({"task_id": task["task_id"], "required_wire_bytes": wire_bytes([material]),
                             "disposition": "unknown", "reason": "Complete lossless task exceeds the registered wire bound"})
            continue
        if current and (len(current) >= max_items or wire_bytes(current + [material]) > max_wire_bytes):
            packets.append(_packet(current, binding))
            current = []
        current.append(material)
    if current:
        packets.append(_packet(current, binding))
    actual_ids = [identifier for packet in packets for identifier in packet["expected_task_ids"]]
    if Counter(actual_ids + [row["task_id"] for row in oversize]) != Counter(task["task_id"] for task in scan["tasks"]):
        raise AssertionError("Every model task must be a complete packet member or a visible wire failure")
    return {"schema_version": VERSION + ".preflight", "paper_id": project.paper_id, "binding": binding,
            "max_wire_bytes": max_wire_bytes, "max_items": max_items,
            "packets": packets, "oversize_tasks": oversize, "supplier_starts": 0,
            "counts": {**scan["counts"], "packets": len(packets), "oversize_tasks": len(oversize),
                       "planned_task_ids": len(actual_ids), "max_packet_wire_bytes": max((wire_bytes(packet["items"]) for packet in packets), default=0)},
            "fixed_nodes_unchanged": True, "quality_accepted": False}


def _selectors(project, material, selectors):
    menu = {row["source_id"]: row for row in material["source_menu"]}
    if not isinstance(selectors, list) or not selectors:
        raise ValueError("A semantic decision needs supplied original-source evidence")
    spans = []
    for selector in selectors:
        if not isinstance(selector, dict) or set(selector) != {"source_id", "quote"} or selector["source_id"] not in menu:
            raise ValueError("Evidence must select this task's supplied source ID")
        row = menu[selector["source_id"]]
        span = copy.deepcopy(row["span"])
        quote = selector["quote"]
        if quote is not None:
            if not isinstance(quote, str) or not quote or row["text"].count(quote) != 1:
                raise ValueError("Evidence quote must be unique exact source characters")
            start = row["text"].index(quote)
            span.update(char_start=span["char_start"] + start, char_end=span["char_start"] + start + len(quote))
        if project.resolve_span(span) != (row["text"] if quote is None else quote):
            raise ValueError("Evidence differs from current pinned source")
        spans.append(span)
    return spans


def validate_response(project, nodes, packet, response):
    """Check complete identity and actual-use witness locations, not truth."""
    validate_nodes(project, nodes)
    if (not isinstance(response, dict) or set(response) != {"packet_id", "decisions"}
            or response["packet_id"] != packet["packet_id"]):
        raise ValueError("Response must bind exactly this explicit packet")
    decisions = response["decisions"]
    if not isinstance(decisions, list) or Counter(row.get("task_id") for row in decisions) != Counter(packet["expected_task_ids"]):
        raise ValueError("Exactly one decision for every explicit task is required")
    materials = {row["task"]["task_id"]: row for row in packet["items"]}
    units = {row["id"]: row for row in nodes["units"]}
    normalized = []
    for decision in decisions:
        if set(decision) != {"task_id", "disposition", "reason", "uses", "evidence", "source_requests"}:
            raise ValueError("Unexpected/missing explicit decision fields")
        if decision["disposition"] not in DISPOSITIONS or not isinstance(decision["reason"], str) or not decision["reason"].strip():
            raise ValueError("A concrete explicit-use decision/reason is required")
        material = materials[decision["task_id"]]
        task = material["task"]
        requests = decision["source_requests"]
        if (not isinstance(requests, list) or len(requests) > 2 or requests and decision["disposition"] != "unknown"
                or any(not isinstance(request, dict) or set(request) != {"kind", "value"}
                    or request["kind"] not in ("node_original_source", "original_source_phrase")
                    or not isinstance(request["value"], str) or not request["value"].strip() for request in requests)):
            raise ValueError("Only unknown judgments may request at most two concrete source reads")
        for request in requests:
            if request["kind"] == "node_original_source" and request["value"] not in units:
                raise ValueError("Source followup cannot invent or extract a node")
        evidence = _selectors(project, material, decision["evidence"])
        if (not isinstance(decision["uses"], list) or (decision["disposition"] == "accepted") != bool(decision["uses"])):
            raise ValueError("Only accepted decisions contain actual direct uses")
        sources = {row["node_id"] for row in task["premise_candidates"] + task["proof_premise_candidates"]}
        targets = {row["node_id"] for row in task["user_candidates"]}
        uses = []
        for use in decision["uses"]:
            if (set(use) != {"source", "target", "type", "explanation", "evidence"}
                    or use["source"] not in sources or use["target"] not in targets or use["source"] == use["target"]
                    or not isinstance(use["explanation"], str) or not use["explanation"].strip()):
                raise ValueError("Use must have distinct current candidate endpoints and explanation")
            source, target = units[use["source"]], units[use["target"]]
            expected = "proves" if task["kind"] == "proves" else "proof_dependency" if target["category"] == "proof" else "statement_dependency"
            if (use["type"] != expected or expected == "statement_dependency" and source["category"] == "proof"
                    or expected == "proves" and (source["category"], target["category"]) != ("proof", "statement")):
                raise ValueError("Formal direction is premise -> user; proves is proof -> statement")
            spans = _selectors(project, material, use["evidence"])
            if task["kind"] == "reference":
                if not any(_intersects(span, task["reference"]["span"]) for span in spans):
                    raise ValueError("Explicit-use evidence must contain its exact reference occurrence")
                body = [_span(span) for span in target["sources"]]
                contexts = {context["id"]: context for context in nodes["contexts"]}
                allowed = body + [row["span"] for row in _context_sources(target, contexts)]
                if not any(_intersects(span, candidate) for span in spans for candidate in allowed):
                    raise ValueError("Explicit evidence does not reach the chosen mathematical user")
                fragments = project.decision_fragments([task["reference"]["span"]], reference=True)
                if fragments and not any(all(_covered(part, spans) for part in fragment) for fragment in fragments):
                    raise ValueError("Explicit-use evidence must preserve the full direct-use clause/governed argument")
            elif (not any(_intersects(span, _span(part)) for span in spans for part in source["sources"])
                  or not any(_intersects(span, _span(part)) for span in spans for part in target["sources"])):
                raise ValueError("proves evidence must reach both the exact proof and statement")
            proof_statements = [row["statement_id"] for row in task.get("proof_premise_candidates", [])
                                if row["node_id"] == use["source"]]
            if expected == "proof_dependency" and source["category"] == "proof":
                premise = [row for row in task["premise_candidates"] if row["node_id"] == source["id"]]
                direct_locator = any(basis in ("label_inside_node", "mathematical_owner_overlap",
                        "same_source_compiled_numbering_anchor_candidate_only",
                        "literal_case_locator_carried_by_existing_proof_requires_review") for row in premise for basis in row["bases"])
                if not direct_locator and not proof_statements:
                    raise ValueError("A statement locator cannot borrow an unpaired proof; direct proof locators remain legal")
            uses.append({**copy.deepcopy(use), "evidence_spans": spans,
                         "proof_locator_statement_ids": sorted(set(proof_statements)),
                         "reference_id": task["reference_id"], "reference_key": task["reference_key"]})
        normalized.append({**copy.deepcopy(decision), "evidence_spans": evidence, "uses": uses,
                           "task_binding_sha256": digest(material)})
    return normalized


def _covered(required, evidence):
    intervals = sorted((max(required["char_start"], row["char_start"]), min(required["char_end"], row["char_end"]))
                       for row in evidence if _intersects(required, row))
    cursor = required["char_start"]
    for left, right in intervals:
        if left > cursor:
            return False
        cursor = max(cursor, right)
    return cursor >= required["char_end"]


def source_followup(project, nodes, material, requests):
    """Read requested original source locally; no search for implicit edges."""
    result = copy.deepcopy(material)
    unit_by_id = {unit["id"]: unit for unit in nodes["units"]}
    additions, requested_ids, search_results = [], set(), []
    for request in requests:
        if request["kind"] == "node_original_source":
            requested_ids.add(request["value"])
        else:
            phrase = request["value"]
            matches, rows, offset = None, [], 0
            while True:
                matches = project.find_text(phrase, limit=100, offset=offset)
                rows.extend(matches["matches"])
                if matches["next_offset"] is None:
                    break
                offset = matches["next_offset"]
            search_results.append({"phrase": phrase, "total": matches["total"], "matches": copy.deepcopy(rows),
                                   "full_exact_phrase_scan": True})
            for row in rows:
                span = row.get("span", row)
                if all(key in span for key in ("file", "char_start", "char_end")):
                    text = project._read_text(span["file"])
                    left = text.rfind("\n\n", 0, span["char_start"])
                    right = text.find("\n\n", span["char_end"])
                    original = {**_span(span), "char_start": max(0, left + 2), "char_end": len(text) if right < 0 else right}
                    additions.append(original)
                    requested_ids.update(unit["id"] for unit in nodes["units"]
                        if any(_intersects(_span(source), span) for source in unit["sources"]))
    for identifier in sorted(requested_ids):
        if identifier not in unit_by_id:
            raise ValueError("Requested source is not a frozen node")
        unit = unit_by_id[identifier]
        if result["task"]["kind"] == "proves" and unit["category"] == "statement":
            if identifier not in {row["node_id"] for row in result["task"]["user_candidates"]}:
                result["task"]["user_candidates"].append({"node_id": identifier, "body_occurrence": True, "context_ids": [],
                    "basis": "requested_existing_proof_target_requires_semantic_scope_confirmation"})
            continue
        direct_owner = any(candidate["id"] == identifier
                           for endpoint in result["task"]["referenced_endpoints"]
                           for candidate, _, _ in _reference_sources([_node_adapter(unit)], endpoint))
        if unit["category"] == "proof" and not direct_owner:
            for hint in nodes.get("proof_attachment_hints", []):
                if hint["proof_unit_id"] != identifier:
                    continue
                for statement_id in hint.get("candidate_target_unit_ids", []):
                    if statement_id not in unit_by_id or unit_by_id[statement_id]["category"] != "statement":
                        continue
                    if statement_id not in {row["node_id"] for row in result["task"]["premise_candidates"]}:
                        result["task"]["premise_candidates"].append({"node_id": statement_id,
                            "bases": ["source_requested_proof_statement_locator_requires_semantic_confirmation"], "owner_ids": []})
                    bridge = {"node_id": identifier, "statement_id": statement_id,
                              "basis": "frozen_attachment_hint_requires_proves_review"}
                    if bridge not in result["task"]["proof_premise_candidates"]:
                        result["task"]["proof_premise_candidates"].append(bridge)
        if identifier not in {row["node_id"] for row in result["task"]["premise_candidates"]}:
            result["task"]["premise_candidates"].append({"node_id": identifier,
                "bases": ["label_inside_node" if direct_owner else
                          "explicit_locator_source_followup_requires_semantic_owner_confirmation"], "owner_ids": []})
    updated = _materials(project, nodes, result["task"])
    menus = {row["source_id"]: row for row in updated["source_menu"]}
    for span in additions:
        identifier = "source:" + digest({"source": project.fingerprint, "span": project.normalize_span(span)})[:24]
        menus.setdefault(identifier, {"source_id": identifier, "span": project.normalize_span(span),
            "text": project.resolve_span(span), "roles": ["requested_original_source_phrase_background"]})
    updated["source_menu"] = list(menus.values())
    updated["source_followup"] = {"requests": copy.deepcopy(requests), "search_results": search_results,
                                  "new_frozen_node_reads": sorted(requested_ids), "node_extraction": False}
    return updated


def run_judgments(project, nodes, scan, plan, judge, *, save=None, max_source_followups=2):
    """Content-bound model calls, preserving every original/followup attempt."""
    if max_source_followups != 2:
        raise ValueError("The registered explicit task permits exactly at most two source followup rounds")
    prompt = PROMPT.read_text(encoding="utf-8")
    decisions, calls = {}, []
    wire = plan["binding"].get("wire_options", {})
    max_wire = plan.get("max_wire_bytes", 160000)
    for packet in plan["packets"]:
        pending = copy.deepcopy(packet)
        for round_number in range(max_source_followups + 1):
            body = actual_wire_bytes(prompt, pending, **wire)
            if len(body) > max_wire:
                for material in pending["items"]:
                    task_id = material["task"]["task_id"]
                    if task_id in decisions:
                        decisions[task_id]["reason"] += "; complete source followup exceeds the registered wire bound"
                break
            response = judge("explicit_dependency_production" if round_number == 0 else "explicit_dependency_source_followup",
                             prompt, pending)
            normalized = validate_response(project, nodes, pending, response)
            record = {"packet": pending, "response": response, "validated_decisions": normalized,
                      "source_followup_round": round_number, "wire_bytes": len(body),
                      "wire_sha256": hashlib.sha256(body).hexdigest()}
            calls.append(record)
            if save:
                save(record)
            for decision in normalized:
                decisions[decision["task_id"]] = decision
            requested = [row for row in normalized if row["disposition"] == "unknown" and row["source_requests"]]
            if not requested or round_number == max_source_followups:
                break
            materials = {row["task"]["task_id"]: row for row in pending["items"]}
            pending = _packet([source_followup(project, nodes, materials[row["task_id"]], row["source_requests"])
                               for row in requested], plan["binding"])
    for row in plan["oversize_tasks"]:
        decisions[row["task_id"]] = {"task_id": row["task_id"], "disposition": "unknown", "reason": row["reason"],
            "uses": [], "evidence": [], "evidence_spans": [], "source_requests": [], "task_binding_sha256": None}
    return assemble(project, nodes, scan, plan, list(decisions.values())), calls


def assemble(project, nodes, scan, plan, decisions):
    validate_nodes(project, nodes)
    expected = [task["task_id"] for task in scan["tasks"]]
    if Counter(row["task_id"] for row in decisions) != Counter(expected):
        raise ValueError("Complete explicit task population must be retained in assembly")
    proved_pairs = {(use["source"], use["target"]) for row in decisions for use in row["uses"] if use["type"] == "proves"}
    task_by_id = {task["task_id"]: task for task in scan["tasks"]}
    relation_map, evidences, gaps = {}, {}, []
    for row in decisions:
        task = task_by_id[row["task_id"]]
        if row["disposition"] == "unknown":
            gaps.append(copy.deepcopy(row))
        for use in row["uses"]:
            if use["type"] == "proof_dependency":
                statements = use.get("proof_locator_statement_ids", [])
                if statements and not any((use["source"], statement_id) in proved_pairs for statement_id in statements):
                    raise ValueError("A reused proof premise requires an accepted matching proves association")
            key = (use["source"], use["target"], use["type"])
            identifier = "relation:" + digest({"source": project.fingerprint, "endpoints": key})[:24]
            evidence_id = "evidence:" + digest({"task": row["task_id"], "use": use})[:24]
            evidences[evidence_id] = {"id": evidence_id, "method": "agent_analysis", "explanation": use["explanation"],
                "sources": [project.source(span).model_dump(mode="json") for span in use["evidence_spans"]]}
            record = relation_map.setdefault(key, {"id": identifier, "source": key[0], "target": key[1], "type": key[2],
                    "explanation": use["explanation"], "conditions": None, "evidence": []})
            record["evidence"].append(evidence_id)
    graph_dict = {"schema_version": "gangmu.mathir.v4", "snapshot_id": "explicit:" + digest({"nodes": digest(nodes), "decisions": decisions})[:24],
                  "documents": copy.deepcopy(nodes["documents"]),
                  "contexts": copy.deepcopy(nodes["contexts"]), "units": copy.deepcopy(nodes["units"]),
                  "relations": list(relation_map.values()), "evidence": [], "identity_views": []}
    bare = Graph.model_validate_json(json.dumps(graph_dict))
    index = GraphIndex(bare)
    for record in graph_dict["relations"]:
        relation = index.get(record["id"])
        for identifier in record["evidence"]:
            value = evidences[identifier]
            sources = tuple(Source.model_validate_json(json.dumps(source)) for source in value["sources"])
            value["claim_sha256"] = index.claim_digest(relation, sources)
    graph_dict["evidence"] = list(evidences.values())
    graph = Graph.model_validate_json(json.dumps(graph_dict))
    graph_validation = project.verify_graph(graph)
    remaining_attachments = [error for error in graph_validation.get("errors", [])
                             if error.get("reason") == "a proof requires a same-document proves edge to a statement"]
    other_errors = [error for error in graph_validation.get("errors", []) if error not in remaining_attachments]
    if other_errors or graph_validation.get("source_validation") != "verified":
        raise ValueError("Assembled explicit graph failed exact schema/source validation: " + json.dumps(graph_validation["errors"]))
    return {"schema_version": VERSION + ".result", "paper_id": project.paper_id,
            "source_fingerprint": project.fingerprint, "frozen_nodes_value_sha256": digest(nodes),
            "scan_value_sha256": scan["scan_value_sha256"], "plan_binding": plan["binding"],
            "reference_ledger": copy.deepcopy(scan["reference_ledger"]), "structural_exclusions": copy.deepcopy(scan["structural_exclusions"]),
            "decisions": copy.deepcopy(decisions), "unresolved": gaps, "graph": graph.model_dump(mode="json"),
            "graph_validation": graph_validation,
            "unresolved_proof_attachment_validation": remaining_attachments,
            "counts": {**scan["counts"], "model_decisions": len(decisions), "unresolved": len(gaps),
                       "relations_by_type": dict(Counter(relation["type"] for relation in graph_dict["relations"]))},
            "fixed_nodes_unchanged": True, "node_extraction": False, "implicit_retrieval": False,
            "accepted_graph": False, "mathematical_quality_accepted": False,
            "claim": "Model explicit-use decisions over frozen nodes; source/schema replay is not a mathematical certificate."}


def replay_calls(project, nodes, scan, plan, calls):
    """Derive every actual packet from the initial plan and source requests."""
    cursor, decisions = 0, {}
    prompt = PROMPT.read_text(encoding="utf-8")
    wire = plan["binding"].get("wire_options", {})
    for packet in plan["packets"]:
        pending = copy.deepcopy(packet)
        for round_number in range(3):
            body = actual_wire_bytes(prompt, pending, **wire)
            if len(body) > plan["max_wire_bytes"]:
                for material in pending["items"]:
                    identifier = material["task"]["task_id"]
                    if identifier in decisions:
                        decisions[identifier]["reason"] += "; complete source followup exceeds the registered wire bound"
                break
            if cursor >= len(calls):
                raise ValueError("Retained calls omit a planned explicit packet")
            call = calls[cursor]
            cursor += 1
            if (call.get("packet") != pending or call.get("source_followup_round") != round_number
                    or call.get("wire_bytes") != len(body) or call.get("wire_sha256") != hashlib.sha256(body).hexdigest()):
                raise ValueError("Retained call is not the exact lossless planned/source-followup packet")
            actual = validate_response(project, nodes, pending, call["response"])
            if actual != call.get("validated_decisions"):
                raise ValueError("Normalized decisions differ from retained raw response")
            for row in actual:
                decisions[row["task_id"]] = row
            requested = [row for row in actual if row["disposition"] == "unknown" and row["source_requests"]]
            if not requested or round_number == 2:
                break
            materials = {row["task"]["task_id"]: row for row in pending["items"]}
            pending = _packet([source_followup(project, nodes, materials[row["task_id"]], row["source_requests"])
                               for row in requested], plan["binding"])
    if cursor != len(calls):
        raise ValueError("Retained calls add unplanned explicit tasks")
    for row in plan["oversize_tasks"]:
        decisions[row["task_id"]] = {"task_id": row["task_id"], "disposition": "unknown", "reason": row["reason"],
            "uses": [], "evidence": [], "evidence_spans": [], "source_requests": [], "task_binding_sha256": None}
    return assemble(project, nodes, scan, plan, list(decisions.values()))
