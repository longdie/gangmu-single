"""Independent precision for a fixed local population after Context-only changes.

The caller pins the selected IDs and owns fresh judge sessions, transport,
accounting and admission. This component reads no coverage references and never
certifies a whole paper, a node stage, or unchanged historical judgments.
"""
from __future__ import annotations

from collections import Counter
import copy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from gangmu.representation.model import Context, Unit, digest
from . import staged_node_evaluation as base
from . import staged_review_completion as completion
from . import staged_review_record_pool_v1 as record_pool
from . import staged_nodes_wire_strategy as wire
from .source import SourceError

VERSION = "gangmu.e4.staged-context-local-precision.v1"
TASK = "staged_nodes_precision"
AXES = base.AXES
SOURCE_FIELDS = {"source_id", "kind", "original_text", "source_spans", "reading_ranges"}
SOURCE_OPTIONAL = {"title", "category_hint", "anchor_before", "anchor_after"}
MUTABLE_TOP = {"units", "contexts", "producer_version", "producer_binding",
               "summary", "context_selection_audit"}


def _require(condition, reason):
    if not condition:
        raise SourceError(reason)


def program_bindings():
    root = Path(__file__).resolve().parents[3]
    paths = [Path(__file__), *(Path(module.__file__) for module in
                              (base, completion, record_pool, wire)),
             base.PROMPTS / "precision.md"]
    return {"version": VERSION, "files": [
        {"path": path.relative_to(root).as_posix(),
         "sha256": hashlib.sha256(path.read_bytes()).hexdigest()} for path in paths]}


def _records(project, artifact):
    _require(isinstance(artifact, dict), "A complete node artifact is required")
    _require(artifact.get("paper_id") == project.paper_id
             and artifact.get("source_fingerprint") == project.fingerprint
             and artifact.get("documents") == [project.document.model_dump(mode="json")]
             and artifact.get("accepted_graph") is False,
             "Local precision requires the pinned same-document node-only artifact")
    units, contexts = artifact.get("units"), artifact.get("contexts")
    _require(isinstance(units, list) and isinstance(contexts, list),
             "The complete accepted Unit and Context populations are required")
    context_map, unit_map = {}, {}
    for raw in contexts:
        context = Context.model_validate_json(json.dumps(raw)).model_dump(mode="json")
        key = context["id"]
        _require(key not in context_map and context["document"] == project.document.id,
                 "Context IDs must be unique and belong to the pinned document")
        context["source_evidence"] = base._source_records(
            project, context, literal=False, document=True)
        context["sources"] = [item["source"] for item in context["source_evidence"]]
        context_map[key] = context

    def chain(key):
        attached, seen = [], set()
        while key is not None:
            _require(key not in seen and key in context_map,
                     "Attached Context chain is missing, invalid or cyclic")
            seen.add(key)
            attached.append(context_map[key])
            key = context_map[key]["parent"]
        return attached

    # Check unused Context chains too; invalid records cannot hide outside the
    # selected population in a purported Context-only revision.
    for key in context_map:
        chain(key)
    chains = {}
    for raw in units:
        unit = Unit.model_validate_json(json.dumps(raw)).model_dump(mode="json")
        key = unit["id"]
        _require(key not in unit_map and unit["document"] == project.document.id
                 and unit["semantics"] == {},
                 "Unit IDs must be unique, source-bound and have empty staged semantics")
        evidence = base._source_records(project, unit, literal=True, document=True)
        unit["sources"] = [item["source"] for item in evidence]
        unit_map[key] = unit
        chains[key] = chain(unit["context"])
    return unit_map, chains


def _source_rows(project, source_records):
    _require(isinstance(source_records, dict) and bool(source_records),
             "A complete pinned source-ID table is required")
    rows = {}
    for key, raw in source_records.items():
        _require(isinstance(key, str) and key.strip() and isinstance(raw, dict)
                 and raw.get("source_id") == key
                 and SOURCE_FIELDS <= set(raw) <= SOURCE_FIELDS | SOURCE_OPTIONAL
                 and isinstance(raw["kind"], str) and raw["kind"].strip()
                 and all(raw[name] is None or isinstance(raw[name], str)
                         for name in SOURCE_OPTIONAL if name in raw),
                 "Source records may contain original source fields only")
        ranges = raw["reading_ranges"]
        _require(isinstance(ranges, list) and bool(ranges), "Complete reading ranges are required")
        for region in ranges:
            _require(isinstance(region, dict) and set(region) == {"char_start", "char_end"}
                     and type(region.get("char_start")) is int
                     and type(region.get("char_end")) is int
                     and 0 <= region["char_start"] < region["char_end"] <= len(project.reading),
                     "Invalid complete source reading range")
        row = copy.deepcopy(raw)
        spans = row["source_spans"]
        _require(isinstance(spans, list) and bool(spans)
                 and all(isinstance(span, dict)
                         and {"file", "char_start", "char_end"} <= set(span)
                         <= {"file", "char_start", "char_end", "occurrence", "document"}
                         and span.get("document", project.document.id) == project.document.id
                         for span in spans),
                 "Audit source records cannot cross the pinned document")
        evidence = base._source_records(project, row, literal=True, document=False)
        row["source_spans"] = [item["source"] for item in evidence]
        rows[key] = row
    return rows


def prepare_local_precision(project, original, revised, selected_unit_ids,
                            source_records, *, neighborhood_chars=1200):
    """Prepare complete source-bound rows for a preregistered local population.

    Actual Context changes must be a subset of the fixed selected IDs. Selected
    unchanged or uncertain cases remain in the denominator. All other artifact
    fields stay exact except producer metadata, summary.contexts and the added
    context_selection_audit. Source rows are whole audit evidence, never an
    automatic attachment to Context or an independent coverage reference.
    """
    project.assert_unchanged()
    _require(isinstance(selected_unit_ids, list) and bool(selected_unit_ids)
             and all(isinstance(key, str) and key.strip() for key in selected_unit_ids)
             and len(set(selected_unit_ids)) == len(selected_unit_ids),
             "A fixed nonempty list of unique selected Unit IDs is required")
    _require(type(neighborhood_chars) is int and neighborhood_chars > 0,
             "Source neighborhood size must be a positive integer")
    old, new, selected, source = map(copy.deepcopy,
                                     (original, revised, selected_unit_ids, source_records))
    old_units, old_chains = _records(project, old)
    new_units, new_chains = _records(project, new)
    _require([row["id"] for row in old["units"]] == [row["id"] for row in new["units"]]
             and set(selected) <= set(new_units),
             "Local Context revision cannot change or reorder the accepted population")
    _require({key: value for key, value in old.items() if key not in MUTABLE_TOP}
             == {key: value for key, value in new.items() if key not in MUTABLE_TOP},
             "Original extraction history and all unrelated artifact fields must remain exact")
    for value in (old, new):
        _require(isinstance(value.get("summary"), dict)
                 and value["summary"].get("contexts") == len(value["contexts"]),
                 "Context summary must match its complete population")
    _require({key: value for key, value in old["summary"].items() if key != "contexts"}
             == {key: value for key, value in new["summary"].items() if key != "contexts"},
             "Only summary.contexts may change during local Context revision")
    changed = []
    for before, after in zip(old["units"], new["units"]):
        key = before["id"]
        _require({name: value for name, value in before.items() if name != "context"}
                 == {name: value for name, value in after.items() if name != "context"},
                 "Unit content, classification and literal sources must remain exact")
        if before["context"] != after["context"] or old_chains[key] != new_chains[key]:
            changed.append(key)
    _require(set(changed) <= set(selected), "A Context change affected an unselected Unit")
    rows = _source_rows(project, source)
    payload_rows, claims = [], {}
    for key in selected:
        unit, attached = new_units[key], new_chains[key]
        neighborhoods, nearby_spans = [], []
        for span in unit["sources"]:
            text = project._read_text(span["file"])
            left = max(0, span["char_start"] - neighborhood_chars)
            right = min(len(text), span["char_end"] + neighborhood_chars)
            neighborhoods.append({"file": span["file"], "occurrence": span["occurrence"],
                "before": {"char_start": left, "char_end": span["char_start"],
                           "text": text[left:span["char_start"]]},
                "after": {"char_start": span["char_end"], "char_end": right,
                          "text": text[span["char_end"]:right]}})
            nearby_spans.append(dict(span, char_start=left, char_end=right))
        targets = {"source_spans": nearby_spans + [
            span for context in attached for span in context["sources"]]}
        audit_rows = [row for row in rows.values() if base._overlaps(targets, row)]
        _require(all(base._complete_span_cover(record, audit_rows) for record in [unit, *attached]),
                 "Full source rows do not contain every selected Unit and attached Context slice")
        payload_rows.append({"unit_id": key, "unit": unit, "attached_contexts": attached,
                             "audit_only_source_neighborhoods": neighborhoods,
                             "audit_only_source_records": audit_rows})
        claims[key] = digest({"source": project.fingerprint, "unit": unit, "contexts": attached})
    prompt = (base.PROMPTS / "precision.md").read_text(encoding="utf-8")
    payload = {"paper_id": project.paper_id, "source_fingerprint": copy.deepcopy(project.fingerprint),
               "review_scope": "fixed_local_population_after_context_only_revision",
               "units": payload_rows, "response_schema": completion._schema(TASK, selected)}
    binding = {**program_bindings(), "original_node_result_value_sha256": digest(old),
               "revised_node_result_value_sha256": digest(new), "source_records_value_sha256": digest(source),
               "selected_unit_ids": selected, "changed_unit_ids": changed,
               "complete_accepted_population": len(new_units),
               "neighborhood_chars": neighborhood_chars,
               "instructions_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
               "payload_sha256": digest(payload)}
    project.assert_unchanged()
    return {"instructions": prompt, "payload": payload, "claims": claims, "binding": binding}


def score_local_precision(response, claims):
    """Score only the fixed claims; missing, duplicate and malformed rows stay unknown."""
    _require(isinstance(claims, dict) and bool(claims)
             and all(isinstance(key, str) and key.strip() and isinstance(value, str)
                     and len(value) == 64 and all(char in "0123456789abcdef" for char in value)
                     for key, value in claims.items()), "Fixed source-bound local claims are required")
    raw = response.get("judgments") if isinstance(response, dict) and set(response) == {"judgments"} else None
    raw = raw if isinstance(raw, list) else []
    counts = Counter(row["unit_id"] for row in raw if isinstance(row, dict)
                     and isinstance(row.get("unit_id"), str))
    found = {row["unit_id"]: row for row in raw if isinstance(row, dict)
             and isinstance(row.get("unit_id"), str) and row["unit_id"] in claims
             and counts[row["unit_id"]] == 1}
    judgments = []
    for key, claim in claims.items():
        answer = found.get(key, {})
        valid = bool(answer) and completion._valid_row(answer, TASK)
        axes = {axis: answer.get(axis, "unknown") for axis in AXES}
        axes = {axis: value if value in ("pass", "fail", "unknown") else "unknown"
                for axis, value in axes.items()}
        status = "unknown"
        if valid:
            if "fail" in axes.values() or answer["status"] == "unsupported":
                status = "unsupported"
            elif all(value == "pass" for value in axes.values()) and answer["status"] == "supported":
                status = "supported"
        judgments.append({"unit_id": key, "claim_sha256": claim, "status": status, "axes": axes,
                          "response_valid": valid,
                          "reason": answer["reason"] if valid else
                          "omitted, duplicate, malformed or incomplete independent local judgment"})
    return {**base._metric(judgments, "supported", "unsupported"), "judgments": judgments,
            "unexpected_ids": [key for key in counts if key not in claims],
            "duplicate_ids": [key for key, count in counts.items() if count > 1]}


def evaluate_local_precision(project, original, revised, selected_unit_ids,
                             source_records, judge, wire_options=None, *, neighborhood_chars=1200):
    """Run fresh injected local review through lossless complete-item record pooling.

    All packets are planned before any judge dispatch. There are no automatic
    second opinions or missing-ID retries. The result has no whole-stage pass
    claim, no coverage metric and no inherited historical review decisions.
    """
    _require(callable(judge), "An independent local review judge must be injected")
    prepared = prepare_local_precision(project, original, revised, selected_unit_ids,
                                       source_records, neighborhood_chars=neighborhood_chars)
    options = wire.WireOptions.parse(asdict(wire_options) if isinstance(wire_options, wire.WireOptions)
                                     else wire_options)
    def isolated(kind, instructions, payload):
        return judge(kind, instructions, copy.deepcopy(payload))

    pooled = record_pool.RecordPoolJudge(isolated, asdict(options))
    pooled.plan(TASK, prepared["instructions"], prepared["payload"])
    _, trace = base._judgments(pooled, TASK, prepared["instructions"], prepared["payload"],
                              prepared["binding"]["selected_unit_ids"], "unit_id")
    response = trace.get("response")
    project.assert_unchanged()
    return {"schema_version": VERSION, "paper_id": project.paper_id,
            "accepted_graph": False, "node_stage_accepted": False,
            "complete_paper_quality_evaluated": False, "local_precision_only": True,
            "core_and_skeleton_reevaluated": False, "binding": prepared["binding"],
            "wire_options": asdict(options), "precision": score_local_precision(response, prepared["claims"]),
            "judge_calls": [trace], "record_pool_requests": copy.deepcopy(pooled.requests),
            "limitations": {"full_paper_review_required": True,
                            "independent_author_task_references_read": False,
                            "same_payload_as_historical_full_review": False,
                            "audit_sources_automatically_attached_to_context": False,
                            "author_mathematics_certified": False, "human_expert_gold": False,
                            "dependency_or_dag_evaluated": False}}
