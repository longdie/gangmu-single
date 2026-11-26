"""Normal node execution closure, separate from candidate admission.

The frozen node producer continues to select, reject and roll back exactly as
before. This wrapper only distinguishes a completed but partially admitted
population from missing/malformed answers. It never scores node precision.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
from pathlib import Path

from gangmu.runtime.layout import project_of, storage_path
from gangmu.representation.model import digest
from . import staged_nodes as base
from .topology_integration import ROOT, _local, _record, _save_once

VERSION = "gangmu.single-document.node-execution.v1"


class RecordingJudge:
    """Save exact post-compaction packets and returned production decisions."""
    def __init__(self, judge, *, root=ROOT, identity):
        self.root, self.judge = Path(root).resolve(), judge
        self.identity = copy.deepcopy(identity)
        self.directory = storage_path(self.root, "artifacts/single-document-node-execution/" + digest(identity), project_of(judge))
        self.directory.mkdir(parents=True, exist_ok=True)
        _save_once(self.directory / "identity.json", self.identity)
        self.records = []

    def __call__(self, task_kind, instructions, payload):
        payload = copy.deepcopy(payload)
        key = digest({"task_kind": task_kind, "instructions": instructions, "payload": payload})
        directory = self.directory / "calls" / key
        directory.mkdir(parents=True, exist_ok=True)
        request = {"identity": self.identity, "task_kind": task_kind,
                   "instructions": instructions, "payload": payload}
        _save_once(directory / "request.json", request)
        row = {"task_kind": task_kind, "instructions_sha256": hashlib.sha256(instructions.encode()).hexdigest(),
               "payload": payload, "payload_sha256": digest(payload),
               "request": _record(self.root, directory / "request.json")}
        before = len(self.judge.receipts()) if hasattr(self.judge, "receipts") else 0
        try:
            response = self.judge(task_kind, instructions, copy.deepcopy(payload))
        except Exception as error:
            failure = {"identity": self.identity, "status": "technical_failure",
                       "error_type": type(error).__name__}
            _save_once(directory / "failure.json", failure)
            row.update(status="technical_failure", error_type=type(error).__name__,
                       failure=_record(self.root, directory / "failure.json"))
            self.records.append(row)
            raise
        _save_once(directory / "response.json", response)
        receipts = self.judge.receipts()[before:] if hasattr(self.judge, "receipts") else []
        # Reuse may change cache_hit while referring to the identical original
        # invocation. Bind the stable receipt fields without rewriting history.
        stable_receipts = [{k: v for k, v in receipt.items() if k != "cache_hit"} for receipt in receipts]
        receipt_value = {"identity": self.identity, "receipts": stable_receipts}
        _save_once(directory / "receipts.json", receipt_value)
        row.update(status="returned", response=copy.deepcopy(response), response_sha256=digest(response),
                   response_binding=_record(self.root, directory / "response.json"),
                   receipts_binding=_record(self.root, directory / "receipts.json"),
                   provider_receipts=stable_receipts)
        self.records.append(row)
        return response


def classify_execution(result, calls, *, expected_candidate_ids, expected_paragraph_ids,
                       source_fingerprint, max_context_requests=2):
    """Classify technical closure without changing any original host outcome.

    A source-selector or Context proposal rejected *after* a complete valid
    final answer can remain pending. Invalid schema/populations, oversized or
    unanswered requests, or nonterminal Context request chains block output.
    """
    blockers, chains = [], defaultdict(list)
    def block(reason, **details):
        blockers.append({"reason": reason, **details})

    if result.get("source_fingerprint") != source_fingerprint:
        block("node_result_source_binding_differs")
    wire = result.get("wire_protocol", {}).get("actual_calls")
    if not isinstance(wire, list):
        block("wire_call_population_missing")
        wire = []
    if any(row.get("status") != "returned" for row in wire):
        block("oversized_or_failed_wire_task", statuses=[row.get("status") for row in wire])
    returned = [row for row in calls if row.get("status") == "returned"]
    if len(returned) != len(calls):
        block("model_task_unanswered_or_failed")
    if Counter(row.get("actual_payload_sha256") for row in wire) != Counter(
            row.get("payload_sha256") for row in calls):
        block("actual_wire_and_recorded_call_populations_differ")
    logical = result.get("judgment_calls")
    if (not isinstance(logical, list) or len(logical) != len(calls)
            or any(row.get("status") != "returned" for row in logical)):
        block("logical_model_task_population_not_returned")

    for number, row in enumerate(calls):
        if number >= len(wire) or not isinstance(logical, list) or number >= len(logical):
            continue
        physical, journal = wire[number], logical[number]
        if (physical.get("task_kind") != row.get("task_kind")
                or journal.get("task_kind") != row.get("task_kind")
                or physical.get("actual_payload_sha256") != row.get("payload_sha256")
                or physical.get("source_context_followup") != row.get("payload", {}).get("source_context_followup")
                or (row.get("status") == "returned" and journal.get("response_sha256") != row.get("response_sha256"))
                or row.get("instructions_sha256") != result.get("producer_binding", {}).get("prompt_sha256", {}).get(row.get("task_kind"))):
            block("recorded_call_differs_from_original_frozen_journal", call=number)

    for number, row in enumerate(calls):
        if row.get("status") != "returned":
            continue
        payload, response, kind = row.get("payload"), row.get("response"), row.get("task_kind")
        if (not isinstance(payload, dict) or kind not in {base.TASK_ROUND1, base.TASK_ROUND2}
                or row.get("payload_sha256") != digest(payload)
                or row.get("response_sha256") != digest(response)
                or payload.get("source_fingerprint") != source_fingerprint):
            block("returned_call_binding_invalid", call=number)
            continue
        try:
            if payload["response_schema"] != base.response_schema(kind, payload):
                raise ValueError("Payload schema differs from frozen node schema")
            base._check_shape(response, payload["response_schema"])
        except (ValueError, KeyError, TypeError):
            block("response_schema_invalid", call=number)
            continue
        followup = payload.get("source_context_followup")
        if type(followup) is not int or not 0 <= followup <= max_context_requests:
            block("invalid_context_followup_number", call=number)
            continue
        if kind == base.TASK_ROUND1:
            ids = [candidate.get("candidate_id") for candidate in payload.get("candidates", [])]
            decisions = [decision.get("candidate_id") for decision in response["decisions"]]
            identifier = payload.get("batch_id")
            if (payload.get("round") != 1 or not isinstance(identifier, str) or not ids
                    or len(set(ids)) != len(ids) or Counter(decisions) != Counter(ids)):
                block("round1_candidate_population_not_exactly_once", call=number)
                continue
        else:
            ids = [paragraph.get("source_id") for paragraph in payload.get("paragraphs", [])]
            identifier = payload.get("chunk_id")
            if (payload.get("round") != 2 or not isinstance(identifier, str) or not ids
                    or len(set(ids)) != len(ids) or response["chunk_id"] != identifier
                    or bool(response["nodes"]) != (response["decision"] == "nodes")):
                block("round2_chunk_disposition_or_population_invalid", call=number)
                continue
        chains[(kind, identifier)].append((followup, row))

    final_calls = {}
    for (kind, identifier), chain in chains.items():
        numbers = [number for number, _ in chain]
        if numbers != list(range(len(chain))):
            block("context_chain_missing_or_duplicate_followup", task=identifier)
        ids_key = "candidates" if kind == base.TASK_ROUND1 else "paragraphs"
        id_key = "candidate_id" if kind == base.TASK_ROUND1 else "source_id"
        populations = [[item[id_key] for item in row["payload"][ids_key]] for _, row in chain]
        if any(population != populations[0] for population in populations):
            block("context_followup_changes_task_population", task=identifier)
        for position, (_, row) in enumerate(chain):
            requests = row["response"]["context_requests"]
            if position == len(chain) - 1:
                if requests:
                    block("context_request_chain_not_closed", task=identifier)
            elif not requests or chain[position + 1][1]["payload"].get("previous_context_requests") != requests:
                block("context_followup_does_not_answer_previous_request", task=identifier)
        final_calls[(kind, identifier)] = chain[-1][1]

    batches = result.get("round1", {}).get("batches", [])
    chunks = result.get("round2", {}).get("chunks", [])
    if (Counter(identifier for kind, identifier in chains if kind == base.TASK_ROUND1)
            != Counter(batch.get("batch_id") for batch in batches)):
        block("round1_batch_population_missing_or_duplicate")
    if (Counter(identifier for kind, identifier in chains if kind == base.TASK_ROUND2)
            != Counter(chunk.get("chunk_id") for chunk in chunks)):
        block("round2_chunk_population_missing_or_duplicate")
    if (Counter(identifier for batch in batches for identifier in batch.get("candidate_ids", []))
            != Counter(expected_candidate_ids)
            or Counter(decision.get("candidate_id") for decision in result.get("round1", {}).get("decisions", []))
            != Counter(expected_candidate_ids)):
        block("discovered_candidate_population_not_completely_processed")
    if Counter(identifier for chunk in chunks for identifier in chunk.get("paragraph_ids", [])) != Counter(expected_paragraph_ids):
        block("discovered_paragraph_population_not_completely_processed")
    for kind, entries, identifier_key, population_key in (
            (base.TASK_ROUND1, batches, "batch_id", "candidate_ids"),
            (base.TASK_ROUND2, chunks, "chunk_id", "paragraph_ids")):
        for entry in entries:
            row = final_calls.get((kind, entry.get(identifier_key)))
            if row is None:
                continue
            key, field = ("candidates", "candidate_id") if kind == base.TASK_ROUND1 else ("paragraphs", "source_id")
            if [item[field] for item in row["payload"][key]] != entry.get(population_key):
                block("final_answer_task_ids_differ_from_host_receipt", task=entry.get(identifier_key))

    pending = []
    for decision in result.get("round1", {}).get("decisions", []):
        if decision.get("decision") != "uncertain":
            continue
        row = final_calls.get((base.TASK_ROUND1, decision.get("batch_id")))
        proposals = [proposal for proposal in row["response"]["decisions"]
                     if proposal.get("candidate_id") == decision.get("candidate_id")] if row else []
        pending.append({"round": 1, "host_decision": copy.deepcopy(decision),
                        "original_proposals": copy.deepcopy(proposals), "call": row.get("request") if row else None})
    for decision in result.get("round2", {}).get("decisions", []):
        if decision.get("decision") != "uncertain":
            continue
        row = final_calls.get((base.TASK_ROUND2, decision.get("chunk_id")))
        pending.append({"round": 2, "host_decision": copy.deepcopy(decision),
            "original_response": copy.deepcopy(row["response"]) if row else None,
            "chunk_rollback_unchanged": True, "call": row.get("request") if row else None})
    partial = result.get("summary", {}).get("status") != "complete" or bool(result.get("unresolved"))
    return {"schema_version": VERSION, "raw_node_result_sha256": digest(result),
        "technical_complete": not blockers, "scope": "incomplete" if blockers else "partial" if partial else "complete",
        "blockers": blockers, "pending": pending, "original_unresolved": copy.deepcopy(result.get("unresolved", [])),
        "rule_unresolved": copy.deepcopy(result.get("rule_unresolved", [])),
        "call_bindings": [{key: copy.deepcopy(row[key]) for key in (
            "task_kind", "status", "request", "response_binding", "receipts_binding", "failure") if key in row} for row in calls],
        "node_admission_unchanged": True, "source_selection_unchanged": True, "chunk_rollback_unchanged": True,
        "precision_assessed": False, "mathematical_correctness_certified": False,
        "complete_mathematical_content_claimed": False}
