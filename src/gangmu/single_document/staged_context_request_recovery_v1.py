"""Bounded, source-only continuation of exactly replayed Context-loop failures.

This is a new development producer after feedback. Clients, physical retries,
allowances and independent semantic evaluation remain caller responsibilities.
An injected fresh judge owns the assertion that an old uncertainty was caused
only by missing source Context; this module cannot certify that assertion.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import math
from pathlib import Path
import re

from gangmu.representation.model import digest
from . import staged_nodes as base
from . import staged_nodes_wire_strategy as wire
from .source import SourceError

VERSION = "staged-nodes-context-request-recovery-v1"
TASK = "staged_nodes_round1_context_recovery"
POLICY = base.PROMPT_ROOT / "context_request_recovery_v1.md"
CONTEXT_FAILURES = frozenset({"source-context request found no new source",
                              "required source context remains unresolved"})
STATE_FIELDS = ("units", "contexts", "provenance", "unit_keys", "candidate_units")
SCOPES = ("source_context_only", "retain_original_semantics")


def _tokens(value):
    """Keep both TeX control-sequence identity and its searchable macro name."""
    result = []
    for macro, word in re.findall(r"\\([A-Za-z@]+)|([\w]+)", value.lower()):
        if macro:
            result.extend(("tex:" + macro, macro))
        else:
            result.append(word)
    return result


def _query(rows, query, supplied, limit):
    """Exclude fully supplied IDs before BM25 scoring and before the top-k cap."""
    terms = set(_tokens(query))
    documents = [(row, Counter(_tokens("".join(row["original_text"]))))
                 for row in rows.values() if row["source_id"] not in supplied]
    if not terms or not documents:
        return []
    lengths = [sum(tokens.values()) for _, tokens in documents]
    average = sum(lengths) / len(lengths) or 1
    frequencies = {term: sum(term in tokens for _, tokens in documents) for term in terms}
    ranked = []
    for (row, tokens), length in zip(documents, lengths):
        score = 0.0
        for term in sorted(terms):
            frequency = tokens[term]
            if frequency:
                inverse = math.log(1 + (len(documents) - frequencies[term] + 0.5) /
                                   (frequencies[term] + 0.5))
                score += inverse * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / average))
        if score:
            ranked.append((score, row))
    ranked.sort(key=lambda item: (-item[0], item[1]["reading_ranges"][0]["char_start"],
                                  item[1]["source_id"]))
    return [row for _, row in ranked[:limit]]


class _CapturePipeline(wire._BoundedPipeline):
    """Observe the unchanged host at its logical judge boundary; never patch it."""

    def __init__(self, *args):
        super().__init__(*args)
        self.retained_calls, self.context_failures = [], []
        original = self.judge

        def capture(kind, instructions, payload):
            record = {"task_kind": kind, "instructions": instructions,
                      "logical_payload": copy.deepcopy(payload)}
            self.retained_calls.append(record)
            response = original(kind, instructions, payload)
            record["response"] = copy.deepcopy(response)
            return response
        self.judge = capture

    def _ask(self, kind, payload):
        start = len(self.retained_calls)
        try:
            return super()._ask(kind, payload)
        except SourceError as error:
            if kind == base.TASK_ROUND1 and str(error) in CONTEXT_FAILURES:
                calls = self.retained_calls[start:]
                if not calls or "response" not in calls[-1]:
                    raise SourceError("A Context-loop failure requires its returned terminal call") from None
                self.context_failures.append({"batch_id": payload["batch_id"], "reason": str(error),
                    "calls": copy.deepcopy(calls)})
            raise


def _schema(payload):
    schema = base.response_schema(base.TASK_ROUND1, payload)
    decision = schema["properties"]["decisions"]["items"]
    decision["properties"]["resolution_scope"] = {"type": "string", "enum": list(SCOPES)}
    decision["required"].append("resolution_scope")
    requests = schema["properties"]["context_requests"]["items"]
    requests["properties"]["candidate_ids"] = base._array({"type": "string", "enum":
        [row["candidate_id"] for row in payload["candidates"]]})
    requests["required"].append("candidate_ids")
    return schema


def _validate_response(response, payload):
    base._check_shape(response, payload["response_schema"])
    ids = [row["candidate_id"] for row in payload["candidates"]]
    decisions = response["decisions"]
    if len(decisions) != len(ids) or {row["candidate_id"] for row in decisions} != set(ids):
        raise SourceError("Recovery must decide every complete group candidate exactly once")
    if len(response["context_requests"]) > 2 * len(ids):
        raise SourceError("Recovery source requests exceed the finite complete-group bound")
    for request in response["context_requests"]:
        scoped = request["candidate_ids"]
        if not scoped or len(scoped) != len(set(scoped)):
            raise SourceError("Recovery source requests need unique, nonempty candidate scopes")
        if request["source_id"] is None and request["query"] is None:
            raise SourceError("Recovery source request is empty")


def _retrieve(pipeline, requests, supplied):
    added, seen, audit = [], set(supplied), []
    for request in requests:
        selected, error = [], None
        identifier, query = request["source_id"], request["query"]
        if identifier is not None:
            if identifier not in pipeline.rows:
                error = "unknown requested same-paper source ID"
            elif identifier not in seen:
                selected.append(pipeline.rows[identifier])
        if query is not None:
            selected.extend(_query(pipeline.rows, query, seen, pipeline.options.context_query_results))
        fresh = []
        for row in selected:
            if row["source_id"] not in seen:
                fresh.append(row["source_id"])
                added.append(copy.deepcopy(row))
                seen.add(row["source_id"])
        audit.append({"request": copy.deepcopy(request), "added_source_ids": fresh,
                      "status": "invalid_source_id" if error else "source_added" if fresh else "no_new_source",
                      "error": error})
    return added, audit


def _catalog(pipeline, supplied):
    return [{"source_id": row["source_id"], "kind": row["kind"],
             "preview": " ".join("".join(row["original_text"]).split())[:100],
             "full_text_supplied": row["source_id"] in supplied}
            for row in pipeline.rows.values()]


def _snapshot(pipeline):
    return tuple(copy.deepcopy(getattr(pipeline, name)) for name in STATE_FIELDS)


def _restore(pipeline, snapshot):
    for name, value in zip(STATE_FIELDS, snapshot):
        setattr(pipeline, name, value)


def _accept(pipeline, decision, allowed, original):
    before = _snapshot(pipeline)
    try:
        proposal = {key: copy.deepcopy(value) for key, value in decision.items()
                    if key != "resolution_scope"}
        identifier = decision["candidate_id"]
        unit_id = pipeline._accept(proposal, pipeline.rows[identifier]["source_spans"],
                                   [identifier], 1, identifier, allowed)
        for name, old in zip(STATE_FIELDS, original):
            current = getattr(pipeline, name)
            if any(key not in current or current[key] != value for key, value in old.items()):
                raise SourceError("Recovery cannot change an inherited unit, Context or provenance")
        if unit_id in before[0]:
            raise SourceError("Recovery cannot merge into an existing source unit")
        return unit_id
    except (SourceError, ValueError):
        _restore(pipeline, before)
        raise


def _binding(pipeline, old_nodes, instructions, policy_bytes, max_followups):
    return {"producer_version": VERSION,
        "producer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "source_fingerprint_sha256": digest(pipeline.project.fingerprint),
        "rule_result_sha256": pipeline.rule_sha,
        "parent_producer_binding": copy.deepcopy(old_nodes["producer_binding"]),
        "parent_producer_binding_sha256": digest(old_nodes["producer_binding"]),
        "ancestor_result_value_sha256": digest(old_nodes),
        "recovery_policy_sha256": hashlib.sha256(policy_bytes).hexdigest(),
        "prompt_sha256": {TASK: hashlib.sha256(instructions.encode("utf-8")).hexdigest()},
        "retrieval_policy": {"method": "bm25", "k1": 1.2, "b": 0.75,
            "exclude_full_supplied_ids_before_scoring": True, "include_tex_macro_names": True,
            "query_results": pipeline.options.context_query_results,
            "sources": "complete_rows_from_same_pinned_source_only"},
        "followup_policy": {"max_new_logical_model_calls_per_complete_failed_group": max_followups,
            "physical_attempt_policy": "caller_existing_client_only",
            "candidate_scoped_requests": True, "original_negatives_locked": True,
            "uncertainty_cause": "fresh_judge_semantic_assertion_not_host_certified"},
        "feedback_exposed_development": True, "held_out_test_passed": False,
        "original_round2_replayed_with_expanded_units": False}


def _recover_group(pipeline, failure, judge, binding, instructions, max_followups, original):
    terminal = failure["calls"][-1]
    payload = copy.deepcopy(terminal["logical_payload"])
    original_response = terminal["response"]
    ids = [row["candidate_id"] for row in payload["candidates"]]
    old = {row["candidate_id"]: row for row in original_response["decisions"]}
    if len(old) != len(ids) or set(old) != set(ids):
        raise SourceError("Terminal Context failure lacks its complete original decision set")
    trace = {"batch_id": failure["batch_id"], "original_failure_reason": failure["reason"],
        "complete_task_ids": ids, "original_terminal_payload": copy.deepcopy(payload),
        "original_terminal_response": copy.deepcopy(original_response),
        "original_calls": [{"logical_payload_sha256": digest(call["logical_payload"]),
                            "response_sha256": digest(call["response"])} for call in failure["calls"]],
        "new_calls": [], "candidate_dispositions": [], "status": "unresolved"}
    supplied = {row["source_id"] for row in payload["context_menu"]}
    initial_requests = [{**copy.deepcopy(request), "candidate_ids": ids}
                        for request in original_response["context_requests"]]
    added, audit = _retrieve(pipeline, initial_requests, supplied)
    payload["context_menu"].extend(added)
    payload.pop("previous_context_requests", None)
    payload.update(producer_binding=binding,
        recovery_eligible_candidate_ids=[key for key in ids if old[key]["decision"] in {"accept", "uncertain"}],
        retained_original_semantic_candidate_ids=[key for key in ids if old[key]["decision"] in {"reject", "context_only"}],
        original_terminal_response=copy.deepcopy(original_response),
        original_terminal_response_is_uncommitted=True,
        original_context_requests_had_no_candidate_scope=True,
        original_terminal_source_context_followup=terminal["logical_payload"]["source_context_followup"],
        seeded_source_retrieval=audit, previous_recovery_response=None,
        terminal_recovery_candidate_ids=[])
    trace["seeded_source_retrieval"] = copy.deepcopy(audit)
    new_wire = wire._WireJudge(judge, pipeline.wire_options)
    dispositions, resolved = {}, {}
    for followup in range(1, max_followups + 1):
        supplied = {row["source_id"] for row in payload["context_menu"]}
        payload.update(source_context_followup=followup, context_catalog=_catalog(pipeline, supplied),
                       terminal_recovery_candidate_ids=list(dispositions))
        payload["response_schema"] = _schema(payload)
        logical = copy.deepcopy(payload)
        call = {"source_context_followup": followup, "logical_payload": logical,
                "input_sha256": digest({"task_kind": TASK, "instructions": instructions, "payload": logical})}
        trace["new_calls"].append(call)
        try:
            response = new_wire(TASK, instructions, logical)
            call["raw_response"] = copy.deepcopy(response)
            call["response_sha256"] = digest(response)
            _validate_response(response, logical)
        except (SourceError, ValueError, TypeError) as error:
            call.update(status="technical_or_schema_failure", error_type=type(error).__name__, reason=str(error))
            trace["terminal_error"] = str(error)
            break
        call["status"] = "returned"
        pending = set()
        active_requests = []
        for request in response["context_requests"]:
            active = [key for key in request["candidate_ids"] if key not in dispositions]
            if active:
                pending.update(active)
                active_requests.append({**copy.deepcopy(request), "candidate_ids": active})
        for decision in response["decisions"]:
            key = decision["candidate_id"]
            if key in dispositions or key in pending:
                continue
            record = {"candidate_id": key, "decision": "uncertain", "reason": decision["reason"],
                      "unit_id": None, "batch_id": failure["batch_id"]}
            audit_record = {"candidate_id": key, "resolution_scope": decision["resolution_scope"],
                            "raw_decision": decision["decision"], "new_call": followup}
            try:
                if old[key]["decision"] in {"reject", "context_only"}:
                    if (decision["resolution_scope"] != "retain_original_semantics" or
                            decision["decision"] != old[key]["decision"]):
                        raise SourceError("Original negative semantic disposition is locked")
                    record.update(decision=old[key]["decision"], reason=old[key]["reason"])
                elif decision["resolution_scope"] == "retain_original_semantics":
                    if decision["decision"] != old[key]["decision"]:
                        raise SourceError("Retained original semantics cannot be reclassified")
                    record["reason"] = "Original uncommitted judgment retained; source-only recovery cannot resolve it"
                else:
                    record["decision"] = decision["decision"]
                    if decision["decision"] == "accept":
                        record["unit_id"] = _accept(pipeline, decision, supplied, original)
                audit_record["status"] = "validated" if record["decision"] != "uncertain" else "unresolved"
            except (SourceError, ValueError) as error:
                record.update(decision="uncertain", reason=str(error), unit_id=None)
                audit_record.update(status="validation_failed", reason=str(error))
            dispositions[key] = record
            audit_record["disposition"] = copy.deepcopy(record)
            trace["candidate_dispositions"].append(audit_record)
            if record["decision"] != "uncertain":
                resolved[key] = record
        call["pending_candidate_ids"] = [key for key in ids if key in pending]
        if not pending:
            break
        if followup == max_followups:
            trace["terminal_error"] = "required source context remains unresolved at new followup cap"
            break
        added, retrieval = _retrieve(pipeline, active_requests, supplied)
        call["source_retrieval"] = retrieval
        if not added:
            trace["terminal_error"] = "candidate-scoped source requests found no new same-paper source"
            break
        payload["context_menu"].extend(added)
        payload["previous_context_requests"] = copy.deepcopy(active_requests)
        payload["previous_recovery_response"] = copy.deepcopy(response)
    for key in ids:
        if key not in dispositions:
            dispositions[key] = {"candidate_id": key, "decision": "uncertain", "unit_id": None,
                "batch_id": failure["batch_id"], "reason": trace.get("terminal_error", "No new source-context followup authorized")}
            trace["candidate_dispositions"].append({"candidate_id": key, "status": "pending_source_or_call",
                                                    "disposition": copy.deepcopy(dispositions[key])})
    trace["status"] = "resolved" if len(resolved) == len(ids) else "incomplete"
    trace["wire_calls"] = copy.deepcopy(new_wire.records)
    return dispositions, resolved, trace


def resume(project, rule_input, old_nodes, replay_judge, judge,
           options=None, wire_options=None, max_followups=2):
    """Return a new derived result, after exact full old-host replay.

    Both injected judges receive compact packets. ``replay_judge`` must be
    cache-only; the caller validates its complete request/receipt/file bindings.
    ``judge`` may raise SourceError for an offline packet preflight: every failed
    group is still visited. No client, budget ledger or reference is constructed.
    The ancestor value hash is checked here; its saved-file hash is caller-bound.
    """
    if type(max_followups) is not int or not 0 <= max_followups <= 2:
        raise SourceError("At most two new source-context calls per complete failed group are allowed")
    if not callable(replay_judge) or not callable(judge) or not isinstance(old_nodes, dict):
        raise SourceError("A retained original result and two explicitly injected judges are required")
    baseline = copy.deepcopy(old_nodes)
    pipeline = _CapturePipeline(project, rule_input, replay_judge,
                                base.NodeOptions.parse(options), wire_options)
    replayed = pipeline.run()
    expected_calls = len(baseline.get("judgment_calls", []))
    if replayed != baseline or len(pipeline.retained_calls) != expected_calls:
        raise SourceError("Entire retained node result and original logical call count must replay exactly")
    if callable(getattr(replay_judge, "assert_finished", None)):
        replay_judge.assert_finished()
    policy_bytes = POLICY.read_bytes()
    instructions = pipeline.prompts[base.TASK_ROUND1].rstrip() + "\n\n" + policy_bytes.decode("utf-8")
    binding = _binding(pipeline, baseline, instructions, policy_bytes, max_followups)
    original = _snapshot(pipeline)
    traces, dispositions, resolved, resolved_batches = [], {}, {}, set()
    for failure in pipeline.context_failures:
        decisions, completed, trace = _recover_group(pipeline, failure, judge, binding,
                                                     instructions, max_followups, original)
        traces.append(trace)
        dispositions.update(decisions)
        resolved.update(completed)
        if trace["status"] == "resolved":
            resolved_batches.add(failure["batch_id"])
    result = copy.deepcopy(baseline)
    result.update(producer_version=VERSION, producer_binding=binding,
                  units=list(pipeline.units.values()), contexts=list(pipeline.contexts.values()),
                  unit_provenance=list(pipeline.provenance.values()))
    result["round1"]["decisions"] = [copy.deepcopy(dispositions.get(row["candidate_id"], row))
                                       for row in baseline["round1"]["decisions"]]
    for batch in result["round1"]["batches"]:
        if batch["batch_id"] in resolved_batches:
            batch["status"] = "judged"
    result["unresolved"] = [copy.deepcopy(row) for row in baseline["unresolved"]
        if row.get("candidate_id") not in resolved and row.get("batch_id") not in resolved_batches]
    for trace in traces:
        result["wire_protocol"]["actual_calls"].extend(copy.deepcopy(trace["wire_calls"]))
        for call in trace["new_calls"]:
            result["judgment_calls"].append({"task_kind": TASK, "input_sha256": call["input_sha256"],
                "status": "returned" if call["status"] == "returned" else "technical_failure",
                "source_context_followup": call["source_context_followup"],
                **({"response_sha256": call["response_sha256"]} if "response_sha256" in call else
                   {"error_type": call["error_type"]})})
    existing_proofs = {row["proof_unit_id"] for row in result["proof_attachment_hints"]}
    result["proof_attachment_hints"].extend(row for row in pipeline._proof_hints()
                                            if row["proof_unit_id"] not in existing_proofs)
    result["summary"].update(accepted_nodes=len(result["units"]), contexts=len(result["contexts"]),
        by_category=dict(Counter(row["category"] for row in result["units"])),
        unresolved_count=len(result["unresolved"]), status="complete" if not result["unresolved"] else "incomplete")
    result["context_request_recovery"] = {"schema_version": "gangmu.e4.context-request-recovery.v1",
        "binding": copy.deepcopy(binding), "baseline_full_result_equal": True,
        "original_calls_replayed": expected_calls,
        "original_call_bindings": [{"task_kind": row["task_kind"],
            "logical_payload_sha256": digest(row["logical_payload"]), "response_sha256": digest(row["response"])}
            for row in pipeline.retained_calls],
        "ancestry": {"original_unresolved": copy.deepcopy(baseline["unresolved"]),
            "original_round1": copy.deepcopy(baseline["round1"]), "original_summary": copy.deepcopy(baseline["summary"]),
            "original_round2_sha256": digest(baseline["round2"]),
            "inherited_units_sha256": digest(baseline["units"]),
            "inherited_contexts_sha256": digest(baseline["contexts"]),
            "inherited_unit_provenance_sha256": digest(baseline["unit_provenance"])},
        "groups": traces, "resolved_candidate_ids": list(resolved), "resolved_batch_ids": sorted(resolved_batches),
        "new_logical_call_count": sum(len(trace["new_calls"]) for trace in traces),
        "original_round2_preserved": result["round2"] == baseline["round2"],
        "whole_output_semantic_evaluation_required": True,
        "claim": "Development Context-loop continuation only; inherited round2 was not rerun with expanded existing_units. No held-out or mathematical acceptance claim."}
    if old_nodes != baseline:
        raise SourceError("Recovery mutated the caller's original result")
    project.assert_unchanged()
    return result
