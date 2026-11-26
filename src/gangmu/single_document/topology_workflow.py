"""Bounded final verification over a saved whole single-document graph.

Extraction policies are inputs, not parameters to optimize here.  Every model
packet uses saved node fragments and Contexts; raw TeX is only accessible via
the once-reserved complete-reextraction callback.  Journals precede graph binds
so a restart cannot reset counts or revive a removed relationship.
"""
from __future__ import annotations

from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import copy
import json
from pathlib import Path
import time

from gangmu.representation.graph import validate
from gangmu.representation.model import Graph, digest
from .implicit_dependencies import local, record, write_once, retrieve
from .topology import (State, analyze, graph_binding, relation_signature,
                       rank_cycle_edges, remove_relations, remove_units)
from .topology_integration import saved_graph_paper
from .topology_client import POLICY_VERSION, validate_runtime_policy
from . import topology_semantics as semantics

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.e4.topology-workflow.v2"


def _read(path):
    return json.loads(Path(path).read_bytes())


def _canonical(graph):
    return Graph.model_validate_json(json.dumps(graph, ensure_ascii=False)).model_dump(mode="json")


def implementation_bindings(config, root=ROOT):
    names = ["src/gangmu/single_document/" + name for name in (
        "topology.py", "topology_workflow.py", "topology_semantics.py",
        "topology_integration.py", "topology_client.py", "implicit_client.py",
        "implicit_dependencies.py", "core_client.py", "core_response_stream.py",
        "candidates.py", "retrieval.py")]
    names += ["src/gangmu/representation/" + name for name in ("model.py", "graph.py", "sources.py")]
    names += [config[key] for key in ("review_prompt", "repair_prompt", "value_prompt")]
    return [record(name, root) for name in names]


def pair_of(relation):
    return {"candidate_id": relation["id"], **{key: relation[key] for key in ("source", "target", "type")}}


def pack(paper, pairs, limits, *, evaluation=False):
    """Keep complete bodies; reject oversized work instead of silently cutting it."""
    packets, current, oversized = [], [], []
    for pair in pairs:
        proposed = current + [pair]
        packet = semantics.make_packet(paper, proposed, evaluation=evaluation)
        if len(proposed) > limits["max_pairs"] or len(json.dumps(packet, ensure_ascii=False).encode()) > limits["max_payload_bytes"]:
            if current:
                packets.append(semantics.make_packet(paper, current, evaluation=evaluation))
            current = [pair]
            packet = semantics.make_packet(paper, current, evaluation=evaluation)
            if len(json.dumps(packet, ensure_ascii=False).encode()) > limits["max_payload_bytes"]:
                oversized.append(pair["candidate_id"])
                current = []
        else:
            current = proposed
    if current:
        packets.append(semantics.make_packet(paper, current, evaluation=evaluation))
    return packets, oversized


def value_packet(paper, ids):
    fields = {"node_id": {"type": "string", "enum": ids},
              "verdict": {"type": "string", "enum": ["keep", "delete", "unresolved"]},
              "quote": {"type": "string"}, "reason": {"type": "string"}}
    return {"schema_version": VERSION + ".node-value", "paper_id": paper.paper_id,
        "nodes": paper.packet_nodes(ids), "contexts": paper.packet_contexts(ids),
        "response_schema": {"type": "object", "properties": {"decisions": {"type": "array", "items": {
            "type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}},
            "required": ["decisions"], "additionalProperties": False}}


def validate_values(paper, ids, response):
    if not isinstance(response, dict) or set(response) != {"decisions"} or not isinstance(response["decisions"], list):
        raise ValueError("Incomplete node-value population")
    if Counter(d.get("node_id") for d in response["decisions"] if isinstance(d, dict)) != Counter(ids):
        raise ValueError("Each isolated node requires exactly one value judgment")
    checked = []
    for row in response["decisions"]:
        if (set(row) != {"node_id", "verdict", "quote", "reason"}
                or any(not isinstance(v, str) for v in row.values())
                or row["verdict"] not in {"keep", "delete", "unresolved"} or not row["reason"].strip()):
            raise ValueError("Invalid node-value fields")
        value, errors = copy.deepcopy(row), []
        if row["verdict"] in {"delete", "keep"} and (not row["quote"].strip()
                or not any(row["quote"] in part for part in paper.units[row["node_id"]]["original_text"])):
            errors.append("value_quote_not_bound_to_saved_node_fragment")
        if errors:
            value["verdict"] = "unresolved"
        value.update(model_verdict=row["verdict"], host_validation_errors=errors)
        checked.append(value)
    return sorted(checked, key=lambda row: ids.index(row["node_id"]))


def expanded_candidates(paper, focus, config, ordinal, attempted, state):
    options = {key: (value[ordinal - 1] if isinstance(value, list) else value)
               for key, value in config["expanded_retrieval"].items() if key != "max_pairs_per_node_round"}
    found = retrieve(paper, options)
    pool = found["candidates"] + [p for p in found["exclusions"] if p["reason"] not in
            {"frozen_explicit_dependency", "same_assertion_own_proof_goal_not_premise"}]
    # Existing direct-use retrieval is broadened, never loosened at admission.
    eligible, excluded = [], []
    seen = set()
    for candidate in pool:
        pair = {**candidate, "candidate_id": "topology-candidate:" + digest(
                [paper.paper_id, candidate["source"], candidate["target"], candidate["type"]])[:24]}
        if not (set(focus) & {pair["source"], pair["target"]}) or pair["candidate_id"] in seen:
            continue
        seen.add(pair["candidate_id"])
        signature = relation_signature(paper.graph, pair, state.snapshot()["revision_id"])
        error = semantics.host_e2_errors(paper, pair)
        if signature in attempted or state.blocked(paper.graph, pair) or error:
            excluded.append({"candidate": pair, "reason": "already_judged_or_blocked" if not error else "e2_type_gate",
                             "e2_errors": error})
        else:
            pair["processing_signature"] = signature
            eligible.append(pair)
    def rank(pair):
        hits = pair["retrieval"]
        score = max([h.get("score", 0) for h in hits] + [0])
        feature = any(h["channel"] == "saved_declared_math_feature" for h in hits)
        return (not feature, -score, abs(pair["source_order_offset"]), pair["candidate_id"])
    selected, selected_ids = [], set()
    maximum = config["expanded_retrieval"]["max_pairs_per_node_round"]
    for uid in focus:
        incoming = sorted([p for p in eligible if p["target"] == uid], key=rank)
        outgoing = sorted([p for p in eligible if p["source"] == uid], key=rank)
        balanced = []
        for i in range(max(len(incoming), len(outgoing))):
            for direction in (incoming, outgoing):
                if i < len(direction):
                    balanced.append(direction[i])
        for pair in balanced[:maximum]:
            if pair["candidate_id"] not in selected_ids:
                selected.append(pair)
                selected_ids.add(pair["candidate_id"])
    return {"options": options, "focus": focus, "round": ordinal, "candidates": selected,
            "eligible_before_cap": len(eligible), "excluded": excluded,
            "direction_counts": {uid: {"incoming": sum(p["target"] == uid for p in selected),
                                        "outgoing": sum(p["source"] == uid for p in selected)} for uid in focus},
            "raw_source_reading": False}


class TopologyWorkflow:
    def __init__(self, config, judge, *, root=ROOT):
        validate_runtime_policy(config)
        self.root, self.config, self.judge = Path(root).resolve(), copy.deepcopy(config), judge
        # This fork is used for anomalous-cycle diagnosis only.  Local new
        # relationships and node value decisions each have one production call.
        self.evaluator = judge.fork(allocation="evaluation")

    def _paper(self, graph):
        return saved_graph_paper(graph, self.references, predecessor=self.predecessor,
                                 revision_id=self.state.snapshot()["revision_id"], root=self.root)

    def _value_signature(self, uid):
        paper = self._paper(self.graph)
        return digest({"unit": paper.units[uid], "contexts": paper.contexts_for(uid),
            "proof_associations": [r for r in self.graph["relations"] if r["type"] == "proves"
                                    and uid in {r["source"], r["target"]}]})

    def _current_dispositions(self):
        units = {u["id"] for u in self.graph["units"]}
        return [e["details"] for e in self.state.snapshot()["events"] if e["kind"] == "node_value_disposition"
                and e["details"]["node_id"] in units
                and e["details"].get("node_signature") == self._value_signature(e["details"]["node_id"])]

    def _snapshot(self, graph, references):
        value = {"graph": graph, "reference_ledger": references}
        return write_once(self.directory / "snapshots" / (digest(graph) + ".json"), value, self.root)

    def _inherit_history(self, initial):
        """Reuse completed historical work without editing its files or counts.

        A policy migration starts from the exact saved final diagnostic graph.
        Prior expansions consume the new single pass.  Confirmed removals and
        content-bound node dispositions survive, regardless of old score gates.
        """
        history = self.predecessor.get("topology_history")
        self.history = None
        if history is None:
            return
        if not isinstance(history, dict) or set(history) != {"state", "result", "campaign_id"}:
            raise ValueError("Topology history requires bound state, result and campaign identity")
        if history["campaign_id"] != self.config.get("history_campaign_id"):
            raise ValueError("Historical topology campaign differs from the migration policy")
        material = {}
        for key in ("state", "result"):
            pointer = history[key]
            if not isinstance(pointer, dict) or set(pointer) != {"path", "sha256"}:
                raise ValueError("Topology history needs immutable path/SHA256 bindings")
            if record(pointer["path"], self.root) != pointer:
                raise ValueError("Historical topology artifact bytes changed")
            material[key] = _read(local(pointer["path"], self.root))
        old_state, old_result = material["state"], material["result"]
        if (digest(old_result.get("graph")) != digest(initial)
                or old_result.get("binding") != old_state.get("policy_binding")
                or old_result.get("report", {}).get("graph_sha256") != digest(initial)):
            raise ValueError("Migration must reuse the exact finalized historical graph and policy binding")
        source_binding = {**history["state"],
            "path": local(history["state"]["path"], self.root).relative_to(ROOT).as_posix()}
        imported = self.state.import_policy_history(initial, old_state, source_binding=source_binding)
        self.history = {**copy.deepcopy(history), **imported,
            "historical_report_completed": old_result["report"].get("completed"),
            "historical_local_quality": copy.deepcopy(old_result["report"].get("local_quality")),
            "old_quality_gate_applied_to_runtime": False}
        if any(e["kind"] == "historical_node_dispositions_imported" for e in self.state.snapshot()["events"]):
            return
        self.graph = initial
        units = {u["id"] for u in initial["units"]}
        dispositions = old_result["report"].get("isolated_dispositions", [])
        retained = {}
        for row in dispositions:
            uid = row.get("node_id")
            if uid not in units or row.get("action") != "keep":
                continue
            if row.get("node_signature") != self._value_signature(uid):
                continue
            retained[uid] = copy.deepcopy(row)
        for uid, row in retained.items():
            if any(current["node_id"] == uid for current in self._current_dispositions()):
                continue
            row.update(history_result=copy.deepcopy(history["result"]),
                       historical_basis=row.get("basis"), basis="content_bound_historical_disposition")
            self.state.record_event(initial, "node_value_disposition", row)
        self.state.record_event(initial, "historical_node_dispositions_imported", {
            "result": copy.deepcopy(history["result"]), "node_ids": sorted(retained)})

    def _recover(self):
        # Prepared immutable transactions can survive a crash before graph bind.
        for path in sorted((self.directory / "mutations").glob("*.json")):
            transaction = _read(path)
            current = self.state.snapshot()["graph_binding"]["graph_sha256"]
            if current == transaction["before"]["graph_sha256"]:
                if not any(e["kind"] == "graph_change" and e["details"]["change_id"] == transaction["change_id"]
                           for e in self.state.snapshot()["events"]):
                    self.state.record_event(transaction["before_graph"], "graph_change",
                        {"change_id": transaction["change_id"], "transaction": record(path, self.root)})
                self.state.bind_graph(transaction["after_graph"], expected_before_sha256=current,
                                      reason=transaction["change"]["reason"])
        path = self.directory / "snapshots" / (self.state.snapshot()["graph_binding"]["graph_sha256"] + ".json")
        saved = _read(path)
        if digest(saved["graph"]) != self.state.snapshot()["graph_binding"]["graph_sha256"]:
            raise ValueError("Resumed whole-graph bytes differ from persistent state")
        self.graph, self.references = saved["graph"], saved["reference_ledger"]
        checkpoint = self.directory / "complete-reextraction-checkpoint.json"
        if self.state.snapshot()["reextract"] is not None and self.state.snapshot()["reextract"]["status"] == "completed":
            if not any(e["kind"] == "reextract_production" for e in self.state.snapshot()["events"]):
                output = _read(checkpoint)
                self.state.record_event(self.graph, "reextract_production", {
                    "binding": output["binding"], "completion_evidence": output["completion_evidence"],
                    "runtime_precision_evaluation": False})

    def _commit(self, after, change):
        after = _canonical(after)
        before = self.graph
        change["reason"] = change.get("reason") or change["classification"]
        after["snapshot_id"] = "topology:" + digest({"prior": digest(before), "change": change, "after": after})[:24]
        references = copy.deepcopy(self.references)
        removed = {u["id"] for u in change.get("removed_units", [])}
        if removed:
            # External syntactic ledger is versioned too; preserve original pointers.
            for row in references:
                row["topology_removed_unit_ids"] = sorted(removed)
                row["topology_endpoint_revalidation_required"] = True
        self._snapshot(after, references)
        transaction = {"before": graph_binding(before), "after": graph_binding(after),
                       "before_graph": before, "after_graph": after, "change": change,
                       "reference_ledger": references}
        transaction["change_id"] = digest(transaction)
        path = self.directory / "mutations" / (f"{len(self.state.snapshot()['events']):05d}-" + transaction["change_id"] + ".json")
        binding = write_once(path, transaction, self.root)
        self.state.record_event(before, "graph_change", {"change_id": transaction["change_id"], "transaction": binding})
        self.state.bind_graph(after, expected_before_sha256=digest(before), reason=change["reason"])
        self.graph, self.references = after, references

    def _enforce_historical_prohibitions(self):
        """A completed re-extraction does not authorize restoring removed facts.

        Preserve the actual frozen production output and record any ensuing
        topology removal separately.  Unchanged participants and Contexts keep
        their old prohibition even when a new revision/relationship ID is used.
        """
        for relation in list(self.graph["relations"]):
            previous = next((row for row in self.state.snapshot()["blocked_relations"].values()
                if row.get("active") and relation_signature(self.graph, relation, row["revision_id"]) == row["signature"]), None)
            if previous is None:
                continue
            reason = "Do not restore the content-bound historical removal: " + previous["reason"]
            self.state.record_event(self.graph, "historical_relation_removal_enforced", {
                "relation": copy.deepcopy(relation), "historical_removal": previous,
                "classification": previous["classification"], "reason": reason})
            after, change = remove_relations(self.graph, [relation["id"]],
                classification="historical_exclusion", reason=reason)
            self._commit(after, change)

    def _call(self, task, prompt, payload, *, evaluation, directory):
        judge = self.evaluator if evaluation else self.judge
        prompt_binding = record(prompt, self.root)
        plan = {"task": task, "prompt": prompt_binding, "payload": payload,
                "graph_binding": graph_binding(self.graph), "execution_binding": self.binding,
                "allocation": "evaluation" if evaluation else "production"}
        key = digest(plan)
        path = directory / (key + ".json")
        if path.exists():
            saved = _read(path)
            if saved["plan"] != plan or saved["result_sha256"] != digest(saved["response"]):
                raise ValueError("Cached topology call differs from complete inputs")
            return saved["response"]
        write_once(directory / "plans" / (key + ".json"), plan, self.root)
        try:
            response = judge(task, local(prompt, self.root).read_text(), payload)
        except Exception as exc:
            write_once(directory / "failures" / (key + "-" + str(time.time_ns()) + ".json"),
                       {"plan": plan, "error_type": type(exc).__name__, "error": str(exc)}, self.root)
            raise
        receipts = [r for r in judge.receipts() if r["task_kind"] == task and
                    r["request_key"] == digest({"purpose": "topology:" + plan["allocation"] + ":" + task,
                        "instructions": local(prompt, self.root).read_text(), "payload": payload, "model": judge.client.model})]
        # Multiple in-process reuses refer to the same request; retain one original receipt.
        if not receipts:
            raise ValueError("A real gate receipt is required for each topology call")
        write_once(path, {"plan": plan, "response": response, "result_sha256": digest(response),
                          "receipt": receipts[0]}, self.root)
        return response

    def _judge_pairs(self, pairs, task, *, evaluation, directory):
        paper = self._paper(self.graph)
        material, oversized = pack(paper, pairs, self.config["packets"], evaluation=evaluation)
        plan = {"task": task, "pairs": pairs, "packets": material, "oversized": oversized,
                "graph_binding": graph_binding(self.graph)}
        write_once(directory / "plan.json", plan, self.root)
        if oversized:
            raise ValueError("Technical unfinished oversized topology candidates: " + repr(oversized))
        prompt = self.config["review_prompt" if evaluation else "repair_prompt"]
        def run(payload):
            response = self._call(task, prompt, payload, evaluation=evaluation, directory=directory / "calls")
            return semantics.validate_decisions(paper, payload, response, evaluation=evaluation)
        with ThreadPoolExecutor(max_workers=min(3, max(1, len(material)))) as pool:
            batches = list(pool.map(run, material))
        positions = {p["candidate_id"]: i for i, p in enumerate(pairs)}
        decisions = sorted([d for batch in batches for d in batch], key=lambda d: positions[d["candidate_id"]])
        write_once(directory / "decisions.json", decisions, self.root)
        return decisions

    def _review(self, row):
        status = {"supported": "supported", "contradicted": "incorrect", "unresolved": "unresolved"}[row["verdict"]]
        review = {"status": status, "witness_quality": "exact" if status == "supported" else "unknown",
                  "scope_quality": "explicit" if status == "supported" else "unknown",
                  "model_decision": row, "independent_session": True}
        if row.get("host_e2_errors"):
            review.update(status="incorrect", error_kind="existing_e2_type_or_ownership_rule",
                          integrity_errors=row["host_e2_errors"])
        return review

    def _cycle_phase(self, reextract):
        while analyze(self.graph)["cycle_relation_ids"]:
            diagnosis = analyze(self.graph)
            by_id = {r["id"]: r for r in self.graph["relations"]}
            pairs = [pair_of(by_id[i]) for i in diagnosis["cycle_relation_ids"]]
            decisions = self._judge_pairs(pairs, "independent-cycle-edge-review", evaluation=True,
                directory=self.directory / "cycle-reviews" / digest(self.graph))
            reviews = {d["candidate_id"]: self._review(d) for d in decisions}
            for rid, review in reviews.items():
                self.state.record_review(self.graph, by_id[rid], review)
            wrong = [rid for rid in diagnosis["cycle_relation_ids"] if reviews[rid]["status"] == "incorrect"]
            for rid in wrong:
                row, review = by_id[rid], reviews[rid]
                classification = "e2_integrity_error" if review.get("integrity_errors") else "mathematical_error"
                reason = json.dumps(review, ensure_ascii=False)
                self.state.record_relation_removal(self.graph, row, classification=classification, reason=reason, review=review)
                after, change = remove_relations(self.graph, [rid], reason=reason, classification=classification)
                self._commit(after, change)
            if wrong:
                continue
            reservation = self.state.snapshot()["reextract"]
            if reservation is None or reservation["status"] != "completed":
                reservation = self.state.reserve_reextract(self.graph, reviews=reviews,
                    reason="Cycle persists after independent complete edge audit; one frozen-stage re-extraction")
                if reextract is None:
                    raise ValueError("Persisting cycle requires the authorized fixed-stage re-extraction callback")
                self.state.mark_reextract_attempt(reservation["revision_id"])
                try:
                    output = reextract(reservation["revision_id"], diagnosis, self.state.snapshot())
                    write_once(self.directory / "complete-reextraction-checkpoint.json", output, self.root)
                    new = _canonical(output["graph"])
                    references = output["reference_ledger"]
                    self._snapshot(new, references)
                    old = {u["id"]: u for u in self.graph["units"]}
                    mapping = {u["id"]: u["id"] for u in new["units"] if u["id"] in old
                               and u["original_text"] == old[u["id"]]["original_text"]
                               and u["category"] == old[u["id"]]["category"]}
                    self.state.mark_reextract_completed(reservation["revision_id"], new,
                        node_mapping=mapping, completion_evidence=output["completion_evidence"])
                    self.graph, self.references = new, references
                    self.state.record_event(new, "reextract_production", {
                        "binding": output["binding"], "completion_evidence": output["completion_evidence"],
                        "runtime_precision_evaluation": False})
                    self._enforce_historical_prohibitions()
                except Exception as exc:
                    if self.state.snapshot()["reextract"]["status"] != "completed":
                        self.state.mark_reextract_failed(reservation["revision_id"], reason=str(exc))
                    raise
                continue
            # Review completed re-extraction first; only now may supported edges be cropped.
            ranking = rank_cycle_edges(self.graph, reviews)
            selected = ranking[0]["relation"]
            reason = "One complete re-extraction and full cycle audit still leave a cycle; predeclared weakest-first crop"
            self.state.record_relation_removal(self.graph, selected, classification="structural_crop",
                                              reason=reason, review=reviews[selected["id"]], ranking=ranking)
            after, change = remove_relations(self.graph, [selected["id"]], classification="structural_crop", reason=reason)
            self._commit(after, change)

    def _isolation_phase(self, reextract):
        while True:
            report = analyze(self.graph)
            if report["structural_errors"]:
                raise ValueError("Whole graph has unresolved structural endpoint errors")
            snapshot = self.state.snapshot()
            values = {row["node_id"]: row for row in self._current_dispositions()}
            focus = [uid for uid in report["completely_isolated"] if uid not in values]
            # Resume reserved work rather than allocating a new document round.
            complete = {e["details"]["round_index"] for e in snapshot["events"] if e["kind"] == "isolation_round_complete"}
            pending = [e for e in snapshot["events"] if e["kind"] == "isolation_round" and e["index"] not in complete]
            if not pending and (not focus or snapshot["document_rounds"] >= self.config["max_document_isolation_rounds"]):
                break
            if pending:
                event = pending[-1]
            else:
                event = self.state.begin_isolation_round(self.graph, focus, reason="Expand both premise and consumer directions")
            ordinal = event["details"]["document_round"]
            directory = self.directory / "isolation-rounds" / f"{ordinal:02d}"
            attempted = {s for e in self.state.snapshot()["events"] if e["kind"] == "local_candidate_judgments"
                         for s in e["details"]["processing_signatures"]}
            path = directory / "retrieval.json"
            if path.exists():
                retrieval = _read(path)
            else:
                retrieval = expanded_candidates(self._paper(self.graph), event["details"]["node_ids"],
                    self.config, ordinal, attempted, self.state)
                write_once(path, retrieval, self.root)
            pairs = retrieval["candidates"]
            # Persist the sole production decision before mutating the graph.
            results_path = directory / "round-results.json"
            if results_path.exists():
                result = _read(results_path)
            else:
                decisions = self._judge_pairs(pairs, "isolated-node-direct-use", evaluation=False, directory=directory / "production")
                relations, evidence, witnesses = semantics.accept_relations(self._paper(self.graph), pairs, decisions)
                result = {"pairs": pairs, "production_decisions": decisions, "relations": relations,
                          "evidence": evidence, "witnesses": witnesses,
                          "runtime_precision_evaluation": False}
                write_once(results_path, result, self.root)
            if not any(e["kind"] == "local_candidate_judgments" and e["details"]["round_index"] == event["index"]
                       for e in self.state.snapshot()["events"]):
                self.state.record_event(self.graph, "local_candidate_judgments", {"round_index": event["index"],
                    "processing_signatures": [p["processing_signature"] for p in pairs], "results": record(results_path, self.root)})
            if result["relations"]:
                after = copy.deepcopy(self.graph)
                known = {r["id"] for r in after["relations"]}
                retain = [r for r in result["relations"] if r["id"] not in known
                          and not self.state.blocked(self.graph, r)]
                if retain:
                    after["relations"].extend(retain)
                    eids = {i for r in retain for i in r["evidence"]}
                    after["evidence"].extend(e for e in result["evidence"] if e["id"] in eids)
                    self._commit(after, {"classification": "local_dependency_addition", "reason": "First-pass model accepted isolated-node direct use with saved-input host gates",
                        "added_relations": retain, "all_producer_accepted": result["relations"],
                        "runtime_precision_evaluation": False, "results": record(results_path, self.root)})
            if not any(e["kind"] == "isolation_round_complete" and e["details"]["round_index"] == event["index"]
                       for e in self.state.snapshot()["events"]):
                self.state.record_event(self.graph, "isolation_round_complete", {"round_index": event["index"],
                    "results": record(results_path, self.root), "diagnostics": analyze(self.graph)})
            self._cycle_phase(reextract)
        report = analyze(self.graph)
        # Association-only proof/statement pairs are legitimate; no paid forced link.
        for uid in report["proves_only_isolated"]:
            details = {"node_id": uid, "action": "keep", "reason": "No dependency incidence, but a real saved proves association connects this mathematical node",
                       "proves_relation_ids": report["isolated_proves_relations"][uid], "basis": "saved_association_and_whole_node",
                       "node_signature": self._value_signature(uid)}
            if not any(row["node_id"] == uid for row in self._current_dispositions()):
                self.state.record_event(self.graph, "node_value_disposition", details)
        known = {row["node_id"] for row in self._current_dispositions()}
        ids = [uid for uid in report["completely_isolated"] if uid not in known]
        for uid in ids:
            if uid not in analyze(self.graph)["completely_isolated"]:
                continue
            paper = self._paper(self.graph)
            node_signature = self._value_signature(uid)
            payload = value_packet(paper, [uid])
            response = self._call("isolated-node-value", self.config["value_prompt"], payload,
                evaluation=False, directory=self.directory / "node-values" / digest(self.graph) / "calls")
            decision = validate_values(paper, [uid], response)[0]
            action = "keep" if decision["verdict"] != "delete" else "delete"
            if action == "delete":
                try:
                    after, change = remove_units(self.graph, [uid], reason=decision["reason"])
                    self._commit(after, change)
                except ValueError as exc:
                    action = "keep"
                    decision["integrity_retention_reason"] = str(exc)
            self.state.record_event(self.graph, "node_value_disposition", {"node_id": uid, "action": action,
                "reason": decision["reason"], "model_decision": decision, "basis": "saved_node_value_review",
                "node_signature": node_signature})

    def __call__(self, *, graph, predecessor, revision_id, output, reextract=None):
        self.directory = local(output, self.root)
        self.directory.mkdir(parents=True, exist_ok=True)
        initial = _canonical(graph)
        self.predecessor = copy.deepcopy(predecessor)
        self.references = copy.deepcopy(predecessor.get("reference_ledger", []))
        self.binding = {"config_sha256": digest(self.config), "predecessor_sha256": digest(predecessor),
            "runtime_policy": POLICY_VERSION, "implementation": implementation_bindings(self.config, self.root),
            "model": copy.deepcopy(self.judge.client.model)}
        write_once(self.directory / "execution-binding.json", self.binding, self.root)
        self._snapshot(initial, self.references)
        self.state = State(self.directory / "state.json", initial["documents"][0]["id"], revision_id,
                           initial, binding=self.binding, max_node_rounds=1, max_document_rounds=1)
        self._inherit_history(initial)
        self._recover()
        self._enforce_historical_prohibitions()
        result_path = self.directory / "workflow-result.json"
        if result_path.exists():
            result = _read(result_path)
            if result["binding"] != self.binding or digest(result["graph"]) != digest(self.graph):
                raise ValueError("Final cache or current state differs from full execution binding")
            return {"graph": result["graph"], "report": result["report"]}
        write_once(self.directory / "initial-diagnostics.json", analyze(initial), self.root)
        self._cycle_phase(reextract)
        self._isolation_phase(reextract)
        self._cycle_phase(reextract)
        final = analyze(self.graph)
        snapshot = self.state.snapshot()
        dispositions = [e["details"] for e in snapshot["events"] if e["kind"] == "node_value_disposition"]
        handled = {r["node_id"] for r in self._current_dispositions()}
        production = [_read(e["details"]["results"]["path"] if Path(e["details"]["results"]["path"]).is_absolute()
                            else self.root / e["details"]["results"]["path"])
                      for e in snapshot["events"] if e["kind"] == "local_candidate_judgments"]
        decisions = [d for r in production for d in r["production_decisions"]]
        repairs = {"candidate_count": sum(len(r["pairs"]) for r in production),
            "production_accepted": sum(len(r["relations"]) for r in production),
            "production_rejected": sum(d["verdict"] == "reject" for d in decisions),
            "production_unresolved": sum(d["verdict"] == "unresolved" for d in decisions),
            "retained_added_relations": [r["id"] for batch in production for r in batch["relations"]
                                        if r["id"] in {row["id"] for row in self.graph["relations"]}],
            "runtime_precision_evaluation": False, "mathematical_correctness_certified": False}
        rerun_production = [e["details"] for e in snapshot["events"] if e["kind"] == "reextract_production"]
        qualification = validate(Graph.model_validate_json(json.dumps(self.graph)))
        rerun_complete = not snapshot["reextract"] or snapshot["reextract"]["status"] == "completed"
        report = {"schema_version": VERSION + ".report", "paper_id": initial["documents"][0]["paper_id"],
            "runtime_policy": POLICY_VERSION,
            "completed": final["dag"] and not final["structural_errors"]
                         and set(final["dependency_isolated"]) <= handled and rerun_complete,
            "initial": analyze(initial), "final": final, "local_repairs": repairs,
            "isolated_dispositions": dispositions, "reextraction": snapshot["reextract"],
            "reextraction_production": rerun_production, "document_isolation_rounds": snapshot["document_rounds"],
            "historical_policy": getattr(self, "history", None),
            "runtime_precision_gate": False,
            "mathematical_error_removals": [r for r in snapshot["blocked_relations"].values() if r["classification"] == "mathematical_error"],
            "e2_integrity_removals": [r for r in snapshot["blocked_relations"].values() if r["classification"] == "e2_integrity_error"],
            "structural_crops": [r for r in snapshot["blocked_relations"].values() if r["classification"] == "structural_crop"],
            "formal_e2_eligible": not qualification.errors, "formal_e2_errors": list(qualification.errors),
            "automatic_e3_publication": False, "whole_e4_t5_complete": False,
            "recall_measured": False, "graph_sha256": digest(self.graph)}
        write_once(self.directory / "final-diagnostics.json", final, self.root)
        write_once(result_path, {"binding": self.binding, "graph": self.graph, "report": report}, self.root)
        return {"graph": self.graph, "report": report}
