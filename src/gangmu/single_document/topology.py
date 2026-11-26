"""Whole accepted-graph topology and durable bounded repair bookkeeping.

This module makes no model or source call.  Topology is a structural property,
not mathematical certification.  Run artifacts carry repair decisions; MathIR
records acquire no extra fields.  Raw diagnostic dictionaries may contain
self-edges forbidden by the formal E2 schema, so ``analyze`` deliberately does
not pass them through the E2 validator before locating the fault.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import heapq
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterator

from gangmu.representation.graph import GraphIndex, validate as validate_graph
from gangmu.representation.model import Graph, Relation, Source, digest

VERSION = "gangmu.e4.topology.v1"
STATE_VERSION = "gangmu.e4.topology_state.v1"
RANKING_VERSION = "gangmu.e4.topology_evidence_ranking.v1"
DEPENDENCIES = frozenset({"statement_dependency", "proof_dependency"})
ASSOCIATIONS = DEPENDENCIES | {"proves"}
ROOT = Path(__file__).resolve().parents[3]
REVIEW_STATUSES = frozenset({"supported", "incorrect", "unresolved", "technical_incomplete"})
REMOVAL_CLASSIFICATIONS = frozenset({"mathematical_error", "e2_integrity_error", "structural_crop", "historical_exclusion"})
RANKING_POLICY = {
    "schema_version": RANKING_VERSION,
    "weakest_first": ["independent_review", "use_witness", "scope", "stable_relation_id"],
    "independent_review": {"unresolved": 0, "supported": 1},
    "use_witness": {"missing": 0, "unknown": 0, "partial": 1, "exact": 2},
    "scope": {"missing": 0, "unknown": 0, "limited": 1, "explicit": 2},
    "source_mode_is_probability": False,
}


class TopologyError(ValueError):
    """Invalid topology input, binding or attempted repair transition."""


class EvidenceInvalidationError(TopologyError):
    def __init__(self, relation_ids: list[str]):
        self.relation_ids = relation_ids
        super().__init__("change invalidates claim-bound evidence: " + ", ".join(relation_ids))


def _copy_graph(graph: Graph | dict) -> dict:
    if isinstance(graph, Graph):
        value = graph.model_dump(mode="json")
    elif isinstance(graph, dict):
        value = copy.deepcopy(graph)
    else:
        raise TopologyError("graph must be a Graph or JSON object")
    # Also rejects non-JSON or nonfinite values before hashes or mutations.
    digest(value)
    for key in ("documents", "contexts", "units", "relations", "evidence", "identity_views"):
        records = value.setdefault(key, [])
        if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
            raise TopologyError(f"{key} must contain record objects")
    seen = set()
    for key in ("documents", "contexts", "units", "relations", "evidence", "identity_views"):
        for record in value[key]:
            record_id = record.get("id")
            if not isinstance(record_id, str) or not record_id:
                raise TopologyError(f"{key} has an empty or invalid ID")
            if record_id in seen:
                raise TopologyError(f"duplicate snapshot ID: {record_id}")
            seen.add(record_id)
    return value


def graph_binding(graph: Graph | dict) -> dict:
    value = _copy_graph(graph)
    return {"snapshot_id": value.get("snapshot_id"), "schema_version": value.get("schema_version"),
            "graph_sha256": digest(value)}


def _components(nodes: list[str], adjacency: dict[str, list[str]]) -> list[list[str]]:
    """Iterative, deterministic connected components including singleton units."""
    remaining = set(nodes)
    result = []
    for start in nodes:
        if start not in remaining:
            continue
        remaining.remove(start)
        stack, members = [start], []
        while stack:
            node = stack.pop()
            members.append(node)
            for target in reversed(adjacency[node]):
                if target in remaining:
                    remaining.remove(target)
                    stack.append(target)
        result.append(sorted(members))
    return sorted(result, key=lambda part: tuple(part))


def _sccs(nodes: list[str], outgoing: dict[str, list[tuple[str, str]]],
          incoming: dict[str, list[tuple[str, str]]]) -> list[list[str]]:
    """Iterative Kosaraju; no recursion limit on a long mathematical chain."""
    visited, finished = set(), []
    for start in nodes:
        if start in visited:
            continue
        visited.add(start)
        stack: list[tuple[str, Iterator[tuple[str, str]]]] = [(start, iter(outgoing[start]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                finished.append(node)
                stack.pop()
            elif child[0] not in visited:
                visited.add(child[0])
                stack.append((child[0], iter(outgoing[child[0]])))
    visited.clear()
    components = []
    for start in reversed(finished):
        if start in visited:
            continue
        visited.add(start)
        stack, members = [start], []
        while stack:
            node = stack.pop()
            members.append(node)
            for target, _ in reversed(incoming[node]):
                if target not in visited:
                    visited.add(target)
                    stack.append(target)
        components.append(sorted(members))
    return sorted(components, key=lambda part: tuple(part))


def _witness(members: list[str], outgoing: dict[str, list[tuple[str, str]]]) -> dict:
    allowed = set(members)
    color, parent = {}, {}
    for start in members:
        if color.get(start):
            continue
        color[start] = 1
        stack = [(start, iter(outgoing[start]))]
        while stack:
            node, children = stack[-1]
            child = next(children, None)
            if child is None:
                color[node] = 2
                stack.pop()
                continue
            target, relation_id = child
            if target not in allowed:
                continue
            if not color.get(target):
                color[target] = 1
                parent[target] = (node, relation_id)
                stack.append((target, iter(outgoing[target])))
            elif color[target] == 1:
                route_nodes, route_relations = [node], []
                cursor = node
                while cursor != target:
                    upper, edge_id = parent[cursor]
                    route_relations.append(edge_id)
                    route_nodes.append(upper)
                    cursor = upper
                return {"node_ids": list(reversed(route_nodes)) + [target],
                        "relation_ids": list(reversed(route_relations)) + [relation_id]}
    raise TopologyError("cyclic SCC has no cycle witness")


def analyze(graph: Graph | dict) -> dict:
    """Check ALL units and the union of accepted mathematical dependency types.

    One deterministic cycle witness per cyclic SCC is returned, together with
    every internal relation ID; all internal edges belong to some cycle.  This
    avoids exponential enumeration of all simple cycles.  ``proves`` affects
    association/isolation diagnostics only; ``identity`` affects neither.
    """
    value = _copy_graph(graph)
    units = {row["id"]: row for row in value["units"]}
    nodes = sorted(units)
    outgoing, incoming = {u: [] for u in nodes}, {u: [] for u in nodes}
    weak_dep, weak_assoc = {u: set() for u in nodes}, {u: set() for u in nodes}
    relation_counts = Counter(row.get("type") for row in value["relations"])
    errors, dependency_records = [], []
    for relation in sorted(value["relations"], key=lambda row: row["id"]):
        source, target = relation.get("source"), relation.get("target")
        if source not in units or target not in units:
            errors.append({"relation_id": relation["id"], "code": "missing_unit_endpoint",
                           "source": source, "target": target})
            continue
        kind = relation.get("type")
        if kind not in {"statement_dependency", "proof_dependency", "proves", "identity"}:
            errors.append({"relation_id": relation["id"], "code": "unknown_relation_type", "type": kind})
        if kind in DEPENDENCIES:
            dependency_records.append(relation)
            outgoing[source].append((target, relation["id"]))
            incoming[target].append((source, relation["id"]))
            weak_dep[source].add(target)
            weak_dep[target].add(source)
        if kind in ASSOCIATIONS:
            weak_assoc[source].add(target)
            weak_assoc[target].add(source)
    for node in nodes:
        outgoing[node].sort()
        incoming[node].sort()
    degrees = {node: {"in": len(incoming[node]), "out": len(outgoing[node])} for node in nodes}
    heap = [node for node in nodes if not incoming[node]]
    heapq.heapify(heap)
    remaining_degree, order = {node: len(incoming[node]) for node in nodes}, []
    while heap:
        node = heapq.heappop(heap)
        order.append(node)
        for target, _ in outgoing[node]:
            remaining_degree[target] -= 1
            if remaining_degree[target] == 0:
                heapq.heappush(heap, target)
    components = _sccs(nodes, outgoing, incoming)
    membership = {node: i for i, members in enumerate(components) for node in members}
    internal_edges = defaultdict(list)
    for row in dependency_records:
        source_component, target_component = membership[row["source"]], membership[row["target"]]
        if source_component == target_component:
            internal_edges[source_component].append(row["id"])
    cyclic_components = []
    for i, members in enumerate(components):
        internal = sorted(internal_edges[i])
        if len(members) > 1 or internal:
            witness = _witness(members, outgoing)
            cyclic_components.append({"node_ids": members, "relation_ids": internal, "cycle_witness": witness})
    dag = len(order) == len(nodes) and not errors
    if dag:
        position = {node: i for i, node in enumerate(order)}
        if any(position[row["source"]] >= position[row["target"]] for row in dependency_records):
            raise TopologyError("internal error: invalid complete topological order")
    dep_isolated = sorted(node for node in nodes if not incoming[node] and not outgoing[node])
    completely_isolated = sorted(node for node in nodes if not weak_assoc[node])
    assoc_map = {node: [] for node in dep_isolated}
    for row in value["relations"]:
        if row.get("type") == "proves":
            for node in {row.get("source"), row.get("target")}:
                if node in assoc_map:
                    assoc_map[node].append(row["id"])
    assoc_map = {node: sorted(ids) for node, ids in assoc_map.items()}
    return {"schema_version": VERSION, "graph_binding": graph_binding(value),
            "node_count": len(nodes), "dependency_count": len(dependency_records),
            "relation_counts": dict(sorted(relation_counts.items(), key=lambda pair: str(pair[0]))),
            "dag": dag, "structural_errors": errors,
            "topological_order": order if dag else None, "partial_topological_order": order,
            "degrees": degrees, "roots": sorted(node for node in nodes if not incoming[node]),
            "terminals": sorted(node for node in nodes if not outgoing[node]),
            "sccs": components, "cyclic_sccs": cyclic_components,
            "cycle_witnesses": [row["cycle_witness"] for row in cyclic_components],
            "cycle_relation_ids": sorted(edge for row in cyclic_components for edge in row["relation_ids"]),
            "dependency_isolated": dep_isolated, "completely_isolated": completely_isolated,
            "proves_only_isolated": sorted(set(dep_isolated) - set(completely_isolated)),
            "isolated_proves_relations": assoc_map,
            "dependency_weak_components": _components(nodes, {k: sorted(v) for k, v in weak_dep.items()}),
            "association_weak_components": _components(nodes, {k: sorted(v) for k, v in weak_assoc.items()})}


def _environment(graph: dict, unit_ids: list[str]) -> dict:
    records = {row["id"]: row for key in ("documents", "contexts", "units") for row in graph[key]}
    closure, pending = {}, list(unit_ids)
    associations = {}
    while pending:
        record_id = pending.pop()
        if record_id in closure:
            continue
        if record_id not in records:
            raise TopologyError(f"signature references a missing mathematical record: {record_id}")
        row = records[record_id]
        if "category" in row:
            # Labels/classification metadata are not new mathematical authority.
            closure[record_id] = {key: copy.deepcopy(row.get(key)) for key in
                                  ("id", "category", "document", "context", "original_text", "sources", "semantics")}
            pending.extend([row["document"], row["context"]])
            if row["category"] == "proof":
                goals = [{key: relation.get(key) for key in ("type", "source", "target", "conditions")}
                         for relation in graph["relations"] if relation.get("type") == "proves"
                         and relation.get("source") == record_id]
                associations[record_id] = sorted(goals, key=digest)
                pending.extend(goal["target"] for goal in goals)
        elif "description" in row:
            closure[record_id] = copy.deepcopy(row)
            pending.append(row["document"])
            if row.get("parent"):
                pending.append(row["parent"])
        else:
            closure[record_id] = copy.deepcopy(row)
        pending.extend(source["document"] for source in row.get("sources", []))
    return {"participants_and_contexts": closure, "proof_targets": associations}


def relation_signature(graph: Graph | dict, relation: dict | Relation, revision_id: str) -> str:
    """Block re-adding a rejected/cropped claim with renamed ID or explanation.

    Signatures bind revision, endpoints, original text, Context ancestors,
    documents and proof goals, plus type.  Renaming ID, explanation or conditions
    does not by itself create new authority for a previously removed pair.
    An unrelated graph edit
    leaves this signature unchanged.  New explicit authority is separately
    logged by State.reopen_relation, never silently inferred from graph hashes.
    """
    value = _copy_graph(graph)
    row = relation.model_dump(mode="json") if isinstance(relation, Relation) else relation
    if not isinstance(row, dict) or not isinstance(revision_id, str) or not revision_id:
        raise TopologyError("relation signature requires a relation and revision identity")
    claim = {key: row.get(key) for key in ("type", "source", "target")}
    return digest({"schema_version": VERSION, "revision_id": revision_id, "claim": claim,
                   "environment": _environment(value, [row["source"], row["target"]])})


def _review_status(review: dict) -> str:
    status = review.get("status")
    if status not in REVIEW_STATUSES:
        raise TopologyError("independent review needs a recognized status")
    return status


def rank_cycle_edges(graph: Graph | dict, reviews: dict[str, dict]) -> list[dict]:
    """Predeclared weakest-first ranking of CURRENT cyclic dependency edges.

    Every cycle edge must be independently reviewed first.  Incorrect edges
    must go through mathematical deletion; technical failures block cropping.
    Model self-confidence and explicit/implicit mode do not determine order.
    """
    value, report = _copy_graph(graph), analyze(graph)
    records = {row["id"]: row for row in value["relations"]}
    rows = []
    for relation_id in report["cycle_relation_ids"]:
        review = reviews.get(relation_id)
        if not isinstance(review, dict):
            raise TopologyError(f"cycle edge was not independently reviewed: {relation_id}")
        status = _review_status(review)
        if status not in {"supported", "unresolved"}:
            raise TopologyError(f"cycle edge not eligible for structural cropping: {relation_id} ({status})")
        witness, scope = review.get("witness_quality", "unknown"), review.get("scope_quality", "unknown")
        if witness not in RANKING_POLICY["use_witness"] or scope not in RANKING_POLICY["scope"]:
            raise TopologyError("unknown evidence ranking factor")
        factors = {"independent_review": status, "use_witness": witness, "scope": scope}
        rank = [RANKING_POLICY["independent_review"][status],
                RANKING_POLICY["use_witness"][witness], RANKING_POLICY["scope"][scope], relation_id]
        rows.append({"relation_id": relation_id, "relation": copy.deepcopy(records[relation_id]),
                     "factors": factors, "rank": rank})
    return sorted(rows, key=lambda row: tuple(row["rank"]))


def _model(graph: dict) -> Graph:
    return Graph.model_validate_json(json.dumps(graph, ensure_ascii=False, allow_nan=False))


def _binding_model(graph: dict) -> Graph:
    """Allow deletion of a diagnosed self-loop, never normalize other faults.

    Formal E2 forbids self-relations.  Before-graph hash/evidence calculations
    omit only these already invalid relations; original records remain in the
    mutation record.  Any other schema defect is still a hard failure.
    """
    if any(row.get("source") == row.get("target") for row in graph["relations"]):
        value = copy.deepcopy(graph)
        value["relations"] = [row for row in value["relations"] if row.get("source") != row.get("target")]
        return _model(value)
    return _model(graph)


def _integrity_report(graph: dict):
    report = validate_graph(_binding_model(graph))
    for row in graph["relations"]:
        if row.get("source") == row.get("target"):
            report.errors.append({"record": row["id"], "reason": "formal E2 forbids a self-relation"})
    return report


def _claim_bindings(graph: dict) -> dict[str, dict[str, str]]:
    """Compute genuine E2 claim digests; accept diagnostic proof gaps only."""
    model = _binding_model(graph)
    index = GraphIndex(model)
    evidence = {row.id: row for row in model.evidence}
    return {row.id: {eid: index.claim_digest(row, evidence[eid].sources) for eid in row.evidence}
            for row in model.relations}


def _guard_proofs(before: dict, after: dict) -> None:
    units_before = {row["id"]: row for row in before["units"]}
    units_after = {row["id"]: row for row in after["units"]}
    for proof_id, proof in units_after.items():
        if proof.get("category") != "proof":
            continue
        def goals(graph, units):
            return {row["target"] for row in graph["relations"] if row.get("type") == "proves"
                    and row.get("source") == proof_id and row.get("target") in units
                    and units[row["target"]].get("category") == "statement"
                    and units[row["target"]].get("document") == proof.get("document")}
        if goals(before, units_before) and not goals(after, units_after):
            raise TopologyError(f"deletion would leave a formerly associated proof unowned: {proof_id}")


def _finish_change(before: dict, after: dict, change: dict, *, invalidated_evidence: str) -> tuple[dict, dict]:
    if invalidated_evidence not in {"reject", "detach"}:
        raise TopologyError("invalidated_evidence must be reject or explicit detach")
    _guard_proofs(before, after)
    # Orphan evidence is optional data but cannot remain attached to a stale claim.
    bindings_before, bindings_after = _claim_bindings(before), _claim_bindings(after)
    invalid = {rid: [eid for eid, claim in claims.items()
                     if bindings_before.get(rid, {}).get(eid) != claim]
               for rid, claims in bindings_after.items()}
    invalid = {rid: eids for rid, eids in invalid.items() if eids}
    if invalid and invalidated_evidence == "reject":
        raise EvidenceInvalidationError(sorted(invalid))
    stale_records = []
    if invalid:
        evidence = {row["id"]: row for row in before["evidence"]}
        for relation in after["relations"]:
            removed = set(invalid.get(relation["id"], []))
            if removed:
                stale_records.append({"relation_id": relation["id"],
                                      "evidence": [copy.deepcopy(evidence[eid]) for eid in sorted(removed)],
                                      "reason": "participant/environment/proof-goal changed; independent revalidation required"})
                relation["evidence"] = [eid for eid in relation.get("evidence", []) if eid not in removed]
    referenced = {eid for relation in after["relations"] for eid in relation.get("evidence", [])}
    pruned = [row for row in after["evidence"] if row["id"] not in referenced]
    after["evidence"] = [row for row in after["evidence"] if row["id"] in referenced]
    # Keep pre-existing diagnostic proof gaps, but reject newly introduced shape,
    # direction, dangling references, identity, or evidence-binding errors.
    before_errors = {(row.get("record"), row.get("reason")) for row in _integrity_report(before).errors}
    new_errors = [row for row in _integrity_report(after).errors
                  if (row.get("record"), row.get("reason")) not in before_errors]
    if new_errors:
        raise TopologyError("change creates E2 integrity errors: " + json.dumps(new_errors, ensure_ascii=False))
    change.update({"before": graph_binding(before), "after": graph_binding(after),
                   "invalidated_evidence": stale_records, "pruned_evidence": copy.deepcopy(pruned),
                   "formal_e2_report": _integrity_report(after).to_dict()})
    return after, change


def remove_relations(graph: Graph | dict, relation_ids: list[str], *, reason: str,
                     classification: str = "mathematical_error", snapshot_id: str | None = None,
                     invalidated_evidence: str = "reject") -> tuple[dict, dict]:
    """Remove claims without mutating originals; error and cropping stay distinct."""
    if classification not in REMOVAL_CLASSIFICATIONS or not reason.strip():
        raise TopologyError("relation deletion needs a reason and explicit classification")
    before = _copy_graph(graph)
    known = {row["id"]: row for row in before["relations"]}
    selected = set(relation_ids)
    if not selected or len(selected) != len(relation_ids) or not selected <= known.keys():
        raise TopologyError("deletion needs distinct known relation IDs")
    if classification == "structural_crop":
        cyclic = set(analyze(before)["cycle_relation_ids"])
        if len(selected) != 1 or not selected <= cyclic:
            raise TopologyError("a structural crop must remove one current cyclic dependency")
    after = copy.deepcopy(before)
    after["relations"] = [row for row in after["relations"] if row["id"] not in selected]
    if snapshot_id is not None:
        after["snapshot_id"] = snapshot_id
    dropped_views = [row for row in after["identity_views"]
                     if selected & set(row.get("justifications", []))]
    after["identity_views"] = [row for row in after["identity_views"] if row not in dropped_views]
    change = {"classification": classification, "reason": reason,
              "removed_relations": [copy.deepcopy(known[rid]) for rid in sorted(selected)],
              "removed_identity_views": copy.deepcopy(dropped_views),
              "loss": "may remove a true relationship to enforce DAG" if classification == "structural_crop" else None}
    return _finish_change(before, after, change, invalidated_evidence=invalidated_evidence)


def remove_units(graph: Graph | dict, unit_ids: list[str], *, reason: str,
                 snapshot_id: str | None = None, invalidated_evidence: str = "reject") -> tuple[dict, dict]:
    """Delete value-reviewed units and incident claims; reject orphan proof goals.

    Affected identity views are removed as whole derived views; no representative
    or justification is invented.  Context/document/source records are retained.
    Unaffected proof goals and evidence are checked with actual GraphIndex hashes.
    """
    if not isinstance(reason, str) or not reason.strip():
        raise TopologyError("node deletion requires a mathematical-value reason")
    before = _copy_graph(graph)
    known = {row["id"]: row for row in before["units"]}
    selected = set(unit_ids)
    if not selected or len(selected) != len(unit_ids) or not selected <= known.keys():
        raise TopologyError("deletion needs distinct known node IDs")
    after = copy.deepcopy(before)
    after["units"] = [row for row in after["units"] if row["id"] not in selected]
    removed_relations = [row for row in after["relations"] if selected & {row["source"], row["target"]}]
    after["relations"] = [row for row in after["relations"] if row not in removed_relations]
    removed_relation_ids = {row["id"] for row in removed_relations}
    removed_views = [row for row in after["identity_views"] if selected & set(row["members"])
                     or removed_relation_ids & set(row["justifications"])]
    after["identity_views"] = [row for row in after["identity_views"] if row not in removed_views]
    if snapshot_id is not None:
        after["snapshot_id"] = snapshot_id
    change = {"classification": "node_value_deletion", "reason": reason,
              "removed_units": [copy.deepcopy(known[uid]) for uid in sorted(selected)],
              "removed_relations": copy.deepcopy(removed_relations),
              "removed_identity_views": copy.deepcopy(removed_views)}
    return _finish_change(before, after, change, invalidated_evidence=invalidated_evidence)


def _local(path: str | Path) -> Path:
    result = Path(path).resolve()
    if not result.is_relative_to(ROOT) or result == ROOT:
        raise TopologyError("topology state writes must stay inside the project")
    return result


def _atomic(path: Path, value: dict) -> None:
    content = (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    fd, temporary = tempfile.mkstemp(prefix="." + path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class State:
    """One persistent state per document/campaign, independent of graph hashes.

    ``path`` MUST be reused after edits/restart/re-extraction.  Initial identity
    and policy bindings cannot be replaced.  Reserve a full re-extraction once;
    failure leaves the SAME reserved identity resumable and never permits crop.
    Isolation rounds survive graph/revision changes.  Ledger/model cache remain
    separate concerns; this state contains no secret or supplier interaction.
    """

    def __init__(self, path: str | Path, document_id: str, initial_revision: str,
                 graph: Graph | dict, binding: dict | None = None, *, max_node_rounds: int = 1,
                 max_document_rounds: int = 1):
        if not isinstance(document_id, str) or not document_id or not isinstance(initial_revision, str) or not initial_revision:
            raise TopologyError("state needs document and initial revision IDs")
        if type(max_node_rounds) is not int or not 1 <= max_node_rounds <= 3 or type(max_document_rounds) is not int or not 1 <= max_document_rounds <= 3:
            raise TopologyError("repair round limits must be positive and no greater than three")
        self.path = _local(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.document_id, self.initial_revision = document_id, initial_revision
        self.binding = copy.deepcopy(binding or {})
        self.limits = {"node_rounds": max_node_rounds, "document_rounds": max_document_rounds, "reextractions": 1}
        self.initial_graph = graph_binding(graph)
        with self._locked(create=True) as value:
            if not value:
                value.update({"schema_version": STATE_VERSION, "document_id": document_id,
                              "initial_revision": initial_revision, "initial_graph": self.initial_graph,
                              "policy_binding": self.binding, "limits": self.limits,
                              "ranking_policy": copy.deepcopy(RANKING_POLICY),
                              "revision_id": initial_revision, "graph_binding": self.initial_graph,
                              "document_rounds": 0, "node_rounds": {}, "node_aliases": {},
                              "reextract": None, "blocked_relations": {}, "reviews": {}, "events": []})
                self._event(value, "initialized", {"graph_binding": self.initial_graph})

    def _validate(self, value: dict) -> None:
        required = {"schema_version", "document_id", "initial_revision", "initial_graph", "policy_binding", "limits",
                    "ranking_policy", "revision_id", "graph_binding", "document_rounds", "node_rounds", "node_aliases",
                    "reextract", "blocked_relations", "reviews", "events", "state_sha256"}
        if not isinstance(value, dict) or set(value) != required:
            raise TopologyError("malformed topology state")
        sealed = {key: val for key, val in value.items() if key != "state_sha256"}
        if digest(sealed) != value["state_sha256"]:
            raise TopologyError("topology state checksum differs")
        if (value["schema_version"] != STATE_VERSION or value["document_id"] != self.document_id
                or value["initial_revision"] != self.initial_revision or value["initial_graph"] != self.initial_graph
                or value["policy_binding"] != self.binding or value["limits"] != self.limits
                or value["ranking_policy"] != RANKING_POLICY):
            raise TopologyError("state belongs to a different document, initial input or fixed policy")
        previous, rounds, node_rounds, reservations = None, 0, Counter(), []
        for index, event in enumerate(value["events"]):
            unsigned = {key: val for key, val in event.items() if key != "event_sha256"}
            if (event.get("index") != index or event.get("previous_event_sha256") != previous
                    or digest(unsigned) != event.get("event_sha256")):
                raise TopologyError("topology event chain differs")
            previous = event["event_sha256"]
            if event["kind"] == "isolation_round":
                rounds += 1
                node_rounds.update(event["details"]["counted_node_ids"])
            if event["kind"] == "reextract_reserved":
                reservations.append(event["details"]["revision_id"])
        if (type(value["document_rounds"]) is not int or value["document_rounds"] != rounds
                or rounds > self.limits["document_rounds"] or dict(node_rounds) != value["node_rounds"]
                or any(count > self.limits["node_rounds"] for count in node_rounds.values())):
            raise TopologyError("isolation counters were reset or exceed limits")
        if len(reservations) > 1:
            raise TopologyError("more than one full re-extraction identity")
        reextract = value["reextract"]
        if bool(reservations) != (reextract is not None):
            raise TopologyError("re-extraction state and events disagree")
        if reextract is not None:
            if (reextract.get("revision_id") != reservations[0] or reextract.get("status") not in
                    {"reserved", "failed", "completed"} or type(reextract.get("attempts")) is not int
                    or reextract["attempts"] < 0):
                raise TopologyError("invalid re-extraction status")

    @contextmanager
    def _locked(self, *, create: bool = False):
        descriptor = os.open(self.path.with_suffix(self.path.suffix + ".lock"), os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
        with os.fdopen(descriptor, "a+b") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            _local(self.path)
            if self.path.exists():
                value = json.loads(self.path.read_text(encoding="utf-8"))
                self._validate(value)
            elif create:
                value = {}
            else:
                raise TopologyError("topology state disappeared; refusing to reset repair progress")
            yield value
            value.pop("state_sha256", None)
            value["state_sha256"] = digest(value)
            self._validate(value)
            _atomic(self.path, value)

    @staticmethod
    def _event(value: dict, kind: str, details: dict) -> dict:
        event = {"index": len(value["events"]), "kind": kind, "time": _now(),
                 "revision_id": value["revision_id"], "graph_binding": copy.deepcopy(value["graph_binding"]),
                 "previous_event_sha256": value["events"][-1]["event_sha256"] if value["events"] else None,
                 "details": copy.deepcopy(details)}
        event["event_sha256"] = digest(event)
        value["events"].append(event)
        return event

    def snapshot(self) -> dict:
        with self._locked() as value:
            return copy.deepcopy(value)

    def import_policy_history(self, graph: Graph | dict, historical_state: dict, *, source_binding: dict) -> dict:
        """Migrate a saved diagnostic graph without buying another repair pass.

        The old state stays immutable.  Its actual counts remain in the import
        event; a completed old expansion consumes the new policy's single pass.
        Content-bound prohibitions are re-signed for the new revision so a
        policy change cannot restore an already rejected relationship.
        """
        if (not isinstance(source_binding, dict) or set(source_binding) != {"path", "sha256"}
                or not isinstance(historical_state, dict)):
            raise TopologyError("history migration requires a bound saved state")
        source = _local(source_binding["path"])
        if source == self.path:
            raise TopologyError("historical state must remain separate from the new policy state")
        content = source.read_bytes()
        if hashlib.sha256(content).hexdigest() != source_binding["sha256"] or json.loads(content) != historical_state:
            raise TopologyError("saved topology history binding differs")
        legacy = object.__new__(State)
        legacy.document_id = historical_state.get("document_id")
        legacy.initial_revision = historical_state.get("initial_revision")
        legacy.initial_graph = historical_state.get("initial_graph")
        legacy.binding = historical_state.get("policy_binding")
        legacy.limits = historical_state.get("limits")
        legacy._validate(historical_state)
        if historical_state["document_id"] != self.document_id:
            raise TopologyError("cannot import topology history from another document")
        reserved = {e["index"] for e in historical_state["events"] if e["kind"] == "isolation_round"}
        complete = {e["details"]["round_index"] for e in historical_state["events"]
                    if e["kind"] == "isolation_round_complete"}
        if reserved != complete or historical_state["reextract"] is not None:
            raise TopologyError("pending work or a reserved re-extraction needs explicit continuation, not policy migration")
        with self._locked() as value:
            prior = [e for e in value["events"] if e["kind"] == "policy_history_imported"]
            if prior:
                if len(prior) != 1 or prior[0]["details"]["source_binding"] != source_binding:
                    raise TopologyError("cannot replace imported repair history")
                return copy.deepcopy(prior[0]["details"])
            self._current(value, graph)
            if value["events"][-1]["kind"] != "initialized" or historical_state["graph_binding"] != graph_binding(graph):
                raise TopologyError("migration must start from the exact saved diagnostic graph")
            imported = []
            for old_signature, row in historical_state["blocked_relations"].items():
                if not row.get("active"):
                    continue
                if relation_signature(graph, row["relation"], historical_state["revision_id"]) != old_signature:
                    raise TopologyError("historical blocked relationship has stale content")
                signature = relation_signature(graph, row["relation"], value["revision_id"])
                saved = copy.deepcopy(row)
                saved.update(signature=signature, revision_id=value["revision_id"],
                             historical_revision_id=row["revision_id"], history_source=copy.deepcopy(source_binding))
                value["blocked_relations"][signature] = saved
                imported.append(signature)
            consumed = historical_state["document_rounds"] > 0
            if consumed:
                current_nodes = {u["id"] for u in _copy_graph(graph)["units"]}
                counted = sorted(u for u, n in historical_state["node_rounds"].items() if n and u in current_nodes)
                value["document_rounds"] = 1
                value["node_rounds"] = dict.fromkeys(counted, 1)
                event = self._event(value, "isolation_round", {"node_ids": counted, "counted_node_ids": counted,
                    "document_round": 1, "reason": "Previously completed expansion consumes the single production pass",
                    "history_source": source_binding, "migration": True})
                self._event(value, "isolation_round_complete", {"round_index": event["index"],
                    "diagnostics": analyze(graph), "migration": True, "history_source": source_binding})
            details = {"source_binding": copy.deepcopy(source_binding),
                       "historical_state_sha256": historical_state["state_sha256"],
                       "historical_document_rounds": historical_state["document_rounds"],
                       "historical_node_rounds": copy.deepcopy(historical_state["node_rounds"]),
                       "single_pass_consumed": consumed, "imported_blocked_signatures": imported}
            self._event(value, "policy_history_imported", details)
            return copy.deepcopy(details)

    @staticmethod
    def _current(value: dict, graph: Graph | dict) -> None:
        if graph_binding(graph) != value["graph_binding"]:
            raise TopologyError("operation is bound to a stale full graph snapshot")

    def bind_graph(self, graph: Graph | dict, *, expected_before_sha256: str | None = None,
                   reason: str = "versioned graph change") -> dict:
        with self._locked() as value:
            if expected_before_sha256 is not None and value["graph_binding"]["graph_sha256"] != expected_before_sha256:
                raise TopologyError("graph changed before repair commit")
            new = graph_binding(graph)
            if new == value["graph_binding"]:
                return copy.deepcopy(value["graph_binding"])
            old = copy.deepcopy(value["graph_binding"])
            value["graph_binding"] = new
            self._event(value, "graph_bound", {"before": old, "after": new, "reason": reason})
            return copy.deepcopy(new)

    def reserve_reextract(self, graph: Graph | dict, *, reviews: dict[str, dict], reason: str) -> dict:
        """Only after all current cycle edges reviewed; reserve the single identity."""
        with self._locked() as value:
            self._current(value, graph)
            if value["reextract"] is not None:
                return copy.deepcopy(value["reextract"])
            if not analyze(graph)["cycle_relation_ids"]:
                raise TopologyError("a full re-extraction requires a persisting real cycle")
            rank_cycle_edges(graph, reviews)
            if not reason.strip():
                raise TopologyError("re-extraction requires a saved diagnosis")
            revision = value["initial_revision"] + ":full-reextract-01"
            value["reextract"] = {"revision_id": revision, "status": "reserved", "attempts": 0,
                                   "from_graph": copy.deepcopy(value["graph_binding"]), "completed_graph": None}
            self._event(value, "reextract_reserved", {"revision_id": revision, "reason": reason,
                                                       "cycle_report": analyze(graph), "reviews": reviews})
            return copy.deepcopy(value["reextract"])

    def mark_reextract_attempt(self, revision_id: str) -> dict:
        with self._locked() as value:
            row = value["reextract"]
            if row is None or row["revision_id"] != revision_id or row["status"] == "completed":
                raise TopologyError("no incomplete reserved re-extraction with this identity")
            row["attempts"] += 1
            self._event(value, "reextract_attempt", {"revision_id": revision_id, "attempt": row["attempts"]})
            return copy.deepcopy(row)

    def mark_reextract_failed(self, revision_id: str, *, reason: str) -> dict:
        with self._locked() as value:
            row = value["reextract"]
            if row is None or row["revision_id"] != revision_id or row["status"] == "completed":
                raise TopologyError("cannot fail an unreserved or completed re-extraction")
            row["status"] = "failed"
            self._event(value, "reextract_failed", {"revision_id": revision_id, "reason": reason})
            return copy.deepcopy(row)

    def mark_reextract_completed(self, revision_id: str, graph: Graph | dict, *, node_mapping: dict[str, str] | None = None,
                                 completion_evidence: dict) -> dict:
        """Record completion of the fixed extraction stages, not a precision score.

        Evidence must identify all fixed stages as completed and immutable output
        hashes.  No host claim can turn failed supplier work into complete input.
        """
        stages = {"nodes", "contexts", "explicit_dependencies", "implicit_dependencies"}
        if (not isinstance(completion_evidence, dict) or completion_evidence.get("technical_complete") is not True
                or set(completion_evidence.get("completed_stages", [])) != stages
                or not completion_evidence.get("artifact_bindings")):
            raise TopologyError("full re-extraction lacks complete fixed-stage evidence")
        mapping = copy.deepcopy(node_mapping or {})
        with self._locked() as value:
            row = value["reextract"]
            if row is None or row["revision_id"] != revision_id:
                raise TopologyError("wrong full re-extraction identity")
            new = graph_binding(graph)
            if row["status"] == "completed":
                if row["completed_graph"] != new:
                    raise TopologyError("cannot replace completed full re-extraction output")
                return copy.deepcopy(row)
            new_units = {unit["id"] for unit in _copy_graph(graph)["units"]}
            if len(set(mapping.values())) != len(mapping) or not set(mapping.values()) <= new_units:
                raise TopologyError("node mapping must be injective and refer to new nodes")
            for old, new_id in mapping.items():
                canonical = value["node_aliases"].get(old, old)
                if new_id in value["node_aliases"] and value["node_aliases"][new_id] != canonical:
                    raise TopologyError("node mapping would reset or conflate repair counters")
                value["node_aliases"][new_id] = canonical
            row.update({"status": "completed", "completed_graph": new, "completion_evidence": copy.deepcopy(completion_evidence),
                        "node_mapping": mapping})
            value["revision_id"], value["graph_binding"] = revision_id, new
            self._event(value, "reextract_completed", {"revision_id": revision_id, "graph_binding": new,
                                                       "completion_evidence": completion_evidence, "node_mapping": mapping})
            return copy.deepcopy(row)

    def begin_isolation_round(self, graph: Graph | dict, node_ids: list[str], *, reason: str) -> dict:
        """Count BEFORE execution so a crash cannot permit unlimited new rounds.

        The saved event/ordinal is the resumable work identity.  Re-entry should
        resume that event, not invoke this method again to obtain a free round.
        """
        units = {row["id"] for row in _copy_graph(graph)["units"]}
        if not node_ids or len(set(node_ids)) != len(node_ids) or not set(node_ids) <= units:
            raise TopologyError("isolation round needs distinct current nodes")
        with self._locked() as value:
            self._current(value, graph)
            counted = sorted({value["node_aliases"].get(uid, uid) for uid in node_ids})
            if value["document_rounds"] >= value["limits"]["document_rounds"]:
                raise TopologyError("document isolation repair round limit exhausted")
            if any(value["node_rounds"].get(uid, 0) >= value["limits"]["node_rounds"] for uid in counted):
                raise TopologyError("node isolation expansion round limit exhausted")
            value["document_rounds"] += 1
            for uid in counted:
                value["node_rounds"][uid] = value["node_rounds"].get(uid, 0) + 1
            event = self._event(value, "isolation_round", {"node_ids": sorted(node_ids), "counted_node_ids": counted,
                                                          "document_round": value["document_rounds"], "reason": reason})
            return copy.deepcopy(event)

    def record_review(self, graph: Graph | dict, relation: dict | Relation, review: dict) -> dict:
        _review_status(review)
        with self._locked() as value:
            self._current(value, graph)
            signature = relation_signature(graph, relation, value["revision_id"])
            record = {"signature": signature, "graph_binding": graph_binding(graph), "review": copy.deepcopy(review)}
            value["reviews"][signature] = record
            self._event(value, "independent_review", record)
            return copy.deepcopy(record)

    def record_relation_removal(self, before: Graph | dict, relation: dict | Relation, *, classification: str,
                                reason: str, review: dict | None = None, ranking: list[dict] | None = None) -> dict:
        """Persist the prohibition BEFORE removing a claim from the graph."""
        if classification not in REMOVAL_CLASSIFICATIONS or not reason.strip():
            raise TopologyError("invalid removal classification or empty reason")
        if classification != "structural_crop" and (not isinstance(review, dict) or _review_status(review) != "incorrect"):
            raise TopologyError("error deletion requires an independently supported invalid judgment")
        if classification == "e2_integrity_error" and (not review.get("error_kind") or not review.get("integrity_errors")):
            raise TopologyError("E2 integrity deletion requires a specific validator rule and error record")
        with self._locked() as value:
            self._current(value, before)
            row = relation.model_dump(mode="json") if isinstance(relation, Relation) else copy.deepcopy(relation)
            signature = relation_signature(before, row, value["revision_id"])
            if classification == "structural_crop":
                reextract = value["reextract"]
                if reextract is None or reextract["status"] != "completed" or value["revision_id"] != reextract["revision_id"]:
                    raise TopologyError("structural cropping requires successfully completed single re-extraction")
                current_reviews = {}
                for candidate in _copy_graph(before)["relations"]:
                    if candidate["id"] in analyze(before)["cycle_relation_ids"]:
                        saved = value["reviews"].get(relation_signature(before, candidate, value["revision_id"]))
                        if saved is not None:
                            current_reviews[candidate["id"]] = saved["review"]
                expected_ranking = rank_cycle_edges(before, current_reviews)
                if ranking != expected_ranking or not ranking or ranking[0]["relation_id"] != row["id"]:
                    raise TopologyError("crop must follow saved current-cycle evidence ranking")
            record = {"signature": signature, "revision_id": value["revision_id"], "relation": row,
                      "classification": classification, "reason": reason, "review": copy.deepcopy(review),
                      "ranking": copy.deepcopy(ranking), "active": True,
                      "loss": "potential true dependency omitted to enforce DAG" if classification == "structural_crop" else None}
            value["blocked_relations"][signature] = record
            self._event(value, "relation_removed", record)
            return copy.deepcopy(record)

    def blocked(self, graph: Graph | dict, relation: dict | Relation) -> bool:
        with self._locked() as value:
            self._current(value, graph)
            return self._blocked_claim(value, graph, relation)

    @staticmethod
    def _blocked_claim(value: dict, graph: Graph | dict, relation: dict | Relation) -> bool:
        # Complete re-extraction changes the revision identity, not authority
        # for an unchanged rejected claim.  Check the current claim using each
        # saved removal's revision and its exact endpoint/Context environment.
        return any(row.get("active") is True
                   and relation_signature(graph, relation, row["revision_id"]) == row["signature"]
                   for row in value["blocked_relations"].values())

    def blocked_in_graph(self, graph: Graph | dict, relation: dict | Relation) -> bool:
        """Check a new re-extraction snapshot against content-bound removals.

        This reads the same persistent prohibitions without requiring the new
        graph to have been committed as the current topology revision yet.
        """
        with self._locked() as value:
            return self._blocked_claim(value, graph, relation)

    def reopen_relation(self, graph: Graph | dict, relation: dict | Relation, *, new_evidence: dict, reason: str) -> dict:
        """Exceptional explicit authority only, within existing remaining rounds."""
        if not new_evidence or not reason.strip():
            raise TopologyError("reopening requires separately saved new evidence and reason")
        with self._locked() as value:
            self._current(value, graph)
            signature = relation_signature(graph, relation, value["revision_id"])
            row = value["blocked_relations"].get(signature)
            if row is None or not row.get("active"):
                raise TopologyError("no active rejected/cropped claim to reopen")
            if value["document_rounds"] >= value["limits"]["document_rounds"]:
                raise TopologyError("new evidence cannot bypass exhausted document repair rounds")
            row["active"] = False
            event = self._event(value, "relation_reopened", {"signature": signature, "new_evidence": new_evidence, "reason": reason,
                                                            "requires_independent_review_and_dag_recheck": True})
            return copy.deepcopy(event)

    def record_event(self, graph: Graph | dict, kind: str, details: dict) -> dict:
        if kind in {"initialized", "isolation_round", "reextract_reserved", "reextract_completed", "reextract_attempt",
                    "reextract_failed", "relation_removed", "relation_reopened", "independent_review", "graph_bound",
                    "policy_history_imported"}:
            raise TopologyError("use the dedicated guarded transition for reserved event kinds")
        with self._locked() as value:
            self._current(value, graph)
            return copy.deepcopy(self._event(value, kind, details))
