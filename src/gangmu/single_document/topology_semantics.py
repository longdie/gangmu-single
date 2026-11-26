"""Saved-input dependency acceptance for topology-local repairs.

Unlike the frozen implicit-only experiment, a local repair can discover an
explicit or implicit direct use.  Both require identical saved-fragment use
witnesses.  Supplier verdicts remain separate from deterministic host gates;
these gates neither certify mathematics nor rewrite historical evaluations.
No original paper, producer reason or previous evaluation is read here.
"""
from __future__ import annotations

from collections import Counter
import copy
import json

from gangmu.representation.graph import GraphIndex
from gangmu.representation.model import Graph, Relation, Source, digest
from . import implicit_dependencies as implicit

VERSION = "gangmu.e4.topology_dependency_semantics.v1"
FIELDS = frozenset({"candidate_id", "verdict", "explicitness", "source_quote", "target_quote",
                    "scope_explanation", "reason"})
HOST_FIELDS = frozenset({"model_verdict", "host_validation_errors", "host_e2_errors"})
DEPENDENCIES = frozenset({"statement_dependency", "proof_dependency"})


def _pairs(pairs):
    if not isinstance(pairs, list) or any(not isinstance(row, dict) for row in pairs):
        raise ValueError("dependency candidates must be an array of objects")
    identifiers, claims = {}, set()
    for pair in pairs:
        if any(not isinstance(pair.get(key), str) or not pair[key] for key in ("candidate_id", "source", "target", "type")):
            raise ValueError("candidate identity, participants and type must be nonempty strings")
        candidate_id = pair["candidate_id"]
        if candidate_id in identifiers:
            raise ValueError("duplicate candidate IDs in dependency material")
        claim = (pair["source"], pair["target"], pair["type"])
        if claim in claims:
            raise ValueError("duplicate participant/type claims under different candidate IDs")
        identifiers[candidate_id] = pair
        claims.add(claim)
    return identifiers


def host_e2_errors(paper, pair):
    """Deterministic type/endpoint constraints, independent of model opinions.

    A proof-to-statement formulation dependency violates formal E2 even if a
    model describes a valid proof justification.  This is an output-model rule,
    not a declaration that the paper's mathematical argument is false.
    """
    if not isinstance(pair, dict):
        raise ValueError("dependency pair must be an object")
    source, target, kind = pair.get("source"), pair.get("target"), pair.get("type")
    errors = []
    for role, identifier in (("source", source), ("target", target)):
        if not isinstance(identifier, str) or identifier not in paper.units:
            errors.append({"code": "missing_mathematical_endpoint", "role": role,
                           "endpoint": identifier, "reason": "dependency endpoint is not a retained mathematical unit"})
    if not isinstance(kind, str) or kind not in DEPENDENCIES:
        errors.append({"code": "nondependency_type", "type": kind,
                       "reason": "local dependency repair cannot invent proves or identity relationships"})
    if source == target:
        errors.append({"code": "self_dependency", "source": source, "target": target,
                       "reason": "formal E2 requires distinct relation participants"})
    if not isinstance(source, str) or not isinstance(target, str) or source not in paper.units or target not in paper.units:
        return errors
    source_unit, target_unit = paper.units[source], paper.units[target]
    expected = "proof_dependency" if target_unit["category"] == "proof" else "statement_dependency"
    if isinstance(kind, str) and kind in DEPENDENCIES and kind != expected:
        errors.append({"code": "dependency_target_category_mismatch", "type": kind, "expected_type": expected,
                       "target_category": target_unit["category"],
                       "reason": "dependency type is determined by whether the actual user is a proof"})
    if kind == "statement_dependency" and source_unit["category"] == "proof":
        errors.append({"code": "proof_source_in_statement_dependency", "source_category": "proof", "type": kind,
                       "reason": "a proof cannot be a statement interpretation prerequisite under formal E2"})
    if source_unit.get("document") != target_unit.get("document"):
        errors.append({"code": "cross_document_local_dependency", "source_document": source_unit.get("document"),
                       "target_document": target_unit.get("document"),
                       "reason": "this finalizer handles one document; cross-document dependencies require a separate policy"})
    goals = set(getattr(paper, "saved_proof_goals", set())) | set(getattr(paper, "proves_pairs", set()))
    if (source, target) in goals or (target, source) in goals:
        errors.append({"code": "own_proof_goal_as_prerequisite", "source": source, "target": target,
                       "reason": "a proof's own asserted goal/target association is not a dependency premise"})
    return errors


def make_packet(paper, pairs, evaluation=False):
    """Use the frozen saved-node packet, admitting both pointer classifications.

    No retrieval scores, accepted witness selections, producer reasons or old
    judgments are introduced.  Literal reference metadata remain syntactic
    source evidence rather than an endorsement of the proposed relation.
    """
    _pairs(pairs)
    payload = implicit.make_packet(paper, pairs, evaluation)
    payload["schema_version"] = VERSION + ".saved-input"
    payload["context_policy"] = (
        "Saved Contexts help scope only; actual use must be witnessed in target original_text. "
        "Local repair permits explicit and implicit direct dependencies. A uniquely resolvable saved "
        "pointer implies explicit classification; otherwise actual use is implicit. Unknown classification "
        "cannot be accepted. Proof-goal associations are not premise uses.")
    # The reused schema already contains both explicitness categories.
    return payload


def validate_decisions(paper, payload, response, evaluation=False):
    """Normalize a complete population without replacing supplier evidence.

    Positive decisions failing host gates become unresolved.  ``model_verdict``
    always preserves the supplier verdict, and specific E2 rule failures are
    reported independently of source-quote/classification gates.
    """
    if not isinstance(payload, dict):
        raise ValueError("decision material must be a packet object")
    pairs = _pairs(payload.get("candidates"))
    if (not isinstance(response, dict) or set(response) != {"decisions"}
            or not isinstance(response["decisions"], list)
            or any(not isinstance(row, dict) for row in response["decisions"])):
        raise ValueError("invalid complete dependency decision response")
    if any(not isinstance(row.get("candidate_id"), str) for row in response["decisions"]):
        raise ValueError("individual decisions need a string candidate identity")
    if Counter(row.get("candidate_id") for row in response["decisions"]) != Counter(pairs.keys()):
        raise ValueError("dependency decisions must cover every planned candidate exactly once")
    verdicts = {"supported", "contradicted", "unresolved"} if evaluation else {"accept", "reject", "unresolved"}
    raw_by_id = {row["candidate_id"]: row for row in response["decisions"]}
    normalized = []
    for candidate_id, pair in pairs.items():
        raw = raw_by_id[candidate_id]
        if (set(raw) != FIELDS or any(not isinstance(value, str) for value in raw.values())
                or raw["verdict"] not in verdicts or raw["explicitness"] not in {"explicit", "implicit", "unknown"}):
            raise ValueError("malformed individual dependency decision")
        if pair["source"] not in paper.units or pair["target"] not in paper.units:
            raise ValueError("model decision refers to an unavailable saved endpoint")
        checked = copy.deepcopy(raw)
        checked["model_verdict"] = raw["verdict"]
        e2_errors = host_e2_errors(paper, pair)
        errors = []
        positive = raw["verdict"] == ("supported" if evaluation else "accept")
        if positive:
            for key, endpoint in (("source_quote", pair["source"]), ("target_quote", pair["target"])):
                quote = raw[key]
                if not quote.strip() or not any(quote in fragment for fragment in paper.units[endpoint]["original_text"]):
                    errors.append({"code": "positive_quote_not_bound_to_one_saved_original_fragment", "field": key,
                                   "endpoint": endpoint})
            pointer = paper.literal_explicit(pair["source"], pair["target"])
            expected = "explicit" if pointer else "implicit"
            if raw["explicitness"] == "unknown" or raw["explicitness"] != expected:
                errors.append({"code": "positive_source_classification_mismatch", "model_classification": raw["explicitness"],
                               "expected_classification": expected, "saved_pointer": pointer})
            if not raw["scope_explanation"].strip() or not raw["reason"].strip():
                errors.append({"code": "positive_missing_concrete_scope_or_reason"})
            if errors or e2_errors:
                checked["verdict"] = "unresolved"
        checked["host_validation_errors"], checked["host_e2_errors"] = errors, e2_errors
        normalized.append(checked)
    return normalized


def accept_relations(paper, pairs, decisions):
    """Create new, correctly bound E2 dependencies only; never a proves edge.

    All planned decisions stay in the population.  Any accepted claim must pass
    the saved-input gates again; changing host fields cannot override them.
    IDs live in a topology namespace and old accepted claims cannot duplicate.
    """
    by_id = _pairs(pairs)
    if not isinstance(decisions, list) or any(not isinstance(row, dict) for row in decisions):
        raise ValueError("accepted decisions must be objects")
    if any(not isinstance(row.get("candidate_id"), str) for row in decisions):
        raise ValueError("accepted decisions need string candidate identities")
    if Counter(row.get("candidate_id") for row in decisions) != Counter(by_id.keys()):
        raise ValueError("accepted decision population must be complete and unique")
    raw_decisions = []
    for row in decisions:
        if not FIELDS <= set(row) or set(row) - FIELDS - HOST_FIELDS:
            raise ValueError("unexpected or missing accepted-decision fields")
        raw = {key: copy.deepcopy(row[key]) for key in FIELDS}
        raw["verdict"] = row.get("model_verdict", row["verdict"])
        raw_decisions.append(raw)
    normalized = validate_decisions(paper, {"candidates": pairs}, {"decisions": raw_decisions})
    current_ids = {row["candidate_id"]: row for row in decisions}
    existing_claims = {(row["source"], row["target"], row["type"]) for row in paper.graph["relations"]}
    existing_ids = {row["id"] for key in ("documents", "contexts", "units", "relations", "evidence", "identity_views")
                    for row in paper.graph.get(key, [])}
    graph = Graph.model_validate_json(json.dumps(paper.graph, ensure_ascii=False, allow_nan=False))
    index = GraphIndex(graph)
    relations, evidence, witnesses = [], [], []
    for decision in normalized:
        supplied = current_ids[decision["candidate_id"]]
        if supplied["verdict"] != decision["verdict"]:
            raise ValueError("accepted host decision disagrees with recomputed source/type gates")
        if decision["verdict"] != "accept":
            continue
        pair = by_id[decision["candidate_id"]]
        claim = (pair["source"], pair["target"], pair["type"])
        if claim in existing_claims:
            raise ValueError("local repair duplicates a retained dependency")
        origin = decision["explicitness"]
        relation_id = "relation:topology:" + digest([paper.paper_id, graph.snapshot_id, *claim, origin])[:24]
        evidence_id = "evidence:topology:" + digest([relation_id, decision])[:24]
        if relation_id in existing_ids or evidence_id in existing_ids:
            raise ValueError("topology record ID collides with a retained record")
        relation = {"id": relation_id, "type": pair["type"], "source": pair["source"], "target": pair["target"],
                    "explanation": decision["reason"], "conditions": decision["scope_explanation"], "evidence": [evidence_id]}
        sources = copy.deepcopy(paper.units[pair["target"]]["sources"])
        model_relation = Relation.model_validate_json(json.dumps(relation, ensure_ascii=False, allow_nan=False))
        source_models = tuple(Source.model_validate_json(json.dumps(row)) for row in sources)
        claim_sha256 = index.claim_digest(model_relation, source_models)
        relations.append(relation)
        evidence.append({"id": evidence_id, "claim_sha256": claim_sha256, "method": "agent_analysis", "sources": sources,
                         "explanation": decision["reason"], "certificate": None})
        witnesses.append({"candidate_id": pair["candidate_id"], "relation_id": relation_id,
                          "source_quote": decision["source_quote"], "target_quote": decision["target_quote"],
                          "scope_explanation": decision["scope_explanation"], "explicitness": origin,
                          "saved_pointer": paper.literal_explicit(pair["source"], pair["target"]),
                          "model_verdict": decision["model_verdict"],
                          "host_validation_errors": [], "host_e2_errors": []})
        existing_claims.add(claim)
        existing_ids.update({relation_id, evidence_id})
    return relations, evidence, witnesses


# Explicit name matches the frozen implicit component for adapter convenience.
accepted_relations = accept_relations
