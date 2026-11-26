"""Implicit-only experiments over byte-bound saved graphs, never SourceProject.

Retrieval provides candidates. Witness binding is a structural check, not a
mathematical certificate. Independent scores retain the entire accepted output.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import copy
import hashlib
import json
import math
from pathlib import Path
import re

from gangmu.representation.model import Graph, Relation, Source, digest
from gangmu.representation.graph import GraphIndex
from .candidates import _features, _construction_features
from .retrieval import BM25Index, tokenize, term_weight

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.e4.implicit_dependencies.v1"
DEPENDENCIES = {"statement_dependency", "proof_dependency"}


def local(path, root=ROOT):
    path = Path(path)
    if not path.is_absolute():
        path = root / path
    if any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Implicit experiment paths cannot contain symbolic links")
    path = path.resolve()
    if not path.is_relative_to(root.resolve()) or path == root.resolve():
        raise ValueError("Implicit experiment paths must stay inside the project")
    return path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def record(path, root=ROOT):
    path = local(path, root)
    return {"path": path.relative_to(root).as_posix(), "sha256": sha(path.read_bytes())}


def write_once(path, value, root=ROOT):
    path = local(path, root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("Immutable implicit output already has different contents")
        return record(path, root)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    return record(path, root)


def load_config(path, root=ROOT):
    value = json.loads(local(path, root).read_bytes())
    if value.get("schema_version") != "gangmu.e4.implicit_campaign.v1":
        raise ValueError("Unknown implicit campaign")
    if (value.get("node_reextraction") is not False or value.get("raw_paper_reading") is not False
            or value.get("topology_acceptance") is not False or value.get("recall_acceptance") is not False):
        raise ValueError("Implicit experiment scope must preserve frozen inputs")
    if len(set(value["papers"])) < 3 or len(value["papers"]) != len(set(value["papers"])):
        raise ValueError("At least three distinct, preregistered papers required")
    if value["acceptance"]["precision_strictly_greater_than"] != 0.85:
        raise ValueError("Do not silently change the user precision threshold")
    return value


class SavedPaper:
    """Read only a manifest and its saved graph/node JSON objects.

    No source tree, source parser, or previous semantic evaluation is available
    through this adapter. Reference records are syntactic saved metadata only.
    """

    def __init__(self, config, paper_id, root=ROOT):
        self.root = root.resolve()
        manifest = local(config["baseline_manifest"], self.root)
        raw = manifest.read_bytes()
        if sha(raw) != config["baseline_manifest_sha256"]:
            raise ValueError("Frozen baseline manifest differs")
        self.manifest = json.loads(raw)
        self.files = {r["origin_path"]: r for r in self.manifest["files"]}
        self.paper = next(p for p in self.manifest["papers"] if p["paper_id"] == paper_id)
        self.allowed = {self.paper["production_result"]["path"], self.paper["node_result"]["path"]}
        self.read_paths = []
        production = self.bound_json(self.paper["production_result"])
        nodes = self.bound_json(self.paper["node_result"])
        self.graph = production["graph"]
        for key in ("documents", "units", "contexts"):
            if self.graph[key] != nodes[key]:
                raise ValueError("Saved graph differs from frozen nodes")
        self.baseline_binding = record(manifest, self.root)
        self.production_binding = self.paper["production_result"]
        self.reference_ledger = production["reference_ledger"]
        self.units = {u["id"]: u for u in self.graph["units"]}
        self.contexts = {c["id"]: c for c in self.graph["contexts"]}
        if len(self.units) != len(self.graph["units"]) or len(self.contexts) != len(self.graph["contexts"]):
            raise ValueError("Duplicate frozen node/context IDs")
        self.paper_id = paper_id
        self.order = sorted(self.units, key=lambda identifier: self.position(self.units[identifier]))
        self.indices = {identifier: i for i, identifier in enumerate(self.order)}
        self.explicit_pairs = {(r["source"], r["target"], r["type"])
                               for r in self.graph["relations"] if r["type"] in DEPENDENCIES}
        self.proves_pairs = {(r["source"], r["target"]) for r in self.graph["relations"] if r["type"] == "proves"}
        headings = defaultdict(set)
        for identifier, unit in self.units.items():
            if unit["category"] != "proof":
                for kind, number in self.manual_heading(unit):
                    headings[(kind, number)].add(identifier)
        self.named_headings = {key: next(iter(ids)) for key, ids in headings.items() if len(ids) == 1}
        self.saved_proof_goals = set(self.proves_pairs)
        for identifier, unit in self.units.items():
            if unit["category"] != "proof":
                continue
            first_line = self.body(unit).splitlines()[0]
            for (kind, number), target in self.named_headings.items():
                if re.search(r"\bproof\s+of\s+(?:the\s+)?" + self.numbered_pattern(kind, number), first_line, re.I):
                    self.saved_proof_goals.add((identifier, target))

    def bound_json(self, binding):
        name = binding["path"]
        if name not in self.allowed:
            raise ValueError("Adapter may only read saved graph and node JSON")
        row = self.files[name]
        raw = local(row["snapshot_path"], self.root).read_bytes()
        if row["sha256"] != binding["sha256"] or sha(raw) != row["sha256"] or len(raw) != row["bytes"]:
            raise ValueError("Frozen saved input byte binding differs")
        self.read_paths.append(row["snapshot_path"])
        return json.loads(raw)

    @staticmethod
    def body(unit):
        return "\n".join(unit["original_text"])

    @staticmethod
    def position(unit):
        return min((s["file"], s.get("occurrence") or "", s["char_start"], s["char_end"], unit["id"])
                   for s in unit["sources"])

    @staticmethod
    def manual_heading(unit):
        # Literal heading in saved first line only: no inferred TeX numbering.
        first = SavedPaper.body(unit).splitlines()[0]
        if any(s['file'].lower().endswith(('.md', '.markdown')) for s in unit['sources']):
            first = re.sub(r'^[#* _>]+', '', first)
        prefix = re.sub(r"\\[A-Za-z@]+|[{}$]", " ", first)
        match = re.match(r"\s*(Theorem|Lemma|Definition-Proposition|Definition|Proposition|Corollary|Remark|Example)"
                         r"\s+(\d+(?:\.\d+)*)\b", prefix, re.I)
        if not match:
            return []
        kinds = match[1].lower().split('-')
        return [(kind, match[2]) for kind in kinds]

    @staticmethod
    def numbered_pattern(kind, number):
        return r"(?<![A-Za-z])" + re.escape(kind) + r"\s+" + re.escape(number) + r"(?!\d|\.\d)"

    @staticmethod
    def overlap(left, right):
        return (left["file"] == right["file"] and left.get("occurrence") == right.get("occurrence")
                and left["char_start"] < right["char_end"] and right["char_start"] < left["char_end"])

    def contexts_for(self, identifier):
        result, seen = [], set()
        context_id = self.units[identifier]["context"]
        while context_id:
            if context_id in seen or context_id not in self.contexts:
                raise ValueError("Frozen Context chain is missing or cyclic")
            seen.add(context_id)
            context = self.contexts[context_id]
            result.append(context)
            context_id = context.get("parent")
        return result

    def context_text(self, identifier):
        return "\n".join(json.dumps({k: c.get(k) for k in ("description", "assumptions", "symbols")},
                                    ensure_ascii=False, sort_keys=True) for c in self.contexts_for(identifier))

    def references_for(self, identifier):
        spans = self.units[identifier]["sources"]
        return [item for item in self.reference_ledger
                if item.get("activity") != "inactive"
                and any(self.overlap(s, r) for s in spans
                        for r in item.get("source_spans", [item["span"]]))]

    def literal_explicit(self, source, target):
        # Saved label/ref text and unique saved endpoint spans only. Unknown
        # numbered pointers remain available for semantic classification review.
        source_unit, target_unit = self.units[source], self.units[target]
        labels = re.findall(r"\\label\s*\{([^{}]+)\}", self.body(source_unit))
        references = re.findall(r"\\(?:ref|eqref|autoref|[cC]ref|vref)\s*\{([^{}]+)\}", self.body(target_unit))
        if set(labels) & {x.strip() for r in references for x in r.split(',')}:
            return "saved_literal_label_pointer"
        for (kind, number), owner in self.named_headings.items():
            if owner == source and re.search(self.numbered_pattern(kind, number), self.body(target_unit), re.I):
                return "saved_unique_handwritten_numbered_pointer"
        for item in self.references_for(target):
            resolved = set()
            for endpoint in item.get("targets", []):
                if endpoint.get("resolution") != "resolved":
                    continue
                locations = endpoint.get("target_spans", [])
                for unit in self.units.values():
                    if any(self.overlap(s, t) for s in unit["sources"] for t in locations):
                        resolved.add(unit["id"])
            if resolved == {source}:
                return "saved_unique_reference_endpoint"
        return None

    def packet_nodes(self, identifiers):
        return [{k: copy.deepcopy(self.units[i][k]) for k in
                 ("id", "category", "label", "document", "context", "sources", "original_text")}
                for i in sorted(set(identifiers), key=self.indices.__getitem__)]

    def packet_contexts(self, identifiers):
        contexts = {}
        for identifier in identifiers:
            for context in self.contexts_for(identifier):
                contexts[context["id"]] = context
        return [copy.deepcopy(contexts[i]) for i in sorted(contexts)]


class SavedFeatureAdapter:
    """Compatibility only for existing lexical feature functions; no file IO."""
    structure_data = {"definitions": []}

    def __init__(self, paper):
        self.paper = paper

    def resolve_span(self, span):
        if set(span) != {"saved_unit_id"}:
            raise ValueError("Feature adapter cannot resolve arbitrary source locations")
        return self.paper.body(self.paper.units[span["saved_unit_id"]])


def retrieve(paper, options):
    ordered = paper.order
    body_tokens = [tokenize(paper.body(paper.units[i])) for i in ordered]
    context_tokens = [tokenize(paper.context_text(i)) for i in ordered]
    weight = options["context_weight"]
    documents = []
    for body, context in zip(body_tokens, context_tokens):
        counts = Counter(body)
        for token, number in Counter(context).items():
            counts[token] += weight * number
        documents.append(counts)
    index = BM25Index(documents)
    feature_adapter = SavedFeatureAdapter(paper)
    profiles = {}
    for identifier in ordered:
        unit = paper.units[identifier]
        adapter_node = {"label": unit["label"], "spans": [{"saved_unit_id": identifier}]}
        profiles[identifier] = (_features(adapter_node, feature_adapter)
                                if unit["category"] == "definition"
                                else _construction_features(adapter_node, feature_adapter))
    candidates, exclusions = [], []
    for target_index, target in enumerate(ordered):
        target_unit = paper.units[target]
        typ = "proof_dependency" if target_unit["category"] == "proof" else "statement_dependency"
        eligible = {i for i, source in enumerate(ordered) if source != target
                    and (options.get("source_proofs", True) or paper.units[source]["category"] != "proof")}
        union = defaultdict(list)
        query = set(body_tokens[target_index]) | set(context_tokens[target_index])
        query_weights = {term: (1.0 if term in body_tokens[target_index] else weight) for term in query}
        hits = index.search(query, weights=query_weights, eligible=eligible)
        floor = hits[0]["score"] * options["bm25_relative_floor"] if hits else 0
        filtered = [hit for hit in hits if hit["score"] >= floor and any(
            term_weight(term) >= .5 for term in set(body_tokens[target_index]) & set(body_tokens[hit["index"]]))]
        for hit in filtered[:options["bm25_top_k"]]:
            source = ordered[hit["index"]]
            union[source].append({"channel": "body_bm25_with_weighted_saved_context",
                                  "score": hit["score"], "shared_terms": hit["shared_terms"][:12]})
        if options["rule_features"]:
            target_text = paper.body(target_unit)
            for source_index in sorted(eligible):
                source = ordered[source_index]
                for feature in profiles[source]:
                    matches = list(feature["pattern"].finditer(target_text))
                    if matches:
                        union[source].append({"channel": "saved_declared_math_feature", "kind": feature["kind"],
                            "value": feature["value"], "priority": feature["priority"],
                            "body_match_offsets": [[m.start(), m.end()] for m in matches[:4]]})
        same_file = [source for source in ordered if paper.position(paper.units[source])[:2]
                     == paper.position(target_unit)[:2]]
        position = same_file.index(target)
        neighbours = [("fixed_previous", same_file[max(0, position - options["previous_k"]):position]),
                      ("fixed_following", same_file[position + 1:position + 1 + options["following_k"]])]
        for channel, sources in neighbours:
            for source in sources:
                if paper.indices[source] in eligible:
                    union[source].append({"channel": channel, "source_order_offset": paper.indices[source] - target_index})
        for source in sorted(union, key=paper.indices.__getitem__):
            exclusion = None
            if (source, target, typ) in paper.explicit_pairs:
                exclusion = "frozen_explicit_dependency"
            elif (source, target) in paper.saved_proof_goals or (target, source) in paper.saved_proof_goals:
                exclusion = "same_assertion_own_proof_goal_not_premise"
            else:
                exclusion = paper.literal_explicit(source, target)
            pair = {"candidate_id": "implicit:" + digest([paper.paper_id, source, target, typ])[:24],
                    "source": source, "target": target, "type": typ,
                    "retrieval": union[source], "source_order_offset": paper.indices[source] - target_index}
            if exclusion:
                exclusions.append({**pair, "reason": exclusion})
            else:
                candidates.append(pair)
    return {"schema_version": VERSION + ".retrieval", "paper_id": paper.paper_id,
            "options": options, "candidates": candidates, "exclusions": exclusions,
            "baseline_manifest": paper.baseline_binding, "production_binding": paper.production_binding,
            "node_count": len(ordered), "frozen_explicit_dependencies": len(paper.explicit_pairs),
            "input_read_paths": paper.read_paths, "raw_source_reading": False,
            "candidate_count": len(candidates), "counts_by_channel": dict(Counter(
                hit["channel"] for candidate in candidates for hit in candidate["retrieval"])),
            "backward_writing_order_candidates": sum(c["source_order_offset"] > 0 for c in candidates),
            "long_range_candidates": sum(abs(c["source_order_offset"]) > 5 for c in candidates)}


def decision_schema(identifiers, evaluation=False):
    verdicts = ["supported", "contradicted", "unresolved"] if evaluation else ["accept", "reject", "unresolved"]
    fields = {"candidate_id": {"type": "string", "enum": identifiers},
              "verdict": {"type": "string", "enum": verdicts},
              "explicitness": {"type": "string", "enum": ["implicit", "explicit", "unknown"]},
              **{key: {"type": "string"} for key in ("source_quote", "target_quote", "scope_explanation", "reason")}}
    return {"type": "object", "properties": {"decisions": {"type": "array", "items": {
        "type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}}},
        "required": ["decisions"], "additionalProperties": False}


def make_packet(paper, pairs, evaluation=False):
    identifiers = {c["source"] for c in pairs} | {c["target"] for c in pairs}
    payload = {"schema_version": VERSION + ".saved-input", "paper_id": paper.paper_id,
        "candidates": [{k: c[k] for k in ("candidate_id", "source", "target", "type")} for c in pairs],
        "nodes": paper.packet_nodes(identifiers), "contexts": paper.packet_contexts(identifiers),
        "saved_reference_records": {i: paper.references_for(i) for i in sorted(identifiers)
                                    if paper.references_for(i)},
        "context_policy": "Earlier saved interpretation, for scope only; use must be witnessed in target original_text.",
        "response_schema": decision_schema([c["candidate_id"] for c in pairs], evaluation)}
    # Do not leak production decisions, witness selections, scores or reasons.
    if any(key in payload for key in ("retrieval", "judgments", "producer_reasons")):
        raise ValueError("Production judgments leaked into independent material")
    return payload


def packets(paper, pairs, limits, evaluation=False):
    result, current, oversized = [], [], []
    for pair in pairs:
        proposal = current + [pair]
        payload = make_packet(paper, proposal, evaluation)
        if len(proposal) > limits["max_pairs"] or len(json.dumps(payload, ensure_ascii=False).encode()) > limits["max_payload_bytes"]:
            if current:
                result.append(make_packet(paper, current, evaluation))
                current = []
            single = make_packet(paper, [pair], evaluation)
            if len(json.dumps(single, ensure_ascii=False).encode()) > limits["max_payload_bytes"]:
                oversized.append({"candidate_id": pair["candidate_id"], "reason": "saved_input_exceeds_packet_limit_no_truncation"})
            else:
                current = [pair]
        else:
            current = proposal
    if current:
        result.append(make_packet(paper, current, evaluation))
    return result, oversized


def validate_decisions(paper, payload, response, evaluation=False):
    if not isinstance(response, dict) or set(response) != {"decisions"} or not isinstance(response["decisions"], list):
        raise ValueError("Invalid complete decision response")
    pairs = {p["candidate_id"]: p for p in payload["candidates"]}
    if len(pairs) != len(payload["candidates"]):
        raise ValueError("Duplicate candidate IDs in model material")
    if any(not isinstance(d, dict) for d in response["decisions"]):
        raise ValueError("Individual decisions must be objects")
    if Counter(d.get("candidate_id") for d in response["decisions"]) != Counter(pairs.keys()):
        raise ValueError("Model response must cover every candidate exactly once")
    fields = {"candidate_id", "verdict", "explicitness", "source_quote", "target_quote", "scope_explanation", "reason"}
    verdicts = {"supported", "contradicted", "unresolved"} if evaluation else {"accept", "reject", "unresolved"}
    normalized = []
    for decision in response["decisions"]:
        if set(decision) != fields or any(not isinstance(v, str) for v in decision.values()):
            raise ValueError("Invalid individual decision fields")
        if decision["verdict"] not in verdicts or decision["explicitness"] not in {"implicit", "explicit", "unknown"}:
            raise ValueError("Invalid decision state")
        pair = pairs[decision["candidate_id"]]
        checked = copy.deepcopy(decision)
        positive = decision["verdict"] == ("supported" if evaluation else "accept")
        errors = []
        if positive:
            for quote, endpoint in ((decision["source_quote"], pair["source"]), (decision["target_quote"], pair["target"])):
                # A quote must be inside one saved fragment, never span an
                # invented join between disjoint source fragments.
                if not quote.strip() or not any(quote in part for part in paper.units[endpoint]["original_text"]):
                    errors.append("positive_quote_not_bound_to_one_saved_original_fragment")
            if decision["explicitness"] != "implicit":
                errors.append("positive_not_confirmed_implicit")
            if not decision["scope_explanation"].strip() or not decision["reason"].strip():
                errors.append("positive_missing_concrete_scope_or_reason")
        if errors:
            checked["verdict"] = "unresolved"
        checked["host_validation_errors"] = errors
        normalized.append(checked)
    return normalized


def accepted_relations(paper, pairs, decisions):
    by_id = {p["candidate_id"]: p for p in pairs}
    if len(by_id) != len(pairs):
        raise ValueError("Duplicate candidate IDs in acceptance material")
    decision_ids = [d["candidate_id"] for d in decisions]
    if len(decision_ids) != len(set(decision_ids)) or set(decision_ids) != set(by_id):
        raise ValueError("Invalid accepted decision population")
    relations, evidence, witnesses = [], [], []
    index = GraphIndex(Graph.model_validate_json(json.dumps(paper.graph)))
    for decision in decisions:
        if decision["verdict"] != "accept":
            continue
        pair = by_id[decision["candidate_id"]]
        if (pair["source"], pair["target"], pair["type"]) in paper.explicit_pairs:
            raise ValueError("Implicit output duplicated frozen explicit dependency")
        if pair["source"] == pair["target"] or pair["type"] not in DEPENDENCIES:
            raise ValueError("Invalid implicit relation endpoints/type")
        expected = "proof_dependency" if paper.units[pair["target"]]["category"] == "proof" else "statement_dependency"
        if pair["type"] != expected:
            raise ValueError("Implicit dependency type does not match target category")
        if ((pair["source"], pair["target"]) in paper.saved_proof_goals
                or (pair["target"], pair["source"]) in paper.saved_proof_goals
                or paper.literal_explicit(pair["source"], pair["target"])):
            raise ValueError("Implicit acceptance violates saved explicit/proof-goal exclusions")
        fields = ("candidate_id", "verdict", "explicitness", "source_quote", "target_quote", "scope_explanation", "reason")
        checked = validate_decisions(paper, {"candidates": [pair]},
                                     {"decisions": [{k: decision[k] for k in fields}]})
        if checked[0]["verdict"] != "accept":
            raise ValueError("Implicit acceptance lacks valid saved-fragment witnesses")
        relation_id = "relation:" + digest([paper.paper_id, pair["source"], pair["target"], pair["type"], "implicit"])[:24]
        evidence_id = "evidence:" + digest([relation_id, decision])[:24]
        relation = {"id": relation_id, "type": pair["type"], "source": pair["source"], "target": pair["target"],
                    "explanation": decision["reason"], "conditions": decision["scope_explanation"], "evidence": [evidence_id]}
        relations.append(relation)
        witness = {"candidate_id": pair["candidate_id"], "relation_id": relation_id,
                   "source_quote": decision["source_quote"], "target_quote": decision["target_quote"],
                   "scope_explanation": decision["scope_explanation"], "explicitness": "implicit"}
        witnesses.append(witness)
        sources = copy.deepcopy(paper.units[pair["target"]]["sources"])
        claim_sha256 = index.claim_digest(Relation.model_validate_json(json.dumps(relation)), tuple(Source.model_validate(s) for s in sources))
        evidence.append({"id": evidence_id, "claim_sha256": claim_sha256, "method": "agent_analysis",
                         "sources": sources, "explanation": decision["reason"], "certificate": None})
    if len({(r["source"], r["target"], r["type"]) for r in relations}) != len(relations):
        raise ValueError("Duplicate newly accepted implicit relations")
    return relations, evidence, witnesses


def precision_report(paper, production, judgments, technical_missing=()):
    accepted = production["accepted_candidate_ids"]
    if len(accepted) != len(set(accepted)):
        raise ValueError("Duplicate accepted candidate IDs in evaluated population")
    ids = [d["candidate_id"] for d in judgments]
    if len(ids) != len(set(ids)) or set(ids) - set(accepted):
        raise ValueError("Independent evaluation population differs from accepted output")
    missing = sorted((set(accepted) - set(ids)) | set(technical_missing))
    if set(technical_missing) - set(accepted):
        raise ValueError("Technical missing population differs")
    correct = sum(d["verdict"] == "supported" and d["explicitness"] == "implicit" for d in judgments)
    counts = Counter(d["verdict"] for d in judgments)
    total = len(accepted)
    explicit = len(paper.explicit_pairs)
    return {"schema_version": VERSION + ".precision", "paper_id": paper.paper_id,
        "total_accepted_implicit": total, "correct_direct_implicit": correct,
        "contradicted": counts["contradicted"], "unresolved": counts["unresolved"],
        "classification_errors": sum(d["explicitness"] == "explicit" for d in judgments),
        "classification_unknown": sum(d["explicitness"] == "unknown" for d in judgments),
        "technical_unfinished": len(missing), "technical_missing_ids": missing,
        "precision": correct / total if total else None,
        "complete_independent_evaluation": not missing,
        "passed": bool(total and not missing and correct / total > .85),
        "frozen_explicit_dependencies": explicit,
        "accepted_implicit_to_explicit_ratio": total / explicit if explicit else None,
        "supported_implicit_to_explicit_ratio": correct / explicit if explicit else None,
        "by_type": {typ: {"total": sum(r["type"] == typ for r in production["relations"])} for typ in sorted(DEPENDENCIES)},
        "judgments": judgments, "same_model_independent_sessions": True,
        "expert_gold": False, "development_acceptance": True,
        "not_e8_formal_experiment": True, "whole_e4_t5_complete": False,
        "recall_measured": False, "topology_assessed": False}
