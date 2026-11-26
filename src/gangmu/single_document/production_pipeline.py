"""Source-independent configuration of the accepted single-document stages.

Production does not import any accuracy evaluator or old experiment result.
The existing mathematical policies remain in their versioned components;
this module supplies normal source admission and formal-output constraints.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gangmu.representation.graph import GraphIndex, validate
from gangmu.representation.model import Evidence, Graph, Relation, digest
from .topology_integration import FixedStagePipeline, VERSION, ROOT, _record, _local

POLICY_SCHEMA = "gangmu.single-document.policy.v1"
DEFAULT_POLICY = ROOT / "configs/e4-single-document-v1.json"

# Explicit production dependencies: historical evaluation hosts and reports do
# not participate. All executed helpers and runtime prompts are bound below.
PROGRAMS = (
    "rule_nodes_source_control_v1.py", "rule_nodes_task_project_v6.py",
    "rule_nodes.py", "inventory.py", "rule_nodes_named_heading_project_v2.py",
    "rule_nodes_named_heading_project_v3.py", "rule_nodes_named_headings_v2.py",
    "rule_nodes_qed_proofs_v3.py", "rule_nodes_project_supplement_v3.py",
    "rule_nodes_qed_proofs_v4.py", "rule_nodes_handwritten_v5.py", "rule_nodes_handwritten_v6.py",
    "rule_nodes_numbered_wrappers_v5.py", "rule_scan_review_v3.py", "rule_source_conditional_v1.py",
    "staged_nodes.py", "staged_nodes_context_strategy.py", "staged_nodes_wire_strategy.py",
    "staged_nodes_source_query_v1.py", "staged_nodes_context_scope_v1.py",
    "staged_context_request_recovery_v1.py", "staged_explicit_dependencies.py",
    "explicit_reference_inventory.py", "explicit_reference_inventory_v2.py", "candidates.py",
    "retrieval.py", "source.py", "implicit_dependencies.py", "implicit_client.py", "core_client.py",
    "core_response_stream.py", "backend.py",
    "budget.py", "cache.py", "topology.py", "topology_semantics.py", "topology_workflow.py",
    "topology_client.py", "topology_integration.py", "production_source_scan.py", "production_pipeline.py",
    "production_node_state.py",
)
PROMPTS = (
    "prompts/single_document/staged_nodes/round1.md",
    "prompts/single_document/staged_nodes/round2.md",
    "prompts/single_document/staged_nodes/context_policy_v4.md",
    "prompts/single_document/staged_nodes/context_scope_v1.md",
    "prompts/single_document/staged_explicit_dependencies/decide_v1.md",
    "prompts/single_document/source_scan_v1.md",
)


def load_policy(path=DEFAULT_POLICY, *, root=ROOT):
    root = Path(root).resolve()
    policy = json.loads(_local(root, path).read_bytes())
    required = {"schema_version", "agent_version", "authorization_config", "node_options",
                "node_wire_options", "stage_model_options", "explicit_options", "implicit_retrieval",
                "implicit_packets", "implicit_prompt"}
    if not isinstance(policy, dict) or set(policy) != required or policy["schema_version"] != POLICY_SCHEMA:
        raise ValueError("Unknown single-document production policy")
    from .staged_nodes import NodeOptions
    from .staged_nodes_wire_strategy import WireOptions
    from .topology_client import load_config
    NodeOptions.parse(policy["node_options"])
    WireOptions.parse(policy["node_wire_options"])
    load_config(_local(root, policy["authorization_config"]))
    if (not isinstance(policy["agent_version"], str) or not policy["agent_version"].strip()
            or set(policy["stage_model_options"]) != {"nodes", "explicit", "implicit"}
            or policy["explicit_options"].get("max_source_followups") != 2):
        raise ValueError("Single-document stages need a concrete version and bounded fixed options")
    for stage in policy["stage_model_options"].values():
        if (stage.get("reasoning_effort") not in {"low", "medium", "high", "xhigh"}
                or stage.get("response_transport") not in {"json", "sse"}
                or type(stage.get("max_output_tokens")) is not int or stage["max_output_tokens"] < 16
                or type(stage.get("timeout_seconds")) is not int or stage["timeout_seconds"] <= 0):
            raise ValueError("Invalid production stage model options")
    if any(policy["node_wire_options"][key] != policy["stage_model_options"]["nodes"][key]
           for key in ("reasoning_effort", "max_output_tokens", "response_transport")):
        raise ValueError("Node wire measurement must match the actual node model")
    _local(root, policy["implicit_prompt"])
    return copy.deepcopy(policy)


def source_stage_profile(project, policy, *, root=ROOT):
    """Pin the new input and production policy; never read an old result."""
    root = Path(root).resolve()
    if getattr(project, 'source_format', 'latex') == 'markdown':
        from .markdown_pipeline import markdown_profile
        return markdown_profile(project, policy, root=root)
    profile = {"schema_version": VERSION + ".fixed-profile", "paper_id": project.paper_id,
        "source_fingerprint": copy.deepcopy(project.fingerprint),
        "node_producer": "staged-nodes-context-scope-v1", "normal_source_profile": True,
        "agent_version": policy["agent_version"],
        **{key: copy.deepcopy(policy[key]) for key in (
            "node_options", "node_wire_options", "stage_model_options", "explicit_options",
            "implicit_retrieval", "implicit_packets", "implicit_prompt")},
        "program_bindings": [_record(root, "src/gangmu/single_document/" + name) for name in PROGRAMS]
            + [_record(root, name) for name in (
                "src/gangmu/latex_structure.py", "src/gangmu/representation/model.py",
                "src/gangmu/representation/graph.py", "src/gangmu/representation/sources.py",
                "src/gangmu/representation/storage.py", "src/gangmu/runtime/config.py",
                "src/gangmu/runtime/protocol.py")],
        "prompt_bindings": [_record(root, name) for name in (*PROMPTS, policy["implicit_prompt"])],
        "normal_production_source_admission": True,
        "runtime_precision_evaluation": False, "automatic_e3_publication": False}
    profile["profile_sha256"] = digest(profile)
    return profile


def production_pipeline(profile, judge, *, root=ROOT):
    if profile.get('source_format') == 'markdown':
        from .markdown_pipeline import MarkdownPipeline
        return MarkdownPipeline(profile, judge, root=root)
    return ProductionPipeline(profile, judge, root=root)


class ProductionPipeline(FixedStagePipeline):
    """Run the existing four stages on a newly pinned source project."""
    def __init__(self, profile, judge, *, root=ROOT, components=None, strict_nodes=False):
        super().__init__(profile, judge, root=root, components=components)
        self.strict_nodes = strict_nodes

    def for_complete_reextraction(self):
        """The once-only complete cycle re-extraction keeps its strict guard."""
        return ProductionPipeline(self.profile, self.judge, root=self.root,
                                  components=self.components, strict_nodes=True)

    def _rules(self, project, diagnosis, identity):
        from .production_source_scan import prepare_rules
        result = prepare_rules(project, self.judge, root=self.root,
                               profile_sha256=self.profile["profile_sha256"])
        if result["reviewed"]["metadata"]["scan_ready_for_discovery_acceptance"] is not True:
            raise ValueError("Source preparation has unresolved structural obligations; retained for continuation")
        return result

    def _nodes(self, project, rules, diagnosis, identity):
        if self.strict_nodes:
            return super()._nodes(project, rules, diagnosis, identity)
        from . import staged_nodes_context_scope_v1 as scope
        from .staged_nodes import NodeOptions
        from .production_node_state import RecordingJudge, classify_execution
        from .topology_integration import _save_once, _record
        recorder = RecordingJudge(self._judge("nodes"), root=self.root, identity=identity)
        pipeline = scope._ContextScopePipeline(project, rules["reviewed"]["rule_input"], recorder,
            NodeOptions.parse(self.profile["node_options"]), self.profile["node_wire_options"])
        pipeline.binding["topology_revision"] = identity
        pipeline.binding["cycle_diagnosis"] = copy.deepcopy(diagnosis)
        result = pipeline.run()
        # The exact same production identity can recover from a retained
        # technical failure using the remaining authorized supplier attempt.
        # Keep every raw outcome immutable without locking that identity to
        # its first failed outcome.
        raw_path = recorder.directory / ("raw-node-result-" + digest(result) + ".json")
        _save_once(raw_path, result)
        execution = classify_execution(result, recorder.records,
            expected_candidate_ids=[candidate["candidate_id"] for candidate in pipeline.candidates],
            expected_paragraph_ids=[row["source_id"] for row in pipeline.paragraphs],
            source_fingerprint=project.fingerprint, max_context_requests=pipeline.options.max_context_requests)
        execution["raw_node_result"] = _record(self.root, raw_path)
        report_path = recorder.directory / ("execution-report-" + digest(execution) + ".json")
        _save_once(report_path, execution)
        execution["execution_report"] = _record(self.root, report_path)
        if not execution["technical_complete"]:
            raise ValueError("Normal node production has unclosed technical tasks; inspect its retained execution report")
        # Preserve the exact original summary, unresolved records, accepted
        # population and every complete-chunk rollback; only output scope is new.
        result["production_execution"] = execution
        return result

    def _implicit(self, project, explicit, diagnosis, identity):
        from . import implicit_dependencies as implicit
        from .topology_integration import saved_graph_paper
        from .topology_semantics import host_e2_errors
        graph, references = explicit["result"]["graph"], explicit["result"]["reference_ledger"]
        paper = saved_graph_paper(graph, references, predecessor=identity,
                                 revision_id=identity["revision_id"], root=self.root)
        retrieval = implicit.retrieve(paper, self.profile["implicit_retrieval"])
        eligible, exclusions = [], []
        for pair in retrieval["candidates"]:
            errors = host_e2_errors(paper, pair)
            if errors:
                exclusions.append({"candidate": copy.deepcopy(pair), "errors": errors,
                                   "reason": "formal_endpoint_constraint"})
            else:
                eligible.append(pair)
        material, oversized = implicit.packets(paper, eligible, self.profile["implicit_packets"])
        if oversized:
            raise ValueError("Implicit production has unprocessed oversized candidates")
        judge, decisions, calls = self._judge("implicit"), [], []
        prompt = _local(self.root, self.profile["implicit_prompt"]).read_text(encoding="utf-8")
        for payload in material:
            payload["topology_revision"], payload["cycle_diagnosis"] = identity, copy.deepcopy(diagnosis)
            response = judge("fixed-implicit-direct-use", prompt, payload)
            decisions.extend(implicit.validate_decisions(paper, payload, response))
            calls.append({"payload": payload, "response": response})
        order = {p["candidate_id"]: i for i, p in enumerate(eligible)}
        decisions.sort(key=lambda row: order[row["candidate_id"]])
        relations, evidence, witnesses = implicit.accepted_relations(paper, eligible, decisions)
        combined = copy.deepcopy(graph)
        combined["relations"].extend(relations)
        combined["evidence"].extend(evidence)
        combined["snapshot_id"] = "single-document:" + digest({"identity": identity, "graph": combined})[:24]
        Graph.model_validate_json(json.dumps(combined))
        return {"graph": combined, "reference_ledger": references, "retrieval": retrieval,
                "decisions": decisions, "calls": calls, "witnesses": witnesses,
                "formal_candidate_exclusions": exclusions, "runtime_precision_evaluation": False}


def formal_projection(graph, *, root=ROOT):
    """Preserve raw extraction; isolate unbound proofs and invalid types.

    No mathematical content is invented, no relation is relabeled, and no
    unsupported evidence is made valid by merely recomputing its checksum.
    """
    model = Graph.model_validate_json(json.dumps(graph, ensure_ascii=False))
    original = model.model_dump(mode="json")
    index = GraphIndex(model)
    # Check every original evidence claim before allowing a structural subset.
    for relation in model.relations:
        for eid in relation.evidence:
            ev = index.get(eid, Evidence)
            if ev.claim_sha256 != index.claim_digest(relation, ev.sources):
                raise ValueError("Original evidence binding is invalid: " + eid)
    units = {unit.id: unit for unit in model.units}
    removed_relations, retained = [], []
    for relation in model.relations:
        src, dst = index.get(relation.source), index.get(relation.target)
        invalid = (relation.type == "statement_dependency" and (src.category == "proof" or dst.category == "proof")
                   or relation.type == "proof_dependency" and dst.category != "proof"
                   or relation.type == "proves" and (src.category != "proof" or dst.category != "statement"
                                                     or src.document != dst.document))
        if invalid:
            removed_relations.append({"relation": relation.model_dump(mode="json"),
                                      "reason": "formal_endpoint_type_constraint"})
        else:
            retained.append(relation)
    paired = {r.source for r in retained if r.type == "proves"}
    pending = {u.id for u in model.units if u.category == "proof" and u.id not in paired}
    relations = []
    for relation in retained:
        if relation.source in pending or relation.target in pending:
            removed_relations.append({"relation": relation.model_dump(mode="json"),
                                      "reason": "endpoint_is_unattributed_proof_candidate"})
        else:
            relations.append(relation.model_dump(mode="json"))
    output = copy.deepcopy(original)
    output["units"] = [u for u in output["units"] if u["id"] not in pending]
    output["relations"] = relations
    removed_ids = {r["relation"]["id"] for r in removed_relations}
    views = [v for v in output["identity_views"] if set(v["members"]) & pending
             or set(v["justifications"]) & removed_ids]
    output["identity_views"] = [v for v in output["identity_views"] if v not in views]
    evidence_ids = {eid for r in relations for eid in r["evidence"]}
    output["evidence"] = [e for e in output["evidence"] if e["id"] in evidence_ids]
    projected = GraphIndex(Graph.model_validate_json(json.dumps(output)))
    rebound = []
    targets = {}
    for relation in projected.graph.relations:
        for eid in relation.evidence:
            ev = projected.get(eid, Evidence)
            expected = projected.claim_digest(relation, ev.sources)
            if eid in targets and targets[eid] != expected:
                raise ValueError("An evidence record would bind different projected claims")
            targets[eid] = expected
    for ev in output["evidence"]:
        expected = targets[ev["id"]]
        if ev["claim_sha256"] != expected:
            rebound.append({"evidence_id": ev["id"], "old": ev["claim_sha256"], "new": expected})
            ev["claim_sha256"] = expected
    if digest(output) != digest(original):
        output["snapshot_id"] = "single-document-formal:" + digest(output)[:24]
    final = Graph.model_validate_json(json.dumps(output))
    errors = validate(final).errors
    if errors:
        raise ValueError("Unresolved formal output errors: " + json.dumps(errors, ensure_ascii=False))
    report = {"schema_version": "gangmu.single-document.formal-projection.v1",
        "raw_graph_sha256": digest(original), "formal_graph_sha256": digest(output),
        "unattributed_proof_candidates": [units[uid].model_dump(mode="json") for uid in sorted(pending)],
        "excluded_relations": removed_relations, "excluded_identity_views": views,
        "excluded_evidence": [e for e in original["evidence"] if e["id"] not in evidence_ids],
        "evidence_rebindings": rebound, "formal_e2_eligible": True,
        "mathematical_correctness_certified": False, "proof_ownership_invented": False,
        "counts": {"raw_nodes": len(original["units"]), "formal_nodes": len(output["units"]),
                   "raw_relations": len(original["relations"]), "formal_relations": len(relations),
                   "proof_candidates": len(pending), "excluded_relations": len(removed_relations)}}
    return {"graph": output, "report": report}
