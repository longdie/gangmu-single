"""Fixed extraction orchestration and a finalization hook before graph output.

The old Codex extractor is deliberately not called here.  The accepted staged
node, explicit-use and saved-node implicit policies are reused without changing
their prompts or mathematical admission rules.  Source access belongs to normal
production or to the one authorized complete re-extraction, never a saved-graph
finalization.  All semantic calls use the injected shared topology judge.
"""
from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from gangmu.representation.graph import validate
from gangmu.representation.model import Graph, digest
from .cache import _atomic_json
from .implicit_dependencies import SavedPaper

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.e4.topology-integration.v2"
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.:-]{0,159}")
_STAGES = ("rules", "nodes", "explicit", "implicit")


def _local(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Finalization paths must be project-local without symlinks")
    path = path.resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError("Finalization artifacts must remain inside the project")
    return path


def _record(root, path):
    path = _local(root, path)
    return {"path": path.relative_to(root).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _bound(root, record):
    if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
        raise ValueError("A complete path and SHA256 binding is required")
    if _record(root, record["path"]) != record:
        raise ValueError("Fixed stage input bytes changed: " + record["path"])
    return json.loads(_local(root, record["path"]).read_bytes())


def _save_once(path, value):
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("An immutable finalization artifact changed")
    else:
        _atomic_json(path, value)


@contextmanager
def _locked(directory):
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".integration.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        yield


def saved_graph_paper(graph, reference_ledger=(), *, predecessor=None, revision_id="saved", root=ROOT):
    """Use the existing implicit methods over an in-memory saved MathIR graph.

    No file/source capability is installed.  Explicit pointers and proof goals
    are excluded in exactly the same way as the accepted saved-input adapter.
    """
    model = Graph.model_validate_json(json.dumps(graph, ensure_ascii=False))
    data = model.model_dump(mode="json")
    if len(data["documents"]) != 1:
        raise ValueError("Finalization expects one mathematical document")
    paper = SavedPaper.__new__(SavedPaper)
    paper.root, paper.graph = Path(root).resolve(), data
    paper.paper_id = data["documents"][0]["paper_id"]
    paper.units = {u["id"]: u for u in data["units"]}
    paper.contexts = {c["id"]: c for c in data["contexts"]}
    if len(paper.units) != len(data["units"]) or len(paper.contexts) != len(data["contexts"]):
        raise ValueError("Duplicate saved Unit or Context identities")
    paper.reference_ledger, paper.read_paths = copy.deepcopy(list(reference_ledger)), []
    paper.baseline_binding = copy.deepcopy(predecessor or {"revision_id": revision_id, "graph_sha256": digest(data)})
    paper.production_binding = {"revision_id": revision_id, "graph_sha256": digest(data)}
    paper.order = sorted(paper.units, key=lambda i: paper.position(paper.units[i]))
    paper.indices = {identifier: i for i, identifier in enumerate(paper.order)}
    paper.explicit_pairs = {(r["source"], r["target"], r["type"]) for r in data["relations"]
                            if r["type"] in ("statement_dependency", "proof_dependency")}
    paper.proves_pairs = {(r["source"], r["target"]) for r in data["relations"] if r["type"] == "proves"}
    headings = defaultdict(set)
    for identifier, unit in paper.units.items():
        if unit["category"] != "proof":
            for kind, number in paper.manual_heading(unit):
                headings[(kind, number)].add(identifier)
    paper.named_headings = {key: next(iter(ids)) for key, ids in headings.items() if len(ids) == 1}
    paper.saved_proof_goals = set(paper.proves_pairs)
    for identifier, unit in paper.units.items():
        if unit["category"] == "proof":
            first_line = paper.body(unit).splitlines()[0]
            for (kind, number), target in paper.named_headings.items():
                if re.search(r"\bproof\s+of\s+(?:the\s+)?" + paper.numbered_pattern(kind, number), first_line, re.I):
                    paper.saved_proof_goals.add((identifier, target))
    for identifier in paper.units:
        paper.contexts_for(identifier)
    return paper


def frozen_stage_profile(baseline_manifest, implicit_config, paper_id, *, root=ROOT):
    """Read existing bound production metadata; this function never opens TeX.

    It records the actual accepted node options, wire limits and source-review
    packet, rather than relying on the historical full-extractor defaults.
    Static source-local endpoint repairs are provenance, not a generic strategy
    to silently splice into a newly extracted node population.
    """
    root = Path(root).resolve()
    manifest = _bound(root, baseline_manifest)
    row = next((p for p in manifest["papers"] if p["paper_id"] == paper_id), None)
    if row is None:
        raise ValueError("Paper is not present in this fixed predecessor manifest")
    nodes = _bound(root, row["node_result"])
    binding = nodes["producer_binding"]
    if nodes["producer_version"] != "staged-nodes-context-scope-v1":
        raise ValueError("This adapter only implements the accepted context-scope node strategy")
    exact_review = row["production_exact_review"]
    while "config" not in exact_review and "base_exact_review" in exact_review:
        exact_review = exact_review["base_exact_review"]
    config_binding = exact_review["config"]
    config = _bound(root, config_binding)
    implicit = _bound(root, implicit_config)
    freeze = _bound(root, manifest["frozen_nodes_manifest"])
    source_control = binding["source_control_admission"]
    stages = {"nodes": {"reasoning_effort": binding["wire_options"]["reasoning_effort"],
                       "max_output_tokens": binding["wire_options"]["max_output_tokens"],
                       "response_transport": binding["wire_options"]["response_transport"],
                       "timeout_seconds": config["node_options"]["timeout_seconds"]},
              "explicit": {key: config["node_options"][key] for key in
                           ("reasoning_effort", "max_output_tokens", "response_transport", "timeout_seconds")},
              "implicit": copy.deepcopy(implicit["model_options"])}
    names = ("rule_nodes_source_control_v1.py", "rule_nodes_task_project_v6.py",
        "rule_nodes.py", "inventory.py", "rule_nodes_named_heading_project_v2.py",
        "rule_nodes_named_heading_project_v3.py", "rule_nodes_named_headings_v2.py",
        "rule_nodes_qed_proofs_v3.py", "rule_nodes_project_supplement_v3.py",
        "rule_nodes_qed_proofs_v4.py", "rule_nodes_handwritten_v5.py", "rule_nodes_handwritten_v6.py",
        "rule_nodes_numbered_wrappers_v5.py", "rule_scan_review_v3.py", "rule_source_conditional_v1.py",
        "staged_nodes.py", "staged_nodes_context_strategy.py", "staged_nodes_wire_strategy.py",
        "staged_nodes_source_query_v1.py", "staged_nodes_context_scope_v1.py",
        "staged_context_request_recovery_v1.py", "staged_explicit_dependencies.py",
        "explicit_reference_inventory.py", "explicit_reference_inventory_v2.py", "candidates.py",
        "retrieval.py", "source.py", "implicit_dependencies.py")
    programs = [_record(root, "src/gangmu/single_document/" + name) for name in names]
    programs += [_record(root, name) for name in ("src/gangmu/latex_structure.py",
        "src/gangmu/representation/model.py", "src/gangmu/representation/graph.py",
        "src/gangmu/representation/sources.py")]
    prompts = [_record(root, name) for name in (
        "prompts/single_document/staged_nodes/round1.md", "prompts/single_document/staged_nodes/round2.md",
        "prompts/single_document/staged_nodes/context_policy_v4.md",
        "prompts/single_document/staged_nodes/context_scope_v1.md",
        "prompts/single_document/staged_explicit_dependencies/decide_v1.md", implicit["judge_prompt"])]
    profile = {"schema_version": VERSION + ".fixed-profile", "paper_id": paper_id,
        "baseline_manifest": baseline_manifest, "node_result": row["node_result"],
        "explicit_config": config_binding, "implicit_config": implicit_config,
        "source_artifact_root": freeze["source_artifact_root"], "source_fingerprint": row["source_fingerprint"],
        "node_producer": nodes["producer_version"], "node_options": binding["options"],
        "node_wire_options": binding["wire_options"], "stage_model_options": stages,
        "source_review_packet": source_control["source_review_packet"],
        "source_control_binding": copy.deepcopy(source_control),
        "explicit_scanner": "gangmu.e4.explicit-reference-inventory.v2",
        "explicit_options": config["explicit_options"],
        "implicit_retrieval": implicit["retrieval"], "implicit_packets": implicit["packets"],
        "implicit_prompt": implicit["judge_prompt"], "program_bindings": programs,
        "prompt_bindings": prompts, "baseline_local_endpoint_repairs": nodes.get("local_endpoint_repairs", []),
        "historical_explicit_recovery_kind": row["production_exact_review"].get("producer_kind"),
        "old_cache_reused_as_new_production": False, "automatic_e3_publication": False}
    profile["profile_sha256"] = digest(profile)
    return profile


class FixedStagePipeline:
    """The current stage composition, followed by a mandatory finalizer hook.

    ``judge.fork(stage_options, allocation=...)`` must retain the same campaign
    gate.  Component injection is only for offline checks; production defaults
    call the frozen existing extraction functions below.
    """
    def __init__(self, profile, judge, *, root=ROOT, components=None):
        self.root, self.profile, self.judge = Path(root).resolve(), copy.deepcopy(profile), judge
        if self.profile.get("profile_sha256") != digest({k: v for k, v in self.profile.items() if k != "profile_sha256"}):
            raise ValueError("Fixed-stage profile digest differs")
        if self.profile.get("schema_version") != VERSION + ".fixed-profile":
            raise ValueError("Unknown fixed-stage profile")
        self.components, self._stage_judges = components or {}, []

    def _check_profile(self):
        for binding in self.profile["program_bindings"] + self.profile["prompt_bindings"]:
            if _record(self.root, binding["path"]) != binding:
                raise ValueError("A fixed extraction policy changed during finalization")

    def _judge(self, stage):
        judge = self.judge.fork(copy.deepcopy(self.profile["stage_model_options"][stage]), allocation="production")
        if judge.ledger.path != self.judge.ledger.path or judge.ledger.scope != self.judge.ledger.scope:
            raise ValueError("All extraction stages must use the same topology campaign budget")
        expected = self.profile["stage_model_options"][stage]
        actual = judge.client.model
        if (actual["model"] != "gpt-6.1-sol" or actual["provider"] != "yxai"
                or actual["effort"] != expected.get("reasoning_effort", expected.get("effort"))
                or actual["max_output_tokens"] != expected["max_output_tokens"]
                or actual.get("response_transport", "json") != expected["response_transport"]):
            raise ValueError("The actual stage model differs from the fixed profile")
        self._stage_judges.append((stage, judge))
        return judge

    def _rules(self, project, diagnosis, identity):
        from . import rule_nodes_source_control_v1 as control
        envelope = control.prepare_source_control(project)
        packet = _bound(self.root, self.profile["source_review_packet"]) if self.profile["source_review_packet"] else None
        reviewed = control.resolve_source_control_review(project, envelope, packet)
        if reviewed["metadata"]["scan_ready_for_discovery_acceptance"] is not True:
            raise ValueError("Complete re-extraction cannot bypass unresolved source-scan obligations")
        expected = self.profile["source_control_binding"]
        if (control._value_sha256(envelope) != expected["source_control_envelope_value_sha256"]
                or control._value_sha256(reviewed) != expected["reviewed_rule_envelope_value_sha256"]):
            raise ValueError("The fixed complete rule/source-control production changed")
        return {"envelope": envelope, "reviewed": reviewed}

    def _nodes(self, project, rules, diagnosis, identity):
        from . import staged_nodes_context_scope_v1 as scope
        from .staged_nodes import NodeOptions
        pipeline = scope._ContextScopePipeline(project, rules["reviewed"]["rule_input"], self._judge("nodes"),
            NodeOptions.parse(self.profile["node_options"]), self.profile["node_wire_options"])
        pipeline.binding["topology_revision"] = identity
        pipeline.binding["cycle_diagnosis"] = copy.deepcopy(diagnosis)
        result = pipeline.run()
        if result["summary"]["status"] != "complete":
            raise ValueError("Complete re-extraction has unresolved node-production work")
        return result

    def _explicit(self, project, nodes, diagnosis, identity):
        from . import explicit_reference_inventory_v2 as scanner
        from . import staged_explicit_dependencies as explicit
        references = scanner.scan_references(project)
        scan = explicit.build_scan(project, nodes, references=references)
        options = self.profile["explicit_options"]
        model = self.profile["stage_model_options"]["explicit"]
        wire = {"model": "gpt-6.1-sol", **{key: model[key] for key in
                ("reasoning_effort", "max_output_tokens", "response_transport")}}
        plan = explicit.prepare_packets(project, nodes, scan, max_wire_bytes=options["max_wire_bytes"],
            max_items=options["batch_max_items"], wire_options=wire)
        # Rebuild packets with revision metadata before run_judgments measures
        # their actual wires.  Fixed task populations/admission remain unchanged.
        plan["binding"]["topology_revision"] = identity
        plan["binding"]["cycle_diagnosis"] = copy.deepcopy(diagnosis)
        plan["packets"] = [explicit._packet(p["items"], plan["binding"]) for p in plan["packets"]]
        result, calls = explicit.run_judgments(project, nodes, scan, plan, self._judge("explicit"),
            max_source_followups=options["max_source_followups"])
        if plan["oversize_tasks"]:
            raise ValueError("Complete re-extraction has technically unprocessed explicit tasks")
        return {"result": result, "calls": calls, "scan": scan, "plan": plan,
                "actual_reference_scanner": _record(self.root, scanner.__file__)}

    def _implicit(self, project, explicit, diagnosis, identity):
        from . import implicit_dependencies as implicit
        graph, reference_ledger = explicit["result"]["graph"], explicit["result"]["reference_ledger"]
        paper = saved_graph_paper(graph, reference_ledger, predecessor=identity,
                                  revision_id=identity["revision_id"], root=self.root)
        retrieval = implicit.retrieve(paper, self.profile["implicit_retrieval"])
        material, oversized = implicit.packets(paper, retrieval["candidates"], self.profile["implicit_packets"])
        if oversized:
            raise ValueError("Complete re-extraction has unprocessed oversized implicit candidates")
        judge, decisions, calls = self._judge("implicit"), [], []
        prompt = _local(self.root, self.profile["implicit_prompt"]).read_text(encoding="utf-8")
        for payload in material:
            payload["topology_revision"] = identity
            payload["cycle_diagnosis"] = copy.deepcopy(diagnosis)
            response = judge("fixed-implicit-direct-use", prompt, payload)
            decisions.extend(implicit.validate_decisions(paper, payload, response))
            calls.append({"payload": payload, "response": response})
        positions = {p["candidate_id"]: i for i, p in enumerate(retrieval["candidates"])}
        decisions.sort(key=lambda row: positions[row["candidate_id"]])
        relations, evidence, witnesses = implicit.accepted_relations(paper, retrieval["candidates"], decisions)
        combined = copy.deepcopy(graph)
        combined["relations"].extend(relations)
        combined["evidence"].extend(evidence)
        combined["snapshot_id"] = "fixed-stages:" + digest({"identity": identity, "graph": combined})[:24]
        Graph.model_validate_json(json.dumps(combined))
        return {"graph": combined, "reference_ledger": reference_ledger, "retrieval": retrieval,
                "decisions": decisions, "calls": calls, "witnesses": witnesses}

    def extract(self, project, *, revision_id, diagnosis, output):
        if not _IDENTIFIER.fullmatch(revision_id):
            raise ValueError("Use a stable bounded revision identity, never random force generations")
        directory = _local(self.root, output)
        identity = {"revision_id": revision_id, "profile_sha256": self.profile["profile_sha256"],
                    "source_fingerprint_sha256": digest(project.fingerprint), "diagnosis_sha256": digest(diagnosis)}
        if (project.paper_id != self.profile["paper_id"] or project.fingerprint != self.profile["source_fingerprint"]):
            raise ValueError("Complete re-extraction must use the same pinned document source")
        self._check_profile()
        with _locked(directory):
            _save_once(directory / "identity.json", identity)
            state = {}
            for stage in _STAGES:
                path = directory / (stage + ".json")
                if path.exists():
                    saved = json.loads(path.read_bytes())
                    if (saved.get("identity") != identity or saved.get("stage") != stage
                            or saved.get("value_sha256") != digest(saved.get("value"))):
                        raise ValueError("Retained stage checkpoint differs from its revision binding")
                    state[stage] = saved["value"]
                    continue
                function = self.components.get(stage, getattr(self, "_" + stage))
                prior = state["rules"] if stage == "nodes" else state["nodes"] if stage == "explicit" else state["explicit"] if stage == "implicit" else None
                args = (project, diagnosis, identity) if stage == "rules" else (project, prior, diagnosis, identity)
                value = function(*args)
                self._check_profile()
                project.assert_unchanged()
                _save_once(path, {"identity": identity, "stage": stage, "value": value, "value_sha256": digest(value)})
                state[stage] = value
            result = {"schema_version": VERSION + ".fixed-production", "graph": state["implicit"]["graph"],
                "reference_ledger": state["implicit"]["reference_ledger"], "identity": identity,
                "binding": _record(self.root, directory / "identity.json"),
                "stages": {stage: _record(self.root, directory / (stage + ".json")) for stage in _STAGES},
                "technical_complete": True,
                "completed_stages": ["nodes", "contexts", "explicit_dependencies", "implicit_dependencies"],
                "automatic_e3_publication": False}
            _save_once(directory / "production.json", result)
            _save_once(directory / ("request-receipts-" + digest([stage_judge.receipts() for _, stage_judge in self._stage_judges])[:16] + ".json"),
                {"revision_id": revision_id, "stage_receipts": [{"stage": stage, "receipts": stage_judge.receipts()}
                                                               for stage, stage_judge in self._stage_judges]})
        return result

    def run(self, project, *, revision_id, output, finalizer, diagnosis=None, reextract=None):
        """Normal Agent path: complete predecessor stages -> finalizer -> output."""
        output = _local(self.root, output)
        production = self.extract(project, revision_id=revision_id, diagnosis=diagnosis or {}, output=output / "predecessor")
        return finalize_saved_graph(graph=production["graph"], predecessor=production,
            revision_id=revision_id, output=output / "final", reextract=reextract, finalizer=finalizer, root=self.root)


def finalize_saved_graph(*, graph, predecessor, revision_id, output, reextract=None, finalizer, root=ROOT):
    """The final Agent hook, usable directly for cached predecessor artifacts.

    Only after the finalizer returns a verified result is diagnostic graph.json
    emitted.  Formal publication remains a separate existing E2/E3 gate; DAG
    success never grants mathematical acceptance or fixes proof ownership.
    """
    root = Path(root).resolve()
    directory = _local(root, output)
    if not _IDENTIFIER.fullmatch(revision_id):
        raise ValueError("A concrete stable finalization revision is required")
    model = Graph.model_validate_json(json.dumps(graph, ensure_ascii=False))
    with _locked(directory):
        result = finalizer(graph=model.model_dump(mode="json"), predecessor=copy.deepcopy(predecessor),
            revision_id=revision_id, output=directory, reextract=reextract)
        if not isinstance(result, dict) or not isinstance(result.get("graph"), dict) or not isinstance(result.get("report"), dict):
            raise ValueError("The finalizer must return graph and report objects")
        final = Graph.model_validate_json(json.dumps(result["graph"], ensure_ascii=False))
        report = copy.deepcopy(result["report"])
        if report.get("completed") is not True:
            raise ValueError("Incomplete topology processing cannot emit the final Agent output")
        # Recompute the all-node dependency union locally.  This output guard is
        # independent of the semantic repair callback and includes isolated nodes.
        _check_all_node_dag(final.model_dump(mode="json"))
        qualification = validate(final)
        report.update(agent_finalization_position="after_fixed_stages_before_output_and_publication",
            formal_e2_eligible=not bool(qualification.errors), formal_e2_errors=list(qualification.errors),
            automatic_e3_publication=False, graph_sha256=digest(final.model_dump(mode="json")))
        _save_once(directory / "agent-final-graph.json", final.model_dump(mode="json"))
        _save_once(directory / "agent-finalization-report.json", report)
        return {"graph": final.model_dump(mode="json"), "report": report}


def _check_all_node_dag(graph):
    indegree = {u["id"]: 0 for u in graph["units"]}
    outgoing = {identifier: [] for identifier in indegree}
    for relation in graph["relations"]:
        if relation["type"] in ("statement_dependency", "proof_dependency"):
            if relation["source"] not in indegree or relation["target"] not in indegree:
                raise ValueError("A dependency endpoint is missing from the final graph")
            outgoing[relation["source"]].append(relation["target"])
            indegree[relation["target"]] += 1
    ready = sorted(identifier for identifier, degree in indegree.items() if degree == 0)
    visited = 0
    while ready:
        identifier = ready.pop()
        visited += 1
        for target in outgoing[identifier]:
            indegree[target] -= 1
            if indegree[target] == 0:
                ready.append(target)
    if visited != len(indegree):
        raise ValueError("The final Agent graph still contains a dependency-union cycle")


class CompleteReextractAdapter:
    """Lazy, once-authorized complete staged production with stable continuation.

    The core controller owns the persistent cumulative retry count.  This
    adapter additionally locks its first revision and diagnosis on disk, so
    neither a restart nor a caller's changed graph creates another identity.
    Complete production and topology checks are runtime requirements.  Precision
    evaluation belongs to an explicitly invoked offline acceptance experiment.
    """
    def __init__(self, pipeline, source_loader, quality_evaluator=None, *, output):
        self.pipeline, self.source_loader, self.quality_evaluator = pipeline, source_loader, quality_evaluator
        self.output = _local(pipeline.root, output)

    def __call__(self, revision_id, diagnosis, state):
        reservation = state.get("reextract")
        events = [row for row in state.get("events", []) if row.get("kind") == "reextract_reserved"]
        if (not isinstance(reservation, dict) or reservation.get("revision_id") != revision_id
                or reservation.get("status") not in {"reserved", "failed", "completed"} or len(events) != 1):
            raise ValueError("The core must retain exactly one complete re-extraction revision")
        self.output.mkdir(parents=True, exist_ok=True)
        with _locked(self.output):
            _save_once(self.output / "one-revision-authorization.json", {
                "revision_id": revision_id, "profile_sha256": self.pipeline.profile["profile_sha256"],
                "diagnosis": copy.deepcopy(diagnosis), "complete_reextraction_count": 1})
            # SourceProject is deliberately created only after the above checks.
            project = self.source_loader()
            result = self.pipeline.extract(project, revision_id=revision_id, diagnosis=diagnosis,
                                            output=self.output / "fixed-production")
            expected = {"nodes", "contexts", "explicit_dependencies", "implicit_dependencies"}
            if (not isinstance(result, dict) or result.get("technical_complete") is not True
                    or set(result.get("completed_stages", [])) != expected
                    or not isinstance(result.get("stages"), dict) or set(result["stages"]) != set(_STAGES)
                    or not isinstance(result.get("reference_ledger"), list)):
                raise ValueError("Complete re-extraction has not technically completed every fixed production stage")
            Graph.model_validate_json(json.dumps(result["graph"], ensure_ascii=False))
            _bound(self.pipeline.root, result["binding"])
            for pointer in result["stages"].values():
                _bound(self.pipeline.root, pointer)
            # The optional historical constructor argument is retained for API
            # compatibility only.  A production re-extraction never invokes it.
            return {**result, "runtime_precision_evaluation": False,
                "completion_evidence": {"technical_complete": True,
                    "completed_stages": result["completed_stages"], "artifact_bindings": result["stages"]}}



def __getattr__(name):
    # Historical offline callers may still explicitly import this API. Normal
    # extraction does not import the evaluator or bind its implementation bytes.
    if name in {"WholeOutputQualityEvaluator", "_validate_whole_quality", "_whole_edge_decisions"}:
        from gangmu.evaluation import single_document_topology
        return getattr(single_document_topology, name)
    raise AttributeError(name)
