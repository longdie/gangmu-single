"""The production single-document module: staged extraction, finalization, export.

This module does not import an acceptance evaluator or publish E3 proposals.
Completed results are immutable and every reuse rechecks both their retained
bytes and the current pinned source. Failed work keeps its stage/topology
checkpoints; an invalid cache is an error, never permission for paid rework.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import threading
import time

from gangmu.runtime.layout import project_of, storage_path
from gangmu.representation.graph import validate
from gangmu.representation.model import Graph, digest
from gangmu.representation.storage import load_bundle, save_bundle, SCHEMA_VERSION
from .cache import _atomic_json, _public, file_digests, stable_fingerprint
from .source import SourceProject
from .topology_integration import CompleteReextractAdapter, _check_all_node_dag, _bound

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.single_document.production.v2"
DEFAULT_POLICY = "configs/e4-single-document-v1.json"


class CacheIntegrityError(ValueError):
    """Retained work must be inspected, rather than silently re-extracted."""


def _local(root, value):
    path = Path(value)
    path = path if path.is_absolute() else root / path
    if ".." in path.parts or any(p.is_symlink() for p in (path, *path.parents)):
        raise ValueError("Module paths must not contain traversal or symbolic links")
    path = path.resolve()
    if path == root or not path.is_relative_to(root):
        raise ValueError("Module outputs must remain inside the project")
    return path


def _read(path):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise CacheIntegrityError("Duplicate key in a retained JSON object")
            result[key] = value
        return result
    def constant(_):
        raise CacheIntegrityError("Nonfinite retained JSON number")
    return json.loads(Path(path).read_bytes(), object_pairs_hook=pairs, parse_constant=constant)


def _save_once(path, value):
    if path.exists():
        if _read(path) != value:
            raise CacheIntegrityError("An immutable module artifact differs from its binding")
    else:
        _atomic_json(path, _public(value))


def _record(path, value):
    public = _public(value)
    _atomic_json(path, {**public, "record_sha256": digest(public)})


def _read_record(path):
    try:
        value = _read(path)
        if not isinstance(value, dict) or value.get("record_sha256") != digest(
                {k: v for k, v in value.items() if k != "record_sha256"}):
            raise CacheIntegrityError("Module cache record checksum differs")
        return value
    except (OSError, ValueError, TypeError, KeyError) as exc:
        raise CacheIntegrityError("Module cache record is invalid; no extraction was authorized") from exc


@contextmanager
def _lock(root, directory, name, timeout):
    directory = _local(root, directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    descriptor = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        until = None if timeout is None else time.monotonic() + max(0, timeout)
        while True:
            try:
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if until is not None and time.monotonic() >= until:
                    raise TimeoutError("Another process owns this document extraction")
                time.sleep(.05)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _semantic_profile(profile):
    # Authorization and storage identities control execution, not mathematical
    # reuse. They are separately bound on incomplete work below.
    excluded = {"profile_sha256", "policy_sha256", "authorization_config",
        "authorization_config_sha256", "budget", "budget_inheritance", "campaign_id",
        "ledger_id", "source_artifact_root", "source_review_packet", "cache_root"}
    return {key: copy.deepcopy(value) for key, value in profile.items() if key not in excluded}


def _topology_parameters(config):
    names = ("runtime_policy", "model_options", "expanded_retrieval", "packets",
        "max_node_expansion_rounds", "max_document_isolation_rounds", "max_reextractions_per_document",
        "pruning_order", "repair_relation_evaluation", "runtime_precision_gate", "whole_output_runtime_evaluation")
    return {name: copy.deepcopy(config[name]) for name in names}


def _ledger_rows(judge):
    path = getattr(judge.ledger, "path", None)
    if path is None or not Path(path).exists():
        return []
    value = _read(path)
    return value.get("invocations", [])


def _verify_retained_stages(root, directory):
    """Check checkpoints and the external evidence of structural admission.

    Only declared rule-review evidence is interpreted as a binding. Arbitrary
    mathematical semantics is not scanned for path-shaped user dictionaries.
    This is a byte/source integrity guard, never another semantic assessment.
    """
    work = directory / "work"
    if not work.exists():
        return
    for path in work.rglob("*.json"):
        _local(root, path)
        value = _read(path)
        if not isinstance(value, dict) or value.get("stage") not in {"rules", "nodes", "explicit", "implicit"}:
            continue
        if value.get("value_sha256") != digest(value.get("value")):
            raise CacheIntegrityError("A retained production stage checksum differs")
        if value["stage"] == "nodes":
            nodes = value.get("value", {})
            execution = nodes.get("production_execution")
            if execution is None:
                continue
            if not isinstance(execution, dict) or not isinstance(execution.get("call_bindings"), list):
                raise CacheIntegrityError("Retained node execution evidence is invalid")
            bindings = [execution.get("raw_node_result"), execution.get("execution_report")]
            for call in execution["call_bindings"]:
                if not isinstance(call, dict) or call.get("status") != "returned":
                    raise CacheIntegrityError("A retained node execution task is not returned")
                bindings.extend(call.get(key) for key in ("request", "response_binding", "receipts_binding"))
            for binding in bindings:
                if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                    raise CacheIntegrityError("Retained node execution binding is invalid")
                target = _local(root, binding["path"])
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != binding["sha256"]:
                    raise CacheIntegrityError("External node execution evidence changed")
            raw = _read(_local(root, execution["raw_node_result"]["path"]))
            if raw != {key: val for key, val in nodes.items() if key != "production_execution"}:
                raise CacheIntegrityError("Node execution changed the original node outcomes")
            if _read(_local(root, execution["execution_report"]["path"])) != {
                    key: val for key, val in execution.items() if key != "execution_report"}:
                raise CacheIntegrityError("Node execution report differs from its stage declaration")
            continue
        if value["stage"] != "rules":
            continue
        packet = value.get("value", {}).get("reviewed", {}).get("source_review_packet")
        if packet is None:
            continue
        if not isinstance(packet, dict) or not isinstance(packet.get("decisions"), list):
            raise CacheIntegrityError("Retained source-structure review packet is invalid")
        for decision in packet["decisions"]:
            evidence = decision.get("evidence") if isinstance(decision, dict) else None
            if not isinstance(evidence, list) or not evidence:
                raise CacheIntegrityError("A retained structural admission decision lacks evidence")
            for binding in evidence:
                if not isinstance(binding, dict) or set(binding) != {"path", "sha256"}:
                    raise CacheIntegrityError("Retained source-structure evidence binding is invalid")
                target = _local(root, binding["path"])
                if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != binding["sha256"]:
                    raise CacheIntegrityError("External source-structure admission evidence changed")


def _node_output_scope(root, directory, *, required=False):
    """Retain unresolved extraction material without promoting it to complete.

    Production node stages declare a separately checked execution contract.
    Historical strict stages still need their original complete-node guard.
    An injected offline fixture may omit real stage files; the public source
    profile cannot. Pending earlier attempts remain visible even if a later
    strict re-extraction produces a larger accepted graph.
    """
    stages, pending = [], []
    for path in sorted((directory / "work").rglob("nodes.json")):
        value = _read(path)
        if not isinstance(value, dict) or value.get("stage") != "nodes":
            continue
        if value.get("value_sha256") != digest(value.get("value")):
            raise CacheIntegrityError("A retained node stage checksum differs")
        nodes = value["value"]
        execution = nodes.get("production_execution")
        if execution is not None:
            if (not isinstance(execution, dict) or execution.get("technical_complete") is not True
                    or execution.get("scope") not in {"complete", "partial"}):
                raise CacheIntegrityError("A node stage has no closed execution contract")
            scope = execution["scope"]
            if scope != ("partial" if nodes.get("summary", {}).get("status") != "complete"
                         or nodes.get("unresolved") else "complete"):
                raise CacheIntegrityError("Node extraction scope contradicts retained unresolved work")
        elif required:
            if nodes.get("summary", {}).get("status") != "complete" or nodes.get("unresolved"):
                raise CacheIntegrityError("A strict node stage is not complete")
            scope = "complete"
        else:
            scope = "complete"
        row = {"path": path.relative_to(root).as_posix(),
               "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
               "scope": scope, "unresolved_count": len(nodes.get("unresolved", []))}
        stages.append(row)
        if scope == "partial":
            pending.append(row)
    if required and not stages:
        raise CacheIntegrityError("Production node stages are missing")
    return {"extraction_scope": "partial" if pending else "complete",
            "node_pending_count": sum(row["unresolved_count"] for row in pending),
            "pending_node_stages": pending}


class _ModuleCompleteReextractAdapter(CompleteReextractAdapter):
    """A partial normal graph never qualifies as a complete re-extraction."""
    def __call__(self, *args, **kwargs):
        result = super().__call__(*args, **kwargs)
        if self.pipeline.profile.get("normal_source_profile") is True:
            stage = _bound(self.pipeline.root, result["stages"]["nodes"])
            nodes = stage["value"]
            execution = nodes.get("production_execution")
            if (nodes.get("summary", {}).get("status") != "complete" or nodes.get("unresolved")
                    or execution is not None and (execution.get("technical_complete") is not True
                                                  or execution.get("scope") != "complete")):
                raise ValueError("Complete re-extraction cannot use partial node production")
        return result


class _RequestJudge:
    """Attribute all attempts, including failures, to this document request.

    Shared ledgers may contain simultaneous other documents in the same host
    process. Request-cache identities provide a more precise boundary than a
    before/after count of the entire shared ledger.
    """
    def __init__(self, delegate, keys=None, lock=None):
        self.delegate = delegate
        self.keys = set() if keys is None else keys
        self.lock = threading.Lock() if lock is None else lock

    def __getattr__(self, name):
        return getattr(self.delegate, name)

    def __call__(self, task_kind, instructions, payload):
        purpose = "topology:" + self.delegate.allocation + ":" + task_kind
        key = digest({"purpose": purpose, "instructions": instructions,
                      "payload": payload, "model": self.delegate.client.model})
        with self.lock:
            self.keys.add(key)
        return self.delegate(task_kind, instructions, payload)

    def fork(self, stage_options=None, allocation=None):
        return _RequestJudge(self.delegate.fork(stage_options, allocation), self.keys, self.lock)


class SingleDocumentModule:
    """One integrated Agent, with deterministic components and shared model gate.

    Optional factories are for offline engineering checks. Production uses the
    fixed node/explicit/implicit policies and the one-pass topology workflow.
    """
    def __init__(self, root=ROOT, *, policy_path=DEFAULT_POLICY, policy=None, judge=None,
                 profile_builder=None, pipeline_factory=None, projection=None,
                 finalizer_factory=None, authorization=None):
        from .production_pipeline import load_policy, source_stage_profile, production_pipeline, formal_projection
        from .topology_client import TopologyJudge, load_config
        from .topology_workflow import TopologyWorkflow
        self.root = Path(root).resolve(strict=True)
        self.policy = copy.deepcopy(policy) if policy is not None else load_policy(
            _local(self.root, policy_path), root=self.root)
        self.authorization = copy.deepcopy(authorization) if authorization is not None else load_config(
            _local(self.root, self.policy["authorization_config"]))
        self.judge = judge if judge is not None else TopologyJudge(self.authorization, root=self.root)
        self.profile_builder = profile_builder or source_stage_profile
        self.pipeline_factory = pipeline_factory or production_pipeline
        self.projection = projection or formal_projection
        self.finalizer_factory = finalizer_factory or TopologyWorkflow
        self.agent_version = self.policy.get("agent_version", "single-document-module-v1")
        self.cache_root = storage_path(self.root, self.policy.get("cache_root", "artifacts/single-document"), project_of(self.judge))

    def fingerprint(self, project):
        from .topology_workflow import implementation_bindings
        profile = self.profile_builder(project, self.policy, root=self.root)
        programs = list(profile["program_bindings"])
        topology = implementation_bindings(self.authorization, self.root)
        prompt_names = {binding["path"] for binding in profile["prompt_bindings"]}
        prompt_names.update(self.authorization[name] for name in ("review_prompt", "repair_prompt", "value_prompt"))
        code = {binding["path"]: binding["sha256"] for binding in programs + topology
                if binding["path"] not in prompt_names}
        names = ("src/gangmu/single_document/production.py", "src/gangmu/single_document/production_pipeline.py",
                 "src/gangmu/representation/storage.py", "scripts/extract_single_document.py")
        code.update(file_digests(self.root, names))
        prompts = file_digests(self.root, tuple(sorted(prompt_names)))
        model = _public(self.judge.client.model)
        fingerprint = stable_fingerprint(source_files=project.source_files, preprocessing=project.fingerprint,
            code_files=code, prompt_files=prompts, schema_version=SCHEMA_VERSION,
            provider=model["provider"], actual_model=model["model"], agent_version=self.agent_version,
            result_parameters={"profile": _semantic_profile(profile), "model": model,
                "topology": _topology_parameters(self.authorization), "production_version": VERSION})
        return fingerprint, profile

    def _check_inputs(self, project, fingerprint):
        project.assert_unchanged()
        for key in ("code_files", "prompt_files"):
            if file_digests(self.root, tuple(fingerprint.manifest[key])) != fingerprint.manifest[key]:
                raise CacheIntegrityError("Production implementation changed during this request")

    def _resume_binding(self, profile):
        return {"profile_sha256": profile["profile_sha256"],
                "authorization_sha256": digest(self.authorization),
                "ledger_scope_sha256": digest(self.judge.ledger.scope)}

    def _verify(self, directory, record, project, fingerprint):
        try:
            if (record.get("schema_version") != VERSION + ".cache" or record.get("key") != fingerprint.key
                    or record.get("fingerprint") != fingerprint.manifest or record.get("status") != "complete"):
                raise CacheIntegrityError("Complete module cache has a different input binding")
            for relative, expected in record["artifacts"].items():
                path = _local(self.root, directory / relative)
                if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise CacheIntegrityError("A completed module artifact is missing or changed")
            retained = {name: sha for name, sha in file_digests(directory).items()
                        if name != "cache-record.json"}
            if retained != record["artifacts"]:
                raise CacheIntegrityError("Completed module artifact inventory differs")
            if _read(directory / "fingerprint.json") != {"key": fingerprint.key, "manifest": fingerprint.manifest}:
                raise CacheIntegrityError("Completed source/implementation fingerprint is missing or changed")
            _verify_retained_stages(self.root, directory)
            scope = _node_output_scope(self.root, directory, required=fingerprint.manifest[
                "result_parameters"]["profile"].get("normal_source_profile") is True)
            if (any(record["result"].get(key) != value for key, value in scope.items())
                    or record["result"].get("status") != ("partial" if scope["extraction_scope"] == "partial" else "success")):
                raise CacheIntegrityError("Cached extraction scope differs from retained node stages")
            graph = load_bundle(directory / "bundle")
            source = project.verify_graph(graph)
            if not source["structurally_valid"] or source["source_validation"] != "verified":
                raise CacheIntegrityError("Cached graph no longer matches pinned source/E2 requirements")
            if validate(graph).errors:
                raise CacheIntegrityError("Cached graph fails formal E2 validation")
            _check_all_node_dag(graph.model_dump(mode="json"))
            if digest(graph.model_dump(mode="json")) != record["graph_sha256"]:
                raise CacheIntegrityError("Cache graph digest differs")
            self._check_inputs(project, fingerprint)
            return graph
        except (OSError, KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, CacheIntegrityError):
                raise
            raise CacheIntegrityError("Completed cache validation failed; no new model call is allowed") from exc

    def _request_usage(self, before, started, request_keys=()):
        rows = [row for row in _ledger_rows(self.judge) if row.get("id") not in before
                and row.get("owner", {}).get("pid") == os.getpid() and row.get("started_at", 0) >= started
                and Path(row.get("run", "")).name in request_keys]
        return {"extraction_model_invocations_this_request": len(rows),
                "observed_tokens_this_request": sum(row.get("observed_tokens", 0) for row in rows),
                "usage_unknown_invocations_this_request": sum(not row.get("usage_known") for row in rows),
                "invocation_ids_this_request": [row["id"] for row in rows]}

    def _output(self, output, report, graph=None):
        output = _local(self.root, output)
        output.mkdir(parents=True, exist_ok=True)
        if graph is not None:
            bundle = output / "bundle"
            if bundle.exists():
                if load_bundle(bundle).model_dump(mode="json") != graph.model_dump(mode="json"):
                    raise CacheIntegrityError("Output contains a different immutable graph bundle")
            else:
                save_bundle(graph, bundle)
            report["bundle"] = bundle.relative_to(self.root).as_posix()
        requests = output / "requests"
        _local(self.root, requests)
        requests.mkdir(exist_ok=True)
        request = requests / (str(time.time_ns()) + ".json")
        _save_once(request, report)
        _atomic_json(output / "run-report.json", report)
        return report

    def run(self, project, output, *, check_cache=False, lock_timeout=60):
        """Return success/partial/cached/incomplete without accuracy evaluation."""
        started = time.time()
        fingerprint, profile = self.fingerprint(project)
        directory = _local(self.root, self.cache_root / fingerprint.key)
        output = _local(self.root, output)
        if output.is_relative_to(self.cache_root) or self.cache_root.is_relative_to(output):
            raise ValueError("Output and canonical cache must be separate project directories")
        with _lock(self.root, self.cache_root, fingerprint.key + ".lock", lock_timeout):
            with _lock(self.root, output, ".module-output.lock", lock_timeout):
                previous_output = output / "run-report.json"
                if previous_output.exists() and _read(previous_output).get("fingerprint") != fingerprint.key:
                    raise CacheIntegrityError("Output belongs to another document or Agent version")
                record_path = directory / "cache-record.json"
                _local(self.root, record_path)
                if not record_path.exists() and directory.exists() and any(directory.iterdir()):
                    raise CacheIntegrityError("Canonical cache record is missing; retained work cannot authorize re-extraction")
                record = _read_record(record_path) if record_path.exists() else None
                if record is not None and (record.get("key") != fingerprint.key
                        or record.get("fingerprint") != fingerprint.manifest):
                    raise CacheIntegrityError("Retained module work has another fingerprint")
                if record is not None and record.get("status") not in {"running", "incomplete", "complete"}:
                    raise CacheIntegrityError("Unknown module cache state")
                if record is not None and record.get("status") == "complete":
                    graph = self._verify(directory, record, project, fingerprint)
                    report = {**copy.deepcopy(record["result"]), "status": "cached", "cache_hit": True,
                              "cached_result_status": record["result"]["status"],
                              "check_cache_only": check_cache, "elapsed_seconds": time.time() - started,
                              "extraction_model_invocations_this_request": 0, "observed_tokens_this_request": 0,
                              "usage_unknown_invocations_this_request": 0, "invocation_ids_this_request": []}
                    return self._output(output, report, graph)
                if (output / "bundle").exists():
                    raise CacheIntegrityError("Output bundle has no complete canonical cache; choose a fresh output")
                if check_cache:
                    return self._output(output, {"schema_version": VERSION + ".report", "status": "incomplete",
                        "paper_id": project.paper_id, "fingerprint": fingerprint.key, "cache_hit": False,
                        "reason": "complete_cache_not_available", "check_cache_only": True,
                        "automatic_e3_publication": False, "runtime_precision_evaluation": False,
                        "extraction_model_invocations_this_request": 0, "observed_tokens_this_request": 0,
                        "elapsed_seconds": time.time() - started})
                directory.mkdir(parents=True, exist_ok=True)
                _save_once(directory / "fingerprint.json", {"key": fingerprint.key, "manifest": fingerprint.manifest})
                _save_once(directory / "resume-binding.json", self._resume_binding(profile))
                attempts = copy.deepcopy(record.get("attempts", [])) if record else []
                attempts.append({"started_at": started, "pid": os.getpid(), "resumed": record is not None})
                record = {"schema_version": VERSION + ".cache", "key": fingerprint.key,
                          "fingerprint": fingerprint.manifest, "status": "running", "attempts": attempts}
                _record(record_path, record)
                before = {row.get("id") for row in _ledger_rows(self.judge)}
                request_judge = _RequestJudge(self.judge)
                try:
                    self._check_inputs(project, fingerprint)
                    _verify_retained_stages(self.root, directory)
                    pipeline = self.pipeline_factory(profile, request_judge, root=self.root)
                    finalizer = self.finalizer_factory(self.authorization, request_judge, root=self.root)
                    strict_pipeline = pipeline.for_complete_reextraction() if hasattr(
                        pipeline, "for_complete_reextraction") else pipeline
                    reextract = _ModuleCompleteReextractAdapter(strict_pipeline, lambda: project,
                        output=directory / "work" / "complete-reextraction")
                    result = pipeline.run(project, revision_id="single-document:" + fingerprint.key[:40],
                        output=directory / "work", finalizer=finalizer, reextract=reextract)
                    if result.get("report", {}).get("completed") is not True:
                        raise ValueError("Topology finalization is technically incomplete")
                    scope = _node_output_scope(self.root, directory,
                        required=profile.get("normal_source_profile") is True)
                    _save_once(directory / "diagnostic-graph.json", result["graph"])
                    _save_once(directory / "topology-report.json", result["report"])
                    projected = self.projection(result["graph"], root=self.root)
                    formal = Graph.model_validate_json(json.dumps(projected["graph"], ensure_ascii=False))
                    _save_once(directory / "formal-projection-report.json", projected["report"])
                    source = project.verify_graph(formal)
                    _save_once(directory / "source-validation-report.json", source)
                    if not source["structurally_valid"] or source["source_validation"] != "verified":
                        raise ValueError("Final formal graph fails pinned-source/E2 validation")
                    if validate(formal).errors:
                        raise ValueError("Final formal graph fails E2 validation")
                    _check_all_node_dag(formal.model_dump(mode="json"))
                    self._check_inputs(project, fingerprint)
                    if (directory / "bundle").exists():
                        if load_bundle(directory / "bundle").model_dump(mode="json") != formal.model_dump(mode="json"):
                            raise CacheIntegrityError("Interrupted export differs from resumed final graph")
                    else:
                        save_bundle(formal, directory / "bundle")
                    data = formal.model_dump(mode="json")
                    report = {"schema_version": VERSION + ".report",
                        "status": "partial" if scope["extraction_scope"] == "partial" else "success", "cache_hit": False,
                        "paper_id": project.paper_id, "fingerprint": fingerprint.key,
                        "agent_version": self.agent_version, "cache": directory.relative_to(self.root).as_posix(),
                        "graph_sha256": digest(data), "snapshot_id": formal.snapshot_id,
                        "nodes": len(formal.units), "relations": len(formal.relations),
                        "dependency_relations": sum(r.type in {"statement_dependency", "proof_dependency"} for r in formal.relations),
                        "formal_e2_eligible": True, "dag": True, "source_validation": "verified",
                        "projection_report": (directory / "formal-projection-report.json").relative_to(self.root).as_posix(),
                        "topology_report": (directory / "topology-report.json").relative_to(self.root).as_posix(),
                        "runtime_precision_evaluation": False, "automatic_e3_publication": False,
                        "mathematical_correctness_certified": False, "check_cache_only": False,
                        **scope,
                        "resumed_checkpoints": attempts[-1]["resumed"],
                        **self._request_usage(before, started, request_judge.keys), "elapsed_seconds": time.time() - started}
                    saved_report = directory / "production-report.json"
                    retained_report = _read(saved_report) if saved_report.exists() else report
                    if (retained_report.get("fingerprint") != fingerprint.key
                            or retained_report.get("graph_sha256") != digest(data)
                            or retained_report.get("status") != report["status"]
                            or any(retained_report.get(key) != value for key, value in scope.items())):
                        raise CacheIntegrityError("Retained production report differs from resumed formal output")
                    _save_once(saved_report, retained_report)
                    artifacts = {name: sha for name, sha in file_digests(directory).items()
                                 if name != "cache-record.json"}
                    record.update(status="complete", graph_sha256=digest(data), result=retained_report, artifacts=artifacts,
                                  completed_at=time.time())
                    _record(record_path, record)
                    graph = self._verify(directory, _read_record(record_path), project, fingerprint)
                    return self._output(output, report, graph)
                except Exception as exc:
                    if record.get("status") == "complete":
                        # Materializing another output cannot turn a valid
                        # immutable cache into permission for another extraction.
                        raise
                    # Keep source/model text and credentials out of public errors.
                    record.update(status="incomplete", error_type=type(exc).__name__, finished_at=time.time())
                    _record(record_path, record)
                    report = {"schema_version": VERSION + ".report", "status": "incomplete", "cache_hit": False,
                        "paper_id": project.paper_id, "fingerprint": fingerprint.key,
                        "cache": directory.relative_to(self.root).as_posix(), "error_type": type(exc).__name__,
                        "reason": "retained_work_requires_inspection" if isinstance(exc, CacheIntegrityError) else "production_not_complete",
                        "runtime_precision_evaluation": False, "automatic_e3_publication": False,
                        **self._request_usage(before, started, request_judge.keys), "elapsed_seconds": time.time() - started}
                    return self._output(output, report)


def main(argv=None):
    """CLI for real extraction or no-call verification of a completed cache."""
    import argparse
    parser = argparse.ArgumentParser(description="从LaTeX源码构建完整单文档六文件图谱；评估不属于正式流程。")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--source", type=Path)
    inputs.add_argument("--artifact", type=Path)
    parser.add_argument("--paper", required=True)
    parser.add_argument("--arxiv")
    parser.add_argument("--main")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path(DEFAULT_POLICY))
    parser.add_argument("--check-cache", action="store_true", help="只核验已有完整缓存，绝不调用模型")
    parser.add_argument("--lock-timeout", type=float, default=60)
    args = parser.parse_args(argv)
    if args.source and (not args.arxiv or not args.main):
        parser.error("--source requires --main and --arxiv (including the source version)")
    if args.lock_timeout < 0:
        parser.error("--lock-timeout must be nonnegative")
    try:
        project = (SourceProject.from_source(args.source if args.source.is_absolute() else ROOT / args.source,
                      args.main, args.paper, args.arxiv) if args.source else SourceProject.from_artifact(
                      args.artifact if args.artifact.is_absolute() else ROOT / args.artifact, paper_id=args.paper))
        module = SingleDocumentModule(ROOT, policy_path=args.policy)
        report = module.run(project, args.output, check_cache=args.check_cache, lock_timeout=args.lock_timeout)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return 0 if report["status"] in {"success", "partial", "cached"} else 1
    except Exception as exc:
        print(json.dumps({"status": "incomplete", "error_type": type(exc).__name__,
            "reason": "cache_integrity_refusal" if isinstance(exc, CacheIntegrityError) else "initialization_or_input_failure",
            "extraction_model_invocations_this_request": 0}, ensure_ascii=False))
        return 1
