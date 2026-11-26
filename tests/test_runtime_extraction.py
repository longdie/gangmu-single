"""Offline provider contract check of the complete production E4 composition.

The synthetic response selects an original theorem; it makes no claim about
model accuracy. Real source admission, both node rounds, topology finalization,
source verification and authenticated cache reuse remain in the execution.
"""
from io import BytesIO
import json
from pathlib import Path
import shutil
import tempfile
import os
import subprocess
import sys

from gangmu.representation.storage import load_bundle
from gangmu.runtime.config import Credentials, ModelConfig, ProviderConfig, RuntimeConfig
from gangmu.runtime.inputs import UserSource, register_source
from gangmu.runtime.session import SessionTopologyJudge, make_session_config
from gangmu.single_document.production import SingleDocumentModule
from gangmu.single_document.source import SourceProject


class _CompletedResponse(BytesIO):
    status = 200
    headers = {"Content-Type": "text/event-stream"}

    def __init__(self, value):
        response = {"id": "resp_offline_e4", "object": "response", "status": "completed",
            "output": [{"type": "message", "role": "assistant", "status": "completed",
                "content": [{"type": "output_text", "text": json.dumps(value)}]}],
            "usage": {"input_tokens": 20, "output_tokens": 10, "total_tokens": 30}}
        super().__init__(("event: response.completed\ndata: " + json.dumps({
            "type": "response.completed", "sequence_number": 0, "response": response}) + "\n\n").encode())

    def set_read_timeout(self, remaining):
        assert remaining > 0


def test_actual_single_document_rounds_topology_and_authenticated_cache():
    repository = Path(__file__).resolve().parents[1]
    work = repository / ".work"
    work.mkdir(exist_ok=True)
    # Source-control evidence has a deliberately repository-local trust root.
    with tempfile.TemporaryDirectory(prefix="offline-production-", dir=work) as temporary:
        root = Path(temporary)
        for name in ("src", "prompts", "configs"):
            shutil.copytree(repository / name, root / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "goals"))
        (root / "scripts").mkdir()
        shutil.copy2(repository / "scripts/extract_single_document.py", root / "scripts")
        # Import the copied modules in a fresh interpreter so their source
        # trust root and __file__ bindings are the workspace being verified.
        result = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(root)],
            cwd=root, env={**os.environ, "PYTHONPATH": str(root / "src")},
            capture_output=True, text=True, timeout=60)
        assert result.returncode == 0, result.stdout + result.stderr


def _exercise(root):
    source = root / "input"
    source.mkdir()
    original = r"""\documentclass{article}
\newtheorem{theorem}{Theorem}
\begin{document}
This paper considers elementary finite objects.

\begin{theorem}
Every zero module is finite.
\end{theorem}

This concluding paragraph adds no further mathematical claim.
\end{document}
"""
    (source / "main.tex").write_text(original)
    registered = register_source(root, UserSource(source, "offline-theorem", "v1"))
    project = SourceProject.from_artifact(registered.artifact_dir, "offline-theorem")
    policy = json.loads((root / "configs/single-document.json").read_bytes())
    authorization = json.loads((root / "configs/topology.json").read_bytes())
    runtime = RuntimeConfig(1, {"yxai": ProviderConfig("yxai", "https://example.invalid/v1", "openai_responses")},
        {"single_document": ModelConfig("single_document", "yxai", "gpt-6.1-sol")}, {},
        Credentials({"yxai": "synthetic-offline-credential"}, {"yxai": "test"}), root)
    observed = []

    def transport(request, **_):
        body = json.loads(request.data)
        payload = json.loads(body["input"][1]["content"])
        if payload.get("round") == 1:
            observed.append("round1")
            value = {"decisions": [{"candidate_id": item["candidate_id"],
                "decision": "accept", "category": "statement", "label": "Theorem 1",
                "field": None, "subfield": None, "reason": "Select the complete source theorem.",
                "context": {"description": "The theorem is self-contained.", "self_contained": True,
                    "selectors": [], "assumptions": None, "symbols": None}}
                for item in payload["candidates"]], "context_requests": []}
        elif payload.get("round") == 2:
            observed.append("round2")
            value = {"chunk_id": payload["chunk_id"], "decision": "no_supplement", "nodes": [],
                "reason": "The source paragraphs add no further mathematical node.", "context_requests": []}
        elif payload.get("schema_version", "").endswith(".node-value"):
            observed.append("node-value")
            value = {"decisions": [{"node_id": item["id"], "verdict": "keep",
                "quote": item["original_text"][0], "reason": "Retain the meaningful theorem."}
                for item in payload["nodes"]]}
        else:
            raise AssertionError("Unexpected synthetic provider task: " + str(payload.keys()))
        return _CompletedResponse(value)

    config = make_session_config("offline-extraction", max_total_tokens=1_000_000)
    judge = SessionTopologyJudge(config, root=root, runtime=runtime,
        stage_options=authorization["model_options"], transport=transport)
    judge.ledger.initialize()
    module = SingleDocumentModule(root, policy=policy, authorization=authorization, judge=judge)
    output = root / "results"
    report = module.run(project, output)
    assert report["status"] == "success", report
    assert set(observed) == {"round1", "round2", "node-value"}
    assert report["source_validation"] == "verified" and report["dag"] is True
    graph = load_bundle(output / "bundle")
    assert len(graph.units) == 1
    assert "Every zero module is finite." in "\n".join(graph.units[0].original_text)
    assert len(graph.relations) == 0
    stages = list((root / report["cache"] / "work").rglob("nodes.json"))
    assert len(stages) == 1
    execution = json.loads(stages[0].read_bytes())["value"]["production_execution"]
    assert execution["technical_complete"] is True and execution["scope"] == "complete"
    calls_before = list(observed)
    cached = module.run(project, output, check_cache=True)
    assert cached["status"] == "cached" and observed == calls_before
    assert cached["extraction_model_invocations_this_request"] == 0
    assert judge.ledger.snapshot()["supplier_starts"] == len(observed)


if __name__ == "__main__":
    _exercise(Path(sys.argv[1]))
