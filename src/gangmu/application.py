"""Small command-line wrapper for the standalone single-document pipeline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

from gangmu.representation.graph import validate
from gangmu.representation.model import digest
from gangmu.representation.storage import load_bundle
from gangmu.runtime.config import load_config
from gangmu.runtime.inputs import UserSource, register_source
from gangmu.runtime.session import (
    SessionTopologyJudge,
    make_session_config,
    validate_session_config,
)
from gangmu.single_document.cache import _atomic_json
from gangmu.single_document.production import SingleDocumentModule
from gangmu.single_document.source import SourceProject


ROOT = Path(__file__).resolve().parents[2]


class ApplicationError(ValueError):
    """An actionable command-line error."""


def _read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _json(value):
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def _source(args):
    source = args.source.resolve(strict=True)
    if args.main:
        main = args.main
    elif args.format == "markdown":
        main = "main.md"
    elif args.format == "latex":
        main = "main.tex"
    elif (source / "main.tex").is_file():
        main = "main.tex"
    elif (source / "main.md").is_file():
        main = "main.md"
    else:
        raise ApplicationError("Could not infer the main source file; pass --main")
    source_format = args.format
    if source_format is None:
        source_format = (
            "markdown"
            if main.lower().endswith((".md", ".markdown"))
            or ((source / "main.md").is_file() and not (source / "main.tex").is_file())
            else "latex"
        )
    return UserSource(source, args.paper, args.version, main_file=main, source_format=source_format)


def _prepare(args):
    if args.source is None:
        raise ApplicationError("prepare requires --source")
    source = _source(args)
    registered = register_source(ROOT, source)
    result = {
        "status": "prepared",
        "paper": source.paper_id,
        "version": source.version,
        "format": source.source_format,
        "artifact": str(registered.artifact_dir.relative_to(ROOT)),
        "source_fingerprint": registered.fingerprint,
        "reused": registered.reused,
        "model_calls": 0,
    }
    session = ROOT / "runs" / (args.session or "prepare")
    _atomic_json(session / "prepared.json", result)
    return result


def _session(args, route):
    session_id = args.session or args.paper
    path = ROOT / "runs" / session_id / "session.json"
    if path.exists():
        config = validate_session_config(_read(path))
        if config["provider"] != route.provider.alias or config["model"] != route.model.name:
            raise ApplicationError("The saved session uses a different provider or model")
        if args.max_tokens is not None and args.max_tokens != config["budget"]["max_total_tokens"]:
            raise ApplicationError("The saved session has a fixed token ceiling")
        return config
    if args.max_tokens is None or args.max_tokens < 1:
        raise ApplicationError("A new extraction session requires --max-tokens")
    config = make_session_config(
        session_id,
        max_total_tokens=args.max_tokens,
        provider=route.provider.alias,
        model=route.model.name,
    )
    _atomic_json(path, config)
    return config


def _project(args):
    if args.source is not None:
        source = _source(args)
        registered = register_source(ROOT, source)
        return SourceProject.from_artifact(registered.artifact_dir, source.paper_id), registered
    if args.artifact is None:
        raise ApplicationError("extract requires --source or --artifact")
    artifact = args.artifact.resolve(strict=True)
    return SourceProject.from_artifact(artifact, args.paper), None


def _extract(args):
    project, registered = _project(args)
    runtime = load_config(ROOT, require_private=True)
    route = runtime.require_route("single_document", implemented_protocols={"openai_responses"})
    config = _session(args, route)
    policy = _read(ROOT / "configs/single-document.json")
    policy["node_wire_options"]["model"] = route.model.name
    authorization = _read(ROOT / "configs/topology.json")
    authorization["provider"] = route.provider.alias
    authorization["model"] = route.model.name
    judge = SessionTopologyJudge(
        config,
        root=ROOT,
        runtime=runtime,
        stage_options=authorization["model_options"],
    )
    module = SingleDocumentModule(
        ROOT,
        policy=policy,
        authorization=authorization,
        judge=judge,
    )
    fingerprint, _ = module.fingerprint(project)
    output = (args.output.resolve() if args.output else ROOT / "results" / project.paper_id / fingerprint.key)
    report = module.run(project, output, check_cache=args.check_cache)
    result = {
        **report,
        "paper": project.paper_id,
        "source_format": project.source_format,
        "output": str(output.relative_to(ROOT)),
        "model": route.model.name,
        "session": config["session_id"],
        "source_fingerprint": registered.fingerprint if registered else project.artifact_tree_sha256,
    }
    _atomic_json(ROOT / "runs" / config["session_id"] / "summary.json", result)
    return result


def _inspect(args):
    bundle = args.bundle.resolve(strict=True)
    graph = load_bundle(bundle)
    report = validate(graph)
    return {
        "status": "valid" if report.structurally_valid else "invalid",
        "bundle": str(bundle.relative_to(ROOT)) if bundle.is_relative_to(ROOT) else str(bundle),
        "graph_sha256": digest(graph.model_dump(mode="json")),
        "documents": len(graph.documents),
        "contexts": len(graph.contexts),
        "nodes": len(graph.units),
        "relations": len(graph.relations),
        "validation": report.to_dict(),
    }


def parser():
    result = argparse.ArgumentParser(description="Extract a source-preserving single-document mathematical graph")
    commands = result.add_subparsers(dest="command", required=True)

    init = commands.add_parser("init", help="Create private configuration and local directories")
    init.add_argument("--force", action="store_true")

    for name in ("prepare", "extract"):
        command = commands.add_parser(name)
        command.add_argument("--source", type=Path)
        command.add_argument("--artifact", type=Path, help="Existing registered source artifact")
        command.add_argument("--paper", required=True)
        command.add_argument("--version", default="v1")
        command.add_argument("--main")
        command.add_argument("--format", choices=("latex", "markdown"))
        command.add_argument("--session")
        if name == "extract":
            command.add_argument("--max-tokens", type=int)
            command.add_argument("--output", type=Path)
            command.add_argument("--check-cache", action="store_true")

    inspect = commands.add_parser("inspect", help="Validate one exported graph bundle")
    inspect.add_argument("bundle", type=Path)
    return result


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "init":
            (ROOT / "data").mkdir(exist_ok=True)
            (ROOT / "artifacts").mkdir(exist_ok=True)
            (ROOT / "runs").mkdir(exist_ok=True)
            (ROOT / "results").mkdir(exist_ok=True)
            config = ROOT / "config.toml"
            if not config.exists() or args.force:
                config.write_bytes((ROOT / "config.example.toml").read_bytes())
            result = {"status": "initialized", "configuration": "config.toml"}
        elif args.command == "prepare":
            result = _prepare(args)
        elif args.command == "extract":
            result = _extract(args)
        else:
            result = _inspect(args)
        print(_json(result))
        return 0
    except (ApplicationError, OSError, ValueError, KeyError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 2
