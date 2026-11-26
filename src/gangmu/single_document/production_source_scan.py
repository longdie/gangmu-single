"""Source-only structural admission for the composed document producer.

This is a production decision before mathematical node extraction, never an
independent precision/coverage assessment.  The historical rule envelope and
all blockers remain intact.  Only the two dispositions already admitted by
the frozen v3 resolver can discharge a blocker; missing support content,
unknown activity, unsupported obligations and technical failures stay pending.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re

from gangmu.runtime.layout import project_of, storage_path
from gangmu.representation.model import digest
from .cache import _atomic_json
from . import rule_nodes_source_control_v1 as control
from .rule_scan_review_v3 import VERSION as REVIEW_VERSION, scan_obligations

ROOT = Path(__file__).resolve().parents[3]
VERSION = "gangmu.single-document.production-source-scan.v1"
PROMPT = ROOT / "prompts/single_document/source_scan_v1.md"
MAX_PAYLOAD_BYTES = 120_000
NEIGHBOR_CHARS = 4_000
_HASH = re.compile(r"[0-9a-f]{64}\Z")


def _local_root(root):
    root = Path(root).absolute()
    if any(path.is_symlink() for path in (root, *root.parents)):
        raise ValueError("Production source-scan paths must not contain symlinks")
    root = root.resolve(strict=True)
    # The frozen v3 evidence resolver has one repository-owned trust root.
    # Test roots may be subdirectories of it, but cannot widen that root.
    if not root.is_relative_to(ROOT):
        raise ValueError("Production source-scan artifacts must stay inside the repository")
    return root


def _record(path):
    path = Path(path).absolute()
    if (any(item.is_symlink() for item in (path, *path.parents))
            or not path.resolve().is_relative_to(ROOT) or not path.is_file()):
        raise ValueError("Source-scan evidence needs a local regular file")
    return {"path": path.relative_to(ROOT).as_posix(),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def _save_once(path, value):
    if path.is_symlink():
        raise ValueError("Source-scan artifacts cannot be symlinks")
    if path.exists():
        if json.loads(path.read_bytes()) != value:
            raise ValueError("An immutable production source-scan artifact changed")
    else:
        _atomic_json(path, value)


@contextmanager
def _locked(directory):
    if any(item.is_symlink() for item in (directory, *directory.parents)):
        raise ValueError("Source-scan cache cannot contain symlinks")
    directory.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(directory / ".source-scan.lock", os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW, 0o600)
    with os.fdopen(descriptor, "a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _allowed(item):
    if item.get("kind") == "unheaded_qed_candidate_full_boundary_review":
        return "complete_original_proof_boundary"
    if (item.get("kind") == "named_heading_input_expansion_unresolved"
            and item.get("status") == "system_input_expected"
            and item.get("activity") == "active" and not item.get("conditional_scope")):
        return "non_task_support_input"
    return None


def _span(span):
    return {key: span[key] for key in ("file", "char_start", "char_end", "occurrence")
            if span.get(key) is not None}


def _material(project, item, envelope, identity):
    sources = []
    for number, original in enumerate(item["source_spans"]):
        span = project.normalize_span(_span(original))
        text = project._read_text(span["file"])
        start, end = span["char_start"], span["char_end"]
        sources.append({"source_id": "body:" + str(number), "role": "complete_obligation_body",
                        "span": span, "text": project.resolve_span(span)})
        # Occurrence marks apply to the proposed body; surrounding file text
        # may lie outside a statically traversed occurrence, so do not claim it.
        for role, left, right in (("before", max(0, start - NEIGHBOR_CHARS), start),
                                  ("after", end, min(len(text), end + NEIGHBOR_CHARS))):
            if left < right:
                neighbor = {"file": span["file"], "char_start": left, "char_end": right}
                sources.append({"source_id": role + ":" + str(number), "role": role + "_boundary_context",
                                "span": neighbor, "text": project.resolve_span(neighbor)})
    identifier = control._value_sha256(item)
    expected = _allowed(item)
    payload = {"schema_version": VERSION + ".input", "binding": copy.deepcopy(identity),
               "paper_id": project.paper_id, "source_fingerprint": copy.deepcopy(project.fingerprint),
               "raw_wrapper_value_sha256": control._value_sha256(envelope),
               "obligation_value_sha256": identifier, "obligation": copy.deepcopy(item),
               "allowed_positive_disposition": expected, "sources": sources,
               "full_original_body_included": True,
               "external_support_content_included": False,
               "task_scope": "production_source_structure_before_node_extraction",
               "independent_from_rule_discovery": True,
               "independent_precision_evaluation": False, "expert_gold": False}
    fields = {"obligation_value_sha256": {"type": "string", "enum": [identifier]},
              "disposition": {"type": "string", "enum": [expected, "unresolved"]},
              "reason": {"type": "string", "minLength": 1},
              "boundary_start_complete": {"type": "boolean"},
              "boundary_end_complete": {"type": "boolean"},
              "activity_confirmed": {"type": "boolean"},
              "support_content_checked": {"type": "boolean"},
              "evidence_quotes": {"type": "array", "items": {
                  "type": "object", "properties": {
                      "source_id": {"type": "string", "enum": [row["source_id"] for row in sources]},
                      "quote": {"type": "string", "minLength": 1}},
                  "required": ["source_id", "quote"], "additionalProperties": False}}}
    payload["response_schema"] = {"type": "object", "properties": fields,
                                  "required": list(fields), "additionalProperties": False}
    return payload


def _decision(item, payload, response):
    fields = set(payload["response_schema"]["properties"])
    if (not isinstance(response, dict) or set(response) != fields
            or response["obligation_value_sha256"] != payload["obligation_value_sha256"]
            or response["disposition"] not in {_allowed(item), "unresolved"}
            or not isinstance(response["reason"], str) or not response["reason"].strip()
            or any(type(response[key]) is not bool for key in (
                "boundary_start_complete", "boundary_end_complete", "activity_confirmed", "support_content_checked"))
            or not isinstance(response["evidence_quotes"], list)):
        raise ValueError("Source-structure response does not match its exact obligation")
    sources = {row["source_id"]: row for row in payload["sources"]}
    selected = set()
    for quote in response["evidence_quotes"]:
        if (not isinstance(quote, dict) or set(quote) != {"source_id", "quote"}
                or quote["source_id"] not in sources or not isinstance(quote["quote"], str)
                or not quote["quote"].strip() or quote["quote"] not in sources[quote["source_id"]]["text"]):
            raise ValueError("Source-structure evidence must quote supplied exact source")
        selected.add(quote["source_id"])
    if response["disposition"] == "unresolved":
        return None, "model_unresolved"
    if not selected or not response["activity_confirmed"]:
        return None, "positive_without_bound_source_or_known_activity"
    if _allowed(item) == "complete_original_proof_boundary":
        if (not response["boundary_start_complete"] or not response["boundary_end_complete"]
                or any(row["source_id"] not in selected for row in payload["sources"]
                       if row["role"] == "complete_obligation_body")):
            return None, "complete_proof_boundary_not_confirmed"
    elif (not payload["external_support_content_included"] or not response["support_content_checked"]):
        # A system-library name and an import line do not establish the unseen
        # file's content or exclude hidden mathematical task definitions.
        return None, "external_support_content_unavailable"
    return {"obligation_value_sha256": payload["obligation_value_sha256"],
            "disposition": response["disposition"], "reason": response["reason"],
            "source_spans": copy.deepcopy(item["source_spans"])}, None


def prepare_rules(project, judge, *, root=ROOT, profile_sha256=None):
    """Prepare exact raw rules and, when possible, the existing v3 review.

    The injected judge is called at most once per eligible obligation. Durable
    raw responses (including negative/invalid replies) are reused on restart.
    Supplier retry/cache accounting remains the injected judge's responsibility.
    Any pending obligation returns a non-ready reviewed envelope; the caller
    must not launch node production with it. No TeX, node addition or metric
    evaluation is performed here.
    """
    root = _local_root(root)
    prompt_bytes = PROMPT.read_bytes()
    implementation_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    profile_sha256 = profile_sha256 or digest({"version": VERSION, "code": implementation_sha,
                                              "prompt": hashlib.sha256(prompt_bytes).hexdigest()})
    if not isinstance(profile_sha256, str) or not _HASH.fullmatch(profile_sha256):
        raise ValueError("Production source scan needs a SHA256 profile identity")
    envelope = control.prepare_source_control(project)
    obligations = scan_obligations(envelope["metadata"])
    identity = {"schema_version": VERSION, "profile_sha256": profile_sha256,
                "source_fingerprint_sha256": digest(project.fingerprint),
                "raw_wrapper_value_sha256": control._value_sha256(envelope),
                "implementation_sha256": implementation_sha,
                "prompt_sha256": hashlib.sha256(prompt_bytes).hexdigest()}
    directory = storage_path(root, "artifacts/single-document-source-control", project_of(judge)) / profile_sha256 / digest(identity)
    with _locked(directory):
        _save_once(directory / "identity.json", identity)
        _save_once(directory / "raw-envelope.json", envelope)
        decisions, pending, attempts = [], [], []
        for item in obligations:
            identifier = control._value_sha256(item)
            allowed = _allowed(item)
            if allowed is None:
                pending.append({"obligation_value_sha256": identifier,
                                "reason": "unsupported_or_unknown_activity_obligation"})
                continue
            payload = _material(project, item, envelope, identity)
            call_dir = directory / "calls" / identifier
            call_dir.mkdir(parents=True, exist_ok=True)
            _save_once(call_dir / "request.json", {"instructions": prompt_bytes.decode("utf-8"), "payload": payload})
            response_path, failure_path = call_dir / "response.json", call_dir / "failure.json"
            if response_path.exists():
                response = json.loads(response_path.read_bytes())
            elif failure_path.exists():
                failure_record = _record(failure_path)
                failure = json.loads(failure_path.read_bytes())
                if failure.get("binding") != identity:
                    raise ValueError("Retained source-scan failure belongs to another request")
                pending.append({"obligation_value_sha256": identifier,
                                "reason": failure["reason"], "failure": failure_record})
                continue
            elif len(json.dumps(payload, ensure_ascii=False).encode()) > MAX_PAYLOAD_BYTES:
                failure = {"schema_version": VERSION + ".failure", "binding": identity,
                           "reason": "full_source_payload_exceeds_limit_no_truncation", "supplier_call_attempted": False}
                _save_once(failure_path, failure)
                pending.append({"obligation_value_sha256": identifier, "reason": failure["reason"],
                                "failure": _record(failure_path)})
                continue
            else:
                try:
                    response = judge("production-source-structure", prompt_bytes.decode("utf-8"), payload)
                    _save_once(response_path, response)
                except Exception as error:
                    failure = {"schema_version": VERSION + ".failure", "binding": identity,
                               "reason": "technical_judgment_failure", "error_type": type(error).__name__,
                               "supplier_call_attempted": True}
                    _save_once(failure_path, failure)
                    pending.append({"obligation_value_sha256": identifier, "reason": failure["reason"],
                                    "failure": _record(failure_path)})
                    continue
            request_record, response_record = _record(call_dir / "request.json"), _record(response_path)
            attempts.append({"obligation_value_sha256": identifier, "request": request_record, "response": response_record})
            try:
                decision, reason = _decision(item, payload, response)
            except (ValueError, TypeError, KeyError) as error:
                decision, reason = None, "invalid_response_" + type(error).__name__
            if decision is None:
                pending.append({"obligation_value_sha256": identifier, "reason": reason,
                                "request": request_record, "response": response_record})
            else:
                decision["evidence"] = [request_record, response_record]
                decisions.append(decision)
        packet = None
        if obligations and not pending:
            packet = {"schema_version": REVIEW_VERSION, "paper_id": project.paper_id,
                      "source_fingerprint": copy.deepcopy(project.fingerprint),
                      "raw_wrapper_value_sha256": control._value_sha256(envelope),
                      "independent_source_review": True, "decisions": decisions,
                      "production_source_structure": True, "independent_precision_evaluation": False,
                      "expert_gold": False,
                      "independence_scope": "A separate source-structure judgment of rule discoveries before mathematical node production; no precision assessor or second node/edge evaluation."}
            _save_once(directory / "review-packet.json", packet)
        reviewed = control.resolve_source_control_review(project, envelope, packet)
        summary = {"schema_version": VERSION + ".result", "binding": identity,
                   "raw_envelope": _record(directory / "raw-envelope.json"),
                   "review_packet": _record(directory / "review-packet.json") if packet else None,
                   "obligations": len(obligations), "eligible_positive_decisions": len(decisions),
                   "pending": pending, "attempts": attempts,
                   "completed": reviewed["metadata"]["scan_ready_for_discovery_acceptance"],
                   "pending_blocks_node_production": True, "mathematical_nodes_added": 0,
                   "independent_precision_evaluation": False, "expert_gold": False}
        _save_once(directory / "reviewed-envelope.json", reviewed)
        _save_once(directory / "source-scan-report.json", summary)
        project.assert_unchanged()
        return {"envelope": envelope, "reviewed": reviewed}
