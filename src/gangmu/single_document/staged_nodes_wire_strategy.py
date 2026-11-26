"""Two-round node strategy with complete-source menus and bounded API packets.

The frozen v4 host still validates source selectors, Context and node content.
Only presentation and grouping change. Round two preserves each complete gap
between explicit anchors/sections; it never splits a multi-paragraph task to
make a request fit. Oversize complete tasks remain explicitly unresolved.
"""
from __future__ import annotations

import copy
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path

from gangmu.representation.model import digest
from . import staged_nodes as base
from . import staged_nodes_context_strategy as context
from .source import SourceError

VERSION = "staged-nodes-v5-wire-bounded"


@dataclass(frozen=True)
class WireOptions:
    max_wire_bytes: int = 160000
    round1_max_items: int = 12
    model: str = "gpt-6.1-sol"
    reasoning_effort: str = "high"
    max_output_tokens: int = 12000
    response_transport: str = "sse"

    @classmethod
    def parse(cls, value=None):
        result = cls(**(value or {}))
        if (type(result.max_wire_bytes) is not int or result.max_wire_bytes < 1024
                or type(result.round1_max_items) is not int or not 1 <= result.round1_max_items <= 12
                or type(result.max_output_tokens) is not int or result.max_output_tokens < 16
                or not isinstance(result.model, str) or not result.model.strip()
                or result.reasoning_effort not in {"low", "medium", "high", "xhigh"}
                or result.response_transport not in {"sse", "json"}):
            raise SourceError("Invalid bounded node packet options")
        return result


def compact_payload(original):
    """Remove geometry from the model menu; preserve full text and ID authority."""
    payload = copy.deepcopy(original)
    full_targets = {}
    for name, key in (("candidates", "candidate_id"), ("paragraphs", "source_id")):
        for row in payload.get(name, []):
            identifier = row[key]
            if identifier in full_targets:
                raise SourceError("Duplicate complete task source ID")
            texts = row.get("original_text")
            if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
                raise SourceError("Complete task requires its original text array")
            full_targets[identifier] = (name, texts)
    menu, seen = [], set()
    for row in payload["context_menu"]:
        identifier = row["source_id"]
        if identifier in seen:
            raise SourceError("Duplicate Context menu source ID")
        seen.add(identifier)
        item = {"source_id": identifier, "kind": row["kind"]}
        if "original_text" in row:
            texts = row["original_text"]
            if not isinstance(texts, list) or not texts or any(not isinstance(t, str) for t in texts):
                raise SourceError("Context packet requires its complete original text array")
            if identifier in full_targets and full_targets[identifier][1] != texts:
                raise SourceError("Same source ID has different complete text")
            item["original_text"] = copy.deepcopy(texts)
        else:
            collection = row.get("full_text_in")
            if identifier not in full_targets or full_targets[identifier][0] != collection:
                raise SourceError("Context alias has no complete task text in this packet")
            item["full_text_in"] = collection
        menu.append(item)
    payload["context_menu"] = menu
    return payload


def wire_bytes(instructions, payload, options):
    """Match SourceResponsesClient's full API body, including both schemas."""
    body = {
        "model": options.model, "store": False,
        "reasoning": {"effort": options.reasoning_effort},
        "max_output_tokens": options.max_output_tokens,
        "input": [{"role": "developer", "content": instructions},
                  {"role": "user", "content": json.dumps(payload, ensure_ascii=False, allow_nan=False)}],
    }
    if options.response_transport == "sse":
        body["stream"] = True
    if payload.get("response_schema") is not None:
        body["text"] = {"format": {"type": "json_schema", "name": "source_evaluation",
                                   "schema": payload["response_schema"], "strict": True}}
    return json.dumps(body, ensure_ascii=False, allow_nan=False,
                      sort_keys=options.response_transport == "sse").encode("utf-8")


class _WireJudge:
    def __init__(self, judge, options):
        self.judge, self.options, self.records = judge, options, []

    def __call__(self, kind, instructions, logical):
        actual = compact_payload(logical)
        packet = wire_bytes(instructions, actual, self.options)
        record = {"task_kind": kind, "logical_payload_sha256": digest(logical),
                  "actual_payload_sha256": digest(actual), "wire_bytes": len(packet),
                  "wire_sha256": hashlib.sha256(packet).hexdigest(),
                  "source_context_followup": actual["source_context_followup"],
                  "complete_task_ids": [r.get("candidate_id", r.get("source_id"))
                                        for r in actual.get("candidates", actual.get("paragraphs", []))],
                  "status": "ready"}
        self.records.append(record)
        if len(packet) > self.options.max_wire_bytes:
            record["status"] = "oversize_complete_task_unresolved_no_call"
            raise SourceError("Complete node task or required source followup exceeds the registered wire bound")
        try:
            answer = self.judge(kind, instructions, actual)
        except Exception:
            record["status"] = "judge_failed"
            raise
        record["status"] = "returned"
        return answer


class _BoundedPipeline(context._ContextPipeline):
    def __init__(self, project, rule_result, judge, options, wire_options):
        self.wire_options = WireOptions.parse(wire_options)
        self.wire_judge = _WireJudge(judge, self.wire_options)
        super().__init__(project, rule_result, self.wire_judge, options)
        self.binding = {
            **self.binding, "producer_version": VERSION,
            "producer_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "context_producer_code_sha256": hashlib.sha256(Path(context.__file__).read_bytes()).hexdigest(),
            "wire_options": asdict(self.wire_options),
            "round2_grouping": "pack_complete_source_gaps_between_explicit_anchors_or_sections",
            "base_call_input_hashes_are_logical_packets": True,
        }
        self.group_records = []

    def _packet(self, rows, kind):
        ids = [r["source_id"] for r in rows]
        payload = {"source_fingerprint": self.project.fingerprint,
                   "context_menu": self._context_menu(rows),
                   "producer_binding": self.binding, "context_catalog": self._catalog(),
                   "source_context_followup": 0}
        if kind == base.TASK_ROUND1:
            payload.update(round=1, batch_id="round1:" + digest(ids)[:24], fixed_complete_source=True,
                           candidates=[{**copy.deepcopy(c), "accepted": False}
                                       for c in self.candidates if c["candidate_id"] in ids])
        else:
            payload.update(round=2, chunk_id="round2:" + digest(ids)[:24], paragraphs=copy.deepcopy(rows),
                           existing_units=[{"unit_id": u["id"], "category": u["category"],
                                            "label": u["label"], "sources": u["sources"]}
                                           for u in self.units.values()])
        payload["response_schema"] = base.response_schema(kind, payload)
        return compact_payload(payload)

    def _groups(self, rows, limit, max_items=None):
        if not rows:
            return
        kind = base.TASK_ROUND1 if rows[0]["kind"] == "rule_candidate" else base.TASK_ROUND2
        if kind == base.TASK_ROUND2:
            # A body task can span blank-line paragraphs. Keep its entire source
            # gap selectable; Context-only overlap cannot replace node selectors.
            group, assigned = [], set()
            for gap in self.gaps:
                whole = [r for r in rows if gap["char_start"] <= r["reading_ranges"][0]["char_start"]
                         and r["reading_ranges"][-1]["char_end"] <= gap["char_end"]]
                if not whole:
                    continue
                identifiers = {r["source_id"] for r in whole}
                if identifiers & assigned:
                    raise SourceError("A body paragraph belongs to more than one complete gap")
                assigned.update(identifiers)
                combined = [*group, *whole]
                if group and (sum(self._size(r) for r in combined) > limit or
                        len(wire_bytes(self.prompts[kind], self._packet(combined, kind),
                                       self.wire_options)) > self.wire_options.max_wire_bytes):
                    yield self._record_group(group, kind)
                    group = whole
                else:
                    group = combined
            if assigned != {r["source_id"] for r in rows}:
                raise SourceError("A body paragraph is outside the complete source gaps")
            if group:
                yield self._record_group(group, kind)
            return
        for parent in super()._groups(rows, limit, min(max_items or self.wire_options.round1_max_items,
                                                      self.wire_options.round1_max_items)):
            yield from self._split_candidates(parent, kind)

    def _record_group(self, group, kind, parent_ids=None):
        size = len(wire_bytes(self.prompts[kind], self._packet(group, kind), self.wire_options))
        self.group_records.append({"task_kind": kind, "complete_task_ids": [r["source_id"] for r in group],
                                   "parent_task_ids": parent_ids, "estimated_initial_wire_bytes": size,
                                   "complete_source_text_sha256": digest([r["original_text"] for r in group]),
                                   "oversize_kept_whole": size > self.wire_options.max_wire_bytes})
        return group

    def _split_candidates(self, group, kind, parent_ids=None):
        size = len(wire_bytes(self.prompts[kind], self._packet(group, kind), self.wire_options))
        if size <= self.wire_options.max_wire_bytes or len(group) == 1:
            yield self._record_group(group, kind, parent_ids)
            return
        parent_ids = [r["source_id"] for r in group]
        middle = len(group) // 2
        yield from self._split_candidates(group[:middle], kind, parent_ids)
        yield from self._split_candidates(group[middle:], kind, parent_ids)

    def run(self):
        result = super().run()
        result["producer_version"] = VERSION
        result["wire_protocol"] = {"options": asdict(self.wire_options),
                                   "groups": copy.deepcopy(self.group_records),
                                   "actual_calls": copy.deepcopy(self.wire_judge.records),
                                   "complete_source_text_preserved": True,
                                   "oversize_is_unresolved_not_absence": True}
        return result


def extract_complete_nodes(project, rule_result, judge, options=None, wire_options=None):
    return _BoundedPipeline(project, rule_result, judge,
                            base.NodeOptions.parse(options), wire_options).run()
