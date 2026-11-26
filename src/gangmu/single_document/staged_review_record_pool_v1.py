"""Lossless display of repeated review evidence, with bounded full-item batches.

This changes transport presentation, never evidence, reference denominators,
quality axes or scoring. A fresh independent judge must still assess every ID.
"""
from __future__ import annotations

from collections import Counter
import copy
import json

from gangmu.representation.model import digest
from .source import SourceError
from .staged_nodes_wire_strategy import WireOptions, wire_bytes
from .staged_review_completion import TASKS, _schema

VERSION = "gangmu.e4.review-record-pool.v1"
REF = "$review_record"
POOL = "shared_review_records"
POLICY = """Evidence display uses a lossless shared-record table. An object of the
form {\"$review_record\": \"r1\"} means the complete JSON value stored as r1 in
shared_review_records. Expand references recursively, including references in
table entries. No source text, Context, condition, candidate or audit evidence
has been removed. Repeated objects refer to the same full record; this says
nothing about mathematical identity or correctness. Judge every requested ID
using the original review instructions and full expanded evidence. A table
reference is neither a supported judgment nor permission to omit a quality
axis. The response schema and all quality requirements remain unchanged.
"""


def _canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def pack(original):
    """Replace only repeated large JSON values; preserve schema and every value."""
    payload = copy.deepcopy(original)
    if not isinstance(payload, dict) or POOL in payload or REF in payload:
        raise SourceError("Review record packing requires an unpooled object")
    counts, objects = Counter(), {}

    def visit(value):
        if isinstance(value, (dict, list, str)):
            canonical = _canonical(value)
            if len(canonical.encode("utf-8")) >= 300:
                counts[canonical] += 1
                objects.setdefault(canonical, value)
        if isinstance(value, dict):
            if REF in value:
                raise SourceError("Source data conflicts with the reserved review reference key")
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    for key, value in payload.items():
        if key != "response_schema":
            visit(value)
    mapping = {value: f"r{index + 1}" for index, value in
               enumerate(value for value in counts if counts[value] > 1)}

    def transform(value, own=None):
        if isinstance(value, (dict, list, str)):
            canonical = _canonical(value)
            if canonical in mapping and canonical != own:
                return {REF: mapping[canonical]}
        if isinstance(value, dict):
            return {key: transform(item) for key, item in value.items()}
        if isinstance(value, list):
            return [transform(item) for item in value]
        return copy.deepcopy(value)

    result = {key: copy.deepcopy(value) if key == "response_schema" else transform(value)
              for key, value in payload.items()}
    result[POOL] = {key: transform(objects[value], value) for value, key in mapping.items()}
    if expand(result) != original:
        raise SourceError("Shared-record presentation does not restore the entire review input")
    return result


def expand(packed):
    """Decode the self-contained presentation; reject missing records or cycles."""
    if not isinstance(packed, dict) or not isinstance(packed.get(POOL), dict):
        raise SourceError("A complete shared review record table is required")
    pool, memo = packed[POOL], {}

    def restore(value, visiting=()):
        if isinstance(value, dict):
            if REF in value:
                if set(value) != {REF}:
                    raise SourceError("A shared record reference cannot contain extra fields")
                key = value[REF]
                if not isinstance(key, str) or key not in pool or key in visiting:
                    raise SourceError("Invalid, missing or cyclic shared review record")
                if key not in memo:
                    memo[key] = restore(pool[key], (*visiting, key))
                return copy.deepcopy(memo[key])
            return {key: restore(item, visiting) for key, item in value.items()}
        if isinstance(value, list):
            return [restore(item, visiting) for item in value]
        return copy.deepcopy(value)

    # Validate unused entries too; a corrupt pool never becomes admissible by
    # hiding its corruption in an unselected record.
    for key in pool:
        restore({REF: key})
    return {key: copy.deepcopy(value) if key == "response_schema" else restore(value)
            for key, value in packed.items() if key != POOL}


class RecordPoolJudge:
    """Split complete review items by actual wire bytes before any dispatch.

    The caller applies the unchanged strict review schema first. Each ID occurs
    in one bounded packet; responses combine without altering any judgment.
    Structural missing-ID completion remains owned by the existing wrapper.
    """

    def __init__(self, judge, options):
        self.judge = judge
        self.options = WireOptions.parse(options)
        self.requests = []

    def _packet(self, kind, instructions, original):
        payload = pack(original)
        actual_instructions = instructions.rstrip() + "\n\n" + POLICY
        packet = wire_bytes(actual_instructions, payload, self.options)
        return actual_instructions, payload, packet

    def plan(self, kind, instructions, original):
        if kind not in TASKS:
            packet = self._packet(kind, instructions, original)
            if len(packet[2]) > self.options.max_wire_bytes:
                raise SourceError("Complete structural completion exceeds the review wire bound")
            return [packet]
        collection, key = TASKS[kind]
        items = original[collection]
        identifiers = [row[key] for row in items]
        if not items or len(set(identifiers)) != len(identifiers):
            raise SourceError("Review items must have nonempty unique IDs")

        def candidate(rows):
            logical = copy.deepcopy(original)
            logical[collection] = copy.deepcopy(rows)
            logical["response_schema"] = _schema(kind, [row[key] for row in rows])
            return self._packet(kind, instructions, logical)

        packets, current = [], []
        for row in items:
            proposed = candidate([*current, row])
            if len(proposed[2]) <= self.options.max_wire_bytes:
                current.append(row)
                continue
            if current:
                packets.append(candidate(current))
                current = []
            single = candidate([row])
            if len(single[2]) > self.options.max_wire_bytes:
                raise SourceError("One complete review item exceeds the wire bound; no calls dispatched")
            current.append(row)
        if current:
            packets.append(candidate(current))
        if [row[key] for _, payload, _ in packets for row in expand(payload)[collection]] != identifiers:
            raise SourceError("Packed review partition changed the complete item population")
        return packets

    def __call__(self, kind, instructions, original):
        packets = self.plan(kind, instructions, original)
        answers = []
        for actual_instructions, payload, packet in packets:
            record = {"task_kind": kind, "logical_input_sha256": digest(original),
                      "actual_payload_sha256": digest(payload), "wire_bytes": len(packet),
                      "decoded_payload_sha256": digest(expand(payload)), "status": "started"}
            self.requests.append(record)
            try:
                response = self.judge(kind, actual_instructions, payload)
                if not isinstance(response, dict) or set(response) != {"judgments"} or not isinstance(response["judgments"], list):
                    raise SourceError("Packed review must return its original strict judgment object")
                if kind in TASKS:
                    collection, key = TASKS[kind]
                    allowed = {row[key] for row in expand(payload)[collection]}
                    if any(not isinstance(row, dict) or row.get(key) not in allowed
                           for row in response["judgments"]):
                        raise SourceError("A packed response cannot judge an ID from another packet")
                answers.extend(copy.deepcopy(response["judgments"]))
                record.update(status="returned", response_sha256=digest(response))
            except Exception as error:
                record.update(status="failed", error_type=type(error).__name__)
                raise
        return {"judgments": answers}
