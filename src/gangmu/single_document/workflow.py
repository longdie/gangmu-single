"""Source-bound dependency review workflow, separate from the E2 schema.

This checks observable tool work and retained review records. Candidate pages
and an Agent's review are not mathematical truth or a recall certificate.
"""
from __future__ import annotations

import copy
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Any

from gangmu.representation.model import digest

from .cache import _atomic_json
from .draft import Draft, DraftError


LEGACY_VERSION = "gangmu.e4.dependency-review-workflow.v2"
VERSION = "gangmu.e4.precision-review-workflow.v3"
INDIVIDUAL_DECISION_BINDING = "source_specific_explanations_v1"
SOURCE_REVIEW_BATCH_LIMIT = 8
TARGET_REVIEW_BATCH_LIMIT = 4

SOURCE_OPERATIONS = {"explicit": "explicit_inventory", "prose": "paragraph_inventory",
                     "reference": "reference_ledger"}


class WorkflowIncomplete(DraftError):
    """The mandatory observable review work has not been durably completed."""


class WorkflowGate:
    def __init__(self, project, draft):
        if project.fingerprint != draft.project.fingerprint:
            raise DraftError("workflow and draft source projects differ")
        self.project, self.draft = project, draft
        self.path = draft.work_dir / "relation-review.json"
        self.trace_path = draft.work_dir / "tool-trace.jsonl"
        if self.path.is_symlink() or self.trace_path.is_symlink():
            raise DraftError("workflow files cannot be symbolic links")
        self.state = {"version": VERSION, "source_fingerprint": project.fingerprint,
                      "reference_endpoint_binding": "incident_endpoints_v1",
                      "source_occurrence_binding": "unambiguous_occurrence_v1",
                      "individual_decision_binding": INDIVIDUAL_DECISION_BINDING,
                      "candidate_pages": [], "audits": {}, "history": [],
                      "source_pages": [], "source_reviews": {key: {} for key in SOURCE_OPERATIONS}}
        if draft.data.get('use_witness_binding') is not None:
            self.state['use_witness_binding'] = draft.data['use_witness_binding']
        if draft.data.get('context_source_policy') is not None:
            self.state['context_source_policy'] = draft.data['context_source_policy']
        if draft.data.get('node_content_policy') is not None:
            self.state['node_content_policy'] = draft.data['node_content_policy']
        if self.path.exists():
            self.state = json.loads(self.path.read_bytes())
        self._validate_state()

    def _validate_state(self):
        if (not isinstance(self.state, dict) or self.state.get("version") != VERSION
                or self.state.get("source_fingerprint") != self.project.fingerprint
                or not isinstance(self.state.get("candidate_pages"), list)
                or not isinstance(self.state.get("audits"), dict)
                or not isinstance(self.state.get("history"), list)):
            raise DraftError("invalid or differently source-bound workflow checkpoint")
        if self.version == VERSION and (not isinstance(self.state.get("source_pages"), list)
                or not isinstance(self.state.get("source_reviews"), dict)
                or set(self.state["source_reviews"]) != set(SOURCE_OPERATIONS)
                or any(not isinstance(value, dict) for value in self.state["source_reviews"].values())):
            raise DraftError("invalid source-review checkpoint")
        if self.state.get("reference_endpoint_binding") not in (None, "incident_endpoints_v1"):
            raise DraftError("unknown reference endpoint binding revision")
        if self.state.get("source_occurrence_binding") not in (None, "unambiguous_occurrence_v1"):
            raise DraftError("unknown source occurrence binding revision")
        marker = self.state.get("individual_decision_binding")
        if marker != INDIVIDUAL_DECISION_BINDING:
            raise DraftError("current producer requires source-specific individual decision binding")
        if marker not in (None, INDIVIDUAL_DECISION_BINDING):
            raise DraftError("unknown individual decision binding revision")
        if (self.state.get('use_witness_binding') != self.draft.data.get('use_witness_binding')
                or self.state.get('use_witness_binding') not in (None, Draft.USE_WITNESS_BINDING)):
            raise DraftError('workflow dependency use witness contract differs from draft')
        self.draft.validate_context_source_policy()
        if self.state.get('context_source_policy') != self.draft.data.get('context_source_policy'):
            raise DraftError('workflow explicit context source policy differs from draft')
        self.draft.validate_node_content_policy()
        if self.state.get('node_content_policy') != self.draft.data.get('node_content_policy'):
            raise DraftError('workflow node content policy differs from draft')

    @property
    def version(self):
        return self.state["version"]

    def _save(self):
        if self.path.is_symlink() or self.path.with_suffix(".json.partial").is_symlink():
            raise DraftError("workflow checkpoint cannot be a symbolic link")
        _atomic_json(self.path, self.state)

    def _nodes(self):
        return {key: node for key, node in sorted(self.draft.data["nodes"].items())
                if node["judgment"] == "accepted"}

    def _node_content(self, node):
        records, seen = [], set()
        context_id = node["context"]
        while context_id is not None:
            if context_id in seen or context_id not in self.draft.data["contexts"]:
                raise DraftError("workflow requires complete acyclic context ancestry")
            seen.add(context_id)
            context = self.draft.data["contexts"][context_id]
            records.append(context)
            context_id = context.get("parent")
        # Cache only after hashing current node and entire actual ancestry.
        # Updates on a reused gate therefore cannot reuse an earlier binding.
        key = digest({"node": node, "contexts": records})
        cache = self.__dict__.setdefault("_content_cache", {})
        if key not in cache:
            cache[key] = {"node": copy.deepcopy(node), "original_text": [self.project.resolve_span(span)
                          for span in node["spans"]], "contexts": [
                {"record": copy.deepcopy(context), "original_text": [
                    self.project.resolve_span(span) for span in context["spans"]]}
                for context in records]}
        return cache[key]

    def _inventory(self):
        self.project.assert_unchanged()
        nodes = self._nodes()
        contents = {key: self._node_content(node) for key, node in nodes.items()}
        # Proof locator candidates consume these current associations. Bind
        # only their accepted IDs/endpoints; ordinary dependency edits must
        # not invalidate the candidate inventory used to review those edges.
        proves = [{"id": relation["id"], "source": relation["source"], "target": relation["target"]}
                  for relation in sorted(self.draft.data["relations"].values(), key=lambda value: value["id"])
                  if relation["type"] == "proves" and relation["judgment"] == "accepted"]
        return nodes, digest({"source_fingerprint": self.project.fingerprint,
                              "accepted_nodes": contents, "accepted_proves": proves})

    def _binding(self, node_id, inventory_sha, candidates):
        nodes = self._nodes()
        if node_id not in nodes:
            raise DraftError("relation review target must be an accepted node")
        incoming = [copy.deepcopy(relation) for _, relation in
                    sorted(self.draft.data["relations"].items()) if relation["target"] == node_id]
        value = {"version": self.version, "inventory_sha256": inventory_sha,
                 "node_sha256": digest(self._node_content(nodes[node_id])),
                 "incoming_sha256": digest(incoming),
                 "candidate_set_sha256": digest(candidates)}
        if self.version == VERSION:
            value["individual_decision_binding"] = INDIVIDUAL_DECISION_BINDING
        if self.state.get('use_witness_binding') is not None:
            value['use_witness_binding'] = self.state['use_witness_binding']
        return {**value, "review_binding": digest(value),
                "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
                "incoming_relation_ids": [relation["id"] for relation in incoming]}

    def _trace(self):
        return self.trace_path.read_bytes() if self.trace_path.exists() else b""

    def _receipts(self, trace):
        receipts = {}
        lines = trace.splitlines(keepends=True)
        identity = digest(self.project.fingerprint)
        for number, line in enumerate(lines):
            try:
                row = json.loads(line)
            except (ValueError, UnicodeError):
                if number == len(lines) - 1 and not line.endswith(b"\n"):
                    break  # An interrupted final append never receives credit.
                raise DraftError("corrupt workflow tool trace")
            event, call_id = row.get("event"), row.get("call_id")
            if event not in ("started", "completed"):
                continue
            if type(call_id) is not int or call_id < 1:
                raise DraftError("invalid workflow tool call identity")
            receipt = receipts.setdefault(call_id, {})
            if event in receipt:
                raise DraftError("duplicate workflow tool call event")
            if event == "started" and (row.get("source_identity") != identity
                    or digest(row.get("arguments")) != row.get("arguments_sha256")):
                raise DraftError("workflow tool request source or arguments differ")
            receipt[event] = row
            receipt[event + "_index"] = number
        if self.version == VERSION:
            for pair in receipts.values():
                start, end = pair.get("started"), pair.get("completed")
                if start and end and end.get("response", {}).get("ok") is True:
                    self._individual_batch(start.get("operation"), start.get("arguments"))
        return receipts

    @staticmethod
    def _has_receipt(record, receipts, operation):
        pair = receipts.get(record.get("call_id"), {})
        start, end = pair.get("started"), pair.get("completed")
        if not start or not end or start.get("operation") != operation or end.get("operation") != operation:
            return False
        response = end.get("response", {})
        return (response.get("ok") is True and response.get("call_id") == record["call_id"]
                and pair["started_index"] < pair["completed_index"]
                and start.get("arguments_sha256") == record.get("arguments_sha256")
                and digest(response.get("result")) == record.get("result_sha256"))

    @staticmethod
    def _covers(pages):
        if not pages:
            return False
        totals = {page["total"] for page in pages}
        if len(totals) != 1:
            return False
        total = totals.pop()
        if total == 0:
            return any(page["offset"] == 0 and page["count"] == 0 for page in pages)
        end = 0
        for page in sorted(pages, key=lambda value: value["offset"]):
            if end >= total:
                return True
            if page["offset"] > end:
                return False
            end = max(end, page["offset"] + page["count"])
        return end >= total

    def _candidate_record(self, record, inventory_sha):
        raw = {key: value for key, value in record.items() if key != "candidate_id"}
        identifier = "candidate:" + digest({"version": self.version,
            "source_fingerprint": self.project.fingerprint,
            "inventory_sha256": inventory_sha, "record": raw})
        return {**copy.deepcopy(raw), "candidate_id": identifier}

    def _candidate_state(self, inventory_sha, nodes, receipts, *, independent=False,
                         before_call_id=None):
        groups, stale, expected = {}, [], {}
        catalog, page_calls = {key: {} for key in nodes}, {key: [] for key in nodes}
        boundary = None
        if before_call_id is not None:
            boundary = receipts.get(before_call_id, {}).get("started_index")
            if boundary is None:
                raise DraftError("candidate review has no actual started receipt")
        for page in self.state["candidate_pages"]:
            if page.get("inventory_sha256") != inventory_sha:
                continue
            if not self._has_receipt(page, receipts, "relation_candidates"):
                stale.append(page.get("call_id"))
                continue
            if boundary is not None and receipts[page["call_id"]]["completed_index"] >= boundary:
                continue
            result = receipts[page["call_id"]]["completed"]["response"]["result"]
            query = page.get("query", {})
            arguments = receipts[page["call_id"]]["started"]["arguments"]
            actual_query = {"node_id": arguments.get("node_id"), "offset": arguments.get("offset", 0),
                            "limit": arguments.get("limit", 50)}
            if (result.get("workflow_inventory_sha256") != inventory_sha
                    or result.get("workflow_query") != query
                    or actual_query != query
                    or page.get("total") != result.get("total")
                    or page.get("count") != len(result.get("records", []))
                    or page.get("offset") != query.get("offset")
                    or type(page.get("total")) is not int or page["total"] < 0
                    or type(page.get("offset")) is not int or page["offset"] < 0):
                raise DraftError("candidate workflow page differs from completed response")
            if independent:
                if self.version == LEGACY_VERSION:
                    from .candidates_v2 import discover
                else:
                    from .candidates import discover
                key = (query.get("node_id"), query["offset"], query["limit"])
                if key not in expected:
                    expected[key] = discover(self.project, self.draft, **query)
                    expected[key]["records"] = [self._candidate_record(record, inventory_sha)
                                                 for record in expected[key]["records"]]
                stripped = {key: value for key, value in result.items()
                            if key not in ("workflow_inventory_sha256", "workflow_query")}
                if digest(stripped) != digest(expected[key]):
                    raise DraftError("retained candidate page differs from source-bound recomputation")
            if query.get("node_id") is not None and query["node_id"] not in nodes:
                raise DraftError("retained candidate query target is not accepted")
            for record in result["records"]:
                target_id = record.get("target")
                expected_type = "proof_dependency" if nodes.get(target_id, {}).get("category") == "proof" else "statement_dependency"
                if (target_id not in nodes or record.get("source") not in nodes
                        or record.get("source") == target_id or record.get("type") != expected_type
                        or (query.get("node_id") is not None and query["node_id"] != target_id)
                        or record != self._candidate_record(record, inventory_sha)):
                    raise DraftError("candidate record differs from source-bound target/type/ID")
                existing = catalog[target_id].get(record["candidate_id"])
                if existing is not None and existing != record:
                    raise DraftError("candidate ID has inconsistent source records")
                catalog[target_id][record["candidate_id"]] = copy.deepcopy(record)
            for target_id in nodes if query.get("node_id") is None else [query["node_id"]]:
                page_calls[target_id].append(page["call_id"])
            groups.setdefault(query.get("node_id"), []).append(page)
        global_complete = self._covers(groups.get(None, []))
        complete = {node_id for node_id in nodes if global_complete or self._covers(groups.get(node_id, []))}
        return complete, {"global_complete": global_complete, "reviewed_targets": sorted(complete),
                          "missing_targets": sorted(set(nodes) - complete),
                          "uncompleted_page_calls": sorted(stale)}, {
            target: [catalog[target][key] for key in sorted(catalog[target])]
            for target in nodes}, {target: sorted(set(values)) for target, values in page_calls.items()}

    def _candidate_coverage(self, inventory_sha, nodes, receipts, *, independent=False):
        complete, summary, _, _ = self._candidate_state(inventory_sha, nodes, receipts,
                                                      independent=independent)
        return complete, summary

    def record_candidate_page(self, arguments, result, call_id):
        nodes, inventory_sha = self._inventory()
        query = {"node_id": arguments.get("node_id"), "offset": arguments.get("offset", 0),
                 "limit": arguments.get("limit", 50)}
        if query["node_id"] is not None and query["node_id"] not in nodes:
            raise DraftError("candidate workflow target must be an accepted node")
        output = {**result, "records": [self._candidate_record(record, inventory_sha)
                  for record in result["records"]],
                  "workflow_inventory_sha256": inventory_sha, "workflow_query": query}
        record = {"call_id": call_id, "arguments_sha256": digest(arguments),
                  "result_sha256": digest(output), "inventory_sha256": inventory_sha,
                  "query": query, "offset": query.get("offset", 0), "total": result["total"],
                  "count": len(result["records"])}
        self.state["candidate_pages"].append(record)
        self.state["history"].append({"kind": "candidate_page", "record": copy.deepcopy(record)})
        self._save()
        return output

    def bindings(self, offset=0, limit=100):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise DraftError("invalid review binding pagination")
        nodes, inventory_sha = self._inventory()
        covered, _, catalog, page_calls = self._candidate_state(inventory_sha, nodes, self._receipts(self._trace()))
        identifiers = list(nodes)
        return {"inventory_sha256": inventory_sha, "total": len(nodes),
                "records": [{"node_id": key, **self._binding(key, inventory_sha, catalog[key]),
                             "candidates": copy.deepcopy(catalog[key]),
                             "candidate_page_call_ids": page_calls[key],
                             "candidate_pages_complete": key in covered}
                            for key in identifiers[offset:offset + limit]],
                "next_offset": offset + limit if offset + limit < len(nodes) else None,
                "notice": "Review every retrieved candidate as accepted/excluded/unresolved with a specific explanation, and review source against the full inventory for retrieval misses. Bindings do not certify mathematics."}

    def _individual_batch(self, operation, arguments):
        if self.version != VERSION or operation not in ("review_extraction", "review_relations"):
            return
        if not isinstance(arguments, dict) or not isinstance(arguments.get("reviews"), list):
            raise DraftError("individual review batch lacks records")
        limit = (TARGET_REVIEW_BATCH_LIMIT if operation == "review_relations" else
                 SOURCE_REVIEW_BATCH_LIMIT if arguments.get("kind") in ("explicit", "prose") else 100)
        if not 1 <= len(arguments["reviews"]) <= limit:
            raise DraftError("individual " + operation + " batch must contain 1.." + str(limit) + " reviews")

    @staticmethod
    def _negative_reason(value, identifiers=()):
        value = unicodedata.normalize("NFKC", value).casefold()
        identifiers = sorted({unicodedata.normalize("NFKC", str(identifier)).casefold()
                              for identifier in identifiers if identifier}, key=len, reverse=True)
        if identifiers:
            # Strip host-issued record identifiers, not mathematical numbers.
            # A known name followed by an equality may be a formula parameter.
            value = re.sub(r"(?<!\w)(?:" + "|".join(re.escape(v) for v in identifiers)
                           + r")(?!\w)(?!\s*=(?!=))", "", value)
        value = re.sub(r"(?<!\w)#\s*\d+\b", "", value)
        # Editorial offsets and list ordinals must not salt a generic reason.
        # Keep genuine mathematical numerals (n=2, characteristic 2, etc.).
        value = re.sub(r"(?m)^\s*(?:\(\d+\)|\d+[.):：])\s+", "", value)
        value = re.sub(r"\b((?:source|target|reading|record|item|char(?:acter)?)\s+"
                       r"(?:offset|position|span|range))\s*[:#=]?\s*\(?\s*\d+"
                       r"(?:\s*[-–:,]\s*\d+)?\s*\)?", r"\1", value)
        value = re.sub(r"\b(review\s+item|item|node|candidate|source|target|call(?:_id)?|"
                       r"record|row|paragraph|line|log\s+(?:entry|position)|char(?:acter)?\s+offset)"
                       r"\s*[:#]?\s*\d+(?:\s*[-–:]\s*\d+)?\b", r"\1", value)
        return re.sub(r"[\W_]+", "", value)

    def _decision_identifiers(self, new_source=(), new_targets=()):
        inventory = self.project._inventory()
        identifiers = set(self.draft.data["nodes"]) | set(self.draft.data["relations"])
        for records in (inventory.explicit, inventory.prose, inventory.references):
            identifiers.update(record["item_id"] for record in records)
        for value in list(self.state["audits"].values()) + list(new_targets):
            identifiers.update(value.get("candidate_ids", []))
            identifiers.update(d.get("candidate_id") for d in value.get("candidate_dispositions", []))
        identifiers.update(value.get("item_id") for value in new_source)
        return identifiers - {None}

    def _literal_scope(self, spans, extra=None):
        # Literal identity and a host-derived scope, never a model's navigation
        # or external claim, may explain repetition. This is not semantic identity.
        return digest({"source": self.project.fingerprint,
            "literal": [self.project.resolve_span(s) for s in spans],
            "scope": [(s["file"], self._reading_occurrence(s)) for s in spans], "extra": extra})

    def _covers_original_fragments(self, evidence, required):
        for need in required:
            occurrence = self._reading_occurrence(need)
            if occurrence is None:
                return False
            cursor = need["char_start"]
            intervals = sorted((s["char_start"], s["char_end"]) for s in evidence
                if s["file"] == need["file"] and self._reading_occurrence(s) == occurrence)
            for first, last in intervals:
                if first > cursor:
                    break
                if last > cursor:
                    cursor = last
                if cursor >= need["char_end"]:
                    break
            if cursor < need["char_end"]:
                return False
        return bool(required)

    def _decision_fragments(self, spans, *, reference=False):
        return self.project.decision_fragments(spans, reference=reference)

    def target_use_fragments(self, node_id, offset=0, limit=20):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise DraftError('invalid target fragment pagination')
        node = self._nodes().get(node_id)
        if node is None:
            raise DraftError('target fragment node is not accepted')
        fragments = self._decision_fragments(node['spans'])
        return {'node_id': node_id, 'total': len(fragments),
                'records': [self.project.fragment_record(value) for value in fragments[offset:offset + limit]],
                'next_offset': offset + limit if offset + limit < len(fragments) else None,
                'notice': 'Exact original fragment locations, not relation approval or source-review completion. '
                          'Use original_text for actual content. A single-span use witness can copy span only when it is not null.'}

    def _require_decision_fragment(self, evidence, scopes, *, reference=False, node_id=None,
                                   source_kind=None, item_id=None):
        if self.version != VERSION:
            return
        fragments = self._decision_fragments(scopes, reference=reference)
        if any(self._covers_original_fragments(evidence, needed) for needed in fragments):
            return
        def distance(fragment):
            values = [max(0, need['char_start'] - span['char_end'], span['char_start'] - need['char_end'])
                      for need in fragment for span in evidence if need['file'] == span['file']
                      and self._reading_occurrence(need) == self._reading_occurrence(span)]
            return min(values) if values else float('inf')
        details = {'constraint': 'complete_original_fragment',
                   'submitted_evidence_spans': copy.deepcopy(evidence),
                   'candidate_fragments': [self.project.fragment_record(value)
                       for value in sorted(fragments, key=distance)[:4]],
                   'fragment_count': len(fragments),
                   'notice': 'Copy complete original positions after reading them; these hints do not establish the mathematical relationship.'}
        if node_id is not None:
            details['target'] = node_id
            details['read_hint'] = {'operation': 'target_use_fragments', 'arguments': {'node_id': node_id, 'offset': 0, 'limit': 20}}
        if item_id is not None:
            details.update(source_kind=source_kind, item_id=item_id,
                read_hint={'operation': SOURCE_OPERATIONS[source_kind], 'arguments': {'item_id': item_id, 'preview_chars': 24000}})
        raise DraftError('individual decision evidence must cover a complete original clause or source fragment', details=details)

    def _source_negative_decisions(self, kind, value, record):
        spans = record.get("source_spans", [record["span"]])
        if kind != "reference":
            negative = value.get("decision") in ("excluded", "unresolved") or (
                kind == "prose" and not value.get("node_ids"))
            return [("source", value["explanation"], self._literal_scope(spans,
                {"sections": record.get("section_ids", []), "parent": record.get("parent_environment")}),
                {'source_kind': kind, 'item_id': value['item_id'], 'scope_spans': spans})] if negative else []
        result = []
        for decision in value["key_decisions"]:
            targets = [target for target in record.get("targets", []) if target.get("key") == decision["key"]]
            external = bool(targets) and all(target.get("resolution") == "external" for target in targets)
            if decision["disposition"] == "dependency" or (decision["disposition"] == "external" and external):
                continue
            clauses = self._decision_fragments(spans, reference=True)
            scope = self._literal_scope(spans, {"key": decision["key"], "targets": targets,
                "sections": record.get("section_ids", []), "clauses": [self._literal_scope(c) for c in clauses]})
            result.append(("reference", decision["explanation"], scope,
                {'source_kind': kind, 'item_id': value['item_id'], 'reference_key': decision['key'], 'scope_spans': spans}))
        return result

    def _candidate_negative_decisions(self, value):
        result = []
        nodes = self._nodes()
        for decision in value["candidate_dispositions"]:
            if decision["judgment"] == "accepted":
                continue
            source, target = nodes[decision["source"]], nodes[value["node_id"]]
            literal = {"type": decision["type"], "source": self._literal_scope(source["spans"],
                self._node_content(source)["contexts"]), "target": self._literal_scope(target["spans"],
                self._node_content(target)["contexts"])}
            result.append(("candidate", decision["explanation"], digest(literal),
                {'source': decision['source'], 'target': value['node_id'], 'type': decision['type'],
                 'candidate_id': decision['candidate_id'], 'source_spans': source['spans'], 'target_spans': target['spans']}))
        # A target without any accepted mathematical dependency still makes
        # a negative source-use decision, even with an empty candidate list.
        # A proves edge identifies a proof owner and is not a dependency here.
        target = nodes[value["node_id"]]
        accepted_dependency = any(relation["target"] == value["node_id"]
            and relation["type"] in ("statement_dependency", "proof_dependency")
            and relation["judgment"] == "accepted" for relation in self.draft.data["relations"].values())
        if not accepted_dependency:
            literal = self._literal_scope(target["spans"], self._node_content(target)["contexts"])
            result.append(("target", value["explanation"], literal,
                {'target': value['node_id'], 'scope_spans': target['spans']}))
        return result

    def _negative_conflict_details(self, context, reason, family):
        result = {key: copy.deepcopy(value) for key, value in context.items() if not key.endswith('spans')}
        result.update(family=family, explanation=reason)
        for key in ('scope_spans', 'source_spans', 'target_spans'):
            if key not in context:
                continue
            spans = context[key]
            fragments = self._decision_fragments(spans, reference=context.get('source_kind') == 'reference')
            result[key.replace('spans', 'fragments')] = [self.project.fragment_record(value)
                for value in (fragments[:3] if fragments else [spans])]
        return result

    def _individual_decisions(self, receipts, *, new_source=(), new_targets=(), source_kind=None):
        if self.version != VERSION:
            return
        tokens = []
        replace_source = {v["item_id"] for v in new_source}
        replace_targets = {v["node_id"] for v in new_targets}
        for kind in SOURCE_OPERATIONS:
            catalog, inventory_sha = self._source_catalog(kind)
            for identifier, value in self.state["source_reviews"][kind].items():
                if identifier not in catalog or (kind == source_kind and identifier in replace_source):
                    continue
                if not self._has_receipt(value, receipts, "review_extraction"):
                    continue
                binding = self._source_binding(kind, catalog[identifier], inventory_sha)
                if value.get("review_binding") != binding["review_binding"]:
                    continue  # Invalidated old decisions do not occupy a reason.
                args = receipts[value["call_id"]]["started"]["arguments"]
                submitted = [v for v in args.get("reviews", []) if v.get("item_id") == identifier]
                try:
                    if args.get("kind") != kind or len(submitted) != 1:
                        continue
                    expected = self._source_review_value(kind, submitted[0], catalog[identifier], inventory_sha,
                                                         receipts=receipts, before_call_id=value["call_id"])
                    stripped = {k: v for k, v in value.items() if k not in ("call_id", "arguments_sha256", "result_sha256")}
                    if stripped != expected:
                        continue
                except (DraftError, KeyError, TypeError, ValueError):
                    continue
                tokens.extend(self._source_negative_decisions(kind, value, catalog[identifier]))
        if self.state["audits"]:
            nodes, inventory_sha = self._inventory()
            _, _, catalog, _ = self._candidate_state(inventory_sha, nodes, receipts)
            for identifier, value in self.state["audits"].items():
                if identifier in replace_targets or identifier not in nodes:
                    continue
                if not self._has_receipt(value, receipts, "review_relations"):
                    continue
                binding = self._binding(identifier, inventory_sha, catalog[identifier])
                if any(value.get(k) != v for k, v in binding.items()):
                    continue
                args = receipts[value["call_id"]]["started"]["arguments"]
                submitted = [v for v in args.get("reviews", []) if v.get("node_id") == identifier]
                try:
                    if len(submitted) != 1 or self._dispositions(submitted[0].get("candidate_dispositions"),
                            catalog[identifier], identifier) != value["candidate_dispositions"]:
                        continue
                except (DraftError, KeyError, TypeError, ValueError):
                    continue
                tokens.extend(self._candidate_negative_decisions(value))
        if new_source:
            catalog, _ = self._source_catalog(source_kind)
            for value in new_source:
                tokens.extend(self._source_negative_decisions(source_kind, value, catalog[value["item_id"]]))
        for value in new_targets:
            tokens.extend(self._candidate_negative_decisions(value))
        seen = {}
        identifiers = self._decision_identifiers(new_source, new_targets)
        for family, reason, literal, context in tokens:
            key = (family, self._negative_reason(reason, identifiers))
            if key in seen and seen[key][0] != literal:
                previous = seen[key]
                raise DraftError("different current " + family + " decisions reuse a generic negative explanation; explain each source use or gap",
                    details={'constraint': 'source_specific_negative_reason', 'family': family,
                        'normalized_reason': key[1], 'conflicts': [
                            self._negative_conflict_details(previous[2], previous[1], family),
                            self._negative_conflict_details(context, reason, family)],
                        'notice': 'Read both actual source fragments and explain each mathematical use or gap; changing an ID or ordinal is not a new reason.'})
            seen[key] = (literal, reason, context)

    def _dispositions(self, submitted, candidates, node_id):
        if not isinstance(submitted, list):
            raise DraftError("review requires candidate_dispositions, including [] for no candidates")
        by_pair = {(record["source"], record["type"]): record for record in candidates}
        if len(by_pair) != len(candidates):
            raise DraftError("duplicate retrieved source/type candidate for target")
        normalized, seen = [], set()
        for disposition in submitted:
            if (not isinstance(disposition, dict)
                    or set(disposition) != {"source", "type", "judgment", "explanation"}
                    or not isinstance(disposition.get("source"), str)
                    or not isinstance(disposition.get("type"), str)):
                raise DraftError("candidate disposition requires source/type/judgment/explanation")
            pair = (disposition["source"], disposition["type"])
            if pair not in by_pair or pair in seen:
                raise DraftError("candidate disposition is unknown or repeated for target")
            seen.add(pair)
            judgment, explanation = disposition["judgment"], disposition["explanation"]
            if judgment not in ("accepted", "excluded", "unresolved"):
                raise DraftError("candidate judgment must be accepted/excluded/unresolved")
            if not isinstance(explanation, str) or not explanation.strip():
                raise DraftError("every candidate disposition requires a specific source-use explanation")
            accepted_relations = sorted(relation["id"] for relation in self.draft.data["relations"].values()
                if relation["source"] == pair[0] and relation["target"] == node_id
                and relation["type"] == pair[1] and relation["judgment"] == "accepted")
            if (judgment == "accepted") != bool(accepted_relations):
                raise DraftError("candidate disposition conflicts with current accepted dependency edges")
            normalized.append({**copy.deepcopy(disposition),
                "candidate_id": by_pair[pair]["candidate_id"],
                "accepted_relation_ids": accepted_relations})
        if seen != set(by_pair):
            raise DraftError("candidate_dispositions must cover exactly every retrieved source/type for target")
        return sorted(normalized, key=lambda record: record["candidate_id"])

    def record_reviews(self, arguments, call_id):
        self._individual_batch("review_relations", arguments)
        reviews = arguments.get("reviews")
        if set(arguments) != {"reviews"} or not isinstance(reviews, list) or not 1 <= len(reviews) <= 100:
            raise DraftError("review_relations requires a batch of 1..100 reviews")
        nodes, inventory_sha = self._inventory()
        receipts = self._receipts(self._trace())
        covered, _, catalog, page_calls = self._candidate_state(inventory_sha, nodes, receipts,
                                                               before_call_id=call_id)
        prepared, seen = [], set()
        for review in reviews:
            if not isinstance(review, dict):
                raise DraftError("relation review must be a JSON object")
            node_id = review.get("node_id")
            if node_id in seen:
                raise DraftError("duplicate relation review target in one batch")
            seen.add(node_id)
            binding = self._binding(node_id, inventory_sha, catalog.get(node_id, []))
            if node_id not in covered:
                raise WorkflowIncomplete("complete candidate pages for review target " + str(node_id))
            if review.get("review_binding") != binding["review_binding"]:
                raise WorkflowIncomplete("relation review binding changed for target " + str(node_id))
            dispositions = self._dispositions(review.get("candidate_dispositions"), catalog[node_id], node_id)
            identifiers = review.get("reviewed_relation_ids")
            if (not isinstance(identifiers, list) or any(not isinstance(value, str) for value in identifiers)
                    or sorted(identifiers) != sorted(binding["incoming_relation_ids"])
                    or len(set(identifiers)) != len(identifiers)):
                raise DraftError("review must name exactly the target's current incoming relations")
            explanation = review.get("explanation")
            if not isinstance(explanation, str) or not explanation.strip():
                raise DraftError("relation review requires an original-source explanation, including when no relations were found")
            spans = review.get("evidence_spans")
            if not isinstance(spans, list) or not spans:
                raise DraftError("relation review requires original-source positions")
            spans = [self.project.normalize_span(span) for span in spans]
            if not any(a["file"] == b["file"] and a["char_start"] < b["char_end"]
                       and b["char_start"] < a["char_end"] for a in spans for b in nodes[node_id]["spans"]):
                raise DraftError("relation review evidence must include a position within the reviewed node")
            scopes = nodes[node_id]["spans"] + [s for c in self._node_content(nodes[node_id])["contexts"] for s in c["record"]["spans"]]
            self._require_decision_fragment(spans, scopes, node_id=node_id)
            prepared.append({"node_id": node_id, **binding, "explanation": explanation,
                "candidate_dispositions": dispositions,
                "candidate_page_call_ids": page_calls[node_id],
                "evidence_spans": spans, "original_text": [self.project.resolve_span(span) for span in spans],
                "reviewed_relation_ids": identifiers, "call_id": call_id,
                "arguments_sha256": digest(arguments)})
        self._individual_decisions(receipts, new_targets=prepared)
        output = {"reviewed": len(prepared), "node_ids": sorted(seen), "inventory_sha256": inventory_sha,
                  "candidate_disposition_counts": {judgment: sum(d["judgment"] == judgment
                      for record in prepared for d in record["candidate_dispositions"])
                      for judgment in ("accepted", "excluded", "unresolved")},
                  "notice": "Agent source-use reviews are retained; mathematical correctness and recall are not certified."}
        for record in prepared:
            record["result_sha256"] = digest(output)
            self.state["audits"][record["node_id"]] = record
        self.state["history"].append({"kind": "relation_reviews", "records": copy.deepcopy(prepared)})
        self._save()
        return output

    def _source_catalog(self, kind):
        if kind not in SOURCE_OPERATIONS:
            raise DraftError("extraction kind must be explicit, prose or reference")
        if self.version != VERSION:
            raise DraftError("source-round operations require workflow v3")
        inventory = self.project._inventory()
        values = {"explicit": inventory.explicit, "prose": inventory.prose,
                  "reference": inventory.references}[kind]
        records = {record["item_id"]: copy.deepcopy(record) for record in values}
        if len(records) != len(values):
            raise DraftError("source inventory has repeated item IDs")
        return records, inventory.digests[kind]

    def _reading_occurrence(self, span):
        """A physical range is attributable only to one actual reading occurrence.

        Explicit anchors can select a known occurrence of repeated source;
        unanchored ambiguous source cannot authorize a source-round match.
        """
        key = (span["file"], span["char_start"], span["char_end"])
        cache = self.__dict__.setdefault("_source_occurrence_cache", {})
        if key not in cache:
            cache[key] = {row["occurrence"] for row in self.project.source_map
                if row["source"]["file"] == span["file"]
                and row["source"]["char_start"] < span["char_end"]
                and span["char_start"] < row["source"]["char_end"]}
        candidates = cache[key]
        stated = span.get("occurrence")
        if stated is not None:
            return stated if stated in candidates else None
        return next(iter(candidates)) if len(candidates) == 1 else None

    def _overlaps(self, left, right):
        anchored = self.state.get("source_occurrence_binding") == "unambiguous_occurrence_v1"
        for a in left:
            for b in right:
                if (a["file"] != b["file"] or a["char_start"] >= b["char_end"]
                        or b["char_start"] >= a["char_end"]):
                    continue
                if not anchored:
                    return True
                occurrence = self._reading_occurrence(a)
                if occurrence is not None and occurrence == self._reading_occurrence(b):
                    return True
        return False

    def _source_binding(self, kind, record, inventory_sha):
        spans = record.get("source_spans", [record["span"]])
        if kind == "reference":
            spans = spans + [span for target in record.get("targets", [])
                             for span in target.get("target_spans", []) + target.get("owner_source_spans", [])]
        related = {}
        for identifier, node in self._nodes().items():
            ranges = node["spans"]
            if kind == "reference":
                ranges = ranges + [span for context in self._node_content(node)["contexts"]
                                   for span in context["record"]["spans"]]
            if self._overlaps(ranges, spans):
                related[identifier] = self._node_content(node)
        relations = {key: value for key, value in sorted(self.draft.data["relations"].items())
                     if kind == "reference" and
                     (value["source"] in related or value["target"] in related)}
        # Historical v3 records retain their original algorithm for replay.
        # New checkpoints bind actual endpoints even when a handwritten locator
        # has no statically parsed target. Expand only the incident relations.
        if self.state.get("reference_endpoint_binding") == "incident_endpoints_v1":
            nodes = self._nodes()
            for relation in relations.values():
                for identifier in (relation["source"], relation["target"]):
                    if identifier in nodes:
                        related[identifier] = self._node_content(nodes[identifier])
        value = {"version": self.version, "kind": kind, "inventory_sha256": inventory_sha,
                 "item_sha256": digest(record), "related_nodes_sha256": digest(related),
                 "related_relations_sha256": digest(relations)}
        if self.state.get("source_occurrence_binding") is not None:
            value["source_occurrence_binding"] = self.state["source_occurrence_binding"]
        if self.version == VERSION:
            value["individual_decision_binding"] = INDIVIDUAL_DECISION_BINDING
        if self.state.get('use_witness_binding') is not None:
            value['use_witness_binding'] = self.state['use_witness_binding']
        heading_ids = sorted({heading['item_id'] for target in record.get('targets', [])
                              for heading in target.get('heading_anchors', [])}) if kind == 'reference' else []
        if heading_ids:
            value['heading_ownership_sha256'] = digest({identifier:
                self.state['source_reviews']['explicit'].get(identifier) for identifier in heading_ids})
        return {**value, "review_binding": digest(value),
                "related_node_ids": sorted(related), "related_relation_ids": sorted(relations)}

    def _accepted_heading_owner(self, heading, node):
        """A printed title needs a current source-specific admitted owner.

        Formatting alone also finds ordinary emphasis/navigation. Accepting an
        enclosing mathematical node does not silently change an excluded title
        into that node's identifier. A genuine duplicate title can be reviewed
        as accepted with the already existing complete owner's node ID.
        """
        catalog, inventory_sha = self._source_catalog('explicit')
        record = catalog.get(heading['item_id'])
        review = self.state['source_reviews']['explicit'].get(heading['item_id'])
        return bool(record and review and review.get('decision') == 'accepted'
            and node['id'] in review.get('node_ids', [])
            and review.get('review_binding') == self._source_binding('explicit', record, inventory_sha)['review_binding'])

    def record_source_page(self, kind, arguments, result, call_id):
        catalog, inventory_sha = self._source_catalog(kind)
        if result.get("inventory_sha256") != inventory_sha:
            raise DraftError("source page has different inventory binding")
        query = copy.deepcopy(arguments)
        record = {"kind": kind, "call_id": call_id, "arguments_sha256": digest(arguments),
                  "result_sha256": digest(result), "inventory_sha256": inventory_sha,
                  "query": query, "offset": query.get("offset", 0), "total": result["total"],
                  "count": len(result["records"]),
                  "item_ids": [item["item_id"] for item in result["records"]]}
        if any(identifier not in catalog for identifier in record["item_ids"]):
            raise DraftError("source page contains unknown inventory item")
        self.state["source_pages"].append(record)
        self.state["history"].append({"kind": "source_page", "record": copy.deepcopy(record)})
        self._save()
        return result

    def _source_page_state(self, kind, inventory_sha, catalog, receipts, *, independent=False,
                           before_call_id=None):
        operation = SOURCE_OPERATIONS[kind]
        pages, seen = [], set()
        boundary = receipts.get(before_call_id, {}).get("started_index") if before_call_id else None
        if before_call_id and boundary is None:
            raise DraftError("source review has no actual started tool receipt")
        for page in self.state["source_pages"]:
            if page.get("kind") != kind or page.get("inventory_sha256") != inventory_sha:
                continue
            if not self._has_receipt(page, receipts, operation):
                continue
            if boundary is not None and receipts[page["call_id"]]["completed_index"] >= boundary:
                continue
            pair = receipts[page["call_id"]]
            result, arguments = pair["completed"]["response"]["result"], pair["started"]["arguments"]
            query = copy.deepcopy(arguments)
            if (query != page.get("query") or result.get("inventory_sha256") != inventory_sha
                    or page.get("total") != result.get("total") or page.get("count") != len(result.get("records", []))
                    or page.get("offset") != query.get("offset", 0)
                    or page.get("item_ids") != [item["item_id"] for item in result["records"]]
                    or any({key: value for key, value in item.items()
                            if key not in ("original_text", "preview", "text_truncated", "text_complete")}
                           != {key: value for key, value in catalog.get(item.get("item_id"), {}).items()
                               if key != "original_text"} for item in result["records"])
                    or type(page.get("offset")) is not int or page["offset"] < 0):
                raise DraftError("source inventory page differs from retained tool response")
            if independent and digest(result) != digest(getattr(self.project, operation)(**arguments)):
                raise DraftError("source inventory page differs from independent recomputation")
            if query.get("item_id") is None:
                if page["total"] != len(catalog):
                    raise DraftError("source page has incorrect complete-inventory total")
                pages.append(page)
            seen.update(page["item_ids"])
        return self._covers(pages), seen

    def _source_reads(self, record, receipts, before_call_id, *, independent=False):
        """Credit returned original ranges, never a preview or self-reported read."""
        boundary = receipts.get(before_call_id, {}).get("started_index")
        if boundary is None:
            return False
        needed = record.get("source_spans", [record["span"]])
        cache = self.__dict__.setdefault("_read_cache", {})
        cache_key = (id(receipts), boundary, independent)
        if cache_key in cache and cache[cache_key][0] is receipts:
            returned = cache[cache_key][1]
        else:
            returned = []
            for pair in receipts.values():
                start, end = pair.get("started"), pair.get("completed")
                if not start or not end or pair["completed_index"] >= boundary:
                    continue
                response = end.get("response", {})
                if (response.get("ok") is not True or response.get("call_id") != start["call_id"]
                        or start.get("operation") != end.get("operation")
                        or pair["started_index"] >= pair["completed_index"]):
                    continue
                operation, result = start["operation"], response.get("result", {})
                candidates = []
                if operation in SOURCE_OPERATIONS.values():
                    for item in result.get("records", []):
                        spans = item.get("source_spans", [item.get("span")])
                        text = item.get("original_text")
                        if (isinstance(text, list) and len(text) == len(spans)
                                and all(isinstance(span, dict) for span in spans)
                                and text == [self.project.resolve_span(span) for span in spans]):
                            candidates.extend(spans)
                elif operation in ("read", "read_section", "read_paragraph"):
                    if "span" in result and result.get("text") == self.project.resolve_span(result["span"]):
                        candidates.append(result["span"])
                    for item in result.get("records", []):
                        spans = item.get("source_spans", [item.get("span")])
                        if item.get("original_text") == [self.project.resolve_span(span) for span in spans]:
                            candidates.extend(spans)
                    for segment in result.get("segments", []):
                        if segment.get("text") == self.project.resolve_span(segment["span"]):
                            candidates.append(segment["span"])
                if candidates and independent and operation in ("read", "read_section", "read_paragraph"):
                    if digest(result) != digest(getattr(self.project, operation)(**start["arguments"])):
                        raise DraftError("retained source read differs from independent recomputation")
                returned.extend(candidates)
            cache[cache_key] = (receipts, returned)
        for span in needed:
            cursor = span["char_start"]
            for value in sorted((value for value in returned if value["file"] == span["file"]),
                                key=lambda value: value["char_start"]):
                if value["char_start"] <= cursor:
                    cursor = max(cursor, value["char_end"])
            if cursor < span["char_end"]:
                return False
        return True

    def extraction_bindings(self, kind, offset=0, limit=100):
        if type(offset) is not int or offset < 0 or type(limit) is not int or not 1 <= limit <= 100:
            raise DraftError("invalid extraction binding pagination")
        catalog, inventory_sha = self._source_catalog(kind)
        complete, seen = self._source_page_state(kind, inventory_sha, catalog, self._receipts(self._trace()))
        identifiers = list(catalog)
        return {"kind": kind, "inventory_sha256": inventory_sha, "total": len(catalog),
                'record_format': 'metadata_only',
                "records": [{"item_id": identifier, "record": {key: value for key, value in catalog[identifier].items() if key != "original_text"},
                    **self._source_binding(kind, catalog[identifier], inventory_sha),
                    "source_page_seen": identifier in seen, "source_pages_complete": complete}
                    for identifier in identifiers[offset:offset + limit]],
                "next_offset": offset + limit if offset + limit < len(catalog) else None,
                "notice": "Bindings contain metadata only: record has no original_text, and a missing text key does not mean empty source. "
                          "Read actual original_text from a complete inventory page, or segments[].text from read_paragraph; complete original-read coverage remains required. "
                          "Source dispositions and returned reading ranges are checked, not mathematical truth."}

    def _pure_prose_markup(self, record):
        # Only a host-inventoried, wholly structural fragment gets this
        # negative-review exception. Never treat an arbitrary TeX command or
        # a substring of mathematical source as a source-use decision.
        text = "".join(self.project.resolve_span(span)
                       for span in record.get("source_spans", [record["span"]]))
        command = (r"\\(?:smallskip|medskip|bigskip|noindent|par|newpage|pagebreak|maketitle)\b"
                   r"|\\(?:begin|end)\s*\{(?:document|itemize|enumerate|description|"
                   r"thebibliography|center|flushleft|flushright|quote|quotation)\}"
                   r"|\\label\s*\{[A-Za-z0-9_.:@*+\-]+\}")
        if not text.strip() or not re.fullmatch(r"(?:\s|" + command + r")+", text):
            return False
        names = set(re.findall(r"\\([A-Za-z]+)", text))
        # A project macro can override a familiar layout spelling. Such a
        # fragment is not certified as built-in syntax by its name alone.
        environment_names = set(re.findall(r"\\(?:begin|end)\s*\{([^{}]+)\}", text))
        declared = {value.get("name") for value in self.project.structure_data.get("definitions", [])}
        return not (names | environment_names).intersection(declared)

    def _prose_unresolved_receipt(self, scopes, receipts, before_call_id):
        # A missing textual anchor is an interpretation gap, never evidence of
        # absent mathematics. Bind only a prior actual add_unresolved receipt
        # that remains identical in the current draft and names this exact item.
        if not receipts or before_call_id not in receipts:
            return None
        boundary = receipts[before_call_id].get("started_index")
        if boundary is None:
            return None

        def exact_ranges(spans):
            normalized = [self.project.normalize_span(span) for span in spans]
            occurrences = [self._reading_occurrence(span) for span in normalized]
            if not normalized or any(value is None for value in occurrences):
                return None
            return sorted((span["file"], span["char_start"], span["char_end"], occurrence)
                          for span, occurrence in zip(normalized, occurrences))

        required = exact_ranges(scopes)
        if required is None:
            return None
        current = self.draft.data["unresolved"]
        for call_id, pair in sorted(receipts.items()):
            start, end = pair.get("started"), pair.get("completed")
            if (not start or not end or start.get("operation") != "add_unresolved"
                    or end.get("operation") != "add_unresolved"
                    or not pair["started_index"] < pair["completed_index"] < boundary):
                continue
            response = end.get("response", {})
            result = response.get("result")
            if (not isinstance(result, dict) or type(result.get("revision")) is not int
                    or end.get("draft_revision") != result["revision"]
                    or not 1 <= result["revision"] <= self.draft.data["revision"]):
                continue
            record = {key: value for key, value in result.items() if key != "revision"}
            if (response.get("ok") is not True or response.get("call_id") != call_id
                    or record not in current
                    or record.get("kind") not in ("node_boundary", "mathematical_judgment")
                    or not record.get("spans") or exact_ranges(record["spans"]) != required):
                continue
            args = start["arguments"]
            expected = {"id": args.get("id", "unresolved:" + digest(args)[:20]),
                        "reason": args.get("reason"),
                        "spans": [self.project.normalize_span(span) for span in args.get("spans", [])],
                        "kind": args.get("kind", "mathematical_judgment")}
            if (record != expected or not isinstance(record.get("reason"), str)
                    or not record["reason"].strip() or exact_ranges(record.get("spans", [])) != required):
                continue
            return {"id": record["id"], "record_sha256": digest(record),
                    "call_id": call_id, "arguments_sha256": start["arguments_sha256"],
                    "result_sha256": digest(result)}
        return None

    def _source_review_value(self, kind, submitted, record, inventory_sha,
                             *, receipts=None, before_call_id=None):
        if not isinstance(submitted, dict) or submitted.get("item_id") != record["item_id"]:
            raise DraftError("source review has unknown item ID")
        binding = self._source_binding(kind, record, inventory_sha)
        if submitted.get("review_binding") != binding["review_binding"]:
            raise WorkflowIncomplete("source item or related node/relations changed")
        explanation = submitted.get("explanation")
        if not isinstance(explanation, str) or not explanation.strip():
            raise DraftError("source review needs a specific source-use or exclusion explanation")
        evidence = submitted.get("evidence_spans")
        if not isinstance(evidence, list) or not evidence:
            raise DraftError("source review requires original-source evidence")
        evidence = [self.project.normalize_span(span) for span in evidence]
        if not self._overlaps(evidence, record.get("source_spans", [record["span"]])):
            raise DraftError("source review evidence must reach the reviewed original item")
        scopes = record.get("source_spans", [record["span"]])
        static_external = kind == "reference" and bool(record.get("targets")) and all(t.get("resolution") == "external" for t in record["targets"])
        pure_markup_exclusion = (kind == "prose" and submitted.get("decision") == "reviewed"
                                 and submitted.get("node_ids") == [] and self._pure_prose_markup(record))
        unresolved_receipt = None
        anchorless_prose = (kind == "prose" and submitted.get("decision") == "reviewed"
                            and submitted.get("node_ids") == [] and not binding["related_node_ids"]
                            and not pure_markup_exclusion and not self._decision_fragments(scopes))
        if anchorless_prose:
            unresolved_receipt = self._prose_unresolved_receipt(scopes, receipts, before_call_id)
            if unresolved_receipt is None:
                raise DraftError("anchorless prose requires a prior successful add_unresolved for its exact full source item and occurrence")
        if static_external or pure_markup_exclusion or unresolved_receipt:
            if not self._covers_original_fragments(evidence, scopes):
                raise DraftError("structural, external or unresolved source decision must cover its complete original item")
        else:
            self._require_decision_fragment(evidence, scopes, reference=kind == "reference",
                                            source_kind=kind, item_id=record['item_id'])
        value = {"item_id": record["item_id"], **binding, "explanation": explanation,
                 "evidence_spans": evidence, "original_text": [self.project.resolve_span(span) for span in evidence]}
        if unresolved_receipt:
            value.update(disposition="unresolved", unresolved_receipt=unresolved_receipt)
        if kind in ("explicit", "prose"):
            decision, identifiers = submitted.get("decision"), submitted.get("node_ids")
            allowed = ("accepted", "excluded", "unresolved") if kind == "explicit" else ("reviewed",)
            if decision not in allowed or not isinstance(identifiers, list) or len(set(identifiers)) != len(identifiers):
                raise DraftError("invalid node-round decision or node_ids")
            for identifier in identifiers:
                node = self._nodes().get(identifier)
                if not node or not self._overlaps(node["spans"], record.get("source_spans", [record["span"]])):
                    raise DraftError("source-round node is missing or outside reviewed source item")
            if kind == "explicit" and ((decision == "accepted") != bool(identifiers)):
                raise DraftError("accepted explicit item needs node IDs; excluded/unresolved items do not admit nodes")
            value.update(decision=decision, node_ids=sorted(identifiers))
            if kind == "explicit":
                value["category_hint_mismatches"] = [{"node_id": identifier,
                    "category_hint": record.get("category_hint"), "accepted_category": self._nodes()[identifier]["category"],
                    "explanation": explanation} for identifier in sorted(identifiers)
                    if record.get("category_hint") is not None
                    and self._nodes()[identifier]["category"] != record["category_hint"]]
        else:
            decisions = submitted.get("key_decisions")
            keys = list(dict.fromkeys(record.get("keys", []))) or [None]
            if (not isinstance(decisions, list) or len(decisions) != len(keys)
                    or {digest(item.get("key")) for item in decisions} != {digest(key) for key in keys}):
                raise DraftError("reference review must decide every referenced key exactly once")
            normalized = []
            for decision in decisions:
                key, disposition = decision.get("key"), decision.get("disposition")
                if disposition not in ("dependency", "nondependency", "nonmath", "external", "unresolved", "missing_endpoint"):
                    raise DraftError("unknown reference disposition")
                detail, relation_ids = decision.get("explanation"), decision.get("relation_ids")
                if (not isinstance(detail, str) or not detail.strip() or not isinstance(relation_ids, list)
                        or len(set(relation_ids)) != len(relation_ids)
                        or (disposition == "dependency") != bool(relation_ids)):
                    raise DraftError("reference decision needs explanation and exactly its accepted dependency IDs")
                targets = [span for target in record.get("targets", []) if target.get("key") == key
                           for span in target.get("target_spans", []) + target.get("owner_source_spans", [])]
                endpoints = [endpoint for endpoint in record.get("targets", []) if endpoint.get("key") == key]
                has_heading_anchors = any(endpoint.get('heading_anchors') for endpoint in endpoints)
                for identifier in relation_ids:
                    relation = self.draft.data["relations"].get(identifier)
                    if not relation or relation["judgment"] != "accepted" or relation["type"] not in ("statement_dependency", "proof_dependency"):
                        raise DraftError("reference dependency must name a current accepted mathematical dependency")
                    source, target = self._nodes()[relation["source"]], self._nodes()[relation["target"]]
                    contexts = [span for context in self._node_content(target)["contexts"] for span in context["record"]["spans"]]
                    if not self._overlaps(target["spans"] + contexts, record.get("source_spans", [record["span"]])):
                        raise DraftError("reference dependency target does not reach source use/context")
                    from .candidates import _reference_sources, _paired_proof_sources
                    matches_heading = has_heading_anchors and any(node['id'] == source['id'] for endpoint in endpoints
                        for node, _, _ in _reference_sources(list(self._nodes().values()), endpoint,
                            occurrence_resolver=self._reading_occurrence,
                            heading_owner_filter=self._accepted_heading_owner))
                    wrong_owner = ((has_heading_anchors and not matches_heading)
                        or (not has_heading_anchors and targets and not self._overlaps(source["spans"], targets)))
                    if wrong_owner:
                        # A statement locator also addresses a separately stored
                        # proof through its current accepted proves association.
                        # This only resolves the source owner; it does not establish
                        # actual mathematical use, which still needs review evidence.
                        # Unlike a retrieval hint, association is not restricted to
                        # literal English proof-of wording.
                        paired = (relation["type"] == "proof_dependency" and target["category"] == "proof"
                                  and any(proof["id"] == source["id"]
                                     and (has_heading_anchors or self._overlaps(statement["spans"], targets))
                                     for endpoint in endpoints
                                     for proof, statement, _, _, _ in _paired_proof_sources(
                                         list(self._nodes().values()), self.draft.data["relations"], endpoint,
                                         occurrence_resolver=self._reading_occurrence,
                                         heading_owner_filter=self._accepted_heading_owner)))
                        if not paired:
                            raise DraftError("reference dependency source does not match referenced target")
                    if self.state.get('use_witness_binding') is not None and not any(
                            witness['reference_id'] == record['item_id'] and witness['reference_key'] == key
                            for witness in relation.get('use_witnesses', [])):
                        raise DraftError('reference dependency must have an accepted use witness for this exact occurrence/key')
                normalized.append({"key": key, "disposition": disposition, "explanation": detail,
                                   "relation_ids": sorted(relation_ids)})
            value["key_decisions"] = sorted(normalized, key=lambda item: digest(item["key"]))
        return value

    def record_extraction_reviews(self, arguments, call_id):
        self._individual_batch("review_extraction", arguments)
        kind, reviews = arguments.get("kind"), arguments.get("reviews")
        if set(arguments) != {"kind", "reviews"} or not isinstance(reviews, list) or not 1 <= len(reviews) <= 100:
            raise DraftError("review_extraction requires kind and 1..100 reviews")
        catalog, inventory_sha = self._source_catalog(kind)
        receipts = self._receipts(self._trace())
        _, seen = self._source_page_state(kind, inventory_sha, catalog, receipts, before_call_id=call_id)
        if kind == "prose" and not self._inspect_source_stage("explicit", receipts)["complete"]:
            raise WorkflowIncomplete("finish the current explicit-node round before reviewing prose")
        prepared, identifiers = [], set()
        for submitted in reviews:
            identifier = submitted.get("item_id") if isinstance(submitted, dict) else None
            if identifier not in catalog or identifier in identifiers or identifier not in seen:
                raise DraftError("source review item must be unique and previously returned by its inventory tool")
            identifiers.add(identifier)
            if not self._source_reads(catalog[identifier], receipts, call_id):
                raise WorkflowIncomplete("read the complete original ranges before source review")
            value = self._source_review_value(kind, submitted, catalog[identifier], inventory_sha,
                                              receipts=receipts, before_call_id=call_id)
            prepared.append({**value, "call_id": call_id, "arguments_sha256": digest(arguments)})
        self._individual_decisions(receipts, new_source=prepared, source_kind=kind)
        # The existing structural draft contract also tracks syntax candidates.
        # Reuse the actual source decision; never require a second ceremonial
        # review or mark an excluded environment as an accepted math node.
        updates = [{"event_id": catalog[value["item_id"]].get("event_id"),
                    "outcome": value["decision"], "node_ids": value["node_ids"],
                    "explanation": value["explanation"]} for value in prepared
                   if kind == "explicit" and catalog[value["item_id"]].get("event_id") in self.project._events]
        if updates:
            self.draft.apply("review_candidates", {"reviews": updates})
        output = {"kind": kind, "reviewed": len(prepared), "item_ids": sorted(identifiers),
                  "inventory_sha256": inventory_sha, "mathematical_truth": "not_certified"}
        for record in prepared:
            record["result_sha256"] = digest(output)
            self.state["source_reviews"][kind][record["item_id"]] = record
        self.state["history"].append({"kind": "extraction_reviews", "stage": kind, "records": copy.deepcopy(prepared)})
        self._save()
        return output

    def _inspect_source_stage(self, kind, receipts, *, independent=False):
        catalog, inventory_sha = self._source_catalog(kind)
        pages_complete, _ = self._source_page_state(kind, inventory_sha, catalog, receipts, independent=independent)
        reviewed, missing, admitted, decisions, category_mismatches = [], [], set(), {}, []
        for identifier, source_record in catalog.items():
            audit = self.state["source_reviews"][kind].get(identifier)
            reason = None
            if not audit or not self._has_receipt(audit, receipts, "review_extraction"):
                reason = "no_successfully_completed_source_review"
            else:
                arguments = receipts[audit["call_id"]]["started"]["arguments"]
                submitted = [item for item in arguments.get("reviews", []) if item.get("item_id") == identifier]
                try:
                    _, old_seen = self._source_page_state(kind, inventory_sha, catalog, receipts,
                                                         before_call_id=audit["call_id"])
                    if arguments.get("kind") != kind or len(submitted) != 1 or identifier not in old_seen:
                        raise DraftError("source-review arguments or prior page receipt differ")
                    expected = self._source_review_value(kind, submitted[0], source_record, inventory_sha,
                                                         receipts=receipts, before_call_id=audit["call_id"])
                    stripped = {key: value for key, value in audit.items()
                                if key not in ("call_id", "arguments_sha256", "result_sha256")}
                    if stripped != expected:
                        raise DraftError("source-review source/content binding differs")
                    if not self._source_reads(source_record, receipts, audit["call_id"], independent=independent):
                        raise DraftError("source-review complete reading receipts differ")
                    admitted.update(expected.get("node_ids", []))
                    category_mismatches.extend({"item_id": identifier, **mismatch}
                        for mismatch in expected.get("category_hint_mismatches", []))
                    for decision in expected.get("key_decisions", [expected]):
                        label = decision.get("disposition", decision.get("decision"))
                        decisions[label] = decisions.get(label, 0) + 1
                except (DraftError, TypeError, KeyError, ValueError):
                    reason = "source_node_context_relation_or_read_binding_changed"
            (missing if reason else reviewed).append({"item_id": identifier, "reason": reason} if reason else identifier)
        return {"inventory_sha256": inventory_sha, "total": len(catalog), "pages_complete": pages_complete,
                "complete": pages_complete and not missing, "reviewed_items": reviewed,
                "missing_reviews": missing, "admitted_node_ids": sorted(admitted), "disposition_counts": decisions,
                "category_hint_mismatches": category_mismatches}

    def inspect_extraction(self, *, receipts=None, independent=False):
        if self.version != VERSION:
            return {"version": self.version, "source_rounds": "not_required_by_historical_v2"}
        receipts = self._receipts(self._trace()) if receipts is None else receipts
        stages = {kind: self._inspect_source_stage(kind, receipts, independent=independent) for kind in SOURCE_OPERATIONS}
        admitted = set(stages["explicit"]["admitted_node_ids"]) | set(stages["prose"]["admitted_node_ids"])
        missing_nodes = sorted(set(self._nodes()) - admitted)
        phase = "explicit_nodes" if not stages["explicit"]["complete"] else (
            "prose_nodes" if not stages["prose"]["complete"] else "dependency_review")
        return {"stages": stages, "phase": phase, "unreviewed_accepted_nodes": missing_nodes,
                "complete": all(stage["complete"] for stage in stages.values()) and not missing_nodes,
                "mathematical_truth": "not_certified", "semantic_recall": "not_certified"}

    def _inspect_use_witnesses(self, reference_stage):
        """Reverse reconciliation is occurrence-local, never a pair-wide veto."""
        errors, checked = [], []
        catalog, _ = self._source_catalog('reference')
        reviewed = set(reference_stage['reviewed_items'])
        for relation in self.draft.data['relations'].values():
            if (relation['judgment'] != 'accepted'
                    or relation['type'] not in ('statement_dependency', 'proof_dependency')):
                continue
            try:
                expected = self.draft._use_witnesses(relation, relation.get('use_witnesses'))
                if expected != relation.get('use_witnesses'):
                    raise DraftError('current source or endpoint role differs from retained witness')
                for witness in expected:
                    reference_id, key = witness['reference_id'], witness['reference_key']
                    if reference_id is None:
                        # A witness cannot silently relabel an exact referenced
                        # endpoint as implicit. Unrelated navigation is allowed.
                        for row in catalog.values():
                            if not self._overlaps([witness['span']], row.get('source_spans', [row['span']])):
                                continue
                            for refkey in row.get('keys', []):
                                endpoints = [target for target in row.get('targets', []) if target.get('key') == refkey]
                                targets = [span for target in endpoints
                                           for span in target.get('target_spans', []) + target.get('owner_source_spans', [])]
                                source = self._nodes()[relation['source']]
                                matches = targets and self._overlaps(source['spans'], targets)
                                if any(endpoint.get('heading_anchors') for endpoint in endpoints):
                                    from .candidates import _reference_sources, _paired_proof_sources
                                    matches = any(node['id'] == source['id'] for endpoint in endpoints
                                        for node, _, _ in _reference_sources(list(self._nodes().values()), endpoint,
                                            occurrence_resolver=self._reading_occurrence,
                                            heading_owner_filter=self._accepted_heading_owner))
                                    if not matches and source['category'] == 'proof':
                                        matches = any(proof['id'] == source['id'] for endpoint in endpoints
                                            for proof, _, _, _, _ in _paired_proof_sources(
                                                list(self._nodes().values()), self.draft.data['relations'], endpoint,
                                                occurrence_resolver=self._reading_occurrence,
                                                heading_owner_filter=self._accepted_heading_owner))
                                if not matches and source['category'] == 'proof':
                                    statements = [self._nodes()[r['target']] for r in self.draft.data['relations'].values()
                                                  if r['type'] == 'proves' and r['judgment'] == 'accepted'
                                                  and r['source'] == source['id']]
                                    matches = any(self._overlaps(s['spans'], targets) for s in statements)
                                if matches:
                                    raise DraftError('implicit witness contains an explicit reference to its source endpoint')
                        continue
                    if reference_id not in reviewed:
                        raise DraftError('explicit use lacks a current independently replayed reference review')
                    audit = self.state['source_reviews']['reference'][reference_id]
                    decisions = [value for value in audit['key_decisions'] if value['key'] == key]
                    if (len(decisions) != 1 or decisions[0]['disposition'] != 'dependency'
                            or relation['id'] not in decisions[0]['relation_ids']):
                        raise DraftError('accepted explicit use disagrees with the exact reference disposition')
                checked.append(relation['id'])
            except (DraftError, KeyError, TypeError, ValueError) as exc:
                errors.append({'relation_id': relation['id'], 'reason': str(exc)})
        return {'binding': Draft.USE_WITNESS_BINDING, 'complete': not errors,
                'checked_dependency_ids': sorted(checked), 'errors': errors,
                'mathematical_truth': 'not_certified'}

    def node_content_bindings(self):
        """Host-only mutation snapshot; incomplete Context chains bind to null."""
        bindings = {}
        for node_id, node in self._nodes().items():
            try:
                bindings[node_id] = digest(self._node_content(node))
            except DraftError:
                bindings[node_id] = None
        return bindings

    def inspect_node_readbacks(self, *, receipts=None):
        """Check actual successful get_node replies against all current bytes.

        Equality includes the entire current Context ancestry. Dependency
        changes, unrelated Context edits and byte-identical updates do not
        invalidate a read. A -> B -> A does: the trace retains each effective
        change. No reply is a mathematical acceptance certificate.
        """
        self._validate_state()
        if receipts is None:
            receipts = self._receipts(self._trace())
        read, missing, snapshot_errors = [], [], []
        ordered = sorted(receipts.items(), key=lambda row: row[1].get('started_index', -1))
        snapshots = []
        for call_id, pair in ordered:
            start, end = pair.get('started'), pair.get('completed')
            if (not start or not end or start.get('operation') not in
                    ('upsert_node', 'upsert_nodes', 'upsert_context', 'remove_node')
                    or end.get('response', {}).get('ok') is not True):
                continue
            bindings = end.get('node_content_bindings')
            if (end.get('operation') != start.get('operation')
                    or pair['started_index'] >= pair['completed_index']
                    or not isinstance(bindings, dict)
                    or any(not isinstance(key, str) or not key
                           or (value is not None and
                               (not isinstance(value, str) or not re.fullmatch(r'[0-9a-f]{64}', value)))
                           for key, value in bindings.items())):
                snapshot_errors.append({'call_id': call_id,
                                        'reason': 'missing_or_invalid_host_mutation_snapshot'})
            else:
                snapshots.append((pair['completed_index'], bindings))
        for node_id, node in self._nodes().items():
            content = self._node_content(node)
            current_digest = digest(content)
            previous, last_change = 'absent', -1
            for index, bindings in snapshots:
                binding = bindings.get(node_id, 'absent')
                if binding != previous:
                    last_change = index
                previous = binding
            if previous != current_digest:
                snapshot_errors.append({'node_id': node_id,
                                        'reason': 'final_host_mutation_snapshot_differs'})
            expected = {**copy.deepcopy(content['node']),
                        'original_text': content['original_text'],
                        'context_record': content['contexts'][0]['record'],
                        'context_chain': content['contexts']}
            eligible = []
            for call_id, pair in receipts.items():
                start, end = pair.get('started'), pair.get('completed')
                if (start and end and start.get('operation') == 'get_node'
                        and end.get('operation') == 'get_node'
                        and pair['started_index'] < pair['completed_index']
                        and pair['started_index'] > last_change
                        and start.get('arguments', {}).get('id') == node_id
                        and end.get('response', {}).get('ok') is True
                        and end['response'].get('call_id') == call_id
                        and end['response'].get('result') == expected):
                    eligible.append(call_id)
            if eligible:
                read.append({'node_id': node_id, 'call_id': max(eligible),
                             'current_content_sha256': current_digest})
            else:
                missing.append(node_id)
        return {'binding': Draft.NODE_CONTENT_POLICY, 'complete': not missing and not snapshot_errors,
                'current_reads': read, 'missing_current_node_reads': missing,
                'mutation_snapshot_errors': snapshot_errors,
                'mathematical_truth': 'not_certified'}

    def inspect(self, *, independent=False, trace=None):
        self._validate_state()
        nodes, inventory_sha = self._inventory()
        receipts = self._receipts(self._trace() if trace is None else trace)
        covered, candidates, catalog, _ = self._candidate_state(inventory_sha, nodes, receipts, independent=independent)
        reviewed, missing, disposition_counts, unresolved = [], [], {key: 0 for key in ("accepted", "excluded", "unresolved")}, []
        for node_id in nodes:
            audit = self.state["audits"].get(node_id)
            binding = self._binding(node_id, inventory_sha, catalog[node_id])
            reason = None
            if not audit:
                reason = "no_source_bound_review"
            elif any(audit.get(key) != value for key, value in binding.items()):
                reason = "node_inventory_context_or_incoming_relations_changed"
            elif not self._has_receipt(audit, receipts, "review_relations"):
                reason = "review_call_not_successfully_completed"
            elif node_id not in covered:
                reason = "candidate_pages_incomplete"
            else:
                spans = audit.get("evidence_spans", [])
                submitted = receipts[audit['call_id']]['started']['arguments'].get('reviews', [])
                submitted = [record for record in submitted if record.get('node_id') == node_id]
                if (not spans or not isinstance(audit.get("explanation"), str)
                        or not audit["explanation"].strip()
                        or [self.project.resolve_span(span) for span in spans] != audit.get("original_text")
                        or sorted(audit.get("reviewed_relation_ids", [])) != sorted(binding["incoming_relation_ids"])
                        or not any(a["file"] == b["file"] and a["char_start"] < b["char_end"]
                                   and b["char_start"] < a["char_end"] for a in spans for b in nodes[node_id]["spans"])
                        or len(submitted) != 1
                        or submitted[0].get('review_binding') != binding['review_binding']
                        or submitted[0].get('explanation') != audit['explanation']
                        or sorted(submitted[0].get('reviewed_relation_ids', [])) != sorted(binding['incoming_relation_ids'])
                        or [self.project.normalize_span(span) for span in submitted[0].get('evidence_spans', [])] != spans):
                    reason = "review_source_or_incoming_set_differs"
                else:
                    try:
                        scopes = nodes[node_id]["spans"] + [s for c in self._node_content(nodes[node_id])["contexts"]
                                                          for s in c["record"]["spans"]]
                        self._require_decision_fragment(spans, scopes)
                        old_covered, _, old_catalog, old_page_calls = self._candidate_state(
                            inventory_sha, nodes, receipts, before_call_id=audit["call_id"])
                        dispositions = self._dispositions(submitted[0].get("candidate_dispositions"), old_catalog[node_id], node_id)
                        if (node_id not in old_covered or old_catalog[node_id] != catalog[node_id]
                                or audit.get("candidate_page_call_ids") != old_page_calls[node_id]
                                or audit.get("candidate_dispositions") != dispositions):
                            reason = "candidate_dispositions_or_prior_page_receipts_differ"
                        else:
                            for disposition in dispositions:
                                disposition_counts[disposition["judgment"]] += 1
                                if disposition["judgment"] == "unresolved":
                                    unresolved.append({"target": node_id, **copy.deepcopy(disposition)})
                    except DraftError:
                        reason = "candidate_dispositions_or_prior_page_receipts_differ"
            (missing if reason else reviewed).append({"node_id": node_id, "reason": reason} if reason else node_id)
        report = {"version": self.version, "complete": bool(nodes) and len(reviewed) == len(nodes)
                and len(covered) == len(nodes), "inventory_sha256": inventory_sha,
                "accepted_nodes": len(nodes), "reviewed_nodes": reviewed, "missing_reviews": missing,
                "candidate_disposition_counts": disposition_counts,
                "unresolved_candidates": unresolved,
                "unresolved_candidate_pairs": [{"source": record["source"], "target": record["target"],
                                                 "type": record["type"]} for record in unresolved],
                "candidate_coverage": candidates, "mathematical_truth": "not_certified",
                "dependency_recall": "not_certified",
                "notice": "This verifies completed candidate pagination and source-bound Agent review records, not mathematical reliability."}
        if self.version == VERSION:
            source_review = self.inspect_extraction(receipts=receipts, independent=independent)
            nonempty = sorted(key for key, node in nodes.items() if node.get("semantics"))
            report.update(extraction=source_review, unverified_semantics_nodes=nonempty)
            report["complete"] = report["complete"] and source_review["complete"] and not nonempty
            if self.state.get('use_witness_binding') is not None:
                witnesses = self._inspect_use_witnesses(source_review['stages']['reference'])
                report['use_witnesses'] = witnesses
                report['complete'] = report['complete'] and witnesses['complete']
            report["phase"] = "ready_for_host" if report["complete"] else source_review["phase"]
        if self.state.get('node_content_policy') is not None:
            readbacks = self.inspect_node_readbacks(receipts=receipts)
            report['node_readbacks'] = readbacks
            report['complete'] = report['complete'] and readbacks['complete']
            if not readbacks['complete']:
                report['phase'] = 'current_node_readback_required'
        self._individual_decisions(receipts)
        return report

    def require_complete(self, graph=None):
        if graph is not None:
            rebuilt = self.draft.to_graph(graph.snapshot_id)
            if rebuilt != graph:
                raise WorkflowIncomplete("workflow draft does not reproduce the retained graph")
        report = self.inspect(independent=True)
        if not report["complete"]:
            raise WorkflowIncomplete("dependency review workflow incomplete: " + json.dumps({
                "candidate_targets": report["candidate_coverage"]["missing_targets"],
                "reviews": report["missing_reviews"],
                "node_readbacks": report.get('node_readbacks')}, ensure_ascii=False))
        return copy.deepcopy(report)

    def export_report(self, graph):
        summary = self.require_complete(graph=graph)
        trace = self._trace()
        return {"version": self.version, "source_fingerprint": copy.deepcopy(self.project.fingerprint),
                "graph_sha256": digest(graph.model_dump(mode="json")),
                "draft_data": copy.deepcopy(self.draft.data), "workflow_state": copy.deepcopy(self.state),
                "trace_sha256": hashlib.sha256(trace).hexdigest(), "summary": summary,
                "mathematical_truth": "not_certified", "dependency_recall": "not_certified"}

    @classmethod
    def verify_retained(cls, project, graph, report, trace):
        try:
            return cls._verify_retained(project, graph, report, trace)
        except (KeyError, TypeError, AttributeError, IndexError) as exc:
            raise WorkflowIncomplete("malformed retained workflow records") from exc

    @classmethod
    def _verify_retained(cls, project, graph, report, trace):
        if (not isinstance(report, dict) or report.get("version") != VERSION
                or report.get("source_fingerprint") != project.fingerprint
                or report.get("graph_sha256") != digest(graph.model_dump(mode="json"))
                or not isinstance(trace, bytes)
                or report.get("trace_sha256") != hashlib.sha256(trace).hexdigest()):
            raise WorkflowIncomplete("retained workflow graph, source or trace binding differs")
        data = report.get("draft_data")
        if not isinstance(data, dict) or data.get("source_fingerprint") != project.fingerprint:
            raise WorkflowIncomplete("retained draft source binding differs")
        # Rebuild from actual retained records and current source bytes. No work
        # directory or model-provided completion label is used as authority.
        draft = Draft.__new__(Draft)
        draft.project, draft.data = project, copy.deepcopy(data)
        gate = cls.__new__(cls)
        gate.project, gate.draft = project, draft
        gate.state = copy.deepcopy(report.get("workflow_state"))
        gate._validate_state()
        if gate.version != report["version"]:
            raise WorkflowIncomplete("retained report and checkpoint workflow versions differ")
        if draft.to_graph(graph.snapshot_id) != graph:
            raise WorkflowIncomplete("retained workflow records do not reproduce graph")
        summary = gate.inspect(independent=True, trace=trace)
        if not summary["complete"] or summary != report.get("summary"):
            raise WorkflowIncomplete("retained workflow does not independently close all required reviews")
        return copy.deepcopy(summary)
