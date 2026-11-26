"""Regular two-round production with supplied-source exclusion before top-k.

The bounded producer still owns both rounds, grouping, schemas and Context
validation. Only source-request retrieval changes. This new development
strategy requires fresh complete development acceptance before freezing tests.
"""
from __future__ import annotations

import copy
import hashlib
from pathlib import Path

from gangmu.representation.model import digest
from . import staged_nodes as base
from . import staged_nodes_context_strategy as context
from . import staged_nodes_wire_strategy as wire
from . import staged_context_request_recovery_v1 as retrieval
from .source import SourceError

VERSION="staged-nodes-source-query-v1"


class _RetrievalPipeline(wire._BoundedPipeline):
    def __init__(self,project,rule_result,judge,options,wire_options):
        super().__init__(project,rule_result,judge,options,wire_options)
        inherited=copy.deepcopy(self.binding)
        self.binding={**inherited,"producer_version":VERSION,
            "producer_code_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "parent_producer_binding":inherited,
            "parent_producer_binding_sha256":digest(inherited),
            "source_query_dependency_sha256":{
                name:hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                for name,module in (("base",base),("context",context),("wire",wire),
                                   ("bm25_query",retrieval))},
            "source_query_policy":{"method":"bm25","k1":1.2,"b":0.75,
                "include_tex_macro_names":True,"exclude_fully_supplied_before_scoring":True,
                "query_results":options.context_query_results,
                "exclude_explicit_ids_and_earlier_requests_in_same_response":True,
                "sources":"complete_rows_from_same_pinned_source_only"},
            "regular_rounds":[1,2],"new_strategy_development_required":True}

    def _query(self,query,supplied):
        return retrieval._query(self.rows,query,supplied,self.options.context_query_results)

    def _ask(self,task_kind,payload):
        # Retain the original short bounded request loop and response semantics.
        # Build seen before each query, rather than excluding only after top-k.
        payload=copy.deepcopy(payload)
        payload["producer_binding"]=self.binding
        payload["context_catalog"]=self._catalog()
        for followup in range(self.options.max_context_requests+1):
            payload["source_context_followup"]=followup
            payload["source_context_availability"]={
                "further_followups_remaining":self.options.max_context_requests-followup,
                "previous_request_found_new_source":payload.pop("_previous_request_found_new_source",None),
                "instruction":"Use only supplied original sources. If necessary meaning remains "
                    "unresolved and no informative request remains, return context_requests:[] "
                    "with the existing uncertain disposition. On the final followup, resolve "
                    "the task from supplied sources or mark it uncertain; do not invent context."}
            payload["response_schema"]=base.response_schema(task_kind,payload)
            instructions=self.prompts[task_kind]
            request_sha=digest({"task_kind":task_kind,"instructions":instructions,"payload":payload})
            try:
                response=self.judge(task_kind,instructions,copy.deepcopy(payload))
                if not isinstance(response,dict):
                    raise SourceError("judge response must be a JSON object")
                digest(response)
            except Exception as error:
                self.calls.append({"task_kind":task_kind,"input_sha256":request_sha,
                    "status":"technical_failure","error_type":type(error).__name__,
                    "source_context_followup":followup})
                raise SourceError("semantic judge task failed") from None
            self.calls.append({"task_kind":task_kind,"input_sha256":request_sha,
                "response_sha256":digest(response),"status":"returned",
                "source_context_followup":followup})
            base._check_shape(response,payload["response_schema"])
            requests=response.get("context_requests")
            if not isinstance(requests,list):
                raise SourceError("judge omitted context request disposition")
            if not requests:
                return response,{row["source_id"] for row in payload["context_menu"]}
            if followup==self.options.max_context_requests:
                raise SourceError("required source context remains unresolved")
            seen={row["source_id"] for row in payload["context_menu"]}
            seen.update(row["candidate_id"] for row in payload.get("candidates",[]))
            seen.update(row["source_id"] for row in payload.get("paragraphs",[]))
            added=[]
            for request in requests:
                if not isinstance(request,dict) or set(request)!={"source_id","query"}:
                    raise SourceError("invalid source-context request")
                identifier,query=request["source_id"],request["query"]
                if identifier is not None:
                    if identifier not in self.rows:
                        raise SourceError("unknown requested source-context ID")
                    if identifier not in seen:
                        added.append(self.rows[identifier])
                        seen.add(identifier)
                if query is not None:
                    if not isinstance(query,str) or not query.strip():
                        raise SourceError("source-context query must be nonempty text")
                    selected=self._query(query,seen)
                    added.extend(selected)
                    seen.update(row["source_id"] for row in selected)
                if identifier is None and query is None:
                    raise SourceError("empty source-context request")
            # A pinned paper may omit the requested definition. Tell the judge
            # what the source lookup found within the existing followup bound;
            # only its terminal answer may close the task. Never erase requests
            # or manufacture an uncertain/accepted mathematical decision here.
            payload["_previous_request_found_new_source"]=bool(added)
            payload["context_menu"].extend(copy.deepcopy(added))
            payload["previous_context_requests"]=copy.deepcopy(requests)
        raise AssertionError("unreachable context followup")

    def run(self):
        result=super().run()
        result["producer_version"]=VERSION
        return result


def extract_complete_nodes(project,rule_result,judge,options=None,wire_options=None):
    """Use the new source query in both original production rounds; no client."""
    return _RetrievalPipeline(project,rule_result,judge,
        base.NodeOptions.parse(options),wire_options).run()
