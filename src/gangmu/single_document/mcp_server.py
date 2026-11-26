"""Small JSON-RPC stdio MCP server; exposes only source-bound operations.

No network, generic file, shell, process or evaluation-reference tool exists.
The JSON lines transport is deliberately dependency-free.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable


TOOL = {"name":"document", "description":
    "Read and inspect the pinned mathematical source, search its entire text, "
    "inspect structural candidates, and persist extraction decisions. Use overview first. "
    "Operations include overview, read, read_section, search, find_text, structure, labels, symbols, "
    "candidates, dependency_candidates, relation_candidates, explicit_inventory, paragraph_inventory, reference_ledger, read_paragraph, status, list_nodes, get_node, target_use_fragments, retention_status, list_relations, "
    "upsert_node/upsert_nodes, upsert_relation/upsert_relations, remove_node, remove_relation, "
    "upsert_context, review_candidates, extraction_bindings, review_extraction, review_bindings, review_relations, workflow_status, "
    "add_unresolved, coverage, checkpoint, validate, finalize/finish. "
    "get_node returns original_text and the full context_chain with original_text. "
    "retention_status takes spans (1..32 exact source ranges) and optional node_id; it compares saved Unit/Context "
    "ranges with actual accepted-node body/Context-chain/direct statement/proof-dependency bindings. It reports stored, "
    "accessible and uncovered ranges, real consumers and content hashes; isolated Context, descriptions and "
    "evidence-only windows do not imply node coverage. It is read-only and supplies no complete source-read receipt "
    "or applicability/semantic sufficiency certificate; dependency source Context is its own background, not "
    "inherited target scope. Ambiguous repeated-include occurrences stay explicit. "
    "target_use_fragments returns exact complete source_spans/original_text and span when single-file, with node_id/offset/limit pagination. "
    "extraction_bindings records are metadata only, never empty source text; read actual original_text from inventory or read_paragraph segments[].text. "
    "Source-fragment and individual-explanation failures include specific error_details. "
    "Only source files belonging to this document are readable. Arguments are operation-specific JSON.",
    "annotations":{"readOnlyHint":False,"destructiveHint":False,"openWorldHint":False},
    "inputSchema":{"type":"object","properties":{"operation":{"type":"string"},
                   "arguments":{"type":"object","additionalProperties":True}},
                   "required":["operation","arguments"],"additionalProperties":False}}


def serve(call: Callable[[str,dict[str,Any]],dict[str,Any]]) -> None:
    for line in sys.stdin:
        try:
            message=json.loads(line)
            method=message.get('method'); request_id=message.get('id')
            if request_id is None: continue
            params=message.get('params') or {}
            if method=='initialize':
                result={'protocolVersion':params.get('protocolVersion','2024-11-05'),
                        'capabilities':{'tools':{'listChanged':False}},
                        'serverInfo':{'name':'gangmu-source-tools','version':'0.1.0'}}
            elif method=='tools/list': result={'tools':[TOOL]}
            elif method=='tools/call':
                if params.get('name')!='document': raise ValueError('Unknown tool')
                data=params.get('arguments') or {}
                try:
                    value=call(data['operation'],data.get('arguments') or {})
                    result={'content':[{'type':'text','text':json.dumps(value,ensure_ascii=False)}],
                            'isError':False}
                except (ValueError, KeyError, TypeError) as exc:
                    result={'content':[{'type':'text','text':json.dumps({'error':str(exc)},ensure_ascii=False)}],
                            'isError':True}
            elif method=='ping': result={}
            elif method in ('resources/list','resources/templates/list','prompts/list'):
                key={'resources/list':'resources','resources/templates/list':'resourceTemplates',
                     'prompts/list':'prompts'}[method]; result={key:[]}
            else:
                print(json.dumps({'jsonrpc':'2.0','id':request_id,'error':{'code':-32601,'message':'Unknown method'}}),flush=True)
                continue
            print(json.dumps({'jsonrpc':'2.0','id':request_id,'result':result},ensure_ascii=False),flush=True)
        except Exception:
            # Do not echo arbitrary invalid input or host paths.
            print(json.dumps({'jsonrpc':'2.0','id':None,'error':{'code':-32603,'message':'Document server request failed'}}),flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument('--session',required=True)
    args=parser.parse_args(); spec=json.loads(Path(args.session).read_text())
    if spec.get('mode')=='probe':
        def call(operation: str, arguments: dict[str,Any]) -> dict[str,Any]:
            if operation=='overview':
                if spec.get('probe_trace'):
                    with Path(spec['probe_trace']).open('a') as log:
                        log.write(json.dumps({'operation':'overview','success':True})+'\n')
                return {'message':'document tools connection verified','only_tool':'document'}
            raise ValueError('Probe only permits overview')
    else:
        from .source import SourceProject
        from .draft import Draft
        from .tools import DocumentTools
        if 'artifact_dir' in spec:
            project=SourceProject.from_artifact(Path(spec['artifact_dir']),paper_id=spec.get('paper_id'))
        else:
            project=SourceProject.from_source(Path(spec['source_dir']),spec['main_file'],
                spec['paper_id'],spec['arxiv_id'])
        draft=Draft(project,Path(spec['work_dir']),
                    require_use_witnesses=spec.get('require_use_witnesses', False),
                    require_explicit_context_sources=spec.get('require_explicit_context_sources', False),
                    require_node_content_checks=spec.get('require_node_content_checks', False))
        tools=DocumentTools(project,draft,max_calls=spec.get('max_calls',160))
        call=tools.call
    serve(call)


if __name__=='__main__': main()
