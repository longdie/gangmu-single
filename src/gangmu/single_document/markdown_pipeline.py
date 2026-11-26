"""Markdown discovery and explicit-use adapters for the common production module.

All model calls use the existing authenticated judge, ledger and retry gate.
Only exact original selections become MathIR; the shared implicit and topology
stages operate on that MathIR without knowing its input format.
"""
from __future__ import annotations

import copy
import json
from pathlib import Path

from gangmu.markdown_structure import labels, references
from gangmu.representation.graph import GraphIndex
from gangmu.representation.model import Context, Evidence, Graph, Relation, Unit, digest
from gangmu.representation.sources import SourceError
from .production_pipeline import ProductionPipeline, PROGRAMS
from .production_node_state import RecordingJudge
from .staged_nodes import _check_shape
from .staged_nodes_wire_strategy import WireOptions, wire_bytes
from .topology_integration import VERSION, _record, _save_once

PROMPTS = ('prompts/single_document/markdown/nodes.md', 'prompts/single_document/markdown/relations.md')
OPTIONS = {'chunk_chars': 10000, 'block_chars': 6000, 'context_chars': 6000,
           'max_followups': 2, 'relation_batch': 6, 'max_wire_bytes': 160000}


def markdown_profile(project, policy, *, root):
    profile = {'schema_version': VERSION + '.fixed-profile', 'source_format': 'markdown',
        'paper_id': project.paper_id, 'source_fingerprint': copy.deepcopy(project.fingerprint),
        'node_producer': 'markdown-source-spans-v1', 'normal_source_profile': True,
        'agent_version': policy['agent_version'], 'markdown_options': OPTIONS,
        **{key: copy.deepcopy(policy[key]) for key in ('stage_model_options', 'implicit_retrieval', 'implicit_packets', 'implicit_prompt')},
        'program_bindings': [_record(root, 'src/gangmu/single_document/' + name) for name in
            (*PROGRAMS, 'markdown_pipeline.py')]
            + [_record(root, 'src/gangmu/' + name) for name in ('markdown_structure.py',
                'representation/model.py', 'representation/graph.py', 'representation/sources.py',
                'representation/storage.py', 'runtime/config.py', 'runtime/protocol.py')],
        'prompt_bindings': [_record(root, name) for name in (*PROMPTS, policy['implicit_prompt'])],
        'runtime_precision_evaluation': False, 'automatic_e3_publication': False}
    profile['profile_sha256'] = digest(profile)
    return profile


def _object(**properties):
    return {'type': 'object', 'properties': properties, 'required': list(properties), 'additionalProperties': False}


def _array(items):
    return {'type': 'array', 'items': items}


STRING = {'type': 'string', 'minLength': 1}
SELECTION = _object(block_id=STRING, quote={'type': ['string', 'null']})
NODE_SCHEMA = _object(reviewed_blocks=_array(STRING), nodes=_array(_object(
    category={'type': 'string', 'enum': ['definition', 'statement', 'expression', 'proof']},
    label=STRING, selections=_array(SELECTION), context=_array(SELECTION))),
    unresolved=_array(_object(block_id=STRING, reason=STRING)), context_queries=_array(STRING))
RELATION_SCHEMA = _object(decisions=_array(_object(candidate_id=STRING,
    decision={'type': 'string', 'enum': ['accept', 'reject', 'unknown']}, explanation=STRING)))


def source_blocks(project, limit):
    """Keep every eligible block and mark unavoidable large-block cuts explicitly."""
    result = []
    for row in project.structure_data['blocks']:
        if row['kind'] in {'blank', 'code', 'comment'}:
            continue
        span = row['source']
        for start in range(span['char_start'], span['char_end'], limit):
            end = min(start + limit, span['char_end'])
            source = dict(span, char_start=start, char_end=end)
            result.append({**row, 'id': f'md:{start}:{end}', 'source': source,
                           'text': project.resolve_span(source),
                           'fragmented': end - start < span['char_end'] - span['char_start']})
    return result


def chunks(blocks, limit):
    current, size = [], 0
    for block in blocks:
        if current and size + len(block['text']) > limit:
            yield current
            current, size = [], 0
        current.append(block)
        size += len(block['text'])
    if current:
        yield current


def selections(project, rows, available, *, ordered=True):
    if not rows:
        raise SourceError('Markdown Unit needs original source selections')
    spans = []
    for row in rows:
        if row['block_id'] not in available:
            raise SourceError('Markdown selection is outside the supplied source blocks')
        block = available[row['block_id']]
        quote = row['quote']
        start, end = 0, len(block['text'])
        if quote is not None:
            if not quote or block['text'].count(quote) != 1:
                raise SourceError('Markdown quotation must occur exactly once in its host block')
            start = block['text'].index(quote)
            end = start + len(quote)
        source = dict(block['source'], char_start=block['source']['char_start'] + start,
                      char_end=block['source']['char_start'] + end)
        spans.append(project.normalize_span(source))
    ordered_spans = sorted(spans, key=lambda s: (s['file'], s['char_start'], s['char_end']))
    if ordered and spans != ordered_spans:
        raise SourceError('Markdown selections must follow original reading order')
    if not ordered:
        # Context is a set of exact evidence intervals, not a reconstructed
        # narrative. Normalize its presentation order without changing text.
        spans = ordered_spans
    merged = []
    for span in spans:
        if merged and span['file'] == merged[-1]['file']:
            previous = merged[-1]
            if span['char_start'] < previous['char_end']:
                if ordered:
                    raise SourceError('Markdown selections overlap')
                previous['char_end'] = max(previous['char_end'], span['char_end'])
                continue
            gap = project._read_text(span['file'])[previous['char_end']:span['char_start']]
            if not gap.strip():
                previous['char_end'] = span['char_end']
                continue
        merged.append(dict(span))
    return merged


def _graph(graph):
    return Graph.model_validate_json(json.dumps(graph, ensure_ascii=False))


def bind_relations(graph, proposals):
    """Bind evidence after all proof ownership links have been installed."""
    result = copy.deepcopy(graph)
    for proposal in proposals:
        rid = 'relation:' + digest(proposal)[:24]
        result['relations'].append(Relation(id=rid, type=proposal['type'], source=proposal['source'],
            target=proposal['target'], explanation=proposal['explanation'], evidence=('evidence:' + rid,)).model_dump(mode='json'))
    index = GraphIndex(_graph(result))
    for relation in index.graph.relations:
        if any(e['id'] == relation.evidence[0] for e in result['evidence']):
            continue
        target = index.get(relation.source if relation.type == 'proves' else relation.target, Unit)
        result['evidence'].append(Evidence(id=relation.evidence[0], method='agent_analysis', sources=target.sources,
            claim_sha256=index.claim_digest(relation, target.sources), explanation=relation.explanation).model_dump(mode='json'))
    return result


class MarkdownPipeline(ProductionPipeline):
    def for_complete_reextraction(self):
        return MarkdownPipeline(self.profile, self.judge, root=self.root, strict_nodes=True)

    def _rules(self, project, diagnosis, identity):
        if project.source_format != 'markdown':
            raise SourceError('Markdown production requires a registered Markdown source')
        return {'source_fingerprint': project.fingerprint, 'structure': project.structure_data,
                'quality': project.quality, 'ocr_performed': False}

    def _call(self, judge, kind, prompt, payload):
        # Measure the actual serialized model body, including instructions and
        # schema. Oversize work remains a technical failure, never a skipped chunk.
        stage = 'explicit' if kind == 'markdown-explicit-use' else 'nodes'
        model = self.profile['stage_model_options'][stage]
        options = WireOptions.parse({key: model[key] for key in
            ('reasoning_effort', 'max_output_tokens', 'response_transport')})
        payload = json.loads(json.dumps(payload, ensure_ascii=False, allow_nan=False, sort_keys=True))
        if len(wire_bytes(prompt, payload, options)) > self.profile['markdown_options']['max_wire_bytes']:
            raise SourceError('Markdown packet exceeds the bound wire size; retained for continuation')
        response = judge(kind, prompt, payload)
        _check_shape(response, payload['response_schema'])
        return response

    def _nodes(self, project, rules, diagnosis, identity):
        options = self.profile['markdown_options']
        blocks = source_blocks(project, options['block_chars'])
        recorder = RecordingJudge(self._judge('nodes'), root=self.root, identity=identity)
        prompt = (self.root / PROMPTS[0]).read_text()
        graph = Graph(snapshot_id='markdown:' + digest(identity)[:24], documents=(project.document,)).model_dump(mode='json')
        unresolved = copy.deepcopy(project.quality.get('unresolved', []))
        occurrences, tasks = {}, []
        for chunk in chunks(blocks, options['chunk_chars']):
            core = [row['id'] for row in chunk]
            start, stop = chunk[0]['source']['char_start'], chunk[-1]['source']['char_end']
            # Nearby complete blocks and ancestor headings are evidence only;
            # a unit belongs to the chunk containing its first original span.
            context = [b for b in blocks if (b['kind'] == 'heading' and b['source']['char_start'] < start)
                       or (start - 2000 <= b['source']['char_start'] < start)
                       or (stop <= b['source']['char_start'] < stop + 2000)]
            context = context[-12:]
            while sum(len(b['text']) for b in context) > options['context_chars']:
                context.pop(0)
            for phase in ('formal', 'prose'):
                available = {b['id']: b for b in (*context, *chunk)}
                payload = {'source_fingerprint': project.fingerprint, 'topology_revision': identity,
                    'cycle_diagnosis': diagnosis, 'phase': phase, 'core_blocks': core,
                    'blocks': list(available.values()), 'previous_nodes': [u for u in graph['units']
                        if start <= u['sources'][0]['char_start'] < stop], 'response_schema': NODE_SCHEMA}
                for followup in range(options['max_followups'] + 1):
                    payload['source_context_followup'] = followup
                    response = self._call(recorder, 'markdown-nodes-' + phase, prompt, payload)
                    if sorted(response['reviewed_blocks']) != sorted(core):
                        raise SourceError('Markdown response must cover every core block exactly once')
                    if not response['context_queries']:
                        break
                    if followup == options['max_followups']:
                        unresolved.append({'reason': 'context_followup_limit', 'blocks': core,
                                           'queries': response['context_queries']})
                        break
                    supplements = []
                    for query in response['context_queries'][:3]:
                        if len(query) > 240:
                            raise SourceError('Markdown context query exceeds 240 characters')
                        found = [b for b in blocks if query in b['text'] and b['id'] not in available][:2]
                        for block in found:
                            if sum(len(b['text']) for b in supplements) + len(block['text']) <= options['context_chars']:
                                supplements.append(block)
                                available[block['id']] = block
                    payload['blocks'] = list(available.values())
                    payload['requested_queries'] = response['context_queries']
                tasks.append({'phase': phase, 'blocks': core, 'response_sha256': digest(response)})
                if any(r['block_id'] not in available for r in response['unresolved']):
                    raise SourceError('Markdown unresolved item refers to an unknown source block')
                unresolved.extend(response['unresolved'])
                for proposal in response['nodes']:
                    try:
                        spans = selections(project, proposal['selections'], available)
                        if not start <= spans[0]['char_start'] < stop:
                            raise SourceError('Markdown node begins outside its owning core chunk')
                        key = digest(spans)
                        if key in occurrences:
                            if occurrences[key]['category'] != proposal['category']:
                                raise SourceError('Same Markdown occurrence has conflicting categories')
                            continue
                        context_spans = selections(project, proposal['context'], available, ordered=False) if proposal['context'] else spans
                        cid = 'context:' + digest([project.document.id, context_spans])[:24]
                        if not any(c['id'] == cid for c in graph['contexts']):
                            graph['contexts'].append(Context(id=cid, document=project.document.id,
                                sources=tuple(project.source(s) for s in context_spans),
                                description='\n'.join(project.resolve_span(s) for s in context_spans),
                                assumptions=None, symbols=None).model_dump(mode='json'))
                        unit = Unit(id='unit:' + digest([project.document.id, spans])[:24], category=proposal['category'],
                            label=proposal['label'], document=project.document.id, context=cid,
                            sources=tuple(project.source(s) for s in spans),
                            original_text=tuple(project.resolve_span(s) for s in spans)).model_dump(mode='json')
                        graph['units'].append(unit)
                        occurrences[key] = unit
                    except SourceError as error:
                        unresolved.append({'reason': str(error), 'proposal': proposal, 'blocks': core})
        result = {'graph': graph, 'source_fingerprint': project.fingerprint, 'unresolved': unresolved,
                  'summary': {'status': 'partial' if unresolved else 'complete'}, 'tasks': tasks}
        if self.strict_nodes and unresolved:
            raise SourceError('Complete Markdown re-extraction has unresolved source/node work')
        raw_path = recorder.directory / ('raw-node-result-' + digest(result) + '.json')
        _save_once(raw_path, result)
        execution = {'technical_complete': True, 'scope': 'partial' if unresolved else 'complete',
                     'call_bindings': recorder.records, 'raw_node_result': _record(self.root, raw_path),
                     'contract': 'markdown-two-pass-all-core-blocks.v1', 'tasks': tasks}
        execution_path = recorder.directory / ('execution-report-' + digest(execution) + '.json')
        _save_once(execution_path, execution)
        execution['execution_report'] = _record(self.root, execution_path)
        result['production_execution'] = execution
        return result

    def _explicit(self, project, nodes, diagnosis, identity):
        graph, ledger, pairs = nodes['graph'], [], {}
        units = {u['id']: u for u in graph['units']}
        by_label = {}
        for uid, unit in units.items():
            if unit['category'] != 'proof':
                for label in labels('\n'.join(unit['original_text'])):
                    by_label.setdefault(label, []).append(uid)
        for uid, unit in units.items():
            for source, text in zip(unit['sources'], unit['original_text']):
                for ref in references(text):
                    targets = [other for other in by_label.get(ref['label'], []) if other != uid]
                    span = {k: v for k, v in source.items() if k != 'document'}
                    span.update(char_start=source['char_start'] + ref['char_start'], char_end=source['char_start'] + ref['char_end'])
                    ledger.append({'span': span, 'activity': 'active', 'label': ref['label'],
                        'targets': [{'resolution': 'resolved' if len(targets) == 1 else 'ambiguous',
                                     'target_spans': [s for target in targets for s in units[target]['sources']]}]})
                    for other in targets:
                        kind = 'proof_dependency' if unit['category'] == 'proof' else 'statement_dependency'
                        pairs[(other, uid, kind)] = {'source': other, 'target': uid, 'type': kind}
        # Proof ownership is judged separately from prerequisite use. Candidate
        # retrieval includes both preceding and following statements.
        ordered = sorted(units.values(), key=lambda u: u['sources'][0]['char_start'])
        for i, unit in enumerate(ordered):
            if unit['category'] != 'proof':
                continue
            candidates = [u for u in ordered[max(0, i-3):i+4] if u['category'] == 'statement']
            for ref in references('\n'.join(unit['original_text'])):
                candidates.extend(units[uid] for uid in by_label.get(ref['label'], []) if units[uid]['category'] == 'statement')
            for target in candidates:
                key = (unit['id'], target['id'], 'proves')
                pairs[key] = dict(zip(('source', 'target', 'type'), key))
        candidates = [dict(pair, candidate_id='md-pair:' + digest(pair)[:24]) for pair in pairs.values()]
        prompt, judge = (self.root / PROMPTS[1]).read_text(), self._judge('explicit')
        decisions, calls = [], []
        for batch in chunks([dict(p, text=json.dumps([units[p['source']], units[p['target']]], ensure_ascii=False)) for p in candidates], 26000):
            for offset in range(0, len(batch), self.profile['markdown_options']['relation_batch']):
                packet = [{k: v for k, v in p.items() if k != 'text'} for p in batch[offset:offset+self.profile['markdown_options']['relation_batch']]]
                ids = {p[side] for p in packet for side in ('source', 'target')}
                payload = {'topology_revision': identity, 'cycle_diagnosis': diagnosis, 'candidates': packet,
                    'units': [units[uid] for uid in sorted(ids)],
                    'contexts': [c for c in graph['contexts'] if c['id'] in {units[uid]['context'] for uid in ids}],
                    'response_schema': RELATION_SCHEMA}
                response = self._call(judge, 'markdown-explicit-use', prompt, payload)
                if sorted(d['candidate_id'] for d in response['decisions']) != sorted(p['candidate_id'] for p in packet):
                    raise SourceError('Markdown relation decision population differs')
                decisions.extend(response['decisions'])
                calls.append({'payload': payload, 'response': response})
        accepted = {d['candidate_id']: d for d in decisions if d['decision'] == 'accept'}
        proposals = [dict(p, explanation=accepted[p['candidate_id']]['explanation']) for p in candidates if p['candidate_id'] in accepted]
        ownership = {}
        for p in proposals:
            if p['type'] == 'proves':
                ownership.setdefault(p['source'], []).append(p['target'])
        conflicts = {uid for uid, targets in ownership.items() if len(targets) > 1}
        proposals = [p for p in proposals if not (p['type'] == 'proves' and p['source'] in conflicts)]
        result = bind_relations(graph, proposals)
        return {'result': {'graph': result, 'reference_ledger': ledger}, 'calls': calls,
                'decisions': decisions, 'proof_ownership_conflicts': sorted(conflicts),
                'unattributed_proofs': [u['id'] for u in ordered if u['category'] == 'proof' and
                    not any(p['type'] == 'proves' and p['source'] == u['id'] for p in proposals)]}
