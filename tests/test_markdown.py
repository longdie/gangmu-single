"""Source fidelity, production adapters, and downstream Markdown consumption."""
import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from gangmu import markdown_structure as md
from gangmu.representation.graph import validate
from gangmu.representation.model import Graph, digest
from gangmu.representation.sources import SourceError
from gangmu.representation.storage import save_bundle
from gangmu.runtime.inputs import UserSource, register_source
from gangmu.single_document.source import SourceProject
from gangmu.single_document.markdown_pipeline import MarkdownPipeline, selections, source_blocks

ROOT = Path(__file__).resolve().parents[1]
TEXT = ('# Mathematics\r\n\r\nThroughout this chapter let $K$ be a field. 50% is a proportion, not a comment.\r\n\r\n'
        'Definition 1. A zero vector is denoted by $0$.\r\n\r\n'
        'Theorem 2. For any vector $v$, $v+0=v$, using Definition 1.\r\n\r\n'
        'Proof of Theorem 2. Use Definition 1. Thus $v+0=v$. □\r\n\r\n'
        '```python\r\n# Theorem 99. fake\r\nprint("100%")\r\n```\r\n\r\n'
        '$$\r\n# this is math, not a heading\r\nx=x\r\n$$\r\n\r\n'
        '![diagram](images/figure.png)\r\n')


def registered(tmp_path, text=TEXT):
    root = tmp_path / 'project'
    source = root / 'data/example/markdown/paper/v1'
    source.mkdir(parents=True)
    (source / 'images').mkdir()
    (source / 'images/figure.png').write_bytes(b'fake image asset\x00\xff')
    (source / 'main.md').write_bytes(text.encode())
    result = register_source(root, UserSource(source, 'paper', 'v1', main_file='main.md', source_format='markdown'), namespace='example')
    return root, source, result, SourceProject.from_artifact(result.artifact_dir)


def test_markdown_snapshot_preserves_offsets_assets_and_source_identity(tmp_path):
    root, source, registered_source, project = registered(tmp_path)
    corpus = registered_source.corpus()
    assert corpus.verify()['papers'] == 1
    assert project.reading == TEXT and project.source_format == 'markdown'
    assert project.fingerprint['source_format'] == 'markdown'
    assert corpus.read('paper/source/images/figure.png') == (source / 'images/figure.png').read_bytes()
    blocks = project.structure_data['blocks']
    assert ''.join(project.resolve_span(b['source']) for b in blocks) == TEXT
    assert len(project.structure_data['sections']) == 1
    assert any(b['kind'] == 'math' for b in blocks)
    assert '# Theorem 99' not in md.active_text(TEXT)
    assert '50% is a proportion, not a comment.' in md.active_text(TEXT)
    assert md.references('50% then use Theorem 2.')[0]['label'] == 'theorem:2'
    assert register_source(root, UserSource(source, 'paper', 'v1', main_file='main.md', source_format='markdown'), namespace='example').reused
    (source / 'main.md').write_bytes(b'changed')
    with pytest.raises(SourceError, match='different source bytes'):
        register_source(root, UserSource(source, 'paper', 'v1', main_file='main.md', source_format='markdown'), namespace='example')
    assert project.assert_unchanged()['unchanged']


def test_markdown_and_latex_versions_coexist_without_aliasing(tmp_path):
    root, _, result, _ = registered(tmp_path)
    tex = root / 'data/example/latex/paper/v1'
    tex.mkdir(parents=True)
    (tex / 'main.tex').write_text(r'\begin{document}Text.\end{document}')
    second = register_source(root, UserSource(tex, 'paper', 'v1'), namespace='example')
    assert second.artifact_dir != result.artifact_dir
    (result.artifact_dir.parent / 'identity-index.json').unlink()
    assert register_source(root, UserSource(tex, 'paper', 'v1'), namespace='example').reused


def test_unique_exact_quotes_and_duplicate_occurrences(tmp_path):
    _, _, _, project = registered(tmp_path, 'Repeated word word.\n\nRepeated word word.\n')
    blocks = source_blocks(project, 6000)
    available = {b['id']: b for b in blocks}
    with pytest.raises(SourceError, match='exactly once'):
        selections(project, [{'block_id': blocks[0]['id'], 'quote': 'word'}], available)
    spans = [selections(project, [{'block_id': b['id'], 'quote': None}], available) for b in blocks]
    assert spans[0] != spans[1]
    assert project.resolve_span(spans[0][0]) == project.resolve_span(spans[1][0])


def test_context_quotes_are_a_set_but_node_quotes_preserve_reading_order(tmp_path):
    _, _, _, project = registered(tmp_path, 'Assumption one.\n\nAssumption two.\n')
    blocks = source_blocks(project, 6000)
    available = {b['id']: b for b in blocks}
    rows = [{'block_id': b['id'], 'quote': None} for b in reversed(blocks)]
    with pytest.raises(SourceError, match='reading order'):
        selections(project, rows, available)
    context = selections(project, rows, available, ordered=False)
    assert len(context) == 1
    assert project.resolve_span(context[0]) == project.reading
    assert md.labels('**Assertion** 11.4. An assertion.') == ['assertion:11.4']
    assert md.references('By Assertion 11.4 and (2.3).')[0]['label'] == 'assertion:11.4'


class Judge:
    """Synthetic mathematical decisions; transport/accounting are tested separately."""
    def __init__(self, root, *, bad_quote=False, missing_block=False, unresolved=False):
        self.config = {'project': 'example'}
        self.ledger = SimpleNamespace(path=root / 'fake-ledger', scope={})
        self.client = SimpleNamespace(model={'model': 'gpt-6.1-sol', 'provider': 'yxai', 'effort': 'high',
            'max_output_tokens': 12000, 'response_transport': 'sse'})
        self.bad_quote, self.missing_block, self.unresolved = bad_quote, missing_block, unresolved
        self.calls = []

    def fork(self, *args, **kwargs):
        return self

    def receipts(self):
        return []

    def __call__(self, kind, prompt, payload):
        self.calls.append(kind)
        if kind.startswith('markdown-nodes'):
            nodes = []
            if payload['phase'] == 'formal':
                for block in payload['blocks']:
                    category = next((category for prefix, category in [('Definition ', 'definition'),
                        ('Theorem ', 'statement'), ('Proof ', 'proof')] if block['text'].startswith(prefix)), None)
                    if category and block['id'] in payload['core_blocks']:
                        nodes.append({'category': category, 'label': block['text'].split('.')[0],
                            'selections': [{'block_id': block['id'], 'quote': 'fabricated' if self.bad_quote else None}], 'context': []})
            return {'reviewed_blocks': [] if self.missing_block else payload['core_blocks'], 'nodes': nodes,
                    'unresolved': [{'block_id': payload['core_blocks'][0], 'reason': 'missing image meaning'}] if self.unresolved else [],
                    'context_queries': []}
        if kind == 'markdown-explicit-use':
            return {'decisions': [{'candidate_id': p['candidate_id'],
                'decision': 'reject' if p['type'] == 'proof_dependency' and p['source'] == next(u['id'] for u in payload['units'] if u['category'] == 'statement') else 'accept',
                'explanation': 'Synthetic fixture source attribution or direct use.'} for p in payload['candidates']]}
        raise AssertionError(kind)


def pipeline(root, project, **options):
    for name in ('nodes.md', 'relations.md'):
        target = root / 'prompts/single_document/markdown' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / 'prompts/single_document/markdown' / name, target)
    from gangmu.single_document.markdown_pipeline import OPTIONS
    from gangmu.single_document.topology_integration import VERSION
    profile = {'schema_version': VERSION + '.fixed-profile', 'markdown_options': OPTIONS,
        'stage_model_options': {key: {'reasoning_effort': 'high', 'max_output_tokens': 12000,
            'response_transport': 'sse', 'timeout_seconds': 600} for key in ('nodes', 'explicit')}}
    profile['profile_sha256'] = digest(profile)
    return MarkdownPipeline(profile, Judge(root, **options), root=root)


def test_markdown_nodes_proofs_and_downstream_source_reading(tmp_path):
    root, _, registered_source, project = registered(tmp_path)
    pipe = pipeline(root, project)
    nodes = pipe._nodes(project, {}, {}, {'revision_id': 'test'})
    assert nodes['summary']['status'] == 'complete'
    assert len(nodes['graph']['units']) == 3
    explicit = pipe._explicit(project, nodes, {}, {'revision_id': 'test'})
    graph = Graph.model_validate_json(json.dumps(explicit['result']['graph']))
    assert not validate(graph, registered_source.corpus()).errors
    assert {r.type for r in graph.relations} == {'proves', 'statement_dependency', 'proof_dependency'}
    bundle = root / 'results/example/single_doc/paper/test/bundle'
    save_bundle(graph, bundle)
    statement = next(u for u in graph.units if u.category == 'statement')
    assert project.source_format == 'markdown'
    assert project.resolve_span(statement.sources[0].model_dump()).strip()
    assert '50% is a proportion' in project.reading
    assert 'paper/source/images/figure.png' in registered_source.corpus().files


@pytest.mark.parametrize('option', ['bad_quote', 'unresolved'])
def test_semantically_unresolved_markdown_is_partial(tmp_path, option):
    root, _, _, project = registered(tmp_path)
    nodes = pipeline(root, project, **{option: True})._nodes(project, {}, {}, {'revision_id': 'test'})
    assert nodes['production_execution']['technical_complete'] is True
    assert nodes['production_execution']['scope'] == 'partial'
    assert nodes['unresolved']


def test_missing_model_coverage_is_technical_failure(tmp_path):
    root, _, _, project = registered(tmp_path)
    with pytest.raises(SourceError, match='every core block'):
        pipeline(root, project, missing_block=True)._nodes(project, {}, {}, {'revision_id': 'test'})
