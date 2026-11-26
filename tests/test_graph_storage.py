"""Meaningful bundle integrity and immutable publication boundaries."""
import hashlib
import json
from pathlib import Path

import pytest

from gangmu.representation.graph import GraphError
from gangmu.representation.model import Context, Document, Graph, Unit, Source, digest
from gangmu.representation.storage import COLLECTIONS, load_bundle, save_bundle
import gangmu.representation.storage as storage


@pytest.fixture
def graph():
    return Graph(snapshot_id='测试：有限维代数', documents=(Document(
        id='document:甲', paper_id='论文甲', arxiv_id='2401.00001',
        artifact_version='原文版本一', artifact_tree_sha256='a' * 64,
    ),))


@pytest.fixture
def bundle(tmp_path, graph):
    path = tmp_path / 'snapshot-v4'
    save_bundle(graph, path)
    return path


def manifest(bundle):
    return json.loads((bundle / 'manifest.json').read_text(encoding='utf-8'))


def write_manifest(bundle, value):
    (bundle / 'manifest.json').write_text(json.dumps(value, ensure_ascii=False), encoding='utf-8')


def update_collection(bundle, collection, content, *, count=None, rehash_graph=False):
    """Update storage metadata too, exercising validation beyond byte hashes."""
    filename = f'{collection}.json'
    (bundle / filename).write_bytes(content)
    value = manifest(bundle)
    value['files'][collection]['sha256'] = hashlib.sha256(content).hexdigest()
    if count is not None:
        value['files'][collection]['count'] = count
    if rehash_graph:
        data = {'schema_version': value['schema_version'], 'snapshot_id': value['snapshot_id']}
        data.update({c: json.loads((bundle / f'{c}.json').read_text()) for c in COLLECTIONS})
        value['graph_sha256'] = digest(data)
    write_manifest(bundle, value)


def test_six_arrays_and_unicode_roundtrip(bundle, graph):
    assert load_bundle(bundle) == graph
    assert {p.name for p in bundle.iterdir()} == {'manifest.json', *(f'{c}.json' for c in COLLECTIONS)}
    value = manifest(bundle)
    assert value['schema_version'] == 'gangmu.mathir.v4'
    assert value['graph_sha256'] == digest(graph.model_dump(mode='json'))
    assert '论文甲' in (bundle / 'documents.json').read_text(encoding='utf-8')
    for collection in COLLECTIONS:
        content = (bundle / f'{collection}.json').read_bytes()
        assert isinstance(json.loads(content), list)
        assert value['files'][collection]['sha256'] == hashlib.sha256(content).hexdigest()
        assert value['files'][collection]['count'] == len(getattr(graph, collection))


def test_empty_graph(tmp_path):
    graph = Graph(snapshot_id='empty')
    save_bundle(graph, tmp_path / 'empty')
    assert load_bundle(tmp_path / 'empty') == graph


def test_record_storage_uses_snapshot_ids_without_legacy_metadata(bundle):
    records = json.loads((bundle / 'documents.json').read_text(encoding='utf-8'))
    assert records[0]['id'] == 'document:甲'
    assert 'kind' not in records[0]
    assert 'revision' not in records[0]
    assert manifest(bundle)['snapshot_id'] == '测试：有限维代数'


def test_existing_bundle_is_unchanged(bundle, graph):
    before = {p.name: p.read_bytes() for p in bundle.iterdir()}
    with pytest.raises(GraphError, match='already exists'):
        save_bundle(graph.model_copy(update={'snapshot_id': 'new'}), bundle)
    assert before == {p.name: p.read_bytes() for p in bundle.iterdir()}


def test_existing_empty_directory_is_not_replaced(tmp_path, graph):
    target = tmp_path / 'existing'
    target.mkdir()
    with pytest.raises(GraphError, match='already exists'):
        save_bundle(graph, target)
    assert target.is_dir() and list(target.iterdir()) == []


def test_concurrent_destination_creation_is_not_overwritten(tmp_path, graph, monkeypatch):
    original = storage._publish

    def race(staging, target):
        target.mkdir()
        original(staging, target)

    monkeypatch.setattr(storage, '_publish', race)
    target = tmp_path / 'race'
    with pytest.raises(GraphError, match='already exists'):
        save_bundle(graph, target)
    assert list(target.iterdir()) == []
    assert list(tmp_path.iterdir()) == [target]


def test_failed_publication_removes_staging_only(tmp_path, graph, monkeypatch):
    existing = tmp_path / 'keep.txt'
    existing.write_text('untouched')

    def failure(staging, target):
        raise OSError('simulated publication failure')

    monkeypatch.setattr(storage, '_publish', failure)
    with pytest.raises(GraphError, match='simulated publication failure'):
        save_bundle(graph, tmp_path / 'new')
    assert list(tmp_path.iterdir()) == [existing]
    assert existing.read_text() == 'untouched'


def test_invalid_in_memory_graph_never_publishes(tmp_path, graph):
    forged = graph.model_copy(update={'documents': graph.documents * 2})
    with pytest.raises(GraphError, match='duplicate'):
        save_bundle(forged, tmp_path / 'invalid')
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize('value', [float('nan'), float('inf'), -float('inf')])
def test_bypassed_nonfinite_semantics_are_not_silently_serialized_as_null(tmp_path, graph, value):
    document = graph.documents[0].id
    source = Source(document=document, file='main.tex', char_start=0, char_end=1)
    context = Context(id='context', document=document, sources=(source,), description='test context')
    unit = Unit(id='unit', category='definition', label='X', document=document,
                         context=context.id, sources=(source,), original_text=('X',))
    invalid = unit.model_copy(update={'semantics': {'invalid': value}})
    forged = graph.model_copy(update={'contexts': (context,), 'units': (invalid,)})
    with pytest.raises(GraphError, match='Out of range float values'):
        save_bundle(forged, tmp_path / 'invalid')
    assert list(tmp_path.iterdir()) == []


def test_missing_file(bundle):
    (bundle / 'units.json').unlink()
    with pytest.raises(GraphError, match='file set mismatch.*units.json'):
        load_bundle(bundle)


def test_unexpected_file(bundle):
    (bundle / 'unlisted.json').write_text('[]')
    with pytest.raises(GraphError, match='file set mismatch.*unlisted.json'):
        load_bundle(bundle)


def test_tampered_file(bundle):
    (bundle / 'documents.json').write_text('[]')
    with pytest.raises(GraphError, match='byte digest mismatch: documents.json'):
        load_bundle(bundle)


def test_mixed_snapshot_files(bundle, tmp_path, graph):
    other = tmp_path / 'other'
    changed = graph.documents[0].model_copy(update={'paper_id': 'different source'})
    save_bundle(graph.model_copy(update={'documents': (changed,)}), other)
    (bundle / 'documents.json').write_bytes((other / 'documents.json').read_bytes())
    with pytest.raises(GraphError, match='byte digest mismatch'):
        load_bundle(bundle)


@pytest.mark.parametrize('changes, match', [
    ({'schema_version': 'gangmu.mathir.v999'}, 'unsupported bundle schema_version'),
    ({'schema_version': 'gangmu.mathir.v1'}, 'unsupported bundle schema_version'),
    ({'schema_version': 'gangmu.mathir.v2'}, 'unsupported bundle schema_version'),
    ({'schema_version': 'gangmu.mathir.v3'}, 'unsupported bundle schema_version'),
    ({'storage_version': 'anything'}, 'unsupported bundle storage_version'),
    ({'snapshot_id': ''}, 'nonempty snapshot_id'),
    ({'graph_sha256': 'bad'}, 'invalid graph_sha256'),
    ({'unexpected': 1}, 'exactly the documented manifest fields'),
])
def test_manifest_metadata(bundle, changes, match):
    value = manifest(bundle)
    value.update(changes)
    write_manifest(bundle, value)
    with pytest.raises(GraphError, match=match):
        load_bundle(bundle)


@pytest.mark.parametrize('filename', ['../documents.json', '/documents.json', 'other.json'])
def test_manifest_cannot_redirect_files(bundle, filename):
    value = manifest(bundle)
    value['files']['documents']['file'] = filename
    write_manifest(bundle, value)
    with pytest.raises(GraphError, match='must use fixed file'):
        load_bundle(bundle)


@pytest.mark.parametrize('field, value', [('count', True), ('count', -1), ('count', 1.0), ('sha256', 'x')])
def test_manifest_entry_validation(bundle, field, value):
    metadata = manifest(bundle)
    metadata['files']['documents'][field] = value
    write_manifest(bundle, metadata)
    with pytest.raises(GraphError, match='invalid'):
        load_bundle(bundle)


def test_manifest_rejects_missing_collection(bundle):
    value = manifest(bundle)
    del value['files']['units']
    write_manifest(bundle, value)
    with pytest.raises(GraphError, match='exactly the six fixed collections'):
        load_bundle(bundle)


def test_snapshot_id_is_bound_to_graph_hash(bundle):
    value = manifest(bundle)
    value['snapshot_id'] = 'falsely relabelled'
    write_manifest(bundle, value)
    with pytest.raises(GraphError, match='canonical graph digest mismatch'):
        load_bundle(bundle)


@pytest.mark.parametrize('content, match', [
    (b'{"x": 1, "x": 2}', 'duplicate JSON key'),
    (b'[NaN]', 'nonfinite JSON number'),
    (b'[Infinity]', 'nonfinite JSON number'),
    (b'[-Infinity]', 'nonfinite JSON number'),
    (b'[1e9999]', 'nonfinite JSON number'),
    (b'[', 'invalid UTF-8 JSON'),
    (b'["\xff"]', 'invalid UTF-8 JSON'),
    (b'{}', 'JSON array of records'),
    (b'[null]', 'JSON array of records'),
])
def test_rehashed_malformed_collection_is_rejected(bundle, content, match):
    update_collection(bundle, 'units', content)
    with pytest.raises(GraphError, match=match):
        load_bundle(bundle)


def test_duplicate_manifest_keys(bundle):
    (bundle / 'manifest.json').write_bytes(b'{"files": {}, "files": {}}')
    with pytest.raises(GraphError, match='duplicate JSON key'):
        load_bundle(bundle)


def test_record_count_checked_before_graph_validation(bundle):
    update_collection(bundle, 'documents', b'[]')
    with pytest.raises(GraphError, match='record count mismatch: documents.json'):
        load_bundle(bundle)


def test_duplicate_record_ids_rejected_after_integrity_checks(bundle):
    docs = json.loads((bundle / 'documents.json').read_text()) * 2
    update_collection(bundle, 'documents', json.dumps(docs).encode(), count=2, rehash_graph=True)
    with pytest.raises(GraphError, match='duplicate'):
        load_bundle(bundle)


def test_unknown_record_fields_rejected_after_integrity_checks(bundle):
    docs = json.loads((bundle / 'documents.json').read_text())
    docs[0]['unexpected'] = 'not in schema'
    update_collection(bundle, 'documents', json.dumps(docs).encode(), rehash_graph=True)
    with pytest.raises(GraphError, match='Extra inputs are not permitted'):
        load_bundle(bundle)


def test_missing_directory_has_clear_error(tmp_path):
    with pytest.raises(GraphError, match='cannot load bundle'):
        load_bundle(tmp_path / 'missing')


def test_collection_symlink_rejected(bundle, tmp_path):
    source = tmp_path / 'source.json'
    source.write_bytes((bundle / 'documents.json').read_bytes())
    (bundle / 'documents.json').unlink()
    (bundle / 'documents.json').symlink_to(source)
    with pytest.raises(GraphError, match='symlink in bundle'):
        load_bundle(bundle)
    assert source.read_bytes()


def test_manifest_symlink_rejected(bundle, tmp_path):
    source = tmp_path / 'manifest-source.json'
    source.write_bytes((bundle / 'manifest.json').read_bytes())
    (bundle / 'manifest.json').unlink()
    (bundle / 'manifest.json').symlink_to(source)
    with pytest.raises(GraphError, match='symlink in bundle'):
        load_bundle(bundle)


def test_bundle_directory_symlink_rejected(bundle, tmp_path, graph):
    alias = tmp_path / 'alias'
    alias.symlink_to(bundle, target_is_directory=True)
    with pytest.raises(GraphError, match='symlink in bundle path'):
        load_bundle(alias)
    with pytest.raises(GraphError, match='symlink in bundle path'):
        save_bundle(graph, alias / 'new')
    assert not (bundle / 'new').exists()


def test_dangling_target_symlink_rejected(tmp_path, graph):
    alias = tmp_path / 'alias'
    alias.symlink_to(tmp_path / 'missing')
    with pytest.raises(GraphError, match='symlink in bundle path'):
        save_bundle(graph, alias)
    assert alias.is_symlink()


def test_non_regular_collection_rejected(bundle):
    (bundle / 'units.json').unlink()
    (bundle / 'units.json').mkdir()
    with pytest.raises(GraphError, match='not a regular file'):
        load_bundle(bundle)


def test_parent_traversal_rejected(tmp_path, graph):
    target = tmp_path / 'child' / '..' / 'new'
    with pytest.raises(GraphError, match='parent traversal'):
        save_bundle(graph, target)
    with pytest.raises(GraphError, match='parent traversal'):
        load_bundle(target)
    assert list(tmp_path.iterdir()) == []
