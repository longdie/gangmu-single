"""User source registration exercises the actual E1 and E4 source consumers."""
from pathlib import Path
import json
import shutil

import pytest

from gangmu.representation.sources import E1Corpus, SourceError
from gangmu.runtime.inputs import UserSource, register_source, register_sources
from gangmu.single_document.source import SourceProject


def source(tmp_path: Path, name='paper') -> Path:
    root = tmp_path / name
    root.mkdir()
    (root / 'chapters').mkdir()
    (root / 'main.tex').write_bytes(
        b'\\documentclass{article}\r\n\\begin{document}\r\n\\input{chapters/result}\r\n\\end{document}\r\n')
    (root / 'chapters' / 'result.tex').write_text(
        '\\begin{theorem}For α, $α=α$.\\label{one}\\end{theorem}\n', encoding='utf-8')
    (root / 'local.sty').write_text('\\newcommand{\\local}{yes}\n')
    (root / '.latexmetadata').write_bytes(b'keep hidden source assets\n')
    return root


def project(tmp_path: Path) -> Path:
    root = tmp_path / 'project'
    root.mkdir()
    return root


def test_registered_source_is_consumable_and_preserves_all_original_bytes(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    before = {path.relative_to(original): path.read_bytes() for path in original.rglob('*') if path.is_file()}
    result = register_source(target, UserSource(original, 'my-paper', 'v1', title='My paper'))
    corpus = E1Corpus(result.artifact_dir)
    assert corpus.verify()['papers'] == 1
    assert result.reused is False
    assert result.artifact_dir.parent == target / 'artifacts/preprocessed/user'
    for name, raw in before.items():
        assert corpus.read(f'my-paper/source/{name.as_posix()}') == raw
        assert (original / name).read_bytes() == raw
    pinned = SourceProject.from_artifact(result.artifact_dir)
    assert pinned.paper_id == 'my-paper'
    assert 'For α' in pinned.reading
    assert pinned.assert_unchanged()['unchanged'] is True
    provenance = json.loads(corpus.read('preprocessing.json'))
    assert provenance['compilation_performed'] is False
    assert provenance['semantic_extraction_performed'] is False
    assert not (result.artifact_dir / 'my-paper/build.json').exists()


def test_identical_relocated_inputs_reuse_exact_snapshot(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    first = register_source(target, UserSource(original, 'paper', 'v1'))
    before = {path.relative_to(first.artifact_dir): path.read_bytes()
              for path in first.artifact_dir.rglob('*') if path.is_file()}
    relocated = tmp_path / 'relocated'
    shutil.copytree(original, relocated)
    second = register_source(target, UserSource(relocated, 'paper', 'v1'))
    assert second.reused is True
    assert second.artifact_dir == first.artifact_dir
    assert before == {path.relative_to(first.artifact_dir): path.read_bytes()
                      for path in first.artifact_dir.rglob('*') if path.is_file()}


def test_changed_source_requires_explicit_new_version(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    first = register_source(target, UserSource(original, 'paper', 'v1'))
    old = E1Corpus(first.artifact_dir).read('paper/source/main.tex')
    (original / 'main.tex').write_bytes(old + b'\n% new source revision\n')
    with pytest.raises(SourceError, match='choose a new document version'):
        register_source(target, UserSource(original, 'paper', 'v1'))
    revised = register_source(target, UserSource(original, 'paper', 'v2'))
    assert revised.artifact_dir != first.artifact_dir
    assert E1Corpus(first.artifact_dir).read('paper/source/main.tex') == old
    assert SourceProject.from_artifact(revised.artifact_dir).arxiv_id == 'v2'


def test_corrupt_registered_copy_is_rejected_instead_of_rebuilt(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    first = register_source(target, UserSource(original, 'paper', 'v1'))
    (first.artifact_dir / 'paper/source/main.tex').write_text('corrupt')
    with pytest.raises(SourceError, match='digest mismatch'):
        register_source(target, UserSource(original, 'paper', 'v1'))
    assert (first.artifact_dir / 'paper/source/main.tex').read_text() == 'corrupt'


def test_batch_of_arbitrary_size_has_complete_and_order_independent_identity(tmp_path):
    a, b, target = source(tmp_path, 'a'), source(tmp_path, 'b'), project(tmp_path)
    first = register_sources(target, [UserSource(b, 'B', 'v1'), UserSource(a, 'A', 'v1')])
    assert first.paper_ids == ('A', 'B')
    assert E1Corpus(first.artifact_dir).verify()['papers'] == 2
    assert SourceProject.from_artifact(first.artifact_dir, 'B').paper_id == 'B'
    second = register_sources(target, [UserSource(a, 'A', 'v1'), UserSource(b, 'B', 'v1')])
    assert second.artifact_dir == first.artifact_dir
    with pytest.raises(SourceError, match='duplicate paper_id'):
        register_sources(target, [UserSource(a, 'A', 'v1'), UserSource(b, 'A', 'v2')])


def test_missing_index_cannot_hide_old_document_version_binding(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    first = register_source(target, UserSource(original, 'paper', 'v1'))
    (first.artifact_dir.parent / 'identity-index.json').unlink()
    (original / 'local.sty').write_text('changed')
    with pytest.raises(SourceError, match='choose a new document version'):
        register_source(target, UserSource(original, 'paper', 'v1'))


def test_source_and_artifact_symlinks_are_refused(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    (original / 'linked.tex').symlink_to(original / 'main.tex')
    with pytest.raises(SourceError, match='symlinks are not accepted'):
        register_source(target, UserSource(original, 'paper', 'v1'))
    (original / 'linked.tex').unlink()
    outside = tmp_path / 'outside'
    outside.mkdir()
    (target / 'artifacts').symlink_to(outside, target_is_directory=True)
    with pytest.raises(SourceError, match='output directory must not contain symlinks'):
        register_source(target, UserSource(original, 'paper', 'v1'))
    assert list(outside.iterdir()) == []


def test_source_overlap_and_escaping_main_are_rejected_without_source_mutation(tmp_path):
    original = source(tmp_path)
    before = sorted(path.relative_to(original) for path in original.rglob('*'))
    with pytest.raises(SourceError, match='must not overlap'):
        register_source(original, UserSource(original, 'paper', 'v1'))
    assert before == sorted(path.relative_to(original) for path in original.rglob('*'))
    target = project(tmp_path)
    with pytest.raises(SourceError, match='noncanonical source path'):
        register_source(target, UserSource(original, 'paper', 'v1', '../outside.tex'))


def test_unresolved_tex_is_reported_without_fake_acceptance(tmp_path):
    original, target = source(tmp_path), project(tmp_path)
    (original / 'main.tex').write_text('\\begin{document}\\input{missing}\\end{document}')
    result = register_source(target, UserSource(original, 'paper', 'v1'))
    corpus = result.corpus()
    quality = corpus.record('paper', 'preprocess/quality.json')
    assert quality['source_roundtrip_failures'] == []
    assert quality['unresolved_by_reason']
    assert quality['semantic_extraction_performed'] is False


def test_source_edit_during_preprocessing_never_publishes_mixed_snapshot(tmp_path, monkeypatch):
    from gangmu.runtime import inputs
    original, target = source(tmp_path), project(tmp_path)
    real = inputs.latex_structure.preprocess

    def concurrent_edit(*args, **kwargs):
        answer = real(*args, **kwargs)
        (original / 'local.sty').write_text('concurrently updated')
        return answer

    monkeypatch.setattr(inputs.latex_structure, 'preprocess', concurrent_edit)
    with pytest.raises(SourceError, match='source changed'):
        register_source(target, UserSource(original, 'paper', 'v1'))
    parent = target / 'artifacts/preprocessed/user'
    assert not list(parent.glob('*/artifact.json'))
    assert not list(parent.glob('.register-*'))
