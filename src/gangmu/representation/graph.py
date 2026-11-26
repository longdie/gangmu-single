"""Snapshot-local ID lookup and structural/source checks for MathIR v4.

These checks do not implement the reserved formal-semantics interface and do
not certify mathematical assertions. Old schemas use explicit v1/v2/v3 modules.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path

from pydantic import ValidationError

from .model import (Context, Dependency, Document, Evidence, Graph, Identity,
                    IdentityView, Proves, Record, Relation, Source, Unit, digest)
from .sources import E1Corpus, SourceError, _json


class GraphError(ValueError):
    pass


@dataclass
class Report:
    errors: list[dict] = field(default_factory=list)
    unresolved: list[dict] = field(default_factory=list)
    semantic_validation: str = 'not_performed'
    source_validation: str = 'not_performed'

    @property
    def structurally_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {'structurally_valid': self.structurally_valid, **asdict(self)}


def _claim(relation: Relation) -> dict:
    value = relation.model_dump(mode='json', exclude={'id', 'evidence'})
    if relation.type == 'identity':
        value['source'], value['target'] = sorted((relation.source, relation.target))
    return value


def claim_key(relation: Relation) -> str:
    """One assertion per direction, participant pair, conditions and policy."""
    value = relation.model_dump(mode='json', exclude={'id', 'evidence', 'explanation'})
    return digest(value)


class GraphIndex:
    def __init__(self, graph: Graph):
        self.graph = graph
        self.index: dict[str, Record] = {}
        for record in graph.records():
            if record.id in self.index:
                raise GraphError(f'duplicate snapshot record ID: {record.id}')
            self.index[record.id] = record

    def get(self, record_id: str, expected: type[Record] | None = None) -> Record:
        try:
            record = self.index[record_id]
        except (KeyError, TypeError) as exc:
            raise GraphError(f'missing record: {record_id}') from exc
        if expected is not None and not isinstance(record, expected):
            raise GraphError(f'wrong reference kind: {record_id}, expected {expected.__name__}')
        return record

    def claim_digest(self, relation: Relation, sources: tuple[Source, ...] = ()) -> str:
        """Bind a flat claim to its full participants and interpretation environment.

        Both directions of one symmetric fact deliberately have the same digest.
        IDs stay short, but copying evidence to changed content cannot authorize it.
        Proof-target associations and ancestor contexts are included transitively.
        Build all relations before binding evidence. Associations omit their own
        evidence references, avoiding circular hashes. Explicit
        evidence locations bind additional source documents and their versions.
        """
        self.get(relation.source, Unit)
        self.get(relation.target, Unit)
        pending = [relation.source, relation.target, *(s.document for s in sources)]
        closure: dict[str, dict] = {}
        proof_targets: dict[str, list[dict]] = {}
        while pending:
            record = self.get(pending.pop())
            if record.id in closure:
                continue
            if not isinstance(record, (Document, Context, Unit)):
                raise GraphError('a claim participant/environment must be a mathematical record')
            closure[record.id] = record.model_dump(mode='json')
            if isinstance(record, (Context, Unit)):
                pending.append(record.document)
                pending.extend(s.document for s in record.sources)
            if isinstance(record, Context) and record.parent is not None:
                pending.append(record.parent)
            if isinstance(record, Unit):
                pending.append(record.context)
                if record.category == 'proof':
                    associations = [r for r in self.graph.relations
                                    if r.type == 'proves' and r.source == record.id]
                    proof_targets[record.id] = sorted((_claim(r) for r in associations), key=digest)
                    pending.extend(r.target for r in associations)
        return digest({'schema_version': 'gangmu.mathir.claim.v4',
                       'claim': _claim(relation), 'participants': closure,
                       'proof_targets': proof_targets,
                       'evidence_sources': [s.model_dump(mode='json') for s in sources]})

    def relation_records(self) -> tuple[Relation, ...]:
        return self.graph.relations

    def relations_for(self, unit: str) -> tuple[Relation, ...]:
        self.get(unit, Unit)
        return tuple(r for r in self.graph.relations if unit in (r.source, r.target))

    def proofs_for(self, statement: str) -> tuple[Unit, ...]:
        target = self.get(statement, Unit)
        if target.category != 'statement':
            raise GraphError('proof target must be a statement')
        proof_ids = {r.source for r in self.graph.relations if r.type == 'proves' and r.target == statement}
        return tuple(u for u in self.graph.units if u.category == 'proof' and u.id in proof_ids)

    def dependency_projection(self) -> list[dict]:
        return [{'source': r.source, 'target': r.target, 'relation': r.id, 'type': r.type,
                 'individually_sufficient': False}
                for r in self.graph.relations if isinstance(r, Dependency)]


def claim_digest(graph: Graph, relation: Relation, sources: tuple[Source, ...] = ()) -> str:
    return GraphIndex(graph).claim_digest(relation, sources)


def validate(graph: Graph, corpus: E1Corpus | None = None) -> Report:
    report = Report()
    try:
        graph = Graph.model_validate_json(json.dumps(graph.model_dump(mode='json'), allow_nan=False))
    except (ValidationError, ValueError, TypeError) as exc:
        report.errors.append({'record': 'graph', 'reason': f'invalid record shape: {exc}'})
        return report
    try:
        index = GraphIndex(graph)
    except GraphError as exc:
        report.errors.append({'record': 'graph', 'reason': str(exc)})
        return report

    def notice(record, reason):
        report.unresolved.append({'record': record.id, 'reason': reason})

    resolved: dict[Source, str] = {}

    def source_check(source: Source, document: str | None = None):
        doc = index.get(source.document, Document)
        if document is not None and doc.id != document:
            raise SourceError('source belongs to a different document')
        if corpus is not None and source not in resolved:
            row = corpus._paper(doc.paper_id)
            if (doc.arxiv_id, doc.artifact_version, doc.artifact_tree_sha256) != (
                    row['arxiv_id'], corpus.version, corpus.tree_sha256):
                raise SourceError('document does not match pinned corpus')
            anchor = corpus.anchor(doc.paper_id, {
                'file': source.file, 'char_start': source.char_start, 'char_end': source.char_end,
            }, occurrence=source.occurrence)
            resolved[source] = corpus.resolve(anchor)

    source_errors = 0
    for record in graph.records():
        try:
            if isinstance(record, Document) and corpus is not None:
                row = corpus._paper(record.paper_id)
                if (record.arxiv_id, record.artifact_version, record.artifact_tree_sha256) != (
                        row['arxiv_id'], corpus.version, corpus.tree_sha256):
                    raise SourceError('document does not match pinned corpus')
            for source in getattr(record, 'sources', ()):
                source_check(source, record.document if isinstance(record, (Context, Unit)) else None)
            if isinstance(record, Unit):
                for source, original in zip(record.sources, record.original_text):
                    if len(original) != source.char_end - source.char_start:
                        raise SourceError('original_text length differs from source character range')
                    if corpus is not None and resolved[source] != original:
                        raise SourceError('original_text differs from pinned source bytes')
        except (GraphError, SourceError, ValueError, UnicodeError, OSError) as exc:
            source_errors += 1
            report.errors.append({'record': record.id, 'reason': str(exc), 'category': 'source'})
    if corpus is not None:
        report.source_validation = 'verified' if not source_errors else 'failed_or_incomplete'

    for record in graph.records():
        try:
            if isinstance(record, Context):
                index.get(record.document, Document)
                if record.assumptions is None:
                    notice(record, 'context assumptions are unknown or unextracted')
                if record.symbols is None:
                    notice(record, 'context symbols are unknown or unextracted')
                seen = {record.id}
                parent = record.parent
                while parent is not None:
                    if parent in seen:
                        raise GraphError('cyclic context containment')
                    seen.add(parent)
                    upper = index.get(parent, Context)
                    if upper.document != record.document:
                        raise GraphError('context containment crosses documents')
                    parent = upper.parent
            elif isinstance(record, Unit):
                index.get(record.document, Document)
                context = index.get(record.context, Context)
                if context.document != record.document:
                    raise GraphError('unit context belongs to another document')
                if not record.semantics:
                    notice(record, 'semantics is an empty extension point, not a formal interpretation or empty assumptions')
                else:
                    notice(record, 'semantics has JSON content but no formal language/checker is defined')
                if record.category == 'proof':
                    targets = [index.get(r.target, Unit) for r in graph.relations
                               if r.type == 'proves' and r.source == record.id]
                    if not any(t.category == 'statement' and t.document == record.document for t in targets):
                        raise GraphError('a proof requires a same-document proves edge to a statement')
                    notice(record, 'proof-target association is not checker-certified proof validity')
            elif isinstance(record, Relation):
                source = index.get(record.source, Unit)
                target = index.get(record.target, Unit)
                if isinstance(record, Dependency):
                    if (record.type == 'proof_dependency') != (target.category == 'proof'):
                        raise GraphError('dependency type does not match statement/proof target')
                    if record.type == 'statement_dependency' and source.category == 'proof':
                        raise GraphError('a proof cannot be a statement interpretation prerequisite')
                elif isinstance(record, Identity):
                    if source.category != target.category or source.category not in ('definition', 'statement', 'expression'):
                        raise GraphError('identity requires same-category definitions, statements or expressions; proof identity is undefined')
                elif isinstance(record, Proves):
                    if source.category != 'proof' or target.category != 'statement':
                        raise GraphError('proves requires a proof source and a statement target')
                evs = [index.get(e, Evidence) for e in record.evidence]
                if any(e.claim_sha256 != index.claim_digest(record, e.sources) for e in evs):
                    raise GraphError('evidence binding differs from claim or participant/environment content')
                if isinstance(record, Identity) and all(e.method == 'source_observation' for e in evs):
                    raise GraphError('source observation alone cannot establish a mathematical alignment')
                if isinstance(record, Proves) and source.document != target.document:
                    if record.conditions is None or not evs:
                        raise GraphError('cross-document proves requires explicit conditions and evidence')
                    if all(e.method == 'source_observation' for e in evs):
                        raise GraphError('source observation alone cannot establish cross-document proof reuse')
                notice(record, 'accepted record is not checker-certified mathematics')
            elif isinstance(record, Evidence) and record.method == 'formal_certificate':
                notice(record, 'certificate backend not implemented')
            elif isinstance(record, IdentityView):
                for member in record.members:
                    index.get(member, Unit)
                covered = {record.representative}
                policy_scope = None
                for ref in record.justifications:
                    relation = index.get(ref, Identity)
                    pair = {relation.source, relation.target}
                    if record.representative not in pair or not pair <= set(record.members):
                        raise GraphError('identity view requires direct representative-member justifications')
                    value = (relation.policy, relation.conditions)
                    if policy_scope is not None and value != policy_scope:
                        raise GraphError('identity view combines different policies or common scopes')
                    policy_scope = value
                    covered.update(pair)
                if covered != set(record.members):
                    raise GraphError('identity view has an unjustified member')
        except (GraphError, SourceError, ValueError, TypeError) as exc:
            report.errors.append({'record': record.id, 'reason': str(exc)})

    keys: dict[str, str] = {}
    for relation in graph.relations:
        key = claim_key(relation)
        if key in keys:
            report.errors.append({'record': relation.id, 'reason': 'duplicate directed claim: ' + keys[key]})
        keys[key] = relation.id
    for relation in graph.relations:
        if not isinstance(relation, Identity):
            continue
        reverse = relation.model_copy(update={'source': relation.target, 'target': relation.source})
        reverse_id = keys.get(claim_key(reverse))
        if reverse_id is None:
            report.errors.append({'record': relation.id, 'reason': 'symmetric relation lacks reverse direction'})
            continue
        counterpart = index.get(reverse_id, Relation)
        fields = {'id', 'source', 'target'}
        if relation.model_dump(exclude=fields) != counterpart.model_dump(exclude=fields):
            report.errors.append({'record': relation.id, 'reason': 'symmetric directions must share explanation, conditions, evidence and policy'})
    return report


def loads(content: bytes | str) -> Graph:
    try:
        data = _json(content.encode() if isinstance(content, str) else content)
        graph = Graph.model_validate_json(json.dumps(data, allow_nan=False))
        report = validate(graph)
        if report.errors:
            raise GraphError(json.dumps(report.errors, ensure_ascii=False))
        return graph
    except (ValueError, TypeError) as exc:
        if isinstance(exc, GraphError):
            raise
        raise GraphError(str(exc)) from exc


def load(path: str | Path) -> Graph:
    if Path(path).is_dir():
        from .storage import load_bundle
        return load_bundle(path)
    return loads(Path(path).read_bytes())


def save(graph: Graph, path: str | Path) -> None:
    """Export one JSON file for interchange/debugging, replacing that file.

    This helper is not immutable snapshot publication. Use storage.save_bundle
    for the six-file, manifest-bound format and no-overwrite guarantees.
    """
    report = validate(graph)
    if report.errors:
        raise GraphError(json.dumps(report.errors, ensure_ascii=False))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(graph.model_dump(mode='json'), ensure_ascii=False, indent=2,
                                 allow_nan=False) + '\n', encoding='utf-8')
