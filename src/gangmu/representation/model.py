"""MathIR v4: accepted records in one immutable, manifest-bound snapshot.

IDs resolve within that snapshot. ``semantics`` is only a JSON-object extension
point; this module deliberately defines no formal mathematical language.
"""
from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from .sources import SourceAnchor, _relative

Text = Annotated[str, Field(min_length=1)]
Digest = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]


def digest(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':'), allow_nan=False).encode()).hexdigest()


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)


class Record(Model):
    id: Text


class Document(Record):
    paper_id: Text
    arxiv_id: Text
    artifact_version: Text
    artifact_tree_sha256: Digest


class Source(Model):
    """A half-open character range in a document's pinned source snapshot."""
    document: Text
    file: Text
    char_start: Annotated[int, Field(ge=0)]
    char_end: Annotated[int, Field(ge=0)]
    occurrence: Text | None = None

    @model_validator(mode='after')
    def bounds(self) -> Self:
        _relative(self.file)
        if self.char_end < self.char_start:
            raise ValueError('reversed source character range')
        return self

    @classmethod
    def from_anchor(cls, anchor: SourceAnchor, document: str) -> Self:
        return cls(document=document, file=anchor.file, char_start=anchor.char_start,
                   char_end=anchor.char_end, occurrence=anchor.occurrence)


class Context(Record):
    document: Text
    parent: Text | None = None
    sources: Annotated[tuple[Source, ...], Field(min_length=1)]
    description: Text
    # None means unknown/unextracted; () explicitly asserts no listed entries.
    assumptions: tuple[Text, ...] | None = None
    symbols: tuple[Text, ...] | None = None


class Unit(Record):
    category: Literal['definition', 'statement', 'expression', 'proof']
    label: Text
    document: Text
    context: Text
    field: Text | None = None
    subfield: Text | None = None
    sources: Annotated[tuple[Source, ...], Field(min_length=1)]
    original_text: Annotated[tuple[Text, ...], Field(min_length=1)]
    semantics: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode='after')
    def content(self) -> Self:
        if len(self.sources) != len(self.original_text):
            raise ValueError('original_text must match sources one-to-one')
        # JsonValue enforces JSON types; explicitly exclude NaN and infinities.
        digest(self.semantics)
        return self


class Relation(Record):
    type: Literal['statement_dependency', 'proof_dependency', 'proves', 'identity']
    source: Text
    target: Text
    explanation: Text
    conditions: Text | None = None
    evidence: tuple[Text, ...] = ()

    @model_validator(mode='after')
    def endpoints_and_evidence(self) -> Self:
        if self.source == self.target:
            raise ValueError('a relation must have distinct participants')
        if len(set(self.evidence)) != len(self.evidence):
            raise ValueError('duplicate evidence references')
        return self


class Dependency(Relation):
    type: Literal['statement_dependency', 'proof_dependency']


class Proves(Relation):
    """A source proof's target association, not a proof-validity certificate."""
    type: Literal['proves'] = 'proves'


class Identity(Relation):
    type: Literal['identity'] = 'identity'
    conditions: Text
    evidence: Annotated[tuple[Text, ...], Field(min_length=1)]
    policy: Text


MathematicalRelation = Annotated[Dependency | Proves | Identity, Field(discriminator='type')]


class Evidence(Record):
    # Only evidence carries this content binding; endpoint references stay IDs.
    claim_sha256: Digest
    method: Literal['source_observation', 'agent_analysis', 'human_review', 'formal_certificate']
    sources: tuple[Source, ...] = ()
    explanation: Text
    certificate: Text | None = None

    @model_validator(mode='after')
    def form(self) -> Self:
        if self.method == 'source_observation' and not self.sources:
            raise ValueError('source observation needs original source locations')
        if (self.method == 'formal_certificate') != (self.certificate is not None):
            raise ValueError('formal certificate requires an explicit certificate locator')
        return self


class IdentityView(Record):
    representative: Text
    members: Annotated[tuple[Text, ...], Field(min_length=2)]
    justifications: Annotated[tuple[Text, ...], Field(min_length=1)]

    @model_validator(mode='after')
    def members_and_justifications(self) -> Self:
        if len(set(self.members)) != len(self.members) or self.representative not in self.members:
            raise ValueError('identity members must be distinct and include representative')
        if len(set(self.justifications)) != len(self.justifications):
            raise ValueError('duplicate identity justification references')
        return self


class Graph(Model):
    schema_version: Literal['gangmu.mathir.v4'] = 'gangmu.mathir.v4'
    snapshot_id: Text
    documents: tuple[Document, ...] = ()
    contexts: tuple[Context, ...] = ()
    units: tuple[Unit, ...] = ()
    relations: tuple[MathematicalRelation, ...] = ()
    evidence: tuple[Evidence, ...] = ()
    identity_views: tuple[IdentityView, ...] = ()

    def records(self) -> tuple[Record, ...]:
        return self.documents + self.contexts + self.units + self.relations + self.evidence + self.identity_views
