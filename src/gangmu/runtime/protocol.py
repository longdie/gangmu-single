"""Content-bound proposals; these types alone certify no mathematics.

Runtime certificates describe a finite proof/check graph outside MathIR. A
registered checker, never an agent's success flag, closes each primitive/rule.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


def canonical(value: object) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode('utf-8')


def sha(value: object) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def same(left: object, right: object) -> bool:
    """Exact JSON content: Python's True == 1 is not semantic identity."""
    return canonical(left) == canonical(right)


def clone(value):
    return json.loads(canonical(value))


class Reject(ValueError):
    """A protocol outcome; ``mismatch`` must come from a scoped checker."""

    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f'{code}: {message}')


class IntegrityError(Reject):
    def __init__(self, message: str):
        super().__init__('integrity_error', message)


Text = Annotated[str, Field(min_length=1)]
Hash = Annotated[str, Field(pattern=r'^[0-9a-f]{64}$')]
Kind = Literal['source', 'interpretation', 'context', 'premise', 'rule',
               'transport', 'identity', 'usage', 'reuse', 'unresolved',
               'revise', 'withdraw', 'restore', 'graph']


class Model(BaseModel):
    model_config = ConfigDict(extra='forbid', frozen=True, strict=True)


class Claim(Model):
    id: Text
    kind: Kind
    payload: dict[str, JsonValue]
    scope: dict[str, JsonValue] = Field(default_factory=dict)


class Node(Model):
    id: Text
    mode: Literal['primitive', 'derive', 'reference']
    conclusion: dict[str, JsonValue]
    scope: dict[str, JsonValue] = Field(default_factory=dict)
    obligation: Text
    checker: Text | None = None
    method: Text | None = None
    premises: tuple[Text, ...] = ()
    resources: dict[Text, Hash] = Field(default_factory=dict)
    reference: Text | None = None

    @model_validator(mode='after')
    def shape(self):
        if len(set(self.premises)) != len(self.premises):
            raise ValueError('duplicate premise')
        if self.mode == 'reference':
            if not self.reference or self.checker or self.method or self.premises:
                raise ValueError('reference nodes name only an existing support')
        elif not self.checker or not self.method or self.reference:
            raise ValueError('primitive/rule nodes require a checker and method')
        if self.mode == 'primitive' and self.premises:
            raise ValueError('primitive nodes have no logical premises')
        if self.mode == 'derive' and not self.premises:
            raise ValueError('a zero-premise axiom needs a primitive checker')
        return self


class Certificate(Model):
    nodes: Annotated[tuple[Node, ...], Field(min_length=1)]
    root: Text


class Operation(Model):
    kind: Kind
    resource: Text | None = None
    value: JsonValue = None
    claim: Claim | None = None
    certificate: Certificate | None = None
    target: Text | None = None
    reason: Text | None = None
    graph: Hash | None = None
    delete: bool = False

    @model_validator(mode='after')
    def shape(self):
        active = {key for key, value in self.model_dump().items()
                  if value is not None and (key != 'delete' or value is True) and key != 'kind'}
        if self.kind == 'graph':
            allowed = {'graph'}
            required = {'graph'}
        elif self.kind == 'source':
            allowed = {'resource', 'value', 'claim', 'certificate'}
            required = {'resource', 'value'}
            if bool(self.claim) != bool(self.certificate):
                raise ValueError('source claim and certificate must be supplied together')
        elif self.kind == 'unresolved':
            allowed = {'target', 'reason', 'value'}
            required = {'target', 'reason'}
        elif self.kind == 'withdraw':
            allowed = required = {'target', 'reason'}
        elif self.kind == 'restore':
            allowed = {'target', 'certificate', 'reason'}
            required = {'target', 'certificate'}
        elif self.kind == 'revise' and self.delete:
            allowed = required = {'resource', 'reason', 'delete'}
        else:
            allowed = {'resource', 'value', 'claim', 'certificate', 'reason'}
            required = {'claim', 'certificate'}
        if not required <= active or not active <= allowed:
            raise ValueError(f'invalid {self.kind} operation fields')
        return self


class Proposal(Model):
    id: Text
    revision: Annotated[int, Field(ge=0)]
    operations: Annotated[tuple[Operation, ...], Field(min_length=1)]

    @property
    def content_hash(self) -> str:
        return sha(self.model_dump(mode='json'))


@dataclass(frozen=True)
class ValidationTicket:
    """A process-local authenticated, immutable validation record."""

    proposal: bytes
    observations: bytes
    signature: str


@dataclass(frozen=True)
class Receipt:
    proposal_id: str
    revision: int
    proposal_hash: str
    release: int
    release_hash: str
    event: int

    def as_dict(self) -> dict:
        return dict(self.__dict__)
