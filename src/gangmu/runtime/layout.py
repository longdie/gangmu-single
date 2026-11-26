"""Explicit project storage namespaces; code and provider admission stay shared.

No implicit path remapping, process globals, or symlinks are used. A missing
project means the historical layout, whose records must remain at their paths.
"""
from dataclasses import dataclass
from pathlib import Path
import re


class LayoutError(ValueError):
    """A public, credential-free project/path error."""


def identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}", value):
        raise LayoutError("Identifiers must use letters, digits, '.', '_' or '-' (1–96 characters)")
    return value


def project_of(judge):
    return getattr(judge, "config", {}).get("project")


def storage_relative(value, project=None):
    path = Path(value)
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] not in {"artifacts", "runs"}:
        raise LayoutError("Storage paths must be relative artifact or run paths")
    return (Path(path.parts[0]) / identifier(project) / Path(*path.parts[1:])
            if project is not None else path).as_posix()


def storage_path(root, value, project=None):
    from gangmu.single_document.implicit_client import _local
    return _local(root, storage_relative(value, project))


@dataclass(frozen=True)
class ProjectLayout:
    root: Path
    project: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "root", Path(self.root).resolve(strict=True))
        if self.project is not None:
            identifier(self.project)

    def local(self, value):
        from gangmu.single_document.implicit_client import _local
        return _local(self.root, value)

    def artifact(self, name):
        return storage_path(self.root, "artifacts/" + name, self.project)

    def run(self, session):
        return storage_path(self.root, "runs/" + identifier(session), self.project)

    def knowledge_base(self, name):
        return self.artifact("knowledge-bases/" + identifier(name))

    def single(self, paper=None, fingerprint=None, *, legacy_kb="default"):
        base = (Path("results") / self.project / "single_doc" if self.project else
                Path("results") / identifier(legacy_kb) / "documents")
        if paper is not None:
            if self.project and paper == "index.json":
                raise LayoutError("Document ID 'index.json' is reserved for the project result index")
            base /= identifier(paper)
        if fingerprint is not None:
            base /= identifier(fingerprint)
        return self.local(base)

    def cross(self, kb):
        base = Path("results")
        if self.project:
            base = base / self.project / "cross_doc"
        return self.local(base / identifier(kb))

    def input(self):
        return self.local(Path("data") / (self.project or "input"))

    def require_owned_input(self, path):
        """Inputs may be external, but another managed project's input is not implicit."""
        path = Path(path).resolve(strict=True)
        data = self.root / "data"
        if self.project and path.is_relative_to(data) and not path.is_relative_to(self.input()):
            raise LayoutError("Input belongs to another data project; select its --project explicitly")

    def require_owned_artifact(self, path):
        path = self.local(path)
        if self.project and not any(path.is_relative_to(base) for base in (
                self.artifact(""), self.local(Path("runs") / self.project),
                self.local(Path("results") / self.project))):
            raise LayoutError("Integration inputs must belong to the selected project; legacy data needs explicit migration")
        return path
