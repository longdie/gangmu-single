"""Build commands for an offline, filesystem-isolated TeX process.

Only the supplied per-build directory is writable on the host.  The caller
must pass TeX's ``-no-shell-escape`` and latexmk's ``-norc`` options, enforce
runtime/resource limits, and run against a copy of the original sources.
"""

from __future__ import annotations

from pathlib import Path
import shutil


_PROJECT_ROOT = Path(__file__).resolve().parents[3]
_READ_ONLY_MOUNTS = (
    "/usr",
    "/bin",
    "/lib",
    "/lib64",
    "/etc/texmf",
    "/etc/fonts",
    "/etc/paperspecs",
    "/etc/papersize",
    "/var/lib/texmf",
    "/var/cache/fontconfig",
)


def _project_child(path: Path, label: str) -> Path:
    resolved = path.resolve()
    if resolved == _PROJECT_ROOT or not resolved.is_relative_to(_PROJECT_ROOT):
        raise ValueError(f"{label} must be a directory below the project root")
    return resolved


def sandbox_command(
    workdir: Path,
    argv: list[str],
    *,
    tex_overlay: Path | None = None,
) -> list[str]:
    """Return a bubblewrap command; do not execute a compiler.

    ``workdir`` is mounted read/write at ``/work`` and becomes the process
    working directory.  All paths in ``argv`` must therefore be relative to
    ``/work`` or use sandbox paths.  ``tex_overlay``, if present, must be a
    project-local TeX Directory Structure tree; it is mounted read-only and
    searched before system TeX trees.  Original archives, other runs, the
    project configuration and the host home directory are not mounted.

    Creating the command also creates ``workdir/_sandbox/`` for per-build
    home, TeX state and temporary/cache directories.  Bubblewrap or namespace
    failures must be reported by the caller, never retried without isolation.
    """
    if not argv or any(not isinstance(arg, str) or "\0" in arg for arg in argv):
        raise ValueError("argv must be a nonempty argument list without NUL")
    if not argv[0]:
        raise ValueError("argv[0] must name a command")
    bwrap = shutil.which("bwrap")
    if not bwrap:
        raise RuntimeError("bubblewrap is required; unsandboxed execution is disabled")
    build_dir = _project_child(Path(workdir), "workdir")
    if not build_dir.is_dir():
        raise ValueError("workdir must already exist and contain the build copy")
    overlay = None
    if tex_overlay is not None:
        overlay = _project_child(Path(tex_overlay), "tex_overlay")
        if not overlay.is_dir():
            raise ValueError("tex_overlay must be an existing TeX tree directory")
        if overlay.is_relative_to(build_dir) or build_dir.is_relative_to(overlay):
            raise ValueError("tex_overlay and workdir must be separate directory trees")
    state = build_dir / "_sandbox"
    directories = ("home", "tmp", "cache", "texmf-home", "texmf-var", "texmf-config")
    # Reject a source archive that pre-populated state paths with symlinks.
    for directory in (state, *(state / name for name in directories)):
        if directory.is_symlink() or not directory.resolve().is_relative_to(build_dir):
            raise ValueError("sandbox state directories may not redirect outside workdir")
        if directory.exists() and not directory.is_dir():
            raise ValueError("sandbox state path must be a directory")
    for name in directories:
        (state / name).mkdir(parents=True, exist_ok=True)

    command = [
        bwrap,
        "--unshare-all",
        "--unshare-user",
        "--disable-userns",
        "--assert-userns-disabled",
        "--die-with-parent",
        "--new-session",
        "--cap-drop", "ALL",
        "--clearenv",
    ]
    for mount in _READ_ONLY_MOUNTS:
        if Path(mount).exists():
            command.extend(("--ro-bind", mount, mount))
    command.extend((
        "--proc", "/proc",
        "--dev", "/dev",
        "--bind", str(build_dir), "/work",
        "--bind", str(state / "tmp"), "/tmp",
        "--chdir", "/work",
    ))
    if overlay is not None:
        command.extend(("--ro-bind", str(overlay), "/texmf-overlay"))
    environment = {
        "PATH": "/usr/bin:/bin",
        "HOME": "/work/_sandbox/home",
        "TMPDIR": "/work/_sandbox/tmp",
        "XDG_CACHE_HOME": "/work/_sandbox/cache",
        "TEXMFHOME": "/texmf-overlay" if overlay is not None else "/work/_sandbox/texmf-home",
        "TEXMFVAR": "/work/_sandbox/texmf-var",
        "TEXMFCONFIG": "/work/_sandbox/texmf-config",
        "TEXMFCACHE": "/work/_sandbox/texmf-var",
        "TEXMFOUTPUT": "/work",
        # Absolute TeX-tree reads are required by LuaTeX. The filesystem
        # namespace is the access boundary; 'r' additionally rejects dotfiles.
        "openin_any": "r",
        "openout_any": "p",
        "shell_escape": "f",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
    }
    for name, value in environment.items():
        command.extend(("--setenv", name, value))
    # The synthetic root must become read-only after creating mount targets.
    # This does not recursively remount the explicit writable /work and /tmp.
    command.extend(("--remount-ro", "/", "--", *argv))
    return command
