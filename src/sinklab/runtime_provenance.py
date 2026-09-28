"""Fail closed when production Python differs from its immutable source milestone."""

from __future__ import annotations

import subprocess
from pathlib import Path

from .provenance import COMMIT_PATTERN


EXECUTION_CRITICAL_PATH_SET_VERSION = 1
EXECUTION_CRITICAL_PATHS = ("src/sinklab", "pyproject.toml", "uv.lock")
IMPORTABLE_SUFFIXES = {".py", ".pyw", ".pyc", ".pyd", ".so", ".dll"}


class RuntimeSourceError(ValueError):
    """The active production implementation has unapproved source content."""


def _git(repo: Path, *arguments: str) -> subprocess.CompletedProcess[bytes]:
    result = subprocess.run(("git", "-C", str(repo), *arguments),
                            capture_output=True, check=False)
    if result.returncode not in (0, 1):
        raise RuntimeSourceError(
            f"git provenance check failed: {arguments[0]}: "
            f"{result.stderr.decode('utf-8', errors='replace').strip()}")
    return result


def validate_runtime_source(repo: Path, production_runtime_source_commit: str,
                            path_set_version: int, *, loaded_package_dir: Path) -> dict:
    """Accept an exact source milestone or a descendant changing only other paths.

    The versioned path set includes every tracked file in the package directory,
    pyproject.toml and uv.lock. Git compares committed content using its normal
    clean filters, so Windows checkout line endings do not create false changes.
    An additional filesystem walk catches ignored importable modules too.
    """
    repo = Path(repo).resolve()
    if path_set_version != EXECUTION_CRITICAL_PATH_SET_VERSION:
        raise RuntimeSourceError("unsupported execution-critical path-set version")
    if (not isinstance(production_runtime_source_commit, str) or
            not COMMIT_PATTERN.fullmatch(production_runtime_source_commit)):
        raise RuntimeSourceError("production runtime source must be a full commit SHA")
    if Path(loaded_package_dir).resolve() != (repo / "src/sinklab").resolve():
        raise RuntimeSourceError("loaded sinklab package is not the approved checkout")
    top = _git(repo, "rev-parse", "--show-toplevel")
    if top.returncode != 0 or Path(top.stdout.decode().strip()).resolve() != repo:
        raise RuntimeSourceError("production source is not a Git checkout root")
    if _git(repo, "cat-file", "-e", f"{production_runtime_source_commit}^{{commit}}").returncode:
        raise RuntimeSourceError("production runtime source commit does not exist")
    head = _git(repo, "rev-parse", "HEAD")
    if head.returncode or not COMMIT_PATTERN.fullmatch(head.stdout.decode().strip()):
        raise RuntimeSourceError("current production HEAD is unavailable")
    head_sha = head.stdout.decode().strip()
    if _git(repo, "merge-base", "--is-ancestor",
            production_runtime_source_commit, head_sha).returncode:
        raise RuntimeSourceError("production runtime source is not an ancestor of HEAD")
    if _git(repo, "diff", "--no-ext-diff", "--quiet",
            production_runtime_source_commit, head_sha, "--",
            *EXECUTION_CRITICAL_PATHS).returncode:
        raise RuntimeSourceError("execution-critical committed tree differs from source milestone")
    if _git(repo, "diff", "--no-ext-diff", "--quiet", "HEAD", "--",
            *EXECUTION_CRITICAL_PATHS).returncode or _git(
                repo, "diff", "--cached", "--no-ext-diff", "--quiet", "HEAD",
                "--", *EXECUTION_CRITICAL_PATHS).returncode:
        raise RuntimeSourceError("dirty execution-critical tracked file")
    status = _git(repo, "status", "--porcelain=v1", "--untracked-files=all", "-z",
                  "--", *EXECUTION_CRITICAL_PATHS)
    if status.returncode or status.stdout:
        raise RuntimeSourceError("dirty or untracked execution-critical path")
    tracked = _git(repo, "ls-files", "-z", "--", "src/sinklab")
    if tracked.returncode:
        raise RuntimeSourceError("cannot enumerate tracked runtime package files")
    tracked_paths = {item.decode("utf-8").replace("\\", "/")
                     for item in tracked.stdout.split(b"\0") if item}
    package = repo / "src/sinklab"
    if not package.is_dir():
        raise RuntimeSourceError("runtime package directory is missing")
    for path in package.rglob("*"):
        if "__pycache__" in path.relative_to(package).parts:
            continue
        if path.is_symlink():
            raise RuntimeSourceError("execution-critical package contains a symlink")
        if path.is_file() and path.suffix.lower() in IMPORTABLE_SUFFIXES:
            if path.relative_to(repo).as_posix() not in tracked_paths:
                raise RuntimeSourceError("untracked importable execution-critical file")
    return {"production_runtime_source_commit": production_runtime_source_commit,
            "current_head": head_sha,
            "execution_critical_path_set_version": path_set_version,
            "execution_critical_paths": list(EXECUTION_CRITICAL_PATHS),
            "tracked_tree_equal": True, "working_tree_clean": True}
