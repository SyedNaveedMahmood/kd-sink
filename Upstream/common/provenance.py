# -*- coding: utf-8 -*-
"""provenance.py — the provenance fields every E6/E7 artefact must carry.

``05_SCHEMAS_AND_CONTRACTS.md`` §7.3 makes provenance mandatory: "A row without provenance
cannot enter the paper." Before this module the repo had no way to obtain a git sha at
runtime — ``BASELINE_HASHES.json`` carries one as a *literal*, captured once at freeze time.
That is fine for the frozen tag and useless for a run started three commits later.

Scope is deliberately narrow: git identity, UTC timestamps, library versions, file and
tensor digests. No model loading, no HF imports, no torch import at module level — WP8's
manifest builders and WP5's training loop both use this and neither should pay for torch
just to stamp a git sha.

Nothing here writes to a frozen file, and nothing here is called by E1–E5 code.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]

#: Bumped whenever the *meaning* of a provenance field changes (``05`` §1.1 discipline).
PROVENANCE_VERSION = "provenance_v1"


# ═══════════════════════════════════════════════════════════════════════════════
# Git identity
# ═══════════════════════════════════════════════════════════════════════════════


def git_sha(repo: Optional[Path] = None, *, short: bool = False) -> str:
    """Current commit sha, or a sentinel when git is unavailable.

    Returns the sentinel ``"unknown"`` rather than raising: a run that cannot see git (a
    tarball export, a container without the ``.git`` directory) must still produce a
    complete artefact, and a sentinel is honest where a fabricated hash would not be
    (CLAUDE.md rule 4). Callers that need a real sha should assert on it themselves.
    """
    root = Path(repo) if repo is not None else _REPO
    try:
        out = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short" if short else "HEAD"],
            capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    if out.returncode != 0:
        return "unknown"
    return out.stdout.strip() or "unknown"


def git_is_dirty(repo: Optional[Path] = None) -> Optional[bool]:
    """``True`` when the working tree has uncommitted changes; ``None`` if git is unavailable.

    Recorded alongside ``git_sha`` because a sha alone does not identify the code that ran
    when the tree is dirty.
    """
    root = Path(repo) if repo is not None else _REPO
    try:
        out = subprocess.run(["git", "-C", str(root), "status", "--porcelain"],
                             capture_output=True, text=True, timeout=15, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    return bool(out.stdout.strip())


def utc_now() -> str:
    """ISO-8601 UTC timestamp, second resolution, for ``created_utc`` / ``measured_utc``."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


# ═══════════════════════════════════════════════════════════════════════════════
# Digests
# ═══════════════════════════════════════════════════════════════════════════════


def sha256_file(path) -> str:
    """SHA-256 of a file's bytes, streamed (checkpoints are hundreds of MB)."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(root, *, patterns: Iterable[str] = ("*",)) -> str:
    """SHA-256 over the sorted (relative-path, file-sha) pairs beneath ``root``.

    Used for ``checkpoint_sha256``: a directory of shards hashes to one stable value, and
    the relative paths are included so a renamed shard changes the hash. Paths are
    normalised with forward slashes so a checkpoint written on Windows and one written on
    Linux hash identically.
    """
    root = Path(root)
    entries = []
    for pattern in patterns:
        for path in sorted(root.rglob(pattern)):
            if path.is_file():
                entries.append((path.relative_to(root).as_posix(), sha256_file(path)))
    entries = sorted(set(entries))
    payload = json.dumps(entries, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_json(obj: Any) -> str:
    """SHA-256 over canonical JSON — the hashing convention used repo-wide.

    Matches ``corpus_providers.compute_manifest_sha256``: ``sort_keys=True``,
    ``ensure_ascii=False``. Two artefacts hashed by different helpers must agree on this
    convention or ``05`` §7.1 ("hash before compare") silently fails open.
    """
    payload = json.dumps(obj, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_int_rows(rows: Iterable[Sequence[int]]) -> str:
    """SHA-256 of a sequence of integer rows, **byte-identical** to ``sha256_json(rows)``.

    Same digest, streamed. ``sha256_json`` materialises one JSON string for the whole
    object, which is fine for a manifest and catastrophic for a corpus: TinyStories packs
    to 3.57M blocks of 128 tokens, and hashing them through ``sha256_json`` costs an extra
    ~8.8 GB of peak RSS and ~12 minutes on top of the ~20 GB the blocks already occupy —
    measured, not estimated. That is what made E6A training unrunnable on a 16 GB host.

    The digest **value** is unchanged, deliberately. ``json.dumps`` of a list of lists of
    ints is fully determined — ``[[1, 2], [3, 4]]`` — so the identical byte stream can be
    fed to ``hashlib`` incrementally. ``sort_keys`` and ``ensure_ascii`` do not apply to
    arrays of ints, so the two agree exactly; ``tests/test_provenance_streaming.py`` asserts
    that on random inputs rather than trusting the argument. Any run hashed with either
    helper is therefore comparable under ``05`` §7.1, which is the property that would
    break if this returned a "different but equally good" digest.

    Accepts anything iterable of integer sequences — a list of lists, or the rows of a
    numpy array — and never holds more than one row's text at a time.
    """
    digest = hashlib.sha256()
    digest.update(b"[")
    first = True
    for row in rows:
        if not first:
            digest.update(b", ")
        first = False
        digest.update(b"[")
        digest.update(", ".join(str(int(value)) for value in row).encode("utf-8"))
        digest.update(b"]")
    digest.update(b"]")
    return digest.hexdigest()


def sha256_state_dict(state_dict, *, sample: int = 0) -> str:
    """SHA-256 of a torch ``state_dict``, over parameters in sorted-name order.

    ``sample=0`` (the default) hashes every byte of every tensor — the exact-equality check
    ``tests/test_distillation_init.py`` needs to prove three conditions share byte-identical
    initial weights.

    ``sample=k > 0`` hashes only each tensor's name, dtype, shape and its first/last ``k``
    flattened elements. That is a *fingerprint*, not a hash: it is for cheaply detecting
    that two handles are not the same model (WP7 invariant 8), and it is not proof of
    equality. Both modes are deterministic across machines because the tensors are moved to
    CPU and, for the sampled mode, serialised as text.
    """
    digest = hashlib.sha256()
    for name in sorted(state_dict):
        tensor = state_dict[name]
        detached = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(detached.dtype).encode("utf-8"))
        digest.update(str(tuple(detached.shape)).encode("utf-8"))
        if sample <= 0:
            digest.update(detached.numpy().tobytes())
            continue
        flat = detached.reshape(-1)
        if flat.numel() == 0:
            continue
        take = min(sample, int(flat.numel()))
        head = flat[:take].to("cpu", copy=True).float().tolist()
        tail = flat[-take:].to("cpu", copy=True).float().tolist()
        digest.update(repr([round(v, 12) for v in head + tail]).encode("utf-8"))
    return digest.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Environment block
# ═══════════════════════════════════════════════════════════════════════════════


def library_versions() -> Dict[str, Optional[str]]:
    """Installed versions of every library whose behaviour can move a reported number.

    Missing optional libraries record ``None`` rather than raising — ``peft`` is only
    present for E6B, and ``nnsight`` only for evaluation.
    """
    import importlib
    import platform

    versions: Dict[str, Optional[str]] = {"python": platform.python_version()}
    for name in ("torch", "transformers", "nnsight", "peft", "numpy", "scipy",
                 "pandas", "datasets"):
        try:
            module = importlib.import_module(name)
        except Exception:  # pragma: no cover - depends on the installed environment
            versions[name] = None
            continue
        versions[name] = str(getattr(module, "__version__", "unknown"))
    return versions


def provenance_block(*, repo: Optional[Path] = None,
                     extra: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The common provenance tail for any artefact (``05`` §1 and §7.3).

    ``extra`` is merged last so a caller can add ``manifest_sha256``, ``seed``, ``dtype``,
    ``device`` and registry versions without a second dict merge at every call site.
    """
    block: Dict[str, Any] = {
        "git_sha": git_sha(repo),
        "git_dirty": git_is_dirty(repo),
        "frozen_baseline_tag": "frozen-e1-e5",
        "created_utc": utc_now(),
        "provenance_version": PROVENANCE_VERSION,
        "versions": library_versions(),
    }
    if extra:
        block.update(extra)
    return block


def write_json(path, obj: Dict[str, Any]) -> Path:
    """Write ``obj`` as UTF-8 JSON, creating parents. Returns the path written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str),
                    encoding="utf-8")
    return path
