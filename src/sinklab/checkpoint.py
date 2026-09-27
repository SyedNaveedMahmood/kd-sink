"""Verified, atomic engineering checkpoints with safe Torch loading."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from safetensors.torch import load_file, save_file

from .provenance import canonical_json_bytes, payload_digest


class CheckpointError(ValueError):
    pass


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sync_file(path: Path) -> None:
    with path.open("rb+") as file:
        os.fsync(file.fileno())


def _atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_bytes(canonical_json_bytes(data) + b"\n")
        _sync_file(temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


@contextmanager
def writer_lock(root: Path) -> Iterator[None]:
    root.mkdir(parents=True, exist_ok=True)
    lock = root / ".writer.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as exc:
        raise CheckpointError("duplicate writer; inspect the existing lock") from exc
    try:
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _name(step: int, kind: str) -> str:
    if type(step) is not int or step < 0 or kind not in {"weights", "rolling", "final"}:
        raise CheckpointError("invalid checkpoint step/class")
    return f"{kind}-{step:06d}"


def save_checkpoint(root: Path, *, step: int, kind: str, model: torch.nn.Module,
                    state: dict[str, Any] | None, identity: dict[str, Any],
                    fail_at: str | None = None) -> Path:
    """Call at update boundaries while holding writer_lock."""
    root = Path(root)
    name = _name(step, kind)
    destination = root / name
    if destination.exists():
        verify_checkpoint(destination, identity=identity)
        prior = load_file(str(destination / "model.safetensors"), device="cpu")
        current = model.state_dict()
        if set(prior) != set(current) or any(not torch.equal(prior[k], v.detach().cpu())
                                             for k, v in current.items()):
            raise CheckpointError("same-step checkpoint has different model content")
        return destination
    if kind != "weights" and state is None or kind == "weights" and state is not None:
        raise CheckpointError("full and weights-only classes require distinct payloads")
    temporary = root / f".{name}.{uuid.uuid4().hex}.tmp"
    temporary.mkdir(parents=True)
    try:
        tensors = {k: v.detach().cpu().contiguous().clone() for k, v in model.state_dict().items()}
        save_file(tensors, str(temporary / "model.safetensors"))
        _sync_file(temporary / "model.safetensors")
        if state is not None:
            torch.save(state, temporary / "state.pt")
            _sync_file(temporary / "state.pt")
        manifest = {"schema_version": 1, "step": step, "kind": kind,
                    "identity": identity, "identity_sha256": payload_digest(identity),
                    "files": {p.name: _sha(p) for p in temporary.iterdir() if p.is_file()}}
        (temporary / "manifest.json").write_bytes(canonical_json_bytes(manifest) + b"\n")
        _sync_file(temporary / "manifest.json")
        if fail_at == "before_complete":
            raise OSError("injected pre-complete save failure")
        (temporary / "COMPLETE").write_text(_sha(temporary / "manifest.json") + "\n", encoding="ascii")
        _sync_file(temporary / "COMPLETE")
        if fail_at == "before_rename":
            raise OSError("injected pre-rename save failure")
        os.replace(temporary, destination)
        if fail_at == "after_rename":
            raise OSError("injected post-rename save interruption")
        verify_checkpoint(destination, identity=identity)
        return destination
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def verify_checkpoint(path: Path, *, identity: dict[str, Any] | None = None) -> dict:
    path = Path(path)
    try:
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        if set(manifest) != {"schema_version", "step", "kind", "identity", "identity_sha256", "files"}:
            raise CheckpointError("checkpoint manifest schema mismatch")
        if path.name != _name(manifest["step"], manifest["kind"]):
            raise CheckpointError("checkpoint path and manifest disagree")
        if manifest["identity_sha256"] != payload_digest(manifest["identity"]):
            raise CheckpointError("checkpoint identity hash mismatch")
        if identity is not None and manifest["identity"] != identity:
            raise CheckpointError("immutable run identity mismatch")
        if (path / "COMPLETE").read_text(encoding="ascii").strip() != _sha(path / "manifest.json"):
            raise CheckpointError("checkpoint incomplete")
        if set(manifest["files"]) != ({"model.safetensors"} if manifest["kind"] == "weights" else
                                       {"model.safetensors", "state.pt"}):
            raise CheckpointError("checkpoint file list mismatch")
        for filename, digest in manifest["files"].items():
            if _sha(path / filename) != digest:
                raise CheckpointError("checkpoint content hash mismatch")
        return manifest
    except (OSError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, CheckpointError):
            raise
        raise CheckpointError(f"cannot verify checkpoint: {exc}") from exc


def load_checkpoint(path: Path, *, model: torch.nn.Module,
                    identity: dict[str, Any]) -> tuple[dict, dict | None]:
    manifest = verify_checkpoint(path, identity=identity)
    weights = load_file(str(Path(path) / "model.safetensors"), device="cpu")
    model.load_state_dict(weights, strict=True)
    state = None
    if manifest["kind"] != "weights":
        # weights_only rejects arbitrary classes; all saved non-tensor state is primitive.
        state = torch.load(Path(path) / "state.pt", map_location="cpu", weights_only=True)
    return manifest, state


def latest_full(root: Path, *, identity: dict[str, Any]) -> Path:
    candidates = []
    for path in Path(root).glob("*"):
        if not path.is_dir() or not (path.name.startswith("rolling-") or path.name.startswith("final-")):
            continue
        try:
            manifest = verify_checkpoint(path, identity=identity)
            candidates.append((manifest["step"], manifest["kind"] == "final", path))
        except CheckpointError:
            continue
    if not candidates:
        raise CheckpointError("no verified full checkpoint available")
    return max(candidates)[2]


def prune_rolling(root: Path, *, identity: dict[str, Any]) -> None:
    """Keep two verified rolling generations; protected final is never pruned."""
    valid = []
    for path in Path(root).glob("rolling-*"):
        try:
            valid.append((verify_checkpoint(path, identity=identity)["step"], path))
        except CheckpointError:
            continue
    for _, path in sorted(valid, reverse=True)[2:]:
        shutil.rmtree(path)
