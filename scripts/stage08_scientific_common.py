"""Shared, fail-closed preflight helpers for Stage08 S4/S6 runs."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import socket
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from sinklab.followup_policy import (D24_PATH, D24_SHA256, admit_s1_followup,
                                     validate_followup_amendment)
from sinklab.provenance import canonical_json_bytes, seal_payload, verify_envelope

from scripts.stage08_readiness import (read_json, sha256, verify_required_checkpoint,
                                       verify_training_log)

REPO = Path(__file__).resolve().parents[1]
CONDITIONS = tuple(f"C{i}" for i in range(7))
RUN_INVENTORY = REPO / "reports" / "stage08_adrita_preparation.json"
EXCEPTION_PATH = REPO / "protocols" / "s1_c3_seed0_transferred_log_exception.json"
EXPECTED_GPU_UUID = "GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf"
EXPECTED_GPU_NAME = "NVIDIA GeForce RTX 4080 SUPER"


class Stage08Blocked(RuntimeError):
    """A required execution resource or external artifact is unavailable."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scientific_source_hashes(relative_paths: tuple[str, ...]) -> dict[str, str]:
    return {relative: sha256_file(REPO / relative) for relative in relative_paths}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_new(path: Path, value: dict) -> None:
    write_new(path, canonical_json_bytes(value) + b"\n")


def write_sealed_new(path: Path, payload: dict) -> str:
    document = seal_payload(payload)
    write_json_new(path, document)
    return document["sha256"]


def current_gpu() -> dict:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,uuid,pci.bus_id,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=15)
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        raise Stage08Blocked(f"cannot verify the authorized local GPU: {exc}") from exc
    rows = list(csv.reader(result.stdout.splitlines(), skipinitialspace=True))
    matches = [row for row in rows if len(row) == 4 and row[1].strip() == EXPECTED_GPU_UUID]
    if len(matches) != 1:
        raise Stage08Blocked("Adrita-PC's recorded RTX 4080 SUPER UUID is not uniquely visible")
    name, uuid, pci_bus_id, memory_mib = (value.strip() for value in matches[0])
    if name != EXPECTED_GPU_NAME:
        raise Stage08Blocked(f"inference GPU model differs from Adrita-PC lock: {name}")
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise Stage08Blocked("CUDA device is unavailable")
    active_name = torch.cuda.get_device_name(0)
    if active_name != EXPECTED_GPU_NAME:
        raise Stage08Blocked(f"CUDA device 0 is not the approved local RTX 4080 SUPER: {active_name}")
    return {"name": name, "uuid": uuid, "pci_bus_id": pci_bus_id,
            "memory_mib": int(memory_mib), "cuda_index": 0}


def source_run_directories() -> dict[str, Path]:
    report = read_json(RUN_INVENTORY)
    run_map = report.get("s4", {}).get("required_run_paths", {})
    if set(run_map) != set(CONDITIONS):
        raise ValueError("Stage08 readiness report lacks the seven exact S1 source paths")
    paths = {condition: Path(run_map[condition]).resolve() for condition in CONDITIONS}
    if len({path.name for path in paths.values()}) != len(CONDITIONS):
        raise ValueError("S1 source run IDs are not unique")
    return paths


def preflight_sources(*, steps: tuple[int, ...], study: str,
                      runs_root: Path | None = None,
                      progress=None) -> dict:
    """Verify original run logs, protocol seals and every required checkpoint."""
    d24_document = read_json(REPO / D24_PATH)
    d24_payload = validate_followup_amendment(d24_document)
    if d24_document["sha256"] != D24_SHA256:
        raise ValueError("D24 seal mismatch")
    exception_document = read_json(EXCEPTION_PATH)
    report_paths = source_run_directories()
    if runs_root is not None:
        runs_root = runs_root.resolve()
        if any(not path.is_relative_to(runs_root) for path in report_paths.values()):
            raise ValueError("readiness source path escapes the requested central run root")

    sources = {}
    for condition in CONDITIONS:
        run_dir = report_paths[condition]
        if progress:
            progress({"event": "source_preflight_start", "study": study,
                      "condition": condition, "run": str(run_dir)})
        final_manifest_path = run_dir / "checkpoints" / "final-010000" / "manifest.json"
        final_manifest = read_json(final_manifest_path)
        identity = final_manifest.get("identity")
        if (not isinstance(identity, dict) or identity.get("study") != "S1" or
                identity.get("condition") != condition or identity.get("seed") != 0 or
                identity.get("run_id") != run_dir.name or identity.get("protocol_hash") is None):
            raise ValueError(f"original S1 identity mismatch for {condition}: {final_manifest_path}")
        protocol_path = REPO / "protocols" / "superseded" / f"s1-protocol-{identity['protocol_hash']}.json"
        protocol_document = read_json(protocol_path)
        protocol_payload, protocol_sha = verify_envelope(protocol_document)
        if (protocol_sha != identity["protocol_hash"] or
                protocol_payload.get("study") != "S1" or
                protocol_payload.get("condition_variants", {}).get(condition) is None or
                identity.get("hardware_hash") != protocol_payload.get("hardware_lock_digest") or
                identity.get("calibration_hash") != protocol_payload.get("calibration_lock_digest")):
            raise ValueError(f"original sealed protocol lineage mismatch for {condition}")
        log_audit = verify_training_log(run_dir, identity, exception_document=exception_document)
        if (condition == "C3" and log_audit["status"] != "accepted_historical_prefix_gap"):
            if log_audit["status"] != "complete":
                raise ValueError("C3 log status unexpected")
        elif condition != "C3" and log_audit["status"] != "complete":
            raise ValueError(f"unexpected training provenance status for {condition}")
        if log_audit["last_update"] != 10000:
            raise ValueError(f"source training history does not end at 10000: {condition}")
        admit_s1_followup(identity, study=study, step=steps[0])

        checkpoint_rows = []
        for step in steps:
            path = (run_dir / "checkpoints" /
                    ("final-010000" if step == 10000 else f"weights-{step:06d}"))
            verified = verify_required_checkpoint(path, identity=identity, step=step)
            expected_kind = "final" if step == 10000 else "weights"
            if verified["kind"] != expected_kind:
                raise ValueError(f"checkpoint kind mismatch for {condition}/step{step}")
            # D24 is checked against each retained state, not just the final run manifest.
            followup = admit_s1_followup(verified["identity"], study=study, step=step)
            checkpoint_rows.append({**verified, "followup_policy": followup,
                "manifest_path": str(path / "manifest.json"),
                "manifest_file_sha256": sha256(path / "manifest.json"),
                "model_path": str(path / "model.safetensors"),
                "model_file_sha256": verified["payload_sha256"]["model.safetensors"],
                "model_stat": {"bytes": (path / "model.safetensors").stat().st_size,
                               "mtime_ns": (path / "model.safetensors").stat().st_mtime_ns}})
            if progress:
                progress({"event": "checkpoint_verified", "study": study,
                          "condition": condition, "step": step,
                          "path": str(path), "model_sha256": checkpoint_rows[-1]["model_file_sha256"]})
        sources[condition] = {"run_path": str(run_dir), "run_id": identity["run_id"],
            "identity": identity, "original_protocol_root_sha256": identity["protocol_hash"],
            "protocol_path": str(protocol_path), "protocol_file_sha256": sha256(protocol_path),
            "protocol_payload_sha256": protocol_sha,
            "condition_variant": protocol_payload["condition_variants"][condition],
            "training_log": log_audit,
            "final_manifest_sha256": sha256(final_manifest_path),
            "checkpoints": checkpoint_rows}
        if progress:
            progress({"event": "source_preflight_complete", "study": study,
                      "condition": condition, "checkpoint_count": len(checkpoint_rows),
                      "training_log_status": log_audit["status"]})
    return {"study": study, "D24_sha256": D24_SHA256,
            "D24_interpretation": d24_payload["interpretation"],
            "sources": sources}


def model_stat_is_stable(path: Path, stat: dict) -> bool:
    current = path.stat()
    return current.st_size == stat["bytes"] and current.st_mtime_ns == stat["mtime_ns"]


def output_path_is_external(path: Path, source_paths: list[Path]) -> None:
    path = path.resolve()
    if path == REPO or REPO in path.parents:
        raise ValueError("scientific result output must remain outside Git")
    if any(path == source or source in path.parents for source in source_paths):
        raise ValueError("scientific result output cannot be inside an original source run")


def require_free_space(path: Path, minimum_free_bytes: int) -> int:
    anchor = Path(path.resolve().anchor)
    free = shutil.disk_usage(anchor).free
    if free < minimum_free_bytes:
        raise Stage08Blocked(f"insufficient safe free space on {anchor}: {free} < {minimum_free_bytes} bytes")
    return free


def failure_status(exc: Exception) -> str:
    """Separate unavailable execution resources from scientific/integrity failures."""
    if isinstance(exc, (Stage08Blocked, FileNotFoundError, PermissionError)):
        return "SKIPPED_BLOCKED"
    try:
        import torch
        if isinstance(exc, torch.cuda.OutOfMemoryError):
            return "SKIPPED_BLOCKED"
    except (ImportError, AttributeError):
        pass
    return "FAILED_INTEGRITY_CHECK"


def write_or_validate_run_manifest(path: Path, payload: dict) -> str:
    """Create a sealed immutable run binding or verify an existing resume binding."""
    volatile = {"created_utc", "repository_commit"}
    expected = {k: v for k, v in payload.items() if k not in volatile}
    if path.exists():
        existing, digest = verify_envelope(read_json(path))
        actual = {k: v for k, v in existing.items() if k not in volatile}
        if actual != expected:
            raise ValueError(f"resume run manifest conflicts with current sources/panel/device: {path}")
        return digest
    manifest = seal_payload(payload)
    write_json_new(path, manifest)
    return manifest["sha256"]
