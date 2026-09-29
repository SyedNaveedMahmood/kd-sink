"""Seal the prospective D20 C3/4080 successor from the exact D19 root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinklab.provenance import canonical_json_bytes, verify_envelope
from sinklab.runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION, validate_runtime_source
from sinklab.stage06_readiness import (
    D19_PROTOCOL_ROOT, LOCK_NAMES, build_d20_lock_set, build_production_configs,
    validate_final_lock_set, validate_production_configs,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_or_verify(path: Path, document: dict) -> None:
    if path.exists():
        if _read(path) != document:
            raise ValueError(f"historical artifact differs: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--production-runtime-source-commit", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    validate_runtime_source(repo, args.production_runtime_source_commit,
                            EXECUTION_CRITICAL_PATH_SET_VERSION,
                            loaded_package_dir=repo / "src/sinklab")
    locks = repo / "protocols"
    old = validate_final_lock_set(locks)
    if old["protocol"]["sha256"] != D19_PROTOCOL_ROOT:
        parser.error("D20 sealer requires the exact current D19 root")
    new, reviewed = build_d20_lock_set(repo, old, args.production_runtime_source_commit)
    configs, production_plan = build_production_configs(
        repo, new["protocol"], reviewed_plan=reviewed)
    if len(configs) != 9 or len(production_plan["jobs"]) != 9:
        raise ValueError("D20 requires precisely nine physical jobs")
    for document in new.values():
        verify_envelope(document)

    # Every historical document is checked before any authoritative path changes.
    history = locks / "superseded"
    _write_or_verify(history / f"s1-protocol-{D19_PROTOCOL_ROOT}.json", old["protocol"])
    _write_or_verify(history / f"s1-hardware-{old['hardware']['sha256']}.json", old["hardware"])
    old_plan = _read(repo / "configs/production/s1_jobs_seed0.json")
    old_reviewed = _read(repo / "configs/s1_jobs_seed0_reviewed.json")
    _write_or_verify(repo / "configs/superseded/d19/s1_jobs_seed0.json", old_plan)
    _write_or_verify(repo / "configs/superseded/d19/s1_jobs_seed0_reviewed.json", old_reviewed)
    for row in old_plan["jobs"]:
        path = repo / "configs" / row["config"]
        _write_or_verify(repo / "configs/superseded/d19" / row["config"], _read(path))

    (repo / "configs/s1_jobs_seed0_reviewed.json").write_bytes(canonical_json_bytes(reviewed) + b"\n")
    for name in ("hardware", "protocol"):
        (locks / f"{name}.lock.json").write_bytes(canonical_json_bytes(new[name]) + b"\n")
    old_c3 = repo / "configs/production/s1/c3_rtx3090.json"
    if old_c3.exists():
        old_c3.unlink()
    for name, document in configs.items():
        path = repo / "configs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(document) + b"\n")
    (repo / "configs/production/s1_jobs_seed0.json").write_bytes(
        canonical_json_bytes(production_plan) + b"\n")
    validated = validate_final_lock_set(locks)
    summary = validate_production_configs(repo, validated["protocol"])
    print(json.dumps({"predecessor_root": D19_PROTOCOL_ROOT,
                      "successor_root": new["protocol"]["sha256"],
                      "hardware_lock_sha256": new["hardware"]["sha256"],
                      "job_plan": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
