"""Seal the prospective D22 C0/C2 seed-1/2 successor from exact D21."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinklab.config import production_binding_for, resolve_config
from sinklab.provenance import canonical_json_bytes, verify_envelope
from sinklab.runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION, validate_runtime_source
from sinklab.stage06_readiness import (
    D21_PROTOCOL_ROOT, LOCK_NAMES, build_d22_lock_set, build_production_configs,
    validate_final_lock_set, validate_production_configs,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _archive(source: Path, target: Path) -> None:
    contents = source.read_bytes()
    if target.exists():
        if target.read_bytes() != contents:
            raise ValueError(f"historical artifact differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(contents)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--production-runtime-source-commit", required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    args = parser.parse_args()
    repo, run_root = args.repo.resolve(), args.run_root.resolve()
    validate_runtime_source(repo, args.production_runtime_source_commit,
                            EXECUTION_CRITICAL_PATH_SET_VERSION,
                            loaded_package_dir=repo / "src/sinklab")
    jobs = [(condition, seed) for seed in (1, 2) for condition in ("C0", "C2")]
    for condition, seed in jobs:
        path = run_root / f"s1-{condition.lower()}-seed{seed}-rtx4080super"
        if path.exists() and any(path.iterdir()):
            parser.error(f"prospective run directory is occupied: {path}")
    current = validate_final_lock_set(repo / "protocols")
    if current["protocol"]["sha256"] != D21_PROTOCOL_ROOT:
        parser.error("D22 requires the exact D21 predecessor root")
    successor = build_d22_lock_set(repo, current, args.production_runtime_source_commit)
    for document in successor.values():
        verify_envelope(document)
    seed0, plan = build_production_configs(repo, successor["protocol"])
    if len(seed0) != 9 or len(plan["jobs"]) != 9:
        raise ValueError("D22 must preserve all nine seed0 jobs")

    optional: dict[str, dict] = {}
    queue = []
    for seed in (1, 2):
        for condition in ("C0", "C2"):
            source = _read(repo / f"configs/production/s1/{condition.lower()}_rtx4080super.json")
            source["protocol_digest"] = successor["protocol"]["sha256"]
            source["production_binding"] = production_binding_for(
                successor["protocol"]["payload"], "rtx4080super", seed)
            resolve_config(source, seed=seed, protocol_lock=successor["protocol"], production=True)
            name = f"{condition.lower()}_seed{seed}_rtx4080super.json"
            optional[name] = source
            queue.append({"condition": condition, "seed": seed,
                          "device_role": "rtx4080super",
                          "run_id": f"s1-{condition.lower()}-seed{seed}-rtx4080super",
                          "config": f"production/s1/{name}",
                          "artifacts": f"D21_seed{seed}_sealed_repack",
                          "predecessor_required": ("D21_C3_seed12_complete" if not queue
                                                   else queue[-1]["run_id"] + "_complete")})

    old_plan = repo / "configs/production/s1_jobs_seed0.json"
    old_paths = [repo / "configs" / row["config"] for row in _read(old_plan)["jobs"]]
    old_optional = [repo / f"configs/production/s1/c3_seed{seed}_rtx4080super.json"
                    for seed in (1, 2)]
    archives = [(repo / "protocols/protocol.lock.json",
                 repo / "protocols/superseded" / f"s1-protocol-{D21_PROTOCOL_ROOT}.json"),
                (old_plan, repo / "configs/superseded/d21/s1_jobs_seed0.json")]
    archives.extend((path, repo / "configs/superseded/d21/s1" / path.name)
                    for path in [*old_paths, *old_optional])
    for source, target in archives:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"historical artifact differs: {target}")
    for source, target in archives:
        _archive(source, target)
    _write(repo / "protocols/protocol.lock.json", successor["protocol"])
    for name, document in seed0.items():
        _write(repo / "configs" / name, document)
    for path in old_optional:
        path.unlink()
    for name, document in optional.items():
        _write(repo / "configs/production/s1" / name, document)
    queue_doc = {"schema_version": 1, "status": "prospective_serial_queue",
                 "predecessor_root_sha256": D21_PROTOCOL_ROOT,
                 "protocol_root_sha256": successor["protocol"]["sha256"],
                 "jobs": queue}
    _write(repo / "configs/production/s1_c0_c2_seed12_queue.json", queue_doc)
    verified = validate_final_lock_set(repo / "protocols")
    summary = validate_production_configs(repo, verified["protocol"])
    for row in queue:
        resolve_config(_read(repo / "configs" / row["config"]), seed=row["seed"],
                       protocol_lock=verified["protocol"], production=True)
    print(json.dumps({"predecessor_root": D21_PROTOCOL_ROOT,
                      "successor_root": successor["protocol"]["sha256"],
                      "seed0_jobs": summary, "optional_jobs": queue}, sort_keys=True))


if __name__ == "__main__":
    main()
