"""Seal optional C3-only seeds 1/2 from the exact D20 production root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinklab.config import production_binding_for, resolve_config
from sinklab.provenance import canonical_json_bytes, verify_envelope
from sinklab.runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION, validate_runtime_source
from sinklab.stage06_readiness import (
    D20_PROTOCOL_ROOT, LOCK_NAMES, build_d21_lock_set, build_production_configs,
    validate_final_lock_set, validate_production_configs,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _archive(path: Path, target: Path) -> None:
    contents = path.read_bytes()
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
    parser.add_argument("--seed1-run-dir", type=Path, required=True)
    parser.add_argument("--seed2-run-dir", type=Path, required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    validate_runtime_source(repo, args.production_runtime_source_commit,
                            EXECUTION_CRITICAL_PATH_SET_VERSION,
                            loaded_package_dir=repo / "src/sinklab")
    for path in (args.seed1_run_dir, args.seed2_run_dir):
        if path.exists() and any(path.iterdir()):
            parser.error(f"prospective seed run directory is already occupied: {path}")
    current = validate_final_lock_set(repo / "protocols")
    if current["protocol"]["sha256"] != D20_PROTOCOL_ROOT:
        parser.error("D21 requires the exact D20 predecessor root")
    successor = build_d21_lock_set(repo, current,
                                   args.production_runtime_source_commit)
    for document in successor.values():
        verify_envelope(document)
    seed0, plan = build_production_configs(repo, successor["protocol"])
    if len(seed0) != 9 or len(plan["jobs"]) != 9:
        raise ValueError("D21 must preserve all nine seed0 jobs")
    c3_source = _read(repo / "configs/production/s1/c3_rtx4080super.json")
    optional = {}
    queue = []
    for seed in (1, 2):
        config = dict(c3_source)
        config["protocol_digest"] = successor["protocol"]["sha256"]
        config["production_binding"] = production_binding_for(
            successor["protocol"]["payload"], "rtx4080super", seed)
        resolve_config(config, seed=seed, protocol_lock=successor["protocol"],
                       production=True)
        name = f"c3_seed{seed}_rtx4080super.json"
        optional[name] = config
        queue.append({"condition": "C3", "seed": seed, "device_role": "rtx4080super",
                      "run_id": f"s1-c3-seed{seed}-rtx4080super",
                      "config": f"production/s1/{name}",
                      "predecessor_required": "seed0_final_010000_verified" if seed == 1
                                              else "seed1_final_010000_verified"})

    # Verify all historical targets before modifying the authoritative paths.
    old_plan = repo / "configs/production/s1_jobs_seed0.json"
    old_config_paths = [repo / "configs" / row["config"] for row in _read(old_plan)["jobs"]]
    archives = [(repo / "protocols/protocol.lock.json",
                 repo / "protocols/superseded" / f"s1-protocol-{D20_PROTOCOL_ROOT}.json"),
                (old_plan, repo / "configs/superseded/d20/s1_jobs_seed0.json")]
    archives.extend((path, repo / "configs/superseded/d20/s1" / path.name)
                    for path in old_config_paths)
    for source, target in archives:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"historical artifact differs: {target}")
    for source, target in archives:
        _archive(source, target)
    _write(repo / "protocols/protocol.lock.json", successor["protocol"])
    for name, document in seed0.items():
        _write(repo / "configs" / name, document)
    for name, document in optional.items():
        _write(repo / "configs/production/s1" / name, document)
    _write(repo / "configs/production/s1_c3_seed12_queue.json",
           {"schema_version": 1, "status": "prospective_after_separate_seed0_completion",
            "predecessor_root_sha256": D20_PROTOCOL_ROOT,
            "protocol_root_sha256": successor["protocol"]["sha256"],
            "jobs": queue})
    verified = validate_final_lock_set(repo / "protocols")
    summary = validate_production_configs(repo, verified["protocol"])
    print(json.dumps({"predecessor_root": D20_PROTOCOL_ROOT,
                      "successor_root": successor["protocol"]["sha256"],
                      "seed0_jobs": summary,
                      "optional_c3_seeds": [1, 2]}, sort_keys=True))


if __name__ == "__main__":
    main()
