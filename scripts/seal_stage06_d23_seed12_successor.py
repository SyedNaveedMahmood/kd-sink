"""Seal all-condition optional seed configs from the exact D22 production root."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinklab.provenance import canonical_json_bytes, verify_envelope
from sinklab.runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION, validate_runtime_source
from sinklab.stage06_readiness import (
    D22_PROTOCOL_ROOT, LOCK_NAMES, build_d23_lock_set, build_optional_seed_configs,
    build_production_configs, validate_final_lock_set, validate_optional_seed_configs,
    validate_production_configs,
)


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def _archive(source: Path, target: Path) -> None:
    content = source.read_bytes()
    if target.exists():
        if target.read_bytes() != content:
            raise ValueError(f"historical artifact differs: {target}")
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--production-runtime-source-commit", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    validate_runtime_source(repo, args.production_runtime_source_commit,
                            EXECUTION_CRITICAL_PATH_SET_VERSION,
                            loaded_package_dir=repo / "src/sinklab")
    current = validate_final_lock_set(repo / "protocols")
    if current["protocol"]["sha256"] != D22_PROTOCOL_ROOT:
        parser.error("D23 requires the exact D22 predecessor root")
    successor = build_d23_lock_set(repo, current, args.production_runtime_source_commit)
    for document in successor.values():
        verify_envelope(document)
    seed0, seed0_plan = build_production_configs(repo, successor["protocol"])
    optional, optional_plan = build_optional_seed_configs(repo, successor["protocol"])
    if len(seed0) != 9 or len(optional) != 14:
        raise ValueError("D23 requires nine unchanged seed0 jobs and fourteen optional jobs")

    old_seed0_plan = repo / "configs/production/s1_jobs_seed0.json"
    old_seed0 = [repo / "configs" / row["config"] for row in _read(old_seed0_plan)["jobs"]]
    old_optional_plan = repo / "configs/production/s1_c0_c2_seed12_queue.json"
    old_optional = [path for path in (repo / "configs/production/s1").glob("*_seed[12]_*.json")]
    old_protocol = repo / "protocols/protocol.lock.json"
    old_hardware = repo / "protocols/hardware.lock.json"
    archives = [
        (old_protocol, repo / "protocols/superseded" / f"s1-protocol-{D22_PROTOCOL_ROOT}.json"),
        (old_hardware, repo / "protocols/superseded" / f"s1-hardware-{current['hardware']['sha256']}.json"),
        (old_seed0_plan, repo / "configs/superseded/d22/s1_jobs_seed0.json"),
        (old_optional_plan, repo / "configs/superseded/d22/s1_c0_c2_seed12_queue.json"),
    ]
    archives.extend((path, repo / "configs/superseded/d22/s1" / path.name)
                    for path in [*old_seed0, *old_optional])
    for source, target in archives:
        if target.exists() and target.read_bytes() != source.read_bytes():
            raise ValueError(f"historical artifact differs; refusing to overwrite {target}")
    for source, target in archives:
        _archive(source, target)

    _write(repo / "protocols/hardware.lock.json", successor["hardware"])
    _write(repo / "protocols/protocol.lock.json", successor["protocol"])
    for name, document in seed0.items():
        _write(repo / "configs" / name, document)
    _write(repo / "configs/production/s1_jobs_seed0.json", seed0_plan)
    for name, document in optional.items():
        _write(repo / "configs" / name, document)
    _write(repo / "configs/production/s1_seed12_jobs.json", optional_plan)

    verified = validate_final_lock_set(repo / "protocols")
    seed0_result = validate_production_configs(repo, verified["protocol"])
    optional_result = validate_optional_seed_configs(repo, verified["protocol"])
    print(json.dumps({"predecessor_root": D22_PROTOCOL_ROOT,
                      "production_root": verified["protocol"]["sha256"],
                      "locks": {name: document["sha256"] for name, document in verified.items()},
                      "seed0_plan": seed0_result,
                      "optional_seed_plan": optional_result}, sort_keys=True))


if __name__ == "__main__":
    main()
