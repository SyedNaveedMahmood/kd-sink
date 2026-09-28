"""Seal the prospective D19 4080-class root after the tested source is pushed."""

import argparse
import json
from pathlib import Path

from sinklab.runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION, validate_runtime_source
from sinklab.stage06_readiness import (
    PREDECESSOR_PROTOCOL_ROOT, build_production_configs, build_protocol_lock,
    build_successor_component_locks, validate_final_lock_set,
    validate_production_configs, validate_scientific_unchanged,
    write_production_configs, write_successor_lock_set,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--production-runtime-source-commit", required=True)
    args = parser.parse_args()
    repo = args.repo.resolve()
    validate_runtime_source(repo, args.production_runtime_source_commit,
                            EXECUTION_CRITICAL_PATH_SET_VERSION,
                            loaded_package_dir=repo / "src/sinklab")
    predecessor = json.loads((repo / "protocols/protocol.lock.json").read_text(encoding="utf-8"))
    if predecessor["sha256"] != PREDECESSOR_PROTOCOL_ROOT:
        parser.error("expected exact predecessor root before sealing D19")
    components = build_successor_component_locks(repo)
    protocol = build_protocol_lock(repo, components, fingerprint_denominator_floor=1e-8,
                                   production_runtime_source_commit=args.production_runtime_source_commit)
    validate_scientific_unchanged(predecessor, protocol)
    configs, plan = build_production_configs(repo, protocol)
    write_successor_lock_set(repo / "protocols", {**components, "protocol": protocol})
    write_production_configs(repo, configs, plan,
                             supersede_root_sha256=PREDECESSOR_PROTOCOL_ROOT)
    validated = validate_final_lock_set(repo / "protocols")
    summary = validate_production_configs(repo, validated["protocol"])
    print(json.dumps({"predecessor_root": PREDECESSOR_PROTOCOL_ROOT,
                      "successor_root": protocol["sha256"],
                      "hardware_lock": components["hardware"]["sha256"],
                      "environment_lock": components["environment"]["sha256"],
                      "job_plan": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
