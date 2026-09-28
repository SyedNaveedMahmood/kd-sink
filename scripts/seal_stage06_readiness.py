"""Seal the measured S1 locks and nine non-executing production configs."""

import argparse
import json
from pathlib import Path

from sinklab.stage06_readiness import (
    SUPERSEDED_PROTOCOL_ROOT,
    build_component_locks, build_production_configs, build_protocol_lock,
    validate_final_lock_set, validate_production_configs, write_lock_set,
    write_component_locks, write_production_configs,
)
from sinklab.runtime_provenance import (EXECUTION_CRITICAL_PATH_SET_VERSION,
                                       validate_runtime_source)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--calibration-result", type=Path, required=True)
    parser.add_argument("--restored-inventory", type=Path, required=True)
    parser.add_argument("--fingerprint-denominator-floor", type=float,
                        help="researcher-approved positive numerical reporting floor")
    parser.add_argument("--production-runtime-source-commit",
                        help="immutable tested source milestone for the production runner")
    args = parser.parse_args()
    if args.fingerprint_denominator_floor is not None:
        if args.production_runtime_source_commit is None:
            parser.error("final sealing requires --production-runtime-source-commit")
        validate_runtime_source(
            args.repo, args.production_runtime_source_commit,
            EXECUTION_CRITICAL_PATH_SET_VERSION,
            loaded_package_dir=Path(validate_runtime_source.__code__.co_filename).resolve().parent)
    components = build_component_locks(args.repo, args.calibration_result,
                                       args.restored_inventory)
    if args.fingerprint_denominator_floor is None:
        write_component_locks(args.repo / "protocols", components)
        print(json.dumps({"status": "measured_components_only_pending_approved_metric_floor",
                          "locks": {key: value["sha256"] for key, value in components.items()}},
                         sort_keys=True))
        return
    protocol = build_protocol_lock(args.repo, components,
                                   fingerprint_denominator_floor=args.fingerprint_denominator_floor,
                                   production_runtime_source_commit=args.production_runtime_source_commit)
    documents = {**components, "protocol": protocol}
    configs, plan = build_production_configs(args.repo, protocol)
    write_lock_set(args.repo / "protocols", documents,
                   supersede_root_sha256=SUPERSEDED_PROTOCOL_ROOT)
    write_production_configs(args.repo, configs, plan,
                             supersede_root_sha256=SUPERSEDED_PROTOCOL_ROOT)
    validated = validate_final_lock_set(args.repo / "protocols")
    summary = validate_production_configs(args.repo, validated["protocol"])
    print(json.dumps({"locks": {key: value["sha256"] for key, value in documents.items()},
                      "production_root_sha256": protocol["sha256"],
                      "job_plan": summary}, sort_keys=True))


if __name__ == "__main__":
    main()
