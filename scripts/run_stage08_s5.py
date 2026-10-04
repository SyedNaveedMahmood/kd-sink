"""Reaggregate the approved S5 seed-0 S1 records; never loads a model."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socket
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.stage08_readiness import read_json, verify_evaluation_records, verify_training_log  # noqa: E402
from sinklab.metrics import METRIC_VERSION  # noqa: E402
from sinklab.provenance import canonical_json_bytes, verify_envelope  # noqa: E402
from sinklab.s5_analysis import CONDITIONS, extract_s1_record, join_s5  # noqa: E402
from sinklab.s5_compatibility import (D26_PATH, build_critical_invariants,
                                      validate_d26)  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_new(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def write_json_new(path: Path, value: dict) -> None:
    write_new(path, canonical_json_bytes(value) + b"\n")


def _load_protocol_and_environment(condition: str, identity: dict,
                                   compatibility: dict) -> tuple[dict, dict]:
    approved = compatibility["allowed_runs"][condition]
    if (identity.get("run_id") != approved["run_id"] or identity.get("condition") != condition or
            identity.get("seed") != 0 or identity.get("study") != "S1" or
            identity.get("protocol_hash") != approved["protocol_root_sha256"] or
            identity.get("gpu_uuid") != approved["gpu_uuid"] or
            identity.get("hardware_hash") != approved["hardware_lock_sha256"] or
            identity.get("device_role") != "rtx4080super" or identity.get("precision") != "bf16" or
            identity.get("backend") != "eager"):
        raise ValueError(f"D26 checkpoint/run identity mismatch for {condition}")
    protocol_path = REPO / "protocols" / "superseded" / f"s1-protocol-{identity['protocol_hash']}.json"
    protocol_document = read_json(protocol_path)
    protocol_payload, protocol_sha = verify_envelope(protocol_document)
    if protocol_sha != identity["protocol_hash"]:
        raise ValueError(f"original protocol-root seal mismatch for {condition}")
    if (protocol_payload.get("source_commit") != approved["source_commit"] or
            protocol_payload.get("production_runtime_source_commit") != approved["production_runtime_source_commit"] or
            protocol_payload.get("environment_lock_digest") != approved["environment_lock_sha256"]):
        raise ValueError(f"D26 source/runtime/environment binding mismatch for {condition}")
    env_sha = protocol_payload["environment_lock_digest"]
    env_path = REPO / "protocols" / "superseded" / f"s1-environment-{env_sha}.json"
    if not env_path.exists():
        env_path = REPO / "protocols" / "environment.lock.json"
    environment_document = read_json(env_path)
    environment_payload, actual_sha = verify_envelope(environment_document)
    if actual_sha != env_sha:
        raise ValueError(f"historical environment-lock seal mismatch for {condition}")
    return protocol_payload, environment_payload


def run(*, runs_root: Path, artifact_root: Path, output: Path) -> dict:
    d26_document = read_json(REPO / D26_PATH)
    compatibility = validate_d26(d26_document)
    runs_root, artifact_root, output = runs_root.resolve(), artifact_root.resolve(), output.resolve()
    if output == REPO or REPO in output.parents or runs_root in output.parents or artifact_root in output.parents:
        raise ValueError("S5 output must remain outside Git and source/artifact roots")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"S5 output directory is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)

    artifact_document = read_json(REPO / "protocols" / "artifact.lock.json")
    _, artifact_sha = verify_envelope(artifact_document)
    panel_paths = list((artifact_root / "panels").glob("owt-panels-*.json"))
    if len(panel_paths) != 1:
        raise ValueError("the unique pinned S1 OWT panel manifest is required")
    panel_path = panel_paths[0]
    panel_document = read_json(panel_path)
    panel_payload, panel_sha = verify_envelope(panel_document)
    expected_invariants = compatibility["critical_invariants"]
    if panel_sha != expected_invariants["panel_manifest_sha256"]:
        raise ValueError("S1 panel manifest digest differs from sealed D26")
    if panel_payload.get("corpus_sha256") != expected_invariants["data_sha256"]:
        raise ValueError("S1 panel/corpus identity differs from sealed D26")

    training_exception = read_json(REPO / "protocols" / "s1_c3_seed0_transferred_log_exception.json")
    source_audits = {}
    run_manifests = {}
    for condition in CONDITIONS:
        approved = compatibility["allowed_runs"][condition]
        run_dir = runs_root / approved["run_id"]
        if not run_dir.is_dir():
            raise FileNotFoundError(f"approved S5 source run missing: {run_dir}")
        final_path = run_dir / "checkpoints" / "final-010000"
        checkpoint_manifest = read_json(final_path / "manifest.json")
        identity = checkpoint_manifest["identity"]
        protocol_payload, environment_payload = _load_protocol_and_environment(
            condition, identity, compatibility)
        derived = build_critical_invariants(protocol_payload=protocol_payload, identity=identity,
            environment_payload=environment_payload, artifact_sha256=artifact_sha,
            panel_sha256=panel_sha, panel_payload=panel_payload, metric_version=METRIC_VERSION)
        if derived != expected_invariants:
            raise ValueError(f"S5 comparison-critical source invariants differ from D26: {condition}")

        train_audit = verify_training_log(run_dir, identity, exception_document=training_exception)
        if train_audit["status"] != "complete" or train_audit["last_update"] != 10000:
            raise ValueError(f"incomplete/unexpected S5 source training log: {condition}")
        eval_audit = verify_evaluation_records(run_dir, identity)
        if eval_audit["status"] != "verified" or eval_audit["aggregate_count"] != 222:
            raise ValueError(f"incomplete S1 evaluation inventory for S5: {condition}")

        run_manifests[condition] = {
            "run_id": approved["run_id"], "condition": condition, "seed": 0,
            "device_role": identity["device_role"],
            "initialization_sha256": identity["init_hash"], "data_sha256": identity["data_hash"],
            "protocol_sha256": identity["protocol_hash"],
            "comparison_invariants": derived,
            "objective_variant": protocol_payload["condition_variants"][condition],
            "gpu_uuid": identity["gpu_uuid"], "hardware_sha256": identity["hardware_hash"],
            "source_commit": protocol_payload["source_commit"],
            "environment_lock_sha256": protocol_payload["environment_lock_digest"],
        }
        source_audits[condition] = {
            "source_run_path": str(run_dir),
            "run_id": identity["run_id"], "seed": identity["seed"],
            "protocol_root_sha256": identity["protocol_hash"],
            "initialization_sha256": identity["init_hash"], "data_sha256": identity["data_hash"],
            "model_sha256": identity["model_hash"], "gpu_uuid": identity["gpu_uuid"],
            "hardware_lock_sha256": identity["hardware_hash"],
            "train_jsonl_sha256": train_audit["sha256"],
            "training_update_count": train_audit["update_count"],
            "training_start_event_count": train_audit["start_event_count"],
            "final_input_tokens": train_audit["final_input_tokens"],
            "final_shifted_targets": train_audit["final_shifted_targets"],
            "evaluation": eval_audit,
        }

    dense_steps = list(range(0, 10001, 100))
    full_steps = [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]
    records_by_panel = {"owt_dense64": [], "owt_full300": []}
    seen_groups = {condition: set() for condition in CONDITIONS}
    for condition in CONDITIONS:
        approved = compatibility["allowed_runs"][condition]
        run_dir = runs_root / approved["run_id"]
        for aggregate_path in sorted((run_dir / "evaluation").glob("aggregate-*.json")):
            aggregate_document = read_json(aggregate_path)
            aggregate, _ = verify_envelope(aggregate_document)
            key = aggregate["key"]
            panel_name = key["panel"]
            if key["model_role"] != "student" or panel_name not in records_by_panel:
                continue
            allowed_steps = dense_steps if panel_name == "owt_dense64" else full_steps
            if key["step"] not in allowed_steps:
                raise ValueError(f"unexpected S5 aggregate step: {condition}/{panel_name}/{key['step']}")
            group = (key["step"], panel_name)
            if group in seen_groups[condition]:
                raise ValueError(f"duplicate S5 source aggregate: {condition}/{group}")
            seen_groups[condition].add(group)
            row = extract_s1_record(aggregate_path=aggregate_path,
                                    store_root=run_dir / "evaluation",
                                    run_manifest=run_manifests[condition])
            if row["item_ids"] != panel_payload[panel_name]:
                raise ValueError(f"frozen S1 panel item identity mismatch: {condition}/{panel_name}/{key['step']}")
            records_by_panel[panel_name].append(row)
        expected_groups = {(step, name) for name, steps in (
            ("owt_dense64", dense_steps), ("owt_full300", full_steps)) for step in steps}
        if seen_groups[condition] != expected_groups:
            raise ValueError(f"S5 aggregate coverage differs for {condition}; missing={sorted(expected_groups-seen_groups[condition])}")

    dense_join = join_s5(records_by_panel["owt_dense64"], device_role="rtx4080super",
        panel_sha256=panel_sha, steps=dense_steps, seeds=[0], compatibility_document=d26_document)
    full_join = join_s5(records_by_panel["owt_full300"], device_role="rtx4080super",
        panel_sha256=panel_sha, steps=full_steps, seeds=[0], compatibility_document=d26_document)
    if (dense_join["status"] != "complete" or len(dense_join["joined"]) != len(dense_steps) or
            full_join["status"] != "complete" or len(full_join["joined"]) != len(full_steps)):
        raise ValueError("S5 condition/step join coverage incomplete")

    source_rows_path = output / "S5_REAGGREGATED_SOURCE_ROWS.jsonl"
    with source_rows_path.open("xb") as stream:
        for row in records_by_panel["owt_dense64"] + records_by_panel["owt_full300"]:
            stream.write(canonical_json_bytes(row) + b"\n")
        stream.flush()
        os.fsync(stream.fileno())
    dense_path = output / "S5_JOINED_OWT_DENSE64.json"
    full_path = output / "S5_JOINED_OWT_FULL300.json"
    write_json_new(dense_path, dense_join)
    write_json_new(full_path, full_join)
    verification = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "hostname": socket.gethostname(), "scope": "S5 seed0 C1/C2/C5/C6 read-only reaggregation",
        "source_runs": source_audits,
        "aggregate_reaggregation": {"condition_panel_record_count": sum(map(len, records_by_panel.values())),
            "dense64_rows": len(records_by_panel["owt_dense64"]), "full300_rows": len(records_by_panel["owt_full300"]),
            "dense64_steps": dense_steps, "full300_steps": full_steps,
            "source_s1_item_records_verified": sum(a["evaluation"]["item_record_count"] for a in source_audits.values()),
            "s5_item_operation_records_reaggregated": sum(len(row["item_ids"]) * 3
                for panel_rows in records_by_panel.values() for row in panel_rows),
            "all_s1_evaluation_aggregates_per_run_verified": 222,
            "source_panel_item_identity": {p: {"item_count": len(panel_payload[p]),
                "item_ids_sha256": expected_invariants["panel_item_id_sha256"][p]} for p in records_by_panel}},
        "joins": {"dense64": {"status": dense_join["status"], "joined_steps": len(dense_join["joined"]), "missing": dense_join["missing"]},
                  "full300": {"status": full_join["status"], "joined_steps": len(full_join["joined"]), "missing": full_join["missing"]}},
        "D24_sha256": "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6",
        "D26_sha256": compatibility["sha256"], "critical_invariants_sha256": compatibility["critical_invariants_sha256"],
        "panel_manifest_sha256": panel_sha, "metric_version": METRIC_VERSION,
        "hardware_variation": compatibility["hardware_variation"],
        "execution_boundary": {"model_loaded": False, "new_inference": False, "training": False, "S2": False, "Stage09": False},
    }
    verification_path = output / "S5_SOURCE_VERIFICATION.json"
    write_json_new(verification_path, verification)

    checksum_lines = []
    for path in sorted(p for p in output.rglob("*") if p.is_file()):
        checksum_lines.append(f"{sha256_file(path)}  {path.relative_to(output).as_posix()}")
    checksum_path = output / "SHA256SUMS.txt"
    write_new(checksum_path, ("\n".join(checksum_lines) + "\n").encode("ascii"))
    audit = {"schema_version": 1, "status": "COMPLETE", "scope": verification["scope"],
        "created_utc": verification["created_utc"], "hostname": verification["hostname"],
        "d26_sha256": compatibility["sha256"], "source_rows_sha256": sha256_file(source_rows_path),
        "dense_join_sha256": sha256_file(dense_path), "full300_join_sha256": sha256_file(full_path),
        "source_verification_sha256": sha256_file(verification_path),
        "sha256sums_sha256": sha256_file(checksum_path),
        "source_row_count": len(records_by_panel["owt_dense64"]) + len(records_by_panel["owt_full300"]),
        "joined_row_count": len(dense_join["joined"]) + len(full_join["joined"]),
        "source_run_directories_modified": False, "model_loaded": False, "new_inference": False}
    audit_path = output / "STAGE08_S5_AUDIT.json"
    write_json_new(audit_path, audit)
    write_new(output / "STAGE08_S5_AUDIT.json.sha256",
              f"{sha256_file(audit_path)}  STAGE08_S5_AUDIT.json\n".encode("ascii"))
    return audit


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", type=Path, default=Path(r"D:\KD-SINK-central\runs"))
    parser.add_argument("--artifact-root", type=Path,
        default=Path(r"E:\KD-SINK-stage06-Adrita\artifacts\KD-SINK-stage06-production"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = run(runs_root=args.runs_root, artifact_root=args.artifact_root, output=args.output)
    print(json.dumps(audit, indent=2, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
