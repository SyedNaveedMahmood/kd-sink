"""Read-only verification and numerical summary of the sealed S5 reuse bundle."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CONDITIONS = ("C1", "C2", "C5", "C6")
PAIRS = (("C2", "C1"), ("C5", "C1"), ("C5", "C2"), ("C6", "C1"), ("C6", "C2"))
DENSE_STEPS = list(range(0, 10001, 100))
FULL_STEPS = [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]
MEASURES = {
    "sink_s", "clean_ce_nats", "full_jsd_nats", "nonsink_jsd_nats",
    "full_mse_probability_cells", "nonsink_mse_probability_cells",
    *(f"{op}_{name}" for op in ("delete", "relocate") for name in
      ("delta_ce_nats", "self_kl_nats", "absolute_target_logprob_change_nats",
       "prediction_flip_fraction")),
}
D24 = "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6"
D26 = "888b21509000b570c83c2574f5a6bfbc3d1ec2dc197e88d841d085bac222f182"


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    value = {}
    for key, item in pairs:
        require(key not in value, f"duplicate JSON key: {key}")
        value[key] = item
    return value


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return hasher.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)


def read_sealed(path: Path) -> tuple[dict, str]:
    document = read_json(path)
    require(set(document) == {"payload", "schema_version", "sha256"} and
            document["schema_version"] == 1 and
            document["sha256"] == hashlib.sha256(canonical(document["payload"])).hexdigest(),
            f"invalid seal: {path}")
    return document["payload"], document["sha256"]


def finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite {label}: {value}")
    return float(value)


def auc(rows: list[dict], condition: str, field: str) -> float:
    return sum((a["measures_by_condition"][condition][field] +
                b["measures_by_condition"][condition][field]) *
               (b["step"] - a["step"]) / 2 for a, b in zip(rows, rows[1:])) / 10000


def verify_report(path: Path, derived: dict) -> None:
    report = path.read_text(encoding="utf-8")
    require("## Dense64 trajectory snapshots" in report and
            "## Full300 endpoint" in report and "## Source identities" in report,
            "S5 report sections missing")
    section = report.split("## Dense64 trajectory snapshots", 1)[1].split(
        "### Dense64 time summaries", 1)[0]
    seen = {}
    for line in section.splitlines():
        if not line.startswith("| C") or line.startswith("| C |"):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        key = (cells[0], int(cells[1]))
        require(key not in seen and len(cells) == 8, f"bad trajectory table row: {line}")
        seen[key] = [float(value) for value in cells[2:]]
    snapshot_steps = (0, 100, 500, 2000, 5000, 7500, 10000)
    require(set(seen) == {(c, s) for c in CONDITIONS for s in snapshot_steps},
            "S5 trajectory snapshot coverage differs")
    dense = {row["step"]: row for row in derived["dense_rows"]}
    for (condition, step), displayed in seen.items():
        measures = dense[step]["measures_by_condition"][condition]
        actual = [measures[name] for name in ("sink_s", "clean_ce_nats", "full_jsd_nats",
                  "nonsink_jsd_nats", "delete_delta_ce_nats", "relocate_delta_ce_nats")]
        require(all(abs(a-b) <= 0.00000501 for a, b in zip(displayed, actual)),
                f"S5 report trajectory value differs: {condition}/{step}")

    def table(start: str, end: str, *, step_column: bool = False) -> dict:
        require(start in report and end in report, f"missing report table: {start}")
        section = report.split(start, 1)[1].split(end, 1)[0]
        parsed = {}
        for line in section.splitlines():
            if not line.startswith("| C") or line.startswith("| C |"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if cells[0] == "Contrast":
                continue
            key = (cells[0], int(cells[1])) if step_column else cells[0]
            require(key not in parsed, f"duplicate S5 report row: {start}/{key}")
            parsed[key] = [float(value) for value in cells[2 if step_column else 1:]]
        return parsed

    def check(actual: dict, expected: dict, label: str, tolerance: float = 0.00000501) -> None:
        require(set(actual) == set(expected), f"S5 report coverage differs: {label}")
        for key, displayed in actual.items():
            require(len(displayed) == len(expected[key]) and all(
                abs(value-source) <= tolerance for value, source in zip(displayed, expected[key])),
                f"S5 report numeric value differs: {label}/{key}")

    check(table("### Dense64 time summaries", "## Full300 endpoint"), {
        c: [derived["dense_stats"][c][group][metric] for group, metric in
            (("auc_mean", "sink_s"), ("auc_mean", "clean_ce_nats"),
             ("auc_mean", "full_jsd_nats"), ("auc_mean", "nonsink_jsd_nats"),
             ("auc_mean", "delete_delta_ce_nats"),
             ("late_8000_10000_mean", "sink_s"), ("late_8000_10000_mean", "clean_ce_nats"),
             ("late_8000_10000_mean", "full_jsd_nats"),
             ("late_8000_10000_mean", "nonsink_jsd_nats"),
             ("late_8000_10000_mean", "delete_delta_ce_nats"))] for c in CONDITIONS},
          "dense time summaries")

    full_endpoint = derived["full300_rows"][-1]["measures_by_condition"]
    check(table("| C | `S` | Clean CE | Full JSD | Non-sink JSD | Full MSE | Non-sink MSE |",
                "| C | Delete `ΔCE`"), {
        c: [full_endpoint[c][name] for name in
            ("sink_s", "clean_ce_nats", "full_jsd_nats", "nonsink_jsd_nats",
             "full_mse_probability_cells", "nonsink_mse_probability_cells")] for c in CONDITIONS},
          "Full300 endpoint structure")
    check(table("| C | Delete `ΔCE`", "### Prespecified signed condition contrasts"), {
        c: [full_endpoint[c][f"{op}_{name}"] for op in ("delete", "relocate") for name in
            ("delta_ce_nats", "self_kl_nats", "absolute_target_logprob_change_nats",
             "prediction_flip_fraction")] for c in CONDITIONS},
          "Full300 endpoint behavior", tolerance=0.000051)

    contrast_names = {f"{left} − {right}": f"{left}_minus_{right}" for left, right in PAIRS}
    actual_contrasts = table("### Prespecified signed condition contrasts", "## Full300 nine-checkpoint trajectory")
    check(actual_contrasts, {label: [
        derived["full300_rows"][-1]["comparisons"][key][name] for name in
        ("sink_s", "clean_ce_nats", "full_jsd_nats", "nonsink_jsd_nats",
         "delete_delta_ce_nats", "delete_self_kl_nats", "relocate_delta_ce_nats")]
        for label, key in contrast_names.items()}, "Full300 endpoint contrasts")

    check(table("## Full300 nine-checkpoint trajectory", "## Artifacts and limits",
                step_column=True), {
        (c, row["step"]): [row["measures_by_condition"][c][name] for name in
                            ("sink_s", "full_jsd_nats", "nonsink_jsd_nats",
                             "clean_ce_nats", "delete_delta_ce_nats", "relocate_delta_ce_nats")]
        for row in derived["full300_rows"] for c in CONDITIONS}, "Full300 trajectory")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--d26", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.bundle.resolve()
    d26, d26_sha = read_sealed(args.d26)
    require(d26_sha == D26 and set(d26["allowed_runs"]) == set(CONDITIONS), "D26 lock identity")
    audit_path = root / "STAGE08_S5_AUDIT.json"
    audit = read_json(audit_path)
    require(audit["status"] == "COMPLETE" and audit["source_row_count"] == 440 and
            audit["joined_row_count"] == 110 and audit["d26_sha256"] == D26 and
            audit["model_loaded"] is False and audit["new_inference"] is False and
            audit["source_run_directories_modified"] is False, "S5 audit incomplete")
    sidecar = (root / "STAGE08_S5_AUDIT.json.sha256").read_text(encoding="utf-8").strip().split()
    require(len(sidecar) == 2 and sidecar[0] == digest(audit_path) and
            sidecar[1] == audit_path.name, "S5 audit sidecar mismatch")
    expected_sums = {"S5_JOINED_OWT_DENSE64.json": audit["dense_join_sha256"],
                     "S5_JOINED_OWT_FULL300.json": audit["full300_join_sha256"],
                     "S5_REAGGREGATED_SOURCE_ROWS.jsonl": audit["source_rows_sha256"],
                     "S5_SOURCE_VERIFICATION.json": audit["source_verification_sha256"]}
    sums_path = root / "SHA256SUMS.txt"
    require(digest(sums_path) == audit["sha256sums_sha256"], "S5 SHA256SUMS hash mismatch")
    listed = {}
    for line in sums_path.read_text(encoding="utf-8").splitlines():
        sha, name = line.split(maxsplit=1)
        require(name not in listed, f"duplicate checksum entry: {name}")
        listed[name] = sha
    require(listed == expected_sums, "S5 SHA256SUMS entries differ from audit")
    for name, expected in listed.items():
        require(digest(root / name) == expected, f"S5 file hash mismatch: {name}")

    source = read_json(root / "S5_SOURCE_VERIFICATION.json")
    require(source["D24_sha256"] == D24 and source["D26_sha256"] == D26 and
            source["critical_invariants_sha256"] == d26["critical_invariants_sha256"] and
            source["metric_version"] == "e6a-v2-metrics-1" and
            source["joins"]["dense64"]["joined_steps"] == 101 and
            source["joins"]["full300"]["joined_steps"] == 9 and
            not source["joins"]["dense64"]["missing"] and
            not source["joins"]["full300"]["missing"] and
            source["aggregate_reaggregation"]["condition_panel_record_count"] == 440 and
            source["aggregate_reaggregation"]["s5_item_operation_records_reaggregated"] == 109968 and
            source["aggregate_reaggregation"]["source_s1_item_records_verified"] == 235936,
            "S5 source audit coverage/identity")

    rows = {}
    with (root / "S5_REAGGREGATED_SOURCE_ROWS.jsonl").open(encoding="utf-8") as stream:
        for line_no, line in enumerate(stream, 1):
            row = json.loads(line, object_pairs_hook=unique_pairs)
            key = (row["panel"], row["step"], row["condition"])
            allowed = d26["allowed_runs"][row["condition"]]
            require(key not in rows and row["study"] == "S1" and row["seed"] == 0 and
                    row["status"] == "complete" and row["condition"] in CONDITIONS and
                    row["metric_version"] == "e6a-v2-metrics-1" and
                    row["panel_sha256"] == source["panel_manifest_sha256"] and
                    row["run_id"] == allowed["run_id"] and
                    row["protocol_sha256"] == d26["allowed_protocol_roots_by_condition"][row["condition"]] and
                    row["gpu_uuid"] == allowed["gpu_uuid"] and
                    row["hardware_sha256"] == allowed["hardware_lock_sha256"] and
                    row["source_commit"] == allowed["source_commit"] and
                    row["environment_lock_sha256"] == allowed["environment_lock_sha256"] and
                    row["objective_variant"] == d26["condition_objective_variants"][row["condition"]] and
                    row["initialization_sha256"] == d26["critical_invariants"]["initialization_sha256"] and
                    row["data_sha256"] == d26["critical_invariants"]["data_sha256"] and
                    row["precision"] == "bf16" and
                    row["layer_scope"] == list(range(24)) and
                    row["comparison_invariants"] == d26["critical_invariants"] and
                    set(row["measures"]) == MEASURES and
                    all(type(v) in (int, float) and math.isfinite(v) for v in row["measures"].values()),
                    f"invalid S5 source row {line_no}")
            rows[key] = row
    require(len(rows) == 440, "S5 source row count")

    joins = {}
    for panel, name, steps, count in (("owt_dense64", "S5_JOINED_OWT_DENSE64.json", DENSE_STEPS, 64),
                                      ("owt_full300", "S5_JOINED_OWT_FULL300.json", FULL_STEPS, 300)):
        join = read_json(root / name)
        require(join["status"] == "complete" and not join["missing"] and
                join["steps"] == steps and join["seeds"] == [0] and
                join["analysis"] == "reuse_only_no_training" and
                join["s5_compatibility_amendment_sha256"] == D26 and
                join["panel_sha256"] == source["panel_manifest_sha256"] and
                len(join["joined"]) == len(steps), f"S5 join coverage: {panel}")
        for joined, step in zip(join["joined"], steps):
            require(joined["step"] == step and joined["seed"] == 0 and
                    joined["s5_compatibility_amendment_sha256"] == D26 and
                    set(joined["measures_by_condition"]) == set(CONDITIONS) and
                    set(joined["comparisons"]) == {f"{a}_minus_{b}" for a, b in PAIRS},
                    f"S5 joined identity: {panel}/{step}")
            item_ids = None
            for condition in CONDITIONS:
                row = rows[(panel, step, condition)]
                require(len(row["item_ids"]) == count and len(set(row["item_ids"])) == count and
                        joined["measures_by_condition"][condition] == row["measures"] and
                        joined["source_run_ids"][condition] == row["run_id"] and
                        joined["source_protocol_roots"][condition] == row["protocol_sha256"] and
                        joined["source_gpu_uuids"][condition] == row["gpu_uuid"] and
                        joined["source_aggregate_sha256"][condition] == row["source_aggregate_sha256"] and
                        joined["source_item_bundle_sha256"][condition] == row["source_item_bundle_sha256"],
                        f"S5 joined source mismatch: {panel}/{step}/{condition}")
                if item_ids is None:
                    item_ids = row["item_ids"]
                require(row["item_ids"] == item_ids, f"S5 paired item mismatch: {panel}/{step}/{condition}")
            require(hashlib.sha256(canonical({"item_ids": item_ids})).hexdigest() ==
                    d26["critical_invariants"]["panel_item_id_sha256"][panel],
                    f"S5 frozen panel item hash mismatch: {panel}/{step}")
            for left, right in PAIRS:
                comparison = joined["comparisons"][f"{left}_minus_{right}"]
                require(set(comparison) == MEASURES, f"S5 comparison shape: {panel}/{step}")
                for field in MEASURES:
                    actual = (joined["measures_by_condition"][left][field] -
                              joined["measures_by_condition"][right][field])
                    require(math.isclose(actual, comparison[field], rel_tol=0, abs_tol=1e-12),
                            f"S5 comparison arithmetic: {panel}/{step}/{left}-{right}/{field}")
        joins[panel] = join["joined"]

    dense = joins["owt_dense64"]
    stats = {}
    for condition in CONDITIONS:
        stats[condition] = {"auc_mean": {field: auc(dense, condition, field) for field in sorted(MEASURES)},
                            "late_8000_10000_mean": {field: statistics.mean(
                                row["measures_by_condition"][condition][field] for row in dense
                                if row["step"] >= 8000) for field in sorted(MEASURES)}}
    result = {"schema": "s5-report-derived-v1", "bundle": str(root),
              "file_sha256": {name: digest(root / name) for name in
                              (*expected_sums, "SHA256SUMS.txt", "STAGE08_S5_AUDIT.json")},
              "D24_sha256": D24, "D26_sha256": D26,
              "source_verification": source, "source_run_identities": {
                  condition: {key: value for key, value in source["source_runs"][condition].items()
                              if key != "evaluation"} for condition in CONDITIONS},
              "dense_rows": dense, "full300_rows": joins["owt_full300"], "dense_stats": stats}
    if args.report:
        verify_report(args.report, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(json.dumps({"source_rows": len(rows), "joined_dense": len(dense),
                      "joined_full300": len(joins["owt_full300"]), "output": str(args.output),
                      "output_sha256": digest(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
