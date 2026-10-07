"""Read-only S7 report extraction from the sealed analysis and 36 supplements.

The original S1/S5 source trees and checkpoint payloads are not opened here;
their full validation is recorded by the preserved independent campaign audit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sinklab.provenance import _no_duplicate_keys, verify_envelope


CONDITIONS = ("C1", "C2", "C5", "C6")
STEPS = (0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000)
CONTRASTS = (("C5", "C2"), ("C6", "C1"), ("C2", "C1"), ("C5", "C1"), ("C6", "C2"))
DECOMPOSITION = ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats",
                 "key0_jsd_nats", "other_columns_jsd_nats", "conditional_jsd_nats")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def require(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def close(actual: float, expected: float, label: str, tolerance: float = 1e-10) -> None:
    if not math.isfinite(actual) or abs(actual - expected) > tolerance:
        raise ValueError(f"{label}: {actual!r} != {expected!r} (tol {tolerance})")


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    rank = q * (len(values) - 1)
    lo = math.floor(rank)
    hi = math.ceil(rank)
    return values[lo] + (values[hi] - values[lo]) * (rank - lo)


def extract(root: Path, lock_path: Path) -> dict:
    lock, _ = verify_envelope(read_json(lock_path))
    lock_sha = read_json(lock_path)["sha256"]
    audit_path = root / "audit" / "independent_final_audit.json"
    analysis_path = root / "analysis" / "final_d40afe0" / "S7_ANALYSIS.json"
    runner_path = root / "analysis" / "final_d40afe0" / "S7_AUDIT.json"
    audit, analysis, runner = (read_json(path) for path in (audit_path, analysis_path, runner_path))
    require(audit["status"], "PASS", "independent audit status")
    require(audit["approved_lock_sha256"], lock_sha, "audit lock")
    require(audit["D24_sha256"], lock["D24_sha256"], "D24")
    require(audit["D26_sha256"], lock["D26_sha256"], "D26")
    require(audit["S5_audit_sha256"], lock["S5_audit_sha256"], "S5 source audit")
    require(audit["final_analysis_sha256"], sha256(analysis_path), "analysis file SHA")
    require(audit["final_runner_audit_sha256"], sha256(runner_path), "runner audit file SHA")
    require(audit_path.with_name(audit_path.name + ".sha256").read_text().split()[0], sha256(audit_path), "audit sidecar")
    require(audit["verified_state_count"], 36, "audited states")
    require(audit["conditions"], list(CONDITIONS), "audited conditions")
    require(audit["steps"], list(STEPS), "audited steps")
    require(audit["undefined_conditional_layer_count"], 0, "undefined conditional coverage")
    require(analysis["status"], "descriptive_s5_plus_clean_supplements", "analysis status")
    require(analysis["clean_decomposition_analysis_lock_sha256"], lock_sha, "analysis lock")
    require(analysis["s7_campaign"]["verified_supplement_count"], 36, "analysis supplements")
    require(runner["science_complete"], True, "runner completion")
    require(runner["authorized_supplement_count"], 36, "runner supplement count")
    require(runner["s7_analysis_sha256"], sha256(analysis_path), "runner analysis SHA")
    require(len(analysis["extended_source_aggregates"]), 440, "reused S5 source rows")
    require(len(analysis["dense64_trajectory"]), 101, "Dense64 steps")
    require(len(analysis["retained_full300_decomposition"]), 9, "Full300 steps")
    require([r["step"] for r in analysis["retained_full300_decomposition"]], list(STEPS), "decomposition step grid")

    # The independent auditor copied this one field from an *earlier checkpoint
    # preflight*. The actual campaign source is established by the later frozen
    # source/worktree receipts and terminal monitor event, not by that label.
    preflight = read_json(root / "qualification" / "s7_checkpoint_preflight.json")
    require(audit["execution_source_commit"], preflight["execution_source_commit"], "audit's preflight-source field")
    worktree = read_json(root / "campaign" / "s7_execution_worktree_receipt.json")
    source = read_json(root / "campaign" / f"s7_execution_source_receipt_{worktree['execution_source_commit']}.json")
    terminal = [read_json_line for read_json_line in
                (json.loads(line) for line in (root / "monitoring" / "S7_MONITOR.jsonl").read_text(encoding="utf-8").splitlines())
                if read_json_line.get("event") == "s7_campaign_terminal"]
    require(len(terminal), 1, "terminal monitor events")
    require(worktree["status"], "PASS", "worktree receipt")
    require(worktree["execution_source_commit"], worktree["git_head"], "frozen worktree HEAD")
    require(worktree["execution_source_commit"], source["execution_source_commit"], "frozen source receipt")
    require(terminal[0]["execution_source_commit"], source["execution_source_commit"], "terminal source commit")
    require(terminal[0]["completed_states"], 36, "terminal state count")
    require(audit["inference_gpu_uuid"], lock["device_uuid"], "inference GPU")

    expected_ids = analysis["provenance"]["full300_item_ids"]
    require(len(expected_ids), 300, "ordered Full300 IDs")
    require(len(set(expected_ids)), 300, "unique Full300 IDs")
    by_analysis = {row["step"]: row["condition_values"] for row in analysis["retained_full300_decomposition"]}
    seen = set()
    rows, distributions, hashes = [], [], {}
    for receipt in audit["per_state"]:
        condition, step = receipt["condition"], receipt["step"]
        if condition not in CONDITIONS or step not in STEPS or (condition, step) in seen:
            raise ValueError("invalid or duplicate audited condition/step")
        seen.add((condition, step))
        path = Path(receipt["path"])
        path.relative_to(root)
        file_sha = sha256(path)
        require(file_sha, receipt["sha256"], f"{condition}/{step} supplement SHA")
        require(file_sha, audit["all_supplement_hashes"][path.name], f"{condition}/{step} audit inventory")
        require(file_sha, analysis["provenance"]["supplement_file_sha256"][path.name], f"{condition}/{step} analysis inventory")
        supplement = read_json(path)
        source_run = lock["source_runs"][condition]
        for field, expected in (("study", "S7"), ("condition", condition), ("step", step),
                                ("training_seed", 0), ("analysis_lock_sha256", lock_sha),
                                ("D24_sha256", lock["D24_sha256"]), ("s5_source_audit_sha256", lock["S5_audit_sha256"]),
                                ("run_id", source_run["run_id"]),
                                ("original_protocol_root_sha256", source_run["protocol_root_sha256"]),
                                ("panel", "owt_full300"), ("status", "complete")):
            require(supplement[field], expected, f"{condition}/{step} {field}")
        require(supplement["inference_gpu"]["uuid"], lock["device_uuid"], f"{condition}/{step} GPU")
        require(supplement["result"]["precision"], "fp32", f"{condition}/{step} precision")
        require(supplement["panel_receipt"]["panel_item_count"], 300, f"{condition}/{step} panel count")
        require(supplement["loaded_student_tensor_digest"], receipt["loaded_student_tensor_digest"], f"{condition}/{step} tensor digest")
        items = supplement["result"]["items"]
        require([item["item_id"] for item in items], expected_ids, f"{condition}/{step} item order")
        require(len(items), 300, f"{condition}/{step} item count")
        component_values = {field: [] for field in DECOMPOSITION}
        item_means = {field: [] for field in DECOMPOSITION}
        for item in items:
            layers = item["layers"]
            require(len(layers), 24, f"{condition}/{step} layer count")
            for field in DECOMPOSITION:
                values = [layer[field] for layer in layers]
                if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in values):
                    raise ValueError(f"{condition}/{step} nonfinite {field}")
                component_values[field].extend(values)
                item_means[field].append(sum(values) / 24)
            for layer in layers:
                close(layer["full_jsd_nats"], layer["mass_jsd_nats"] + layer["shape_jsd_nats"], f"{condition}/{step} mass+shape", 1e-9)
                close(layer["full_jsd_nats"], layer["key0_jsd_nats"] + layer["other_columns_jsd_nats"], f"{condition}/{step} key0+other", 1e-9)
                require(layer["valid_query_count"], 127, f"{condition}/{step} query count")
        aggregate = supplement["result"]["aggregate"]
        require(aggregate, by_analysis[step][condition], f"{condition}/{step} final analysis aggregate")
        for field in DECOMPOSITION:
            close(sum(component_values[field]) / 7200, aggregate[field], f"{condition}/{step} {field} reduction", 1e-9)
        close(aggregate["full_jsd_nats"], aggregate["mass_jsd_nats"] + aggregate["shape_jsd_nats"], f"{condition}/{step} aggregate mass+shape")
        close(aggregate["full_jsd_nats"], aggregate["key0_jsd_nats"] + aggregate["other_columns_jsd_nats"], f"{condition}/{step} aggregate key0+other")
        rows.append({"condition": condition, "step": step, **{field: aggregate[field] for field in DECOMPOSITION}})
        if step == 10000:
            distributions.append({"condition": condition, **{f"{field}_{suffix}": statistic(item_means[field])
                for field in ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats")
                for suffix, statistic in (("median", statistics.median), ("p90", lambda x: percentile(x, .9)))}})
        hashes[path.name] = file_sha
    require(seen, {(c, s) for c in CONDITIONS for s in STEPS}, "36-state grid")
    require(len(hashes), 36, "unique supplement files")
    rows.sort(key=lambda row: (CONDITIONS.index(row["condition"]), STEPS.index(row["step"])))
    distributions.sort(key=lambda row: CONDITIONS.index(row["condition"]))

    endpoint = {}
    for condition in CONDITIONS:
        endpoint[condition] = {"s5_full300": analysis["full300_endpoint"]["condition_values"][condition],
                               "s5_extended_full300": analysis["extended_full300_endpoint"]["condition_values"][condition],
                               "s7_decomposition": by_analysis[10000][condition]}
    contrast_rows = []
    for first, second in CONTRASTS:
        label = f"{first}_minus_{second}"
        recorded = analysis["retained_full300_decomposition"][-1]["paired_contrasts"][label]
        record = {"contrast": label}
        for field in ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats", "key0_jsd_nats", "other_columns_jsd_nats"):
            difference = endpoint[first]["s7_decomposition"][field] - endpoint[second]["s7_decomposition"][field]
            close(recorded[field], difference, f"{label} {field}")
            record[field] = difference
        for field, group in (("clean_ce_nats", "s5_full300"), ("delete_delta_ce_nats", "s5_full300"),
                             ("relocate_delta_ce_nats", "s5_full300"),
                             ("teacher_kl_nats", "s5_extended_full300")):
            record[field] = endpoint[first][group][field] - endpoint[second][group][field]
        contrast_rows.append(record)
    temporal = []
    dense = analysis["dense64_trajectory"]
    extended_dense = analysis["extended_dense64_trajectory"]
    require([row["step"] for row in dense], list(range(0, 10001, 100)), "Dense64 grid")
    require([row["step"] for row in extended_dense], list(range(0, 10001, 100)), "extended Dense64 grid")
    for condition in CONDITIONS:
        row = {"condition": condition}
        for field, source_rows, temporal_source in (
            ("clean_ce_nats", dense, analysis["dense64_temporal"]),
            ("sink_s", dense, analysis["dense64_temporal"]),
            ("delete_delta_ce_nats", dense, analysis["dense64_temporal"]),
            ("teacher_kl_nats", extended_dense, analysis["extended_dense64_temporal"])):
            values = [point["condition_values"][condition][field] for point in source_rows]
            auc = sum((values[i] + values[i + 1]) / 2 for i in range(100)) / 100
            late = sum(values[80:]) / 21
            row[field] = {"auc_mean": auc, "late_8000_to_10000_mean": late, "endpoint": values[-1]}
            for other in CONDITIONS:
                if other == condition or f"{condition}_minus_{other}" not in temporal_source[field]:
                    continue
                expected = temporal_source[field][f"{condition}_minus_{other}"]
                other_values = [point["condition_values"][other][field] for point in source_rows]
                other_auc = sum((other_values[i] + other_values[i + 1]) / 2 for i in range(100)) / 100
                close(auc - other_auc, expected["auc_mean_difference"], f"Dense AUC {field} {condition}-{other}")
                close(values[-1] - other_values[-1], expected["endpoint_dense64_difference"], f"Dense endpoint {field} {condition}-{other}")
        temporal.append(row)
    lm = analysis["lm2000_endpoints"]
    require([row["step"] for row in lm], [0, 10000], "LM2000 endpoint steps")
    return {"schema": "s7-report-derived-v1", "root": str(root),
            "analysis_sha256": sha256(analysis_path), "runner_audit_sha256": sha256(runner_path),
            "independent_audit_sha256": sha256(audit_path), "lock_sha256": lock_sha,
            "D24_sha256": lock["D24_sha256"], "D26_sha256": lock["D26_sha256"],
            "S5_audit_sha256": lock["S5_audit_sha256"], "frozen_execution_source_commit": source["execution_source_commit"],
            "checkpoint_preflight_source_commit": preflight["execution_source_commit"],
            "independent_audit_execution_source_field_copies_preflight_commit": True,
            "analysis_source_commit": runner["analysis_source_commit"],
            "inference_gpu_uuid": lock["device_uuid"], "run_sources": lock["source_runs"],
            "supplement_file_sha256": hashes, "decomposition_rows": rows,
            "endpoint": endpoint, "contrast_rows": contrast_rows,
            "endpoint_item_distribution": distributions, "dense64_temporal": temporal,
            "lm2000_endpoints": lm, "unavailable": analysis["unavailable"],
            "independent_audit_counts": {key: audit[key] for key in
                ("verified_state_count", "checkpoint_sets_rehashed", "checkpoint_payload_files_rehashed",
                 "source_bytes_rehashed", "s5_source_row_count", "maximum_decomposition_closure_error_nats")}}


def report_tables(data: dict) -> dict[str, str]:
    endpoint = ["| Condition | `S` | Clean CE | Teacher KL | Teacher top-1 agreement | Delete `ΔCE` | Relocate `ΔCE` | S7 full JSD | Mass JSD | Shape JSD | Conditional JSD |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        e = data["endpoint"][c]
        s5, ext, s7 = e["s5_full300"], e["s5_extended_full300"], e["s7_decomposition"]
        endpoint.append(f"| {c} | {s5['sink_s']:.5f} | {s5['clean_ce_nats']:.5f} | {ext['teacher_kl_nats']:.5f} | "
                        f"{ext['teacher_top1_agreement_fraction']:.5f} | {s5['delete_delta_ce_nats']:.5f} | "
                        f"{s5['relocate_delta_ce_nats']:.5f} | {s7['full_jsd_nats']:.5f} | {s7['mass_jsd_nats']:.5f} | "
                        f"{s7['shape_jsd_nats']:.5f} | {s7['conditional_jsd_nats']:.5f} |")
    trajectory = ["| Condition | Step | Full JSD | Sink/rest mass JSD | Weighted shape JSD | Conditional JSD |",
                  "|---|---:|---:|---:|---:|---:|"]
    for row in data["decomposition_rows"]:
        trajectory.append(f"| {row['condition']} | {row['step']:,} | {row['full_jsd_nats']:.5f} | "
                          f"{row['mass_jsd_nats']:.5f} | {row['shape_jsd_nats']:.5f} | {row['conditional_jsd_nats']:.5f} |")
    contrasts = ["| Contrast (A − B) | Full JSD | Mass JSD | Shape JSD | Clean CE | Teacher KL | Delete `ΔCE` | Relocate `ΔCE` |",
                 "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in data["contrast_rows"]:
        contrasts.append(f"| {row['contrast'].replace('_minus_', ' − ')} | {row['full_jsd_nats']:+.5f} | "
                         f"{row['mass_jsd_nats']:+.5f} | {row['shape_jsd_nats']:+.5f} | "
                         f"{row['clean_ce_nats']:+.5f} | {row['teacher_kl_nats']:+.5f} | "
                         f"{row['delete_delta_ce_nats']:+.5f} | {row['relocate_delta_ce_nats']:+.5f} |")
    temporal = ["| Condition | Clean CE AUC / late mean | `S` AUC / late mean | Delete `ΔCE` AUC / late mean | Teacher KL AUC / late mean |",
                "|---|---:|---:|---:|---:|"]
    for row in data["dense64_temporal"]:
        def pair(field: str) -> str:
            return f"{row[field]['auc_mean']:.5f} / {row[field]['late_8000_to_10000_mean']:.5f}"
        temporal.append(f"| {row['condition']} | {pair('clean_ce_nats')} | {pair('sink_s')} | "
                        f"{pair('delete_delta_ce_nats')} | {pair('teacher_kl_nats')} |")
    lm = ["| Condition | Step-0 LM2000 clean CE | Step-10,000 clean CE | Step-10,000 PPL | Step-10,000 next-token accuracy |",
          "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        start = data["lm2000_endpoints"][0]["condition_values"][c]
        finish = data["lm2000_endpoints"][1]["condition_values"][c]
        lm.append(f"| {c} | {start['clean_ce_nats']:.5f} | {finish['clean_ce_nats']:.5f} | "
                  f"{finish['clean_ppl']:.3f} | {finish['clean_accuracy_fraction']:.5f} |")
    distribution = ["| Condition | Per-item full JSD median / p90 | Per-item mass JSD median / p90 | Per-item shape JSD median / p90 |",
                    "|---|---:|---:|---:|"]
    for row in data["endpoint_item_distribution"]:
        def pair(field: str) -> str:
            return f"{row[field+'_median']:.5f} / {row[field+'_p90']:.5f}"
        distribution.append(f"| {row['condition']} | {pair('full_jsd_nats')} | {pair('mass_jsd_nats')} | {pair('shape_jsd_nats')} |")
    columns = ["| Condition | Key-0 column JSD | Other-column JSD | Sink/rest mass JSD | Weighted shape JSD |",
               "|---|---:|---:|---:|---:|"]
    for c in CONDITIONS:
        row = data["endpoint"][c]["s7_decomposition"]
        columns.append(f"| {c} | {row['key0_jsd_nats']:.5f} | {row['other_columns_jsd_nats']:.5f} | "
                       f"{row['mass_jsd_nats']:.5f} | {row['shape_jsd_nats']:.5f} |")
    return {"{{S7_CENTRAL_TABLE}}": "\n".join(endpoint),
            "{{S7_TRAJECTORY_TABLE}}": "\n".join(trajectory),
            "{{S7_CONTRAST_TABLE}}": "\n".join(contrasts),
            "{{S7_TEMPORAL_TABLE}}": "\n".join(temporal),
            "{{S7_LM_TABLE}}": "\n".join(lm),
            "{{S7_DISTRIBUTION_TABLE}}": "\n".join(distribution),
            "{{S7_COLUMNS_TABLE}}": "\n".join(columns)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--report", type=Path)
    args = p.parse_args()
    data = extract(args.root, args.lock)
    encoded = json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output.exists():
        require(args.output.read_text(encoding="utf-8"), encoded, "prior derived inventory")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    if args.report:
        report = args.report.read_text(encoding="utf-8")
        updated = report
        for placeholder, table in report_tables(data).items():
            if placeholder in updated:
                updated = updated.replace(placeholder, table)
            elif table not in updated:
                raise ValueError(f"report table mismatch: {placeholder}")
        if updated != report:
            args.report.write_text(updated, encoding="utf-8")
    print(json.dumps({"states": len(data["decomposition_rows"]), "output_sha256": sha256(args.output),
                      "report_sha256": sha256(args.report) if args.report else None}, sort_keys=True))


if __name__ == "__main__":
    main()
