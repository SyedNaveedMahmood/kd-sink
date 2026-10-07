"""Verify sealed S4 summaries and derive read-only report tables.

The independent Stage08 audit is the authority for the complete item tree. This
script verifies every summary envelope, identity, and probe aggregate, then
writes a compact numerical inventory outside Git. It never runs model inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


CONDITIONS = tuple(f"C{i}" for i in range(7))
STEPS = (0, 100, 500, 2000, 10000)
PROBES = {
    "epe_transport_layer0", "k_top3_all", "position0_to1", "q_bias_all",
    "remove_absolute_positions", *(f"k_random{i}_all" for i in range(5)),
}
D24 = "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6"
GPU = "GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf"


def require(test: bool, message: str) -> None:
    if not test:
        raise ValueError(message)


def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sealed(path: Path) -> dict:
    document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    require(set(document) == {"payload", "schema_version", "sha256"} and
            document["schema_version"] == 1 and
            document["sha256"] == hashlib.sha256(canonical(document["payload"])).hexdigest(),
            f"invalid seal: {path}")
    return document["payload"]


def finite(value: object, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite {label}: {value}")
    return float(value)


def summarize_probe(probe_id: str, record: dict, label: str) -> dict:
    require(record["status"] == "complete" and record["complete_item_count"] == 300 and
            not record["failed_item_ids"], f"incomplete {label}/{probe_id}")
    behavior = record["behavior"]
    require(behavior["item_count"] == 300 and behavior["valid_targets"] == 38100 and
            behavior["schema_version"] == "e6a-v2-metrics-1", f"behavior coverage: {label}/{probe_id}")
    fp = record["fingerprint"]
    baseline = finite(fp["baseline_sink"], f"{label}/{probe_id}/baseline")
    probed = finite(fp["probed_sink"], f"{label}/{probe_id}/probed")
    delta = finite(fp["absolute_delta_sink"], f"{label}/{probe_id}/delta")
    require(math.isclose(probed - baseline, delta, rel_tol=0, abs_tol=1e-10),
            f"sink delta disagreement: {label}/{probe_id}")
    clean = finite(behavior["clean_ce_nats"], f"{label}/{probe_id}/clean")
    edited = finite(behavior["edited_ce_nats"], f"{label}/{probe_id}/edited")
    dce = finite(behavior["delta_ce_nats"], f"{label}/{probe_id}/delta CE")
    require(math.isclose(edited - clean, dce, rel_tol=0, abs_tol=1e-8),
            f"CE delta disagreement: {label}/{probe_id}")
    result = {"baseline_sink": baseline, "edited_sink": probed, "delta_sink": delta,
              "clean_ce_nats": clean, "edited_ce_nats": edited, "delta_ce_nats": dce,
              "self_kl_nats": finite(behavior["self_kl_nats"], f"{label}/{probe_id}/KL"),
              "absolute_target_logprob_change_nats": finite(
                  behavior["absolute_target_logprob_change_nats"], f"{label}/{probe_id}/abs"),
              "median_item_absolute_change_nats": finite(
                  behavior["median_item_absolute_change_nats"], f"{label}/{probe_id}/median"),
              "p90_item_absolute_change_nats": finite(
                  behavior["p90_item_absolute_change_nats"], f"{label}/{probe_id}/p90"),
              "flip_fraction": finite(behavior["prediction_flip_fraction"], f"{label}/{probe_id}/flip"),
              "clean_accuracy_fraction": finite(behavior["clean_accuracy_fraction"], f"{label}/{probe_id}/accuracy"),
              "edited_accuracy_fraction": finite(behavior["edited_accuracy_fraction"], f"{label}/{probe_id}/accuracy"),
              "responsiveness": record["responsiveness"], "probe_plan": record["probe"]}
    require(0 <= baseline <= 1 and 0 <= probed <= 1 and 0 <= result["flip_fraction"] <= 1 and
            0 <= result["clean_accuracy_fraction"] <= 1 and 0 <= result["edited_accuracy_fraction"] <= 1 and
            result["self_kl_nats"] >= -1e-8, f"metric range: {label}/{probe_id}")
    return result


def verify_report(path: Path, rows: list[dict], teacher_probes: dict) -> None:
    """Check the manually edited trajectory table against unrounded source values."""
    report = path.read_text(encoding="utf-8")
    require("## Complete measured checkpoint trajectory" in report and
            "## Endpoint comparison at step 10,000" in report, "S4 report sections missing")
    section = report.split("## Complete measured checkpoint trajectory", 1)[1].split(
        "## Endpoint comparison at step 10,000", 1)[0]
    found = {}
    for line in section.splitlines():
        if not line.startswith("| C") or line.startswith("| C |"):
            continue
        fields = [part.strip() for part in line.strip("|").split("|")]
        require(len(fields) == 8, f"malformed trajectory row: {line}")
        key = (fields[0], int(fields[1]))
        require(key not in found, f"duplicate report row: {key}")
        found[key] = [float(value) for value in fields[2:]]
    expected = {(row["condition"], row["step"]): [
        row["baseline_sink"], row["clean_ce_nats"],
        row["probes"]["position0_to1"]["delta_sink"],
        row["probes"]["position0_to1"]["delta_ce_nats"],
        row["probes"]["k_top3_all"]["delta_ce_nats"],
        row["random_mean_delta_ce_nats"]] for row in rows}
    require(set(found) == set(expected), "report trajectory coverage differs from summaries")
    for key, values in found.items():
        for index, (display, actual) in enumerate(zip(values, expected[key])):
            require(abs(display - actual) <= 0.00000501,
                    f"report trajectory value differs: {key}, column {index}, {display} vs {actual}")

    def check_endpoint(start: str, end: str, values_for_row) -> None:
        require(start in report and end in report, f"report section missing: {start}")
        text = report.split(start, 1)[1].split(end, 1)[0]
        seen = {}
        for line in text.splitlines():
            if line.startswith("| C") and not line.startswith("| C |"):
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                require(cells[0] not in seen, f"duplicate endpoint row: {cells[0]}")
                seen[cells[0]] = [float(value) for value in cells[1:]]
        require(set(seen) == set(CONDITIONS), f"endpoint coverage differs: {start}")
        for row in rows:
            if row["step"] != 10000:
                continue
            actual = values_for_row(row)
            displayed = seen[row["condition"]]
            require(len(actual) == len(displayed), f"endpoint column count: {start}")
            for index, (value, source) in enumerate(zip(displayed, actual)):
                tolerance = 0.000051  # endpoint tables contain four-decimal fractions
                require(abs(value - source) <= tolerance,
                        f"endpoint value differs: {start}, {row['condition']}, column {index}")

    check_endpoint("### Anchor and broad positional probes", "### Parameter routes and matched controls",
                   lambda r: [r["baseline_sink"], r["probes"]["position0_to1"]["delta_sink"],
                              r["probes"]["position0_to1"]["delta_ce_nats"],
                              r["probes"]["position0_to1"]["flip_fraction"],
                              r["probes"]["remove_absolute_positions"]["delta_ce_nats"],
                              r["probes"]["remove_absolute_positions"]["flip_fraction"]])
    check_endpoint("### Parameter routes and matched controls", "### Other behavioral effect sizes",
                   lambda r: [r["probes"]["k_top3_all"]["delta_sink"],
                              r["probes"]["k_top3_all"]["delta_ce_nats"],
                              r["probes"]["k_top3_all"]["flip_fraction"],
                              r["random_mean_delta_ce_nats"],
                              r["probes"]["q_bias_all"]["delta_ce_nats"],
                              r["probes"]["epe_transport_layer0"]["delta_ce_nats"]])
    check_endpoint("### Other behavioral effect sizes", "## Frozen teacher reference",
                   lambda r: [r["probes"]["position0_to1"]["self_kl_nats"],
                              r["probes"]["position0_to1"]["absolute_target_logprob_change_nats"],
                              r["probes"]["position0_to1"]["median_item_absolute_change_nats"],
                              r["probes"]["position0_to1"]["p90_item_absolute_change_nats"],
                              r["probes"]["k_top3_all"]["self_kl_nats"],
                              r["probes"]["k_top3_all"]["absolute_target_logprob_change_nats"]])

    labels = {"Position 0→1": "position0_to1", "All-PE removal": "remove_absolute_positions",
              "Q-bias removal": "q_bias_all", "EPE transport, layer 0": "epe_transport_layer0",
              "K top3": "k_top3_all",
              **{f"K random control {i}": f"k_random{i}_all" for i in range(5)}}
    section = report.split("## Frozen teacher reference", 1)[1].split(
        "## Artifacts, execution history, and limits", 1)[0]
    seen = set()
    for line in section.splitlines():
        if not line.startswith("| "):
            continue
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if cells[0] not in labels:
            continue
        probe_id = labels[cells[0]]
        require(probe_id not in seen and len(cells) == 4, "teacher table duplicate/shape")
        seen.add(probe_id)
        expected_values = [teacher_probes[probe_id][name] for name in
                           ("delta_sink", "delta_ce_nats", "flip_fraction")]
        require(all(abs(float(value) - source) <= (0.000051 if i == 2 else 0.00000501)
                    for i, (value, source) in enumerate(zip(cells[1:], expected_values))),
                f"teacher table value differs: {probe_id}")
    require(seen == PROBES, "teacher table coverage differs")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--independent-audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    audit = read_sealed(args.independent_audit)
    checks = audit["independent_checks"]
    require(audit["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED" and audit["study"] == "S4" and
            checks["student_batteries"] == 35 and checks["student_records"] == 105000 and
            checks["teacher_records"] == 3000 and checks["manifested_file_count"] == 108037 and
            checks["sha256s_entries_verified"] == 108037 and
            checks["record_envelopes_and_file_hashes"] == "PASS" and
            checks["unique_record_coverage"] == "PASS" and
            checks["run_and_checkpoint_identities"] == "PASS" and
            checks["panel_file_and_item_identity"] == "PASS", "independent audit incomplete")
    require(Path(audit["result_directory"]).resolve() == root, "audit points to another result tree")

    summary_dir = root / "summaries"
    expected = {summary_dir / "teacher_full300.json"}
    expected.update(summary_dir / c / f"step-{s:05d}.json" for c in CONDITIONS for s in STEPS)
    actual = set(summary_dir.rglob("*.json"))
    require(actual == expected, f"summary coverage differs: missing={expected-actual}, extra={actual-expected}")
    rows = []
    common_panel = None
    for condition in CONDITIONS:
        for step in STEPS:
            path = summary_dir / condition / f"step-{step:05d}.json"
            payload = read_sealed(path)
            label = f"{condition}/{step}"
            require(payload["kind"] == "s4-probe-battery-summary-v1" and
                    payload["condition"] == condition and payload["step"] == step and
                    payload["training_seed"] == 0 and payload["model_role"] == "student" and
                    payload["precision"] == "fp32" and payload["D24_sha256"] == D24 and
                    payload["run_id"].startswith(f"s1-{condition.lower()}-seed0-") and
                    payload["original_protocol_root_sha256"] ==
                    payload["result"]["source_identity"]["protocol_hash"],
                    f"summary identity: {label}")
            result = payload["result"]
            require(result["status"] == "complete" and result["probe_version"] == "s4-gpt2-route-v1" and
                    result["step"] == step and result["model_role"] == "student" and
                    result["panel_sha256"] == payload["panel_sha256"] and
                    result["provenance"]["inference_device"]["uuid"] == GPU and
                    result["provenance"]["source_checkpoint_manifest_sha256"] ==
                    payload["checkpoint_manifest_sha256"] and
                    result["followup_policy"]["amendment_sha256"] == D24 and
                    result["source_identity"]["run_id"] == payload["run_id"] and
                    result["source_identity"]["seed"] == 0 and
                    set(result["probes"]) == PROBES, f"result identity/coverage: {label}")
            if common_panel is None:
                common_panel = payload["panel_sha256"]
            require(payload["panel_sha256"] == common_panel, f"panel drift: {label}")
            probes = {key: summarize_probe(key, value, label) for key, value in result["probes"].items()}
            baselines = [value["baseline_sink"] for value in probes.values()]
            cleans = [value["clean_ce_nats"] for value in probes.values()]
            require(max(baselines)-min(baselines) < 1e-9 and max(cleans)-min(cleans) < 1e-8,
                    f"probe baseline drift: {label}")
            randoms = [probes[f"k_random{i}_all"] for i in range(5)]
            rows.append({"condition": condition, "step": step, "summary_path": str(path),
                         "summary_file_sha256": sha256(path), "run_id": payload["run_id"],
                         "source_identity": result["source_identity"],
                         "protocol_root": payload["original_protocol_root_sha256"],
                         "checkpoint_manifest_sha256": payload["checkpoint_manifest_sha256"],
                         "checkpoint_model_sha256": payload["checkpoint_model_sha256"],
                         "baseline_sink": baselines[0], "clean_ce_nats": cleans[0],
                         "probes": probes,
                         "random_mean_delta_ce_nats": statistics.mean(x["delta_ce_nats"] for x in randoms),
                         "random_mean_abs_delta_ce_nats": statistics.mean(abs(x["delta_ce_nats"]) for x in randoms),
                         "random_mean_flip_fraction": statistics.mean(x["flip_fraction"] for x in randoms)})
    teacher_path = summary_dir / "teacher_full300.json"
    teacher = read_sealed(teacher_path)
    tr = teacher["result"]
    require(teacher["kind"] == "s4-probe-battery-summary-v1" and
            teacher["model_role"] == "teacher" and teacher["precision"] == "fp32" and
            teacher["panel_sha256"] == common_panel and tr["status"] == "complete" and
            tr["probe_version"] == "s4-gpt2-route-v1" and set(tr["probes"]) == PROBES and
            tr["provenance"]["device"]["uuid"] == GPU and
            tr["provenance"]["D24_sha256"] == D24, "teacher identity/coverage")
    teacher_probes = {key: summarize_probe(key, value, "teacher") for key, value in tr["probes"].items()}
    result = {"schema": "s4-report-derived-v1", "source_root": str(root),
              "independent_audit_path": str(args.independent_audit.resolve()),
              "independent_audit_file_sha256": sha256(args.independent_audit),
              "independent_audit_seal_sha256": hashlib.sha256(canonical(audit)).hexdigest(),
              "runner_audit_file_sha256": sha256(root / "S4_FINAL_AUDIT.json"),
              "D24_sha256": D24, "inference_gpu_uuid": GPU, "panel_sha256": common_panel,
              "independent_checks": checks, "rows": rows,
              "teacher": {"summary_path": str(teacher_path), "summary_file_sha256": sha256(teacher_path),
                          "source_identity": tr["source_identity"], "probes": teacher_probes}}
    if args.report:
        verify_report(args.report, rows, teacher_probes)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(canonical(result) + b"\n")
    print(json.dumps({"student_batteries": len(rows), "teacher_batteries": 1,
                      "panel_sha256": common_panel, "output": str(args.output),
                      "output_sha256": sha256(args.output)}, sort_keys=True))


if __name__ == "__main__":
    main()
