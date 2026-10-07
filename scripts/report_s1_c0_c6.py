"""Read-only extraction of S1 C0-C6 training and aggregate evaluation results.

The compact archive validation receipt is the authority for item-file coverage and
transfer integrity. This script independently checks training and aggregate records
and writes a small derived JSON outside the repository for report traceability.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


DENSE_STEPS = list(range(0, 10001, 100))
FULL_STEPS = [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]
SINK_SNAPSHOT_STEPS = [0, 500, 2000, 5000, 10000]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def payload_digest(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def unique_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def require(value: bool, detail: str) -> None:
    if not value:
        raise ValueError(detail)


def metric(metrics: dict, name: str) -> float:
    value = metrics.get(name)
    require(isinstance(value, (int, float)) and math.isfinite(value), f"nonfinite/missing {name}: {value}")
    return float(value)


def mean_sd(values: list[float]) -> dict:
    return {"mean": statistics.mean(values), "sample_sd": statistics.stdev(values), "n": len(values)}


def clean_item_values(store_root: Path, aggregate: dict) -> list[dict]:
    """Read the clean student item rows named by the sealed aggregate key."""
    key = aggregate["key"]
    values = []
    for item_id in aggregate["operations"]["clean"]["complete_item_ids"]:
        item_key = {name: key[name] for name in ("run_id", "step", "panel", "panel_hash", "checkpoint_hash", "scope", "precision", "model_role", "run_identity", "fingerprint_denominator_floor", "evaluation_mode", "metric_version")}
        item_key.update(item_id=item_id, operation="clean", strength=0.0, intervention_version="prevalue-v1")
        if "followup_policy" in key:
            item_key["followup_policy"] = key["followup_policy"]
        digest = payload_digest(item_key)
        path = store_root / f"{digest}.json"
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
        payload = document["payload"]
        require(document.get("schema_version") == 1 and document.get("sha256") == payload_digest(payload), f"unsealed/corrupt clean item: {path}")
        require(payload["key"] == item_key and payload["status"] == "complete", f"clean item key/status: {path}")
        values.append(payload["value"])
    return values


def weighted_attention(values: list[dict], kind: str, field: str, weight: str) -> float:
    terms = [(layer[kind][field], layer[kind][weight]) for value in values for layer in value["mapped_attention_similarity"]]
    require(terms and sum(n for _, n in terms) > 0, f"empty attention {kind}")
    return sum(v * n for v, n in terms) / sum(n for _, n in terms)


def auc_mean(rows: list[dict], name: str) -> float:
    require([r["step"] for r in rows] == DENSE_STEPS, f"incomplete AUC: {name}")
    return sum((a[name] + b[name]) * (b["step"] - a["step"]) / 2 for a, b in zip(rows, rows[1:])) / 10000


def summarize_run(entry: dict, run_path: Path) -> dict:
    run_id = entry["run_id"]
    condition, seed = entry["condition"], entry["seed"]
    train_path = run_path / "train.jsonl"
    require(train_path.is_file(), f"missing train log: {run_id}")
    train_sha = entry.get("train_sha256", entry.get("train_jsonl_sha256"))
    require(sha256(train_path) == train_sha, f"train SHA differs from receipt: {run_id}")
    starts, updates = [], []
    with train_path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            record = json.loads(line, object_pairs_hook=unique_pairs)
            if record.get("event") == "start":
                starts.append(record)
            elif record.get("event") == "update":
                updates.append(record)
    require(starts and updates, f"missing start/update: {run_id}")
    require(all(r["condition"] == condition and r["seed"] == seed and r["run_id"] == run_id for r in starts), f"start identity: {run_id}")
    require(all(r["protocol_hash"] == entry["protocol_root"] for r in starts), f"protocol root: {run_id}")
    require([u["step"] for u in updates] == list(range(updates[0]["step"], 10001)), f"update gap/duplicate: {run_id}")
    require(updates[-1]["input_tokens"] == 81920000 and updates[-1]["target_tokens"] == 81280000, f"token endpoint: {run_id}")
    expected_first = 2501 if condition == "C3" and seed == 0 else 1
    require(updates[0]["step"] == expected_first, f"unexpected train prefix: {run_id}")
    for update in updates:
        for name in ("ce", "loss", "lr", "grad_norm"):
            metric(update, name)
    last = updates[-1]
    rows: dict[tuple[str, str, int], dict] = {}
    aggregate_payloads: dict[tuple[str, str, int], dict] = {}
    panels: dict[str, set[str]] = {}
    corpora: set[str] = set()
    models: dict[str, set[str]] = {}
    metric_versions: set[str] = set()
    aggregate_paths = sorted((run_path / "evaluation").glob("aggregate-*.json"))
    require(len(aggregate_paths) == 222, f"aggregate count: {run_id} {len(aggregate_paths)}")
    for path in aggregate_paths:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
        payload = document["payload"]
        require(document.get("schema_version") == 1 and document.get("sha256") == payload_digest(payload), f"unsealed/corrupt aggregate: {path}")
        key = payload["key"]
        identity = key["run_identity"]
        panel, role, step = key["panel"], key["model_role"], key["step"]
        require(key["run_id"] == run_id and identity["seed"] == seed and identity["condition"] == condition and identity["study"] == "S1", f"aggregate run identity: {path}")
        require(identity["protocol_sha256"] == entry["protocol_root"], f"aggregate protocol identity: {path}")
        require(panel in ("owt_dense64", "owt_full300", "owt_lm2000"), f"panel: {path}")
        require(step in (DENSE_STEPS if panel == "owt_dense64" else FULL_STEPS if panel == "owt_full300" else [0, 10000]), f"step: {path}")
        require(role in (("student",) if panel == "owt_lm2000" else ("student", "teacher")), f"role: {path}")
        require((panel, role, step) not in rows, f"duplicate aggregate: {path}")
        panels.setdefault(panel, set()).add(key["panel_hash"])
        corpora.add(identity["corpus_sha256"])
        models.setdefault(role, set()).add(identity["model_sha256"])
        metric_versions.add(key["metric_version"])
        operations = payload["operations"]
        expected_ops = {"clean"} if panel == "owt_lm2000" else {"clean", "delete", "relocate"}
        require(set(operations) == expected_ops, f"operations: {path}")
        clean_ids = operations["clean"]["complete_item_ids"]
        clean_targets = operations["clean"]["metrics"]["valid_targets"]
        result = {"step": step}
        for operation, item in operations.items():
            require(item["status"] == "complete" and not item["missing_item_ids"] and not item["failed_item_ids"], f"incomplete aggregate: {path}")
            metrics = item["metrics"]
            n_expected = 64 if panel == "owt_dense64" else 300 if panel == "owt_full300" else 2000
            require((panel == "owt_lm2000" or metrics["item_count"] == n_expected) and len(item["complete_item_ids"]) == n_expected, f"item count: {path}")
            require(item["complete_item_ids"] == clean_ids and len(set(clean_ids)) == n_expected and metrics["valid_targets"] == clean_targets, f"operation item/target mismatch: {path}")
            result[f"{operation}_ce"] = metric(metrics, "clean_ce_nats" if panel == "owt_lm2000" else "edited_ce_nats")
            result[f"{operation}_valid_targets"] = metrics["valid_targets"]
            if operation != "clean":
                for source, target in (("delta_ce_nats", "delta_ce"), ("self_kl_nats", "self_kl"), ("prediction_flip_fraction", "flips"), ("absolute_target_logprob_change_nats", "absolute_logprob_change")):
                    result[f"{operation}_{target}"] = metric(metrics, source)
            else:
                for source, target in (("teacher_kl_nats", "teacher_kl"), ("teacher_top1_agreement_fraction", "teacher_agreement"), ("clean_accuracy_fraction", "accuracy")):
                    value = metrics.get(source)
                    if value is not None:
                        result[target] = metric(metrics, source)
        rows[(panel, role, step)] = result
        if role == "student" and (panel == "owt_dense64" and step in SINK_SNAPSHOT_STEPS or panel == "owt_full300" and step == 10000):
            aggregate_payloads[(panel, role, step)] = payload
    expected_keys = {(panel, role, step) for panel, steps, roles in (("owt_dense64", DENSE_STEPS, ("student", "teacher")), ("owt_full300", FULL_STEPS, ("student", "teacher")), ("owt_lm2000", [0, 10000], ("student",))) for role in roles for step in steps}
    require(set(rows) == expected_keys, f"aggregate coverage: {run_id}")
    require(all(len(v) == 1 for v in panels.values()) and len(corpora) == 1 and len(metric_versions) == 1, f"within-run panel/corpus/metric drift: {run_id}")
    dense = [rows[("owt_dense64", "student", step)] for step in DENSE_STEPS]
    full = [rows[("owt_full300", "student", step)] for step in FULL_STEPS]
    endpoint = full[-1]
    teacher_endpoint = rows[("owt_full300", "teacher", 10000)]
    lm = rows[("owt_lm2000", "student", 10000)]
    dense_sink = []
    for step in SINK_SNAPSHOT_STEPS:
        values = clean_item_values(run_path / "evaluation", aggregate_payloads[("owt_dense64", "student", step)])
        dense_sink.append({"step": step, "sink_s": statistics.mean(metric(v["clean_structure"], "native_layer_mean") for v in values)})
    endpoint_values = clean_item_values(run_path / "evaluation", aggregate_payloads[("owt_full300", "student", 10000)])
    full_sink = statistics.mean(metric(v["clean_structure"], "native_layer_mean") for v in endpoint_values)
    full_jsd = weighted_attention(endpoint_values, "full", "jsd_nats", "valid_query_count")
    nonsink_jsd = weighted_attention(endpoint_values, "sink_excluded", "jsd_nats", "valid_query_count")
    require(len(models["student"]) >= 1, f"missing student model identity: {run_id}")
    return {
        "condition": condition, "seed": seed, "run_id": run_id, "run_path": str(run_path),
        "train_sha256": train_sha, "protocol_root": entry["protocol_root"],
        "gpu_uuid": entry["gpu_uuid"], "gpu_model": entry["gpu_model"],
        "training": {"first_update": updates[0]["step"], "update_count": len(updates), "start_events": len(starts), "resumed_steps": [s["resumed_step"] for s in starts], "final_step": last["step"], "final_input_tokens": last["input_tokens"], "final_target_tokens": last["target_tokens"], "final_loss": last["loss"], "final_ce": last["ce"], "last_100_ce_mean": statistics.mean(u["ce"] for u in updates[-100:]), "last_100_loss_mean": statistics.mean(u["loss"] for u in updates[-100:]), "final_lr": last["lr"], "final_elapsed_seconds": last.get("elapsed_seconds")},
        "identity": {"panel_hashes": {k: next(iter(v)) for k, v in panels.items()}, "corpus_sha256": next(iter(corpora)), "metric_version": next(iter(metric_versions)), "teacher_model_sha256": sorted(models.get("teacher", [])), "student_model_sha256_count": len(models["student"])},
        "aggregate_count": len(rows), "endpoint_full300": dict(endpoint, sink_s=full_sink, full_jsd_nats=full_jsd, nonsink_jsd_nats=nonsink_jsd), "endpoint_full300_teacher": teacher_endpoint, "initial_full300": full[0], "endpoint_lm2000": lm,
        "dense_sink_snapshots": dense_sink,
        "dense_auc_mean": {k: auc_mean(dense, k) for k in ("clean_ce", "delete_delta_ce", "relocate_delta_ce", "delete_self_kl", "relocate_self_kl", "delete_flips", "relocate_flips")},
        "dense_late_8000_10000_mean": {k: statistics.mean(r[k] for r in dense if r["step"] >= 8000) for k in ("clean_ce", "delete_delta_ce", "relocate_delta_ce", "delete_self_kl", "relocate_self_kl")},
    }


def verify_report(report_path: Path, runs: list[dict], across: dict, contrasts: dict, receipt: dict, derived_sha: str) -> None:
    """Reject transcription errors in the published endpoint and summary tables."""
    report = report_path.read_text(encoding="utf-8")
    require(derived_sha in report, "report does not cite derived inventory SHA")
    require(all(archive["archive_sha256"] in report for archive in receipt["archives"]), "report archive SHA omission")
    require(all(run["protocol_root"] in report for run in runs), "report protocol root omission")
    endpoint_section = report.split("## Full300 results at optimizer step 10,000", 1)[1].split("### Across-seed descriptive summaries", 1)[0]
    endpoint_rows = [line for line in endpoint_section.splitlines() if line.startswith("| C") and any(f" | {seed} |" in line for seed in (0, 1, 2))]
    require(len(endpoint_rows) == 21, f"report endpoint rows: {len(endpoint_rows)}")
    for run in runs:
        prefix = f"| {run['condition']} | {run['seed']} |"
        matches = [line for line in endpoint_rows if line.startswith(prefix)]
        require(len(matches) == 1, f"report endpoint identity: {prefix}")
        actual = [part.strip() for part in matches[0].strip("|").split("|")]
        endpoint = run["endpoint_full300"]
        expected = [f"{endpoint[name]:.4f}" for name in ("clean_ce", "sink_s", "full_jsd_nats", "delete_delta_ce", "relocate_delta_ce", "delete_self_kl", "delete_flips", "teacher_kl")]
        expected.append(f"{run['endpoint_lm2000']['clean_ce']:.4f}")
        require(actual[2:] == expected, f"report endpoint values: {prefix}")
    diagnostics_section = report.split("### Additional endpoint diagnostics", 1)[1].split("### Across-seed descriptive summaries", 1)[0]
    for condition in [f"C{i}" for i in range(7)]:
        subset = [run for run in runs if run["condition"] == condition]
        rows = [line for line in diagnostics_section.splitlines() if line.startswith(f"| {condition} |")]
        require(len(rows) == 1, f"report diagnostic row: {condition}")
        actual = [part.strip() for part in rows[0].strip("|").split("|")][1:]
        expected = [" / ".join(f"{run['endpoint_full300'][name]:.4f}" for run in subset) for name in ("nonsink_jsd_nats", "teacher_agreement", "relocate_self_kl")]
        require(actual == expected, f"report diagnostic values: {condition}")
    summary_section = report.split("### Across-seed descriptive summaries", 1)[1].split("### Shared-panel within-seed condition contrasts", 1)[0]
    for condition, values in across.items():
        rows = [line for line in summary_section.splitlines() if line.startswith(f"| {condition} |")]
        require(len(rows) == 1, f"report summary row: {condition}")
        actual = [part.strip().replace("−", "-") for part in rows[0].strip("|").split("|")][1:]
        expected = [f"{values[name]['mean']:.4f} ± {values[name]['sample_sd']:.4f}" for name in ("clean_ce", "sink_s", "full_jsd_nats", "delete_delta_ce", "relocate_delta_ce")]
        require(actual == expected, f"report summary values: {condition}")
    contrast_section = report.split("### Shared-panel within-seed condition contrasts", 1)[1].split("## Longitudinal development", 1)[0]
    for name, values in contrasts.items():
        label = name.replace("-", " − ")
        rows = [line for line in contrast_section.splitlines() if line.startswith(f"| {label} |")]
        require(len(rows) == 1, f"report contrast row: {name}")
        actual = [part.strip().replace("−", "-") for part in rows[0].strip("|").split("|")][1:]
        expected = [f"{values[field]['mean']:+.4f} ± {values[field]['sample_sd']:.4f}" for field in ("clean_ce", "sink_s", "full_jsd_nats", "delete_delta_ce", "relocate_delta_ce")]
        require(actual == expected, f"report contrast values: {name}: {actual} != {expected}")
    training_section = report.split("## Training completion and known provenance limitation", 1)[1].split("## Full300 results", 1)[0]
    trajectory_section = report.split("## Longitudinal development", 1)[1].split("## Interpretation and limits", 1)[0]
    for condition in [f"C{i}" for i in range(7)]:
        subset = [run for run in runs if run["condition"] == condition]
        require([run["seed"] for run in subset] == [0, 1, 2], f"report seed order: {condition}")
        training_rows = [line for line in training_section.splitlines() if line.startswith(f"| {condition} |")]
        require(len(training_rows) == 1, f"training row: {condition}")
        actual = [part.strip() for part in training_rows[0].strip("|").split("|")][1:]
        expected = [" / ".join(f"{run['training']['last_100_ce_mean']:.4f}" for run in subset),
                    " / ".join(f"{run['endpoint_lm2000']['clean_ce']:.4f}" for run in subset)]
        require(actual == expected, f"training/LM2000 values: {condition}")
        trajectory_rows = [line for line in trajectory_section.splitlines() if line.startswith(f"| {condition} |")]
        require(len(trajectory_rows) == 1, f"trajectory row: {condition}")
        actual = [part.strip() for part in trajectory_rows[0].strip("|").split("|")][1:]
        expected = [f"{statistics.mean(run['dense_sink_snapshots'][i]['sink_s'] for run in subset):.4f}" for i in range(5)]
        expected.extend((f"{statistics.mean(run['dense_auc_mean']['delete_delta_ce'] for run in subset):.4f}",
                         f"{statistics.mean(run['dense_late_8000_10000_mean']['delete_delta_ce'] for run in subset):.4f}"))
        require(actual == expected, f"trajectory values: {condition}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    arguments = parser.parse_args()
    receipt = json.loads(arguments.receipt.read_text(encoding="utf-8"), object_pairs_hook=unique_pairs)
    require(receipt.get("overall_status") == "PASS", "source compact archive receipt not PASS")
    runs = []
    for archive in receipt["archives"]:
        for seed in ("0", "1", "2"):
            entry = archive["seeds"][seed]
            runs.append(summarize_run(entry, Path(entry["canonical_run_path"])))
    require(len(runs) == 21, "not 21 runs")
    require({r["identity"]["metric_version"] for r in runs} == {"e6a-v2-metrics-1"}, "metric schema mismatch")
    require({tuple(r["identity"]["teacher_model_sha256"]) for r in runs} == {("ced962846241d165bd652cdb61ee346765709677feb98384c5ef1dc483928d06",)}, "teacher model identity mismatch")
    # Paired condition comparisons require exactly matching panel and corpus per seed.
    by_seed = {}
    for seed in (0, 1, 2):
        subset = [r for r in runs if r["seed"] == seed]
        by_seed[str(seed)] = {"panel_hash_groups": {panel: {r["condition"]: r["identity"]["panel_hashes"][panel] for r in subset} for panel in ("owt_dense64", "owt_full300", "owt_lm2000")}, "corpus_hashes": {r["condition"]: r["identity"]["corpus_sha256"] for r in subset}}
        require(all(len(set(by_seed[str(seed)]["panel_hash_groups"][panel].values())) == 1 for panel in ("owt_dense64", "owt_full300", "owt_lm2000")) and len(set(by_seed[str(seed)]["corpus_hashes"].values())) == 1, f"within-seed panel/corpus mismatch: {seed}")
    across = {}
    for condition in [f"C{i}" for i in range(7)]:
        subset = [r for r in runs if r["condition"] == condition]
        across[condition] = {key: mean_sd([r["endpoint_full300"][key] for r in subset]) for key in ("clean_ce", "sink_s", "full_jsd_nats", "delete_delta_ce", "relocate_delta_ce", "delete_self_kl", "delete_flips")}
    contrasts = {}
    run_map = {(r["seed"], r["condition"]): r for r in runs}
    for left, right in (("C1", "C0"), ("C2", "C1"), ("C3", "C2"), ("C4", "C2"), ("C5", "C2"), ("C6", "C2"), ("C6", "C1")):
        name = f"{left}-{right}"
        contrasts[name] = {}
        for field in ("clean_ce", "sink_s", "full_jsd_nats", "delete_delta_ce", "relocate_delta_ce", "delete_self_kl"):
            differences = [run_map[(seed, left)]["endpoint_full300"][field] - run_map[(seed, right)]["endpoint_full300"][field] for seed in (0, 1, 2)]
            contrasts[name][field] = dict(seed_differences=differences, **mean_sd(differences), positive_sign_count=sum(x > 0 for x in differences), negative_sign_count=sum(x < 0 for x in differences))
    output = {"schema": "s1-c0-c6-read-only-derived-v3", "source_receipt": str(arguments.receipt), "source_receipt_sha256": sha256(arguments.receipt), "aggregate_scope": "222 sealed aggregate records per run; clean student items at five Dense64 snapshot steps and Full300 endpoint; source receipt audits all item files", "runs": runs, "within_seed_identity": by_seed, "across_seed_descriptive": across, "within_seed_shared_panel_contrasts": contrasts}
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(json.dumps(output, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    if arguments.report:
        verify_report(arguments.report, runs, across, contrasts, receipt, sha256(arguments.output))
    print(f"PASS runs={len(runs)} aggregates={sum(r['aggregate_count'] for r in runs)} output={arguments.output} sha256={sha256(arguments.output)}")


if __name__ == "__main__":
    main()
