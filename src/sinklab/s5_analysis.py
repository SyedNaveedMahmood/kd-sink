"""S5 reuse-only joins over verified S1 evaluation records."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .evaluate import RecordStore, _key
from .metrics import METRIC_VERSION, aggregate_behavior
from .provenance import SHA256_PATTERN, _no_duplicate_keys, payload_digest, verify_envelope


class S5Error(ValueError):
    pass


CONDITIONS = ("C1", "C2", "C5", "C6")
MEASURES = ("sink_s", "clean_ce_nats", "full_jsd_nats", "nonsink_jsd_nats",
            "full_mse_probability_cells", "nonsink_mse_probability_cells",
            "delete_delta_ce_nats", "delete_self_kl_nats",
            "delete_absolute_target_logprob_change_nats", "delete_prediction_flip_fraction",
            "relocate_delta_ce_nats", "relocate_self_kl_nats",
            "relocate_absolute_target_logprob_change_nats", "relocate_prediction_flip_fraction")


def _weighted_attention(items: list[dict], kind: str, field: str, weight: str) -> float:
    pairs = []
    for value in items:
        mapped = value["mapped_attention_similarity"]
        if not mapped:
            raise S5Error("teacher-mapped attention records required")
        for layer in mapped:
            record = layer[kind]
            pairs.append((record[field], record[weight]))
    denominator = sum(n for _, n in pairs)
    if denominator <= 0:
        raise S5Error("attention comparison has no valid support")
    return sum(x * n for x, n in pairs) / denominator


def extract_s1_record(*, aggregate_path: Path, store_root: Path,
                      run_manifest: dict) -> dict:
    """Verify a full causal aggregate plus all item records; never evaluate a model."""
    if not isinstance(run_manifest, dict) or set(run_manifest) != {
            "run_id", "condition", "seed", "device_role", "initialization_sha256",
            "data_sha256", "protocol_sha256"}:
        raise S5Error("complete immutable S1 run manifest required")
    for field in ("initialization_sha256", "data_sha256", "protocol_sha256"):
        if not isinstance(run_manifest[field], str) or not SHA256_PATTERN.fullmatch(run_manifest[field]):
            raise S5Error(f"invalid run manifest {field}")
    try:
        document = json.loads(Path(aggregate_path).read_text(encoding="utf-8"),
                              object_pairs_hook=_no_duplicate_keys)
        aggregate, aggregate_sha = verify_envelope(document)
    except (OSError, ValueError) as exc:
        raise S5Error(f"cannot verify S1 aggregate: {exc}") from exc
    key, operations = aggregate["key"], aggregate["operations"]
    if Path(aggregate_path).name != f"aggregate-{payload_digest(key)}.json":
        raise S5Error("aggregate filename/key mismatch")
    identity = key["run_identity"]
    if (key["metric_version"] != METRIC_VERSION or key["evaluation_mode"] != "full" or
            key["model_role"] != "student" or key["run_id"] != run_manifest["run_id"] or
            identity["study"] != "S1" or identity["condition"] != run_manifest["condition"] or
            identity["seed"] != run_manifest["seed"] or
            identity["protocol_sha256"] != run_manifest["protocol_sha256"] or
            identity["corpus_sha256"] != run_manifest["data_sha256"]):
        raise S5Error("S1 aggregate/run provenance mismatch")
    if run_manifest["condition"] not in CONDITIONS or set(operations) != {"clean", "delete", "relocate"}:
        raise S5Error("S5 needs existing C1/C2/C5/C6 clean/delete/relocate records")
    item_ids = operations["clean"]["complete_item_ids"]
    if not item_ids or any(op["status"] != "complete" or op["complete_item_ids"] != item_ids or
                           op["failed_item_ids"] or op["missing_item_ids"] for op in operations.values()):
        raise S5Error("incomplete or mismatched S1 item coverage")
    store = RecordStore(store_root)
    item_values = {}
    item_digests = []
    for op in ("clean", "delete", "relocate"):
        values = []
        for item_id in item_ids:
            row_key = _key(run_id=key["run_id"], step=key["step"], panel=key["panel"],
                panel_hash=key["panel_hash"], checkpoint_hash=key["checkpoint_hash"],
                item_id=item_id, scope=key["scope"], operation=op,
                strength=0. if op == "clean" else 1., precision=key["precision"],
                model_role="student", evaluation_mode="full", run_identity=identity,
                denominator_floor=key["fingerprint_denominator_floor"])
            row = store.read(row_key)
            if row is None or row["status"] != "complete":
                raise S5Error(f"missing or failed S1 item {item_id} {op}")
            values.append(row["value"])
            item_digests.append(payload_digest(row))
        item_values[op] = values
        if aggregate_behavior([v["behavior"] for v in values]) != operations[op]["metrics"]:
            raise S5Error(f"aggregate metrics disagree with {op} item records")
    clean_values = item_values["clean"]
    measures = {"sink_s": sum(v["clean_structure"]["native_layer_mean"] for v in clean_values) / len(clean_values),
                "clean_ce_nats": operations["clean"]["metrics"]["clean_ce_nats"]}
    for prefix, kind in (("full", "full"), ("nonsink", "sink_excluded")):
        measures[f"{prefix}_jsd_nats"] = _weighted_attention(clean_values, kind, "jsd_nats", "valid_query_count")
        measures[f"{prefix}_mse_probability_cells"] = _weighted_attention(
            clean_values, kind, "mse_probability_cells", "valid_edge_count")
    for op in ("delete", "relocate"):
        metrics = operations[op]["metrics"]
        for source, target in (("delta_ce_nats", "delta_ce_nats"),
                               ("self_kl_nats", "self_kl_nats"),
                               ("absolute_target_logprob_change_nats", "absolute_target_logprob_change_nats"),
                               ("prediction_flip_fraction", "prediction_flip_fraction")):
            measures[f"{op}_{target}"] = metrics[source]
    if set(measures) != set(MEASURES) or any(not isinstance(x, (int, float)) or not math.isfinite(x) for x in measures.values()):
        raise S5Error("missing or nonfinite S5 measure")
    return {"study": "S1", "condition": run_manifest["condition"], "seed": run_manifest["seed"],
        "device_role": run_manifest["device_role"], "run_id": key["run_id"],
        "step": key["step"], "panel": key["panel"], "panel_sha256": key["panel_hash"],
        "checkpoint_sha256": key["checkpoint_hash"], "precision": key["precision"],
        "layer_scope": key["scope"], "metric_version": key["metric_version"],
        "protocol_sha256": identity["protocol_sha256"],
        "initialization_sha256": run_manifest["initialization_sha256"],
        "data_sha256": run_manifest["data_sha256"],
        "source_aggregate_sha256": aggregate_sha,
        "source_item_bundle_sha256": payload_digest({"item_record_sha256": item_digests}),
        "item_ids": item_ids, "measures": measures, "status": "complete"}


def join_s5(rows: list[dict], *, device_role: str, panel_sha256: str,
            steps: list[int], seeds: list[int]) -> dict:
    """Join observed quartets only; report every absent record without imputation."""
    if not steps or not seeds or len(set(steps)) != len(steps) or len(set(seeds)) != len(seeds):
        raise S5Error("explicit unique steps and training seeds required")
    selected = {}
    for row in rows:
        if row["device_role"] != device_role or row["panel_sha256"] != panel_sha256 or row["step"] not in steps or row["seed"] not in seeds:
            continue
        if (row["study"] != "S1" or row["condition"] not in CONDITIONS or
                row["status"] != "complete" or set(row["measures"]) != set(MEASURES)):
            raise S5Error("only complete reused S1 C1/C2/C5/C6 rows are allowed")
        if any(not isinstance(row[field], str) or not SHA256_PATTERN.fullmatch(row[field])
               for field in ("checkpoint_sha256", "source_aggregate_sha256",
                             "source_item_bundle_sha256", "protocol_sha256",
                             "initialization_sha256", "data_sha256")):
            raise S5Error("unverified S5 source provenance")
        if any(not isinstance(value, (int, float)) or not math.isfinite(value)
               for value in row["measures"].values()):
            raise S5Error("nonfinite S5 source measure")
        key = (row["step"], row["seed"], row["condition"])
        if key in selected:
            raise S5Error("duplicate condition/seed/step, possibly a hardware replica")
        selected[key] = row
    missing = [{"step": step, "seed": seed, "condition": condition}
               for step in steps for seed in seeds for condition in CONDITIONS
               if (step, seed, condition) not in selected]
    joined = []
    for step in steps:
        for seed in seeds:
            if any((step, seed, condition) not in selected for condition in CONDITIONS):
                continue
            quartet = {condition: selected[step, seed, condition] for condition in CONDITIONS}
            if len({row["run_id"] for row in quartet.values()}) != len(CONDITIONS):
                raise S5Error("one S1 run ID cannot stand for multiple conditions")
            first = quartet["C1"]
            for condition, row in quartet.items():
                for field in ("protocol_sha256", "initialization_sha256", "data_sha256",
                              "panel_sha256", "panel", "precision", "layer_scope",
                              "metric_version", "item_ids"):
                    if row[field] != first[field]:
                        raise S5Error(f"incompatible S5 {field}: {condition}")
            comparisons = {}
            for a, b in (("C1", "C2"), ("C2", "C5"), ("C2", "C6"),
                         ("C1", "C5"), ("C1", "C6")):
                comparisons[f"{b}_minus_{a}"] = {metric: quartet[b]["measures"][metric] -
                    quartet[a]["measures"][metric] for metric in MEASURES}
            joined.append({"step": step, "seed": seed, "device_role": device_role,
                "panel_sha256": panel_sha256,
                "source_run_ids": {c: quartet[c]["run_id"] for c in CONDITIONS},
                "source_aggregate_sha256": {c: quartet[c]["source_aggregate_sha256"] for c in CONDITIONS},
                "source_item_bundle_sha256": {c: quartet[c]["source_item_bundle_sha256"] for c in CONDITIONS},
                "measures_by_condition": {c: quartet[c]["measures"] for c in CONDITIONS},
                "comparisons": comparisons})
    return {"status": "complete" if not missing else "incomplete",
            "joined": joined, "missing": missing, "device_role": device_role,
            "panel_sha256": panel_sha256, "steps": steps, "seeds": seeds,
            "analysis": "reuse_only_no_training"}
