"""Read-only S1 deletion context for all 35 E0 student states (no model imports)."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

from sinklab.mechanistic_e0 import read_sealed, digest_file, close, require
from sinklab.provenance import payload_digest, seal_payload

STEPS = (0, 100, 500, 2000, 10000)
FIELDS = {
    "clean_ce_nats": "clean_nll_sum_nats",
    "edited_ce_nats": "edited_nll_sum_nats",
    "self_kl_nats": "self_kl_sum_nats",
    "absolute_target_logprob_change_nats": "absolute_target_logprob_change_sum_nats",
    "prediction_flip_fraction": "flip_count",
}


def reaggregate(values, metrics):
    require(len(values) == 300 and all(v["behavior"]["valid_targets"] == 127 for v in values), "Full300 target coverage")
    for field, source in FIELDS.items():
        measured = math.fsum(v["behavior"][source] for v in values) / 38100
        close(measured, metrics[field], field)
    close(metrics["delta_ce_nats"], metrics["edited_ce_nats"] - metrics["clean_ce_nats"], "signed delta CE")
    return math.fsum(v["clean_structure"]["native_layer_mean"] for v in values) / 300


def audit_context(s4_root, output):
    require(not output.exists(), "fresh external output required")
    manifest_path = s4_root / "S4_RUN_MANIFEST.json"
    manifest, _ = read_sealed(manifest_path)
    require(digest_file(manifest_path) == "794bf1673c46c00a190e74bea2f2ec1e1468bf9b5d9420090244d87de225da47", "original S4 manifest pin")
    inventory, rows = {}, []
    for condition, source in sorted(manifest["source_inventory"]["sources"].items()):
        require(source["identity"]["seed"] == 0, "seed0 only")
        store = Path(source["run_path"]) / "evaluation"
        selected = {}
        for path in store.glob("aggregate-*.json"):
            aggregate, _ = read_sealed(path)
            key = aggregate["key"]
            if key["panel"] != "owt_full300" or key["model_role"] != "student" or key["step"] not in STEPS:
                continue
            require(key["step"] not in selected and path.name == f"aggregate-{payload_digest(key)}.json", "duplicate/key mismatch")
            selected[key["step"]] = (path, aggregate)
        require(set(selected) == set(STEPS), "five recorded Full300 states required")
        for step in STEPS:
            path, aggregate = selected[step]
            key = aggregate["key"]
            require(key["run_id"] == source["run_id"] and key["precision"] == "bf16" and key["scope"] == list(range(24)) and
                    key["panel_hash"] == "03fa2adf3931f7594fae23d70e402c697ad2c771e53bdb68977b614336cc5013" and
                    key["run_identity"]["condition"] == condition and key["run_identity"]["seed"] == 0 and
                    key["run_identity"]["protocol_sha256"] == source["original_protocol_root_sha256"], "source identity drift")
            inventory[str(path)] = digest_file(path)
            ids = aggregate["operations"]["clean"]["complete_item_ids"]
            require(len(ids) == len(set(ids)) == 300 and set(aggregate["operations"]) == {"clean", "delete", "relocate"}, "operation coverage")
            row = {"condition": condition, "step": step, "precision": key["precision"], "scope": "native24",
                   "panel_sha256": key["panel_hash"], "checkpoint_tensor_sha256": key["checkpoint_hash"], "source_aggregate_sha256": inventory[str(path)]}
            for operation, summary in aggregate["operations"].items():
                require(summary["status"] == "complete" and summary["complete_item_ids"] == ids and not summary["missing_item_ids"] and
                        not summary["failed_item_ids"] and summary["metrics"]["valid_targets"] == 38100, "incomplete operation")
                values = []
                for ident in ids:
                    item_key = {name: key[name] for name in ("run_id", "step", "panel", "panel_hash", "checkpoint_hash", "scope", "precision", "model_role", "run_identity", "fingerprint_denominator_floor", "evaluation_mode", "metric_version")}
                    item_key.update(item_id=ident, operation=operation, strength=0.0 if operation == "clean" else 1.0, intervention_version="prevalue-v1")
                    if "followup_policy" in key:
                        item_key["followup_policy"] = key["followup_policy"]
                    item_path = store / f"{payload_digest(item_key)}.json"
                    item, _ = read_sealed(item_path)
                    require(item["key"] == item_key and item["status"] == "complete", "item binding")
                    inventory[str(item_path)] = digest_file(item_path)
                    values.append(item["value"])
                sink = reaggregate(values, summary["metrics"])
                row.update({f"{operation}_{field}": summary["metrics"][field] for field in (*FIELDS, "delta_ce_nats")})
                row[f"{operation}_baseline_sink"] = sink
            rows.append(row)
            print(json.dumps({"event": "recorded_S1_context", "condition": condition, "step": step, "records": len(rows) * 900}), flush=True)
    require(len(rows) == 35 and len(inventory) == 31535, "complete 35-state coverage")
    # Source file inventory is rehashed after computation to detect concurrent mutation.
    require(all(digest_file(Path(path)) == sha for path, sha in inventory.items()), "source changed during audit")
    output.mkdir(parents=True)
    with (output / "E0_S1_DELETION_CONTEXT.csv").open("x", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    result = {"status": "COMPLETE", "new_inference": False, "precision_limit": "recorded S1 BF16, separate from S4 FP32",
              "rows": rows, "source_file_hashes": inventory, "record_count": 31500, "aggregate_count": 35,
              "script_sha256": digest_file(Path(__file__)), "s4_manifest_sha256": digest_file(manifest_path)}
    (output / "E0_S1_CONTEXT_AUDIT.json").write_text(json.dumps(seal_payload(result), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--s4-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_context(args.s4_root, args.output)
