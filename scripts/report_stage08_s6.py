"""Read-only S6 report extraction from sealed Stage08 summaries and aggregates.

No model, panel text, or item payload is loaded. The independent final audit is
the source for the prior full-tree item-level verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from sinklab.provenance import _no_duplicate_keys, payload_digest, verify_envelope


CONDITIONS = tuple(f"C{i}" for i in range(7))
STEPS = (0, 500, 2000, 10000)
DOMAINS = ("sst2", "gsm8k", "humaneval")
CONTEXTS = (40, 128)
OPERATIONS = ("clean", "delete", "relocate")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sealed(path: Path) -> dict:
    envelope = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    return verify_envelope(envelope)[0]


def require(actual, expected, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: {actual!r} != {expected!r}")


def extract(root: Path, audit_path: Path) -> dict:
    audit = sealed(audit_path)
    require(audit["status"], "COMPLETE_INDEPENDENTLY_VERIFIED", "audit status")
    require(audit["completed_states"], 28, "completed states")
    require(audit["item_operation_records"], 50400, "item records")
    require(audit["aggregate_documents"], 168, "aggregate documents")
    require(audit["conditions"], list(CONDITIONS), "conditions")
    require(audit["steps"], list(STEPS), "steps")
    require(audit["domains"], list(DOMAINS), "domains")
    require(audit["contexts"], list(CONTEXTS), "contexts")
    rows, pooled, paired, summary_hashes, aggregate_hashes, identities = [], [], [], {}, {}, {}
    for condition in CONDITIONS:
        for step in STEPS:
            path = root / "summaries" / condition / f"step-{step:05d}.json"
            state = sealed(path)
            state_key = f"{condition}/{step}"
            summary_hashes[state_key] = sha256(path)
            require(state["condition"], condition, f"{state_key} condition")
            require(state["step"], step, f"{state_key} step")
            require(state["training_seed"], 0, f"{state_key} seed")
            require(state["D24_sha256"], audit["D24_sha256"], f"{state_key} D24")
            require(state["D25_panel_manifest_sha256"], audit["D25_panel_manifest_sha256"], f"{state_key} D25")
            require(state["domains"], list(DOMAINS), f"{state_key} domains")
            require(state["contexts"], list(CONTEXTS), f"{state_key} contexts")
            require(state["precision"], "fp32", f"{state_key} precision")
            require(state["metric_label"], "causal_language_modeling_not_domain_task_accuracy", f"{state_key} label")
            identity = {k: state[k] for k in ("run_id", "original_protocol_root_sha256")}
            if condition in identities:
                require(identity, identities[condition], f"{condition} source identity")
            identities[condition] = identity
            require(set(state["aggregate_keys"]), {f"{d}_{c}" for d in DOMAINS for c in CONTEXTS}, f"{state_key} aggregate keys")
            for context in CONTEXTS:
                for domain in DOMAINS:
                    key = state["aggregate_keys"][f"{domain}_{context}"]
                    aggregate_path = root / "records" / f"aggregate-{payload_digest(key)}.json"
                    aggregate = sealed(aggregate_path)
                    require(aggregate["key"], key, f"{state_key}/{domain}/{context} key")
                    require(key["run_id"], state["run_id"], f"{state_key} run")
                    require(key["step"], step, f"{state_key} key step")
                    require(key["panel"], f"s6_{domain}_{context}", f"{state_key} panel")
                    require(key["run_identity"]["protocol_sha256"], state["original_protocol_root_sha256"], f"{state_key} root")
                    require(key["checkpoint_hash"], state["checkpoint_model_sha256"], f"{state_key} checkpoint model")
                    require(set(aggregate["operations"]), set(OPERATIONS), f"{state_key} operations")
                    metrics = {}
                    for operation in OPERATIONS:
                        entry = aggregate["operations"][operation]
                        require(entry["status"], "complete", f"{state_key}/{domain}/{context}/{operation} status")
                        require(len(entry["complete_item_ids"]), 100, f"{state_key}/{domain}/{context}/{operation} count")
                        require(len(set(entry["complete_item_ids"])), 100, f"{state_key}/{domain}/{context}/{operation} unique IDs")
                        require(entry["failed_item_ids"], [], f"{state_key}/{domain}/{context}/{operation} failed")
                        require(entry["missing_item_ids"], [], f"{state_key}/{domain}/{context}/{operation} missing")
                        metric = entry["metrics"]
                        require(metric["item_count"], 100, f"{state_key}/{domain}/{context}/{operation} metric count")
                        if not all(math.isfinite(metric[name]) for name in ("clean_ce_nats", "delta_ce_nats", "self_kl_nats", "prediction_flip_fraction")):
                            raise ValueError(f"nonfinite metric in {state_key}/{domain}/{context}/{operation}")
                        metrics[operation] = metric
                    require({metrics[op]["valid_targets"] for op in OPERATIONS}, {metrics["clean"]["valid_targets"]}, f"{state_key} target counts")
                    clean_ids = aggregate["operations"]["clean"]["complete_item_ids"]
                    item_base = {name: value for name, value in key.items() if name not in ("kind", "operations")}
                    sink_values = []
                    clean_nll_sum = 0.0
                    item_targets = 0
                    for item_id in clean_ids:
                        item_key = {**item_base, "item_id": item_id, "operation": "clean",
                                    "strength": 0.0, "intervention_version": "prevalue-v1"}
                        item_path = root / "records" / f"{payload_digest(item_key)}.json"
                        item = sealed(item_path)
                        require(item["key"], item_key, f"{state_key}/{domain}/{context} clean item key")
                        require(item["status"], "complete", f"{state_key}/{domain}/{context} clean item status")
                        sink = item["value"]["clean_structure"]["native_layer_mean"]
                        if not math.isfinite(sink) or not 0 <= sink <= 1:
                            raise ValueError(f"{state_key}/{domain}/{context} invalid sink mass")
                        sink_values.append(sink)
                        behavior = item["value"]["behavior"]
                        clean_nll_sum += behavior["clean_nll_sum_nats"]
                        item_targets += behavior["valid_targets"]
                    require(item_targets, metrics["clean"]["valid_targets"], f"{state_key}/{domain}/{context} clean targets")
                    if abs(clean_nll_sum / item_targets - metrics["clean"]["clean_ce_nats"]) > 1e-9:
                        raise ValueError(f"{state_key}/{domain}/{context} clean CE reaggregation mismatch")
                    rows.append({"condition": condition, "step": step, "domain": domain, "context": context,
                                 "valid_targets": metrics["clean"]["valid_targets"],
                                 "sink_mass_equal_item": sum(sink_values) / 100,
                                 "clean_ce_nats": metrics["clean"]["clean_ce_nats"],
                                 "delete_delta_ce_nats": metrics["delete"]["delta_ce_nats"],
                                 "relocate_delta_ce_nats": metrics["relocate"]["delta_ce_nats"],
                                 "delete_self_kl_nats": metrics["delete"]["self_kl_nats"],
                                 "relocate_self_kl_nats": metrics["relocate"]["self_kl_nats"],
                                 "delete_flip_fraction": metrics["delete"]["prediction_flip_fraction"],
                                 "relocate_flip_fraction": metrics["relocate"]["prediction_flip_fraction"],
                                 "panel_sha256": key["panel_hash"]})
                    aggregate_hashes[f"{state_key}/{domain}/{context}"] = sha256(aggregate_path)
                for operation in OPERATIONS:
                    p = state["pooled"][f"{context}_{operation}"]
                    require(p["domain_count"], 3, f"{state_key}/{context}/{operation} pooled domains")
                    require(p["token_weighted"]["item_count"], 300, f"{state_key}/{context}/{operation} pooled items")
                    require(p["equal_item"]["item_count"], 300, f"{state_key}/{context}/{operation} equal items")
                    relevant = [r for r in rows if r["condition"] == condition and r["step"] == step and r["context"] == context]
                    expected_targets = sum(r["valid_targets"] for r in relevant)
                    require(p["token_weighted"]["valid_targets"], expected_targets, f"{state_key}/{context}/{operation} pooled targets")
                    field = "clean_ce_nats" if operation == "clean" else f"{operation}_delta_ce_nats"
                    metric_field = "clean_ce_nats" if operation == "clean" else "delta_ce_nats"
                    expected = sum(r[field] * r["valid_targets"] for r in relevant) / expected_targets
                    if abs(p["token_weighted"][metric_field] - expected) > 1e-9:
                        raise ValueError(f"{state_key}/{context}/{operation} weighted metric mismatch")
                    pooled.append({"condition": condition, "step": step, "context": context, "operation": operation,
                                   "token_weighted": p["token_weighted"], "equal_item": p["equal_item"]})
            for domain in DOMAINS:
                for operation in ("delete", "relocate"):
                    entries = state["paired"][f"{domain}_{operation}"]
                    require(len(entries), 100, f"{state_key}/{domain}/{operation} pair count")
                    require(len({e["document_sha256"] for e in entries}), 100, f"{state_key}/{domain}/{operation} pair identities")
                    values = [e["delta_ce_128_minus_40"] for e in entries]
                    if not all(math.isfinite(v) for v in values):
                        raise ValueError(f"{state_key}/{domain}/{operation} nonfinite pair")
                    paired.append({"condition": condition, "step": step, "domain": domain, "operation": operation,
                                   "item_count": 100, "mean_delta_ce_128_minus_40": sum(values) / 100,
                                   "positive_count": sum(v > 0 for v in values),
                                   "negative_count": sum(v < 0 for v in values),
                                   "zero_count": sum(v == 0 for v in values)})
    require(len(rows), 168, "derived domain/context rows")
    require(len(aggregate_hashes), 168, "aggregate hashes")
    return {"schema": "s6-report-derived-v1", "source_root": str(root), "independent_audit_path": str(audit_path),
            "independent_audit_file_sha256": sha256(audit_path), "independent_audit_payload_sha256": json.loads(audit_path.read_text())["sha256"],
            "audit_counts": {k: audit[k] for k in ("completed_states", "item_operation_records", "aggregate_documents", "S6_result_tree_file_count", "S6_result_tree_total_bytes", "S6_result_tree_SHA256SUMS_sha256")},
            "D24_sha256": audit["D24_sha256"], "D25_panel_manifest_sha256": audit["D25_panel_manifest_sha256"],
            "tokenizer_sha256": audit["tokenizer_sha256"], "identities": identities, "summary_file_sha256": summary_hashes,
            "aggregate_file_sha256": aggregate_hashes, "domain_context_rows": rows,
            "pooled_rows": pooled, "paired_rows": paired}


def report_tables(result: dict) -> dict[str, str]:
    name = {"sst2": "SST-2", "gsm8k": "GSM8K", "humaneval": "HumanEval"}
    endpoint = ["| C | Domain | Max tokens | Targets | `S` | Clean CE | Delete `ΔCE` | Relocate `ΔCE` |",
                "|---|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["domain_context_rows"]:
        if row["step"] == 10000:
            endpoint.append(f"| {row['condition']} | {name[row['domain']]} | {row['context']} | {row['valid_targets']:,} | "
                            f"{row['sink_mass_equal_item']:.4f} | {row['clean_ce_nats']:.4f} | "
                            f"{row['delete_delta_ce_nats']:.4f} | {row['relocate_delta_ce_nats']:.4f} |")
    by_pooled = {(r["condition"], r["step"], r["context"], r["operation"]): r["token_weighted"]
                 for r in result["pooled_rows"]}
    trajectory = ["| C | Step | Clean CE 40 / 128 | Delete `ΔCE` 40 / 128 | Relocate `ΔCE` 40 / 128 |",
                  "|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for step in STEPS:
            def pair(operation: str, field: str) -> str:
                return " / ".join(f"{by_pooled[condition, step, context, operation][field]:.4f}" for context in CONTEXTS)
            trajectory.append(f"| {condition} | {step:,} | {pair('clean', 'clean_ce_nats')} | "
                              f"{pair('delete', 'delta_ce_nats')} | {pair('relocate', 'delta_ce_nats')} |")
    by_pair = {(r["condition"], r["step"], r["domain"], r["operation"]): r for r in result["paired_rows"]}
    paired = ["| C | Domain | Mean delete `ΔCE` 128−40 | Positive / 100 | Mean relocate `ΔCE` 128−40 | Positive / 100 |",
              "|---|---|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for domain in DOMAINS:
            delete = by_pair[condition, 10000, domain, "delete"]
            relocate = by_pair[condition, 10000, domain, "relocate"]
            paired.append(f"| {condition} | {name[domain]} | {delete['mean_delta_ce_128_minus_40']:+.4f} | "
                          f"{delete['positive_count']} | {relocate['mean_delta_ce_128_minus_40']:+.4f} | "
                          f"{relocate['positive_count']} |")
    diagnostics = ["| C | Max tokens | Delete self-KL | Delete flip fraction | Relocate self-KL | Relocate flip fraction |",
                   "|---|---:|---:|---:|---:|---:|"]
    for condition in CONDITIONS:
        for context in CONTEXTS:
            delete = by_pooled[condition, 10000, context, "delete"]
            relocate = by_pooled[condition, 10000, context, "relocate"]
            diagnostics.append(f"| {condition} | {context} | {delete['self_kl_nats']:.4f} | "
                               f"{delete['prediction_flip_fraction']:.3f} | {relocate['self_kl_nats']:.4f} | "
                               f"{relocate['prediction_flip_fraction']:.3f} |")
    return {"{{S6_CENTRAL_TABLE}}": "\n".join(endpoint),
            "{{S6_TRAJECTORY_TABLE}}": "\n".join(trajectory),
            "{{S6_PAIRED_TABLE}}": "\n".join(paired),
            "{{S6_DIAGNOSTIC_TABLE}}": "\n".join(diagnostics)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    result = extract(args.root, args.audit)
    encoded = json.dumps(result, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    if args.output.exists():
        require(args.output.read_text(encoding="utf-8"), encoded, "prior derived inventory")
    else:
        args.output.write_text(encoded, encoding="utf-8")
    if args.report:
        report = args.report.read_text(encoding="utf-8")
        updated = report
        for placeholder, table in report_tables(result).items():
            if placeholder in updated:
                updated = updated.replace(placeholder, table)
            elif table not in updated:
                raise ValueError(f"report table mismatch: {placeholder}")
        if updated != report:
            args.report.write_text(updated, encoding="utf-8")
    print(json.dumps({"output": str(args.output), "sha256": sha256(args.output),
                      "states": 28, "domain_context_rows": len(result["domain_context_rows"]),
                      "aggregate_documents": len(result["aggregate_file_sha256"]),
                      "report_sha256": sha256(args.report) if args.report else None}, sort_keys=True))


if __name__ == "__main__":
    main()
