"""S7 read-only analysis of a sealed S5 seed-0 source bundle.

S5 records are rejoined under their original D26 exception. S7 only derives
descriptive contrasts; it does not extend D26's permission to new inference.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from .metrics import METRIC_VERSION
from .provenance import _no_duplicate_keys, payload_digest, verify_envelope
from .s5_analysis import CONDITIONS, MEASURES, extract_s1_record, join_s5
from .s5_compatibility import D26_PATH, validate_d26


class S7Error(ValueError):
    pass


DENSE_STEPS = tuple(range(0, 10001, 100))
FULL_STEPS = (0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000)
PAIRS = (("C5", "C2"), ("C6", "C1"), ("C2", "C1"),
         ("C5", "C1"), ("C6", "C2"))
PANELS = {"owt_dense64": ("S5_JOINED_OWT_DENSE64.json", DENSE_STEPS),
          "owt_full300": ("S5_JOINED_OWT_FULL300.json", FULL_STEPS)}
S5_AUDIT_SHA256 = "09b3663cda822c7b5b3b697bbc3a44c4849b63e85f24b0a98abfce06c99dfa7c"
FILES = ("S5_REAGGREGATED_SOURCE_ROWS.jsonl", "S5_JOINED_OWT_DENSE64.json",
         "S5_JOINED_OWT_FULL300.json", "S5_SOURCE_VERIFICATION.json")


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_checksum_file(root: Path) -> None:
    lines = (root / "SHA256SUMS.txt").read_text(encoding="ascii").splitlines()
    seen = set()
    for line in lines:
        if len(line) < 67 or line[64:66] != "  ":
            raise S7Error("malformed S5 checksum line")
        name, expected = line[66:], line[:64]
        if (not name or name in seen or name.startswith("/") or "\\" in name or
                Path(name).is_absolute() or ".." in Path(name).parts or
                len(expected) != 64 or any(c not in "0123456789abcdef" for c in expected)):
            raise S7Error("unsafe or duplicate S5 checksum entry")
        seen.add(name)
        if _sha(root / name) != expected:
            raise S7Error(f"S5 checksum mismatch: {name}")
    if seen != set(FILES):
        raise S7Error("S5 checksum file must list the four source artifacts")


def read_s5_bundle(root: Path, *, d26_path: Path,
                   expected_audit_sha256: str | None = None) -> tuple[dict, dict]:
    """Verify bytes, audit, all rows, D26 identities, and both exact joins."""
    root = Path(root)
    audit = _json(root / "STAGE08_S5_AUDIT.json")
    if expected_audit_sha256 is not None and _sha(root / "STAGE08_S5_AUDIT.json") != expected_audit_sha256:
        raise S7Error("S5 audit differs from the pinned completed scientific artifact")
    sidecar = (root / "STAGE08_S5_AUDIT.json.sha256").read_text(encoding="ascii")
    if sidecar != f"{_sha(root / 'STAGE08_S5_AUDIT.json')}  STAGE08_S5_AUDIT.json\n":
        raise S7Error("S5 audit sidecar mismatch")
    if (audit.get("schema_version") != 1 or audit.get("status") != "COMPLETE" or
            audit.get("model_loaded") is not False or audit.get("new_inference") is not False or
            audit.get("source_run_directories_modified") is not False):
        raise S7Error("S5 audit is not an immutable completed reuse-only bundle")
    _check_checksum_file(root)
    expected = {"S5_REAGGREGATED_SOURCE_ROWS.jsonl": "source_rows_sha256",
                "S5_JOINED_OWT_DENSE64.json": "dense_join_sha256",
                "S5_JOINED_OWT_FULL300.json": "full300_join_sha256",
                "S5_SOURCE_VERIFICATION.json": "source_verification_sha256",
                "SHA256SUMS.txt": "sha256sums_sha256"}
    if any(audit.get(key) != _sha(root / name) for name, key in expected.items()):
        raise S7Error("S5 audit file digest mismatch")
    compatibility = validate_d26(_json(d26_path))
    if audit.get("d26_sha256") != compatibility["sha256"]:
        raise S7Error("S5 audit D26 seal mismatch")
    verification = _json(root / "S5_SOURCE_VERIFICATION.json")
    if (verification.get("schema_version") != 1 or
            verification.get("D26_sha256") != compatibility["sha256"] or
            verification.get("metric_version") != METRIC_VERSION or
            verification.get("joins", {}).get("dense64", {}).get("status") != "complete" or
            verification.get("joins", {}).get("full300", {}).get("status") != "complete"):
        raise S7Error("S5 source verification mismatch")
    with (root / FILES[0]).open(encoding="utf-8") as source:
        rows = [json.loads(line, object_pairs_hook=_no_duplicate_keys) for line in source]
    if len(rows) != 440 or audit.get("source_row_count") != 440:
        raise S7Error("S5 source row coverage mismatch")
    panels = {}
    for panel, (filename, steps) in PANELS.items():
        selected = [row for row in rows if row.get("panel") == panel]
        if len(selected) != 4 * len(steps):
            raise S7Error(f"S5 {panel} source coverage mismatch")
        rebuilt = join_s5(selected, device_role="rtx4080super",
            panel_sha256=compatibility["critical_invariants"]["panel_manifest_sha256"],
            steps=list(steps), seeds=[0], compatibility_document=_json(d26_path))
        saved = _json(root / filename)
        if rebuilt != saved or rebuilt["status"] != "complete":
            raise S7Error(f"S5 {panel} join differs from verified source rows")
        if (len(rebuilt["joined"]) != len(steps) or
                [r["step"] for r in rebuilt["joined"]] != list(steps)):
            raise S7Error(f"S5 {panel} step order/coverage mismatch")
        panels[panel] = rebuilt
    if audit.get("joined_row_count") != sum(len(p["joined"]) for p in panels.values()):
        raise S7Error("S5 joined-row audit count mismatch")
    return panels, {"s5_audit_sha256": _sha(root / "STAGE08_S5_AUDIT.json"),
                    "d26_sha256": compatibility["sha256"],
                    "s5_source_rows_sha256": audit["source_rows_sha256"],
                    "full300_item_ids": next(row["item_ids"] for row in rows
                                             if row["panel"] == "owt_full300"),
                    "source_gpu_uuids": {c: compatibility["allowed_runs"][c]["gpu_uuid"]
                                         for c in CONDITIONS}}


def _paired(row: dict, metric: str) -> dict:
    return {f"{a}_minus_{b}": row["measures_by_condition"][a][metric] -
            row["measures_by_condition"][b][metric] for a, b in PAIRS}


def analyze_s7(panels: dict, provenance: dict) -> dict:
    """Prespecified endpoint and dense time summaries, one actual training seed."""
    if set(panels) != set(PANELS):
        raise S7Error("both S5 panels required")
    for panel, (_, steps) in PANELS.items():
        joined = panels[panel]
        if (joined.get("status") != "complete" or joined.get("missing") or
                joined.get("seeds") != [0] or joined.get("steps") != list(steps) or
                [r["step"] for r in joined["joined"]] != list(steps)):
            raise S7Error(f"incomplete or unpaired S7 {panel} grid")
        for row in joined["joined"]:
            if (row["seed"] != 0 or set(row["measures_by_condition"]) != set(CONDITIONS) or
                    any(set(v) != set(MEASURES) for v in row["measures_by_condition"].values())):
                raise S7Error("S7 requires exact seed0 C1/C2/C5/C6 measures")
    dense = panels["owt_dense64"]["joined"]
    endpoint = panels["owt_full300"]["joined"][-1]
    start = panels["owt_full300"]["joined"][0]
    if endpoint["step"] != 10000 or start["step"] != 0:
        raise S7Error("S7 full300 endpoint/start mismatch")
    trajectory = [{"step": r["step"], "panel": "owt_dense64", "seed": 0,
                   "condition_values": r["measures_by_condition"],
                   "paired_contrasts": {m: _paired(r, m) for m in MEASURES}}
                  for r in dense]
    temporal = {}
    for metric in MEASURES:
        contrasts = {}
        for a, b in PAIRS:
            name = f"{a}_minus_{b}"
            values = [r["paired_contrasts"][metric][name] for r in trajectory]
            # Exact 0..10k trapezoidal integral, divided by 10k updates.
            auc = sum((values[i] + values[i + 1]) / 2 *
                      (DENSE_STEPS[i + 1] - DENSE_STEPS[i])
                      for i in range(len(values) - 1)) / 10000
            late = [v for step, v in zip(DENSE_STEPS, values) if step >= 8000]
            contrasts[name] = {"auc_mean_difference": auc,
                               "late_8000_to_10000_mean_difference": sum(late) / len(late),
                               "endpoint_dense64_difference": values[-1]}
        temporal[metric] = contrasts
    return {"schema_version": 1, "study": "S7", "status": "descriptive_s5_reuse_only",
            "training_seed_count": 1, "training_seeds": [0], "device_role": "rtx4080super",
            "source_hardware_caveat": "three physical GPU UUIDs; condition is confounded with device",
            "provenance": provenance, "metric_units": {"clean_ce_nats": "nats_per_target",
                "sink_s": "probability", "full_jsd_nats": "nats_per_query",
                "nonsink_jsd_nats": "nats_per_query"},
            "full300_start": {"step": 0, "condition_values": start["measures_by_condition"],
                              "paired_contrasts": {m: _paired(start, m) for m in MEASURES}},
            "full300_endpoint": {"step": 10000, "condition_values": endpoint["measures_by_condition"],
                                 "paired_contrasts": {m: _paired(endpoint, m) for m in MEASURES}},
            "dense64_trajectory": trajectory, "dense64_temporal": temporal,
            "unavailable": {"teacher_kl_agreement_accuracy": "not_in_S5_source_rows",
                "lm2000_endpoint": "not_in_S5_source_rows",
                "exact_mass_shape_decomposition": "requires_new_verified_clean_feature_supplement",
                "topology_profiles": "requires_original_S1_item_extraction",
                "teacher_causal_reference": "requires_separate_teacher_role_records",
                "noop_key1_controls": "not_in_selected_S5_bundle",
                "publication_figures": "not_generated_by_current_JSON_analysis",
                "equivalence_or_noninferiority": "no_approved_margin_or_multiseed_design"},
            "interpretation": "descriptive paired training recipes; no causal mediation or population inference"}


def evaluate_clean_decomposition(*, adapter, teacher_adapter, teacher_map: list[int],
                                 items: list[dict], precision: str = "fp32",
                                 on_item=None) -> dict:
    """Evaluate clean attention only, with RNG/mode restoration and no writes.

    The caller must preflight the immutable S1 checkpoint, teacher and panel.
    This function provides the numerical supplement, never source admission.
    """
    import torch

    from .evaluate import rng_neutral
    from .metrics import S7_DECOMPOSITION_VERSION, attention_jsd_decomposition

    if (precision not in {"fp32", "bf16"} or not items or
            len(teacher_map) != adapter.layer_count or
            len(set(teacher_map)) != len(teacher_map) or
            any(type(k) is not int or not 0 <= k < teacher_adapter.layer_count
                for k in teacher_map)):
        raise S7Error("invalid S7 supplement scope, map, items, or precision")
    device = next(adapter.model.parameters()).device
    if next(teacher_adapter.model.parameters()).device != device:
        raise S7Error("teacher and student must share an inference device")
    seen = set()
    records = []
    with rng_neutral(adapter.model, teacher_adapter.model):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                            enabled=precision == "bf16" and device.type == "cuda"):
            for number, item in enumerate(items, start=1):
                ident = item.get("id")
                tokens, token_mask = item.get("input_ids"), item.get("attention_mask")
                if (not isinstance(ident, str) or not ident or ident in seen or
                        not isinstance(tokens, list) or not isinstance(token_mask, list) or
                        len(tokens) != len(token_mask) or len(tokens) < 2 or
                        any(type(v) is not int or v < 0 for v in tokens) or
                        any(v not in (0, 1) for v in token_mask)):
                    raise S7Error("invalid or duplicate S7 panel item")
                seen.add(ident)
                ids = torch.tensor([tokens], dtype=torch.long, device=device)
                mask = torch.tensor([token_mask], dtype=torch.bool, device=device)
                student = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
                teacher = teacher_adapter.forward_with_features(input_ids=ids,
                                                                 attention_mask=mask)
                layers = []
                for student_layer, teacher_layer in enumerate(teacher_map):
                    measurement = attention_jsd_decomposition(
                        teacher.attention[teacher_layer].probabilities,
                        student.attention[student_layer].probabilities, mask)
                    layers.append({"student_layer": student_layer,
                                   "teacher_layer": teacher_layer, **measurement})
                records.append({"item_id": ident, "layers": layers})
                if on_item is not None:
                    on_item(number, len(items), ident)
    fields = ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats",
              "key0_jsd_nats", "other_columns_jsd_nats")
    flattened = [layer for item in records for layer in item["layers"]]
    aggregate = {field: sum(layer[field] for layer in flattened) / len(flattened)
                 for field in fields}
    conditional = [layer["conditional_jsd_nats"] for layer in flattened]
    aggregate["conditional_jsd_nats"] = (
        sum(conditional) / len(conditional) if all(v is not None for v in conditional) else None)
    aggregate["max_closure_error_nats"] = max(
        layer["max_closure_error_nats"] for layer in flattened)
    aggregate["valid_query_count"] = sum(layer["valid_query_count"] for layer in flattened)
    aggregate["conditional_valid_query_count"] = sum(
        layer["conditional_valid_query_count"] for layer in flattened)
    if not math.isclose(aggregate["full_jsd_nats"],
                        aggregate["mass_jsd_nats"] + aggregate["shape_jsd_nats"],
                        abs_tol=1e-8, rel_tol=0):
        raise S7Error("S7 aggregate mass/shape closure failed")
    return {"version": S7_DECOMPOSITION_VERSION, "status": "complete",
            "precision": precision, "item_count": len(items),
            "mapped_layer_count": len(teacher_map), "teacher_map": teacher_map,
            "reduction": "equal_query_then_equal_item_layer",
            "aggregate": aggregate, "items": records}


def extract_extended_s1_record(*, aggregate_path: Path, store_root: Path,
                               source_row: dict) -> dict:
    """Reverify an original full S1 aggregate/items before exposing omitted fields."""
    manifest = {field: source_row[field] for field in (
        "run_id", "condition", "seed", "device_role", "initialization_sha256",
        "data_sha256", "protocol_sha256", "comparison_invariants",
        "objective_variant", "gpu_uuid", "hardware_sha256", "source_commit",
        "environment_lock_sha256") if field in source_row}
    reconstructed = extract_s1_record(aggregate_path=aggregate_path,
                                      store_root=store_root, run_manifest=manifest)
    if reconstructed != source_row:
        raise S7Error("original S1 item evidence differs from sealed S5 source row")
    payload, digest = verify_envelope(_json(aggregate_path))
    if digest != source_row["source_aggregate_sha256"]:
        raise S7Error("original S1 aggregate payload digest mismatch")
    ops = payload["operations"]
    clean = ops["clean"]["metrics"]
    if clean["teacher_kl_nats"] is None or clean["teacher_top1_agreement_fraction"] is None:
        raise S7Error("S7 clean teacher matching missing from original S1 records")
    extended = {"clean_ppl": clean["clean_ppl"],
                "clean_accuracy_fraction": clean["clean_accuracy_fraction"],
                "teacher_kl_nats": clean["teacher_kl_nats"],
                "teacher_top1_agreement_fraction": clean["teacher_top1_agreement_fraction"],
                "valid_targets": clean["valid_targets"]}
    for op in ("delete", "relocate"):
        metrics = ops[op]["metrics"]
        extended[f"{op}_relative_delta_ce_percent"] = metrics["relative_delta_ce_percent"]
        extended[f"{op}_edited_accuracy_fraction"] = metrics["edited_accuracy_fraction"]
        extended[f"{op}_median_item_absolute_change_nats"] = metrics["median_item_absolute_change_nats"]
        extended[f"{op}_p90_item_absolute_change_nats"] = metrics["p90_item_absolute_change_nats"]
    return {"condition": source_row["condition"], "seed": source_row["seed"],
            "step": source_row["step"], "panel": source_row["panel"],
            "source_aggregate_sha256": digest, "extended": extended}


def add_extended_s1_records(result: dict, records: list[dict]) -> dict:
    """Attach exact-grid original S1 measurements; retain nullable values."""
    expected = {(panel, step, condition) for panel, (_, steps) in PANELS.items()
                for step in steps for condition in CONDITIONS}
    indexed = {}
    for record in records:
        key = (record["panel"], record["step"], record["condition"])
        if key in indexed or record["seed"] != 0:
            raise S7Error("duplicate or non-seed0 extended S1 source row")
        indexed[key] = record
    if set(indexed) != expected:
        raise S7Error("extended S1 record grid is incomplete")
    fields = set(next(iter(indexed.values()))["extended"])
    if any(set(row["extended"]) != fields for row in indexed.values()):
        raise S7Error("extended S1 measure schemas differ")
    output = json.loads(json.dumps(result))
    output["extended_source_aggregates"] = records
    output["extended_full300_endpoint"] = {}
    output["extended_full300_start"] = {}
    output["extended_dense64_trajectory"] = []
    output["extended_dense64_temporal"] = {}
    for panel, (_, steps) in PANELS.items():
        for step in steps:
            condition_values = {c: indexed[(panel, step, c)]["extended"] for c in CONDITIONS}
            paired = {field: {f"{a}_minus_{b}": (
                condition_values[a][field] - condition_values[b][field]
                if isinstance(condition_values[a][field], (int, float)) and
                   isinstance(condition_values[b][field], (int, float)) and
                   math.isfinite(condition_values[a][field]) and
                   math.isfinite(condition_values[b][field]) else None)
                for a, b in PAIRS} for field in fields if field != "valid_targets"}
            row = {"step": step, "condition_values": condition_values,
                   "paired_contrasts": paired}
            if panel == "owt_dense64":
                output["extended_dense64_trajectory"].append(row)
            elif step == 0:
                output["extended_full300_start"] = row
            elif step == 10000:
                output["extended_full300_endpoint"] = row
    output["unavailable"].pop("teacher_kl_agreement_accuracy")
    for field in fields - {"valid_targets"}:
        output["extended_dense64_temporal"][field] = {}
        for a, b in PAIRS:
            name = f"{a}_minus_{b}"
            values = [row["paired_contrasts"][field][name]
                      for row in output["extended_dense64_trajectory"]]
            if any(value is None for value in values):
                output["extended_dense64_temporal"][field][name] = {
                    "status": "unavailable", "reason": "at_least_one_undefined_step"}
                continue
            output["extended_dense64_temporal"][field][name] = {
                "status": "complete",
                "auc_mean_difference": sum((values[i] + values[i + 1]) / 2 *
                    (DENSE_STEPS[i + 1] - DENSE_STEPS[i])
                    for i in range(len(values) - 1)) / 10000,
                "late_8000_to_10000_mean_difference": sum(values[80:]) / 21,
                "endpoint_dense64_difference": values[-1]}
    return output


def read_extended_s1_records(*, s5_bundle: Path, runs_root: Path) -> list[dict]:
    """Reverify every full source item from the original four S1 run directories."""
    rows_path = Path(s5_bundle) / FILES[0]
    with rows_path.open(encoding="utf-8") as source:
        rows = [json.loads(line, object_pairs_hook=_no_duplicate_keys) for line in source]
    if len(rows) != 440:
        raise S7Error("S7 requires all 440 S5 source rows")
    expected_keys = {(panel, step, condition) for panel, (_, steps) in PANELS.items()
                     for step in steps for condition in CONDITIONS}
    indexed = {}
    for row in rows:
        key = (row["panel"], row["step"], row["condition"])
        if key in indexed or row["seed"] != 0:
            raise S7Error("duplicate or non-seed0 S5 source row")
        indexed[key] = row
    if set(indexed) != expected_keys:
        raise S7Error("S5 source grid incomplete for extended extraction")
    records = []
    for condition in CONDITIONS:
        run_id = next(row["run_id"] for row in rows if row["condition"] == condition)
        if any(row["run_id"] != run_id for row in rows if row["condition"] == condition):
            raise S7Error("S5 source run ID changes within a condition")
        store_root = Path(runs_root) / run_id / "evaluation"
        aggregates = {}
        for path in store_root.glob("aggregate-*.json"):
            payload, digest = verify_envelope(_json(path))
            key = payload["key"]
            if (key.get("model_role") == "student" and
                    key.get("evaluation_mode") == "full" and
                    key.get("panel") in PANELS):
                if digest in aggregates:
                    raise S7Error("duplicate S1 aggregate payload digest")
                aggregates[digest] = path
        for panel, (_, steps) in PANELS.items():
            for step in steps:
                row = indexed[(panel, step, condition)]
                path = aggregates.get(row["source_aggregate_sha256"])
                if path is None:
                    raise S7Error(f"missing original S1 aggregate: {condition}/{panel}/{step}")
                records.append(extract_extended_s1_record(
                    aggregate_path=path, store_root=store_root, source_row=row))
    return records


def validate_s7_teacher_receipt(receipt: dict, expected_teacher_identity: dict) -> None:
    """Fail closed unless teacher and device provenance match the S7 lock."""
    expected_fields = {
        "followup_study": "S7",
        "reference_model": expected_teacher_identity["id"],
        "reference_revision": expected_teacher_identity["revision"],
        "teacher_weights_sha256": expected_teacher_identity["weights_sha256"],
        "teacher_config_sha256": expected_teacher_identity["config_sha256"],
        "precision": expected_teacher_identity["precision"],
        "D24_sha256": expected_teacher_identity["D24_sha256"],
    }
    device = receipt.get("device") if isinstance(receipt, dict) else None
    if (not isinstance(receipt, dict) or
            any(receipt.get(key) != value for key, value in expected_fields.items()) or
            not isinstance(device, dict) or
            device.get("name") != expected_teacher_identity["device_model"] or
            device.get("uuid") != expected_teacher_identity["device_uuid"] or
            any(key in receipt for key in
                ("control_seed", "denominator_floor", "responsiveness_floor"))):
        raise S7Error("S7 teacher receipt differs from the pinned teacher, D24, or inference GPU")


def add_clean_supplements(result: dict, panels: dict, supplements: list[dict], *,
                          expected_analysis_lock_sha256: str,
                          expected_teacher_identity: dict) -> dict:
    """Join the complete 4x9 retained-state clean decomposition grid."""
    from .metrics import S7_DECOMPOSITION_VERSION
    from .training_entry import _teacher_map

    if (not isinstance(expected_analysis_lock_sha256, str) or
            len(expected_analysis_lock_sha256) != 64 or
            not isinstance(expected_teacher_identity, dict)):
        raise S7Error("S7 final join requires the approved lock digest and pinned teacher identity")

    indexed = {}
    expected = {(step, condition) for step in FULL_STEPS for condition in CONDITIONS}
    lock_sha = None
    for record in supplements:
        condition, step = record.get("condition"), record.get("step")
        key = (step, condition)
        if key not in expected or key in indexed:
            raise S7Error("unexpected or duplicate S7 supplement")
        joined = panels["owt_full300"]["joined"][FULL_STEPS.index(step)]
        measure = record.get("result", {})
        if (record.get("study") != "S7" or record.get("status") != "complete" or
                record.get("training_seed") != 0 or record.get("panel") != "owt_full300" or
                record.get("run_id") != joined["source_run_ids"][condition] or
                record.get("original_protocol_root_sha256") != joined["source_protocol_roots"][condition] or
                record.get("s5_source_audit_sha256") != result["provenance"]["s5_audit_sha256"] or
                measure.get("version") != S7_DECOMPOSITION_VERSION or
                measure.get("precision") != "fp32" or
                measure.get("teacher_map") != list(_teacher_map("S1")) or
                measure.get("mapped_layer_count") != 24 or
                measure.get("item_count") != 300 or
                len(measure.get("items", [])) != 300):
            raise S7Error("S7 supplement source, map, or item coverage differs")
        if record.get("analysis_lock_sha256") != expected_analysis_lock_sha256:
            raise S7Error("S7 supplement is not bound to the approved analysis lock")
        validate_s7_teacher_receipt(record.get("teacher_receipt"), expected_teacher_identity)
        if lock_sha is None:
            lock_sha = expected_analysis_lock_sha256
        elif lock_sha != expected_analysis_lock_sha256:
            raise S7Error("S7 supplement analysis lock changed across states")
        items = measure["items"]
        if [item["item_id"] for item in items] != result["provenance"]["full300_item_ids"]:
            raise S7Error("S7 supplement item order/identity differs from S5 panel")
        fields = ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats",
                  "key0_jsd_nats", "other_columns_jsd_nats")
        layers = [layer for item in items for layer in item["layers"]]
        if len(layers) != 7200 or any(len(item["layers"]) != 24 for item in items):
            raise S7Error("S7 supplement layer coverage differs")
        if any(layer["student_layer"] != index or
               layer["teacher_layer"] != measure["teacher_map"][index]
               for item in items for index, layer in enumerate(item["layers"])):
            raise S7Error("S7 supplement teacher map differs per layer")
        if any(layer["valid_query_count"] != 127 or
               not 0 <= layer["conditional_valid_query_count"] <= 127 or
               layer["max_closure_error_nats"] > 1e-8 or
               not math.isclose(layer["full_jsd_nats"],
                                layer["mass_jsd_nats"] + layer["shape_jsd_nats"],
                                abs_tol=1e-8, rel_tol=0) or
               not math.isclose(layer["full_jsd_nats"],
                                layer["key0_jsd_nats"] + layer["other_columns_jsd_nats"],
                                abs_tol=1e-8, rel_tol=0)
               for layer in layers):
            raise S7Error("S7 supplement per-layer support or closure differs")
        for field in fields:
            mean = sum(layer[field] for layer in layers) / len(layers)
            if not math.isclose(mean, measure["aggregate"][field], abs_tol=1e-10, rel_tol=0):
                raise S7Error("S7 supplement item/layer reduction disagrees with aggregate")
        aggregate = measure["aggregate"]
        conditional = [layer["conditional_jsd_nats"] for layer in layers]
        expected_conditional = (sum(conditional) / len(conditional)
                                if all(v is not None for v in conditional) else None)
        if (aggregate.get("valid_query_count") != 7200 * 127 or
                aggregate.get("conditional_valid_query_count") != sum(
                    layer["conditional_valid_query_count"] for layer in layers) or
                (expected_conditional is None) != (aggregate.get("conditional_jsd_nats") is None) or
                (expected_conditional is not None and not math.isclose(
                    expected_conditional, aggregate["conditional_jsd_nats"],
                    abs_tol=1e-10, rel_tol=0))):
            raise S7Error("S7 supplement query counts or conditional reduction differ")
        if (not math.isclose(aggregate["full_jsd_nats"], aggregate["mass_jsd_nats"] +
                            aggregate["shape_jsd_nats"], abs_tol=1e-8, rel_tol=0) or
                not math.isclose(aggregate["full_jsd_nats"], aggregate["key0_jsd_nats"] +
                                 aggregate["other_columns_jsd_nats"], abs_tol=1e-8, rel_tol=0)):
            raise S7Error("S7 supplement decomposition does not close")
        indexed[key] = record
    if set(indexed) != expected:
        raise S7Error("S7 clean decomposition grid incomplete")
    output = json.loads(json.dumps(result))
    output["clean_decomposition_analysis_lock_sha256"] = lock_sha
    output["provenance"]["s7_teacher_identity"] = expected_teacher_identity
    output["retained_full300_decomposition"] = []
    for step in FULL_STEPS:
        values = {c: indexed[(step, c)]["result"]["aggregate"] for c in CONDITIONS}
        contrasts = {f"{a}_minus_{b}": {name: values[a][name] - values[b][name]
            for name in ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats",
                         "key0_jsd_nats", "other_columns_jsd_nats")}
            for a, b in PAIRS}
        gains = {f"{c}_relative_to_C1": {name: values["C1"][name] - values[c][name]
            for name in ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats")}
            for c in ("C2", "C5", "C6")}
        output["retained_full300_decomposition"].append(
            {"step": step, "condition_values": values,
             "paired_contrasts": contrasts, "C1_relative_gains": gains})
    output["unavailable"].pop("exact_mass_shape_decomposition")
    return output


def extract_lm2000_record(*, aggregate_path: Path, store_root: Path,
                          source_full300_row: dict, expected_item_ids: list[str]) -> dict:
    """Reverify the NLL-only S1 endpoint and every clean item record."""
    from .evaluate import RecordStore, _key

    aggregate, digest = verify_envelope(_json(aggregate_path))
    key, operations = aggregate["key"], aggregate["operations"]
    if (Path(aggregate_path).name != f"aggregate-{payload_digest(key)}.json" or
            key.get("panel") != "owt_lm2000" or
            key.get("evaluation_mode") != "nll_only" or
            key.get("model_role") != "student" or
            key.get("metric_version") != METRIC_VERSION or
            key.get("run_id") != source_full300_row["run_id"] or
            key.get("step") != source_full300_row["step"] or
            key.get("panel_hash") != source_full300_row["panel_sha256"] or
            key.get("checkpoint_hash") != source_full300_row["checkpoint_sha256"] or
            key.get("precision") != source_full300_row["precision"] or
            key.get("scope") != source_full300_row["layer_scope"] or
            key.get("run_identity", {}).get("study") != "S1" or
            key["run_identity"].get("condition") != source_full300_row["condition"] or
            key["run_identity"].get("seed") != 0 or
            key["run_identity"].get("protocol_sha256") != source_full300_row["protocol_sha256"] or
            key["run_identity"].get("corpus_sha256") != source_full300_row["data_sha256"] or
            set(operations) != {"clean"}):
        raise S7Error("S1 lm2000 aggregate identity differs from S5 full300 state")
    op = operations["clean"]
    if (op.get("status") != "complete" or op.get("failed_item_ids") or
            op.get("missing_item_ids") or op.get("complete_item_ids") != expected_item_ids or
            len(expected_item_ids) != len(set(expected_item_ids))):
        raise S7Error("S1 lm2000 item coverage differs from frozen panel")
    if not Path(store_root).is_dir():
        raise S7Error("original S1 metric record directory missing")
    store = RecordStore(store_root)
    nll = count = correct = 0
    for item_id in expected_item_ids:
        item_key = _key(run_id=key["run_id"], step=key["step"], panel=key["panel"],
            panel_hash=key["panel_hash"], checkpoint_hash=key["checkpoint_hash"],
            item_id=item_id, scope=key["scope"], operation="clean", strength=0.,
            precision=key["precision"], model_role="student", evaluation_mode="nll_only",
            run_identity=key["run_identity"],
            denominator_floor=key["fingerprint_denominator_floor"],
            followup_policy=key.get("followup_policy"))
        record = store.read(item_key)
        if record is None or record.get("status") != "complete" or record.get("schema_version") != METRIC_VERSION:
            raise S7Error("missing or failed S1 lm2000 item record")
        value = record["value"]["endpoint_nll_only"]
        if value["valid_targets"] <= 0:
            raise S7Error("S1 lm2000 item has no targets")
        count += value["valid_targets"]
        nll += value["clean_nll_sum_nats"]
        correct += value["clean_correct_count"]
    metrics = op["metrics"]
    ce = nll / count
    if (metrics["valid_targets"] != count or metrics["clean_ce_nats"] != ce or
            metrics["clean_accuracy_fraction"] != correct / count or
            metrics["clean_ppl"] != math.exp(ce)):
        raise S7Error("S1 lm2000 aggregate differs from reverified item sums")
    return {"condition": source_full300_row["condition"], "seed": 0,
            "step": key["step"], "panel": "owt_lm2000",
            "run_id": key["run_id"], "source_aggregate_sha256": digest,
            "item_count": len(expected_item_ids), "valid_targets": count,
            "clean_ce_nats": ce, "clean_ppl": metrics["clean_ppl"],
            "clean_accuracy_fraction": metrics["clean_accuracy_fraction"]}


def add_lm2000_endpoints(result: dict, records: list[dict]) -> dict:
    expected = {(step, condition) for step in (0, 10000) for condition in CONDITIONS}
    selected = {}
    for row in records:
        key = (row["step"], row["condition"])
        if key in selected or row["seed"] != 0 or row["panel"] != "owt_lm2000":
            raise S7Error("duplicate or ineligible lm2000 endpoint")
        selected[key] = row
    if set(selected) != expected:
        raise S7Error("S7 lm2000 endpoint grid incomplete")
    output = json.loads(json.dumps(result))
    output["lm2000_endpoints"] = []
    for step in (0, 10000):
        values = {c: {name: selected[(step, c)][name]
                      for name in ("clean_ce_nats", "clean_ppl", "clean_accuracy_fraction")}
                  for c in CONDITIONS}
        contrasts = {name: {f"{a}_minus_{b}": values[a][name] - values[b][name]
                            for a, b in PAIRS} for name in next(iter(values.values()))}
        output["lm2000_endpoints"].append(
            {"step": step, "condition_values": values, "paired_contrasts": contrasts,
             "source_aggregate_sha256": {c: selected[(step, c)]["source_aggregate_sha256"]
                                         for c in CONDITIONS}})
    output["unavailable"].pop("lm2000_endpoint")
    return output


def read_lm2000_records(*, s5_bundle: Path, runs_root: Path,
                        artifact_root: Path, repo_root: Path) -> list[dict]:
    """Bind frozen panel membership and eight original NLL-only aggregates."""
    artifact_document = _json(Path(repo_root) / "protocols" / "artifact.lock.json")
    artifact, _ = verify_envelope(artifact_document)
    pinned = artifact["panels"]
    path = (Path(artifact_root) / "panels" /
            f"owt-panels-{pinned['payload_sha256']}.json")
    if _sha(path) != pinned["file_sha256"]:
        raise S7Error("S7 lm2000 panel file differs from pinned artifact lock")
    panel, panel_sha = verify_envelope(_json(path))
    ids = panel.get("owt_lm2000")
    full_ids = panel.get("owt_full300")
    if (panel_sha != pinned["payload_sha256"] or not isinstance(ids, list) or
            len(ids) != 2000 or len(set(ids)) != 2000 or
            not isinstance(full_ids, list) or set(ids) & set(full_ids)):
        raise S7Error("S7 lm2000 frozen panel identity or disjointness differs")
    with (Path(s5_bundle) / FILES[0]).open(encoding="utf-8") as stream:
        rows = [json.loads(line, object_pairs_hook=_no_duplicate_keys) for line in stream]
    records = []
    for condition in CONDITIONS:
        for step in (0, 10000):
            full = [row for row in rows if row["condition"] == condition and
                    row["panel"] == "owt_full300" and row["step"] == step]
            if len(full) != 1 or full[0]["panel_sha256"] != panel_sha:
                raise S7Error("S7 lm2000/S5 full300 state pairing differs")
            source_row = full[0]
            store_root = Path(runs_root) / source_row["run_id"] / "evaluation"
            candidates = []
            for aggregate_path in store_root.glob("aggregate-*.json"):
                aggregate, _ = verify_envelope(_json(aggregate_path))
                key = aggregate["key"]
                if (key.get("model_role") == "student" and
                        key.get("panel") == "owt_lm2000" and
                        key.get("step") == step):
                    candidates.append(aggregate_path)
            if len(candidates) != 1:
                raise S7Error("S7 lm2000 original aggregate missing or duplicated")
            records.append(extract_lm2000_record(aggregate_path=candidates[0],
                store_root=store_root, source_full300_row=source_row,
                expected_item_ids=ids))
    return records
