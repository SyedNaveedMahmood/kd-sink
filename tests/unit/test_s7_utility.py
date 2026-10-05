import copy
import hashlib
import json
from pathlib import Path

import pytest

from sinklab.metrics import METRIC_VERSION
from sinklab.provenance import canonical_json_bytes
from sinklab.s5_analysis import CONDITIONS, MEASURES, join_s5
from sinklab.s5_compatibility import D26_PATH
from sinklab.s7_utility import (DENSE_STEPS, FULL_STEPS, PANELS, S7Error,
                                add_clean_supplements, add_extended_s1_records,
                                add_lm2000_endpoints, analyze_s7, extract_extended_s1_record,
                                extract_lm2000_record, read_s5_bundle)


def _write(path, value):
    path.write_bytes(canonical_json_bytes(value) + b"\n")


def _sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _bundle(tmp_path):
    root = tmp_path / "s5"
    root.mkdir()
    d26_path = Path(D26_PATH)
    d26 = json.loads(d26_path.read_text())
    critical = d26["payload"]["critical_invariants"]
    rows = []
    for panel, (_, steps) in PANELS.items():
        for step in steps:
            for n, condition in enumerate(CONDITIONS):
                approved = d26["payload"]["allowed_runs"][condition]
                measures = {m: float(n + step / 10000) for m in MEASURES}
                measures["clean_ce_nats"] = 2 + n * .1 + step / 100000
                rows.append({"study": "S1", "condition": condition, "seed": 0,
                    "device_role": "rtx4080super", "run_id": approved["run_id"],
                    "step": step, "panel": panel, "panel_sha256": critical["panel_manifest_sha256"],
                    "checkpoint_sha256": "a" * 64, "precision": "bf16",
                    "layer_scope": list(range(24)), "metric_version": METRIC_VERSION,
                    "protocol_sha256": approved["protocol_root_sha256"],
                    "initialization_sha256": critical["initialization_sha256"],
                    "data_sha256": critical["data_sha256"],
                    "source_aggregate_sha256": "b" * 64,
                    "source_item_bundle_sha256": "c" * 64,
                    "item_ids": ([str(i) for i in range(300)] if panel == "owt_full300"
                                 else [str(i) for i in range(64)]), "measures": measures,
                    "status": "complete", "gpu_uuid": approved["gpu_uuid"],
                    "hardware_sha256": approved["hardware_lock_sha256"],
                    "source_commit": approved["source_commit"],
                    "environment_lock_sha256": approved["environment_lock_sha256"],
                    "objective_variant": d26["payload"]["condition_objective_variants"][condition],
                    "comparison_invariants": critical})
    (root / "S5_REAGGREGATED_SOURCE_ROWS.jsonl").write_bytes(
        b"".join(canonical_json_bytes(r) + b"\n" for r in rows))
    for panel, (filename, steps) in PANELS.items():
        selected = [r for r in rows if r["panel"] == panel]
        result = join_s5(selected, device_role="rtx4080super",
            panel_sha256=critical["panel_manifest_sha256"], steps=list(steps),
            seeds=[0], compatibility_document=d26)
        _write(root / filename, result)
    _write(root / "S5_SOURCE_VERIFICATION.json", {"schema_version": 1,
        "D26_sha256": d26["sha256"], "metric_version": METRIC_VERSION,
        "joins": {"dense64": {"status": "complete"}, "full300": {"status": "complete"}}})
    files = ["S5_REAGGREGATED_SOURCE_ROWS.jsonl", "S5_JOINED_OWT_DENSE64.json",
             "S5_JOINED_OWT_FULL300.json", "S5_SOURCE_VERIFICATION.json"]
    (root / "SHA256SUMS.txt").write_text("".join(f"{_sha(root / f)}  {f}\n" for f in sorted(files)))
    audit = {"schema_version": 1, "status": "COMPLETE", "model_loaded": False,
        "new_inference": False, "source_run_directories_modified": False,
        "d26_sha256": d26["sha256"], "source_row_count": 440,
        "joined_row_count": 110, "source_rows_sha256": _sha(root / files[0]),
        "dense_join_sha256": _sha(root / files[1]),
        "full300_join_sha256": _sha(root / files[2]),
        "source_verification_sha256": _sha(root / files[3]),
        "sha256sums_sha256": _sha(root / "SHA256SUMS.txt")}
    _write(root / "STAGE08_S5_AUDIT.json", audit)
    (root / "STAGE08_S5_AUDIT.json.sha256").write_text(
        f"{_sha(root / 'STAGE08_S5_AUDIT.json')}  STAGE08_S5_AUDIT.json\n")
    return root, d26_path


def test_verified_bundle_rejoined_and_temporal_grid(tmp_path):
    root, d26 = _bundle(tmp_path)
    before = {p.name: p.read_bytes() for p in root.iterdir()}
    panels, provenance = read_s5_bundle(root, d26_path=d26)
    result = analyze_s7(panels, provenance)
    assert result["status"] == "descriptive_s5_reuse_only"
    assert result["training_seed_count"] == 1
    assert len(set(result["provenance"]["source_gpu_uuids"].values())) == 3
    assert len(result["dense64_trajectory"]) == 101
    assert result["full300_endpoint"]["step"] == 10000
    assert result["full300_endpoint"]["paired_contrasts"]["clean_ce_nats"]["C5_minus_C2"] == pytest.approx(.1)
    assert result["dense64_temporal"]["clean_ce_nats"]["C5_minus_C2"]["auc_mean_difference"] == pytest.approx(.1)
    assert result["unavailable"]["exact_mass_shape_decomposition"]
    assert {p.name: p.read_bytes() for p in root.iterdir()} == before


@pytest.mark.parametrize("target", ["S5_JOINED_OWT_DENSE64.json", "SHA256SUMS.txt",
                                    "STAGE08_S5_AUDIT.json", "S5_REAGGREGATED_SOURCE_ROWS.jsonl"])
def test_byte_tampering_rejected(tmp_path, target):
    root, d26 = _bundle(tmp_path)
    with (root / target).open("ab") as stream:
        stream.write(b" ")
    with pytest.raises(S7Error):
        read_s5_bundle(root, d26_path=d26)


def test_missing_grid_and_duplicate_seed_fail(tmp_path):
    root, d26 = _bundle(tmp_path)
    panels, provenance = read_s5_bundle(root, d26_path=d26)
    bad = copy.deepcopy(panels)
    bad["owt_dense64"]["joined"].pop()
    with pytest.raises(S7Error, match="grid"):
        analyze_s7(bad, provenance)
    bad = copy.deepcopy(panels)
    bad["owt_dense64"]["seeds"] = [0, 0]
    with pytest.raises(S7Error, match="grid"):
        analyze_s7(bad, provenance)


def test_extended_original_measure_join_preserves_missing_values(tmp_path):
    root, d26 = _bundle(tmp_path)
    panels, provenance = read_s5_bundle(root, d26_path=d26)
    result = analyze_s7(panels, provenance)
    records = [{"panel": panel, "step": step, "condition": condition, "seed": 0,
                "extended": {"teacher_kl_nats": condition_index + step / 10000,
                             "clean_accuracy_fraction": .5,
                             "delete_relative_delta_ce_percent": None}}
               for panel, (_, steps) in PANELS.items() for step in steps
               for condition_index, condition in enumerate(CONDITIONS)]
    enriched = add_extended_s1_records(result, records)
    assert enriched["extended_full300_endpoint"]["paired_contrasts"]["teacher_kl_nats"]["C5_minus_C2"] == pytest.approx(1)
    assert enriched["extended_full300_endpoint"]["paired_contrasts"]["delete_relative_delta_ce_percent"]["C5_minus_C2"] is None
    assert "teacher_kl_agreement_accuracy" not in enriched["unavailable"]
    with pytest.raises(S7Error, match="grid"):
        add_extended_s1_records(result, records[:-1])


def test_original_s1_reextraction_uses_item_evidence(tmp_path):
    from test_stage08_s5 import _records
    from sinklab.provenance import verify_envelope

    rows = _records(tmp_path)
    row = rows[0]
    aggregate = next(path for path in (tmp_path / "records").glob("aggregate-*.json")
                     if verify_envelope(json.loads(path.read_text()))[1] == row["source_aggregate_sha256"])
    extended = extract_extended_s1_record(aggregate_path=aggregate,
        store_root=tmp_path / "records", source_row=row)
    assert extended["extended"]["teacher_kl_nats"] is not None
    assert extended["extended"]["clean_accuracy_fraction"] >= 0
    wrong = copy.deepcopy(row)
    wrong["measures"]["sink_s"] += .1
    with pytest.raises(S7Error, match="differs"):
        extract_extended_s1_record(aggregate_path=aggregate,
            store_root=tmp_path / "records", source_row=wrong)


def test_complete_supplement_grid_and_signed_mass_shape_gains(tmp_path):
    from sinklab.metrics import S7_DECOMPOSITION_VERSION
    from sinklab.training_entry import _teacher_map

    lock_sha = "f" * 64
    teacher_identity = {
        "id": "openai-community/gpt2-large",
        "revision": "32b71b12589c2f8d625668d2335a01cac3249519",
        "weights_sha256": "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce",
        "config_sha256": "7fccdcfd6622055342a734c663ee0b61ff4fd697f42467595df0bf4448c8c170",
        "device_model": "NVIDIA GeForce RTX 4080 SUPER",
        "device_uuid": "GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf",
        "precision": "fp32",
        "D24_sha256": "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6",
    }
    teacher_receipt = {
        "followup_study": "S7", "reference_model": teacher_identity["id"],
        "reference_revision": teacher_identity["revision"],
        "teacher_weights_sha256": teacher_identity["weights_sha256"],
        "teacher_config_sha256": teacher_identity["config_sha256"],
        "device": {"name": teacher_identity["device_model"],
                   "uuid": teacher_identity["device_uuid"]},
        "precision": "fp32", "D24_sha256": teacher_identity["D24_sha256"],
    }
    root, d26 = _bundle(tmp_path)
    panels, provenance = read_s5_bundle(root, d26_path=d26)
    result = analyze_s7(panels, provenance)
    teacher_map = list(_teacher_map("S1"))
    supplements = []
    for step in FULL_STEPS:
        joined = panels["owt_full300"]["joined"][FULL_STEPS.index(step)]
        for index, condition in enumerate(CONDITIONS):
            # C2 improves mass but worsens shape relative to C1; signed gains
            # must retain both directions and still close to full gain.
            mass = .4 - .1 * index
            shape = .1 + .2 * index
            layers = [{"student_layer": i, "teacher_layer": t,
                       "full_jsd_nats": mass + shape, "mass_jsd_nats": mass,
                       "shape_jsd_nats": shape, "key0_jsd_nats": mass / 2,
                       "other_columns_jsd_nats": mass / 2 + shape,
                       "valid_query_count": 127, "conditional_valid_query_count": 127,
                       "conditional_jsd_nats": .01,
                       "max_closure_error_nats": 0.}
                      for i, t in enumerate(teacher_map)]
            items = [{"item_id": str(i), "layers": layers} for i in range(300)]
            aggregate = {k: layers[0][k] for k in ("full_jsd_nats", "mass_jsd_nats",
                "shape_jsd_nats", "key0_jsd_nats", "other_columns_jsd_nats",
                "conditional_jsd_nats")}
            aggregate["valid_query_count"] = 7200 * 127
            aggregate["conditional_valid_query_count"] = 7200 * 127
            supplements.append({"study": "S7", "status": "complete", "condition": condition,
                "training_seed": 0, "step": step, "panel": "owt_full300",
                "run_id": joined["source_run_ids"][condition],
                "original_protocol_root_sha256": joined["source_protocol_roots"][condition],
                "s5_source_audit_sha256": provenance["s5_audit_sha256"],
                "analysis_lock_sha256": lock_sha,
                "teacher_receipt": teacher_receipt,
                "result": {"version": S7_DECOMPOSITION_VERSION, "precision": "fp32",
                    "teacher_map": teacher_map, "mapped_layer_count": 24,
                    "item_count": 300, "items": items, "aggregate": aggregate}})
    enriched = add_clean_supplements(result, panels, supplements,
        expected_analysis_lock_sha256=lock_sha,
        expected_teacher_identity=teacher_identity)
    gain = enriched["retained_full300_decomposition"][-1]["C1_relative_gains"]["C2_relative_to_C1"]
    assert gain["mass_jsd_nats"] > 0 and gain["shape_jsd_nats"] < 0
    assert gain["full_jsd_nats"] == pytest.approx(gain["mass_jsd_nats"] + gain["shape_jsd_nats"])
    assert "exact_mass_shape_decomposition" not in enriched["unavailable"]
    with pytest.raises(S7Error, match="incomplete"):
        add_clean_supplements(result, panels, supplements[:-1],
            expected_analysis_lock_sha256=lock_sha,
            expected_teacher_identity=teacher_identity)

    wrong_teacher = copy.deepcopy(supplements)
    wrong_teacher[0]["teacher_receipt"]["followup_study"] = "S4"
    with pytest.raises(S7Error, match="teacher receipt"):
        add_clean_supplements(result, panels, wrong_teacher,
            expected_analysis_lock_sha256=lock_sha,
            expected_teacher_identity=teacher_identity)
    wrong_teacher = copy.deepcopy(supplements)
    wrong_teacher[0]["teacher_receipt"]["teacher_weights_sha256"] = "0" * 64
    with pytest.raises(S7Error, match="teacher receipt"):
        add_clean_supplements(result, panels, wrong_teacher,
            expected_analysis_lock_sha256=lock_sha,
            expected_teacher_identity=teacher_identity)
    wrong_teacher = copy.deepcopy(supplements)
    wrong_teacher[0]["teacher_receipt"]["device"]["uuid"] = "GPU-other"
    with pytest.raises(S7Error, match="teacher receipt"):
        add_clean_supplements(result, panels, wrong_teacher,
            expected_analysis_lock_sha256=lock_sha,
            expected_teacher_identity=teacher_identity)
    wrong_lock = copy.deepcopy(supplements)
    wrong_lock[0]["analysis_lock_sha256"] = "a" * 64
    with pytest.raises(S7Error, match="approved analysis lock"):
        add_clean_supplements(result, panels, wrong_lock,
            expected_analysis_lock_sha256=lock_sha,
            expected_teacher_identity=teacher_identity)


def test_lm2000_endpoint_rechecks_original_nll_items(tmp_path):
    from transformers import GPT2Config, GPT2LMHeadModel
    from sinklab.evaluate import RecordStore, evaluate_panel
    from sinklab.models import GPT2Adapter
    from test_stage08_s5 import _records

    source = _records(tmp_path)[0]
    config = GPT2Config(vocab_size=23, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=2, n_head=4, _attn_implementation="eager")
    adapter = GPT2Adapter(GPT2LMHeadModel(config))
    items = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
             {"id": "b", "input_ids": [4, 3, 2, 1], "attention_mask": [1, 1, 1, 1]}]
    identity = {"study": "S1", "condition": "C1", "seed": 0,
                "model_sha256": "a" * 64, "corpus_sha256": "b" * 64,
                "protocol_sha256": "c" * 64}
    result = evaluate_panel(adapter=adapter, items=items, panel="owt_lm2000",
        panel_hash="d" * 64, checkpoint_hash=source["checkpoint_sha256"],
        run_id=source["run_id"], step=100, store=RecordStore(tmp_path / "records"),
        operations=("clean",), behavior_only=True, run_identity=identity,
        denominator_floor=1e-8, execution_context="registered_training", terminal=False)
    extracted = extract_lm2000_record(aggregate_path=Path(result["record_path"]),
        store_root=tmp_path / "records", source_full300_row=source,
        expected_item_ids=["a", "b"])
    assert extracted["item_count"] == 2
    assert extracted["clean_ce_nats"] > 0
    with pytest.raises(S7Error, match="coverage"):
        extract_lm2000_record(aggregate_path=Path(result["record_path"]),
            store_root=tmp_path / "records", source_full300_row=source,
            expected_item_ids=["b", "a"])


def test_lm2000_corroboration_remains_separate_from_full300(tmp_path):
    root, d26 = _bundle(tmp_path)
    panels, provenance = read_s5_bundle(root, d26_path=d26)
    baseline = analyze_s7(panels, provenance)
    rows = [{"condition": condition, "seed": 0, "step": step,
             "panel": "owt_lm2000", "source_aggregate_sha256": "a" * 64,
             "clean_ce_nats": 2 + index * .1, "clean_ppl": 3 + index,
             "clean_accuracy_fraction": .5 - index * .01}
            for step in (0, 10000) for index, condition in enumerate(CONDITIONS)]
    result = add_lm2000_endpoints(baseline, rows)
    assert len(result["lm2000_endpoints"]) == 2
    assert result["lm2000_endpoints"][1]["paired_contrasts"]["clean_ce_nats"]["C5_minus_C2"] == pytest.approx(.1)
    assert result["full300_endpoint"] == baseline["full300_endpoint"]
    with pytest.raises(S7Error, match="incomplete"):
        add_lm2000_endpoints(baseline, rows[:-1])
