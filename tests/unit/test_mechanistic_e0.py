"""Independent E0 fixtures: source integrity, reductions and contrary trajectories."""

import copy
import json
import math
from pathlib import Path

import pytest

from sinklab.mechanistic_e0 import (CONDITIONS, STEPS, PROBES, CHANNELS, E0Error,
    analyze, audit_source, digest_file, fingerprint_recipe, read_sealed,
    trajectory_direction, trajectory_context, verify_bundle)
from sinklab.followup_policy import D24_SHA256, admit_s1_followup
from sinklab.provenance import canonical_json_bytes, payload_digest, seal_payload
from scripts import report_mechanistic_e0 as runner


def sealed(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(canonical_json_bytes(seal_payload(payload)) + b"\n")


def repair_receipts(root, audit_path):
    """Reseal fixtures so semantic corruption is tested independently of hashes."""
    files = sorted([root / "S4_RUN_MANIFEST.json", *(root / "summaries").rglob("*.json"), *(root / "records").glob("*.json")])
    (root / "SHA256SUMS.txt").write_text("".join(f"{digest_file(p)}  {p.relative_to(root).as_posix()}\n" for p in files), encoding="utf-8")
    manifest, _ = read_sealed(root / "S4_RUN_MANIFEST.json")
    run, _ = read_sealed(root / "S4_FINAL_AUDIT.json")
    run.update(run_manifest_sha256=payload_digest(manifest), SHA256SUMS_file_sha256=digest_file(root / "SHA256SUMS.txt"))
    sealed(root / "S4_FINAL_AUDIT.json", run)
    audit, _ = read_sealed(audit_path)
    audit.update(audit_receipt_file_sha256=digest_file(root / "S4_FINAL_AUDIT.json"), run_manifest_envelope_sha256=payload_digest(manifest))
    sealed(audit_path, audit)
    return digest_file(audit_path)


def source_fixture(tmp_path, items=2):
    root, audit_path = tmp_path / "source", tmp_path / "audits" / "audit.json"
    device = {"uuid": "GPU-fixture", "name": "fixture", "cuda_index": 0}
    panel_path = tmp_path / "inputs" / "panel.json"
    panel_payload = {"kind": "owt-upstream-panels-v1", "corpus_sha256": "e" * 64,
                     "owt_full300": [f"item{i}" for i in range(items)]}
    sealed(panel_path, panel_payload)
    panel, teacher_hash = payload_digest(panel_payload), "b" * 64
    identities = {c: {"study": "S1", "condition": c, "seed": 0, "run_id": f"s1-{c.lower()}-seed0-fixture",
        "protocol_hash": str(i + 1) * 64, "gpu_uuid": "GPU-training-fixture", "device_role": "rtx4080super",
        "init_hash": "f" * 64, "data_hash": "e" * 64} for i, c in enumerate(CONDITIONS)}
    manifest = {"D24_sha256": D24_SHA256, "conditions": list(CONDITIONS), "checkpoint_count": 35,
        "probe_ids": list(PROBES), "control_seed": 82, "denominator_floor": 1e-8, "responsiveness_floor": 1e-6,
        "precision": "fp32", "inference_device": device,
        "panel": {"panel_manifest_sha256": panel, "panel_item_count": items, "sequence_length": 128,
            "panel_file_sha256": digest_file(panel_path), "panel_path": str(panel_path), "corpus_payload_sha256": "e" * 64,
            "panel_item_ids_sha256": payload_digest({"item_ids": panel_payload["owt_full300"]})},
        "source_inventory": {"sources": {c: {"identity": identity, "checkpoints": [
            {"step": step, "status": "verified", "identity": identity,
             "followup_policy": admit_s1_followup(identity, study="S4", step=step),
             "manifest_file_sha256": "c" * 64, "model_file_sha256": payload_digest({"condition": c, "step": step})}
            for step in STEPS]} for c, identity in identities.items()}}}
    sealed(root / "S4_RUN_MANIFEST.json", manifest)
    for condition, step in [(c, s) for c in CONDITIONS for s in STEPS] + [(None, None)]:
        role = "teacher" if condition is None else "student"
        identity = None if condition is None else identities[condition]
        model_hash = teacher_hash if condition is None else payload_digest({"condition": condition, "step": step})
        checkpoint_manifest = "c" * 64
        provenance = {"control_seed": 82, "denominator_floor": 1e-8, "responsiveness_floor": 1e-6, "precision": "fp32"}
        provenance.update({"device": device, "teacher_weights_sha256": model_hash} if role == "teacher" else
            {"inference_device": device, "source_checkpoint_manifest_sha256": checkpoint_manifest})
        followup = None if role == "teacher" else admit_s1_followup(identity, study="S4", step=step)
        probes = {}
        for probe_index, probe_id in enumerate(PROBES):
            scope = [] if probe_index < 2 else [0] if probe_id == "epe_transport_layer0" else list(range(36 if role == "teacher" else 24))
            plan = {"status": "applicable", "layers": scope, "locus": probe_id}
            if probe_id.startswith("k_"):
                plan["coordinates"] = [3 * probe_index, 3 * probe_index + 1, 3 * probe_index + 2]
            all_behavior, clean_sinks, edited_sinks = [], [], []
            for item in range(items):
                baseline = .2 + item * .01
                delta_sink = -.02 * (probe_index + 1)
                clean_ce = 3 + item * .1
                delta_ce = .005 * (probe_index + 1) * (1 if role == "teacher" else (1 + (step or 0) / 10000))
                behavior = {"valid_targets": 127, "clean_nll_sum_nats": clean_ce * 127,
                    "edited_nll_sum_nats": (clean_ce + delta_ce) * 127, "self_kl_sum_nats": delta_ce * 2 * 127,
                    "absolute_target_logprob_change_sum_nats": delta_ce * 3 * 127,
                    "flip_count": item + 1, "clean_correct_count": 40, "edited_correct_count": 39,
                    "teacher_kl_sum_nats": None, "teacher_top1_agreement_count": None}
                fp = {"baseline_sink": baseline, "probed_sink": baseline + delta_sink, "absolute_delta_sink": delta_sink,
                    "denominator_floor": 1e-8, "ratio": (baseline + delta_sink) / baseline, "ratio_unavailable_reason": None}
                key = {"study": "S4", "model_role": role, "checkpoint_sha256": model_hash, "panel_sha256": panel,
                    "probe_id": probe_id, "probe": plan, "provenance": provenance, "probe_version": "s4-gpt2-route-v1",
                    "denominator_floor": 1e-8, "responsiveness_floor": 1e-6, "item_id": f"item{item}"}
                if identity is not None:
                    key.update(source_identity=identity, step=step, followup_policy=followup)
                record = {"key": key, "status": "complete", "schema_version": "e6a-v2-metrics-1", "value": {
                    "attention_mask": [[1] * 128], "input_token_count": 128, "behavior": behavior, "fingerprint": fp,
                    "clean_structure": {"native_layer_mean": baseline}, "probed_structure": {"native_layer_mean": baseline + delta_sink}}}
                sealed(root / "records" / f"{payload_digest(key)}.json", record)
                all_behavior.append(behavior)
                clean_sinks.append(baseline)
                edited_sinks.append(baseline + delta_sink)
            n = items * 127
            aggregation = {"schema_version": "e6a-v2-metrics-1", "item_count": items, "valid_targets": n}
            for out, raw in (("clean_ce_nats", "clean_nll_sum_nats"), ("edited_ce_nats", "edited_nll_sum_nats"),
                    ("self_kl_nats", "self_kl_sum_nats"), ("absolute_target_logprob_change_nats", "absolute_target_logprob_change_sum_nats"),
                    ("prediction_flip_fraction", "flip_count"), ("clean_accuracy_fraction", "clean_correct_count"), ("edited_accuracy_fraction", "edited_correct_count")):
                aggregation[out] = sum(b[raw] for b in all_behavior) / n
            aggregation["delta_ce_nats"] = aggregation["edited_ce_nats"] - aggregation["clean_ce_nats"]
            clean_s, edited_s = sum(clean_sinks) / items, sum(edited_sinks) / items
            probes[probe_id] = {"status": "complete", "complete_item_count": items, "failed_item_ids": [], "probe": plan,
                "behavior": aggregation, "fingerprint": {"baseline_sink": clean_s, "probed_sink": edited_s,
                    "absolute_delta_sink": edited_s - clean_s, "denominator_floor": 1e-8,
                    "ratio": edited_s / clean_s, "ratio_unavailable_reason": None}}
        result = {"status": "complete", "probe_version": "s4-gpt2-route-v1", "probes": probes, "provenance": provenance,
            "source_identity": identity, "followup_policy": followup, "step": step, "model_role": role,
            "panel_sha256": panel, "checkpoint_sha256": model_hash}
        summary = {"kind": "s4-probe-battery-summary-v1", "precision": "fp32", "panel_sha256": panel,
            "model_role": role, "step": step, "result": result}
        if role == "student":
            summary.update(condition=condition, training_seed=0, run_id=identity["run_id"], D24_sha256=D24_SHA256,
                original_protocol_root_sha256=identity["protocol_hash"], checkpoint_model_sha256=model_hash,
                checkpoint_manifest_sha256=checkpoint_manifest)
            path = root / "summaries" / condition / f"step-{step:05d}.json"
        else:
            summary["checkpoint_sha256"] = teacher_hash
            path = root / "summaries" / "teacher_full300.json"
        sealed(path, summary)
    checks = {"student_batteries": 35, "student_records": 350 * items, "teacher_records": 10 * items,
        "manifested_file_count": 360 * items + 37, "sha256s_entries_verified": 360 * items + 37,
        **{k: "PASS" for k in ("record_envelopes_and_file_hashes", "unique_record_coverage", "run_and_checkpoint_identities", "panel_file_and_item_identity")}}
    sealed(root / "S4_FINAL_AUDIT.json", {"status": "COMPLETE", "run_manifest_sha256": payload_digest(manifest),
        "SHA256SUMS_file_sha256": "d" * 64, "verified_record_count": 360 * items, "manifested_file_count": 360 * items + 37,
        "training": False, "source_runs_modified": False})
    sealed(audit_path, {"study": "S4", "status": "COMPLETE_INDEPENDENTLY_VERIFIED", "result_directory": str(root),
        "independent_checks": checks, "run_manifest_envelope_sha256": payload_digest(manifest),
        "audit_receipt_file_sha256": "d" * 64, "training": False, "source_runs_modified": False,
        "S1_original_protocol_roots_preserved": {c: x["protocol_hash"] for c, x in identities.items()},
        "runner_session_exit_code": 1, "runner_exit_status_note": "preserved fixture wrapper discrepancy"})
    return root, audit_path, repair_receipts(root, audit_path)


def audit(fixture, **kwargs):
    return audit_source(*fixture, engineering_fixture=True, **kwargs)


@pytest.fixture
def source(tmp_path):
    return source_fixture(tmp_path)


def test_full_audit_preserves_sources_and_individual_controls(source):
    root, _, _ = source
    before = {str(p): digest_file(p) for p in root.rglob("*") if p.is_file()}
    rows, provenance = audit(source)
    assert len(rows) == 360 and provenance["fresh_records_verified"] == 720
    assert provenance["model_loaded"] is False and provenance["new_inference"] is False
    assert {r["probe_id"] for r in rows} == set(PROBES)
    assert {str(p): digest_file(p) for p in root.rglob("*") if p.is_file()} == before
    q = next(r for r in rows if r["condition"] == "C2" and r["step"] == 500 and r["probe_id"] == "q_bias_all")
    assert q["clean_ce_nats"] == pytest.approx(3.05)
    assert q["delta_ce_nats"] == pytest.approx(.01575)
    assert q["prediction_flip_fraction"] == pytest.approx(3 / 254)
    assert q["scope"] == list(range(24))
    assert next(r for r in rows if r["model_role"] == "teacher" and r["probe_id"] == "q_bias_all")["scope"] == list(range(36))
    assert provenance["independent_audit"]["runner_session_exit_code"] == 1


def test_fixture_cannot_claim_production(source):
    with pytest.raises(E0Error, match="300 items"):
        audit_source(*source)


def test_pinned_audit_and_unsealed_json_rejected(source):
    with pytest.raises(E0Error, match="file hash"):
        audit_source(source[0], source[1], "0" * 64, engineering_fixture=True)
    source[1].write_text('{"payload":{},"payload":{}}')
    with pytest.raises(E0Error, match="sealed"):
        read_sealed(source[1])


@pytest.mark.parametrize("mutation", ["missing", "extra", "corrupt", "index_duplicate", "index_escape", "index_backslash"])
def test_file_inventory_corruption(source, mutation):
    root, audit_path, _ = source
    record = next((root / "records").glob("*.json"))
    if mutation == "missing": record.unlink()
    if mutation == "extra": (root / "records" / "extra.json").write_text("{}")
    if mutation == "corrupt": record.write_bytes(record.read_bytes() + b" ")
    if mutation.startswith("index_"):
        line = (root / "SHA256SUMS.txt").read_text().splitlines()[0]
        name = "../escape.json" if mutation == "index_escape" else "records\\escape.json"
        (root / "SHA256SUMS.txt").write_text((root / "SHA256SUMS.txt").read_text() +
            (line if mutation == "index_duplicate" else "0" * 64 + "  " + name) + "\n")
        runner_payload, _ = read_sealed(root / "S4_FINAL_AUDIT.json")
        runner_payload["SHA256SUMS_file_sha256"] = digest_file(root / "SHA256SUMS.txt")
        sealed(root / "S4_FINAL_AUDIT.json", runner_payload)
        receipt, _ = read_sealed(audit_path)
        receipt["audit_receipt_file_sha256"] = digest_file(root / "S4_FINAL_AUDIT.json")
        sealed(audit_path, receipt)
    with pytest.raises(E0Error):
        audit((root, audit_path, digest_file(audit_path)))


@pytest.mark.parametrize("field,value", [("status", "failed"), ("precision", "bf16"), ("training_seed", 1),
    ("panel_sha256", "f" * 64), ("step", 2000), ("checkpoint_model_sha256", "f" * 64)])
def test_resealed_summary_semantic_corruption(source, field, value):
    root, audit_path, _ = source
    path = root / "summaries/C2/step-00500.json"
    payload, _ = read_sealed(path)
    if field == "status": payload["result"][field] = value
    else: payload[field] = value
    sealed(path, payload)
    pinned = repair_receipts(root, audit_path)
    with pytest.raises(E0Error): audit((root, audit_path, pinned))


@pytest.mark.parametrize("mutation", ["aggregate", "record_behavior", "scope", "baseline", "control_seed", "item_count"])
def test_resealed_arithmetic_and_identity_corruption(source, mutation):
    root, audit_path, _ = source
    path = root / "summaries/C2/step-00500.json"
    payload, _ = read_sealed(path)
    probe = payload["result"]["probes"]["q_bias_all"]
    if mutation == "aggregate": probe["behavior"]["self_kl_nats"] += .1
    if mutation == "scope": probe["probe"]["layers"] = [0]
    if mutation == "baseline": probe["fingerprint"]["baseline_sink"] += .1
    if mutation == "control_seed": payload["result"]["provenance"]["control_seed"] += 1
    if mutation == "item_count": probe["behavior"]["valid_targets"] -= 1
    sealed(path, payload)
    if mutation == "record_behavior":
        record_path = next((root / "records").glob("*.json"))
        record, _ = read_sealed(record_path)
        record["value"]["behavior"]["self_kl_sum_nats"] += 1
        sealed(record_path, record)
    pinned = repair_receipts(root, audit_path)
    with pytest.raises(E0Error): audit((root, audit_path, pinned))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "identity", "status", "model", "manifest"])
def test_checkpoint_inventory_is_bound_to_summary(source, mutation):
    root, audit_path, _ = source
    path = root / "S4_RUN_MANIFEST.json"
    manifest, _ = read_sealed(path)
    entries = manifest["source_inventory"]["sources"]["C2"]["checkpoints"]
    if mutation == "missing": entries.pop()
    if mutation == "duplicate": entries[-1] = copy.deepcopy(entries[0])
    if mutation == "identity": entries[2]["identity"] = {**entries[2]["identity"], "seed": 1}
    if mutation == "status": entries[2]["status"] = "failed"
    if mutation == "model": entries[2]["model_file_sha256"] = "a" * 64
    if mutation == "manifest": entries[2]["manifest_file_sha256"] = "a" * 64
    sealed(path, manifest)
    pin = repair_receipts(root, audit_path)
    with pytest.raises(E0Error, match="checkpoint inventory"):
        audit((root, audit_path, pin))


def analysis_rows():
    rows = []
    for c in (*CONDITIONS, None):
        for s in (STEPS if c else (None,)):
            for probe in PROBES:
                rows.append({"model_role": "student" if c else "teacher", "condition": c, "step": s,
                    "probe_id": probe, "scope": [0], **{m: 0. if c is None else .5 for m in CHANNELS}})
    return rows


def recipe():
    return seal_payload({"kind": "e0-fingerprint-recipe-v1", "status": "exploratory",
        "rationale": "synthetic independent unit scale", "discovery_description": "synthetic fixture only",
        "components": [{"probe": "q_bias_all", "metric": "delta_ce_nats", "scale": 2.}]})


def test_no_default_composite_or_pseudoreplication():
    result = analyze(analysis_rows())
    assert result["composite_status"] == "insufficient" and result["distances"] == []
    assert len(result["comparisons"]) == 1750 and len(result["component_trends"]) == 350
    assert result["training_seed_count"] == 1


@pytest.mark.parametrize("values,expected", [([1, .8, .5, .3, .1], "converge"),
    ([0, .1, .2, .3, .9], "diverge"), ([.5] * 5, "unchanged")])
def test_opposite_and_null_results_are_valid(values, expected):
    rows = analysis_rows()
    for row in rows:
        if row["model_role"] == "student": row["delta_ce_nats"] = values[STEPS.index(row["step"])]
    result = analyze(rows, recipe())
    assert {x["endpoint"] for x in result["condition_conclusions"]} == {expected}
    assert result["distances"][0]["exploratory_rms_distance"] == pytest.approx(abs(values[0]) / 2)


def test_nonmonotonic_and_component_disagreement_remain_visible():
    assert trajectory_direction([.5, .1, .2]) == "nonmonotonic"
    rows = analysis_rows()
    for row in rows:
        if row["model_role"] == "student":
            row["delta_ce_nats"] = [1, .9, .8, .3, .1][STEPS.index(row["step"])]
            row["self_kl_nats"] = [0, .1, .2, .8, 1][STEPS.index(row["step"])]
    trends = analyze(rows)["component_trends"]
    assert {t["endpoint_direction"] for t in trends} == {"toward_teacher", "away_from_teacher", "unchanged"}


@pytest.mark.parametrize("bad", [0, -1, True, float("nan"), float("inf")])
def test_invalid_recipe_scale(bad):
    payload = recipe()["payload"]
    payload["components"][0]["scale"] = bad
    if not math.isfinite(bad):
        with pytest.raises(ValueError): seal_payload(payload)
    else:
        with pytest.raises(E0Error): fingerprint_recipe(seal_payload(payload))


def test_missing_duplicate_or_nonfinite_analysis_rejected():
    rows = analysis_rows()
    for malformed in (rows[:-1], rows[:-1] + [rows[0]]):
        with pytest.raises(E0Error): analyze(malformed)
    rows[0]["delta_ce_nats"] = float("nan")
    with pytest.raises(E0Error): analyze(rows)


def test_source_mutation_during_progress_is_rejected(source):
    record = next((source[0] / "records").glob("*.json"))
    def change(_event): record.write_bytes(record.read_bytes() + b" ")
    with pytest.raises(E0Error, match="changed during audit"):
        audit(source, progress=change)


def dummy_plot(_rows, output):
    (output / "E0_ROUTE_TRAJECTORIES.svg").write_text('<svg xmlns="http://www.w3.org/2000/svg"></svg>')
    (output / "E0_ROUTE_TRAJECTORIES.png").write_bytes(b"fixture image")


def test_artifact_bundle_roundtrip_and_source_preservation(source, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "plot_routes", dummy_plot)
    before = {str(p): digest_file(p) for p in source[0].rglob("*") if p.is_file()}
    output = tmp_path / "outputs" / "e0"
    receipt = runner.run(root=source[0], independent_audit=source[1], expected_audit_sha256=source[2],
        output=output, engineering_fixture=True)
    assert verify_bundle(output) == receipt
    assert receipt["engineering_only"] is True and receipt["scientific_record_reanalysis"] is False
    assert (output / "E0_S4_ROUTE_TRAJECTORIES.csv").read_text().count("\n") == 361
    assert (output / "E0_TEACHER_STUDENT_FINGERPRINTS.csv").read_text().count("\n") == 1751
    assert {str(p): digest_file(p) for p in source[0].rglob("*") if p.is_file()} == before
    with pytest.raises(E0Error, match="new directory"):
        runner.run(root=source[0], independent_audit=source[1], expected_audit_sha256=source[2], output=output)
    (output / "E0_S4_ROUTE_TRAJECTORIES.csv").write_text("corrupted")
    with pytest.raises(E0Error, match="hash"): verify_bundle(output)


@pytest.mark.parametrize("inside", ["source", "repository"])
def test_output_cannot_overlap_source_or_checkout(source, inside):
    output = source[0] / "new" if inside == "source" else runner.REPO / "new"
    with pytest.raises(E0Error, match="outside|separate"):
        runner.run(root=source[0], independent_audit=source[1], expected_audit_sha256=source[2], output=output)
    assert not output.exists()


def test_failure_preserves_receipt_without_complete_marker(source, tmp_path, monkeypatch):
    def fail(_rows, _output): raise OSError("injected disk failure")
    monkeypatch.setattr(runner, "plot_routes", fail)
    output = tmp_path / "output"
    with pytest.raises(OSError):
        runner.run(root=source[0], independent_audit=source[1], expected_audit_sha256=source[2], output=output, engineering_fixture=True)
    assert (output / "FAILED.json").exists() and not (output / "COMPLETE.json").exists()
    assert read_sealed(output / "FAILED.json")[0]["error_type"] == "OSError"


def test_missing_artifact_is_blocked_not_passed(tmp_path):
    output = tmp_path / "output"
    with pytest.raises(FileNotFoundError):
        runner.run(root=tmp_path / "missing", independent_audit=tmp_path / "audits" / "absent.json",
            expected_audit_sha256="a" * 64, output=output)
    assert (output / "BLOCKED.json").exists() and not (output / "COMPLETE.json").exists()


def test_frozen_panel_membership_and_relocation(source, tmp_path):
    manifest, _ = read_sealed(source[0] / "S4_RUN_MANIFEST.json")
    original = Path(manifest["panel"]["panel_path"])
    relocated = tmp_path / "relocated" / "panel.json"
    relocated.parent.mkdir()
    relocated.write_bytes(original.read_bytes())
    original.unlink()
    with pytest.raises(FileNotFoundError): audit(source)
    rows, provenance = audit(source, panel_manifest=relocated)
    assert len(rows) == 360 and provenance["frozen_panel_path"] == str(relocated)
    panel, _ = read_sealed(relocated)
    panel["owt_full300"] = ["unexpected1", "unexpected2"]
    sealed(relocated, panel)
    with pytest.raises(E0Error, match="panel binding"): audit(source, panel_manifest=relocated)


def test_ordered_panel_binding_checked_even_if_resealed(source):
    root, audit_path, _ = source
    manifest, _ = read_sealed(root / "S4_RUN_MANIFEST.json")
    manifest["panel"]["panel_item_ids_sha256"] = "0" * 64
    sealed(root / "S4_RUN_MANIFEST.json", manifest)
    with pytest.raises(E0Error, match="ordered frozen"):
        audit((root, audit_path, repair_receipts(root, audit_path)))


def s5_fixture(path, provenance):
    path.mkdir()
    records = []
    manifest = provenance["run_manifest"]
    panel, _ = read_sealed(Path(manifest["panel"]["panel_path"]))
    for c in ("C1", "C2", "C5", "C6"):
        identity = manifest["source_inventory"]["sources"][c]["identity"]
        for name, steps in (("owt_dense64", range(0, 10001, 100)),
                ("owt_full300", (0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000))):
            for step in steps:
                records.append({"condition": c, "panel": name, "step": step, "status": "complete", "study": "S1",
                    "seed": 0, "protocol_sha256": identity["protocol_hash"], "run_id": identity["run_id"],
                    "gpu_uuid": identity["gpu_uuid"], "initialization_sha256": identity["init_hash"],
                    "data_sha256": identity["data_hash"], "layer_scope": list(range(24)), "precision": "bf16",
                    "metric_version": "e6a-v2-metrics-1", "panel_sha256": manifest["panel"]["panel_manifest_sha256"],
                    "item_ids": panel["owt_full300"], "source_aggregate_sha256": "a" * 64, "checkpoint_sha256": "b" * 64,
                    "measures": {"clean_ce_nats": 3., "sink_s": .3, "delete_delta_ce_nats": .01,
                        "delete_self_kl_nats": .02, "relocate_delta_ce_nats": -.01, "relocate_self_kl_nats": .03}})
    (path / "S5_REAGGREGATED_SOURCE_ROWS.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    for name in ("S5_JOINED_OWT_DENSE64.json", "S5_JOINED_OWT_FULL300.json", "S5_SOURCE_VERIFICATION.json"):
        (path / name).write_text("{}")
    (path / "SHA256SUMS.txt").write_text("fixture checksums")
    bindings = {"source_rows_sha256": "S5_REAGGREGATED_SOURCE_ROWS.jsonl", "dense_join_sha256": "S5_JOINED_OWT_DENSE64.json",
        "full300_join_sha256": "S5_JOINED_OWT_FULL300.json", "source_verification_sha256": "S5_SOURCE_VERIFICATION.json",
        "sha256sums_sha256": "SHA256SUMS.txt"}
    audit = {"status": "COMPLETE", "source_row_count": 440, "joined_row_count": 110, "new_inference": False,
        "model_loaded": False, "source_run_directories_modified": False, "d26_sha256": "d" * 64,
        **{key: digest_file(path / name) for key, name in bindings.items()}}
    (path / "STAGE08_S5_AUDIT.json").write_text(json.dumps(audit))
    return path, digest_file(path / "STAGE08_S5_AUDIT.json")


def test_context_preserves_precision_missingness_and_signed_effects(source, tmp_path):
    rows, provenance = audit(source)
    bundle, pin = s5_fixture(tmp_path / "s5", provenance)
    context, proof = trajectory_context(rows, provenance, bundle, pin)
    assert proof["source_rows_verified"] == 440 and proof["joined_context_states"] == 20
    assert len(context) == 35
    available = [r for r in context if r["context_status"] == "available"]
    assert len(available) == 20
    assert all(r["s4_precision"] == "fp32" and r["s5_precision"] == "bf16" for r in available)
    assert all(r["s5_relocate_delta_ce_nats"] == -.01 for r in available)
    assert {r["condition"] for r in context if r["s5_precision"] is None} == {"C0", "C3", "C4"}
    assert all(r["context_status"] == "not_supplied" for r in trajectory_context(rows, provenance)[0])


@pytest.mark.parametrize("field,value", [("seed", 1), ("precision", "fp32"), ("run_id", "wrong"),
    ("gpu_uuid", "wrong"), ("protocol_sha256", "0" * 64), ("item_ids", ["wrong1", "wrong2"])])
def test_resealed_context_identity_rejected(source, tmp_path, field, value):
    rows, provenance = audit(source)
    bundle, _ = s5_fixture(tmp_path / "s5", provenance)
    path = bundle / "S5_REAGGREGATED_SOURCE_ROWS.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(r for r in records if r["panel"] == "owt_full300")
    row[field] = value
    path.write_text("".join(json.dumps(r) + "\n" for r in records))
    audit_path = bundle / "STAGE08_S5_AUDIT.json"
    receipt = json.loads(audit_path.read_text())
    receipt["source_rows_sha256"] = digest_file(path)
    audit_path.write_text(json.dumps(receipt))
    with pytest.raises(E0Error): trajectory_context(rows, provenance, bundle, digest_file(audit_path))


def test_bundle_resealed_csv_content_mismatch(source, tmp_path, monkeypatch):
    monkeypatch.setattr(runner, "plot_routes", dummy_plot)
    output = tmp_path / "output"
    receipt = runner.run(root=source[0], independent_audit=source[1], expected_audit_sha256=source[2], output=output,
        engineering_fixture=True)
    path = output / "E0_S4_ROUTE_TRAJECTORIES.csv"
    path.write_text("resealed but scientifically inconsistent table")
    receipt["files"][path.name] = digest_file(path)
    sealed(output / "COMPLETE.json", receipt)
    with pytest.raises(E0Error, match="CSV semantic"): verify_bundle(output)
