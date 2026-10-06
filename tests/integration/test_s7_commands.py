import json
import sys
from pathlib import Path

import pytest

from scripts import run_stage08_s7
from scripts.run_stage08_s7_supplement import (_supplement_progress_event,
                                               validate_analysis_lock)
from sinklab.provenance import seal_payload
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "unit"))
from test_s7_utility import _bundle, _sha


def test_supplement_progress_reports_allocated_and_reserved_cuda_memory():
    event = _supplement_progress_event(condition="C1", step=100, run_id="run-c1",
        gpu_uuid="GPU-approved", completed_items=10, total_items=300, item_id="item-10",
        elapsed_seconds=12.5, eta_seconds=362.5, cuda_allocated_bytes=1000,
        cuda_reserved_bytes=2000)
    assert event["event"] == "s7_supplement_progress"
    assert event["cuda_allocated_bytes"] == 1000
    assert event["cuda_reserved_bytes"] == 2000
    assert (event["condition"], event["step"], event["completed_items"],
            event["total_items"], event["gpu_uuid"]) == ("C1", 100, 10, 300, "GPU-approved")


def test_analysis_command_writes_traceable_external_result(tmp_path, monkeypatch):
    source, _ = _bundle(tmp_path)
    monkeypatch.setattr(run_stage08_s7, "S5_AUDIT_SHA256", _sha(source / "STAGE08_S5_AUDIT.json"))
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    output = tmp_path / "s7"
    audit = run_stage08_s7.run(s5_bundle=source, output=output)
    assert audit["status"] == "descriptive_s5_reuse_only"
    assert audit["science_complete"] is False
    assert audit["science_complete_scope"] is None
    assert audit["authorized_supplement_count"] == 0
    assert len(audit["analysis_source_commit"]) == 40
    assert audit["s7_analysis_sha256"] == _sha(output / "S7_ANALYSIS.json")
    assert json.loads((output / "S7_ANALYSIS.json").read_text())["training_seed_count"] == 1
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before
    with pytest.raises(FileExistsError):
        run_stage08_s7.run(s5_bundle=source, output=output)


def test_supplemented_analysis_audit_scopes_completion_to_approved_grid(tmp_path, monkeypatch):
    source, _ = _bundle(tmp_path)
    monkeypatch.setattr(run_stage08_s7, "S5_AUDIT_SHA256", _sha(source / "STAGE08_S5_AUDIT.json"))
    runs = tmp_path / "runs"
    artifacts = tmp_path / "artifacts"
    supplements = tmp_path / "supplements"
    for path in (runs, artifacts, supplements):
        path.mkdir()
    lock = tmp_path / "approved-lock.json"
    lock.write_text("{}", encoding="utf-8")
    lock_sha = "f" * 64
    monkeypatch.setattr(run_stage08_s7, "validate_analysis_lock",
        lambda _document: {"sha256": lock_sha, "device_model": "RTX 4080 SUPER",
                           "device_uuid": "GPU-approved", "precision": "fp32",
                           "D24_sha256": "d" * 64})
    monkeypatch.setattr(run_stage08_s7, "validate_source_roots", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(run_stage08_s7, "read_extended_s1_records", lambda **_kwargs: [])
    monkeypatch.setattr(run_stage08_s7, "add_extended_s1_records", lambda result, _rows: result)
    monkeypatch.setattr(run_stage08_s7, "read_lm2000_records", lambda **_kwargs: [])
    monkeypatch.setattr(run_stage08_s7, "add_lm2000_endpoints", lambda result, _rows: result)

    def mark_supplements(result, _panels, _records, **_kwargs):
        result = dict(result)
        result["status"] = "descriptive_s5_plus_clean_supplements"
        result["s7_campaign"] = {"status": "complete", "verified_supplement_count": 36}
        return result
    monkeypatch.setattr(run_stage08_s7, "add_clean_supplements", mark_supplements)
    for condition in ("C1", "C2", "C5", "C6"):
        for step in (0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000):
            path = supplements / f"S7_{condition}_STEP{step:06d}_FULL300.json"
            path.write_text(json.dumps({"analysis_lock_sha256": lock_sha}), encoding="utf-8")

    audit = run_stage08_s7.run(s5_bundle=source, output=tmp_path / "supplemented",
        runs_root=runs, artifact_root=artifacts, supplements_dir=supplements,
        analysis_lock=lock)
    assert audit["status"] == "descriptive_s5_plus_clean_supplements"
    assert audit["science_complete"] is True
    assert audit["science_complete_scope"] == "authorized seed0 clean-attention Full300 supplement grid only"
    assert audit["authorized_supplement_count"] == 36
    assert len(audit["analysis_source_commit"]) == 40


def test_unapproved_draft_and_forged_lock_cannot_launch_new_s7_inference(tmp_path, monkeypatch):
    draft = json.loads(Path("protocols/s7_utility_analysis_draft.json").read_text())
    with pytest.raises(ValueError):
        validate_analysis_lock(draft)
    approved = json.loads(Path("protocols/s7_utility_analysis_approved.json").read_text())
    forged_payload = {**approved["payload"], "training_seed": 2}
    forged = seal_payload(forged_payload)
    fake_checked_in = tmp_path / "forged.json"
    fake_checked_in.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr("scripts.run_stage08_s7_supplement.APPROVED_LOCK_PATH", fake_checked_in)
    with pytest.raises(ValueError, match="exact researcher-approved"):
        validate_analysis_lock(forged)
