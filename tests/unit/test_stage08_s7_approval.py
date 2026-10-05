import copy
import json
from pathlib import Path

import pytest

from scripts import run_stage08_s4, run_stage08_s7_supplement
from sinklab.provenance import seal_payload
from sinklab.s7_utility import S7Error


def test_exact_prospective_s7_lock_is_accepted():
    path = Path("protocols/s7_utility_analysis_approved.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    approved = run_stage08_s7_supplement.validate_analysis_lock(document)
    assert approved["status"] == "approved_prospective"
    assert approved["conditions"] == ["C1", "C2", "C5", "C6"]
    assert approved["training_seed"] == 0
    assert approved["steps"] == [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]


@pytest.mark.parametrize("field,value", [
    ("training_seed", 1),
    ("conditions", ["C1", "C2", "C5"]),
    ("steps", [0, 100, 250, 500, 1000, 2000, 5000, 7500]),
    ("panel", "owt_dense64"),
    ("precision", "bf16"),
    ("device_uuid", "GPU-other"),
    ("D24_sha256", "0" * 64),
])
def test_self_consistent_but_different_s7_lock_fails_closed(tmp_path, monkeypatch, field, value):
    path = Path("protocols/s7_utility_analysis_approved.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    changed = copy.deepcopy(document["payload"])
    changed[field] = value
    forged = seal_payload(changed)
    local_lock = tmp_path / "s7_lock.json"
    local_lock.write_text(json.dumps(forged), encoding="utf-8")
    monkeypatch.setattr(run_stage08_s7_supplement, "APPROVED_LOCK_PATH", local_lock)
    with pytest.raises(S7Error, match="exact researcher-approved"):
        run_stage08_s7_supplement.validate_analysis_lock(forged)


def test_corrupt_s7_lock_envelope_fails_closed(tmp_path, monkeypatch):
    path = Path("protocols/s7_utility_analysis_approved.json")
    document = json.loads(path.read_text(encoding="utf-8"))
    document["sha256"] = "0" * 64
    local_lock = tmp_path / "s7_lock.json"
    local_lock.write_text(json.dumps(document), encoding="utf-8")
    monkeypatch.setattr(run_stage08_s7_supplement, "APPROVED_LOCK_PATH", local_lock)
    with pytest.raises(ValueError):
        run_stage08_s7_supplement.validate_analysis_lock(document)


def test_source_roots_are_bound_to_approved_lock(tmp_path):
    document = json.loads(Path("protocols/s7_utility_analysis_approved.json").read_text())
    lock = run_stage08_s7_supplement.validate_analysis_lock(document)
    roots = {key: Path(value) for key, value in lock["source_roots"].items()}
    run_stage08_s7_supplement.validate_source_roots(lock,
        runs_root=roots["s1_runs"], s5_bundle=roots["s5_bundle"],
        artifact_root=roots["artifact_root"])
    with pytest.raises(S7Error, match="approved source root"):
        run_stage08_s7_supplement.validate_source_roots(lock,
            runs_root=tmp_path, s5_bundle=roots["s5_bundle"],
            artifact_root=roots["artifact_root"])


def test_s7_teacher_receipt_is_not_mislabeled_as_s4():
    teacher = {"id": "teacher", "revision": "rev", "weights_sha256": "a" * 64,
               "config_sha256": "b" * 64}
    gpu = {"name": "NVIDIA GeForce RTX 4080 SUPER", "uuid": "GPU-authorized"}
    s4 = run_stage08_s4._teacher_provenance(teacher, gpu, study="S4")
    s7 = run_stage08_s4._teacher_provenance(teacher, gpu, study="S7")
    assert s4["followup_study"] == "S4"
    assert "control_seed" in s4 and "responsiveness_floor" in s4
    assert s7["followup_study"] == "S7"
    assert "control_seed" not in s7
    assert "denominator_floor" not in s7
    assert "responsiveness_floor" not in s7
