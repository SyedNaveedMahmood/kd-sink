from pathlib import Path

import pytest

from scripts import stage08_scientific_common as common


def _mock_preflight_dependencies(tmp_path, monkeypatch, conditions):
    runs_root = tmp_path / "runs"
    selected = set(conditions)
    run_paths = {}
    for index in range(7):
        condition = f"C{index}"
        path = runs_root / f"s1-{condition.lower()}-seed0-rtx4080super"
        run_paths[condition] = path
        if condition in selected:
            (path / "checkpoints" / "weights-000000").mkdir(parents=True)
            (path / "checkpoints" / "weights-000000" / "model.safetensors").write_bytes(b"model")

    monkeypatch.setattr(common, "source_run_directories", lambda: run_paths)
    monkeypatch.setattr(common, "read_json", lambda path: {"sha256": common.D24_SHA256})
    monkeypatch.setattr(common, "validate_followup_amendment",
                        lambda doc: {"interpretation": "seed0 only"})
    monkeypatch.setattr(common, "verify_training_log",
                        lambda *args, **kwargs: {"status": "complete", "last_update": 10000})
    monkeypatch.setattr(common, "admit_s1_followup", lambda *args, **kwargs: {"status": "allowed"})
    monkeypatch.setattr(common, "sha256", lambda path: "d" * 64)
    protocol_hash = "f" * 64
    hardware_hash = "a" * 64
    calibration_hash = "b" * 64
    protocol_payload = {"study": "S1", "hardware_lock_digest": hardware_hash,
        "calibration_lock_digest": calibration_hash,
        "condition_variants": {condition: {"objective": condition} for condition in common.CONDITIONS}}

    def read_json(path):
        path = Path(path)
        if path.name == "manifest.json" and path.parent.name == "final-010000":
            run_id = path.parents[2].name
            condition = run_id.split("-")[1].upper()
            identity = {"study": "S1", "condition": condition, "seed": 0,
                "run_id": run_id, "protocol_hash": protocol_hash,
                "hardware_hash": hardware_hash, "calibration_hash": calibration_hash}
            return {"identity": identity}
        return {"sha256": common.D24_SHA256}

    monkeypatch.setattr(common, "read_json", read_json)
    monkeypatch.setattr(common, "verify_envelope", lambda doc: (protocol_payload, protocol_hash))
    calls = []

    def verify_checkpoint(path, *, identity, step):
        condition = identity["condition"]
        calls.append((condition, step))
        model_path = Path(path) / "model.safetensors"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        if not model_path.exists():
            model_path.write_bytes(b"model")
        return {"kind": "weights", "identity": identity,
                "payload_sha256": {"model.safetensors": "c" * 64}}

    monkeypatch.setattr(common, "verify_required_checkpoint", verify_checkpoint)
    return runs_root, calls


@pytest.mark.parametrize("requested,expected", [
    (("C1", "C2", "C5", "C6"), {"C1", "C2", "C5", "C6"}),
    (common.CONDITIONS, set(common.CONDITIONS)),
])
def test_source_preflight_only_reads_the_selected_condition_set(
        tmp_path, monkeypatch, requested, expected):
    runs_root, calls = _mock_preflight_dependencies(tmp_path, monkeypatch, requested)
    result = common.preflight_sources(steps=(0,), study="S7" if len(expected) == 4 else "S4",
                                     runs_root=runs_root, conditions=requested)
    assert set(result["sources"]) == expected
    assert {condition for condition, _ in calls} == expected
    assert len(calls) == len(expected)
    assert not (runs_root / "s1-c0-seed0-rtx4080super").exists() or "C0" in expected


def test_unrelated_missing_s1_runs_do_not_block_s7_preflight(tmp_path, monkeypatch):
    requested = ("C1", "C2", "C5", "C6")
    runs_root, calls = _mock_preflight_dependencies(tmp_path, monkeypatch, requested)
    result = common.preflight_sources(steps=(0,), study="S7", runs_root=runs_root,
                                      conditions=requested)
    assert set(result["sources"]) == set(requested)
    assert len(calls) == 4
    for condition in ("C0", "C3", "C4"):
        assert not (runs_root / f"s1-{condition.lower()}-seed0-rtx4080super").exists()


@pytest.mark.parametrize("conditions", [(), ("C1", "C1"), ("C7",)])
def test_source_preflight_rejects_empty_duplicate_or_unknown_conditions(conditions):
    with pytest.raises(ValueError, match="nonempty unique known"):
        common.preflight_sources(steps=(0,), study="S7", conditions=conditions)
