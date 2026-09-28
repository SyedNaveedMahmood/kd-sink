"""Sparse descending-search proof for the reviewed Stage06 hardware allocation."""

import json
import hashlib
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from sinklab.hardware import HardwareError, Profile, build_batch_plan
from sinklab.stage06_evidence import build_reviewed_seed0_candidate
from sinklab import stage06_calibration_entry
from sinklab.provenance import payload_digest, verify_envelope


ROOT = Path(__file__).resolve().parents[2]


def _candidate():
    return build_reviewed_seed0_candidate(
        ROOT / "reports/stage06_3090_profile_evidence.json",
        ROOT / "reports/stage06_4080_profile_evidence.json",
        ROOT / "configs/s1_jobs_seed0_reviewed.json", ROOT / "configs")


def _profile(condition, role, batch, safe):
    total, headroom = 20_000, 2_000
    return Profile(condition, role, role + "-uuid", batch, safe, "mock_cpu",
                   True, True, True, 10_000, 11_000,
                   4_000 if safe else 1_000, total, headroom)


def test_sparse_descending_proof_does_not_require_smaller_divisors():
    required = {("C1", "rtx3090", "rtx3090-uuid"),
                ("C2", "rtx4080super", "rtx4080super-uuid")}
    rows = [_profile("C1", "rtx3090", 16, True),
            _profile("C2", "rtx4080super", 8, False),
            _profile("C2", "rtx4080super", 4, True)]
    proof = build_batch_plan(rows, required, production=False)
    assert (proof["microbatch"], proof["accumulation"], proof["next_larger_disproved"]) == (4, 16, 8)
    assert {p["microbatch"] for p in proof["profiles"]} == {16, 8, 4}
    with pytest.raises(HardwareError, match="next larger"):
        build_batch_plan(rows[:1] + rows[2:], required, production=False)


def test_committed_real_profiles_independently_prove_reviewed_4x16():
    doc = _candidate()
    payload = doc["payload"]
    proof = payload["proof"]
    assert (proof["microbatch"], proof["accumulation"], proof["effective_sequences"],
            proof["input_tokens"], proof["shifted_targets"]) == (4, 16, 64, 8192, 8128)
    assert len(proof["required"]) == 9
    assert proof["next_larger_disproved"] == 8
    assert payload["unmeasured_not_required_for_current_plan"] == [
        ["C0", "rtx3090"], ["C5", "rtx3090"], ["C6", "rtx3090"], ["C3", "rtx4080super"]]
    assert payload["c5_4080_mb4_repeat"]["free_margin_over_required_bytes"] == 15_119_156
    matrix = payload["condition_device_eligibility"]
    assert all(matrix[f"C{i}"]["rtx3090"]["policy_allowed"] for i in range(7))
    assert all(matrix[c]["rtx4080super"]["measured_production_eligible"]
               for c in ("C0", "C1", "C2", "C5", "C6"))
    assert all(matrix[c]["rtx3090"]["current_plan_status"] ==
               "unmeasured_not_required_for_current_plan" for c in ("C0", "C5", "C6"))
    assert matrix["C3"]["rtx4080super"]["future_gate"] == "pending_4080_profile"
    assert matrix["C4"]["rtx4080super"]["current_plan_status"] == "prohibited"
    assert doc == json.loads((ROOT / "reports/stage06_reviewed_batch_candidate.json").read_text())


def test_missing_scheduled_pair_blocks_but_unused_pairs_do_not():
    proof = _candidate()["payload"]["proof"]
    profiles = [Profile(**row) for row in proof["profiles"]]
    required = {tuple(row) for row in proof["required"]}
    omitted = ("C5", "rtx4080super", proof["eligibility_matrix"]["C5"]["rtx4080super"])
    with pytest.raises(HardwareError, match="missing required"):
        build_batch_plan([p for p in profiles if (p.condition, p.device_role, p.device_uuid) != omitted],
                         required, production=True)
    assert build_batch_plan(profiles, required, production=True)["microbatch"] == 4


def test_c4_on_4080_rejected():
    row = _profile("C4", "rtx4080super", 4, True)
    with pytest.raises(HardwareError, match="REL"):
        build_batch_plan([row], {("C4", "rtx4080super", "rtx4080super-uuid")}, production=False)


def test_committed_profile_hashes_accept_lf_or_crlf_worktrees(tmp_path):
    paths = []
    for role in ("3090", "4080"):
        source = ROOT / f"reports/stage06_{role}_profile_evidence.json"
        target = tmp_path / source.name
        target.write_bytes(source.read_bytes().replace(b"\r\n", b"\n"))
        paths.append(target)
    alternate = build_reviewed_seed0_candidate(
        paths[0], paths[1], ROOT / "configs/s1_jobs_seed0_reviewed.json", ROOT / "configs")
    assert alternate == _candidate()


def test_researcher_approval_covers_registered_d01_d18_and_sealed_amendment():
    document = json.loads((ROOT / "protocols/s1_researcher_approval_20260928.json").read_text())
    approval, _ = verify_envelope(document)
    amendment, digest = verify_envelope(json.loads((
        ROOT / "protocols/s1_researcher_amendment_20260928.json").read_text()))
    assert digest == "4660503cebe710f67f508e96ebd9d6002fcd19835951e773c4fb8062e1f4a097"
    assert approval["amendment_sha256"] == digest
    assert set(approval["approved_decisions"]) == {f"D{i:02d}" for i in range(1, 19)}
    register = (ROOT / "design/e6a_v2/DECISIONS.md").read_bytes()
    assert approval["decision_register_sha256"] == hashlib.sha256(
        register.replace(b"\r\n", b"\n")).hexdigest()
    choices = {}
    for line in register.decode("utf-8").splitlines():
        match = re.match(r"^\| (D\d\d)(?: \([^)]*\))? \| ([^|]+) \|", line)
        if match:
            choices[match.group(1)] = match.group(2).strip()
    assert approval["approved_decisions"] == choices
    plan = json.loads((ROOT / "configs/s1_jobs_seed0_reviewed.json").read_text())
    assert approval["reviewed_seed0_plan_sha256"] == payload_digest(plan)
    assert approval["mandatory_seeds"] == [0] and approval["optional_seeds"] == [1, 2]
    assert approval["default_optimizer_updates"] == 10000
    assert approval["optional_s3_enabled"] is False
    assert approval["optional_512_1024_contexts_enabled"] is False


def test_real_calibration_entry_refuses_non_3090_before_model_load(monkeypatch):
    monkeypatch.setattr(stage06_calibration_entry, "verify_calibration_inputs",
                        lambda args: ({}, {}, {}))
    monkeypatch.setattr(stage06_calibration_entry.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(stage06_calibration_entry.torch.cuda, "get_device_name",
                        lambda index: "NVIDIA GeForce RTX 4080 SUPER")
    with pytest.raises(ValueError, match="approved RTX 3090"):
        stage06_calibration_entry.run_calibration(SimpleNamespace())


def test_real_artifact_inventory_binds_approval_and_frozen_windows():
    inventory, _ = verify_envelope(json.loads((
        ROOT / "reports/stage06_production_artifact_inventory.json").read_text()))
    approval = json.loads((ROOT / "protocols/s1_researcher_approval_20260928.json").read_text())
    assert inventory["researcher_approval_sha256"] == approval["sha256"]
    assert inventory["reviewed_plan_sha256"] == _candidate()["payload"]["reviewed_plan_sha256"]
    assert inventory["dataset"]["windows"] == {
        "training": [0, 400000], "evaluation": [400000, 408000],
        "calibration": [408000, 416000]}
    assert inventory["corpus"]["splits"]["training"]["packed_blocks"] > 640000
    assert inventory["panels"]["dense64"] == 64
    assert inventory["panels"]["full300"] == 300
    assert inventory["panels"]["lm2000"] == 2000
    assert inventory["calibration_export"]["effective_batches"] == 16
    assert inventory["calibration_export"]["sequences_per_batch"] == 64
    assert inventory["training_initialization"]["seed"] == 0
    assert inventory["calibration_initialization"]["seed"] == 1729
