"""D20 is an exact measured successor; D19 remains independently historical."""

import copy
import json
from pathlib import Path

import pytest

from sinklab.checkpoint import CheckpointError, save_checkpoint, verify_checkpoint
from sinklab.stage06_artifacts import approved_artifact_plan_digest
from sinklab.config import resolve_config
from sinklab.hardware import HardwareError, authorize_production_device
from sinklab.provenance import LockError, canonical_json_bytes, validate_protocol_lock
from sinklab.stage06_readiness import (
    D19_PROTOCOL_ROOT, ReadinessError, _d20, build_d20_lock_set,
    build_production_configs, validate_final_lock_set,
)


ROOT = Path(__file__).resolve().parents[2]
NODI = "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee"
MODEL = "NVIDIA GeForce RTX 4080 SUPER"
LOCKS = ("artifact", "environment", "hardware", "calibration", "protocol")


@pytest.fixture(scope="module")
def d20():
    old = {name: json.loads((ROOT / "protocols" / f"{name}.lock.json").read_text())
           for name in LOCKS}
    old["protocol"] = json.loads((ROOT / "protocols/superseded" /
        f"s1-protocol-{D19_PROTOCOL_ROOT}.json").read_text())
    old["hardware"] = json.loads((ROOT / "protocols/superseded" /
        f"s1-hardware-{old['protocol']['payload']['hardware_lock_digest']}.json").read_text())
    docs, plan = build_d20_lock_set(ROOT, old, "f" * 40)
    configs, production_plan = build_production_configs(
        ROOT, docs["protocol"], reviewed_plan=plan)
    return old, docs, plan, configs, production_plan


def test_d20_exact_predecessor_science_and_nine_job_assignment(d20):
    old, docs, plan, configs, production = d20
    before, old_digest = validate_protocol_lock(old["protocol"])
    after, _ = validate_protocol_lock(docs["protocol"])
    assert old_digest == D19_PROTOCOL_ROOT
    assert after["protocol"]["predecessor_protocol_root_sha256"] == old_digest
    assert after["approval"]["decision_ids"] == [*before["approval"]["decision_ids"], "D20"]
    assert after["artifact_lock_digest"] == before["artifact_lock_digest"]
    assert after["environment_lock_digest"] == before["environment_lock_digest"]
    assert after["calibration_lock_digest"] == before["calibration_lock_digest"]
    for field in ("teacher", "student", "data", "training", "objectives", "evaluation",
                  "checkpointing", "calibration", "optional_studies"):
        assert after["protocol"][field] == before["protocol"][field]
    assert after["protocol"]["objectives"]["s_MSE"] == 68.00580071126464
    assert after["protocol"]["training"]["microbatch"] == 4
    assert after["protocol"]["training"]["accumulation"] == 16
    assert len(plan["jobs"]) == len(production["jobs"]) == len(configs) == 9
    assert plan["hardware_confounds"] == production["hardware_confounds"] == ["C4-C3-seed0"]
    assert len({(row["run_id"].split("-")[1], row["seed"]) for row in plan["jobs"]}) == 7
    assert "s1-c3-seed0-rtx4080super" in [row["run_id"] for row in plan["jobs"]]
    assert "s1-c3-seed0-rtx3090" not in [row["run_id"] for row in plan["jobs"]]
    assert all(row in plan["jobs"] for row in before["protocol"]["run_plan"]["jobs"]
               if row["run_id"].startswith(("s1-c1-", "s1-c2-", "s1-c4-")))
    assert (docs["hardware"]["payload"]["condition_device_eligibility"]["C3"]
            ["rtx3090"]["measured_production_eligible"] is True)


def test_d20_real_evidence_and_failed_mb8_are_bound(d20):
    decision, decision_sha, evidence, evidence_sha = _d20(ROOT)
    old, docs, _, _, _ = d20
    assert decision["reference_qualification_sha256"] == evidence_sha
    assert docs["protocol"]["payload"]["approval"]["d20_sha256"] == decision_sha
    assert evidence["profiles"]["mb8"]["profile"]["passed"] is False
    assert evidence["profiles"]["mb4"]["profile"]["passed"] is True
    assert evidence["profiles"]["mb4"]["profile"]["free_bytes"] >= 1717095628
    assert docs["hardware"]["payload"]["proof"] == old["hardware"]["payload"]["proof"]


@pytest.mark.parametrize("condition", ("C0", "C1", "C2", "C3", "C5", "C6"))
@pytest.mark.parametrize("uuid", (NODI, "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0",
                                    "GPU-new-exact-model"))
def test_d20_exact_model_fresh_uuid_and_config(d20, condition, uuid):
    _, docs, _, configs, _ = d20
    identity = authorize_production_device(
        docs["protocol"]["payload"], docs["hardware"]["payload"],
        condition=condition, device_role="rtx4080super", gpu_name=MODEL, gpu_uuid=uuid)
    assert identity["gpu_uuid"] == uuid
    config = configs[f"production/s1/{condition.lower()}_rtx4080super.json"]
    assert config["production_binding"]["hardware_binding"]["actual_uuid"] == "runtime_recorded"
    resolve_config(config, seed=0, protocol_lock=docs["protocol"], production=True)


def test_d20_c4_wrong_model_and_same_uuid_resume(d20, tmp_path):
    import torch
    _, docs, _, configs, _ = d20
    for condition, name in (("C4", MODEL), ("C3", "NVIDIA GeForce RTX 4090")):
        with pytest.raises(HardwareError):
            authorize_production_device(docs["protocol"]["payload"],
                docs["hardware"]["payload"], condition=condition,
                device_role="rtx4080super", gpu_name=name, gpu_uuid=NODI)
    cfg = configs["production/s1/c4_rtx3090.json"]
    resolve_config(cfg, seed=0, protocol_lock=docs["protocol"], production=True)
    model = torch.nn.Linear(2, 2)
    identity = {"run_id": "s1-c3-seed0-rtx4080super", "gpu_uuid": NODI}
    path = save_checkpoint(tmp_path, step=0, kind="rolling", model=model,
                           state={"optimizer": {}}, identity=identity)
    assert verify_checkpoint(path, identity=identity)["identity"]["gpu_uuid"] == NODI
    with pytest.raises(CheckpointError, match="identity mismatch"):
        verify_checkpoint(path, identity={**identity, "gpu_uuid": "GPU-other"})


def test_d20_missing_or_altered_evidence_refused(tmp_path):
    files = ["protocols/s1_researcher_amendment_d19_4080_class_20260928.json",
             "protocols/s1_researcher_amendment_d20_c3_4080_20260929.json",
             "protocols/artifact.lock.json", "protocols/environment.lock.json",
             "reports/stage06_second_4080_qualification.json",
             "reports/stage06_reviewed_batch_candidate.json",
             "reports/stage06_production_artifact_inventory.json",
             "reports/stage06_c3_4080_qualification.json",
             "reports/stage06_c3_4080_mb8_raw.json",
             "reports/stage06_c3_4080_mb4_raw.json"]
    for name in files:
        dest = tmp_path / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes((ROOT / name).read_bytes())
    _d20(tmp_path)
    raw = tmp_path / "reports/stage06_c3_4080_mb4_raw.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(ReadinessError, match="raw profile"):
        _d20(tmp_path)
    raw.unlink()
    with pytest.raises(FileNotFoundError):
        _d20(tmp_path)


def test_artifact_inventory_keeps_original_approved_plan_after_d20(tmp_path):
    from sinklab.provenance import payload_digest
    original = {"jobs": ["historically-approved"]}
    prospective = {"jobs": ["D20"]}
    current = tmp_path / "configs/s1_jobs_seed0_reviewed.json"
    current.parent.mkdir(parents=True)
    current.write_text(json.dumps(original), encoding="utf-8")
    approval = {"reviewed_seed0_plan_sha256": payload_digest(original)}
    assert approved_artifact_plan_digest(tmp_path, approval) == payload_digest(original)
    archive = tmp_path / "configs/superseded/d19/s1_jobs_seed0_reviewed.json"
    archive.parent.mkdir(parents=True)
    archive.write_text(json.dumps(original), encoding="utf-8")
    current.write_text(json.dumps(prospective), encoding="utf-8")
    assert approved_artifact_plan_digest(tmp_path, approval) == payload_digest(original)
    archive.write_text(json.dumps(prospective), encoding="utf-8")
    with pytest.raises(ValueError, match="approved artifact plan"):
        approved_artifact_plan_digest(tmp_path, approval)


def test_d20_forged_approval_and_predecessor_refused(d20):
    old, docs, _, _, _ = d20
    bad = copy.deepcopy(docs["protocol"])
    bad["payload"]["approval"]["decision_ids"].remove("D20")
    from sinklab.provenance import seal_payload
    bad = seal_payload(bad["payload"])
    with pytest.raises(LockError):
        validate_protocol_lock(bad)
    bad_old = copy.deepcopy(old)
    bad_old["protocol"]["sha256"] = "0" * 64
    with pytest.raises((ReadinessError, LockError)):
        build_d20_lock_set(ROOT, bad_old, "f" * 40)
