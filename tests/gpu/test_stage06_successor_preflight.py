"""Validate the sealed successor root on NaveedPC without starting a job."""

from argparse import Namespace
from pathlib import Path

import pytest

from sinklab.hardware import HardwareError, authorize_production_device
from sinklab.stage06_readiness import validate_final_lock_set
from sinklab.training_entry import preflight_approved_training


ROOT = Path(__file__).resolve().parents[2]
ARTIFACTS = Path(r"D:\KD-SINK-stage06-production")


def test_naveed_successor_c1_c2_c5_preflight(selected_cuda, gpu_evidence):
    locks = validate_final_lock_set(ROOT / "protocols")
    assert locks["protocol"]["payload"]["protocol"]["production_config_binding_schema"] == 3
    results = {}
    for condition in ("C1", "C2", "C5"):
        args = Namespace(protocol_lock=ROOT / "protocols/protocol.lock.json",
                         hardware_plan=ROOT / "protocols/hardware.lock.json",
                         config=ROOT / f"configs/production/s1/{condition.lower()}_rtx4080super.json",
                         seed=0, artifact_root=ARTIFACTS if condition == "C1" else None)
        result = preflight_approved_training(args)
        assert result["gpu_uuid"] == "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0"
        assert result["gpu_name"] == "NVIDIA GeForce RTX 4080 SUPER"
        assert result["training_started"] is False
        results[condition] = {"authorized": True, "uuid": result["gpu_uuid"],
                              "artifacts_verified": result.get("scientific_artifacts_verified", False)}
    assert results["C1"]["artifacts_verified"] is True
    gpu_evidence[0]["measurements"]["successor_root_preflight"] = {
        "root": locks["protocol"]["sha256"], "conditions": results,
        "training_started": False,
    }


def test_naveed_successor_c3_c4_refused(selected_cuda, gpu_evidence):
    locks = validate_final_lock_set(ROOT / "protocols")
    protocol = locks["protocol"]["payload"]
    hardware = locks["hardware"]["payload"]
    actual_uuid = "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0"
    assert any(actual_uuid in row for row in gpu_evidence[0]["nvidia_smi"])
    refused = {}
    for condition in ("C3", "C4"):
        with pytest.raises(HardwareError, match="condition/device role"):
            authorize_production_device(protocol, hardware, condition=condition,
                                        device_role="rtx4080super",
                                        gpu_name="NVIDIA GeForce RTX 4080 SUPER",
                                        gpu_uuid=actual_uuid)
        refused[condition] = True
    gpu_evidence[0]["measurements"]["successor_root_refusal"] = {
        "root": locks["protocol"]["sha256"], "conditions": refused,
        "training_started": False,
    }
