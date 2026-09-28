"""Real NaveedPC GPU identity against prospective D19 class policy; no training."""

from pathlib import Path

import pytest

from sinklab.hardware import HardwareError, authorize_production_device
from sinklab.stage06_readiness import build_protocol_lock, build_successor_component_locks
from sinklab.training_entry import _check_runtime_environment, _gpu_metadata


ROOT = Path(__file__).resolve().parents[2]


def test_actual_4080_matches_d19_class_without_relabeling_profiles(selected_cuda, gpu_evidence):
    components = build_successor_component_locks(ROOT)
    protocol = build_protocol_lock(ROOT, components, fingerprint_denominator_floor=1e-8,
                                   production_runtime_source_commit="f" * 40)
    gpu = _gpu_metadata()
    assert gpu["name"] == "NVIDIA GeForce RTX 4080 SUPER"
    assert gpu["uuid"] == "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0"
    _check_runtime_environment(components["environment"]["payload"], "rtx4080super", gpu)
    for condition in ("C1", "C2", "C5"):
        accepted = authorize_production_device(protocol["payload"], components["hardware"]["payload"],
            condition=condition, device_role="rtx4080super", gpu_name=gpu["name"],
            gpu_uuid=gpu["uuid"])
        assert accepted["gpu_uuid"] == gpu["uuid"]
    for condition in ("C3", "C4"):
        with pytest.raises(HardwareError, match="absent from approved protocol"):
            authorize_production_device(protocol["payload"], components["hardware"]["payload"],
                condition=condition, device_role="rtx4080super", gpu_name=gpu["name"],
                gpu_uuid=gpu["uuid"])
    gpu_evidence[0]["measurements"]["d19_class_device"] = {
        "uuid": gpu["uuid"], "model": gpu["name"],
        "fresh_authorized": ["C1", "C2", "C5"], "refused": ["C3", "C4"],
        "environment_exact": True, "training_started": False,
    }
