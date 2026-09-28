"""D19 transfers reference permission while keeping each run's physical identity."""

import copy
import json
from pathlib import Path

import pytest
import torch

from sinklab.checkpoint import CheckpointError, save_checkpoint, verify_checkpoint
from sinklab.config import ConfigError, resolve_config
from sinklab.hardware import HardwareError, authorize_production_device
from sinklab.provenance import validate_protocol_lock, verify_envelope
from sinklab.stage06_readiness import (PREDECESSOR_PROTOCOL_ROOT,
    build_production_configs, build_protocol_lock, build_successor_component_locks,
    validate_scientific_unchanged)


ROOT = Path(__file__).resolve().parents[2]
NODI = "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee"
NAVEED = "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0"
ADRITA = "GPU-11111111-2222-3333-4444-555555555555"
MODEL = "NVIDIA GeForce RTX 4080 SUPER"
ALLOWED = ("C0", "C1", "C2", "C5", "C6")


@pytest.fixture(scope="module")
def successor():
    components = build_successor_component_locks(ROOT)
    protocol = build_protocol_lock(ROOT, components, fingerprint_denominator_floor=1e-8,
                                   production_runtime_source_commit="f" * 40)
    configs, plan = build_production_configs(ROOT, protocol)
    return components, protocol, configs, plan


@pytest.mark.parametrize("uuid", (NODI, NAVEED, ADRITA))
@pytest.mark.parametrize("condition", ALLOWED)
def test_exact_4080_class_accepts_fresh_reference_known_and_unseen_uuid(successor, condition, uuid):
    components, protocol, configs, _ = successor
    payload, _ = validate_protocol_lock(protocol)
    hardware, _ = verify_envelope(components["hardware"])
    identity = authorize_production_device(payload, hardware, condition=condition,
        device_role="rtx4080super", gpu_name=MODEL, gpu_uuid=uuid)
    assert identity["gpu_uuid"] == uuid and identity["condition"] == condition
    config = configs[f"production/s1/{condition.lower()}_rtx4080super.json"]
    assert config["production_binding"]["hardware_binding"]["actual_uuid"] == "runtime_recorded"
    assert "gpu_uuid" not in config["production_binding"]
    resolve_config(config, seed=0, protocol_lock=protocol, production=True)


@pytest.mark.parametrize("name", ("NVIDIA GeForce RTX 4090", "NVIDIA GeForce RTX 4070",
                                   "NVIDIA GeForce RTX 4080", "NVIDIA RTX 4080 Laptop GPU",
                                   "NVIDIA RTX A6000"))
def test_similar_or_other_gpu_models_are_refused(successor, name):
    components, protocol, _, _ = successor
    with pytest.raises(HardwareError, match="outside approved"):
        authorize_production_device(protocol["payload"], components["hardware"]["payload"],
            condition="C1", device_role="rtx4080super", gpu_name=name, gpu_uuid=ADRITA)


@pytest.mark.parametrize("condition", ("C3", "C4"))
def test_unqualified_4080_conditions_are_refused(successor, condition):
    components, protocol, _, _ = successor
    with pytest.raises(HardwareError, match="absent from approved protocol"):
        authorize_production_device(protocol["payload"], components["hardware"]["payload"],
            condition=condition, device_role="rtx4080super", gpu_name=MODEL, gpu_uuid=NAVEED)


def test_empty_uuid_wrong_role_and_wrong_3090_uuid_refused(successor):
    components, protocol, _, _ = successor
    hardware = components["hardware"]["payload"]
    for role, name, uuid in (("rtx4080super", MODEL, ""),
                             ("rtx3090", MODEL, NAVEED),
                             ("rtx3090", "NVIDIA GeForce RTX 3090", NAVEED)):
        with pytest.raises(HardwareError):
            authorize_production_device(protocol["payload"], hardware, condition="C1",
                                        device_role=role, gpu_name=name, gpu_uuid=uuid)


@pytest.mark.parametrize("condition", ("C1", "C2", "C3", "C4"))
def test_exact_3090_bridges_and_assignments_remain(successor, condition):
    components, protocol, configs, _ = successor
    uuid = "GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e"
    identity = authorize_production_device(protocol["payload"], components["hardware"]["payload"],
        condition=condition, device_role="rtx3090", gpu_name="NVIDIA GeForce RTX 3090", gpu_uuid=uuid)
    assert identity["gpu_uuid"] == uuid
    config = configs[f"production/s1/{condition.lower()}_rtx3090.json"]
    assert config["production_binding"]["gpu_uuid"] == uuid
    resolve_config(config, seed=0, protocol_lock=protocol, production=True)


def test_config_binding_and_policy_disagreement_refused(successor):
    components, protocol, configs, _ = successor
    wrong = copy.deepcopy(configs["production/s1/c1_rtx4080super.json"])
    wrong["production_binding"]["hardware_binding"]["model"] = "NVIDIA GeForce RTX 4090"
    with pytest.raises(ConfigError, match="production config binding"):
        resolve_config(wrong, seed=0, protocol_lock=protocol, production=True)
    bad_hardware = copy.deepcopy(components["hardware"]["payload"])
    bad_hardware["hardware_classes"]["rtx4080super"]["reference_uuid"] = NAVEED
    with pytest.raises(HardwareError, match="differs from sealed reference"):
        authorize_production_device(protocol["payload"], bad_hardware,
            condition="C1", device_role="rtx4080super", gpu_name=MODEL, gpu_uuid=NAVEED)


def test_run_checkpoint_identity_records_uuid_and_refuses_migration(tmp_path):
    model = torch.nn.Linear(2, 2)
    identity = {"run_id": "s1-c1-seed0-rtx4080super", "gpu_uuid": NAVEED}
    path = save_checkpoint(tmp_path, step=0, kind="rolling", model=model,
                           state={"optimizer": {}}, identity=identity)
    assert verify_checkpoint(path, identity=identity)["identity"]["gpu_uuid"] == NAVEED
    migrated = {**identity, "gpu_uuid": NODI}
    with pytest.raises(CheckpointError, match="identity mismatch"):
        verify_checkpoint(path, identity=migrated)
    started_on_nodi = {**identity, "gpu_uuid": NODI}
    path2 = save_checkpoint(tmp_path / "second", step=0, kind="rolling", model=model,
                            state={"optimizer": {}}, identity=started_on_nodi)
    with pytest.raises(CheckpointError, match="identity mismatch"):
        verify_checkpoint(path2, identity={**identity, "gpu_uuid": ADRITA})


def test_predecessor_and_failed_profiles_remain_truthful(successor):
    components, protocol, _, plan = successor
    environment = components["environment"]["payload"]["devices"]["rtx4080super"]["gpu"]
    assert environment["reference_uuid"] == NODI and "uuid" not in environment
    old = json.loads((ROOT / "protocols/protocol.lock.json").read_text(encoding="utf-8"))
    if old["sha256"] != PREDECESSOR_PROTOCOL_ROOT:
        old = json.loads((ROOT / "protocols/superseded" /
            f"s1-protocol-{PREDECESSOR_PROTOCOL_ROOT}.json").read_text(encoding="utf-8"))
    assert validate_protocol_lock(old)[1] == PREDECESSOR_PROTOCOL_ROOT
    validate_scientific_unchanged(old, protocol)
    assert len(plan["jobs"]) == 9
    d19, d19_sha = verify_envelope(json.loads((ROOT /
        "protocols/s1_researcher_amendment_d19_4080_class_20260928.json").read_text()))
    report, report_sha = verify_envelope(json.loads((ROOT /
        "reports/stage06_second_4080_qualification.json").read_text()))
    attempts = {(x["condition"], x["microbatch"]): x for x in report["profiles"]["attempts"]}
    assert d19["per_card_headroom_profile_required"] is False
    assert d19["naveed_observed_profile_report_sha256"] == report_sha
    assert components["hardware"]["payload"]["d19_researcher_amendment_sha256"] == d19_sha
    assert [(attempts[(name, 4)]["status"], attempts[(name, 4)]["free_margin_bytes"])
            for name in ("C2", "C5")] == [("complete_unsafe", -96025804),
                                           ("complete_unsafe", -228498636)]
