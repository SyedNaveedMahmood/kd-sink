"""D23 extends the existing pinned seed lineages without changing S1 science."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from sinklab.config import ConfigError, production_binding_for, resolve_config
from sinklab.hardware import HardwareError, authorize_production_device
from sinklab.provenance import validate_protocol_lock
from sinklab.stage06_readiness import (
    D22_PROTOCOL_ROOT, D23_SEED_CONDITIONS, D23_SEED_DEVICE_ROLES, LOCK_NAMES,
    build_d23_lock_set, build_optional_seed_configs,
)


ROOT = Path(__file__).resolve().parents[2]


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _predecessor() -> dict[str, dict]:
    current = _read(ROOT / "protocols/protocol.lock.json")
    protocol = (current if current["sha256"] == D22_PROTOCOL_ROOT else
                _read(ROOT / "protocols/superseded" /
                      f"s1-protocol-{D22_PROTOCOL_ROOT}.json"))
    result = {"protocol": protocol}
    for name in LOCK_NAMES[:-1]:
        lock = _read(ROOT / f"protocols/{name}.lock.json")
        expected = protocol["payload"][f"{name}_lock_digest"]
        if lock["sha256"] != expected:
            lock = _read(ROOT / "protocols/superseded" / f"s1-{name}-{expected}.json")
        result[name] = lock
    return result


def _config(condition: str, seed: int, protocol: dict) -> dict:
    role = D23_SEED_DEVICE_ROLES[condition]
    template_role = "rtx3090" if condition == "C4" else "rtx4080super"
    raw = _read(ROOT / f"configs/production/s1/{condition.lower()}_{template_role}.json")
    raw["device_role"] = role
    raw["protocol_digest"] = protocol["sha256"]
    raw["production_binding"] = production_binding_for(protocol["payload"], role, seed)
    return raw


def test_d23_adds_all_optional_condition_seed_pairs_without_recalibration():
    predecessor = _predecessor()
    assert predecessor["protocol"]["sha256"] == D22_PROTOCOL_ROOT
    successor = build_d23_lock_set(ROOT, predecessor, "1" * 40)
    payload, digest = validate_protocol_lock(successor["protocol"])
    body = payload["protocol"]
    assert digest == successor["protocol"]["sha256"]
    assert body["production_config_binding_schema"] == 6
    assert body["optional_seed_conditions"] == D23_SEED_CONDITIONS
    assert body["optional_seed_device_roles"] == D23_SEED_DEVICE_ROLES
    assert [successor[name]["sha256"] for name in ("artifact", "environment", "calibration")] == [
        predecessor[name]["sha256"] for name in ("artifact", "environment", "calibration")]
    assert successor["hardware"]["payload"]["proof"] == predecessor["hardware"]["payload"]["proof"]
    assert body["calibration"] == predecessor["protocol"]["payload"]["protocol"]["calibration"]
    assert body["objectives"] == predecessor["protocol"]["payload"]["protocol"]["objectives"]
    assert body["training"]["microbatch"] == 4 and body["training"]["accumulation"] == 16
    configs, plan = build_optional_seed_configs(ROOT, successor["protocol"])
    assert len(configs) == len(plan["jobs"]) == 14
    assert {(condition, seed) for condition in D23_SEED_CONDITIONS for seed in (1, 2)} == {
        (Path(row["config"]).name.split("_")[0].upper(), row["seed"]) for row in plan["jobs"]}
    assert plan["hardware_confounds"] == [
        f"C4-C{i}-seed{seed}" for seed in (1, 2) for i in (2, 3)]


@pytest.mark.parametrize("seed", (1, 2))
@pytest.mark.parametrize("condition", D23_SEED_CONDITIONS)
def test_d23_each_condition_has_one_correct_primary_role_config(condition: str, seed: int):
    predecessor = _predecessor()
    successor = build_d23_lock_set(ROOT, predecessor, "1" * 40)
    raw = _config(condition, seed, successor["protocol"])
    spec = resolve_config(raw, seed=seed, protocol_lock=successor["protocol"], production=True)
    assert spec.condition == condition and spec.seed == seed
    assert spec.device_role == D23_SEED_DEVICE_ROLES[condition]


def test_d23_seed_uuid_transfer_is_optional_seed_only_and_exact_model():
    successor = build_d23_lock_set(ROOT, _predecessor(), "1" * 40)
    protocol, hardware = successor["protocol"]["payload"], successor["hardware"]["payload"]
    identity = authorize_production_device(protocol, hardware, condition="C4",
        device_role="rtx3090", gpu_name="NVIDIA GeForce RTX 3090",
        gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", seed=1)
    assert identity["gpu_uuid"].endswith("eeeeeeeeeeee")
    with pytest.raises(HardwareError, match="exact RTX3090"):
        authorize_production_device(protocol, hardware, condition="C4",
            device_role="rtx3090", gpu_name="NVIDIA GeForce RTX 3090",
            gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", seed=0)
    with pytest.raises(HardwareError, match="class policy"):
        authorize_production_device(protocol, hardware, condition="C4",
            device_role="rtx3090", gpu_name="NVIDIA GeForce RTX 4090",
            gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", seed=1)
    with pytest.raises(HardwareError, match="absent from approved protocol"):
        authorize_production_device(protocol, hardware, condition="C3",
            device_role="rtx3090", gpu_name="NVIDIA GeForce RTX 3090",
            gpu_uuid="GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee", seed=1)


def test_d23_rejects_wrong_role_stale_binding_and_seed3():
    successor = build_d23_lock_set(ROOT, _predecessor(), "1" * 40)
    wrong_role = _config("C4", 1, successor["protocol"])
    wrong_role["device_role"] = "rtx4080super"
    with pytest.raises(ConfigError, match="rtx3090"):
        resolve_config(wrong_role, seed=1, protocol_lock=successor["protocol"], production=True)
    stale = _config("C4", 1, successor["protocol"])
    stale["production_binding"]["production_runtime_source_commit"] = "2" * 40
    with pytest.raises(ConfigError, match="binding"):
        resolve_config(stale, seed=1, protocol_lock=successor["protocol"], production=True)
    with pytest.raises(ConfigError, match="D23 optional"):
        resolve_config(_config("C4", 3, successor["protocol"]), seed=3,
                       protocol_lock=successor["protocol"], production=True)
    with pytest.raises(Exception):
        bad = copy.deepcopy(successor["protocol"])
        bad["payload"]["protocol"]["optional_seed_conditions"] = ["C0", "C2"]
        validate_protocol_lock(bad)
