import copy

import pytest

from sinklab.config import ConfigError, resolve_config
from sinklab.provenance import LockError, payload_digest, seal_payload, validate_protocol_lock


@pytest.fixture
def draft():
    return {
        "schema_version": 1, "study": "S1", "condition": "C2",
        "variant": "cosine_soft_jsd_v2", "device_role": "rtx3090",
        "teacher": {"layers": 36, "heads": 20, "width": 1280},
        "student": {"layers": 24, "heads": 16, "width": 1024,
                    "initialization": "random_from_config"},
        "protocol_digest": None,
    }


@pytest.fixture
def approved_payload(draft):
    return {
        "status": "approved", "production_ready": True,
        "study": "S1", "condition_variants": {"C2": "cosine_soft_jsd_v2"},
        "allowed_device_roles": {"C2": ["rtx3090"]},
        "source_commit": "a" * 40,
        "approval": {"researcher": "fixture researcher", "approved_at_utc": "2026-09-27T00:00:00Z",
                     "approval_sha256": "b" * 64,
                     "decision_ids": [f"D{number:02d}" for number in range(1, 19)]},
        "artifact_lock_digest": "c" * 64,
        "environment_lock_digest": "d" * 64,
        "hardware_lock_digest": "e" * 64,
        "calibration_lock_digest": "f" * 64,
        "protocol": {"teacher": draft["teacher"], "student": draft["student"]},
    }


def test_one_explicit_seed_and_immutable_spec(draft):
    with pytest.raises(ConfigError, match="seed"):
        resolve_config(draft, seed=None)
    with pytest.raises(ConfigError, match="seed"):
        resolve_config(draft, seed=True)
    spec = resolve_config(draft, seed=0)
    assert spec.seed == 0 and spec.condition == "C2"
    with pytest.raises(AttributeError):
        spec.seed = 1


@pytest.mark.parametrize("edit", [
    lambda config: config.update(unexpected=True),
    lambda config: config.update(study="S2"),
    lambda config: config.update(condition="C8"),
    lambda config: config.update(variant="index_jsd_v1"),
    lambda config: config["teacher"].update(heads=16),
    lambda config: config["student"].update(initialization="pretrained"),
])
def test_rejects_unknown_and_wrong_contracts(draft, edit):
    edit(draft)
    with pytest.raises(ConfigError):
        resolve_config(draft, seed=0)


def test_s3_variant_and_rel_device_rules(draft):
    draft.update(study="S3", condition="C2", variant="index_jsd_v1")
    draft["teacher"] = {"layers": 12, "heads": 12, "width": 768}
    draft["student"] = {"layers": 6, "heads": 12, "width": 768,
                        "initialization": "random_from_config"}
    assert resolve_config(draft, seed=2).variant == "index_jsd_v1"
    draft.update(condition="C4", variant="causal_qq_kk_vv_v1", device_role="rtx4080super")
    with pytest.raises(ConfigError, match="rtx3090"):
        resolve_config(draft, seed=2)


def test_canonical_digest_ignores_key_order_and_rejects_self_hash():
    assert payload_digest({"b": 1, "a": {"y": 2, "x": 3}}) == payload_digest(
        {"a": {"x": 3, "y": 2}, "b": 1})
    with pytest.raises(LockError, match="outside"):
        seal_payload({"sha256": "a" * 64})
    with pytest.raises(LockError, match="finite"):
        payload_digest({"bad": float("nan")})


def test_draft_and_stale_locks_cannot_pass_production(draft, approved_payload):
    with pytest.raises(ConfigError, match="approved"):
        resolve_config(draft, seed=0, production=True)
    envelope = seal_payload(approved_payload)
    validate_protocol_lock(envelope)
    with pytest.raises(ConfigError, match="stale"):
        resolve_config(draft, seed=0, protocol_lock=envelope, production=True)
    draft["protocol_digest"] = envelope["sha256"]
    assert resolve_config(draft, seed=0, protocol_lock=envelope, production=True).protocol_digest == envelope["sha256"]
    tampered = copy.deepcopy(envelope)
    tampered["payload"]["status"] = "draft"
    with pytest.raises(ConfigError, match="hash mismatch"):
        resolve_config(draft, seed=0, protocol_lock=tampered, production=True)
    draft["device_role"] = "rtx4080super"
    with pytest.raises(ConfigError, match="device role"):
        resolve_config(draft, seed=0, protocol_lock=envelope, production=True)


def test_approval_fields_cannot_be_blank(draft, approved_payload):
    approved_payload["approval"]["researcher"] = ""
    envelope = seal_payload(approved_payload)
    draft["protocol_digest"] = envelope["sha256"]
    with pytest.raises(ConfigError, match="researcher approval"):
        resolve_config(draft, seed=0, protocol_lock=envelope, production=True)


def test_incomplete_decision_approval_is_rejected(draft, approved_payload):
    approved_payload["approval"]["decision_ids"].pop()
    envelope = seal_payload(approved_payload)
    draft["protocol_digest"] = envelope["sha256"]
    with pytest.raises(ConfigError, match="D01-D18"):
        resolve_config(draft, seed=0, protocol_lock=envelope, production=True)
