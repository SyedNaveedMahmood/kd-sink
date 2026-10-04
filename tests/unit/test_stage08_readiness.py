import copy
import importlib.util
from pathlib import Path

import pytest

from sinklab.provenance import canonical_json_bytes, payload_digest, seal_payload


ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("stage08_readiness", ROOT / "scripts/stage08_readiness.py")
CHECKS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CHECKS)


def decision():
    return CHECKS.read_json(ROOT / CHECKS.EXCEPTION_PATH)


def test_exception_accepts_only_exact_approved_identity_and_log_digest():
    doc = decision()
    binding = doc["payload"]
    original = copy.deepcopy(binding["identity"])
    assert CHECKS.accepted_c3_log_exception(identity=original,
        log_sha256=binding["train_jsonl_sha256"], document=doc) == binding
    assert original == binding["identity"]
    for field, value in [("condition", "C2"), ("seed", 1), ("run_id", "another-c3"),
                          ("gpu_uuid", "other-gpu"), ("protocol_hash", "a" * 64),
                          ("data_hash", "b" * 64)]:
        changed = dict(original, **{field: value})
        with pytest.raises(ValueError, match="outside exact"):
            CHECKS.accepted_c3_log_exception(identity=changed,
                log_sha256=binding["train_jsonl_sha256"], document=doc)
    with pytest.raises(ValueError, match="outside exact"):
        CHECKS.accepted_c3_log_exception(identity=original, log_sha256="0" * 64, document=doc)
    changed = copy.deepcopy(binding)
    changed["train_jsonl_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="unapproved"):
        CHECKS.accepted_c3_log_exception(identity=original,
            log_sha256="0" * 64, document=seal_payload(changed))


def test_other_missing_history_never_inherits_exception(tmp_path):
    source = dict(decision()["payload"]["identity"], condition="C5", run_id="c5")
    (tmp_path / "train.jsonl").write_text('{"event":"start","resumed_step":2500}\n')
    with pytest.raises(ValueError, match="outside exact"):
        CHECKS.verify_training_log(tmp_path, source, exception_document=decision())


@pytest.mark.parametrize("damage", ["model", "COMPLETE", "identity", "step", "missing_model"])
def test_c3_checkpoint_checks_are_never_waived(tmp_path, damage):
    identity = decision()["payload"]["identity"]
    path = tmp_path / "weights-000500"
    path.mkdir()
    (path / "model.safetensors").write_bytes(b"synthetic test bytes only")
    manifest = dict(schema_version=1, step=500, kind="weights", identity=identity,
        identity_sha256=payload_digest(identity),
        files={"model.safetensors": CHECKS.sha256(path / "model.safetensors")})
    (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    (path / "COMPLETE").write_text(CHECKS.sha256(path / "manifest.json"))
    assert CHECKS.verify_required_checkpoint(path, identity=identity, step=500)["status"] == "verified"
    if damage == "model":
        (path / "model.safetensors").write_bytes(b"corrupt")
    elif damage == "missing_model":
        (path / "model.safetensors").unlink()  # Test-owned tmp_path only.
    elif damage == "COMPLETE":
        (path / "COMPLETE").write_text("0" * 64)
    else:
        if damage == "identity":
            manifest["identity"] = dict(identity, seed=1)
            manifest["identity_sha256"] = payload_digest(manifest["identity"])
        else:
            manifest["step"] = 2000
        (path / "manifest.json").write_bytes(canonical_json_bytes(manifest))
        (path / "COMPLETE").write_text(CHECKS.sha256(path / "manifest.json"))
    with pytest.raises(ValueError):
        CHECKS.verify_required_checkpoint(path, identity=identity, step=500)
