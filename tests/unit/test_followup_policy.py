"""Prospective D24 boundaries; all model fixtures are tiny, synthetic and CPU only."""

import copy
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

from sinklab.followup_policy import (D24_PATH, D24_SHA256, FollowupPolicyError,
    admit_s1_followup, checkpoint_inventory_coverage, required_s1_checkpoints,
    validate_followup_amendment)
from sinklab.provenance import seal_payload, verify_envelope
from sinklab.evaluate import _key, evaluate_panel
from sinklab.probes import ProbeError, evaluate_probe_battery
from sinklab.s6 import evaluate_optional_long_context, evaluate_s6_domains


ROOT = Path(__file__).resolve().parents[2]


def identity(seed=0, condition="C6"):
    return dict(study="S1", seed=seed, condition=condition,
                protocol_sha256="a" * 64, model_sha256="b" * 64,
                corpus_sha256="c" * 64)


def test_sealed_amendment_plan_and_historical_training_root():
    document = json.loads((ROOT / D24_PATH).read_text())
    payload = validate_followup_amendment(document)
    assert payload["rule"] == "All future checkpoint-dependent analyses using S1 trained models are seed0-only."
    assert payload["prospective_only"] is True
    original, original_sha = verify_envelope(json.loads((ROOT / "protocols/protocol.lock.json").read_text()))
    assert original_sha == payload["historical_current_training_root_sha256"]
    assert original_sha == "7b12e5a637c3a8b6ebe66d4bedbd202077fc7960bc98a443babf8bf68de8d8a7"
    assert D24_SHA256 != original_sha
    plan = json.loads((ROOT / "configs/s1_checkpoint_followups_seed0.json").read_text())
    assert plan["amendment_sha256"] == D24_SHA256
    assert plan["training_seeds"] == [0]
    assert plan["conditions"] == [f"C{i}" for i in range(7)]
    assert plan["checkpoints"] == {"S4": [0, 100, 500, 2000, 10000], "S6": [0, 500, 2000, 10000]}
    assert plan["new_s5_inference_training_seeds"] == [0]
    assert plan["existing_s5_record_training_seeds"] == [0, 1, 2]
    altered = copy.deepcopy(payload)
    altered["allowed_training_seeds"] = [0, 1]
    with pytest.raises(FollowupPolicyError, match="unapproved"):
        validate_followup_amendment(seal_payload(altered))


@pytest.mark.parametrize("study,steps", [("S4", [0, 100, 500, 2000, 10000]),
                                       ("S6", [0, 500, 2000, 10000])])
def test_readiness_requires_only_seed0_and_keeps_original_roots(study, steps):
    manifests = []
    for condition in [f"C{i}" for i in range(7)]:
        for step in steps:
            source = dict(identity(condition=condition), run_id=f"{condition}-seed0")
            prior = copy.deepcopy(source)
            policy = admit_s1_followup(source, study=study, step=step)
            assert source == prior and policy["source_protocol_sha256"] == "a" * 64
            manifests.append(dict(identity=source, step=step, kind="weights",
                                  files={"model.safetensors": "d" * 64}))
    report = checkpoint_inventory_coverage(manifests, study=study)
    assert report["required_count"] == 7 * len(steps)
    assert report["status"] == "complete_inventory" and report["missing"] == []
    assert report["execution_authorized"] is False
    assert all(seed == 0 for _, seed, _ in required_s1_checkpoints(study))
    seed1 = copy.deepcopy(manifests[0]); seed1["identity"]["seed"] = 1
    missing = checkpoint_inventory_coverage(manifests[1:] + [seed1], study=study)
    assert missing["missing"] == [["C0", 0, 0]]
    # A bridge supplies another candidate, never an independent seed.
    bridge = copy.deepcopy(manifests[0]); bridge["identity"]["run_id"] = "bridge"
    report = checkpoint_inventory_coverage(manifests + [bridge], study=study)
    assert report["covered_count"] == 7 * len(steps)
    assert len(report["candidates"][0]["checkpoints"]) == 2


@pytest.mark.parametrize("seed", [1, 2, 3, -1, True, "0", None])
@pytest.mark.parametrize("study", ["S4", "S5", "S6", "future_intervention"])
def test_all_future_checkpoint_work_rejects_nonzero_or_ambiguous_seed(seed, study):
    with pytest.raises(FollowupPolicyError, match="seed0-only"):
        admit_s1_followup(identity(seed), study=study, step=10000)


@pytest.mark.parametrize("study,step", [("S4", 250), ("S6", 100), ("S6", 7500)])
def test_followup_rejects_unregistered_checkpoint(study, step):
    with pytest.raises(FollowupPolicyError, match="fixed retained"):
        admit_s1_followup(identity(), study=study, step=step)


@pytest.mark.parametrize("seed", [1, 2])
def test_public_evaluation_paths_reject_before_adapter_or_store_access(seed):
    source = identity(seed)
    with pytest.raises(FollowupPolicyError, match="seed0-only"):
        evaluate_panel(adapter=object(), items=[], panel="future", panel_hash="b"*64,
            checkpoint_hash="c"*64, run_id="original", step=10000, store=object(),
            run_identity=source, followup_study="S5")
    with pytest.raises(FollowupPolicyError, match="seed0-only"):
        evaluate_probe_battery(adapter=object(), items=[], store=object(),
            checkpoint_sha256="b"*64, panel_sha256="c"*64, run_id="original",
            model_role="student", control_seed=1729, denominator_floor=1e-8,
            responsiveness_floor=1e-8, provenance={"source": "fixture"},
            source_identity=source, step=10000)
    with pytest.raises(FollowupPolicyError, match="seed0-only"):
        evaluate_s6_domains(adapter=object(), document={}, tokenizer_sha256="b"*64,
            checkpoint_sha256="c"*64, run_id="original", step=10000, store=object(),
            run_identity=source, denominator_floor=1e-8, precision="fp32")
    with pytest.raises(FollowupPolicyError, match="seed0-only"):
        evaluate_optional_long_context(adapter=object(), items=[], context=512,
            approval={}, memory_validation={}, panel_sha256="b"*64,
            checkpoint_sha256="c"*64, run_id="original", step=10000, store=object(),
            run_identity=source, denominator_floor=1e-8, precision="fp32")


def test_s4_missing_student_identity_fails_closed():
    with pytest.raises(ProbeError, match="source_identity"):
        evaluate_probe_battery(adapter=object(), items=[], store=object(),
            checkpoint_sha256="b"*64, panel_sha256="c"*64, run_id="original",
            model_role="student", control_seed=0, denominator_floor=1e-8,
            responsiveness_floor=1e-8, provenance={"fixture": True})


def test_followup_keys_do_not_collide_with_historical_training_keys():
    kwargs = dict(run_id="original", step=10000, panel="dense", panel_hash="b"*64,
        checkpoint_hash="c"*64, item_id="one", scope=[0], operation="clean",
        strength=0., precision="bf16", model_role="student", evaluation_mode="full",
        run_identity=identity(), denominator_floor=1e-8)
    historical = _key(**kwargs)
    policy = admit_s1_followup(identity(), study="S5", step=10000)
    followup = _key(**kwargs, followup_policy=policy)
    assert followup.pop("followup_policy") == policy
    assert followup == historical and "followup_policy" not in historical


def test_actual_checkpoint_identity_schema_and_conflicting_roots():
    source = identity()
    source["protocol_hash"] = source.pop("protocol_sha256")
    before = copy.deepcopy(source)
    assert admit_s1_followup(source, study="S4", step=0)["source_protocol_sha256"] == "a" * 64
    assert source == before
    source["protocol_sha256"] = "e" * 64
    with pytest.raises(FollowupPolicyError, match="conflicting roots"):
        admit_s1_followup(source, study="S4", step=0)


def test_readonly_readiness_checks_payload_bytes_and_ignores_seed12_requirements(tmp_path, capsys):
    from sinklab.provenance import canonical_json_bytes, payload_digest
    spec = importlib.util.spec_from_file_location("d24_readiness", ROOT / "scripts/check_s1_followup_readiness.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    paths = []
    for seed in (0, 1, 2):
        path = tmp_path / f"seed{seed}" / "weights-000000"
        path.mkdir(parents=True)
        source = dict(identity(seed), run_id=f"original-seed{seed}")
        source["protocol_hash"] = source.pop("protocol_sha256")
        model = b"synthetic bytes, never tensor-loaded"
        manifest = dict(schema_version=1, step=0, kind="weights", identity=source,
            identity_sha256=payload_digest(source), files={"model.safetensors": hashlib.sha256(model).hexdigest()})
        raw = canonical_json_bytes(manifest) + b"\n"
        (path / "manifest.json").write_bytes(raw)
        # No seed1/2 payloads: these are not readiness requirements.
        if seed == 0:
            (path / "model.safetensors").write_bytes(model)
            (path / "COMPLETE").write_text(hashlib.sha256(raw).hexdigest() + "\n")
        paths.append(path)
    before = {p: p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    args = ["--study", "S4"]
    for path in paths:
        args += ["--checkpoint", str(path)]
    assert module.main(args) == 1  # Missing other seed0 states, not seed1/2.
    report = json.loads(capsys.readouterr().out)
    assert report["required_count"] == 35 and report["covered_count"] == 1
    assert len(report["missing"]) == 34
    assert len(report["excluded_from_requirements"]) == 2
    assert all(p.read_bytes() == content for p, content in before.items())
    (paths[0] / "model.safetensors").write_bytes(b"corrupt synthetic fixture")
    with pytest.raises(ValueError, match="content hash mismatch"):
        module.main(args)
