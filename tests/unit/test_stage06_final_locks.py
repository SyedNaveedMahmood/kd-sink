"""Strict final-lock and immutable production-config boundaries."""

import copy
import json
import shutil
from pathlib import Path

import pytest

from sinklab.config import ConfigError, production_binding_for, resolve_config
from sinklab.metrics import fingerprint
from sinklab.provenance import LockError, seal_payload, validate_protocol_lock, verify_envelope
from sinklab.stage06_readiness import (CALIBRATION_SOURCE_COMMIT, SUPERSEDED_PROTOCOL_ROOT,
                                       ReadinessError,
                                       build_production_configs, build_protocol_lock,
                                       validate_calibration_result, validate_final_lock_set,
                                       validate_production_configs, write_lock_set,
                                       write_production_configs)


ROOT = Path(__file__).resolve().parents[2]


def test_approved_stage06_root_uses_pretraining_fingerprint_guard():
    current = json.loads((ROOT / "protocols/protocol.lock.json").read_text(encoding="utf-8"))
    if current["sha256"] == SUPERSEDED_PROTOCOL_ROOT:
        with pytest.raises(ReadinessError, match="superseded"):
            validate_final_lock_set(ROOT / "protocols")
        return
    documents = validate_final_lock_set(ROOT / "protocols")
    protocol = documents["protocol"]
    assert protocol["sha256"] != SUPERSEDED_PROTOCOL_ROOT
    assert protocol["payload"]["calibration_source_commit"] == CALIBRATION_SOURCE_COMMIT
    assert protocol["payload"]["production_runtime_source_commit"] != CALIBRATION_SOURCE_COMMIT
    floor = protocol["payload"]["protocol"]["evaluation"]["fingerprint_denominator_floor"]
    assert floor == 1e-8
    below = fingerprint(0.5e-8, 0.2, denominator_floor=floor)
    assert below == {
        "baseline_sink": 0.5e-8,
        "probed_sink": 0.2,
        "absolute_delta_sink": 0.2 - 0.5e-8,
        "ratio": None,
        "ratio_unavailable_reason": "negligible_baseline",
        "denominator_floor": 1e-8,
    }
    assert fingerprint(1e-8, 0.2, denominator_floor=floor)["ratio"] == 0.2 / 1e-8
    assert validate_production_configs(ROOT, documents["protocol"])["physical_runs"] == 9


def test_measured_component_locks_bind_real_inputs_without_claiming_root():
    expected = {
        "artifact": "2e721e9726e666cb8dc08a9b7d606a6d6ff99b434011b40fe2cf41f8eb315ccd",
        "environment": "0266e734fa97357ed0609e722c8d492284b34a06f726ce37b7301e703a370025",
        "hardware": "c284abebf7b9915053c282d57555a1a28cf9ad8250f3c402a3f313207eaae138",
        "calibration": "fa031af3de63832e054b89539d3867c2904c9b592b1e7a8b3e4ca3e3676c532f",
    }
    locks = {}
    for name, digest in expected.items():
        locks[name], observed = verify_envelope(json.loads((
            ROOT / "protocols" / f"{name}.lock.json").read_text(encoding="utf-8")))
        assert observed == digest
    inventory = _payload("reports/stage06_production_artifact_inventory.json")
    assert locks["artifact"]["corpus"] == inventory["corpus"]
    assert locks["artifact"]["teacher"] == inventory["teacher"]
    assert locks["environment"]["lock_managed_maps_equal"] is True
    assert len(locks["environment"]["lock_managed_distributions"]) == 101
    matrix = locks["hardware"]["condition_device_eligibility"]
    assert matrix["C4"]["rtx4080super"]["policy_allowed"] is False
    assert matrix["C3"]["rtx4080super"]["measured_production_eligible"] is False
    assert matrix["C0"]["rtx3090"]["scheduled_in_current_plan"] is False
    assert locks["hardware"]["proof"]["microbatch"] == 4
    assert locks["calibration"]["source_commit"] == CALIBRATION_SOURCE_COMMIT
    assert all(len(v) == 16 for v in locks["calibration"]["raw_norms"].values())
    assert locks["calibration"]["factors"] == {
        "mse": 68.00580071126464, "rel": 0.120179255876581}


def _payload(path):
    return verify_envelope(json.loads((ROOT / path).read_text(encoding="utf-8")))[0]


def _calibration_fixture():
    inventory = _payload("reports/stage06_production_artifact_inventory.json")
    candidate = _payload("reports/stage06_reviewed_batch_candidate.json")
    environment = _payload("protocols/s1_environment_3090_exact_v1.json")
    result = {
        "kind": "s1-c3-c4-raw-gradient-calibration-v1",
        "status": "measured_on_approved_rtx3090", "source_commit": CALIBRATION_SOURCE_COMMIT,
        "gpu": {"name": environment["gpu"]["model"],
                "uuid": environment["gpu"]["uuid"],
                "driver": environment["gpu"]["nvidia_driver"]},
        "environment": {"python_torch": environment["torch"],
                        "torch_cuda": environment["torch_cuda_runtime"],
                        "transformers": environment["transformers"]},
        "candidate_batch_plan_sha256": candidate["proof"]["sha256"],
        "reviewed_plan_sha256": inventory["reviewed_plan_sha256"],
        "panel_export_sha256": inventory["calibration_export"]["payload_sha256"],
        "frozen_panels_sha256": inventory["panels"]["payload_sha256"],
        "corpus_sha256": inventory["corpus"]["payload_sha256"],
        "student_config_file_sha256": inventory["student_config"]["sha256"],
        "teacher_config_file_sha256": inventory["teacher"]["config_sha256"],
        "teacher_weights_file_sha256": inventory["teacher"]["weights_sha256"],
        "calibration_initialization_tensor_sha256": inventory["calibration_initialization"]["tensor_content_sha256"],
        "calibration_initialization_metadata_sha256": inventory["calibration_initialization"]["metadata_sha256"],
        "numerics": {"accumulation": 16, "autocast": "cuda_bf16",
                     "dropout": "disabled_during_measurement_only", "effective_batch": 64,
                     "gradient_reduction": "fp32_global_l2", "microbatch": 4,
                     "missing_gradients": "zero", "optimizer_updates": 0,
                     "ratio_rule": "per_batch_jsd_over_raw_objective_median"},
        "raw_norms": {"jsd": [2.0] * 16, "mse": [1.0] * 16, "rel": [4.0] * 16},
        "ratios": {"mse": [2.0] * 16, "rel": [0.5] * 16},
        "factors": {"mse": 2.0, "rel": 0.5}, "c2_c5_c6_scale": 1,
    }
    return result, inventory, candidate, environment


def test_calibration_requires_all_raw_norms_ratios_and_exact_medians():
    args = _calibration_fixture()
    assert validate_calibration_result(*args) == {"mse": 2.0, "rel": 0.5}
    bad = copy.deepcopy(args[0])
    bad["raw_norms"]["rel"][0] = 0.0
    with pytest.raises(ReadinessError, match="positive finite norms"):
        validate_calibration_result(bad, *args[1:])
    bad = copy.deepcopy(args[0])
    bad["ratios"]["mse"][0] = 3.0
    with pytest.raises(ReadinessError, match="ratios or median"):
        validate_calibration_result(bad, *args[1:])
    bad = copy.deepcopy(args[0])
    bad["source_commit"] = "0" * 40
    with pytest.raises(ReadinessError, match="source commit"):
        validate_calibration_result(bad, *args[1:])


def test_date_precision_approval_does_not_fabricate_utc_time():
    approval = {"researcher": "user-provided; identity not asserted",
                "approved_at_utc": None, "approved_on_utc_date": "2026-09-28",
                "approval_sha256": "a" * 64,
                "decision_ids": [f"D{i:02d}" for i in range(1, 19)]}
    payload = {"status": "approved", "production_ready": True, "study": "S1",
               "condition_variants": {"C3": "head_mean_probability_mse_v1"},
               "allowed_device_roles": {"C3": ["rtx3090"]},
               "source_commit": CALIBRATION_SOURCE_COMMIT, "approval": approval,
               "artifact_lock_digest": "b" * 64,
               "environment_lock_digest": "c" * 64,
               "hardware_lock_digest": "d" * 64,
               "calibration_lock_digest": "e" * 64,
               "protocol": {"teacher": {"layers": 36, "heads": 20, "width": 1280},
                            "student": {"layers": 24, "heads": 16, "width": 1024,
                                        "initialization": "random_from_config"}}}
    validate_protocol_lock(seal_payload(payload))
    approval["approved_on_utc_date"] = "2026-09-99"
    with pytest.raises(LockError, match="UTC date"):
        validate_protocol_lock(seal_payload(payload))


def test_production_config_binds_uuid_data_batch_locks_and_seed():
    payload = {"status": "approved", "production_ready": True, "study": "S1",
               "condition_variants": {"C3": "head_mean_probability_mse_v1"},
               "allowed_device_roles": {"C3": ["rtx3090"]},
               "source_commit": "f" * 40,
               "production_runtime_source_commit": "f" * 40,
               "calibration_source_commit": CALIBRATION_SOURCE_COMMIT,
               "execution_critical_path_set_version": 1,
               "approval": {"researcher": "fixture", "approved_at_utc": "2026-09-28T00:00:00Z",
                            "approval_sha256": "a" * 64,
                            "decision_ids": [f"D{i:02d}" for i in range(1, 19)]},
               "artifact_lock_digest": "b" * 64,
               "environment_lock_digest": "c" * 64,
               "hardware_lock_digest": "d" * 64,
               "calibration_lock_digest": "e" * 64,
               "protocol": {"production_config_binding_schema": 2,
                            "teacher": {"layers": 36, "heads": 20, "width": 1280},
                            "student": {"layers": 24, "heads": 16, "width": 1024,
                                        "initialization": "random_from_config"},
                            "hardware": {"gpu_uuids": {"rtx3090": "GPU-approved"}},
                            "data": {"seed0_initialization_sha256": "f" * 64,
                                     "seed0_order_sha256": "1" * 64,
                                     "production_corpus_sha256": "2" * 64,
                                     "frozen_panels_sha256": "3" * 64},
                            "training": {"microbatch": 4, "accumulation": 16,
                                         "effective_sequences": 64, "sequence_length": 128,
                                         "primary_optimizer_updates": 10000}}}
    lock = seal_payload(payload)
    config = {"schema_version": 1, "study": "S1", "condition": "C3",
              "variant": "head_mean_probability_mse_v1", "device_role": "rtx3090",
              "teacher": payload["protocol"]["teacher"],
              "student": payload["protocol"]["student"],
              "protocol_digest": lock["sha256"],
              "production_binding": production_binding_for(payload, "rtx3090", 0)}
    resolve_config(config, seed=0, protocol_lock=lock, production=True)
    for field, wrong in (("gpu_uuid", "GPU-wrong"), ("microbatch", 8),
                         ("calibration_lock_digest", "0" * 64), ("seed", 1)):
        changed = copy.deepcopy(config)
        changed["production_binding"][field] = wrong
        with pytest.raises(ConfigError, match="production config binding"):
            resolve_config(changed, seed=0, protocol_lock=lock, production=True)


def test_lock_writer_refuses_existing_target(tmp_path):
    documents = {name: seal_payload({"kind": name}) for name in
                 ("artifact", "environment", "hardware", "calibration", "protocol")}
    prior = seal_payload({"kind": "other"})
    (tmp_path / "artifact.lock.json").write_text(json.dumps(prior))
    with pytest.raises(ReadinessError, match="refuse overwrite"):
        write_lock_set(tmp_path, documents)
    assert json.loads((tmp_path / "artifact.lock.json").read_text()) == prior


def test_full_transitive_root_and_nine_configs_with_fixture_floor_only(tmp_path):
    # 1.0 is an isolated test fixture, not a proposed or sealed study value.
    relative = ["uv.lock", "reports/stage06_production_artifact_inventory.json",
                "reports/stage06_reviewed_batch_candidate.json",
                "reports/stage06_3090_profile_evidence.json",
                "reports/stage06_4080_profile_evidence.json",
                "protocols/s1_environment_4080_exact_v1.json",
                "protocols/s1_environment_3090_exact_v1.json",
                "protocols/s1_researcher_approval_20260928.json",
                "protocols/s1_researcher_amendment_20260928.json",
                "configs/s1_jobs_seed0_reviewed.json"]
    reviewed = json.loads((ROOT / "configs/s1_jobs_seed0_reviewed.json").read_text())
    relative += [f"configs/{job['config']}" for job in reviewed["jobs"]]
    for name in relative:
        target = tmp_path / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, target)
    components = {name: json.loads((ROOT / "protocols" / f"{name}.lock.json")
                                   .read_text(encoding="utf-8"))
                  for name in ("artifact", "environment", "hardware", "calibration")}
    protocol = build_protocol_lock(tmp_path, components,
                                   fingerprint_denominator_floor=1.0,
                                   production_runtime_source_commit="f" * 40)
    configs, plan = build_production_configs(tmp_path, protocol)
    write_lock_set(tmp_path / "protocols", {**components, "protocol": protocol})
    write_production_configs(tmp_path, configs, plan)
    assert validate_final_lock_set(tmp_path / "protocols")["protocol"] == protocol
    assert validate_production_configs(tmp_path, protocol)["physical_runs"] == 9
    changed = copy.deepcopy(configs["production/s1/c4_rtx3090.json"])
    changed["production_binding"]["gpu_uuid"] = "GPU-wrong"
    with pytest.raises(ConfigError, match="production config binding"):
        resolve_config(changed, seed=0, protocol_lock=protocol, production=True)
