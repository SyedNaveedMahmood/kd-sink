"""Verify the measured S1 handoff and seal its transitive production locks."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
import subprocess
from pathlib import Path

from .config import MODELS, S1_VARIANTS, production_binding_for, resolve_config
from .hardware import Profile, build_batch_plan, validate_eligibility_matrix
from .owt_compat import RECIPE
from .provenance import (_no_duplicate_keys, canonical_json_bytes, payload_digest,
                         seal_payload, validate_protocol_lock, verify_envelope)
from .stage06_evidence import build_reviewed_seed0_candidate


SOURCE_COMMIT = "a29bcebc6253a5300452594bbaabe4b8e082a463"
GPU_UUIDS = {
    "rtx4080super": "GPU-72b4b307-b613-c35e-ea32-53f4431de9ee",
    "rtx3090": "GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e",
}
LOCK_NAMES = ("artifact", "environment", "hardware", "calibration", "protocol")
LOCK_SHA256 = "b71f143ff21a0ccdd45e995b006430399134bb1564b1210bdc70fb3c67f4c3d5"


class ReadinessError(ValueError):
    """A measured input or transitive production lock is missing or inconsistent."""


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def _sealed(path: Path) -> tuple[dict, str]:
    return verify_envelope(_read(path))


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ReadinessError(message)


def _committed_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def validate_calibration_result(result: dict, inventory: dict, candidate: dict,
                                environment: dict) -> dict[str, float]:
    """Recompute every registered ratio and median before accepting factors."""
    _require(result.get("kind") == "s1-c3-c4-raw-gradient-calibration-v1" and
             result.get("status") == "measured_on_approved_rtx3090" and
             result.get("source_commit") == SOURCE_COMMIT,
             "calibration identity/source commit differs from approved handoff")
    gpu = result.get("gpu", {})
    _require(gpu == {"name": environment["gpu"]["model"],
                     "uuid": GPU_UUIDS["rtx3090"],
                     "driver": environment["gpu"]["nvidia_driver"]},
             "calibration GPU differs from exact 3090 environment")
    _require(result.get("environment") == {
        "python_torch": environment["torch"],
        "torch_cuda": environment["torch_cuda_runtime"],
        "transformers": environment["transformers"]},
        "calibration software differs from sealed 3090 environment")
    expected_hashes = {
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
    }
    _require(all(result.get(k) == v for k, v in expected_hashes.items()),
             "calibration input hash differs from committed production inventory")
    _require(result.get("numerics") == {
        "accumulation": 16, "autocast": "cuda_bf16",
        "dropout": "disabled_during_measurement_only", "effective_batch": 64,
        "gradient_reduction": "fp32_global_l2", "microbatch": 4,
        "missing_gradients": "zero", "optimizer_updates": 0,
        "ratio_rule": "per_batch_jsd_over_raw_objective_median"},
        "calibration numerical protocol changed")
    norms, ratios, factors = result.get("raw_norms"), result.get("ratios"), result.get("factors")
    _require(isinstance(norms, dict) and set(norms) == {"jsd", "mse", "rel"} and
             isinstance(ratios, dict) and set(ratios) == {"mse", "rel"} and
             isinstance(factors, dict) and set(factors) == {"mse", "rel"},
             "calibration arrays/factors are incomplete")
    _require(all(isinstance(v, list) and len(v) == 16 and
                 all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in v)
                 for v in norms.values()), "calibration requires 16 positive finite norms per term")
    for name in ("mse", "rel"):
        observed = ratios[name]
        expected = [a / b for a, b in zip(norms["jsd"], norms[name])]
        _require(isinstance(observed, list) and len(observed) == 16 and
                 all(type(x) in (int, float) and math.isfinite(x) and x > 0 for x in observed) and
                 all(math.isclose(a, b, rel_tol=1e-14) for a, b in zip(observed, expected)) and
                 type(factors[name]) in (int, float) and math.isfinite(factors[name]) and
                 factors[name] == statistics.median(observed),
                 f"{name} ratios or median differ from raw gradients")
    _require(result.get("c2_c5_c6_scale") == 1, "JSD scales must remain one")
    return factors


def build_component_locks(repo: Path, calibration_result: Path,
                          restored_inventory: Path) -> dict[str, dict]:
    """Build four measured locks without making an unapproved protocol choice."""
    repo = Path(repo)
    inventory, inventory_digest = _sealed(repo / "reports/stage06_production_artifact_inventory.json")
    restored, _ = _sealed(restored_inventory)
    _require({k: v for k, v in restored.items() if k != "observed_external_root"} ==
             {k: v for k, v in inventory.items() if k != "observed_external_root"},
             "restored artifact tree differs from committed inventory")
    plan = _read(repo / "configs/s1_jobs_seed0_reviewed.json")
    candidate, candidate_digest = _sealed(repo / "reports/stage06_reviewed_batch_candidate.json")
    recomputed = build_reviewed_seed0_candidate(
        repo / "reports/stage06_3090_profile_evidence.json",
        repo / "reports/stage06_4080_profile_evidence.json",
        repo / "configs/s1_jobs_seed0_reviewed.json", repo / "configs")
    _require(recomputed["sha256"] == candidate_digest and
             candidate["reviewed_plan_sha256"] == payload_digest(plan),
             "reviewed nine-job plan or committed profiles changed")
    proof = candidate["proof"]
    rebuilt = build_batch_plan((Profile(**row) for row in proof["profiles"]),
                               {tuple(row) for row in proof["required"]}, production=True)
    _require(rebuilt == proof and (proof["microbatch"], proof["accumulation"],
              proof["effective_sequences"], proof["input_tokens"],
              proof["shifted_targets"]) == (4, 16, 64, 8192, 8128),
             "committed hardware proof does not establish common 4x16")
    validate_eligibility_matrix(proof)
    environments = {role: _sealed(repo / f"protocols/s1_environment_{'4080' if role == 'rtx4080super' else '3090'}_exact_v1.json")
                    for role in GPU_UUIDS}
    env4080, digest4080 = environments["rtx4080super"]
    env3090, digest3090 = environments["rtx3090"]
    common = env4080["installed_locked_distributions"]
    _require(common == env3090["installed_locked_distributions"] and len(common) == 101 and
             all(env["dependency_lock_sha256"] == LOCK_SHA256 and
                 env["gpu"]["uuid"] == GPU_UUIDS[role] and
                 env["active_environment_exact_sync"] is True and
                 "Would make no changes" in env["uv_sync_locked_dry_run"]["stderr"] and
                 env["pip_check"]["exit_code"] == 0
                 for role, (env, _) in environments.items()),
             "two-role exact environment gate failed")
    _require(_committed_sha(repo / "uv.lock") == LOCK_SHA256, "uv.lock differs from approved bytes")
    result, result_digest = _sealed(calibration_result)
    factors = validate_calibration_result(result, inventory, candidate, env3090)
    approval, approval_digest = _sealed(repo / "protocols/s1_researcher_approval_20260928.json")
    amendment, amendment_digest = _sealed(repo / "protocols/s1_researcher_amendment_20260928.json")
    _require(set(approval["approved_decisions"]) == {f"D{i:02d}" for i in range(1, 19)} and
             approval["amendment_sha256"] == amendment_digest and
             inventory["researcher_approval_sha256"] == approval_digest,
             "researcher approval/amendment differs from inventory")
    artifact = seal_payload({"kind": "s1-production-artifact-lock-v1",
        "inventory_sha256": inventory_digest,
        "dataset": inventory["dataset"], "source": inventory["source"],
        "tokenizer": inventory["tokenizer"], "teacher": inventory["teacher"],
        "student_config": inventory["student_config"], "corpus": inventory["corpus"],
        "panels": inventory["panels"], "calibration_export": inventory["calibration_export"],
        "order_seed0": inventory["order_seed0"],
        "training_initialization": inventory["training_initialization"],
        "calibration_initialization": inventory["calibration_initialization"]})
    environment = seal_payload({"kind": "s1-production-two-role-environment-lock-v1",
        "uv_lock_sha256": LOCK_SHA256, "rtx4080super_manifest_sha256": digest4080,
        "rtx3090_manifest_sha256": digest3090,
        "lock_managed_distributions": common, "lock_managed_maps_equal": True,
        "numerical_policy": env4080["numerical_policy"],
        "devices": {role: {"gpu": env["gpu"], "os": env["os"], "python": env["python"],
                            "torch": env["torch"], "cuda_runtime": env["torch_cuda_runtime"],
                            "cudnn": env["cudnn_version"], "transformers": env["transformers"]}
                    for role, (env, _) in environments.items()}})
    hardware = seal_payload({"kind": "s1-production-hardware-lock-v1",
        "evidence": "measured_gpu", "reviewed_plan_sha256": payload_digest(plan),
        "reviewed_jobs": plan["jobs"], "candidate_sha256": candidate_digest,
        "profile_report_sha256": candidate["source_report_sha256"],
        "gpu_uuids": GPU_UUIDS, "proof": proof,
        "condition_device_eligibility": candidate["condition_device_eligibility"],
        "c5_4080_mb4_repeat": candidate["c5_4080_mb4_repeat"],
        "sequence_length": 128, "bridge_conditions": ["C1", "C2"],
        "unmeasured_not_required_for_current_plan": candidate["unmeasured_not_required_for_current_plan"],
        "c4_4080_status": "prohibited"})
    calibration = seal_payload({"kind": "s1-production-calibration-lock-v1",
        "raw_result_sha256": result_digest, "source_commit": SOURCE_COMMIT,
        "seed": 1729, "gpu": result["gpu"],
        "environment_manifest_sha256": digest3090, "environment": result["environment"],
        "input_hashes": {k: v for k, v in result.items() if k.endswith("sha256")},
        "numerics": result["numerics"], "raw_norms": result["raw_norms"],
        "ratios": result["ratios"], "factors": factors,
        "c2_c5_c6_scale": 1})
    return {"artifact": artifact, "environment": environment,
            "hardware": hardware, "calibration": calibration}


def build_protocol_lock(repo: Path, components: dict[str, dict],
                        *, fingerprint_denominator_floor: float) -> dict:
    """Seal the production root only with an explicitly approved metric floor."""
    _require(type(fingerprint_denominator_floor) in (int, float) and
             math.isfinite(fingerprint_denominator_floor) and
             fingerprint_denominator_floor > 0,
             "positive approved fingerprint denominator floor required")
    _require(set(components) == {"artifact", "environment", "hardware", "calibration"},
             "all four measured component locks required")
    artifact, environment, hardware, calibration = (
        verify_envelope(components[name])[0] for name in
        ("artifact", "environment", "hardware", "calibration"))
    approval, approval_digest = _sealed(Path(repo) / "protocols/s1_researcher_approval_20260928.json")
    _, amendment_digest = _sealed(Path(repo) / "protocols/s1_researcher_amendment_20260928.json")
    plan = _read(Path(repo) / "configs/s1_jobs_seed0_reviewed.json")
    _require(approval["amendment_sha256"] == amendment_digest and
             approval["reviewed_seed0_plan_sha256"] == payload_digest(plan) and
             approval["mandatory_seeds"] == [0] and approval["optional_seeds"] == [1, 2] and
             approval["default_optimizer_updates"] == 10000 and
             hardware["reviewed_plan_sha256"] == payload_digest(plan),
             "researcher approval or reviewed job plan changed")
    allowed = {condition: sorted(role for role in GPU_UUIDS
                  if hardware["condition_device_eligibility"][condition][role]["scheduled_in_current_plan"])
               for condition in S1_VARIANTS}
    teacher, student = MODELS["S1"]
    teacher_layers = [(s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1)
                      for s in range(24)]
    protocol = {
        "production_config_binding_schema": 1,
        "teacher": teacher, "student": student,
        "teacher_layers_for_student": teacher_layers,
        "data": {"recipe": RECIPE,
                 "dataset_id": artifact["dataset"]["id"],
                 "dataset_revision": artifact["dataset"]["revision"],
                 "source_windows": artifact["dataset"]["windows"],
                 "tokenizer_revision": artifact["tokenizer"]["revision"],
                 "tokenizer_files_sha256": artifact["tokenizer"]["files_sha256"],
                 "production_corpus_sha256": artifact["corpus"]["payload_sha256"],
                 "frozen_panels_sha256": artifact["panels"]["payload_sha256"],
                 "calibration_export_sha256": artifact["calibration_export"]["payload_sha256"],
                 "seed0_order_sha256": artifact["order_seed0"]["payload_sha256"],
                 "seed0_initialization_sha256": artifact["training_initialization"]["tensor_content_sha256"],
                 "seed1729_initialization_sha256": artifact["calibration_initialization"]["tensor_content_sha256"]},
        "training": {"primary_optimizer_updates": 10000, "mandatory_seeds": [0],
                     "optional_seeds": [1, 2], "launch_all_seeds_by_default": False,
                     "sequence_length": 128, "microbatch": 4, "accumulation": 16,
                     "effective_sequences": 64, "input_tokens_per_update": 8192,
                     "shifted_targets_per_update": 8128,
                     "student_parameters": "fp32", "adam_states": "fp32",
                     "compute": "bf16_autocast", "probability_and_loss_reductions": "fp32",
                     "attention_backend": "eager", "optimizer": "AdamW",
                     "lr": 0.0005, "betas": [0.9, 0.95], "eps": 1e-8,
                     "weight_decay": 0.1, "clip_norm": 1.0,
                     "warmup_updates": 500, "schedule": "warmup_cosine_to_10000_then_constant_floor",
                     "floor_ratio": 0.1, "student_dropout": 0.1, "teacher_eval": True,
                     "protected_post_10k_continuation": {
                         "requires_protected_final_10000_full_state": True,
                         "requires_explicit_extension_lineage_id": True,
                         "restores_optimizer_rng_order_and_schedule": True,
                         "lr_floor_after_10000": 0.1,
                         "rolling_full_every_absolute_updates": 500,
                         "dense_every_absolute_updates": 100,
                         "protects_each_extension_endpoint": True}},
        "objectives": {"variants": S1_VARIANTS,
                       "base_ce_weight_c1_to_c6": 0.5,
                       "base_kd_weight_c1_to_c6": 0.5,
                       "kd_temperature": 2.0,
                       "auxiliary_coefficient": 1 / 9,
                       "soft_alignment_temperature": 1.0,
                       "relation_heads": 64, "relation_types": ["QQ", "KK", "VV"],
                       "c2_c5_c6_scale": 1,
                       "s_MSE": calibration["factors"]["mse"],
                       "s_REL": calibration["factors"]["rel"]},
        "evaluation": {"fingerprint_denominator_floor": fingerprint_denominator_floor,
                       "dense_every_updates": 100, "include_step_zero": True,
                       "dense_panel_size": 64, "full_panel_size": 300,
                       "endpoint_nll_panel_size": 2000,
                       "primary_student_scope": "all_native_layers",
                       "primary_teacher_scope": "mapped_teacher_layers",
                       "primary_operations": ["delete", "relocate"],
                       "fp32_precision_checkpoints": [0, 500, 10000]},
        "checkpointing": {"retained_model_steps": [0, 100, 250, 500, 1000,
                                                    2000, 5000, 7500, 10000],
                          "full_resume_every": 500,
                          "active_rolling_generations": 2,
                          "protect_final_full_resume": True},
        "hardware": {"gpu_uuids": GPU_UUIDS,
                     "reviewed_plan_sha256": hardware["reviewed_plan_sha256"],
                     "candidate_sha256": hardware["candidate_sha256"],
                     "condition_device_eligibility": hardware["condition_device_eligibility"],
                     "c4_4080_prohibited": True},
        "calibration": {"lock_sha256": components["calibration"]["sha256"],
                        "source_commit": SOURCE_COMMIT, "seed": 1729,
                        "result_sha256": calibration["raw_result_sha256"],
                        "s_MSE": calibration["factors"]["mse"],
                        "s_REL": calibration["factors"]["rel"],
                        "c2_c5_c6_scale": 1},
        "run_plan": {"reviewed_plan_sha256": payload_digest(plan),
                     "jobs": plan["jobs"], "physical_jobs": 9,
                     "unique_condition_seed_pairs": 7,
                     "bridge_conditions": ["C1", "C2"]},
        "researcher_amendment_sha256": amendment_digest,
        "optional_studies": {"S3": False, "S6_long_context": False},
    }
    document = seal_payload({
        "status": "approved", "production_ready": True, "study": "S1",
        "condition_variants": S1_VARIANTS, "allowed_device_roles": allowed,
        "source_commit": SOURCE_COMMIT,
        "approval": {"researcher": "user-provided researcher decisions; identity not asserted",
                     "approved_at_utc": None, "approved_on_utc_date": "2026-09-28",
                     "approval_sha256": approval_digest,
                     "decision_ids": sorted(approval["approved_decisions"])},
        "artifact_lock_digest": components["artifact"]["sha256"],
        "environment_lock_digest": components["environment"]["sha256"],
        "hardware_lock_digest": components["hardware"]["sha256"],
        "calibration_lock_digest": components["calibration"]["sha256"],
        "protocol": protocol,
    })
    validate_protocol_lock(document)
    return document


def write_component_locks(directory: Path, documents: dict[str, dict]) -> None:
    """Write or verify the four measured locks; never overwrite an existing one."""
    _require(set(documents) == set(LOCK_NAMES[:-1]), "complete four-component lock set required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {name: directory / f"{name}.lock.json" for name in documents}
    for name, path in paths.items():
        verify_envelope(documents[name])
        if path.exists():
            _require(_read(path) == documents[name],
                     f"existing {name} lock differs; refuse overwrite")
            continue
        path.write_bytes(canonical_json_bytes(documents[name]) + b"\n")


def write_lock_set(directory: Path, documents: dict[str, dict]) -> None:
    """Write the protocol root after verifying all measured component locks."""
    _require(set(documents) == set(LOCK_NAMES), "complete five-lock set required")
    write_component_locks(directory, {name: documents[name] for name in LOCK_NAMES[:-1]})
    target = Path(directory) / "protocol.lock.json"
    verify_envelope(documents["protocol"])
    if target.exists():
        _require(_read(target) == documents["protocol"],
                 "existing protocol lock differs; refuse overwrite")
    else:
        target.write_bytes(canonical_json_bytes(documents["protocol"]) + b"\n")


def validate_final_lock_set(directory: Path) -> dict[str, dict]:
    """Check a lock directory's canonical envelopes and transitive identities."""
    directory = Path(directory)
    documents = {name: _read(directory / f"{name}.lock.json") for name in LOCK_NAMES}
    payloads = {name: verify_envelope(document)[0] for name, document in documents.items()}
    protocol, _ = validate_protocol_lock(documents["protocol"])
    _require(all(protocol[f"{name}_lock_digest"] == documents[name]["sha256"]
                 for name in ("artifact", "environment", "hardware", "calibration")),
             "protocol root has a stale transitive lock digest")
    artifact, environment, hardware, calibration = (payloads[name] for name in
                                                     ("artifact", "environment", "hardware", "calibration"))
    repo = directory.parent
    inventory, inventory_digest = _sealed(repo / "reports/stage06_production_artifact_inventory.json")
    candidate, candidate_digest = _sealed(repo / "reports/stage06_reviewed_batch_candidate.json")
    approval, approval_digest = _sealed(repo / "protocols/s1_researcher_approval_20260928.json")
    _, amendment_digest = _sealed(repo / "protocols/s1_researcher_amendment_20260928.json")
    plan = _read(repo / "configs/s1_jobs_seed0_reviewed.json")
    _require(artifact["inventory_sha256"] == inventory_digest and
             all(artifact[key] == inventory[key] for key in (
                 "dataset", "source", "tokenizer", "teacher", "student_config",
                 "corpus", "panels", "calibration_export", "order_seed0",
                 "training_initialization", "calibration_initialization")),
             "artifact lock differs from committed real inventory")
    env_records = {role: _sealed(repo / f"protocols/s1_environment_{'4080' if role == 'rtx4080super' else '3090'}_exact_v1.json")
                   for role in GPU_UUIDS}
    _require(environment["rtx4080super_manifest_sha256"] == env_records["rtx4080super"][1] and
             environment["rtx3090_manifest_sha256"] == env_records["rtx3090"][1] and
             environment["lock_managed_distributions"] == env_records["rtx4080super"][0]["installed_locked_distributions"] ==
             env_records["rtx3090"][0]["installed_locked_distributions"] and
             _committed_sha(repo / "uv.lock") == LOCK_SHA256,
             "environment lock differs from both exact-role manifests or uv.lock")
    proof = hardware["proof"]
    _require(proof["sha256"] == payload_digest({k: v for k, v in proof.items() if k != "sha256"}) and
             proof["microbatch"] == 4 and proof["accumulation"] == 16 and
             hardware["reviewed_plan_sha256"] == protocol["protocol"]["run_plan"]["reviewed_plan_sha256"] ==
             payload_digest(plan) and
             hardware["candidate_sha256"] == candidate_digest and
             proof == candidate["proof"] and
             hardware["condition_device_eligibility"] == candidate["condition_device_eligibility"] and
             hardware["profile_report_sha256"] == candidate["source_report_sha256"] and
             all(_committed_sha(repo / f"reports/stage06_{short}_profile_evidence.json") ==
                 candidate["source_report_sha256"][role]
                 for role, short in (("rtx3090", "3090"), ("rtx4080super", "4080"))),
             "hardware proof or reviewed plan changed")
    validate_eligibility_matrix(proof)
    _require(environment["uv_lock_sha256"] == LOCK_SHA256 and
             environment["lock_managed_maps_equal"] is True and
             calibration["source_commit"] == SOURCE_COMMIT and
             calibration["environment_manifest_sha256"] == environment["rtx3090_manifest_sha256"] and
             calibration["input_hashes"]["corpus_sha256"] == artifact["corpus"]["payload_sha256"] and
             calibration["input_hashes"]["frozen_panels_sha256"] == artifact["panels"]["payload_sha256"] and
             protocol["source_commit"] == SOURCE_COMMIT and
             protocol["approval"]["approval_sha256"] == approval_digest and
             protocol["protocol"]["researcher_amendment_sha256"] == amendment_digest and
             approval["amendment_sha256"] == amendment_digest and
             protocol["condition_variants"] == S1_VARIANTS and
             protocol["protocol"]["run_plan"]["jobs"] == plan["jobs"] and
             protocol["protocol"]["training"]["primary_optimizer_updates"] == 10000 and
             protocol["protocol"]["training"]["mandatory_seeds"] == [0] and
             protocol["protocol"]["training"]["optional_seeds"] == [1, 2],
             "artifact/environment/calibration source identities disagree")
    reconstructed_result = {"kind": "s1-c3-c4-raw-gradient-calibration-v1",
        "status": "measured_on_approved_rtx3090", "source_commit": calibration["source_commit"],
        "gpu": calibration["gpu"], "environment": calibration["environment"],
        "numerics": calibration["numerics"], "raw_norms": calibration["raw_norms"],
        "ratios": calibration["ratios"], "factors": calibration["factors"],
        "c2_c5_c6_scale": calibration["c2_c5_c6_scale"],
        **calibration["input_hashes"]}
    validate_calibration_result(reconstructed_result, inventory, candidate,
                                env_records["rtx3090"][0])
    _require(protocol["protocol"]["objectives"]["s_MSE"] == calibration["factors"]["mse"] and
             protocol["protocol"]["objectives"]["s_REL"] == calibration["factors"]["rel"],
             "calibration ratios/factors differ from protocol")
    _require(type(protocol["protocol"]["evaluation"]["fingerprint_denominator_floor"])
             in (int, float) and
             math.isfinite(protocol["protocol"]["evaluation"]["fingerprint_denominator_floor"]) and
             protocol["protocol"]["evaluation"]["fingerprint_denominator_floor"] > 0,
             "approved metric floor is absent")
    return documents


def build_production_configs(repo: Path, protocol_document: dict) -> tuple[dict[str, dict], dict]:
    """Derive exactly the reviewed nine seed-0 configs; do not launch them."""
    protocol, root_digest = validate_protocol_lock(protocol_document)
    _require(root_digest == protocol_document["sha256"] and
             protocol["source_commit"] == SOURCE_COMMIT and
             protocol["protocol"].get("production_config_binding_schema") == 1,
             "final S1 protocol binding schema is absent")
    reviewed = _read(Path(repo) / "configs/s1_jobs_seed0_reviewed.json")
    _require(payload_digest(reviewed) == protocol["protocol"]["run_plan"]["reviewed_plan_sha256"] and
             len(reviewed["jobs"]) == 9,
             "reviewed nine-job plan differs from protocol root")
    configs: dict[str, dict] = {}
    jobs = []
    for job in reviewed["jobs"]:
        source = _read(Path(repo) / "configs" / job["config"])
        path = f"production/s1/{Path(job['config']).name}"
        _require(path not in configs and job["seed"] == 0,
                 "reviewed production config is duplicate or not seed0")
        source["protocol_digest"] = root_digest
        source["production_binding"] = production_binding_for(
            protocol, source["device_role"], job["seed"])
        resolve_config(source, seed=job["seed"], protocol_lock=protocol_document,
                       production=True)
        configs[path] = source
        jobs.append({**job, "config": path})
    plan = {**reviewed, "jobs": jobs}
    _require(len(configs) == 9 and len({(x["run_id"], x["seed"]) for x in jobs}) == 9,
             "production plan must contain nine unique physical jobs")
    return configs, plan


def write_production_configs(repo: Path, configs: dict[str, dict], plan: dict) -> None:
    root = Path(repo) / "configs"
    paths = {name: root / name for name in configs}
    plan_path = root / "production/s1_jobs_seed0.json"
    _require(len(paths) == 9, "exactly nine production configs required")
    for name, path in paths.items():
        if path.exists():
            _require(_read(path) == configs[name],
                     f"existing production config {name} differs; refuse overwrite")
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(canonical_json_bytes(configs[name]) + b"\n")
    if plan_path.exists():
        _require(_read(plan_path) == plan,
                 "existing production job plan differs; refuse overwrite")
    else:
        plan_path.write_bytes(canonical_json_bytes(plan) + b"\n")


def validate_production_configs(repo: Path, protocol_document: dict) -> dict:
    from .job_plan import validate_job_plan
    root = Path(repo) / "configs"
    plan = _read(root / "production/s1_jobs_seed0.json")
    result = validate_job_plan(plan, root, protocol_lock=protocol_document, production=True)
    _require(result["physical_runs"] == 9 and result["unique_condition_seed_pairs"] == 7 and
             result["seeds"] == [0] and not result["hardware_confounds"] and
             result["launchable"] is True,
             "production nine-job plan does not validate")
    return result
