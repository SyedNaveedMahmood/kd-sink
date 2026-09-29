"""Verify the measured S1 handoff and seal its transitive production locks."""

from __future__ import annotations

import hashlib
import copy
import json
import math
import statistics
import subprocess
from pathlib import Path

from .config import MODELS, S1_VARIANTS, production_binding_for, resolve_config
from .hardware import RTX4080_SUPER_MODEL
from .hardware import Profile, build_batch_plan, validate_eligibility_matrix
from .owt_compat import RECIPE
from .provenance import (COMMIT_PATTERN, _no_duplicate_keys, canonical_json_bytes, payload_digest,
                         seal_payload, validate_protocol_lock, verify_envelope)
from .runtime_provenance import EXECUTION_CRITICAL_PATH_SET_VERSION
from .stage06_evidence import build_reviewed_seed0_candidate


CALIBRATION_SOURCE_COMMIT = "a29bcebc6253a5300452594bbaabe4b8e082a463"
SUPERSEDED_PROTOCOL_ROOT = "2a11da9bb71957a4d6b3a2f93a34bd67491dd9d21d70577a7a10858930a8943e"
PREDECESSOR_PROTOCOL_ROOT = "910961fcc53edaed0df48e3139dbb7ca2e058bf7480b67dab27bae0bcd90f26f"
D19_PROTOCOL_ROOT = "48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791"
D19_PATH = "protocols/s1_researcher_amendment_d19_4080_class_20260928.json"
D20_PATH = "protocols/s1_researcher_amendment_d20_c3_4080_20260929.json"
D20_EVIDENCE_PATH = "reports/stage06_c3_4080_qualification.json"
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


def _d19(repo: Path) -> tuple[dict, str]:
    decision, digest = _sealed(Path(repo) / D19_PATH)
    observed, observed_digest = _sealed(Path(repo) / "reports/stage06_second_4080_qualification.json")
    candidate, _ = _sealed(Path(repo) / "reports/stage06_reviewed_batch_candidate.json")
    attempts = {(p["condition"], p["microbatch"]): p for p in observed["profiles"]["attempts"]}
    _require(decision.get("decision_id") == "D19" and
             decision.get("status") == "approved_prospective_hardware_policy" and
             decision.get("predecessor_production_root") == PREDECESSOR_PROTOCOL_ROOT and
             decision.get("reference_model") == RTX4080_SUPER_MODEL and
             decision.get("reference_uuid") == GPU_UUIDS["rtx4080super"] and
             decision.get("reference_profile_report_sha256") ==
             candidate["source_report_sha256"]["rtx4080super"] and
             decision.get("naveed_observed_profile_report_sha256") == observed_digest and
             decision.get("known_equivalent_machine_uuids", {}).get("NaveedPC") ==
             "GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0" and
             decision.get("allowed_4080_conditions") == ["C0", "C1", "C2", "C5", "C6"] and
             decision.get("transfer_policy") == "researcher_approved_reference_profile_transfer" and
             decision.get("per_card_headroom_profile_required") is False and
             decision.get("common_schedule") == {"microbatch": 4, "accumulation": 16,
                                                 "effective_sequences": 64} and
             decision.get("c3_4080_status") == "not_approved_pending_reference_profile" and
             decision.get("c4_4080_status") == "prohibited" and
             decision.get("resume_uuid_policy") == "same_physical_uuid_only" and
             observed["hardware_qualification"] == "not_qualified_same_4080_capability_class" and
             all(attempts[(condition, 4)]["status"] == "complete_unsafe" and
                 attempts[(condition, 4)]["free_margin_bytes"] == -deficit
                 for condition, deficit in (("C2", 96025804), ("C5", 228498636))),
             "D19 or preserved nonreference measurements differ from researcher decision")
    return decision, digest


def _d20(repo: Path) -> tuple[dict, str, dict, str]:
    """Verify the fixed C3 evidence and D20's exact prospective decision."""
    repo = Path(repo)
    d19, d19_digest = _d19(repo)
    evidence, evidence_digest = _sealed(repo / D20_EVIDENCE_PATH)
    profiles = evidence.get("profiles", {})
    _require(set(profiles) == {"mb8", "mb4"}, "D20 needs both real C3 profiles")
    for mb in (8, 4):
        row = profiles[f"mb{mb}"]
        expected_path = f"reports/stage06_c3_4080_mb{mb}_raw.json"
        raw = (repo / expected_path).read_bytes()
        _require(row.get("raw_file") == expected_path and
                 row.get("raw_file_sha256") == hashlib.sha256(raw).hexdigest() and
                 row.get("profile") == json.loads(raw)["profile"],
                 f"D20 mb{mb} raw profile differs from committed evidence")
        profile = row["profile"]
        _require(profile["condition"] == "C3" and profile["device_role"] == "rtx4080super" and
                 profile["device_uuid"] == GPU_UUIDS["rtx4080super"] and
                 profile["microbatch"] == mb and profile["evidence"] == "measured_gpu" and
                 profile["optimizer_allocated"] is True and profile["evaluation_passed"] is True and
                 profile["save_passed"] is True and profile["failure_status"] is None and
                 profile["total_bytes"] == evidence["total_vram_bytes"] and
                 profile["headroom_bytes"] == max(1610612736, int(profile["total_bytes"] * .1)),
                 f"D20 mb{mb} lacks the registered production-shape cycle")
        safe = (profile["free_bytes"] >= profile["headroom_bytes"] and
                profile["total_bytes"] - profile["peak_reserved_bytes"] >= profile["headroom_bytes"])
        _require(profile["passed"] is safe and safe is (mb == 4),
                 f"D20 mb{mb} headroom verdict changed")
    artifact, _ = _sealed(repo / "protocols/artifact.lock.json")
    required_hashes = {"corpus": artifact["corpus"]["payload_sha256"],
                       "panels": artifact["panels"]["payload_sha256"],
                       "initialization": artifact["training_initialization"]["tensor_content_sha256"],
                       "order": artifact["order_seed0"]["payload_sha256"],
                       "teacher_weights": artifact["teacher"]["weights_sha256"],
                       "tokenizer": artifact["tokenizer"]["files_sha256"]}
    inventory, inventory_digest = _sealed(repo / "reports/stage06_production_artifact_inventory.json")
    environment = _read(repo / "protocols/environment.lock.json")
    _require(evidence.get("kind") == "s1-c3-rtx4080super-reference-qualification-v1" and
             evidence.get("status") == "qualified_at_registered_mb4_accum16" and
             evidence.get("gpu_model") == RTX4080_SUPER_MODEL and
             evidence.get("gpu_uuid") == GPU_UUIDS["rtx4080super"] and
             evidence.get("environment_lock_sha256") == environment["sha256"] and
             evidence.get("artifact_inventory_sha256") == inventory_digest and
             evidence.get("scientific_hashes") == required_hashes and
             evidence.get("condition") == "C3" and
             evidence.get("variant") == S1_VARIANTS["C3"] and
             evidence.get("mse_scale") == 68.00580071126464 and
             (evidence.get("microbatch"), evidence.get("accumulation"),
              evidence.get("effective_batch")) == (4, 16, 64) and
             evidence.get("mb4_checkpoint") == {
                 "path_kind": "external_engineering_profile", "step": 3,
                 "verified_load": True, "condition": "C3",
                 "gpu_uuid": GPU_UUIDS["rtx4080super"]},
             "D20 qualification differs from committed scientific artifacts or environment")
    decision, digest = _sealed(repo / D20_PATH)
    _require(decision.get("kind") == "s1-d20-c3-rtx4080super-successor-amendment-v1" and
             decision.get("decision_id") == "D20" and
             decision.get("status") == "approved_prospective_hardware_policy" and
             decision.get("predecessor_production_root") == D19_PROTOCOL_ROOT and
             decision.get("d19_amendment_sha256") == d19_digest and
             decision.get("reference_qualification_sha256") == evidence_digest and
             decision.get("reference_model") == RTX4080_SUPER_MODEL and
             decision.get("reference_uuid") == GPU_UUIDS["rtx4080super"] and
             decision.get("allowed_4080_conditions") == ["C0", "C1", "C2", "C3", "C5", "C6"] and
             decision.get("c4_4080_status") == "prohibited" and
             decision.get("c3_variant") == S1_VARIANTS["C3"] and
             decision.get("s_MSE") == 68.00580071126464 and
             decision.get("common_schedule") == d19["common_schedule"] and
             decision.get("mb8_headroom_passed") is False and
             decision.get("mb4_headroom_passed") is True and
             decision.get("transfer_policy") == d19["transfer_policy"] and
             decision.get("resume_uuid_policy") == d19["resume_uuid_policy"] and
             decision.get("per_card_headroom_profile_required") is False and
             decision.get("existing_d19_jobs") ==
             "existing_runs_and_preapproved_C5_queue_retain_D19_root_and_identity" and
             decision.get("prospective_only") is True and
             decision.get("recalibration") is False and
             decision.get("scientific_artifact_change") is False,
             "D20 amendment differs from measured reference or D19 transfer policy")
    return decision, digest, evidence, evidence_digest


def build_successor_component_locks(repo: Path) -> dict[str, dict]:
    """Carry forward measured locks and add only the prospective D19 policy."""
    repo = Path(repo)
    protocol_dir = repo / "protocols"
    old = {}
    for name in LOCK_NAMES[:-1]:
        path = protocol_dir / f"{name}.lock.json"
        current = _read(path)
        if name in ("environment", "hardware") and "predecessor_lock_sha256" in current["payload"]:
            predecessor_digest = current["payload"]["predecessor_lock_sha256"]
            path = protocol_dir / "superseded" / f"s1-{name}-{predecessor_digest}.json"
        old[name] = _read(path)
    for document in old.values():
        verify_envelope(document)
    d19, d19_digest = _d19(repo)
    reference = old["hardware"]["payload"]
    _require(reference["gpu_uuids"] == GPU_UUIDS and
             reference["proof"]["microbatch"] == 4 and
             reference["proof"]["accumulation"] == 16,
             "predecessor reference hardware proof differs")
    hardware_class = {
        "policy": d19["transfer_policy"], "model": RTX4080_SUPER_MODEL,
        "reference_uuid": GPU_UUIDS["rtx4080super"],
        "known_equivalent_uuids": d19["known_equivalent_machine_uuids"],
        "allowed_conditions": d19["allowed_4080_conditions"],
        "actual_uuid": "runtime_recorded", "resume_uuid": "same_physical_uuid_only",
        "per_card_headroom_profile_required": False,
        "microbatch": 4, "accumulation": 16,
    }
    environment = dict(old["environment"]["payload"])
    devices = {role: dict(value) for role, value in environment["devices"].items()}
    reference_gpu = dict(devices["rtx4080super"]["gpu"])
    reference_gpu["reference_uuid"] = reference_gpu.pop("uuid")
    devices["rtx4080super"]["gpu"] = reference_gpu
    environment["devices"] = devices
    environment.update(kind="s1-production-two-role-environment-lock-v2",
                       rtx4080super_environment_policy="exact_locked_software_reference_environment_independent_of_gpu_uuid",
                       rtx4080super_reference_uuid=GPU_UUIDS["rtx4080super"],
                       predecessor_lock_sha256=old["environment"]["sha256"])
    hardware = dict(reference)
    hardware.update(kind="s1-production-hardware-lock-v2",
                    evidence="measured_reference_plus_researcher_transfer",
                    predecessor_lock_sha256=old["hardware"]["sha256"],
                    predecessor_protocol_root=PREDECESSOR_PROTOCOL_ROOT,
                    d19_researcher_amendment_sha256=d19_digest,
                    nonreference_observed_profile_sha256=d19["naveed_observed_profile_report_sha256"],
                    nonreference_profile_interpretation=d19["measurement_interpretation"],
                    hardware_classes={"rtx4080super": hardware_class})
    return {"artifact": old["artifact"], "environment": seal_payload(environment),
            "hardware": seal_payload(hardware), "calibration": old["calibration"]}


def d20_reviewed_plan(d19_protocol: dict) -> dict:
    """Change precisely the unstarted C3 assignment, retaining nine physical jobs."""
    old, digest = validate_protocol_lock(d19_protocol)
    _require(digest == D19_PROTOCOL_ROOT, "D20 requires the exact D19 predecessor")
    jobs = copy.deepcopy(old["protocol"]["run_plan"]["jobs"])
    matches = [row for row in jobs if row["run_id"] == "s1-c3-seed0-rtx3090"]
    _require(len(jobs) == 9 and len(matches) == 1, "D19 nine-job plan differs")
    matches[0]["run_id"] = "s1-c3-seed0-rtx4080super"
    matches[0]["config"] = "s1/c3_rtx4080super.json"
    return {"schema_version": 1, "status": "approved", "study": "S1", "jobs": jobs,
            "optional_s3_enabled": False, "hardware_confounds": ["C4-C3-seed0"]}


def build_d20_lock_set(repo: Path, predecessor: dict[str, dict],
                       production_runtime_source_commit: str) -> tuple[dict[str, dict], dict]:
    """Derive D20 from the exact D19 lock set and one committed C3 measurement."""
    repo = Path(repo)
    _require(set(predecessor) == set(LOCK_NAMES) and
             all(verify_envelope(doc)[1] == doc["sha256"] for doc in predecessor.values()),
             "D20 predecessor lock set is incomplete")
    old, old_root = validate_protocol_lock(predecessor["protocol"])
    _require(old_root == D19_PROTOCOL_ROOT and
             all(old[f"{name}_lock_digest"] == predecessor[name]["sha256"]
                 for name in LOCK_NAMES[:-1]),
             "D20 predecessor is not the exact D19 transitive root")
    _d19(repo)
    d20, d20_digest, _, evidence_digest = _d20(repo)
    plan = d20_reviewed_plan(predecessor["protocol"])
    hardware = copy.deepcopy(predecessor["hardware"]["payload"])
    _require(hardware["hardware_classes"]["rtx4080super"]["allowed_conditions"] ==
             ["C0", "C1", "C2", "C5", "C6"] and
             hardware["condition_device_eligibility"]["C3"]["rtx4080super"]
             ["measured_production_eligible"] is False,
             "D19 C3 was already relabeled")
    hardware.update(kind="s1-production-hardware-lock-v3",
                    predecessor_lock_sha256=predecessor["hardware"]["sha256"],
                    predecessor_protocol_root=D19_PROTOCOL_ROOT,
                    d20_researcher_amendment_sha256=d20_digest,
                    c3_4080_qualification_sha256=evidence_digest,
                    reviewed_plan_sha256=payload_digest(plan),
                    reviewed_jobs=plan["jobs"])
    hardware["hardware_classes"]["rtx4080super"]["allowed_conditions"] = d20["allowed_4080_conditions"]
    hardware["condition_device_eligibility"]["C3"]["rtx3090"].update(
        current_plan_status="measured_not_scheduled", scheduled_in_current_plan=False)
    hardware["condition_device_eligibility"]["C3"]["rtx4080super"].update(
        current_plan_status="measured_scheduled", future_gate=None,
        measured_production_eligible=True, scheduled_in_current_plan=True)
    hardware["unmeasured_not_required_for_current_plan"] = [
        pair for pair in hardware["unmeasured_not_required_for_current_plan"]
        if pair != ["C3", "rtx4080super"]]
    hardware_document = seal_payload(hardware)
    current = copy.deepcopy(old)
    current["source_commit"] = production_runtime_source_commit
    current["production_runtime_source_commit"] = production_runtime_source_commit
    current["hardware_lock_digest"] = hardware_document["sha256"]
    current["allowed_device_roles"]["C3"] = ["rtx4080super"]
    current["approval"].update(approved_on_utc_date=d20["approved_on_utc_date"],
                               d20_sha256=d20_digest,
                               decision_ids=[*old["approval"]["decision_ids"], "D20"])
    body = current["protocol"]
    body["predecessor_protocol_root_sha256"] = D19_PROTOCOL_ROOT
    body["d20_researcher_amendment_sha256"] = d20_digest
    body["c3_4080_qualification_sha256"] = evidence_digest
    body["amendment_chain"] = [
        {"decision_id": "D19", "root_sha256": D19_PROTOCOL_ROOT,
         "amendment_sha256": old["approval"]["d19_sha256"]},
        {"decision_id": "D20", "predecessor_root_sha256": D19_PROTOCOL_ROOT,
         "amendment_sha256": d20_digest}]
    body["hardware"].update(
        hardware_classes=hardware["hardware_classes"],
        condition_device_eligibility=hardware["condition_device_eligibility"],
        reviewed_plan_sha256=payload_digest(plan))
    body["run_plan"].update(reviewed_plan_sha256=payload_digest(plan), jobs=plan["jobs"])
    protocol_document = seal_payload(current)
    validate_protocol_lock(protocol_document)
    _require(current["protocol"]["objectives"]["s_MSE"] == d20["s_MSE"] and
             all(current["protocol"][key] == old["protocol"][key] for key in
                 ("teacher", "student", "teacher_layers_for_student", "data", "training",
                  "objectives", "evaluation", "checkpointing", "calibration",
                  "optional_studies", "researcher_amendment_sha256")),
             "D20 changed a scientific field or frozen C3 scale")
    return {**{name: predecessor[name] for name in
               ("artifact", "environment", "calibration")},
            "hardware": hardware_document, "protocol": protocol_document}, plan


def validate_calibration_result(result: dict, inventory: dict, candidate: dict,
                                environment: dict) -> dict[str, float]:
    """Recompute every registered ratio and median before accepting factors."""
    _require(result.get("kind") == "s1-c3-c4-raw-gradient-calibration-v1" and
             result.get("status") == "measured_on_approved_rtx3090" and
             result.get("source_commit") == CALIBRATION_SOURCE_COMMIT,
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
        "raw_result_sha256": result_digest, "source_commit": CALIBRATION_SOURCE_COMMIT,
        "seed": 1729, "gpu": result["gpu"],
        "environment_manifest_sha256": digest3090, "environment": result["environment"],
        "input_hashes": {k: v for k, v in result.items() if k.endswith("sha256")},
        "numerics": result["numerics"], "raw_norms": result["raw_norms"],
        "ratios": result["ratios"], "factors": factors,
        "c2_c5_c6_scale": 1})
    return {"artifact": artifact, "environment": environment,
            "hardware": hardware, "calibration": calibration}


def build_protocol_lock(repo: Path, components: dict[str, dict],
                        *, fingerprint_denominator_floor: float,
                        production_runtime_source_commit: str) -> dict:
    """Seal the production root only with an explicitly approved metric floor."""
    _require(type(fingerprint_denominator_floor) in (int, float) and
             math.isfinite(fingerprint_denominator_floor) and
             fingerprint_denominator_floor > 0,
             "positive approved fingerprint denominator floor required")
    _require(isinstance(production_runtime_source_commit, str) and
             COMMIT_PATTERN.fullmatch(production_runtime_source_commit) is not None and
             production_runtime_source_commit != CALIBRATION_SOURCE_COMMIT,
             "distinct full production runtime source commit required")
    _require(set(components) == {"artifact", "environment", "hardware", "calibration"},
             "all four measured component locks required")
    artifact, environment, hardware, calibration = (
        verify_envelope(components[name])[0] for name in
        ("artifact", "environment", "hardware", "calibration"))
    approval, approval_digest = _sealed(Path(repo) / "protocols/s1_researcher_approval_20260928.json")
    _, amendment_digest = _sealed(Path(repo) / "protocols/s1_researcher_amendment_20260928.json")
    d19, d19_digest = _d19(Path(repo))
    plan = _read(Path(repo) / "configs/s1_jobs_seed0_reviewed.json")
    _require(approval["amendment_sha256"] == amendment_digest and
             approval["reviewed_seed0_plan_sha256"] == payload_digest(plan) and
             approval["mandatory_seeds"] == [0] and approval["optional_seeds"] == [1, 2] and
             approval["default_optimizer_updates"] == 10000 and
             hardware["reviewed_plan_sha256"] == payload_digest(plan) and
             hardware.get("d19_researcher_amendment_sha256") == d19_digest and
             hardware.get("predecessor_protocol_root") == PREDECESSOR_PROTOCOL_ROOT and
             hardware.get("hardware_classes", {}).get("rtx4080super", {}).get("reference_uuid") ==
             d19["reference_uuid"],
             "researcher approval or reviewed job plan changed")
    allowed = {condition: sorted(role for role in GPU_UUIDS
                  if hardware["condition_device_eligibility"][condition][role]["scheduled_in_current_plan"])
               for condition in S1_VARIANTS}
    teacher, student = MODELS["S1"]
    teacher_layers = [(s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1)
                      for s in range(24)]
    protocol = {
        "production_config_binding_schema": 3,
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
                     "hardware_classes": hardware["hardware_classes"],
                     "reviewed_plan_sha256": hardware["reviewed_plan_sha256"],
                     "candidate_sha256": hardware["candidate_sha256"],
                     "condition_device_eligibility": hardware["condition_device_eligibility"],
                     "c4_4080_prohibited": True},
        "calibration": {"lock_sha256": components["calibration"]["sha256"],
                        "source_commit": CALIBRATION_SOURCE_COMMIT,
                        "calibration_source_commit": CALIBRATION_SOURCE_COMMIT,
                        "seed": 1729,
                        "result_sha256": calibration["raw_result_sha256"],
                        "s_MSE": calibration["factors"]["mse"],
                        "s_REL": calibration["factors"]["rel"],
                        "c2_c5_c6_scale": 1},
        "run_plan": {"reviewed_plan_sha256": payload_digest(plan),
                     "jobs": plan["jobs"], "physical_jobs": 9,
                     "unique_condition_seed_pairs": 7,
                     "bridge_conditions": ["C1", "C2"]},
        "researcher_amendment_sha256": amendment_digest,
        "d19_researcher_amendment_sha256": d19_digest,
        "predecessor_protocol_root_sha256": PREDECESSOR_PROTOCOL_ROOT,
        "optional_studies": {"S3": False, "S6_long_context": False},
    }
    document = seal_payload({
        "status": "approved", "production_ready": True, "study": "S1",
        "condition_variants": S1_VARIANTS, "allowed_device_roles": allowed,
        "source_commit": production_runtime_source_commit,
        "production_runtime_source_commit": production_runtime_source_commit,
        "calibration_source_commit": CALIBRATION_SOURCE_COMMIT,
        "execution_critical_path_set_version": EXECUTION_CRITICAL_PATH_SET_VERSION,
        "approval": {"researcher": "user-provided researcher decisions; identity not asserted",
                     "approved_at_utc": None, "approved_on_utc_date": "2026-09-28",
                     "approval_sha256": approval_digest,
                     "d19_sha256": d19_digest,
                     "decision_ids": sorted(approval["approved_decisions"]) + ["D19"]},
        "artifact_lock_digest": components["artifact"]["sha256"],
        "environment_lock_digest": components["environment"]["sha256"],
        "hardware_lock_digest": components["hardware"]["sha256"],
        "calibration_lock_digest": components["calibration"]["sha256"],
        "protocol": protocol,
    })
    validate_protocol_lock(document)
    return document


def validate_scientific_unchanged(predecessor: dict, successor: dict) -> None:
    """D19 may change hardware authorization, never S1 numerical inputs."""
    old, _ = validate_protocol_lock(predecessor)
    new, _ = validate_protocol_lock(successor)
    fixed = ("teacher", "student", "teacher_layers_for_student", "data", "training",
             "objectives", "evaluation", "checkpointing", "calibration", "run_plan",
             "optional_studies", "researcher_amendment_sha256")
    _require(all(old["protocol"][key] == new["protocol"][key] for key in fixed) and
             old["condition_variants"] == new["condition_variants"] and
             old["allowed_device_roles"] == new["allowed_device_roles"] and
             old["artifact_lock_digest"] == new["artifact_lock_digest"] and
             old["calibration_lock_digest"] == new["calibration_lock_digest"],
             "D19 successor changes a scientific field or condition assignment")


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


def write_lock_set(directory: Path, documents: dict[str, dict],
                   *, supersede_root_sha256: str | None = None) -> None:
    """Write the protocol root after verifying all measured component locks."""
    _require(set(documents) == set(LOCK_NAMES), "complete five-lock set required")
    write_component_locks(directory, {name: documents[name] for name in LOCK_NAMES[:-1]})
    target = Path(directory) / "protocol.lock.json"
    verify_envelope(documents["protocol"])
    if target.exists():
        previous = _read(target)
        if previous == documents["protocol"]:
            return
        _require(supersede_root_sha256 == SUPERSEDED_PROTOCOL_ROOT and
                 previous.get("sha256") == SUPERSEDED_PROTOCOL_ROOT,
                 "existing protocol lock differs; refuse overwrite")
        verify_envelope(previous)
        historical = directory / "superseded" / f"s1-protocol-{SUPERSEDED_PROTOCOL_ROOT}.json"
        historical.parent.mkdir(parents=True, exist_ok=True)
        if historical.exists():
            _require(_read(historical) == previous,
                     "superseded historical protocol differs; refuse overwrite")
        else:
            historical.write_bytes(canonical_json_bytes(previous) + b"\n")
        target.write_bytes(canonical_json_bytes(documents["protocol"]) + b"\n")
    else:
        target.write_bytes(canonical_json_bytes(documents["protocol"]) + b"\n")


def write_successor_lock_set(directory: Path, documents: dict[str, dict]) -> None:
    """Archive the exact predecessor locks before replacing only policy locks."""
    directory = Path(directory)
    _require(set(documents) == set(LOCK_NAMES), "complete successor lock set required")
    current = {name: _read(directory / f"{name}.lock.json") for name in LOCK_NAMES}
    if current == documents:
        return
    previous = dict(current)
    for name in ("environment", "hardware"):
        if current[name] == documents[name]:
            digest = documents[name]["payload"]["predecessor_lock_sha256"]
            previous[name] = _read(directory / "superseded" / f"s1-{name}-{digest}.json")
    if current["protocol"] == documents["protocol"]:
        previous["protocol"] = _read(directory / "superseded" /
                                     f"s1-protocol-{PREDECESSOR_PROTOCOL_ROOT}.json")
    for document in (*previous.values(), *documents.values()):
        verify_envelope(document)
    _require(previous["protocol"]["sha256"] == PREDECESSOR_PROTOCOL_ROOT and
             documents["protocol"]["payload"]["protocol"].get("predecessor_protocol_root_sha256") ==
             PREDECESSOR_PROTOCOL_ROOT and
             previous["artifact"] == documents["artifact"] and
             previous["calibration"] == documents["calibration"] and
             documents["environment"]["payload"]["predecessor_lock_sha256"] ==
             previous["environment"]["sha256"] and
             documents["hardware"]["payload"]["predecessor_lock_sha256"] ==
             previous["hardware"]["sha256"],
             "successor lock set does not preserve exact predecessor inputs")
    historical = directory / "superseded"
    historical.mkdir(parents=True, exist_ok=True)
    for name in ("environment", "hardware", "protocol"):
        destination = (historical / f"s1-protocol-{PREDECESSOR_PROTOCOL_ROOT}.json" if name == "protocol"
                       else historical / f"s1-{name}-{previous[name]['sha256']}.json")
        if destination.exists():
            _require(_read(destination) == previous[name], "historical lock differs; refuse overwrite")
        else:
            destination.write_bytes(canonical_json_bytes(previous[name]) + b"\n")
    for name in ("environment", "hardware", "protocol"):
        (directory / f"{name}.lock.json").write_bytes(canonical_json_bytes(documents[name]) + b"\n")


def validate_final_lock_set(directory: Path) -> dict[str, dict]:
    """Check a lock directory's canonical envelopes and transitive identities."""
    directory = Path(directory)
    documents = {name: _read(directory / f"{name}.lock.json") for name in LOCK_NAMES}
    payloads = {name: verify_envelope(document)[0] for name, document in documents.items()}
    protocol, _ = validate_protocol_lock(documents["protocol"])
    if "d20_researcher_amendment_sha256" in protocol["protocol"]:
        historical = directory / "superseded"
        old_protocol = _read(historical / f"s1-protocol-{D19_PROTOCOL_ROOT}.json")
        old_payload, old_root = validate_protocol_lock(old_protocol)
        _require(old_root == D19_PROTOCOL_ROOT and
                 old_payload["protocol"].get("d20_researcher_amendment_sha256") is None,
                 "D20 historical D19 root differs")
        old_hardware = _read(historical / f"s1-hardware-{old_payload['hardware_lock_digest']}.json")
        predecessor = {**{name: documents[name] for name in
                         ("artifact", "environment", "calibration")},
                       "hardware": old_hardware, "protocol": old_protocol}
        expected, plan = build_d20_lock_set(directory.parent, predecessor,
                                             protocol["production_runtime_source_commit"])
        _require(documents == expected and
                 _read(directory.parent / "configs/s1_jobs_seed0_reviewed.json") == plan,
                 "D20 successor differs from the exact D19 root, measured C3 evidence or reviewed plan")
        validate_eligibility_matrix(old_hardware["payload"]["proof"])
        return documents
    _require(protocol.get("production_runtime_source_commit") == protocol.get("source_commit") and
             protocol.get("calibration_source_commit") == CALIBRATION_SOURCE_COMMIT and
             protocol.get("execution_critical_path_set_version") == EXECUTION_CRITICAL_PATH_SET_VERSION and
             protocol["protocol"].get("production_config_binding_schema") in (2, 3) and
             protocol["protocol"].get("calibration", {}).get("calibration_source_commit") ==
             CALIBRATION_SOURCE_COMMIT,
             "old production root is superseded: runtime source provenance binding is absent")
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
    class_binding = protocol["protocol"]["production_config_binding_schema"] == 3
    expected_devices = {}
    for role, (record, _) in env_records.items():
        gpu = dict(record["gpu"])
        if class_binding and role == "rtx4080super":
            gpu["reference_uuid"] = gpu.pop("uuid")
        expected_devices[role] = {
            "gpu": gpu, "os": record["os"], "python": record["python"],
            "torch": record["torch"], "cuda_runtime": record["torch_cuda_runtime"],
            "cudnn": record["cudnn_version"], "transformers": record["transformers"]}
    _require(environment["rtx4080super_manifest_sha256"] == env_records["rtx4080super"][1] and
             environment["rtx3090_manifest_sha256"] == env_records["rtx3090"][1] and
             environment["lock_managed_distributions"] == env_records["rtx4080super"][0]["installed_locked_distributions"] ==
             env_records["rtx3090"][0]["installed_locked_distributions"] and
             environment["numerical_policy"] == env_records["rtx4080super"][0]["numerical_policy"] and
             environment["devices"] == expected_devices and
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
             calibration["source_commit"] == CALIBRATION_SOURCE_COMMIT and
             calibration["environment_manifest_sha256"] == environment["rtx3090_manifest_sha256"] and
             calibration["input_hashes"]["corpus_sha256"] == artifact["corpus"]["payload_sha256"] and
             calibration["input_hashes"]["frozen_panels_sha256"] == artifact["panels"]["payload_sha256"] and
             protocol["production_runtime_source_commit"] != CALIBRATION_SOURCE_COMMIT and
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
    if protocol["protocol"]["production_config_binding_schema"] == 3:
        d19, d19_digest = _d19(repo)
        class_policy = hardware.get("hardware_classes", {}).get("rtx4080super")
        _require(protocol["approval"].get("d19_sha256") == d19_digest and
                 protocol["protocol"].get("d19_researcher_amendment_sha256") == d19_digest and
                 protocol["protocol"].get("predecessor_protocol_root_sha256") ==
                 PREDECESSOR_PROTOCOL_ROOT and
                 hardware.get("d19_researcher_amendment_sha256") == d19_digest and
                 hardware.get("predecessor_protocol_root") == PREDECESSOR_PROTOCOL_ROOT and
                 hardware.get("evidence") == "measured_reference_plus_researcher_transfer" and
                 hardware.get("nonreference_observed_profile_sha256") ==
                 d19["naveed_observed_profile_report_sha256"] and
                 class_policy == protocol["protocol"]["hardware"].get("hardware_classes", {}).get("rtx4080super") and
                 class_policy.get("policy") == d19["transfer_policy"] and
                 class_policy.get("model") == RTX4080_SUPER_MODEL and
                 class_policy.get("reference_uuid") == GPU_UUIDS["rtx4080super"] and
                 class_policy.get("known_equivalent_uuids") == d19["known_equivalent_machine_uuids"] and
                 class_policy.get("allowed_conditions") == d19["allowed_4080_conditions"] and
                 class_policy.get("actual_uuid") == "runtime_recorded" and
                 class_policy.get("resume_uuid") == "same_physical_uuid_only" and
                 environment.get("rtx4080super_environment_policy") ==
                 "exact_locked_software_reference_environment_independent_of_gpu_uuid" and
                 hardware["proof"] == candidate["proof"],
                 "D19 reference transfer or preserved measured proof differs")
        predecessor = directory / "superseded" / f"s1-protocol-{PREDECESSOR_PROTOCOL_ROOT}.json"
        _require(predecessor.exists(), "historical predecessor production root is missing")
        historical = _read(predecessor)
        _require(verify_envelope(historical)[1] == PREDECESSOR_PROTOCOL_ROOT and
                 historical["payload"]["protocol"]["production_config_binding_schema"] == 2,
                 "historical predecessor production root differs")
    return documents


def build_production_configs(repo: Path, protocol_document: dict,
                             *, reviewed_plan: dict | None = None) -> tuple[dict[str, dict], dict]:
    """Derive exactly the reviewed nine seed-0 configs; do not launch them."""
    protocol, root_digest = validate_protocol_lock(protocol_document)
    _require(root_digest == protocol_document["sha256"] and
             protocol.get("source_commit") == protocol.get("production_runtime_source_commit") and
             protocol.get("calibration_source_commit") == CALIBRATION_SOURCE_COMMIT and
             protocol["protocol"].get("production_config_binding_schema") in (2, 3),
             "final S1 protocol binding schema is absent")
    reviewed = (reviewed_plan if reviewed_plan is not None else
                _read(Path(repo) / "configs/s1_jobs_seed0_reviewed.json"))
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


def write_production_configs(repo: Path, configs: dict[str, dict], plan: dict,
                             *, supersede_root_sha256: str | None = None) -> None:
    root = Path(repo) / "configs"
    paths = {name: root / name for name in configs}
    plan_path = root / "production/s1_jobs_seed0.json"
    _require(len(paths) == 9, "exactly nine production configs required")
    for name, path in paths.items():
        if path.exists():
            previous = _read(path)
            if previous == configs[name]:
                continue
            _require((supersede_root_sha256 == SUPERSEDED_PROTOCOL_ROOT and
                      previous.get("protocol_digest") == SUPERSEDED_PROTOCOL_ROOT and
                      previous.get("production_binding", {}).get("source_commit") ==
                      CALIBRATION_SOURCE_COMMIT) or
                     (supersede_root_sha256 == PREDECESSOR_PROTOCOL_ROOT and
                      previous.get("protocol_digest") == PREDECESSOR_PROTOCOL_ROOT),
                     f"existing production config {name} differs; refuse overwrite")
            if supersede_root_sha256 == PREDECESSOR_PROTOCOL_ROOT:
                historical = root / "superseded" / "s1" / Path(name).name
                historical.parent.mkdir(parents=True, exist_ok=True)
                if historical.exists():
                    _require(_read(historical) == previous,
                             f"historical config {name} differs; refuse overwrite")
                else:
                    historical.write_bytes(canonical_json_bytes(previous) + b"\n")
            path.write_bytes(canonical_json_bytes(configs[name]) + b"\n")
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
    d20 = "d20_researcher_amendment_sha256" in protocol_document["payload"]["protocol"]
    expected_confounds = ["C4-C3-seed0"] if d20 else []
    _require(result["physical_runs"] == 9 and result["unique_condition_seed_pairs"] == 7 and
             result["seeds"] == [0] and result["hardware_confounds"] == expected_confounds and
             result["launchable"] is True,
             "production nine-job plan does not validate")
    return result
