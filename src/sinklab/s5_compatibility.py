"""Narrow D26 admission for the four historical S5 seed-0 roots."""

from __future__ import annotations

from .provenance import payload_digest, verify_envelope


D26_PATH = "protocols/s1_researcher_amendment_d26_s5_mixed_roots_20261005.json"
D26_SHA256 = "888b21509000b570c83c2574f5a6bfbc3d1ec2dc197e88d841d085bac222f182"
CONDITIONS = ("C1", "C2", "C5", "C6")
RUN_ROOTS = {
    "C1": "48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791",
    "C2": "910961fcc53edaed0df48e3139dbb7ca2e058bf7480b67dab27bae0bcd90f26f",
    "C5": "fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552",
    "C6": "48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791",
}
OBJECTIVE_VARIANTS = {
    "C1": "logit_kd",
    "C2": "cosine_soft_jsd_v2",
    "C5": "cosine_soft_conditional_nosink_jsd_v1",
    "C6": "cosine_soft_binary_sink_jsd_v1",
}


class S5CompatibilityError(ValueError):
    pass


def validate_d26(document: dict) -> dict:
    """Verify the exact, sealed, four-run S5-only mixed-root decision."""
    try:
        payload, digest = verify_envelope(document)
    except (TypeError, ValueError) as exc:
        raise S5CompatibilityError(f"invalid D26 compatibility envelope: {exc}") from exc
    if digest != D26_SHA256:
        raise S5CompatibilityError("unapproved S5 mixed-root compatibility amendment")
    if (payload.get("decision_id") != "D26" or
            payload.get("kind") != "s5_seed0_mixed_historical_root_compatibility_v1" or
            payload.get("status") != "approved_prospective" or
            payload.get("scope", {}).get("study") != "S5" or
            payload.get("scope", {}).get("seed") != 0 or
            tuple(payload.get("scope", {}).get("conditions", ())) != CONDITIONS or
            payload.get("allowed_protocol_roots_by_condition") != RUN_ROOTS or
            payload.get("condition_objective_variants") != OBJECTIVE_VARIANTS):
        raise S5CompatibilityError("D26 scope or exact source-root allowlist differs")
    critical = payload.get("critical_invariants")
    if (not isinstance(critical, dict) or
            payload.get("critical_invariants_sha256") != payload_digest(critical)):
        raise S5CompatibilityError("D26 comparison-critical invariant seal is invalid")
    allowed_runs = payload.get("allowed_runs")
    if not isinstance(allowed_runs, dict) or set(allowed_runs) != set(CONDITIONS):
        raise S5CompatibilityError("D26 must identify exactly four approved source runs")
    for condition in CONDITIONS:
        expected = allowed_runs[condition]
        if (expected.get("condition") != condition or expected.get("seed") != 0 or
                expected.get("protocol_root_sha256") != RUN_ROOTS[condition] or
                expected.get("objective_variant") != OBJECTIVE_VARIANTS[condition] or
                expected.get("device_role") != "rtx4080super" or
                expected.get("gpu_model") != "NVIDIA GeForce RTX 4080 SUPER"):
            raise S5CompatibilityError(f"D26 source binding is invalid for {condition}")
    return {**payload, "sha256": digest}


def build_critical_invariants(*, protocol_payload: dict, identity: dict,
                              environment_payload: dict, artifact_sha256: str,
                              panel_sha256: str, panel_payload: dict,
                              metric_version: str) -> dict:
    """Rebuild the D26 comparison descriptor from source locks, never results."""
    protocol = protocol_payload["protocol"]
    data = protocol["data"]
    role = environment_payload["devices"]["rtx4080super"]
    role_runtime = {
        "cuda_runtime": role["cuda_runtime"],
        "cudnn": role["cudnn"],
        "gpu_model": role["gpu"]["model"],
        "gpu_driver": role["gpu"]["nvidia_driver"],
        "gpu_total_vram_mib": role["gpu"]["total_vram_mib"],
        "os": role["os"],
        "python": role["python"],
        "torch": role["torch"],
        "transformers": role["transformers"],
    }
    return {
        "study": "S1",
        "training_seed": 0,
        "initialization_sha256": identity["init_hash"],
        "student_model_sha256": identity["model_hash"],
        "data_sha256": identity["data_hash"],
        "data_order_sha256": data["seed0_order_sha256"],
        "data_identity": {key: data[key] for key in (
            "dataset_id", "dataset_revision", "recipe", "production_corpus_sha256",
            "frozen_panels_sha256", "tokenizer_revision", "tokenizer_files_sha256",
            "source_windows")},
        "architecture": {
            "student": protocol["student"],
            "teacher": protocol["teacher"],
            "teacher_layers_for_student": protocol["teacher_layers_for_student"],
        },
        "training_schedule": protocol["training"],
        "objective_parameters_and_registered_variants": protocol["objectives"],
        "evaluation_protocol": protocol["evaluation"],
        "calibration_identity": protocol["calibration"],
        "artifact_lock_sha256": artifact_sha256,
        "panel_manifest_sha256": panel_sha256,
        "panel_item_id_sha256": {
            name: payload_digest({"item_ids": panel_payload[name]})
            for name in ("owt_dense64", "owt_full300")
        },
        "metric_version": metric_version,
        "evaluation_record_schema": {
            "envelope_schema_version": 1,
            "item_record_schema_version": metric_version,
            "aggregate_metric_version": metric_version,
        },
        "layer_scope": {
            "student_scope": "all_native_layers",
            "student_layer_indices": list(range(protocol["student"]["layers"])),
            "teacher_scope": "mapped_teacher_layers",
            "teacher_layer_indices": protocol["teacher_layers_for_student"],
        },
        "device_role": "rtx4080super",
        "gpu_model": "NVIDIA GeForce RTX 4080 SUPER",
        "precision": "bf16",
        "backend": "eager",
        "numerical_policy": environment_payload["numerical_policy"],
        "environment_compatibility": {
            "package_map_sha256": payload_digest(environment_payload["lock_managed_distributions"]),
            "package_maps_equal": True,
            "uv_lock_sha256": environment_payload["uv_lock_sha256"],
            "numerical_policy_equal": True,
            "rtx4080_runtime_identity": role_runtime,
            "role_runtime_equal": True,
        },
    }
