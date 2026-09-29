"""Strict one-run configuration. No default seed or experiment sweep."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .provenance import LockError, SHA256_PATTERN, validate_protocol_lock


S1_VARIANTS = {
    "C0": "ce_only", "C1": "logit_kd", "C2": "cosine_soft_jsd_v2",
    "C3": "head_mean_probability_mse_v1", "C4": "causal_qq_kk_vv_v1",
    "C5": "cosine_soft_conditional_nosink_jsd_v1",
    "C6": "cosine_soft_binary_sink_jsd_v1",
}
S3_VARIANTS = {
    "C0": "ce_only", "C1": "logit_kd", "C2": "index_jsd_v1",
    "C3": "index_probability_mse_v1", "C4": "causal_qq_kk_vv_v1",
}
VARIANTS = {"S1": S1_VARIANTS, "S3": S3_VARIANTS}
MODELS = {
    "S1": ({"layers": 36, "heads": 20, "width": 1280},
           {"layers": 24, "heads": 16, "width": 1024, "initialization": "random_from_config"}),
    "S3": ({"layers": 12, "heads": 12, "width": 768},
           {"layers": 6, "heads": 12, "width": 768, "initialization": "random_from_config"}),
}
DEVICE_ROLES = {"rtx3090", "rtx4080super"}


class ConfigError(ValueError):
    """A run request is ambiguous, unsupported, or unapproved."""


@dataclass(frozen=True, slots=True)
class RunSpec:
    study: str
    condition: str
    variant: str
    seed: int
    device_role: str
    protocol_digest: str | None


def production_binding_for(payload: Mapping[str, Any], device_role: str, seed: int) -> dict[str, Any]:
    """The immutable per-job identities a final S1 config must repeat exactly."""
    body = payload["protocol"]
    schema = body.get("production_config_binding_schema")
    if schema in (3, 4) and device_role == "rtx4080super":
        hardware_binding = body["hardware"]["hardware_classes"][device_role]
        identity = {"hardware_binding": {
            "policy": "researcher_approved_reference_profile_transfer",
            "class": device_role,
            "model": hardware_binding["model"],
            "reference_uuid": hardware_binding["reference_uuid"],
            "actual_uuid": "runtime_recorded",
        }}
    elif schema in (2, 3, 4):
        identity = {"gpu_uuid": body["hardware"]["gpu_uuids"][device_role]}
    else:
        raise KeyError("unsupported production binding schema")
    if schema == 4 and seed in (1, 2):
        replica = body["seed_replications"][str(seed)]
        data_binding = {
            "initialization_sha256": replica["initialization_sha256"],
            "order_sha256": replica["order_sha256"],
            "corpus_sha256": replica["corpus_sha256"],
            "panels_sha256": replica["panels_sha256"],
            "replication_evidence_sha256": body["seed_replications_sha256"],
        }
    else:
        data_binding = {
            "seed0_initialization_sha256": body["data"]["seed0_initialization_sha256"],
            "seed0_order_sha256": body["data"]["seed0_order_sha256"],
            "corpus_sha256": body["data"]["production_corpus_sha256"],
            "panels_sha256": body["data"]["frozen_panels_sha256"],
        }
    return {
        "seed": seed,
        **identity,
        "production_runtime_source_commit": payload["production_runtime_source_commit"],
        "execution_critical_path_set_version": payload["execution_critical_path_set_version"],
        "artifact_lock_digest": payload["artifact_lock_digest"],
        "environment_lock_digest": payload["environment_lock_digest"],
        "hardware_lock_digest": payload["hardware_lock_digest"],
        "calibration_lock_digest": payload["calibration_lock_digest"],
        **data_binding,
        "microbatch": body["training"]["microbatch"],
        "accumulation": body["training"]["accumulation"],
        "effective_batch": body["training"]["effective_sequences"],
        "sequence_length": body["training"]["sequence_length"],
        "optimizer_updates": body["training"]["primary_optimizer_updates"],
    }


def resolve_config(
    raw: Mapping[str, Any], *, seed: int | None,
    protocol_lock: Mapping[str, Any] | None = None, production: bool = False,
) -> RunSpec:
    fields = {
        "schema_version", "study", "condition", "variant", "device_role",
        "teacher", "student", "protocol_digest",
    }
    if not isinstance(raw, dict) or set(raw) not in (fields, fields | {"production_binding"}):
        raise ConfigError("run config has missing or unknown fields")
    if type(raw["schema_version"]) is not int or raw["schema_version"] != 1:
        raise ConfigError("unsupported run config schema version")
    if type(seed) is not int or seed < 0:
        raise ConfigError("one explicit nonnegative --seed is required")
    study, condition, variant = raw["study"], raw["condition"], raw["variant"]
    if not isinstance(study, str) or study not in VARIANTS:
        raise ConfigError("unsupported study for Stage 00 CLI")
    if not isinstance(condition, str) or condition not in VARIANTS[study]:
        raise ConfigError("condition is incompatible with study")
    if variant != VARIANTS[study][condition]:
        raise ConfigError("variant is incompatible with study and condition")
    for field, expected in zip(("teacher", "student"), MODELS[study]):
        observed = raw[field]
        if not isinstance(observed, dict) or set(observed) != set(expected) or any(
            type(observed[key]) is not type(value) or observed[key] != value
            for key, value in expected.items()
        ):
            raise ConfigError(f"{field} dimensions or initialization differ from {study} contract")
    device_role = raw["device_role"]
    if not isinstance(device_role, str) or device_role not in DEVICE_ROLES:
        raise ConfigError("unsupported device role")
    if condition == "C4" and device_role != "rtx3090":
        raise ConfigError("C4 profiling and production require rtx3090")
    digest = raw["protocol_digest"]
    if digest is not None and (not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest)):
        raise ConfigError("protocol_digest must be a SHA-256 digest or null")
    if protocol_lock is not None:
        try:
            payload, actual_digest = validate_protocol_lock(protocol_lock)
        except LockError as exc:
            raise ConfigError(str(exc)) from exc
        if digest != actual_digest:
            raise ConfigError("run config references a stale protocol lock")
        if payload["study"] != study or payload["condition_variants"].get(condition) != variant:
            raise ConfigError("run config differs from approved protocol")
        if payload["protocol"].get("teacher") != raw["teacher"] or payload["protocol"].get("student") != raw["student"]:
            raise ConfigError("model contract differs from approved protocol")
        if device_role not in payload["allowed_device_roles"].get(condition, []):
            raise ConfigError("device role is absent from approved protocol")
        if payload["protocol"].get("production_config_binding_schema") == 4 and seed != 0 and (
                condition != "C3" or device_role != "rtx4080super" or seed not in (1, 2)):
            raise ConfigError("D21 optional replication authorizes only C3 seeds 1 and 2 on RTX4080 SUPER")
        if study == "S3" and payload["protocol"].get("optional_studies", {}).get("S3") is not True:
            raise ConfigError("S3 requires separate approval")
        if payload["protocol"].get("production_config_binding_schema") in (2, 3, 4) and production:
            try:
                expected_binding = production_binding_for(payload, device_role, seed)
            except (KeyError, TypeError) as exc:
                raise ConfigError("approved production binding contract is incomplete") from exc
            if raw.get("production_binding") != expected_binding:
                raise ConfigError("production config binding differs from approved protocol")
        elif "production_binding" in raw:
            raise ConfigError("production binding requires its final production validator")
    elif production:
        raise ConfigError("production requires a verified approved protocol lock")
    elif "production_binding" in raw:
        raise ConfigError("production binding requires an approved protocol lock")
    if production and digest is None:
        raise ConfigError("production requires a pinned protocol digest")
    return RunSpec(study, condition, variant, seed, device_role, digest)
