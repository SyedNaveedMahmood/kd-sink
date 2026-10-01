"""Canonical payload hashing and approval lock validation.

No scientific values are chosen here. A lock is only a transport and integrity
boundary for an externally approved, measured protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import date
from pathlib import Path
from typing import Any, Mapping


SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")


class LockError(ValueError):
    """A lock is malformed, stale, or lacks production approval."""


def _no_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LockError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def canonical_json_bytes(payload: Mapping[str, Any]) -> bytes:
    if not isinstance(payload, dict):
        raise LockError("lock payload must be a JSON object")
    try:
        return json.dumps(
            payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise LockError("lock payload must contain only finite JSON values") from exc


def payload_digest(payload: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def seal_payload(payload: dict[str, Any]) -> dict[str, Any]:
    if "sha256" in payload:
        raise LockError("digest field belongs outside the hashed payload")
    return {"schema_version": 1, "payload": payload, "sha256": payload_digest(payload)}


def verify_envelope(document: Any) -> tuple[dict[str, Any], str]:
    if not isinstance(document, dict) or set(document) != {"schema_version", "payload", "sha256"}:
        raise LockError("lock envelope has missing or unknown fields")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        raise LockError("unsupported lock schema version")
    payload, digest = document["payload"], document["sha256"]
    if not isinstance(payload, dict) or not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
        raise LockError("invalid lock payload or digest")
    if "sha256" in payload or payload_digest(payload) != digest:
        raise LockError("lock payload hash mismatch")
    return payload, digest


def validate_protocol_lock(document: Any) -> tuple[dict[str, Any], str]:
    payload, digest = verify_envelope(document)
    required = {
        "status", "production_ready", "study", "condition_variants", "allowed_device_roles",
        "source_commit", "approval", "artifact_lock_digest",
        "environment_lock_digest", "hardware_lock_digest",
        "calibration_lock_digest", "protocol",
    }
    runtime_fields = {"production_runtime_source_commit", "calibration_source_commit",
                      "execution_critical_path_set_version"}
    if set(payload) not in (required, required | runtime_fields):
        raise LockError("protocol lock has missing or unknown fields")
    if payload["status"] != "approved" or payload["production_ready"] is not True:
        raise LockError("production requires an approved protocol")
    if not isinstance(payload["source_commit"], str) or not COMMIT_PATTERN.fullmatch(payload["source_commit"]):
        raise LockError("source_commit must be an immutable full commit ID")
    if runtime_fields <= set(payload):
        runtime = payload["production_runtime_source_commit"]
        calibration = payload["calibration_source_commit"]
        if (not isinstance(runtime, str) or not COMMIT_PATTERN.fullmatch(runtime) or
                payload["source_commit"] != runtime):
            raise LockError("production runtime source binding is invalid")
        if not isinstance(calibration, str) or not COMMIT_PATTERN.fullmatch(calibration):
            raise LockError("calibration source binding is invalid")
        if type(payload["execution_critical_path_set_version"]) is not int or (
                payload["execution_critical_path_set_version"] != 1):
            raise LockError("execution-critical path-set version is unsupported")
    for field in ("artifact_lock_digest", "environment_lock_digest", "hardware_lock_digest", "calibration_lock_digest"):
        value = payload[field]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise LockError(f"{field} must be a SHA-256 digest")
    approval = payload["approval"]
    base_approval = {"researcher", "approved_at_utc", "approval_sha256", "decision_ids"}
    d19_approval = base_approval | {"approved_on_utc_date", "d19_sha256"}
    d20_approval = d19_approval | {"d20_sha256"}
    d21_approval = d20_approval | {"d21_sha256"}
    d22_approval = d21_approval | {"d22_sha256"}
    if not isinstance(approval, dict) or set(approval) not in (
            base_approval, base_approval | {"approved_on_utc_date"}, d19_approval,
            d20_approval, d21_approval, d22_approval):
        raise LockError("approval record is incomplete")
    if not isinstance(approval["researcher"], str) or not approval["researcher"].strip():
        raise LockError("researcher approval is missing")
    if "approved_on_utc_date" in approval:
        if approval["approved_at_utc"] is not None:
            raise LockError("date-precision approval must not assert an exact UTC time")
        value = approval["approved_on_utc_date"]
        try:
            if not isinstance(value, str) or date.fromisoformat(value).isoformat() != value:
                raise ValueError
        except ValueError as exc:
            raise LockError("approval UTC date is invalid") from exc
    elif not isinstance(approval["approved_at_utc"], str) or not approval["approved_at_utc"].endswith("Z"):
        raise LockError("approval UTC timestamp is missing")
    if not isinstance(approval["approval_sha256"], str) or not SHA256_PATTERN.fullmatch(approval["approval_sha256"]):
        raise LockError("approval evidence digest is missing")
    class_binding = (isinstance(payload.get("protocol"), dict) and
                     payload["protocol"].get("production_config_binding_schema") in (3, 4, 5))
    d20_binding = class_binding and "d20_researcher_amendment_sha256" in payload["protocol"]
    d21_binding = d20_binding and payload["protocol"].get("production_config_binding_schema") == 4
    d22_binding = d20_binding and payload["protocol"].get("production_config_binding_schema") == 5
    required_decisions = {f"D{number:02d}" for number in range(
        1, 23 if d22_binding else 22 if d21_binding else 21 if d20_binding else 20 if class_binding else 19)}
    if class_binding and (set(approval) != (d22_approval if d22_binding else
                                            d21_approval if d21_binding else
                                            d20_approval if d20_binding else d19_approval) or
                          not isinstance(approval["d19_sha256"], str) or
                          not SHA256_PATTERN.fullmatch(approval["d19_sha256"]) or
                          approval["d19_sha256"] != payload["protocol"].get("d19_researcher_amendment_sha256")):
        raise LockError("D19 class-transfer approval digest is missing")
    if d20_binding and (not isinstance(approval["d20_sha256"], str) or
                        not SHA256_PATTERN.fullmatch(approval["d20_sha256"]) or
                        approval["d20_sha256"] != payload["protocol"]["d20_researcher_amendment_sha256"]):
        raise LockError("D20 C3 qualification approval digest is missing")
    if d21_binding and (not isinstance(approval["d21_sha256"], str) or
                        not SHA256_PATTERN.fullmatch(approval["d21_sha256"]) or
                        approval["d21_sha256"] != payload["protocol"].get("d21_researcher_amendment_sha256") or
                        not isinstance(payload["protocol"].get("seed_replications_sha256"), str) or
                        not SHA256_PATTERN.fullmatch(payload["protocol"]["seed_replications_sha256"])):
        raise LockError("D21 C3 seed-replication approval or artifact digest is missing")
    if d22_binding and (not isinstance(approval["d21_sha256"], str) or
                        not SHA256_PATTERN.fullmatch(approval["d21_sha256"]) or
                        approval["d21_sha256"] != payload["protocol"].get("d21_researcher_amendment_sha256") or
                        not isinstance(approval["d22_sha256"], str) or
                        not SHA256_PATTERN.fullmatch(approval["d22_sha256"]) or
                        approval["d22_sha256"] != payload["protocol"].get("d22_researcher_amendment_sha256") or
                        payload["protocol"].get("optional_replication_conditions") != ["C0", "C2"]):
        raise LockError("D22 C0/C2 seed-replication approval is missing")
    decision_ids = approval["decision_ids"]
    if (not isinstance(decision_ids, list) or
            not all(isinstance(item, str) for item in decision_ids) or
            len(decision_ids) != len(required_decisions) or
            set(decision_ids) != required_decisions):
        raise LockError("approval must cover D01-D22 exactly once" if d22_binding else
                        "approval must cover D01-D21 exactly once" if d21_binding else
                        "approval must cover D01-D20 exactly once" if d20_binding else
                        "approval must cover D01-D19 exactly once" if class_binding else
                        "approval must cover D01-D18 exactly once")
    if not isinstance(payload["protocol"], dict) or not payload["protocol"]:
        raise LockError("approved protocol body is missing")
    if not isinstance(payload["condition_variants"], dict) or not payload["condition_variants"]:
        raise LockError("condition variants are missing")
    if not isinstance(payload["allowed_device_roles"], dict) or not payload["allowed_device_roles"]:
        raise LockError("approved device roles are missing")
    return payload, digest


def read_protocol_lock(path: str | Path) -> tuple[dict[str, Any], str]:
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise LockError(f"cannot read protocol lock: {exc}") from exc
    return validate_protocol_lock(document)
