"""Canonical payload hashing and approval lock validation.

No scientific values are chosen here. A lock is only a transport and integrity
boundary for an externally approved, measured protocol.
"""

from __future__ import annotations

import hashlib
import json
import re
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
    if set(payload) != required:
        raise LockError("protocol lock has missing or unknown fields")
    if payload["status"] != "approved" or payload["production_ready"] is not True:
        raise LockError("production requires an approved protocol")
    if not isinstance(payload["source_commit"], str) or not COMMIT_PATTERN.fullmatch(payload["source_commit"]):
        raise LockError("source_commit must be an immutable full commit ID")
    for field in ("artifact_lock_digest", "environment_lock_digest", "hardware_lock_digest", "calibration_lock_digest"):
        value = payload[field]
        if not isinstance(value, str) or not SHA256_PATTERN.fullmatch(value):
            raise LockError(f"{field} must be a SHA-256 digest")
    approval = payload["approval"]
    if not isinstance(approval, dict) or set(approval) != {"researcher", "approved_at_utc", "approval_sha256", "decision_ids"}:
        raise LockError("approval record is incomplete")
    if not isinstance(approval["researcher"], str) or not approval["researcher"].strip():
        raise LockError("researcher approval is missing")
    if not isinstance(approval["approved_at_utc"], str) or not approval["approved_at_utc"].endswith("Z"):
        raise LockError("approval UTC timestamp is missing")
    if not isinstance(approval["approval_sha256"], str) or not SHA256_PATTERN.fullmatch(approval["approval_sha256"]):
        raise LockError("approval evidence digest is missing")
    required_decisions = {f"D{number:02d}" for number in range(1, 19)}
    decision_ids = approval["decision_ids"]
    if (not isinstance(decision_ids, list) or
            not all(isinstance(item, str) for item in decision_ids) or
            len(decision_ids) != len(required_decisions) or
            set(decision_ids) != required_decisions):
        raise LockError("approval must cover D01-D18 exactly once")
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
