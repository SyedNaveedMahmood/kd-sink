"""D24 admission and inventory requirements for new S1 checkpoint follow-ups.

This policy never replaces a source model's training protocol identity.
Recorded-result analysis and registered in-training evaluation are separate.
"""

from __future__ import annotations

from .provenance import SHA256_PATTERN, verify_envelope


D24_SHA256 = "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6"
D24_PATH = "protocols/s1_researcher_amendment_d24_seed0_followups_20261004.json"
CONDITIONS = tuple(f"C{i}" for i in range(7))
FOLLOWUP_STEPS = {"S4": (0, 100, 500, 2000, 10000),
                  "S6": (0, 500, 2000, 10000)}


class FollowupPolicyError(ValueError):
    pass


def validate_followup_amendment(document: dict) -> dict:
    payload, digest = verify_envelope(document)
    if digest != D24_SHA256:
        raise FollowupPolicyError("unapproved D24 follow-up amendment")
    return payload


def original_protocol_sha256(identity: dict) -> str:
    # Checkpoint manifests use protocol_hash; evaluation identities use
    # protocol_sha256. Accept both original schemas without rewriting either.
    roots = [identity[name] for name in ("protocol_hash", "protocol_sha256")
             if name in identity]
    if (not roots or any(not isinstance(root, str) or
                        not SHA256_PATTERN.fullmatch(root) for root in roots)
            or len(set(roots)) != 1):
        raise FollowupPolicyError("original training protocol SHA-256 required; conflicting roots forbidden")
    return roots[0]


def admit_s1_followup(identity: dict, *, study: str, step: int) -> dict:
    """Validate original S1 identity before model access; return separate policy provenance."""
    if not isinstance(identity, dict) or identity.get("study") != "S1":
        raise FollowupPolicyError("original S1 checkpoint identity required")
    if identity.get("condition") not in CONDITIONS:
        raise FollowupPolicyError("S1 follow-ups require condition C0-C6")
    if type(identity.get("seed")) is not int or identity["seed"] != 0:
        raise FollowupPolicyError("D24: all new S1 checkpoint-dependent work is seed0-only")
    if not isinstance(study, str) or not study.strip():
        raise FollowupPolicyError("explicit follow-up study required")
    if type(step) is not int or step < 0:
        raise FollowupPolicyError("original checkpoint step required")
    if study in FOLLOWUP_STEPS and step not in FOLLOWUP_STEPS[study]:
        raise FollowupPolicyError(f"{study} requires one of the fixed retained checkpoints")
    root = original_protocol_sha256(identity)
    return {"amendment_sha256": D24_SHA256, "study": study,
            "source_protocol_sha256": root, "training_seed": 0,
            "interpretation": "designated-seed0 follow-up; no across-training-seed reproducibility claim"}


def required_s1_checkpoints(study: str) -> tuple[tuple[str, int, int], ...]:
    """Logical readiness requirements, independent of machine paths and hardware replicas."""
    if study not in FOLLOWUP_STEPS:
        raise FollowupPolicyError("fixed checkpoint readiness is defined only for S4/S6")
    return tuple((condition, 0, step) for condition in CONDITIONS
                 for step in FOLLOWUP_STEPS[study])


def checkpoint_inventory_coverage(manifests: list[dict], *, study: str) -> dict:
    """Assess already verified manifests; never load, write, or choose between replicas.

    Callers must checksum-verify checkpoint payloads first. This function reports
    logical coverage only, not panel/device/runtime approval or execution readiness.
    Seed1/2 manifests do not supply or add any requirement.
    """
    required = required_s1_checkpoints(study)
    found = {}
    for manifest in manifests:
        identity = manifest["identity"]
        if identity.get("study") != "S1" or identity.get("seed") != 0:
            continue
        step = manifest["step"]
        if step not in FOLLOWUP_STEPS[study]:
            continue
        admit_s1_followup(identity, study=study, step=step)
        key = (identity["condition"], 0, step)
        # Retained weights and final can represent the same state. Keep every
        # candidate's provenance; selection and hardware pairing stay explicit.
        found.setdefault(key, []).append({"run_id": identity["run_id"],
            "kind": manifest["kind"], "protocol_sha256": original_protocol_sha256(identity),
            "source_identity": identity,
            "device_role": identity.get("device_role"),
            "model_sha256": manifest["files"]["model.safetensors"]})
    missing = [list(key) for key in required if key not in found]
    return {"study": study, "amendment_sha256": D24_SHA256,
            "required_training_seeds": [0], "required_count": len(required),
            "covered_count": len(found), "missing": missing,
            "status": "complete_inventory" if not missing else "incomplete_inventory",
            "candidates": [{"condition": k[0], "seed": k[1], "step": k[2],
                            "checkpoints": v} for k, v in sorted(found.items())],
            "execution_authorized": False}
