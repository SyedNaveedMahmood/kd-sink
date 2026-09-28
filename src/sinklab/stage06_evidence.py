"""Normalize the committed Stage06 GPU profiles into one reviewed S1 batch proof."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .hardware import HardwareError, Profile, build_batch_plan
from .job_plan import validate_job_plan
from .provenance import payload_digest, seal_payload


EXPECTED_REPORT_SHA256 = {
    "rtx3090": "c0f612cdc2cc2d94607568bc723eb8d89599a7182cd661d6b5c5079351e2dca5",
    "rtx4080super": "70fbe8fffce9d76838241d81f8ce0558d99a09b9e234d86fce934632971fc9ca",
}
WINDOWS_WORKTREE_REPORT_SHA256 = {
    "rtx3090": "cfa129c971516ae4c42e9290db73b0a3a7f5a9ac5a524c064c099c80b8cbfb6b",
    "rtx4080super": "7834d8638b5d15444e4a50c63c62c269f8b24f71942f604f525c3c8069c91748",
}
REVIEWED_ROLE_CONDITIONS = {
    "rtx3090": {"C1", "C2", "C3", "C4"},
    "rtx4080super": {"C0", "C1", "C2", "C5", "C6"},
}


def _read_report(path: Path, role: str) -> dict:
    raw = Path(path).read_bytes()
    if hashlib.sha256(raw.replace(b"\r\n", b"\n")).hexdigest() != EXPECTED_REPORT_SHA256[role]:
        raise HardwareError(f"{role} committed profile evidence hash changed")
    return json.loads(raw)


def _normalize(report: dict, role: str) -> list[Profile]:
    hardware = report["hardware"] if role == "rtx3090" else report["rtx4080super"]
    rows = report["profiles"] if role == "rtx3090" else hardware["profiles"]
    total, headroom = hardware["total_bytes"], hardware["headroom_bytes"]
    result = []
    for row in rows:
        if row["condition"] not in REVIEWED_ROLE_CONDITIONS[role]:
            continue
        status = row["status"]
        result.append(Profile(
            condition=row["condition"], device_role=role, device_uuid=hardware["uuid"],
            microbatch=row["microbatch"], passed=status == "complete_safe",
            evidence="measured_gpu", optimizer_allocated=row.get("optimizer_allocated", False),
            evaluation_passed=row.get("evaluation_passed", False),
            save_passed=row.get("save_passed", False),
            peak_allocated_bytes=row.get("peak_allocated_bytes") or 0,
            peak_reserved_bytes=row.get("peak_reserved_bytes") or total,
            free_bytes=row.get("free_bytes") or 0, total_bytes=total,
            headroom_bytes=headroom,
            source_sha256=row.get("raw_sha256", EXPECTED_REPORT_SHA256[role]),
            failure_status=None if status == "complete_safe" else status))
    return result


def build_reviewed_seed0_candidate(report3090: Path, report4080: Path,
                                    plan_path: Path, config_root: Path) -> dict:
    plan = json.loads(Path(plan_path).read_text(encoding="utf-8"))
    summary = validate_job_plan(plan, config_root)
    if (plan["status"] != "approved" or summary["seeds"] != [0] or
            summary["physical_runs"] != 9 or summary["unique_condition_seed_pairs"] != 7 or
            summary["hardware_confounds"]):
        raise HardwareError("reviewed seed0 plan must contain exactly nine bridged jobs")
    from .config import resolve_config
    observed = set()
    for job in plan["jobs"]:
        config = json.loads((Path(config_root) / job["config"]).read_text(encoding="utf-8"))
        spec = resolve_config(config, seed=job["seed"])
        observed.add((spec.condition, spec.device_role))
    expected = {(condition, role) for role, conditions in REVIEWED_ROLE_CONDITIONS.items()
                for condition in conditions}
    if observed != expected:
        raise HardwareError("reviewed plan condition/device pairs differ from researcher allocation")
    reports = {"rtx3090": _read_report(report3090, "rtx3090"),
               "rtx4080super": _read_report(report4080, "rtx4080super")}
    profiles = [p for role, report in reports.items() for p in _normalize(report, role)]
    uuids = {role: (report["hardware"] if role == "rtx3090" else report["rtx4080super"])["uuid"]
             for role, report in reports.items()}
    required = {(condition, role, uuids[role]) for condition, role in expected}
    proof = build_batch_plan(profiles, required, production=True)
    if (proof["microbatch"], proof["accumulation"], proof["effective_sequences"],
            proof["input_tokens"], proof["shifted_targets"]) != (4, 16, 64, 8192, 8128):
        raise HardwareError("committed profiles did not derive the reviewed 4x16 schedule")
    repeat = reports["rtx4080super"]["rtx4080super"]["c5_mb4_repeat"]
    eligibility = {}
    for index in range(7):
        condition = f"C{index}"
        eligibility[condition] = {}
        for role in ("rtx3090", "rtx4080super"):
            scheduled = condition in REVIEWED_ROLE_CONDITIONS[role]
            eligibility[condition][role] = {
                "policy_allowed": not (condition == "C4" and role == "rtx4080super"),
                "measured_production_eligible": scheduled,
                "scheduled_in_current_plan": scheduled,
                "current_plan_status": ("measured_scheduled" if scheduled else
                    "prohibited" if condition == "C4" and role == "rtx4080super" else
                    "unmeasured_not_required_for_current_plan"),
                "future_gate": ("pending_4080_profile" if condition == "C3" and role == "rtx4080super"
                                else "new_hardware_lock_amendment" if not scheduled and
                                not (condition == "C4" and role == "rtx4080super") else None),
            }
    return seal_payload({"kind": "stage06-reviewed-batch-candidate-v1",
        "status": "measured_batch_proof_pending_calibration_and_final_locks",
        "reviewed_plan_sha256": payload_digest(plan),
        "source_report_sha256": EXPECTED_REPORT_SHA256,
        "historical_windows_worktree_report_sha256": WINDOWS_WORKTREE_REPORT_SHA256,
        "proof": proof,
        "condition_device_eligibility": eligibility,
        "c5_4080_mb4_repeat": repeat,
        "unmeasured_not_required_for_current_plan": [
            ["C0", "rtx3090"], ["C5", "rtx3090"], ["C6", "rtx3090"],
            ["C3", "rtx4080super"]],
        "c4_4080_status": "prohibited"})
