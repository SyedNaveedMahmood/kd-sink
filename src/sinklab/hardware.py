"""Profile evidence and one common effective-batch schedule.

Synthetic reports are useful for solver tests but cannot create a production lock.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from .provenance import payload_digest


class HardwareError(ValueError):
    pass


DIVISORS = (64, 32, 16, 8, 4, 2, 1)
RTX4080_SUPER_MODEL = "NVIDIA GeForce RTX 4080 SUPER"
RTX3090_MODEL = "NVIDIA GeForce RTX 3090"

# Current-plan evidence status, not policy permission or a production hardware
# lock. 3090 C0/C5/C6 are policy-allowed but unused and unmeasured; a future
# plan assigning them needs its own measured hardware-lock amendment.
S1_ELIGIBILITY_STATUS = {
    "C0": {"rtx3090": "unmeasured_not_required_for_current_plan", "rtx4080super": "measured_full_cycle_candidate"},
    "C1": {"rtx3090": "measured_full_cycle_candidate", "rtx4080super": "measured_full_cycle_candidate"},
    "C2": {"rtx3090": "measured_full_cycle_candidate", "rtx4080super": "measured_full_cycle_candidate"},
    "C3": {"rtx3090": "measured_full_cycle_candidate", "rtx4080super": "pending_4080_profile"},
    "C4": {"rtx3090": "measured_full_cycle_candidate", "rtx4080super": "restricted_rel_3090_only"},
    "C5": {"rtx3090": "unmeasured_not_required_for_current_plan", "rtx4080super": "measured_full_cycle_candidate"},
    "C6": {"rtx3090": "unmeasured_not_required_for_current_plan", "rtx4080super": "measured_full_cycle_candidate"},
}


def authorize_production_device(protocol: dict, hardware: dict, *, condition: str,
                                device_role: str, gpu_name: str, gpu_uuid: str) -> dict:
    """Authorize a fresh job by exact class/model, then return its physical identity.

    The measured batch proof still names the reference 4080 UUID. D19 transfers
    that proof to exact-model peers; it does not change the proof or resume rule.
    """
    if not isinstance(gpu_uuid, str) or not gpu_uuid.strip():
        raise HardwareError("actual GPU UUID is required")
    if device_role not in protocol.get("allowed_device_roles", {}).get(condition, []):
        raise HardwareError("condition/device role is absent from approved protocol")
    if device_role == "rtx4080super":
        class_policy = hardware.get("hardware_classes", {}).get(device_role)
        d20 = "d20_researcher_amendment_sha256" in protocol.get("protocol", {})
        expected_conditions = (["C0", "C1", "C2", "C3", "C5", "C6"] if d20 else
                               ["C0", "C1", "C2", "C5", "C6"])
        if (not isinstance(class_policy, dict) or
                class_policy.get("policy") != "researcher_approved_reference_profile_transfer" or
                class_policy.get("model") != RTX4080_SUPER_MODEL or
                class_policy.get("allowed_conditions") != expected_conditions or
                class_policy.get("reference_uuid") != hardware.get("gpu_uuids", {}).get(device_role) or
                protocol.get("protocol", {}).get("hardware", {}).get("hardware_classes", {}).get(device_role)
                != class_policy):
            raise HardwareError("RTX4080 SUPER class policy differs from sealed reference")
        if condition not in class_policy["allowed_conditions"] or gpu_name != RTX4080_SUPER_MODEL:
            raise HardwareError("actual GPU is outside approved RTX4080 SUPER class")
    elif device_role == "rtx3090":
        if (gpu_name != RTX3090_MODEL or
                gpu_uuid != hardware.get("gpu_uuids", {}).get(device_role) or
                gpu_uuid != protocol.get("protocol", {}).get("hardware", {}).get("gpu_uuids", {}).get(device_role)):
            raise HardwareError("actual GPU differs from exact RTX3090 binding")
    else:
        raise HardwareError("unsupported production device role")
    return {"device_role": device_role, "gpu_name": gpu_name, "gpu_uuid": gpu_uuid,
            "condition": condition}


@dataclass(frozen=True)
class Profile:
    condition: str
    device_role: str
    device_uuid: str
    microbatch: int
    passed: bool
    evidence: str  # "measured_gpu" or "mock_cpu"
    optimizer_allocated: bool
    evaluation_passed: bool
    save_passed: bool
    peak_allocated_bytes: int
    peak_reserved_bytes: int
    free_bytes: int
    total_bytes: int
    headroom_bytes: int
    source_sha256: str | None = None
    failure_status: str | None = None


def _eligible(profile: Profile) -> None:
    if profile.condition == "C4" and profile.device_role != "rtx3090":
        raise HardwareError("production REL profiling is restricted to RTX 3090")
    if profile.microbatch not in DIVISORS or not profile.device_uuid:
        raise HardwareError("invalid profile identity or batch")
    if profile.passed and not (profile.optimizer_allocated and profile.evaluation_passed and profile.save_passed):
        raise HardwareError("forward-only profile cannot pass")


def validate_eligibility_matrix(plan: dict) -> None:
    """Require an exact condition/role/UUID matrix in a reviewed hardware plan."""
    required = plan.get("required")
    if not isinstance(required, list) or not required:
        raise HardwareError("hardware plan requires condition/device/UUID entries")
    matrix: dict[str, dict[str, str]] = {}
    for row in required:
        if (not isinstance(row, list) or len(row) != 3 or
                any(not isinstance(value, str) or not value for value in row)):
            raise HardwareError("invalid condition/device/UUID entry")
        condition, role, uuid = row
        if condition == "C4" and role != "rtx3090":
            raise HardwareError("C4 REL is restricted to RTX 3090")
        if role in matrix.setdefault(condition, {}):
            raise HardwareError("duplicate condition/device eligibility")
        matrix[condition][role] = uuid
    if plan.get("eligibility_matrix") != matrix:
        raise HardwareError("condition/device/UUID eligibility matrix disagrees with required profiles")


def build_batch_plan(profiles: Iterable[Profile], required: set[tuple[str, str, str]],
                     *, production: bool) -> dict:
    profiles = list(profiles)
    if not required:
        raise HardwareError("explicit condition/device/UUID eligibility matrix required")
    by_key: dict[tuple[str, str, str], list[Profile]] = {}
    for p in profiles:
        _eligible(p)
        if production and p.evidence != "measured_gpu":
            raise HardwareError("mock evidence cannot produce a hardware lock")
        if production and (p.total_bytes <= 0 or p.headroom_bytes != max(1_610_612_736, int(p.total_bytes * .1))):
            raise HardwareError("profile differs from registered VRAM headroom rule")
        key = (p.condition, p.device_role, p.device_uuid)
        if key not in required:
            raise HardwareError("unexpected profile outside approved eligibility matrix")
        if any(old.microbatch == p.microbatch for old in by_key.get(key, [])):
            raise HardwareError("duplicate profile candidate")
        by_key.setdefault(key, []).append(p)
    if set(by_key) != required:
        raise HardwareError("missing required condition/device profile")
    safe = []
    for key in sorted(required):
        rows = by_key[key]
        candidates = [p.microbatch for p in rows if p.passed and
                      p.free_bytes >= p.headroom_bytes and
                      p.total_bytes - p.peak_reserved_bytes >= p.headroom_bytes]
        if not candidates:
            raise HardwareError(f"no safe batch for {key}")
        safe.append(max(candidates))
    common = min(safe)
    next_larger = DIVISORS[DIVISORS.index(common) - 1] if common != DIVISORS[0] else None
    if next_larger is not None and not any(
            p.microbatch == next_larger and p.optimizer_allocated and
            p.evaluation_passed and p.save_passed and
            (p.free_bytes < p.headroom_bytes or
             p.total_bytes - p.peak_reserved_bytes < p.headroom_bytes)
            for rows in by_key.values() for p in rows):
        raise HardwareError("next larger common divisor lacks measured unsafe headroom evidence")
    matrix: dict[str, dict[str, str]] = {}
    for condition, role, uuid in sorted(required):
        matrix.setdefault(condition, {})[role] = uuid
    payload = {"schema_version": 1, "evidence": "measured_gpu" if production else "mock_cpu",
               "required": [list(x) for x in sorted(required)],
               "eligibility_matrix": matrix,
               "microbatch": common, "accumulation": 64 // common,
               "next_larger_disproved": next_larger,
               "proof_rule": "safe_at_or_above_common_for_each_pair_and_full_cycle_headroom_failure_at_next_larger",
               "effective_sequences": 64, "input_tokens": 8192, "shifted_targets": 8128,
               "profiles": [asdict(p) for p in profiles]}
    payload["sha256"] = payload_digest(payload)
    validate_eligibility_matrix(payload)
    return payload


def profile_candidate_isolated(command: list[str], output: Path) -> dict:
    """Run one candidate in a fresh child process; never hide OOM by retrying smaller."""
    if not command or command[0] != sys.executable:
        raise HardwareError("profile child must use the current Python interpreter")
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise HardwareError(f"isolated profile failed ({result.returncode}): {result.stderr[-1000:]}")
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise HardwareError("child did not write a valid profile") from exc
    return report


def measure_candidate(*, condition: str, device_role: str, device_uuid: str,
                      microbatch: int, train_update: Callable[[], None],
                      evaluate: Callable[[], None], save: Callable[[], None],
                      headroom_bytes: int) -> Profile:
    """Child-process body: warmup plus two updates, evaluation, and full save.

    The caller owns the isolated process and real model/optimizer. Failed calls
    raise, so no forward-only result can be mistaken for a passing profile.
    """
    if not torch.cuda.is_available():
        raise HardwareError("measured GPU profile requires CUDA")
    _eligible(Profile(condition, device_role, device_uuid, microbatch, False, "measured_gpu",
                      False, False, False, 0, 0, 0, 0, headroom_bytes))
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    try:
        for _ in range(3):  # one warmup, then two state-bearing optimizer updates
            train_update()
            torch.cuda.synchronize()
        evaluate()
        torch.cuda.synchronize()
        save()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as exc:
        raise HardwareError("isolated candidate OOM; preserve child failure evidence") from exc
    free, total = torch.cuda.mem_get_info()
    allocated = torch.cuda.max_memory_allocated()
    reserved = torch.cuda.max_memory_reserved()
    del started  # duration belongs to the caller's signed profile record
    return Profile(condition, device_role, device_uuid, microbatch,
                   free >= headroom_bytes and total - reserved >= headroom_bytes,
                   "measured_gpu", True, True, True, allocated, reserved,
                   free, total, headroom_bytes)
