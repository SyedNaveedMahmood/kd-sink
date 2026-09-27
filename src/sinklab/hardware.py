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


def _eligible(profile: Profile) -> None:
    if profile.condition == "C4" and profile.device_role != "rtx3090":
        raise HardwareError("production REL profiling is restricted to RTX 3090")
    if profile.microbatch not in DIVISORS or not profile.device_uuid:
        raise HardwareError("invalid profile identity or batch")
    if profile.passed and not (profile.optimizer_allocated and profile.evaluation_passed and profile.save_passed):
        raise HardwareError("forward-only profile cannot pass")


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
        if {p.microbatch for p in rows} != set(DIVISORS):
            raise HardwareError(f"missing candidate profiles for {key}")
        candidates = [p.microbatch for p in rows if p.passed and
                      p.free_bytes >= p.headroom_bytes and
                      p.total_bytes - p.peak_reserved_bytes >= p.headroom_bytes]
        if not candidates:
            raise HardwareError(f"no safe batch for {key}")
        safe.append(max(candidates))
    common = min(safe)
    payload = {"schema_version": 1, "evidence": "measured_gpu" if production else "mock_cpu",
               "required": [list(x) for x in sorted(required)],
               "microbatch": common, "accumulation": 64 // common,
               "effective_sequences": 64, "input_tokens": 8192, "shifted_targets": 8128,
               "profiles": [asdict(p) for p in profiles]}
    payload["sha256"] = payload_digest(payload)
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
