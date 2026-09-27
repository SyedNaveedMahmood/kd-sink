"""Audit bounded 4080 profiles against existing 3090 evidence.

The result is a measured engineering candidate, never an approved hardware lock.
All paths are explicit; no GPU work or training is launched here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


CONDITIONS = ("C0", "C1", "C2", "C5", "C6")
CANDIDATES = {"C0": (8,), "C1": (8, 4), "C2": (8, 4),
              "C5": (8, 4), "C6": (8, 4)}


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(profile_dir: Path, prior_3090: Path) -> dict:
    prior = _read(prior_3090)
    if prior["hardware"]["name"] != "NVIDIA GeForce RTX 3090":
        raise ValueError("prior evidence is not the RTX 3090 profile")
    if set(prior["safe_maxima"]) != {"C1", "C2", "C3", "C4"}:
        raise ValueError("prior RTX 3090 condition coverage is incomplete")
    if prior["safe_maxima"]["C4"] != 8:
        raise ValueError("expected RTX 3090 REL upper bound 8 was not established")
    if list(profile_dir.glob("c4_*.json")):
        raise ValueError("C4 must never be profiled on the RTX 4080 SUPER")
    rows = []
    maxima = {}
    role_uuid = None
    total = None
    headroom = None
    for condition in CONDITIONS:
        passing = []
        for batch in CANDIDATES[condition]:
            path = profile_dir / f"{condition.lower()}_mb{batch}.json"
            row = _read(path)
            item = {"condition": condition, "microbatch": batch,
                    "raw_sha256": _sha(path), "source": path.name}
            if "profile" in row:
                profile = row["profile"]
                if (profile["condition"] != condition or profile["microbatch"] != batch or
                        profile["device_role"] != "rtx4080super" or profile["evidence"] != "measured_gpu"):
                    raise ValueError("candidate profile identity differs")
                role_uuid = role_uuid or profile["device_uuid"]
                total = total or profile["total_bytes"]
                headroom = headroom or profile["headroom_bytes"]
                if (role_uuid != profile["device_uuid"] or total != profile["total_bytes"] or
                        headroom != profile["headroom_bytes"]):
                    raise ValueError("candidate GPU/headroom identity changed")
                required = max(1610612736, int(total * .1))
                if headroom != required:
                    raise ValueError("candidate used a different VRAM headroom rule")
                fit = (profile["optimizer_allocated"] and profile["evaluation_passed"] and
                       profile["save_passed"] and profile["free_bytes"] >= headroom and
                       total - profile["peak_reserved_bytes"] >= headroom)
                if profile["passed"] != fit:
                    raise ValueError("candidate pass flag disagrees with full-cycle headroom")
                item.update({"status": "complete_safe" if fit else "complete_unsafe",
                             "peak_allocated_bytes": profile["peak_allocated_bytes"],
                             "peak_reserved_bytes": profile["peak_reserved_bytes"],
                             "free_bytes": profile["free_bytes"],
                             "free_margin_over_required_bytes": profile["free_bytes"] - headroom,
                             "wall_seconds": row["wall_seconds"],
                             "optimizer_allocated": profile["optimizer_allocated"],
                             "evaluation_passed": profile["evaluation_passed"],
                             "save_passed": profile["save_passed"]})
                if fit:
                    passing.append(batch)
            else:
                if (row["condition"] != condition or row["microbatch"] != batch or
                        row["device_role"] != "rtx4080super" or row["completed_full_cycle"] is not False or
                        row["status"] != "terminated_unsafe_wddm_paging"):
                    raise ValueError("terminated candidate evidence is malformed")
                item.update({"status": row["status"],
                             "observed_gpu_used_mib": row["observed_gpu_used_mib"],
                             "observed_gpu_free_mib": row["observed_gpu_free_mib"],
                             "completed_full_cycle": False})
            rows.append(item)
        if not passing:
            raise ValueError(f"no measured safe candidate for {condition}")
        maxima[condition] = max(passing)
    repeat_path = profile_dir / "c5_mb4_repeat.json"
    repeat = _read(repeat_path)["profile"]
    if (repeat["condition"] != "C5" or repeat["microbatch"] != 4 or
            repeat["device_uuid"] != role_uuid or not repeat["passed"] or
            not all(repeat[key] for key in ("optimizer_allocated", "evaluation_passed", "save_passed")) or
            repeat["free_bytes"] < headroom or total - repeat["peak_reserved_bytes"] < headroom):
        raise ValueError("C5 repeat did not confirm the required margin")
    common = min(*prior["safe_maxima"].values(), *maxima.values())
    if 64 % common:
        raise ValueError("common microbatch must divide the effective batch")
    return {"schema_version": 1, "status": "measured_engineering_common_candidate_not_approved_lock",
            "prior_3090_profile_sha256": _sha(prior_3090),
            "rtx3090_safe_maxima": prior["safe_maxima"],
            "rtx4080super": {"name": "NVIDIA GeForce RTX 4080 SUPER",
                             "uuid": role_uuid, "total_bytes": total,
                             "headroom_bytes": headroom,
                             "safe_maxima_within_3090_bound_8": maxima,
                             "profiles": rows,
                             "c5_mb4_repeat": {"raw_sha256": _sha(repeat_path),
                                               "free_bytes": repeat["free_bytes"],
                                               "free_margin_over_required_bytes": repeat["free_bytes"] - headroom,
                                               "wall_seconds": _read(repeat_path)["wall_seconds"]}},
            "two_device_common_candidate": {"microbatch": common,
                                            "accumulation": 64 // common,
                                            "effective_sequences": 64,
                                            "input_tokens_per_update": 8192,
                                            "shifted_targets_per_update": 8128},
            "limits": ["4080 search was capped at microbatch 8 by the measured RTX 3090 REL maximum",
                       "larger 4080 candidates and smaller-than-first-safe candidates were not run",
                       "the build_batch_plan seven-divisor lock is incomplete",
                       "inputs and dense64 panel are deterministic synthetic memory-shape fixtures",
                       "no researcher approval, production calibration or 10k training is inferred"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--prior-3090", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.profile_dir, args.prior_3090)
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["two_device_common_candidate"], sort_keys=True))


if __name__ == "__main__":
    main()
