"""Small, explicit T08 pairing primitives; inferential analysis remains Stage09."""

from __future__ import annotations

import math
from collections.abc import Sequence


class AnalysisError(ValueError):
    pass


def paired_seed_differences(rows: Sequence[dict], *, condition_a: str,
                            condition_b: str, device_role: str,
                            step: int, metric: str) -> list[dict]:
    """Join one observed endpoint per seed within one physical device block."""
    selected = {}
    for row in rows:
        if row["device_role"] != device_role or row["step"] != step or row["condition"] not in {condition_a, condition_b}:
            continue
        key = (row["seed"], row["condition"])
        if key in selected:
            raise AnalysisError("duplicate seed/condition/device endpoint, possibly a hardware replica")
        if row.get("status") != "complete" or metric not in row or not math.isfinite(row[metric]):
            raise AnalysisError("paired endpoint is missing or nonfinite")
        selected[key] = row
    seeds_a = {seed for seed, condition in selected if condition == condition_a}
    seeds_b = {seed for seed, condition in selected if condition == condition_b}
    if not seeds_a or seeds_a != seeds_b:
        raise AnalysisError("paired seed coverage is incomplete")
    result = []
    for seed in sorted(seeds_a):
        a, b = selected[seed, condition_a], selected[seed, condition_b]
        for field in ("study", "protocol_sha256", "initialization_sha256", "data_sha256"):
            if a[field] != b[field]:
                raise AnalysisError(f"paired provenance mismatch: {field}")
        result.append({"seed": seed, "device_role": device_role, "step": step,
                       "metric": metric, "condition_a": condition_a,
                       "condition_b": condition_b,
                       "difference_b_minus_a": b[metric] - a[metric]})
    return result


def matched_ce_pair(a_rows: Sequence[dict], b_rows: Sequence[dict], *, max_gap: float = .05) -> dict | None:
    """Select saved states using CE/steps only; sink and causal fields are ignored."""
    if max_gap <= 0 or not math.isfinite(max_gap):
        raise AnalysisError("positive finite CE gap required")
    candidates = []
    for a in a_rows:
        for b in b_rows:
            if any(a[field] != b[field] for field in ("seed", "device_role", "protocol_sha256")):
                raise AnalysisError("matched-CE candidates require one seed/device/protocol")
            if a.get("status") != "complete" or b.get("status") != "complete":
                continue
            if not all(math.isfinite(row["ce_nats"]) for row in (a, b)):
                raise AnalysisError("nonfinite CE candidate")
            gap = abs(a["ce_nats"] - b["ce_nats"])
            if gap <= max_gap:
                candidates.append((gap, abs(a["step"] - b["step"]),
                                   min(a["step"], b["step"]), a["step"], b["step"]))
    if not candidates:
        return None
    gap, separation, _, a_step, b_step = min(candidates)
    return {"a_step": a_step, "b_step": b_step, "ce_gap_nats": gap,
            "step_separation": separation, "max_gap_nats": max_gap}


def missing_cadence(observed_steps: Sequence[int], expected_steps: Sequence[int]) -> dict:
    observed, expected = set(observed_steps), set(expected_steps)
    if len(observed_steps) != len(observed):
        raise AnalysisError("duplicate checkpoint observations are not independent")
    return {"missing_steps": sorted(expected - observed),
            "unexpected_steps": sorted(observed - expected),
            "status": "complete" if observed == expected else "incomplete"}
