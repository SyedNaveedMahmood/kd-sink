"""Validate an explicit, non-executable S1/S3 job manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, resolve_config
from .provenance import _no_duplicate_keys


class JobPlanError(ValueError):
    pass


def validate_job_plan(plan: Mapping[str, Any], config_root: str | Path) -> dict[str, Any]:
    """Check every named job; validation never launches a job or supplies a seed."""
    if not isinstance(plan, dict) or set(plan) != {"schema_version", "status", "study", "jobs", "optional_s3_enabled"}:
        raise JobPlanError("job plan has missing or unknown fields")
    if plan["schema_version"] != 1 or plan["status"] not in {"draft", "approved"}:
        raise JobPlanError("unsupported plan version or status")
    study = plan["study"]
    if study not in {"S1", "S3"} or type(plan["optional_s3_enabled"]) is not bool:
        raise JobPlanError("unsupported study or optional-study flag")
    if study == "S3" and plan["optional_s3_enabled"] is not True and plan["status"] == "approved":
        raise JobPlanError("S3 cannot be approved while disabled")
    jobs = plan["jobs"]
    if not isinstance(jobs, list) or not jobs:
        raise JobPlanError("explicit nonempty jobs list required")
    root = Path(config_root).resolve()
    seen: set[tuple[str, int, str]] = set()
    per_condition: dict[str, set[int]] = {}
    for row in jobs:
        if not isinstance(row, dict) or set(row) != {"run_id", "config", "seed"}:
            raise JobPlanError("job must name one run_id, config and seed")
        path = (root / row["config"]).resolve()
        if not path.is_relative_to(root) or path.suffix != ".json":
            raise JobPlanError("config must be a JSON file inside config root")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
            spec = resolve_config(raw, seed=row["seed"])
        except (OSError, ValueError, TypeError, ConfigError) as exc:
            raise JobPlanError(f"invalid job config: {exc}") from exc
        if spec.study != study or row["run_id"] != f"{study.lower()}-{spec.condition.lower()}-seed{spec.seed}-{spec.device_role}":
            raise JobPlanError("run_id or study differs from config")
        key = (spec.condition, spec.seed, spec.device_role)
        if key in seen:
            raise JobPlanError("duplicate physical run")
        seen.add(key)
        per_condition.setdefault(spec.condition, set()).add(spec.seed)
    expected = {f"C{i}" for i in range(7 if study == "S1" else 5)}
    if set(per_condition) != expected:
        raise JobPlanError("condition coverage is incomplete")
    seeds = next(iter(per_condition.values()))
    if not seeds or any(value != seeds for value in per_condition.values()):
        raise JobPlanError("conditions do not share the same explicit seed set")
    if study == "S1" and len(seen) == 27:
        condition_roles = {condition: {role for c, _, role in seen if c == condition} for condition in expected}
        if any("rtx3090" not in condition_roles[c] for c in ("C1", "C2", "C3", "C4")):
            raise JobPlanError("3090 comparison block is incomplete")
        if any("rtx4080super" not in condition_roles[c] for c in ("C0", "C1", "C2", "C5", "C6")):
            raise JobPlanError("4080 comparison block is incomplete")
    elif study == "S1" and len(seen) == 21:
        if any(role != "rtx3090" for _, _, role in seen):
            raise JobPlanError("21-run plan must keep comparisons on the 3090")
    elif study == "S1":
        raise JobPlanError("S1 plan must contain 21 or 27 explicit physical runs")
    elif len(seen) != 15 or any(role != "rtx3090" for _, _, role in seen):
        raise JobPlanError("optional S3 plan must contain 15 explicit 3090 runs")
    return {"study": study, "status": plan["status"], "physical_runs": len(seen),
            "unique_condition_seed_pairs": sum(len(value) for value in per_condition.values()),
            "seeds": sorted(seeds), "launchable": False}


def inspect_job(plan: Mapping[str, Any], config_root: str | Path, run_id: str) -> dict[str, Any]:
    """Resolve precisely one named row for operator review, without executing it."""
    summary = validate_job_plan(plan, config_root)
    matches = [row for row in plan["jobs"] if row["run_id"] == run_id]
    if len(matches) != 1:
        raise JobPlanError("exactly one explicit --run-id must match the plan")
    row = matches[0]
    raw = json.loads((Path(config_root) / row["config"]).read_text(encoding="utf-8"),
                     object_pairs_hook=_no_duplicate_keys)
    spec = resolve_config(raw, seed=row["seed"])
    return {"run_id": run_id, "config": row["config"], "seed": spec.seed,
            "study": spec.study, "condition": spec.condition, "device_role": spec.device_role,
            "plan_status": summary["status"], "action": "inspection_only", "launchable": False}
