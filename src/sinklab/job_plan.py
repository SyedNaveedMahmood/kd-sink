"""Validate an explicit, non-executable S1/S3 job manifest."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .config import ConfigError, resolve_config
from .hardware import S1_ELIGIBILITY_STATUS
from .provenance import _no_duplicate_keys


class JobPlanError(ValueError):
    pass


def validate_job_plan(plan: Mapping[str, Any], config_root: str | Path, *,
                      protocol_lock: Mapping[str, Any] | None = None,
                      production: bool = False) -> dict[str, Any]:
    """Check every named job; validation never launches a job or supplies a seed."""
    required_fields = {"schema_version", "status", "study", "jobs", "optional_s3_enabled"}
    if not isinstance(plan, dict) or not required_fields <= set(plan) or set(plan) - required_fields - {"hardware_confounds"}:
        raise JobPlanError("job plan has missing or unknown fields")
    if plan["schema_version"] != 1 or plan["status"] not in {"draft", "approved"}:
        raise JobPlanError("unsupported plan version or status")
    if production and (plan["status"] != "approved" or protocol_lock is None):
        raise JobPlanError("production plan requires approved status and final protocol lock")
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
    roles_by_pair: dict[tuple[str, int], set[str]] = {}
    for row in jobs:
        if not isinstance(row, dict) or set(row) != {"run_id", "config", "seed"}:
            raise JobPlanError("job must name one run_id, config and seed")
        path = (root / row["config"]).resolve()
        if not path.is_relative_to(root) or path.suffix != ".json":
            raise JobPlanError("config must be a JSON file inside config root")
        try:
            raw = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
            spec = resolve_config(raw, seed=row["seed"],
                                  protocol_lock=protocol_lock, production=production)
        except (OSError, ValueError, TypeError, ConfigError) as exc:
            raise JobPlanError(f"invalid job config: {exc}") from exc
        if spec.study != study or row["run_id"] != f"{study.lower()}-{spec.condition.lower()}-seed{spec.seed}-{spec.device_role}":
            raise JobPlanError("run_id or study differs from config")
        key = (spec.condition, spec.seed, spec.device_role)
        if key in seen:
            raise JobPlanError("duplicate physical run")
        seen.add(key)
        per_condition.setdefault(spec.condition, set()).add(spec.seed)
        roles_by_pair.setdefault((spec.condition, spec.seed), set()).add(spec.device_role)
    expected = {f"C{i}" for i in range(7 if study == "S1" else 5)}
    if set(per_condition) != expected:
        raise JobPlanError("condition coverage is incomplete")
    seeds = next(iter(per_condition.values()))
    if not seeds or any(value != seeds for value in per_condition.values()):
        raise JobPlanError("conditions do not share the same explicit seed set")
    if study == "S1":
        if not seeds <= {0, 1, 2}:
            raise JobPlanError("S1 plan requires complete C0-C6 for each explicitly listed seed")
        if any(condition == "C4" and role != "rtx3090" for condition, _, role in seen):
            raise JobPlanError("C4 REL requires RTX 3090")
        if not production and any(S1_ELIGIBILITY_STATUS[condition][role] == "pending_4080_profile"
               for condition, _, role in seen):
            raise JobPlanError("C3 is pending_4080_profile")
        comparisons = (("C1", "C0"), ("C2", "C1"), ("C3", "C2"),
                       ("C4", "C2"), ("C4", "C3"), ("C3", "C1"),
                       ("C5", "C2"), ("C6", "C2"), ("C6", "C1"))
        confounded = {f"{a}-{b}-seed{seed}" for seed in seeds for a, b in comparisons
                      if not roles_by_pair[(a, seed)] & roles_by_pair[(b, seed)]}
        declared = plan.get("hardware_confounds", [])
        if not isinstance(declared, list) or any(not isinstance(x, str) for x in declared) or set(declared) != confounded or len(declared) != len(confounded):
            raise JobPlanError("cross-device primary comparisons require an explicit bridge/replica or exact hardware_confounds declaration")
    elif len(seen) != 15 or any(role != "rtx3090" for _, _, role in seen):
        raise JobPlanError("optional S3 plan must contain 15 explicit 3090 runs")
    return {"study": study, "status": plan["status"], "physical_runs": len(seen),
            "unique_condition_seed_pairs": sum(len(value) for value in per_condition.values()),
            "seeds": sorted(seeds), "hardware_confounds": sorted(confounded) if study == "S1" else [],
            "launchable": production}


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
