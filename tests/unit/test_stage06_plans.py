"""Stage 06 config and explicit job-plan gates."""

import json
from pathlib import Path

import pytest

from sinklab.config import ConfigError, VARIANTS, resolve_config
from sinklab.job_plan import JobPlanError, inspect_job, validate_job_plan


ROOT = Path(__file__).resolve().parents[2] / "configs"


@pytest.mark.parametrize("study", ["S1", "S3"])
def test_every_independent_config_is_strict(study):
    for path in (ROOT / study.lower()).glob("*.json"):
        raw = json.loads(path.read_text())
        assert resolve_config(raw, seed=0).variant == VARIANTS[study][raw["condition"]]
        with pytest.raises(ConfigError, match="seed"):
            resolve_config(raw, seed=None)
        altered = {**raw, "variant": "wrong"}
        with pytest.raises(ConfigError, match="variant"):
            resolve_config(altered, seed=0)
        altered = {**raw, "student": {**raw["student"], "heads": 13}}
        with pytest.raises(ConfigError, match="dimensions"):
            resolve_config(altered, seed=0)
        assert len(list((ROOT / study.lower()).glob("*.json"))) == (12 if study == "S1" else 5)


def test_rel_on_4080_is_refused():
    raw = json.loads((ROOT / "s1" / "c4_rtx3090.json").read_text())
    raw["device_role"] = "rtx4080super"
    with pytest.raises(ConfigError, match="C4"):
        resolve_config(raw, seed=0)


@pytest.mark.parametrize("name,count,unique", [
    ("s1_jobs_draft.json", 7, 7),
    ("s1_jobs_21_draft.json", 21, 21),
    ("s1_jobs_seed1_draft.json", 7, 7),
    ("s1_jobs_seed2_draft.json", 7, 7),
    ("s3_jobs_draft.json", 15, 15),
])
def test_explicit_plans_validate_without_launch(name, count, unique):
    plan = json.loads((ROOT / name).read_text())
    result = validate_job_plan(plan, ROOT)
    assert result["physical_runs"] == count
    assert result["unique_condition_seed_pairs"] == unique
    assert result["launchable"] is False
    assert result["status"] == "draft"
    with pytest.raises(JobPlanError, match="duplicate"):
        validate_job_plan({**plan, "jobs": plan["jobs"] + [plan["jobs"][0]]}, ROOT)
    one = inspect_job(plan, ROOT, plan["jobs"][0]["run_id"])
    assert one["seed"] == plan["jobs"][0]["seed"]
    assert one["action"] == "inspection_only" and one["launchable"] is False
    with pytest.raises(JobPlanError, match="exactly one"):
        inspect_job(plan, ROOT, "unlisted-run")


def test_s1_plans_require_all_3090_and_allow_primary_seed_alone():
    plan = json.loads((ROOT / "s1_jobs_draft.json").read_text())
    assert validate_job_plan(plan, ROOT)["seeds"] == [0]
    changed = json.loads(json.dumps(plan))
    changed["jobs"][0]["config"] = "s1/c0_rtx4080super.json"
    changed["jobs"][0]["run_id"] = "s1-c0-seed0-rtx4080super"
    with pytest.raises(JobPlanError, match="RTX 3090"):
        validate_job_plan(changed, ROOT)
