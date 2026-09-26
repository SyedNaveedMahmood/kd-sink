# -*- coding: utf-8 -*-
"""Contracts for the isolated GPT-2 Medium E6B scale extension."""

from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402
import evaluate_transformation as ev  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import train_sentiment_adaptation as ts  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
EXPERIMENT = "e6b_gpt2_medium"
MODEL = "openai-community/gpt2-medium"
REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
PREREG = CONFIG_DIR / "e6b_gpt2_medium_preregistration.yaml"

FILES = {
    "F1": "e6b_gpt2_medium_f1.yaml",
    "F2": "e6b_gpt2_medium_f2.yaml",
    "F3": "e6b_gpt2_medium_f3.yaml",
    "F4": "e6b_gpt2_medium_f4.yaml",
}


def _load(path: Path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def _without_scale_identity(config):
    payload = deepcopy(config)
    for key in ("experiment_id", "experiment_family", "preregistration", "base_model",
                "base_revision", "tokenizer", "tokenizer_revision"):
        payload.pop(key, None)
    return payload


def test_all_four_scale_extension_configs_are_shipped_and_isolated():
    assert {p.name for p in CONFIG_DIR.glob("e6b_gpt2_medium_f*.yaml")} == set(
        FILES.values())
    for condition, name in FILES.items():
        config = _load(CONFIG_DIR / name)
        assert config["condition"] == condition
        assert config["experiment_id"] == EXPERIMENT
        assert config["experiment_family"] == "e6b"
        assert config["preregistration"] == (
            "transformation_inheritance/configs/e6b_gpt2_medium_preregistration.yaml")


@pytest.mark.parametrize("condition,name", sorted(FILES.items()))
def test_model_and_tokenizer_are_pinned_to_the_same_gpt2_medium_revision(condition, name):
    config = _load(CONFIG_DIR / name)
    assert config["base_model"] == config["tokenizer"] == MODEL
    assert config["base_revision"] == config["tokenizer_revision"] == REVISION
    assert len(REVISION) == 40


@pytest.mark.parametrize("condition,name", sorted(FILES.items()))
def test_scale_extension_changes_only_model_and_experiment_identity(condition, name):
    extension = _load(CONFIG_DIR / name)
    original = _load(CONFIG_DIR / f"e6b_{condition.lower()}.yaml")
    assert _without_scale_identity(extension) == _without_scale_identity(original)
    assert ts.assert_effective_batch(extension) == 32


def test_extension_registration_reuses_the_original_e6b_rules_unchanged():
    original_path = CONFIG_DIR / "e6_preregistration.yaml"
    original = _load(original_path)
    extension = _load(PREREG)
    assert extension["version"] == "e6b_gpt2_medium_prereg_v1"
    assert extension["experiment_id"] == EXPERIMENT
    assert extension["source_preregistration_sha256"] == hashlib.sha256(
        original_path.read_bytes()).hexdigest()
    assert extension["e6b"] == original["e6b"]

    def operational(criteria):
        rows = []
        for criterion in criteria:
            row = deepcopy(criterion)
            row.pop("source", None)
            row.pop("decision_taken", None)
            rows.append(row)
        return rows

    assert operational(extension["e6b_go_no_go"]["criteria"]) == operational(
        original["e6b_go_no_go"]["criteria"])
    assert extension["e6b_go_no_go"]["combine"] == "all"


def test_run_identity_and_evaluator_dispatch_follow_the_extension_id(tmp_path):
    config = _load(CONFIG_DIR / FILES["F1"])
    setup = ts.prepare_run(config, seed=2, output_dir=tmp_path, smoke=True,
                           n_train=8, n_valid=4, epochs=1)
    path = ts.write_run_config(setup)
    payload = json.loads(path.read_text(encoding="utf-8"))

    assert setup.run_dir == tmp_path / EXPERIMENT / "F1" / "seed2"
    assert payload["run_id"] == f"{EXPERIMENT}_F1_seed2"
    assert payload["experiment_id"] == EXPERIMENT
    assert payload["experiment_family"] == "e6b"
    assert ev.is_e6b_run(ev.read_run_info(setup.run_dir)) is True

    unrelated = ev.RunInfo(
        run_dir=Path("."), run_id="e6a_gpt2_G0_seed0",
        experiment_id="e6a_gpt2", condition="G0", seed=0, teacher=None,
        teacher_revision=None, tokenizer_name=None, smoke=True, raw={})
    assert ev.is_e6b_run(unrelated) is False


def test_aggregator_cli_exposes_a_separate_e6b_arm_selector():
    args = ag.build_parser().parse_args(["--e6b-experiment", EXPERIMENT])
    assert args.e6b_experiment == EXPERIMENT


def test_long_local_model_ids_get_bounded_collision_resistant_cache_names():
    assert fr._safe("short/model") == "short_model"
    first = fr._safe("C:/very/long/" + "model/" * 80 + "one")
    second = fr._safe("C:/very/long/" + "model/" * 80 + "two")
    assert len(first) == len(second) <= 96
    assert first != second
