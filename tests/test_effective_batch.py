# -*- coding: utf-8 -*-
"""test_effective_batch.py — the E6B factorial's fixed optimisation budget (WP6).

design §16.2 fixes the **effective** batch at 32 for every E6B condition: LoRA reaches it
as 32×1 and full fine-tuning as 16×2. The 2×2's entire content is
(LoRA vs full) × (clean vs corrupt); if one arm also saw a different number of examples per
update, the "adaptation method" axis would be measuring optimisation instead, and no amount
of downstream statistics could separate the two afterwards.

``06_TEST_PLAN.md`` §1 therefore asks for this over **every shipped F-config**, not over a
hand-made one — a config that drifts is the realistic failure, and a test that builds its
own input would never see it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_sentiment_adaptation as ts  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
SHIPPED = sorted(CONFIG_DIR.glob("e6b_f*.yaml"))


def _load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


# ── the shipped configs ────────────────────────────────────────────────────────


def test_all_four_conditions_are_shipped():
    assert {p.stem for p in SHIPPED} == {"e6b_f1", "e6b_f2", "e6b_f3", "e6b_f4"}


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_every_shipped_config_has_effective_batch_32(path):
    config = _load(path)
    assert ts.assert_effective_batch(config) == ts.REQUIRED_EFFECTIVE_BATCH == 32


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_the_shipped_pairing_matches_design_9_5(path):
    """LoRA is 32x1 and full fine-tuning 16x2 — the split §4.4 pre-registers."""
    config = _load(path)
    optim = config["optim"]
    expected = {"lora": (32, 1), "full": (16, 2)}[config["adaptation"]]
    assert (optim["per_device_batch_size"], optim["grad_accum"]) == expected


@pytest.mark.parametrize("path", SHIPPED, ids=lambda p: p.stem)
def test_prepare_run_refuses_before_a_single_step_is_taken(path):
    """The assertion must sit at startup, not after the first checkpoint."""
    config = _load(path)
    config["optim"]["grad_accum"] = config["optim"]["grad_accum"] + 1
    with pytest.raises(ValueError, match="effective batch"):
        ts.prepare_run(config, seed=0, output_dir=Path("/nonexistent"), smoke=True)


# ── the assertion itself ───────────────────────────────────────────────────────


@pytest.mark.parametrize("per_device,accum", [(32, 1), (16, 2), (8, 4), (4, 8), (1, 32)])
def test_any_factorisation_of_32_is_accepted(per_device, accum):
    assert ts.assert_effective_batch(
        {"condition": "F?", "optim": {"per_device_batch_size": per_device,
                                      "grad_accum": accum}}) == 32


@pytest.mark.parametrize("per_device,accum", [(32, 2), (16, 1), (8, 3), (0, 32), (32, 0)])
def test_anything_else_is_refused_with_the_reason(per_device, accum):
    with pytest.raises(ValueError) as excinfo:
        ts.assert_effective_batch(
            {"condition": "F9", "optim": {"per_device_batch_size": per_device,
                                          "grad_accum": accum}})
    message = str(excinfo.value)
    assert "F9" in message
    assert "§16.2" in message
    assert "confounds" in message, (
        "the refusal must say *why* — a reader who only sees a number will be tempted to "
        "change the number")


# ── the factorial the pre-registration expects ─────────────────────────────────


def test_the_shipped_conditions_cover_the_preregistered_2x2():
    """`e6_preregistration.yaml` names F1..F4 as (adaptation) x (label_quality)."""
    import aggregate_transformation as ag

    declared = ag.load_preregistration()["e6b"]["factorial"]["conditions"]
    shipped = {}
    for path in SHIPPED:
        config = _load(path)
        shipped[config["condition"]] = [
            config["adaptation"],
            "corrupt" if config.get("corrupt_labels") else "clean"]
    assert shipped == {k: list(v) for k, v in declared.items()}, (
        "the configs and the pre-registered factorial must name the same four cells")
