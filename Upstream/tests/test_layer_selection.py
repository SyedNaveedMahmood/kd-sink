# -*- coding: utf-8 -*-
"""test_layer_selection.py — WP11's dev/test layer protocol (04 §5.3).

Screening on dev and reporting on test is *the entire defence* against layer-sweep
overfitting, so every part of it is asserted rather than documented: the screen stage can
only touch dev, the window stage can only touch test, the window stage refuses to start
without a recorded selection, overlapping partitions are refused, and the selected layer is
the argmax of the dev effect with a correctly clamped window.

No model is built — every test drives the pure functions on synthetic frames — so there is
no teardown dance.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))

import extract_sink_representations as ex  # noqa: E402
import run_cross_language_patching as rp  # noqa: E402


def _config(**patching):
    base = {
        "objects": ["K0_prerope", "V0"],
        "screen_layers": [1, 3, 5, 7],
        "window": 5,
        "norm_conditions": ["direct", "rescaled", "random", "identity"],
        "source_conditions": ["parallel_en", "same_label_en", "different_label_en",
                              "random_en"],
        "smoke": {"n_examples": 50, "objects": ["K0_prerope", "V0"]},
    }
    base.update(patching)
    return ex.config_from_dict({"model": "Qwen/Qwen2.5-0.5B", "tag": "t",
                                "patching": base, "extraction": {"objects": ["R0"]}})


def _rows(effects_by_layer, *, objects=("K0_prerope", "V0"), languages=("de", "zh"),
          n_examples=4, partition="dev"):
    """Synthetic dev rows whose parallel-minus-control effect is exactly as specified."""
    rows = []
    for obj in objects:
        for layer, effect in effects_by_layer.items():
            for lang in languages:
                for i in range(n_examples):
                    for condition, delta in (("parallel_en", effect), ("same_label_en", 0.0)):
                        rows.append({
                            "model_tag": "t", "target_language": lang,
                            "semantic_id": f"s{i}", "partition": partition,
                            "patch_object": obj, "patch_layer": layer,
                            "norm_condition": "direct", "source_condition": condition,
                            "margin_delta": delta, "status": "ok", "unit_status": "ok",
                        })
    return pd.DataFrame(rows)


# ── the window ────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("selected,width,num_layers,expected", [
    (11, 5, 24, [9, 10, 11, 12, 13]),
    (0, 5, 24, [0, 1, 2, 3, 4]),          # clamped at the bottom, still five layers
    (23, 5, 24, [19, 20, 21, 22, 23]),    # clamped at the top, still five layers
    (2, 5, 4, [0, 1, 2, 3]),              # shallower than the window: every layer
    (13, 5, 28, [11, 12, 13, 14, 15]),
])
def test_window_is_contiguous_centred_and_clamped(selected, width, num_layers, expected):
    assert rp.clamp_window(selected, width, num_layers) == expected


def test_a_clamped_window_keeps_its_width_rather_than_truncating():
    """Truncating at a boundary would give the edge layers fewer measurements."""
    for selected in range(24):
        window = rp.clamp_window(selected, 5, 24)
        assert len(window) == 5
        assert window == list(range(window[0], window[0] + 5))


def test_window_width_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        rp.clamp_window(3, 0, 24)


# ── the selection ─────────────────────────────────────────────────────────────


def test_selection_picks_the_largest_dev_effect_and_records_every_screened_layer():
    frame = _rows({1: 0.01, 3: 0.05, 5: 0.02, 7: -0.03})
    chosen = rp.select_layer(frame, screen_layers=[1, 3, 5, 7], window=5, num_layers=24,
                             primary_objects=["K0_prerope", "V0"])

    assert chosen["status"] == "ok"
    assert chosen["selected_layer"] == 3
    assert chosen["selected_effect"] == pytest.approx(0.05)
    assert [row["layer"] for row in chosen["per_layer"]] == [1, 3, 5, 7]
    assert [row["effect"] for row in chosen["per_layer"]] == \
        pytest.approx([0.01, 0.05, 0.02, -0.03])
    # Per-object effects are recorded too, so the choice can be checked, not taken on trust.
    assert set(chosen["per_layer"][1]["per_object"]) == {"K0_prerope", "V0"}
    assert chosen["window"] == [1, 2, 3, 4, 5]
    assert chosen["partition_selected_on"] == "dev"
    assert chosen["partition_to_report_on"] == "test"
    assert "argmax" in chosen["selection_rule"]


def test_selection_ignores_test_rows_entirely():
    """A large effect on the test partition must not influence the choice."""
    dev = _rows({1: 0.01, 3: 0.02}, partition="dev")
    test = _rows({1: 0.90, 3: 0.02}, partition="test")
    chosen = rp.select_layer(pd.concat([dev, test], ignore_index=True),
                             screen_layers=[1, 3], window=3, num_layers=8)
    assert chosen["selected_layer"] == 3


def test_selection_ignores_rows_from_invalid_units_and_failed_rows():
    frame = _rows({1: 0.01, 3: 0.50})
    frame.loc[frame["patch_layer"] == 3, "unit_status"] = "identity_violation"
    chosen = rp.select_layer(frame, screen_layers=[1, 3], window=3, num_layers=8)
    assert chosen["selected_layer"] == 1


def test_selection_reports_no_effect_rather_than_guessing_a_layer():
    empty = _rows({1: 0.0}).iloc[0:0]
    chosen = rp.select_layer(empty, screen_layers=[1, 3], window=5, num_layers=24)
    assert chosen["status"] == "no_effect"
    assert chosen["selected_layer"] is None
    assert chosen["window"] == []
    assert "screen" in chosen["reason"]


def test_dev_effect_table_pairs_within_target_example():
    """An example scored under only one condition contributes to neither side."""
    frame = _rows({1: 0.04}, n_examples=3, languages=("de",))
    orphan = frame[(frame["source_condition"] == "same_label_en")].iloc[0:1].copy()
    orphan["semantic_id"] = "orphan"
    frame = pd.concat([frame, orphan], ignore_index=True)

    table = rp.dev_effect_table(frame)
    row = table[table["patch_object"] == "K0_prerope"].iloc[0]
    assert row["n_pairs"] == 3            # the orphan is not counted
    assert row["effect"] == pytest.approx(0.04)


# ── stage resolution ──────────────────────────────────────────────────────────


def test_screen_runs_on_dev_and_window_runs_on_test():
    config = _config()
    screen = rp.resolve_stage(config, 24, stage="screen")
    assert screen.partition == "dev"
    assert screen.layers == [1, 3, 5, 7]

    window = rp.resolve_stage(config, 24, stage="window",
                              selection={"selected_layer": 3, "window": [1, 2, 3, 4, 5]})
    assert window.partition == "test"
    assert window.layers == [1, 2, 3, 4, 5]


def test_the_smoke_stage_is_the_phase_1_study():
    plan = rp.resolve_stage(_config(), 24, stage="smoke")
    assert plan.partition == "dev"
    assert plan.max_examples == 50
    assert plan.objects == ["K0_prerope", "V0"]
    assert plan.norm_conditions == ["direct", "rescaled", "random", "identity"]


def test_window_refuses_to_start_without_a_recorded_selection():
    with pytest.raises(ValueError, match="layer_selection.json"):
        rp.resolve_stage(_config(), 24, stage="window", selection=None)


def test_a_screening_schedule_out_of_range_is_refused():
    """`04` §5.3's schedules are depth-specific; 5/11/17/22 on a 4-layer model is a bug."""
    with pytest.raises(ValueError, match="out of range"):
        rp.resolve_stage(_config(screen_layers=[5, 11, 17, 22]), 4, stage="screen")


def test_dropping_the_identity_control_is_refused():
    config = _config(norm_conditions=["direct", "rescaled", "random"])
    with pytest.raises(ValueError, match="identity"):
        rp.resolve_stage(config, 24, stage="screen")


def test_an_unknown_stage_is_refused():
    with pytest.raises(ValueError, match="unknown stage"):
        rp.resolve_stage(_config(), 24, stage="production")


# ── the patch schedule ────────────────────────────────────────────────────────


def test_identity_is_scheduled_first_so_a_violation_aborts_early():
    plan = rp.resolve_stage(_config(), 24, stage="screen")
    site = rp.cep.PatchSite("K0_prerope", 3)
    controls = {condition: {"source_semantic_id": f"src_{condition}",
                            "source_label": 0, "gold_label": 0}
                for condition in plan.source_conditions}
    schedule = rp.build_specs(plan, site, controls, seed=42)

    assert schedule[0][0].norm_condition == "identity"
    assert schedule[0][1] == rp.NO_SOURCE


def test_source_independent_norms_are_written_once_not_once_per_source():
    plan = rp.resolve_stage(_config(), 24, stage="screen")
    site = rp.cep.PatchSite("V0", 3)
    controls = {condition: {"source_semantic_id": f"src_{condition}",
                            "source_label": 0, "gold_label": 0}
                for condition in plan.source_conditions}
    schedule = rp.build_specs(plan, site, controls, seed=42)

    by_norm: dict = {}
    for spec, source_condition, _control in schedule:
        by_norm.setdefault(spec.norm_condition, []).append(source_condition)
    assert by_norm["identity"] == [rp.NO_SOURCE]
    assert by_norm["random"] == [rp.NO_SOURCE]
    assert sorted(by_norm["direct"]) == sorted(plan.source_conditions)
    assert sorted(by_norm["rescaled"]) == sorted(plan.source_conditions)
    assert len(schedule) == 2 + 2 * len(plan.source_conditions)
