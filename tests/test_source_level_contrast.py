# -*- coding: utf-8 -*-
"""test_source_level_contrast.py — ``kind: source_level`` (WP12, ``08`` §8.1).

Design §10.12 rows 1 and 3 turn on whether K0 "transfers safely" for an **unrelated**
sentence — the unrelated source's own effect against no patch at all. That is not
expressible as any parallel-versus-control contrast: there is no second arm to subtract.
``margin_delta`` is already ``patched − baseline``, so the one-group mean of that column
*is* the quantity.

The trap this shape exists to avoid is quiet: if ``e7_c8k`` were written as an ordinary
paired contrast against some control, it would return a plausible number that answers a
different question, and the matrix row keyed to it would silently mean something else.

Asserted here: the mean is the one-group mean, ``value_b`` and ``relative_difference`` stay
null, the CI comes from the same hierarchical bootstrap every other contrast uses, and the
failure-rate scope covers only the rows this contrast actually reads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "crosslingual_semantics"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_crosslingual as ag  # noqa: E402

ANALYSIS = {"max_failure_rate": 0.02, "bootstrap_n": 200, "bootstrap_alpha": 0.05,
            "bh_alpha": 0.05, "bootstrap_group_cols": ["semantic_id",
                                                       "target_language"]}


def _rows(*, parallel=0.30, same_label=0.05, languages=("de", "zh", "tr"),
          n_examples=4, status="ok", unit_status="ok"):
    rows = []
    for obj in ("K0_prerope", "V0"):
        for lang in languages:
            for i in range(n_examples):
                for condition, delta in (("parallel_en", parallel),
                                         ("same_label_en", same_label),
                                         ("different_label_en", 0.0),
                                         ("random_en", 0.0)):
                    rows.append({
                        "model": "Qwen/Qwen2.5-0.5B", "model_tag": "t05",
                        "model_variant": "base", "target_language": lang,
                        "semantic_id": f"s{i}", "partition": "test",
                        "patch_object": obj, "patch_layer": 3,
                        "norm_condition": "direct", "source_condition": condition,
                        "margin_delta": delta, "status": status,
                        "unit_status": unit_status, "stage": "window",
                    })
    return ag.normalise_rows(pd.DataFrame(rows))


ENTRY = {"id": "e7_c8k", "kind": "source_level", "text": "unrelated K0 effect",
         "metric": "margin_delta", "source_condition": "same_label_en",
         "objects": ["K0_prerope"], "norm_condition": "direct",
         "family": "e7_secondary"}


# ── it is a one-group mean, not a difference ───────────────────────────────────


def test_the_mean_is_the_unrelated_sources_own_effect_against_the_baseline():
    frame = _rows(parallel=0.30, same_label=0.05)
    row = ag.evaluate_contrast(frame, ENTRY, ANALYSIS, raw=frame)

    assert row["status"] == "ok"
    assert row["absolute_difference"] == pytest.approx(0.05), (
        "the value must be same_label's own margin_delta, not parallel minus same_label")
    assert row["n_pairs"] == 12          # 3 languages x 4 examples, one object


def test_the_parallel_condition_does_not_enter_it():
    """Changing the treatment arm must not move a one-group contrast at all."""
    low = ag.evaluate_contrast(_rows(parallel=0.10), ENTRY, ANALYSIS)
    high = ag.evaluate_contrast(_rows(parallel=0.90), ENTRY, ANALYSIS)
    assert low["absolute_difference"] == pytest.approx(high["absolute_difference"])


def test_value_b_and_relative_difference_stay_null():
    """`08` §8.1 — there is no second arm; a denominator would invent one."""
    row = ag.evaluate_contrast(_rows(), ENTRY, ANALYSIS)
    assert row["value_a"] is None
    assert row["value_b"] is None
    assert row["relative_difference"] is None
    assert row["control_source"] is None
    assert row["treatment_source"] == "same_label_en"


def test_the_object_selects_which_rows_are_read():
    frame = _rows()
    frame.loc[(frame["patch_object"] == "V0")
              & (frame["source_condition"] == "same_label_en"), "margin_delta"] = 0.40
    k0 = ag.evaluate_contrast(frame, ENTRY, ANALYSIS)
    v0 = ag.evaluate_contrast(frame, {**ENTRY, "id": "e7_c8v", "objects": ["V0"]},
                              ANALYSIS)
    assert k0["absolute_difference"] == pytest.approx(0.05)
    assert v0["absolute_difference"] == pytest.approx(0.40)


# ── the statistics are the same ones ───────────────────────────────────────────


def test_the_interval_comes_from_the_hierarchical_bootstrap_over_the_same_clusters():
    row = ag.evaluate_contrast(_rows(), ENTRY, ANALYSIS)
    assert row["ci_uncertainty_kind"] == (
        "hierarchical_bootstrap_over_semantic_id_then_target_language")
    assert row["ci_lo"] is not None and row["ci_hi"] is not None
    assert row["ci_lo"] <= row["absolute_difference"] <= row["ci_hi"]
    assert row["p_value"] is not None and row["min_attainable_p"] is not None


def test_direction_counts_are_still_reported():
    row = ag.evaluate_contrast(_rows(same_label=0.05), ENTRY, ANALYSIS)
    assert row["n_languages_same_direction"] == 3
    assert row["n_model_sizes_same_direction"] == 1


# ── refusals ───────────────────────────────────────────────────────────────────


def test_an_entry_naming_no_source_condition_is_pending_not_guessed():
    row = ag.evaluate_contrast(_rows(), {**ENTRY, "source_condition": None}, ANALYSIS)
    assert row["status"] == "pending"
    assert "names no source_condition" in row["reason"]


def test_no_matching_rows_is_no_data_rather_than_zero():
    row = ag.evaluate_contrast(_rows(), {**ENTRY, "source_condition": "never_used"},
                               ANALYSIS)
    assert row["status"] == "no_data"
    assert row["absolute_difference"] is None


def test_the_failure_rate_is_scoped_to_the_rows_this_contrast_reads():
    """A broken *parallel* unit must not refuse a contrast that never touches it."""
    frame = _rows()
    frame.loc[frame["source_condition"] == "parallel_en", "status"] = "nonfinite"
    row = ag.evaluate_contrast(frame, ENTRY, ANALYSIS, raw=frame)
    assert row["status"] == "ok", row["reason"]
    assert row["failure_rate"] == pytest.approx(0.0)


def test_a_broken_unrelated_unit_does_refuse_it():
    frame = _rows()
    frame.loc[frame["source_condition"] == "same_label_en", "status"] = "nonfinite"
    row = ag.evaluate_contrast(ag.usable_rows(frame), ENTRY, ANALYSIS, raw=frame)
    assert row["status"] == "excluded"
    assert "failure rate" in row["reason"]


def test_the_override_computes_the_refused_one_group_contrast():
    frame = _rows()
    frame.loc[frame["source_condition"] == "same_label_en", "status"] = "nonfinite"
    row = ag.evaluate_contrast(frame, ENTRY, ANALYSIS, raw=frame,
                               allow_high_failure=True)
    assert row["status"] == "ok"
    assert row["failure_rate"] == pytest.approx(1.0)
