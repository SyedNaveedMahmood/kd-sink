# -*- coding: utf-8 -*-
"""test_corpus_pair_contrast.py — ``kind: corpus_pair`` (WP12, ``08`` §2.1).

Design §8.7 contrast 5 is "in-domain versus cross-domain inheritance": two *corpora*
within each condition, not two conditions within one corpus. `08` §2.1 adds the shape and
requires the per-condition breakdown to be recorded, so that a single condition driving the
pooled effect is visible rather than averaged away.

Two properties carry most of the weight:

* the cross-domain corpus must actually reach the contrast. Before WP12 ``aggregate()``
  filtered the frame with ``DEFAULT_CORPUS_PREFIXES``, which drops ``e1_100x40`` — the
  contrast would have been computable on paper and empty in practice;
* an entry without ``kind:`` must keep taking the old condition-pair path untouched.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402

IN_DOMAIN = "tinystories_validation_sink_300"
CROSS = "e1_100x40"


def _row(condition, seed, corpus, cos, step=1000):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": corpus, "status": "ok", "n_items": 100, "n_failed": 0,
            "intervention_registry_version": "v1", "validation_ce": 2.5,
            "fingerprint_cosine_to_teacher": cos, "n_keys_used": 9}


def _frame(gaps):
    """``gaps[condition]`` is that condition's in-domain minus cross-domain difference."""
    rows = []
    for condition, gap in gaps.items():
        for seed in (0, 1, 2):
            rows.append(_row(condition, seed, IN_DOMAIN, 0.50))
            rows.append(_row(condition, seed, CROSS, 0.50 - gap))
    return pd.DataFrame(rows)


ENTRY = {"id": "e6a_c5", "kind": "corpus_pair", "text": "in vs cross domain",
         "metric": "fingerprint_cosine_to_teacher", "conditions": ["D0", "D1", "D2"],
         "corpus_a": IN_DOMAIN, "corpus_b": CROSS, "selection": "equal_step",
         "family": "e6a_primary"}


# ── the pairing itself ─────────────────────────────────────────────────────────


def test_pairs_two_corpora_within_each_condition():
    frame = _frame({"D0": 0.10, "D1": 0.10, "D2": 0.10})
    row = ag.evaluate_contrast(frame, ENTRY, {})

    assert row["status"] == "ok"
    assert row["mean_diff"] == pytest.approx(0.10)
    assert row["n_units"] == 9          # 3 conditions x 3 seeds
    assert row["n_seeds"] == 3
    assert row["n_conditions"] == 3


def test_the_per_condition_breakdown_is_recorded_not_averaged_away():
    """`08` §2.1: one condition driving the pooled effect must be visible."""
    frame = _frame({"D0": 0.00, "D1": 0.00, "D2": 0.30})
    row = ag.evaluate_contrast(frame, ENTRY, {})

    assert row["mean_diff"] == pytest.approx(0.10)     # pooled hides the concentration
    breakdown = json.loads(row["per_condition"])
    assert breakdown["D0"]["mean_diff"] == pytest.approx(0.0)
    assert breakdown["D1"]["mean_diff"] == pytest.approx(0.0)
    assert breakdown["D2"]["mean_diff"] == pytest.approx(0.30)
    assert {v["n"] for v in breakdown.values()} == {3}


def test_the_scope_names_both_corpora():
    row = ag.evaluate_contrast(_frame({"D0": 0.1}), ENTRY, {})
    assert IN_DOMAIN in row["corpus_id"] and CROSS in row["corpus_id"]


# ── the cross-domain corpus really reaches the contrast ────────────────────────


def test_the_cross_domain_corpus_survives_the_default_corpus_filter(tmp_path):
    """The regression `08` §2.1 exists to prevent: `e1_100x40` filtered out upstream."""
    path = tmp_path / "e6a" / "all" / "seed0" / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame({"D0": 0.10, "D1": 0.10, "D2": 0.10}).to_csv(path, index=False)

    prereg = {"contrasts": [ENTRY], "go_no_go": [], "analysis": {}, "e6b": {},
              "version": "t"}
    frame = ag.load_all_metrics(tmp_path)
    usable, _ = ag.ok_rows(frame, max_failure_rate=0.02, allow_high_failure=False)
    # The default selection would drop the cross-domain rows; the contrast must not.
    assert CROSS not in set(ag.select_corpus(usable, None)["corpus_id"])
    row = ag.build_contrasts(usable, prereg, {}).iloc[0]
    assert row["status"] == "ok" and row["n_units"] == 9


def test_a_missing_corpus_is_no_data_with_a_reason_not_a_silent_zero():
    frame = _frame({"D0": 0.1})
    frame = frame[frame["corpus_id"] == IN_DOMAIN]          # cross-domain never measured
    row = ag.evaluate_contrast(frame, ENTRY, {})
    assert row["status"] == "no_data"
    assert row["mean_diff"] is None
    units = json.loads(row["per_seed_json"])
    assert units and all("skipped" in unit for unit in units)
    assert CROSS in units[0]["skipped"]


# ── entries without `kind` are untouched ───────────────────────────────────────


def test_an_entry_without_kind_still_takes_the_condition_pair_path():
    frame = _frame({"D0": 0.0, "D2": 0.0})
    frame.loc[(frame["condition"] == "D2") & (frame["corpus_id"] == IN_DOMAIN),
              "fingerprint_cosine_to_teacher"] = 0.80
    row = ag.evaluate_contrast(frame, {"id": "c", "metric":
                                       "fingerprint_cosine_to_teacher",
                                       "condition_a": "D2", "condition_b": "D0",
                                       "corpus_id": IN_DOMAIN}, {})
    assert row["kind"] is None
    assert row["mean_diff"] == pytest.approx(0.30)
    assert row["n_units"] == 3


def test_an_explicit_corpus_override_beats_the_entrys_own_corpus():
    """``--corpus`` is an operator override and is recorded as one."""
    frame = _frame({"D0": 0.0, "D2": 0.0})
    frame.loc[(frame["condition"] == "D2") & (frame["corpus_id"] == CROSS),
              "fingerprint_cosine_to_teacher"] = 0.90
    entry = {"id": "c", "metric": "fingerprint_cosine_to_teacher",
             "condition_a": "D2", "condition_b": "D0", "corpus_id": IN_DOMAIN}
    row = ag.evaluate_contrast(frame, entry, {}, corpus_override=CROSS)
    assert row["corpus_id"] == CROSS
    assert row["corpus_override"] == CROSS
    assert row["mean_diff"] == pytest.approx(0.40)      # 0.90 - 0.50, on the cross corpus
