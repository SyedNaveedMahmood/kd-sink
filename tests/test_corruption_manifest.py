# -*- coding: utf-8 -*-
"""test_corruption_manifest.py — E6B's 20% symmetric label corruption (WP6, ``03`` §4.3).

``06_TEST_PLAN.md`` §1 names five properties, and each exists to close a specific way the
corrupted-vs-clean contrast could be confounded rather than measured:

* **exactly** ``round(rate * N)`` flipped — sampling each row independently at 20% puts the
  realised rate anywhere near 20%, so the contrast would vary by seed for a reason with
  nothing to do with the hypothesis;
* **class-stratified within 1** — flipping more positives than negatives shifts the label
  prior, and a prior shift is a far simpler explanation for sink drift than anything
  mechanistic;
* **validation never touched** — a corrupted validation label makes every accuracy number
  in `eval_log.jsonl` meaningless, including design §18's E6B criterion 1;
* **distinct across seeds** — the seed is the unit of replication (design §15.1), so two
  seeds sharing a corruption set would be one measurement reported twice;
* **reproducible within a seed** — a resumed run must corrupt the same rows, or the model
  trains on a different dataset either side of the interruption.

Following `06` §5, nothing here asserts what corruption *does*; only that the manifest has
the structure the design specifies.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_sentiment_adaptation as ts  # noqa: E402


def _rows(n=200, balanced=True):
    """``n`` SST-2-shaped training rows, balanced unless asked otherwise."""
    if balanced:
        return [{"example_id": i, "label": i % 2} for i in range(n)]
    # 3:1 positive, to prove stratification is by class share and not by a fixed split.
    return [{"example_id": i, "label": 1 if i % 4 else 0} for i in range(n)]


# ── the four §4.3 constraints ──────────────────────────────────────────────────


@pytest.mark.parametrize("n", [100, 200, 873, 1000])
def test_exactly_the_requested_fraction_is_flipped(n):
    manifest = ts.build_corruption_manifest(_rows(n), rate=0.20, seed=0)
    assert int(manifest["flipped"].sum()) == round(0.20 * n)
    assert len(manifest) == n


@pytest.mark.parametrize("rate", [0.0, 0.05, 0.20, 0.5, 1.0])
def test_the_rate_is_honoured_across_its_whole_range(rate):
    manifest = ts.build_corruption_manifest(_rows(200), rate=rate, seed=0)
    assert int(manifest["flipped"].sum()) == round(rate * 200)


@pytest.mark.parametrize("balanced", [True, False])
def test_flips_are_class_stratified_within_one(balanced):
    manifest = ts.build_corruption_manifest(_rows(200, balanced), rate=0.20, seed=0)
    per_class = manifest[manifest["flipped"]].groupby("original_label").size()
    assert len(per_class) == 2, "both classes must contribute flips"
    assert int(per_class.max() - per_class.min()) <= 1, per_class.to_dict()


def test_a_flip_inverts_the_label_and_an_unflipped_row_keeps_it():
    manifest = ts.build_corruption_manifest(_rows(100), rate=0.20, seed=0)
    flipped = manifest[manifest["flipped"]]
    kept = manifest[~manifest["flipped"]]
    assert (flipped["assigned_label"] != flipped["original_label"]).all()
    assert (kept["assigned_label"] == kept["original_label"]).all()
    assert set(manifest["assigned_label"]) <= {0, 1}


def test_validation_is_refused_outright():
    """Not documented-and-hoped-for: the function raises on any split but train."""
    with pytest.raises(ValueError, match="never modified"):
        ts.build_corruption_manifest(_rows(50), rate=0.20, seed=0, split="validation")


def test_every_training_row_appears_exactly_once():
    manifest = ts.build_corruption_manifest(_rows(200), rate=0.20, seed=0)
    assert manifest["example_id"].is_unique
    assert sorted(manifest["example_id"]) == list(range(200))


# ── seed behaviour ─────────────────────────────────────────────────────────────


def test_the_same_seed_reproduces_the_same_manifest():
    """A resumed run must corrupt the same rows, or it trains on a different dataset."""
    a = ts.build_corruption_manifest(_rows(200), rate=0.20, seed=1)
    b = ts.build_corruption_manifest(_rows(200), rate=0.20, seed=1)
    assert a.equals(b)


def test_different_seeds_produce_different_manifests():
    sets = []
    for seed in (0, 1, 2):
        manifest = ts.build_corruption_manifest(_rows(200), rate=0.20, seed=seed)
        sets.append(frozenset(manifest[manifest["flipped"]]["example_id"]))
    assert len(set(sets)) == 3, "the three pre-registered seeds must differ"
    for left in range(3):
        for right in range(left + 1, 3):
            assert sets[left] != sets[right]


def test_the_seed_changes_which_rows_flip_not_how_many():
    counts = {int(ts.build_corruption_manifest(_rows(200), rate=0.20,
                                               seed=s)["flipped"].sum())
              for s in (0, 1, 2)}
    assert counts == {40}


# ── the shape the corpus provider consumes ─────────────────────────────────────


def test_the_manifest_carries_exactly_the_documented_columns():
    manifest = ts.build_corruption_manifest(_rows(50), rate=0.20, seed=0)
    assert list(manifest.columns) == ["example_id", "original_label", "assigned_label",
                                      "flipped"]


def test_corruption_map_holds_only_the_flipped_rows():
    """``sst2_prompt_corpus(corrupted_manifest=...)`` reads exactly this mapping."""
    manifest = ts.build_corruption_manifest(_rows(100), rate=0.20, seed=0)
    mapping = ts.corruption_map(manifest)
    flipped = manifest[manifest["flipped"]]
    assert len(mapping) == len(flipped) == 20
    for _, row in flipped.iterrows():
        assert mapping[int(row["example_id"])] == int(row["assigned_label"])
        assert mapping[int(row["example_id"])] != int(row["original_label"])
    unflipped_ids = set(manifest[~manifest["flipped"]]["example_id"])
    assert not (set(mapping) & unflipped_ids)


def test_an_empty_or_zero_rate_manifest_maps_to_nothing():
    assert ts.corruption_map(None) == {}
    zero = ts.build_corruption_manifest(_rows(50), rate=0.0, seed=0)
    assert ts.corruption_map(zero) == {}


@pytest.mark.parametrize("rate", [-0.1, 1.5])
def test_an_impossible_rate_is_refused(rate):
    with pytest.raises(ValueError, match=r"rate must be in"):
        ts.build_corruption_manifest(_rows(50), rate=rate, seed=0)
