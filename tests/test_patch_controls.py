# -*- coding: utf-8 -*-
"""test_patch_controls.py — the four E7 source conditions (WP8).

``06_TEST_PLAN.md`` row 34 and ``02_MODULE_SPEC_common.md`` §5.1. The control assignment is
what separates "patching the parallel English sentence moved the margin" from "patching
*any* English sentence moved the margin", so its constraints are the experiment's internal
validity, not bookkeeping:

* no id is its own control (``parallel_en`` excepted — that is the treatment);
* ``same_label_en`` shares the gold label; ``different_label_en`` does not;
* ``random_en`` is drawn independently of label;
* every ``(semantic_id, target_lang)`` has all four conditions;
* fully deterministic from the seed.

Offline: monkeypatched ``load_dataset``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets_loader as dl  # noqa: E402
import paired_manifests as pm  # noqa: E402

import wp8_fake_data as fake  # noqa: E402


@pytest.fixture()
def manifest(monkeypatch):
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=36)))
    tokenizer = fake.tiny_tokenizer(None)
    return pm.build_xnli_manifest(tokenizer, n=30, seed=42, max_tokens=200)


def test_every_pair_has_all_four_conditions(manifest):
    controls = pm.assign_patch_controls(manifest, seed=42)
    targets = [lang for lang in manifest.languages if lang != "en"]
    expected = len(manifest.semantic_ids) * len(targets) * len(pm.PATCH_CONTROL_CONDITIONS)
    assert len(controls) == expected

    grouped = controls.groupby(["semantic_id", "target_language"])["control_condition"]
    for (_sid, _lang), conditions in grouped:
        assert set(conditions) == set(pm.PATCH_CONTROL_CONDITIONS)
    assert "en" not in set(controls["target_language"])
    assert (controls["status"] == "ok").all(), controls[controls["status"] != "ok"]


def test_label_constraints_hold_for_every_row(manifest):
    controls = pm.assign_patch_controls(manifest, seed=42)
    for row in controls.itertuples():
        condition = row.control_condition
        if condition == "parallel_en":
            assert row.source_semantic_id == row.semantic_id
            assert row.source_label == row.gold_label
        elif condition == "same_label_en":
            assert row.source_semantic_id != row.semantic_id, row
            assert row.source_label == row.gold_label, row
        elif condition == "different_label_en":
            assert row.source_semantic_id != row.semantic_id, row
            assert row.source_label != row.gold_label, row
        else:  # random_en
            assert row.source_semantic_id != row.semantic_id, row


def test_random_en_is_drawn_independently_of_label(manifest):
    """Across the whole assignment, ``random_en`` must hit every label class.

    A ``random_en`` implementation that accidentally reused the same-label pool would pass
    every per-row check above and quietly turn the unrelated control into a second
    same-label condition.
    """
    controls = pm.assign_patch_controls(manifest, seed=42)
    random_rows = controls[controls["control_condition"] == "random_en"]
    assert set(random_rows["source_label"]) == set(controls["gold_label"])

    matched = (random_rows["source_label"] == random_rows["gold_label"]).mean()
    # With three balanced classes, matching by chance is ~1/3. Anything near 0 or 1 means
    # the draw is not independent of label. The bounds are deliberately wide — this is a
    # structural check, not a test of a scientific quantity.
    assert 0.05 < matched < 0.80, matched


def test_assignment_is_deterministic_and_seed_sensitive(manifest):
    a = pm.assign_patch_controls(manifest, seed=42)
    b = pm.assign_patch_controls(manifest, seed=42)
    c = pm.assign_patch_controls(manifest, seed=7)

    key = ["semantic_id", "target_language", "control_condition"]
    a_sorted = a.sort_values(key).reset_index(drop=True)
    b_sorted = b.sort_values(key).reset_index(drop=True)
    c_sorted = c.sort_values(key).reset_index(drop=True)

    assert list(a_sorted["source_semantic_id"]) == list(b_sorted["source_semantic_id"])
    assert list(a_sorted["source_semantic_id"]) != list(c_sorted["source_semantic_id"])


def test_adding_a_language_does_not_perturb_the_others(manifest):
    """The RNG stream is keyed per (id, language), so a manifest can grow safely.

    If assignments came from one global stream, adding a language would silently
    re-assign every previously computed control and invalidate a completed run.
    """
    full = pm.assign_patch_controls(manifest, seed=42)

    from dataclasses import replace
    trimmed_langs = tuple(lang for lang in manifest.languages if lang != "tr")
    trimmed = replace(manifest, languages=trimmed_langs,
                      rows={sid: {lang: manifest.rows[sid][lang]
                                  for lang in trimmed_langs}
                            for sid in manifest.semantic_ids})
    partial = pm.assign_patch_controls(trimmed, seed=42)

    key = ["semantic_id", "target_language", "control_condition"]
    full_de = full[full["target_language"] == "de"].sort_values(key).reset_index(drop=True)
    part_de = partial[partial["target_language"] == "de"].sort_values(key)\
        .reset_index(drop=True)
    assert list(full_de["source_semantic_id"]) == list(part_de["source_semantic_id"])


def test_unassignable_controls_are_recorded_not_dropped(monkeypatch):
    """A single-label manifest cannot form ``different_label_en``; the row still exists."""
    shards = fake.make_xnli(n_rows=12)
    for lang in fake.XNLI_LANGS:
        for row in shards[lang]:
            row["label"] = 0
    monkeypatch.setattr(dl, "load_dataset", fake.fake_load_dataset(xnli=shards))
    tokenizer = fake.tiny_tokenizer(None)
    manifest = pm.build_xnli_manifest(tokenizer, n=9, seed=42, max_tokens=200,
                                      balanced=False)

    controls = pm.assign_patch_controls(manifest, seed=42)
    different = controls[controls["control_condition"] == "different_label_en"]
    assert len(different) > 0
    assert (different["status"] == "unassignable").all()
    assert different["source_semantic_id"].isna().all()
    # The other three conditions are unaffected.
    others = controls[controls["control_condition"] != "different_label_en"]
    assert (others["status"] == "ok").all()


def test_source_language_must_be_in_the_manifest(manifest):
    with pytest.raises(ValueError, match="source_lang"):
        pm.assign_patch_controls(manifest, seed=42, source_lang="xx")
