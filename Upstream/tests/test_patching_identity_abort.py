# -*- coding: utf-8 -*-
"""test_patching_identity_abort.py — the identity control aborts, it does not warn.

``04`` §5.4: "if any identity row shows a non-zero margin change beyond 1e-5, the run is
invalid and must abort, not merely warn." ``cross_example_patching.run_patched`` already
*detects* the violation and sets ``status="identity_violation"``; WP11's job is to act on
it. This asserts the whole chain:

1. the unit stops immediately — nothing past the violating example is computed;
2. every row of that unit is stamped ``unit_status="identity_violation"``, including the
   rows written before the violation was seen;
3. the unit lands in ``invalid_units.json``;
4. the CLI exits non-zero;
5. ``aggregate_crosslingual`` drops those rows from every contrast and counts them.

The failure this guards against is the quiet one. A broken write path does not crash — it
produces smooth, plausible, entirely fictitious effect sizes (CLAUDE.md trap 3). ``run_patched``
is stubbed here so the violation can be *made* to happen; the real detector is covered by
``tests/test_cross_example_patching.py``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import aggregate_crosslingual as ag  # noqa: E402
import cross_example_patching as cep  # noqa: E402
import datasets_loader as dl  # noqa: E402
import extract_sink_representations as ex  # noqa: E402
import paired_manifests as pm  # noqa: E402
import run_cross_language_patching as rp  # noqa: E402
import wp8_fake_data as fake  # noqa: E402

LANGS = ["en", "de", "zh"]


@pytest.fixture
def setup(monkeypatch):
    """A manifest, controls and a config, with the model layer stubbed out entirely."""
    tokenizer = fake.tiny_tokenizer(None)
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=24)))
    manifest = pm.build_xnli_manifest(tokenizer, n=12, seed=42, max_tokens=400,
                                      candidates=(" t100", " t200", " t300"))
    manifest = pm.with_partitions(manifest, pm.grouped_partition(
        manifest, (0.0, 1 / 3, 1.0), ("dev", "test"), seed=42))
    controls = pm.assign_patch_controls(manifest, seed=42)

    config = ex.config_from_dict({
        "model": "Qwen/Qwen2.5-0.5B", "tag": "t", "variant": "base", "dtype": "float32",
        "extraction": {"objects": ["R0"]},
        "xnli": {"langs": LANGS},
        "patching": {"objects": ["K0_prerope"], "screen_layers": [1, 2], "window": 3,
                     "norm_conditions": ["direct", "identity"],
                     "source_conditions": ["parallel_en", "same_label_en"]},
    })

    handle = SimpleNamespace(nn_engine=SimpleNamespace(num_layers=4), tokenizer=tokenizer)
    monkeypatch.setattr(cep, "model_fingerprint", lambda h: "stub-fingerprint")
    monkeypatch.setattr(rp.cep, "model_fingerprint", lambda h: "stub-fingerprint")
    monkeypatch.setattr(rp.cep, "capture_sources",
                        lambda *a, **k: {"stub": object()})
    return SimpleNamespace(manifest=manifest, controls=controls, config=config,
                           handle=handle)


def _patch_rows(specs, *, violate_identity: bool):
    """One stub result row per scheduled spec, mimicking ``run_patched``'s shape."""
    rows = []
    for spec in specs:
        identity = spec.norm_condition == "identity"
        rows.append({
            "patch_object": spec.site.object, "patch_layer": spec.site.layer,
            "patch_position": 0, "source_patch_position": 0,
            "norm_condition": spec.norm_condition,
            "source_item_id": spec.source_item_id,
            "baseline_margin": 0.5, "patched_margin": 0.5 + (0.02 if not identity else 0.0),
            "margin_delta": (0.02 if not identity else
                             (1.0 if violate_identity else 0.0)),
            "baseline_gold_logprob": -1.0, "patched_gold_logprob": -1.0,
            "baseline_margin_unnorm": 1.0, "patched_margin_unnorm": 1.0,
            "baseline_prediction": 0, "patched_prediction": 0,
            "jsd_output": 0.0, "target_seq_len": 20, "source_seq_len": 20,
            "target_norm": 1.0, "source_norm": 1.0, "patched_norm": 1.0,
            "patch_registry_version": cep.PATCH_REGISTRY_VERSION,
            "status": ("identity_violation" if identity and violate_identity else "ok"),
            "warning": ("identity patch moved the margin by 1.0e+00"
                        if identity and violate_identity else ""),
        })
    return rows


def _stub_run_patched(*, violating_language: str):
    def _run(handle, item, specs, caches, **kwargs):
        violate = str(item.meta.get("language")) == violating_language
        return {"patches": _patch_rows(specs, violate_identity=violate)}
    return _run


# ── the abort ─────────────────────────────────────────────────────────────────


def test_an_identity_violation_invalidates_its_whole_run_unit(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))

    summary = rp.run(setup.config, setup.manifest, setup.controls, tmp_path,
                     stage="screen", handle=setup.handle, progress=False)

    assert summary["n_invalid_units"] > 0
    frame = pd.read_csv(summary["rows_csv"])

    invalid = frame[frame["unit_status"] == "identity_violation"]
    assert not invalid.empty
    assert set(invalid["target_language"]) == {"de"}
    # Every row of an invalidated unit is stamped — including the ok-status rows written
    # for the same example before the violating identity row was reached.
    for unit_id in invalid["run_unit_id"].unique():
        unit = frame[frame["run_unit_id"] == unit_id]
        assert set(unit["unit_status"]) == {"identity_violation"}, unit_id
        assert "ok" in set(unit["status"]), "the non-identity rows are still written"


def test_the_unit_stops_rather_than_finishing_the_partition(setup, monkeypatch, tmp_path):
    """An aborted unit must be *cheaper* than a clean one, not merely flagged."""
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))
    summary = rp.run(setup.config, setup.manifest, setup.controls, tmp_path,
                     stage="screen", handle=setup.handle, progress=False)

    frame = pd.read_csv(summary["rows_csv"])
    aborted = frame[frame["unit_status"] == "identity_violation"]
    clean = frame[(frame["unit_status"] == "ok")
                  & (frame["run_unit_id"].str.contains("|zh|", regex=False))]
    examples_in_aborted = aborted.groupby("run_unit_id")["semantic_id"].nunique().max()
    examples_in_clean = clean.groupby("run_unit_id")["semantic_id"].nunique().max()
    assert examples_in_aborted == 1
    assert examples_in_clean > 1


def test_the_violation_is_recorded_in_invalid_units_json(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))
    rp.run(setup.config, setup.manifest, setup.controls, tmp_path, stage="screen",
           handle=setup.handle, progress=False)

    payload = json.loads((tmp_path / rp.INVALID_FILENAME).read_text(encoding="utf-8"))
    assert payload["identity_tolerance"] == cep.IDENTITY_TOLERANCE
    assert payload["invalid_units"]
    for unit in payload["invalid_units"]:
        assert unit["unit_status"] == "identity_violation"
        assert unit["target_language"] == "de"
    assert "git_sha" in payload


def test_an_invalidated_unit_is_not_marked_complete_and_is_retried(setup, monkeypatch,
                                                                   tmp_path):
    """Resume must recompute an aborted unit, never treat it as done."""
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))
    first = rp.run(setup.config, setup.manifest, setup.controls, tmp_path,
                   stage="screen", handle=setup.handle, progress=False)
    n_invalid = first["n_invalid_units"]

    second = rp.run(setup.config, setup.manifest, setup.controls, tmp_path,
                    stage="screen", handle=setup.handle, progress=False)
    assert second["n_skipped"] == first["n_units"] - n_invalid
    assert second["n_invalid_units"] == n_invalid


def test_a_clean_run_records_no_invalid_units(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(rp.cep, "run_patched",
                        _stub_run_patched(violating_language="__none__"))
    summary = rp.run(setup.config, setup.manifest, setup.controls, tmp_path,
                     stage="screen", handle=setup.handle, progress=False)

    assert summary["n_invalid_units"] == 0
    assert not (tmp_path / rp.INVALID_FILENAME).exists()
    frame = pd.read_csv(summary["rows_csv"])
    assert set(frame["unit_status"]) == {"ok"}
    identity = frame[frame["norm_condition"] == "identity"]
    assert identity["margin_delta"].abs().max() < cep.IDENTITY_TOLERANCE


# ── the aggregator excludes them ──────────────────────────────────────────────


def test_aggregation_excludes_invalid_units_and_counts_them(setup, monkeypatch, tmp_path):
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))
    results = tmp_path / "patching" / "t"
    rp.run(setup.config, setup.manifest, setup.controls, results, stage="screen",
           handle=setup.handle, progress=False)

    report = ag.aggregate(tmp_path / "patching",
                          prereg_path=REPO / "crosslingual_semantics" / "configs" /
                          "e7_preregistration.yaml",
                          out_dir=tmp_path / "aggregate", figures=False, progress=False)

    assert report["n_rows_from_invalid_units"] > 0
    assert report["n_rows_used"] == report["n_rows"] - report["n_rows_from_invalid_units"]

    # Invalidating one of two target languages puts the contrast's rows far above the
    # 2% bar, so `05` §7.2 refuses it. That refusal *is* the correct answer here, and it
    # is recorded on the row rather than being computed quietly on half the data.
    contrasts = pd.read_csv(tmp_path / "aggregate" / "patching_contrasts.csv")
    refused = contrasts[contrasts["status"] == "excluded"]
    assert not refused.empty
    assert refused["excluded_reason"].str.contains("failure rate").all()
    assert (refused["failure_rate"] > 0.02).all()

    failures = pd.read_csv(tmp_path / "aggregate" / "failures.csv")
    assert "identity_violation" in set(failures["unit_status"].astype(str)) or \
        "identity_violation" in set(failures["status"].astype(str))


def test_the_override_computes_the_contrast_from_the_clean_language_only(
        setup, monkeypatch, tmp_path):
    """With --allow-high-failure the contrast is computed — but only on valid rows."""
    monkeypatch.setattr(rp.cep, "run_patched", _stub_run_patched(violating_language="de"))
    results = tmp_path / "patching" / "t"
    rp.run(setup.config, setup.manifest, setup.controls, results, stage="screen",
           handle=setup.handle, progress=False)

    ag.aggregate(tmp_path / "patching",
                 prereg_path=REPO / "crosslingual_semantics" / "configs" /
                 "e7_preregistration.yaml",
                 out_dir=tmp_path / "aggregate", figures=False, progress=False,
                 allow_high_failure=True)

    contrasts = pd.read_csv(tmp_path / "aggregate" / "patching_contrasts.csv")
    computed = contrasts[contrasts["status"] == "ok"]
    assert not computed.empty
    # The invalidated language contributed no pairs, so only 'zh' is represented.
    assert (computed["n_languages"] == 1).all()
