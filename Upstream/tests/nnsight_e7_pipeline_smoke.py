# -*- coding: utf-8 -*-
"""nnsight_e7_pipeline_smoke.py — the whole E7 pipeline, offline (WP9 + WP11).

    ./.venv/Scripts/python.exe tests/nnsight_e7_pipeline_smoke.py

Runs manifests -> extraction -> retrieval -> multilingual fingerprints -> patching
(screen + window) -> aggregation on a random tiny Qwen2 and ``wp8_fake_data``-backed
FLORES/XNLI fixtures. CPU, fp32, no downloads. ``tests/nnsight_e7_smoke.py`` covers the
WP7 patching *primitives*; this covers the WP9/WP11 *scripts* that drive them.

Two fixture details that matter
-------------------------------
* The tiny checkpoint is built with ``vocab_size=wp8_fake_data.FAKE_VOCAB_SIZE`` because
  both builders emit the same WordLevel vocabulary (``t{i} -> i+3``), so the manifest's
  token ids mean the same thing to the model. The manifest is then built with the
  *checkpoint's own* tokenizer, which is what the extraction tokenizer-coupling check
  compares against.
* The XNLI label candidates are overridden to in-vocab tokens. The real
  ``" entailment"/" neutral"/" contradiction"`` all map to ``[UNK]`` under this fixture
  tokenizer, which would make every candidate identical, every margin exactly 0, and the
  layer selection vacuous — a smoke test that passes while measuring nothing.

Exit code is 0 on success; every phase asserts, so a regression fails loudly.
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "crosslingual_semantics",
              Path(__file__).resolve().parent):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import aggregate_crosslingual as ag  # noqa: E402
import cross_example_patching as cep  # noqa: E402
import datasets_loader as dl  # noqa: E402
import evaluate_crosslingual_retrieval as ret  # noqa: E402
import extract_sink_representations as ex  # noqa: E402
import paired_manifests as pm  # noqa: E402
import run_cross_language_patching as rp  # noqa: E402
import run_multilingual_fingerprint as mf  # noqa: E402
import wp8_fake_data as fake  # noqa: E402
from fingerprint_runner import load_handle  # noqa: E402
from nnsight_smoke_utils import create_tiny_qwen_checkpoint  # noqa: E402

N_LAYERS = 4
N_FLORES = 9
N_XNLI = 12
FLORES_LANGS = ["eng_Latn", "deu_Latn", "tur_Latn"]
XNLI_LANGS = ["en", "de", "zh", "tr"]

#: See the module docstring: the real candidates collapse to [UNK] under this tokenizer.
CANDIDATES = (" t100", " t200", " t300")

PREREG = REPO / "crosslingual_semantics" / "configs" / "e7_preregistration.yaml"


def _checkpoint(root: Path):
    from transformers import AutoTokenizer

    path = root / "qwen"
    create_tiny_qwen_checkpoint(path, seed=4003, vocab_size=fake.FAKE_VOCAB_SIZE,
                                n_positions=256, n_layer=N_LAYERS)
    return path, AutoTokenizer.from_pretrained(str(path))


def _manifests(tokenizer):
    """FLORES and XNLI manifests through the real WP8 builders, on fake shards."""
    original = dl.load_dataset
    dl.load_dataset = fake.fake_load_dataset(flores=fake.make_flores(n_rows=18),
                                             xnli=fake.make_xnli(n_rows=24))
    try:
        flores = pm.build_flores_manifest(tokenizer, langs=FLORES_LANGS, n=N_FLORES,
                                          seed=42)
        xnli = pm.build_xnli_manifest(tokenizer, langs=XNLI_LANGS, n=N_XNLI, seed=42,
                                      max_tokens=400, candidates=CANDIDATES)
    finally:
        dl.load_dataset = original

    flores = pm.with_partitions(flores, pm.grouped_partition(
        flores, (0.0, 1 / 3, 1.0), ("procrustes_train", "procrustes_test"), seed=42))
    xnli = pm.with_partitions(xnli, pm.grouped_partition(
        xnli, (0.0, 1 / 3, 1.0), ("dev", "test"), seed=42))

    pm.verify_manifest(flores)
    pm.verify_manifest(xnli)
    controls = pm.assign_patch_controls(xnli, seed=42)
    print(f"  [manifests] flores={len(flores)} ids, xnli={len(xnli)} ids, "
          f"controls={len(controls)} rows")
    return flores, xnli, controls


def _config(ckpt: Path):
    config = ex.smoke_config(str(ckpt), tag="tiny_qwen")
    config.raw["xnli"] = {"langs": XNLI_LANGS}
    config.raw["patching"] = {
        "objects": ["K0_prerope", "V0", "R0", "Kmid_prerope", "Vmid", "Rmid"],
        "screen_layers": [0, 1, 2, 3],
        "window": 3,
        "norm_conditions": ["direct", "rescaled", "random", "identity"],
        "source_conditions": ["parallel_en", "same_label_en", "different_label_en",
                              "random_en"],
        "smoke": {"n_examples": 4, "objects": ["K0_prerope", "V0"]},
    }
    return config


def _extract(config, flores, handle, root: Path) -> Path:
    summary = ex.extract(config, flores, root / "reps", handle=handle,
                         langs=FLORES_LANGS, progress=False)
    assert summary["n_failed"] == 0, summary
    assert summary["layers"] == list(range(N_LAYERS))

    reps = root / "reps" / config.tag
    for lang in FLORES_LANGS:
        index = json.loads((reps / lang / "index.json").read_text(encoding="utf-8"))
        assert sorted(index["row_of"]) == sorted(index["semantic_ids"])
        for obj in config.objects:
            array = np.load(reps / lang / f"{obj}.npy", mmap_mode="r")
            assert array.shape == (N_FLORES, N_LAYERS, ex.object_dim(handle, obj)), obj
            assert array.dtype == np.float32

    # The size bound is what catches an accidental full-hidden-state capture (04 §3.2).
    budget = sum(N_FLORES * N_LAYERS * ex.object_dim(handle, obj)
                 for obj in config.objects) * len(FLORES_LANGS) * 4
    actual = sum(f.stat().st_size for f in reps.rglob("*") if f.is_file())
    assert actual < 3 * budget, f"{actual} bytes against a {budget}-byte budget"

    # Resume must recompute nothing.
    again = ex.extract(config, flores, root / "reps", handle=handle,
                       langs=FLORES_LANGS, progress=False)
    assert again["n_skipped"] == len(FLORES_LANGS), again

    print(f"  [extract] {len(config.objects)} objects x {N_LAYERS} layers x "
          f"{len(FLORES_LANGS)} languages, {actual} bytes (budget {3 * budget})")
    return reps


def _retrieval(reps: Path, flores, root: Path) -> Path:
    out = root / "retrieval"
    summary = ret.evaluate(reps, flores, out, progress=False, seed=42)
    assert summary["n_pair_rows"] > 0
    assert summary["n_procrustes_train"] and summary["n_procrustes_test"]

    pairs = pd.read_csv(out / "retrieval_by_pair.csv")
    assert set(pairs["alignment"]) == {"cosine", "procrustes"}
    # Self-pairs are the continuous correctness monitor: after per-language centring a
    # language must retrieve itself perfectly.
    self_pairs = pairs[pairs["is_self_pair"] & (pairs["alignment"] == "cosine")]
    assert not self_pairs.empty
    assert float(self_pairs["top1"].min()) == 1.0, self_pairs["top1"].min()

    objects = pd.read_csv(out / "retrieval_by_object.csv")
    with_control = objects[objects["control_object"].notna()]
    assert not with_control.empty, "no position-0 row carries its middle-token control"
    assert with_control["position0_minus_mid_top1_points"].notna().all()

    probe = pd.read_csv(out / "language_probe.csv")
    assert set(probe["grouped_by"]) == {"semantic_id"}
    assert (probe["status"] == "ok").all()

    print(f"  [retrieval] {summary['n_pair_rows']} pair rows, "
          f"{summary['n_object_rows']} object rows, {summary['n_probe_rows']} probes")
    return out


def _fingerprints(config, flores, handle, root: Path) -> None:
    summary = mf.run(config, flores, root / "fingerprints", corpus_variant="matched",
                     handle=handle, langs=FLORES_LANGS, progress=False)
    assert summary["n_failed"] == 0, summary
    assert summary["n_written"] == len(FLORES_LANGS)

    frame = pd.read_csv(summary["summary_csv"])
    assert len(frame) == len(FLORES_LANGS)
    assert frame["baseline_sink"].notna().all()
    assert frame["script"].notna().all(), "the §10.5 covariates must travel with the row"
    print(f"  [fingerprint] {summary['n_written']} languages, band "
          f"{summary['layer_band']}")


def _patching(config, xnli, controls, handle, root: Path) -> Path:
    out = root / "patching" / config.tag
    screen = rp.run(config, xnli, controls, out, stage="screen", handle=handle,
                    progress=False, source_cache_path=out / "source_cache.pt")
    assert screen["partition"] == "dev"
    assert screen["n_invalid_units"] == 0, "the identity control must pass on a clean run"
    assert screen["n_failed_rows"] == 0, screen

    frame = pd.read_csv(screen["rows_csv"])
    assert list(frame.columns) == list(rp.ROW_COLUMNS), "05 §3 schema drifted"
    identity = frame[frame["norm_condition"] == "identity"]
    assert not identity.empty
    assert float(identity["margin_delta"].abs().max()) < cep.IDENTITY_TOLERANCE

    chosen = rp.select_layer(frame, screen_layers=screen["layers"], window=3,
                             num_layers=screen["num_layers"],
                             primary_objects=["K0_prerope", "V0"])
    assert chosen["status"] == "ok", chosen
    assert len(chosen["window"]) == 3
    rp.write_layer_selection(out, chosen)

    window = rp.run(config, xnli, controls, out, stage="window", handle=handle,
                    selection=rp.read_layer_selection(out), progress=False,
                    source_cache_path=out / "source_cache.pt", max_examples=4)
    assert window["partition"] == "test"
    assert window["layers"] == chosen["window"]

    both = pd.read_csv(window["rows_csv"])
    dev_ids = set(both[both["partition"] == "dev"]["semantic_id"])
    test_ids = set(both[both["partition"] == "test"]["semantic_id"])
    assert not (dev_ids & test_ids), "the layer was selected on ids it is reported on"
    assert (out / "runtime_estimate.json").exists()

    print(f"  [patch] screen {screen['n_rows']} rows on dev, selected layer "
          f"{chosen['selected_layer']} -> window {chosen['window']}, "
          f"window {window['n_rows']} rows on test")
    return root / "patching"


def _aggregate(results: Path, retrieval: Path, root: Path) -> None:
    report = ag.aggregate(results, prereg_path=PREREG, retrieval_path=retrieval,
                          out_dir=root / "aggregate", figures=True, progress=False)
    out = root / "aggregate"

    for name in ("patching_contrasts.csv", "exploratory_layer_sweep.csv",
                 "table4_patch_effects.csv", "failures.csv",
                 "interpretation_matrix.json", "e7_go_no_go.json",
                 "aggregate_summary.json"):
        assert (out / name).exists(), name
    assert (out / "figures" / "fig6_patch_decomposition.pdf").exists()

    # `e7_prereg_v4` resolves D5/D6/D7, so the verdict now comes from the numbers rather
    # than being suppressed by an unmade decision. What must still hold is that nothing
    # was decided silently: no sentinel remains anywhere.
    assert report["pending_decisions"] == [], report["pending_decisions"]
    assert report["pending_decisions_blocking"] == [], \
        report["pending_decisions_blocking"]
    assert report["interpretation_matrix_status"] != "PENDING_DECISION_THRESHOLD"
    assert report["decision"] in ("continue", "stop"), report["decision"]
    assert report["supported"] is not None

    decision = json.loads((out / "e7_go_no_go.json").read_text(encoding="utf-8"))
    assert decision["preregistration_version"] == "e7_prereg_v4", \
        decision["preregistration_version"]
    assert not [c for c in decision["criteria"]
                if str(c.get("status", "")).startswith("PENDING")]

    # D6 resolved to `any`: §18 criterion 3's two objects fold into one verdict, written
    # to its own key so `n_met` cannot count K0 and V0 as two separate criteria.
    combined = decision["criterion_3"]
    assert combined is not None and combined["combine"] == "any", combined
    assert combined["met"] is not None, combined
    assert set(combined["limbs"]) == {"e7_3k", "e7_3v"}, combined["limbs"]

    # D5 took route C: e7_c6 stays registered and untiered, so it computes no value.
    untiered = pd.read_csv(out / "patching_contrasts.csv").set_index("id")
    assert "e7_c6" in untiered.index, "the contrast must remain registered, not deleted"
    assert pd.isna(untiered.loc["e7_c6", "absolute_difference"]), \
        "an untiered language_group contrast must report no grouped value"

    # §18 criteria 4 and 5 are the claim gate itself and must not be counted beside it.
    assert decision["n_met_all_criteria"] >= decision["n_met"]
    gate_components = [c for c in decision["criteria"]
                       if c.get("evaluation") == "claim_gate_component"]
    assert len(gate_components) == 2, gate_components

    # Criteria e7_1/e7_2 read WP9's retrieval table, so they must have a real observation.
    retrieval_criteria = [c for c in decision["criteria"] if c["id"] in ("e7_1", "e7_2")]
    assert retrieval_criteria
    assert all(c["observed"] is not None for c in retrieval_criteria), retrieval_criteria

    # WP12's one-group contrasts must produce numbers, not sentinels: §10.12 rows 1 and 3
    # are unreadable without them.
    contrast_rows = pd.read_csv(out / "patching_contrasts.csv").set_index("id")
    for contrast_id in ("e7_c7", "e7_c8k", "e7_c8v"):
        assert contrast_id in contrast_rows.index, contrast_id
        assert contrast_rows.loc[contrast_id, "status"] in ("ok", "no_data")

    matrix = json.loads((out / "interpretation_matrix.json").read_text(encoding="utf-8"))
    # With D7 resolved the matrix is live: it may select a row, or none of them. `04` §6.3
    # explicitly permits "matches none of them" — the six rows are pairwise disjoint but
    # NOT exhaustive — so both outcomes are legitimate and only a row id outside the six
    # would be a fault.
    valid_rows = {row["id"] for row in matrix["rows"]} if "rows" in matrix else {
        f"e7_i{n}" for n in range(1, 7)}
    assert matrix["matched_row"] is None or matrix["matched_row"] in valid_rows, \
        matrix["matched_row"]
    assert matrix["threshold_declaration_consistent"] is True, \
        matrix.get("threshold_declaration")
    assert matrix["observed"]["k0_effect"] is not None, \
        "the observed pattern is computed from real rows"
    # `08` §8.2/§11 — the four keys rows 1, 3, 4, 5 and 6 need must all be emitted.
    for key in ("r0_effect", "k0_unrelated_effect", "v0_unrelated_effect",
                "retrieval_top1_over_chance"):
        assert key in matrix["observed"], key
    assert matrix["observed"]["retrieval_top1_over_chance"] is not None, \
        "the retrieval frame must reach the matrix, or rows 5 and 6 can never evaluate"

    contrasts = pd.read_csv(out / "patching_contrasts.csv")
    sweep = pd.read_csv(out / "exploratory_layer_sweep.csv")
    assert "q_value" in contrasts.columns and "q_value" not in sweep.columns
    assert sweep["note"].str.startswith("EXPLORATORY").all()

    print(f"  [aggregate] {report['n_rows_used']}/{report['n_rows']} rows, "
          f"decision={report['decision']}, supported={report['supported']}, "
          f"pending={report['pending_preregistration']}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="e7_pipeline_smoke_",
                                     ignore_cleanup_errors=True) as tmp:
        root = Path(tmp)
        ckpt, tokenizer = _checkpoint(root)
        flores, xnli, controls = _manifests(tokenizer)
        config = _config(ckpt)

        handle = load_handle("qwen", str(ckpt), dtype="float32", device="cpu",
                             local_files_only=True)
        try:
            reps = _extract(config, flores, handle, root)
            retrieval = _retrieval(reps, flores, root)
            _fingerprints(config, flores, handle, root)
            results = _patching(config, xnli, controls, handle, root)
        finally:
            del handle
            gc.collect()

        _aggregate(results, retrieval, root)
        print("E7 pipeline offline smoke test passed")


if __name__ == "__main__":
    main()
