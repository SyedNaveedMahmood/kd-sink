# -*- coding: utf-8 -*-
r"""Offline end-to-end smoke for the medium->small arm and equal-head bridge.

The synthetic GPT-2 models preserve the registered depths and head counts exactly
(24/16 -> 12/12 for the primary arm; 12/12 -> 6/12 for the bridge) while using
four-dimensional heads and 16-token blocks. No Hugging Face model or dataset is fetched.

Run::

    ./.venv/Scripts/python.exe tests/nnsight_e6a_gpt2_medium_small_smoke.py
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from copy import deepcopy
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402
import check_pilot_gate as gate  # noqa: E402
import evaluate_transformation as ev  # noqa: E402
import train_distillation as td  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
PRIMARY = "e6a_gpt2_medium_small"
BRIDGE = "e6a_gpt2_alignment_bridge"
PRIMARY_CONFIGS = {
    "G0": "e6a_gpt2_medium_small_ce.yaml",
    "G1": "e6a_gpt2_medium_small_logit_kd.yaml",
    "G2-aligned": "e6a_gpt2_medium_small_logit_attention_kd_aligned.yaml",
}
BRIDGE_CONFIGS = {
    "G2-legacy": "e6a_gpt2_alignment_bridge_logit_attention_kd_legacy.yaml",
    "G2-aligned": "e6a_gpt2_alignment_bridge_logit_attention_kd_aligned.yaml",
}
PREREG = CONFIG_DIR / "e6a_gpt2_medium_small_preregistration.yaml"
SEED = 0
STEPS = 2


def smoke_config(name: str):
    config = deepcopy(td.load_config(CONFIG_DIR / name))
    config["data"]["block_size"] = 16
    return config


def train_primary(results: Path) -> None:
    initial = {}
    for condition, name in PRIMARY_CONFIGS.items():
        setup = td.prepare_run(
            smoke_config(name), seed=SEED, output_dir=results,
            max_steps=STEPS, smoke=True)
        assert (td._num_layers(setup.teacher_config),
                td._num_heads(setup.teacher_config)) == (24, 16)
        assert (td._num_layers(setup.student_config),
                td._num_heads(setup.student_config)) == (12, 12)
        assert setup.head_alignment["method"] == "amad_jsd"
        initial[condition] = setup.initial_state_sha256
        td.train(setup, resume="none")
        del setup
        gc.collect()
    assert len(set(initial.values())) == 1, initial
    print("  [train primary] 3 conditions; identical initial state; 24/16 -> 12/12")


def train_bridge_pair(results: Path) -> None:
    initial = {}
    for condition, name in BRIDGE_CONFIGS.items():
        setup = td.prepare_run(
            smoke_config(name), seed=SEED, output_dir=results,
            max_steps=1, smoke=True)
        assert (td._num_heads(setup.teacher_config),
                td._num_heads(setup.student_config)) == (12, 12)
        initial[condition] = setup.initial_state_sha256
        td.train(setup, resume="none")
        payload = json.loads((setup.run_dir / "run_config.json").read_text(
            encoding="utf-8"))
        assert payload["head_alignment"]["method"] == (
            "index" if condition == "G2-legacy" else "amad_jsd")
        del setup
        gc.collect()
    assert len(set(initial.values())) == 1, initial
    print("  [train bridge] legacy/aligned G2 have identical initial state")


def evaluate_primary(results: Path) -> str:
    for condition in PRIMARY_CONFIGS:
        run_dir = results / PRIMARY / condition / f"seed{SEED}"
        summary = ev.evaluate_run(
            run_dir, smoke=True, with_delta_ce=True, n_sink_blocks=4,
            bootstrap_n=50, progress=False,
            # Keep the synthetic teacher's absolute temp path out of an already-deep
            # per-run cache root; otherwise Win32's legacy path limit can fire before the
            # evaluator reaches a fingerprint calculation.
            cache_dir=results.parent / "fp" / condition)
        assert summary["n_written"] > 0 and summary["n_failed"] == 0, summary
        assert not summary["teacher_note"], summary["teacher_note"]
    rows = pd.concat([
        pd.read_csv(path) for path in sorted((results / PRIMARY).rglob(
            "checkpoint_metrics.csv"))
    ])
    assert set(rows["experiment_id"]) == {PRIMARY}
    assert rows["carrier_jaccard_to_teacher"].isna().all()
    assert rows["warning"].fillna("").str.contains("equal head counts").all()
    corpus = str(rows["corpus_id"].dropna().iloc[0])
    print(f"  [evaluate] {len(rows)} rows; unequal-head carrier metric refused honestly")
    return corpus


def aggregate_and_gate(results: Path, corpus: str) -> None:
    report = ag.aggregate(
        results, corpus=corpus, prereg_path=PREREG, experiment=PRIMARY,
        reference_condition="G0", figures=False, progress=False)
    assert report["experiment"] == PRIMARY
    assert report["preregistration_version"] == "e6a_gpt2_medium_small_prereg_v1"
    decision = json.loads((results / "aggregate" / "go_no_go.json").read_text(
        encoding="utf-8"))
    assert decision["decision"] == "incomplete"  # one seed cannot resolve null spread

    pilot = gate.evaluate_gate(
        results, seed=SEED, pilot_step=STEPS, parity_report=None,
        corpus_id=corpus, experiment_id=PRIMARY)
    assert pilot["conditions"] == list(PRIMARY_CONFIGS)
    assert "e6a_pilot_2b" in {entry["id"] for entry in pilot["criteria"]}
    assert pilot["proceed"] is False  # parity report intentionally absent

    default = ag.aggregate(results, figures=False, progress=False)
    assert default["experiment"] == "e6a" and default["n_e6a_rows"] == 0
    print("  [aggregate/gate] own prereg loaded; original e6a sees zero rows")


def main() -> None:
    with tempfile.TemporaryDirectory(
            prefix="sinks_e6a_medium_small_", ignore_cleanup_errors=True) as temp:
        results = Path(temp) / "results"
        train_primary(results)
        train_bridge_pair(results)
        corpus = evaluate_primary(results)
        aggregate_and_gate(results, corpus)
        gc.collect()
    print("E6A GPT-2-medium -> GPT-2-small offline smoke test passed")


if __name__ == "__main__":
    main()
