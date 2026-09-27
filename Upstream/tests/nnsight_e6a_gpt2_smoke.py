# -*- coding: utf-8 -*-
r"""nnsight_e6a_gpt2_smoke.py — the E6A-GPT2 arm, end to end, offline.

The sibling of ``nnsight_e6_eval_smoke.py`` for the second distillation arm. Same pipeline,
different instrument: a tiny random **GPT-2** teacher (12 layers) and student (6 layers)
instead of GPT-Neo's 4/8, a 12->6 layer map, and the arm's own experiment id, conditions,
corpus and pre-registration.

1. ``train_distillation.py --smoke`` for G0/G1/G2 at one seed;
2. ``evaluate_transformation.py --smoke`` on each run, with ΔCE;
3. ``aggregate_transformation.py --experiment e6a_gpt2`` against
   ``e6a_gpt2_preregistration.yaml``;
4. ``check_pilot_gate.py --experiment e6a_gpt2``, including the arm's fifth criterion.

What this program is *for*, beyond "it runs": the arm split is the thing most likely to be
wrong in a way nothing shouts about. Three conditions trained for five steps on random data
say nothing about distillation, and nothing here pretends they do (``06`` §5) — but they do
prove that the gpt2 arm writes to its own directories, is scored on its own corpus, reads
its own registration, and cannot be pooled with the TinyStories arm.

Run: ``./.venv/Scripts/python.exe tests/nnsight_e6a_gpt2_smoke.py``.

On Windows the process may still exit non-zero at TemporaryDirectory teardown (the known
file-lock flake); the runner treats a printed pass line as authoritative.
"""

from __future__ import annotations

import gc
import json
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402
import check_pilot_gate as gate  # noqa: E402
import evaluate_transformation as ev  # noqa: E402
import train_distillation as td  # noqa: E402

EXPERIMENT = "e6a_gpt2"
CONDITIONS = ("G0", "G1", "G2")
CONFIGS = {"G0": "e6a_gpt2_ce.yaml", "G1": "e6a_gpt2_logit_kd.yaml",
           "G2": "e6a_gpt2_logit_attention_kd.yaml"}
SEED = 0
STEPS = 5
CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
PREREG = CONFIG_DIR / "e6a_gpt2_preregistration.yaml"


def _train(results: Path) -> None:
    for condition, config_name in CONFIGS.items():
        config = td.load_config(CONFIG_DIR / config_name)
        assert config["condition"] == condition, (config["condition"], condition)
        assert config["experiment_id"] == EXPERIMENT
        setup = td.prepare_run(config, seed=SEED, output_dir=results,
                               max_steps=STEPS, smoke=True)

        # The arm writes to its OWN subtree. If this ever became results/e6a/... the two
        # arms would be pooled by every rglob in the project.
        assert setup.run_dir == results / EXPERIMENT / condition / f"seed{SEED}", \
            setup.run_dir
        # A GPT-2 student, not a GPT-Neo one, and the registered 12 -> 6 map.
        assert type(setup.student).__name__ == "GPT2LMHeadModel", type(setup.student)
        assert sorted(setup.layer_map.values()) == list(range(6)), setup.layer_map

        td.train(setup, resume="none")
        del setup
        gc.collect()
    print(f"  [train] {len(CONFIGS)} conditions x {STEPS} steps OK (GPT-2 12->6)")


def _check_run_config(results: Path, condition: str) -> None:
    payload = json.loads(
        (results / EXPERIMENT / condition / f"seed{SEED}" / "run_config.json")
        .read_text(encoding="utf-8"))
    assert payload["experiment_id"] == EXPERIMENT, payload["experiment_id"]
    assert payload["arch_key"] == "gpt2", (
        f"{condition}: arch_key {payload['arch_key']!r} would send a GPT-2 student at the "
        "GPT-Neo harness")
    assert payload["attn_reduction"] == td.ATTN_REDUCTION
    assert payload["run_id"].startswith(EXPERIMENT)
    # Every L_ATTN pair is global_global on GPT-2 — no local attention layers exist.
    types = payload["vocab_observed"]["student_attention_layers"]
    assert set(types) == {"global"}, types


def _evaluate(results: Path) -> None:
    for condition in CONDITIONS:
        run_dir = results / EXPERIMENT / condition / f"seed{SEED}"
        summary = ev.evaluate_run(run_dir, smoke=True, with_delta_ce=True,
                                  bootstrap_n=200, progress=False)
        assert summary["n_written"] > 0, (condition, summary)
        assert summary["n_failed"] == 0, (condition, summary)
        assert not summary["teacher_note"], (condition, summary["teacher_note"])
        _check_run_config(results, condition)

        # Resumability, same property the TinyStories smoke asserts.
        again = ev.evaluate_run(run_dir, smoke=True, with_delta_ce=True,
                                bootstrap_n=200, progress=False)
        assert again["n_written"] == 0, (condition, again)
        print(f"  [eval] {condition}: {summary['n_written']} units, "
              f"{again['n_skipped']} skipped on resume")


def _aggregate(results: Path) -> str:
    import pandas as pd

    rows = pd.concat([pd.read_csv(p)
                      for p in sorted(results.rglob("checkpoint_metrics.csv"))])
    assert set(rows["experiment_id"]) == {EXPERIMENT}, set(rows["experiment_id"])
    corpus = sorted(rows["corpus_id"].dropna().unique())[0]

    report = ag.aggregate(results, corpus=corpus, prereg_path=PREREG,
                          experiment=EXPERIMENT, reference_condition="G0",
                          figures=True, progress=False)
    assert report["experiment"] == EXPERIMENT, report["experiment"]
    assert report["preregistration_version"] == "e6a_gpt2_prereg_v1"
    assert report["n_e6a_rows"] > 0, report

    decision = json.loads(
        (results / "aggregate" / "go_no_go.json").read_text(encoding="utf-8"))
    assert not decision["pending_preregistration"], decision["pending_preregistration"]
    assert not decision["pending_decisions"], decision["pending_decisions"]
    # Same framing-(B) consequence as the TinyStories arm: one seed cannot resolve a
    # calibrated null, so the arm reads `incomplete` until G0 has three seeds.
    by_id = {c["id"]: c for c in decision["criteria"]}
    for criterion_id in ("e6a_gpt2_3", "e6a_gpt2_4"):
        assert by_id[criterion_id]["met"] is None, criterion_id
    assert decision["decision"] == "incomplete", decision["decision"]
    print(f"  [aggregate] corpus={corpus}, prereg={report['preregistration_version']}, "
          f"decision={decision['decision']}")
    return corpus


def _pilot_gate(results: Path, corpus: str) -> None:
    """Five criteria here, not four: this arm registers the teacher-relative 2b."""
    report = gate.evaluate_gate(results, seed=SEED, pilot_step=STEPS,
                                parity_report=None, corpus_id=corpus,
                                experiment_id=EXPERIMENT)
    assert report["experiment"] == EXPERIMENT
    assert report["conditions"] == list(CONDITIONS)
    by_id = {c["id"]: c for c in report["criteria"]}
    assert "e6a_pilot_2b" in by_id, sorted(by_id)
    assert len(report["criteria"]) == 5, sorted(by_id)

    for criterion_id in ("e6a_pilot_2", "e6a_pilot_3"):
        criterion = by_id[criterion_id]
        assert criterion["met"] is not None, (criterion_id, criterion.get("reason"))
        assert criterion["observed"], criterion

    # 2b reads the teacher's own fingerprint record, which the evaluator wrote per corpus.
    two_b = by_id["e6a_pilot_2b"]
    assert two_b["met"] is not None, two_b.get("reason")
    assert two_b["observed"]["teacher_sink"] > 0, two_b
    assert two_b["fraction_required"] == gate.TEACHER_SINK_FRACTION

    # No parity report, so the gate must still refuse to proceed.
    assert by_id["e6a_pilot_4"]["met"] is None
    assert report["proceed"] is False
    print(f"  [gate] 5 criteria, 2b teacher_sink="
          f"{two_b['observed']['teacher_sink']:.6f}, proceed={report['proceed']}")


def _isolation(results: Path) -> None:
    """The TinyStories arm must see none of this, and vice versa."""
    tinystories = ag.aggregate(results, figures=False, progress=False)  # default e6a
    assert tinystories["experiment"] == "e6a"
    assert tinystories["n_e6a_rows"] == 0, (
        "gpt2-arm rows leaked into the TinyStories aggregation; experiment_id isolation "
        "is not holding and every contrast, matched-loss reference and calibrated null "
        "would be computed across two different teachers")
    print("  [isolation] the default e6a aggregation sees 0 gpt2 rows")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e6a_gpt2_",
                                     ignore_cleanup_errors=True) as temp:
        results = Path(temp) / "results"
        _train(results)
        _evaluate(results)
        corpus = _aggregate(results)
        _pilot_gate(results, corpus)
        _isolation(results)
        gc.collect()
    print("E6A-GPT2 offline smoke test passed")


if __name__ == "__main__":
    main()
