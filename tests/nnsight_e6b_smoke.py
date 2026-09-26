# -*- coding: utf-8 -*-
"""nnsight_e6b_smoke.py — the whole E6B pipeline, offline (WP6).

Mirrors ``nnsight_e6_eval_smoke.py`` for the E6B half:

1. train F1 (LoRA/clean) and F3 (LoRA/corrupt) plus F2 (full/clean) on a tiny random
   GPT-2 with a word-level tokenizer — no downloads;
2. assert the merge parity report, the corruption manifest and the run artefacts `03` §4.6
   names;
3. evaluate every checkpoint **against the untrained base**, which is what fills the three
   ``*_drift_from_base`` columns;
4. aggregate, and assert ``e6b_drift.csv``, ``e6b_early_warning.csv``, ``e6b_factorial.csv``
   and ``e6b_go_no_go.json`` carry real rows rather than the header-only placeholders they
   have held since WP10.

Step 4 is the point. Everything before it existed to make the E6B tables computable; this
is the proof that they compute. It asserts plumbing and invariants, never that a drift
takes a particular value (`06` §5).

Exit code is 0 on success; every phase asserts, so a regression fails loudly.
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
import evaluate_transformation as ev  # noqa: E402
import train_sentiment_adaptation as ts  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
CONDITIONS = ("f1", "f2", "f3")          # LoRA/clean, full/clean, LoRA/corrupt
SEED = 0
N_TRAIN = 32
N_VALID = 8


def _train(results: Path) -> dict:
    """Train the three conditions and check each run directory against `03` §4.6."""
    bases = {}
    for name in CONDITIONS:
        config = ts.load_config(CONFIG_DIR / f"e6b_{name}.yaml")
        setup = ts.prepare_run(config, seed=SEED, output_dir=results, smoke=True,
                               n_train=N_TRAIN, n_valid=N_VALID, epochs=1)
        ts.write_run_config(setup)
        summary = ts.train(setup, eval_every=1, progress=False)

        run_dir = setup.run_dir
        for artefact in ("run_config.json", "train_log.jsonl", "eval_log.jsonl",
                         "runtime_estimate.json", "train_summary.json"):
            assert (run_dir / artefact).exists(), f"{name}: missing {artefact}"

        config_json = json.loads((run_dir / "run_config.json").read_text("utf-8"))
        assert config_json["effective_batch"] == 32, config_json["effective_batch"]

        # `03` §4.6: the manifest exists for the corrupted conditions and only for them.
        manifest = run_dir / "corruption_manifest.csv"
        if config.get("corrupt_labels"):
            assert manifest.exists(), f"{name}: F3/F4 must write a corruption manifest"
            import pandas as pd
            frame = pd.read_csv(manifest)
            assert int(frame["flipped"].sum()) == round(0.20 * N_TRAIN)
        else:
            assert not manifest.exists(), f"{name}: a clean run must not write one"

        # Every eval row carries the three task metrics the evaluator reads back.
        rows = [json.loads(line) for line in
                (run_dir / "eval_log.jsonl").read_text("utf-8").splitlines() if line]
        assert rows and all(
            {"task_accuracy", "task_nll", "ece_10bin"} <= set(row) for row in rows)
        assert all(0.0 <= row["ece_10bin"] <= 1.0 for row in rows)

        steps = sorted(int(p.name.split("_")[1])
                       for p in (run_dir / "checkpoints").iterdir())
        assert 0 in steps, "step 0 must be checkpointed before the first update"

        if setup.adaptation == "lora":
            last = run_dir / "checkpoints" / f"step_{max(steps)}"
            parity = json.loads((last / "merge_parity.json").read_text("utf-8"))
            assert parity["passed"] is True, parity
            # The check runs in float64, NOT the fp32 `03` §4.5 names, and records the
            # deviation rather than hiding it (CLAUDE.md trap 15): §4.5's 1e-5 bar is
            # ABSOLUTE while distilgpt2's logits reach ~104, so meeting it needs ~1e-7
            # relative — below float32's own epsilon, i.e. unreachable by any correct
            # implementation. The fix was to raise the precision, never the tolerance.
            # This asserted the SPEC value rather than the one the code writes, so it
            # contradicted the shipped implementation.
            assert parity["check_dtype"] == "float64", parity
            assert parity["spec_check_dtype"] == "float32", (
                "the deviation from `03` §4.5 must stay recorded in the artefact")
            assert parity["check_device"] == "cpu", parity
            assert parity["tolerance"] == 1e-5, (
                "the tolerance must not move; only the working precision did")
            assert (last / "adapter").is_dir() and (last / "merged").is_dir()
            assert ev.fingerprint_dir(last).name == "merged"

        bases[config_json["condition_id"]] = config_json["base_model"]
        print(f"  [train] {config_json['condition_id']} ({setup.adaptation}) "
              f"steps={summary['steps']} acc={summary['final']['task_accuracy']:.3f} "
              f"corrupted={summary['n_corrupted']}")
        del setup
        gc.collect()
    return bases


def _evaluate(results: Path, bases: dict) -> None:
    """Fingerprint every checkpoint against the untrained base (the drift comparand)."""
    for condition, base in bases.items():
        run_dir = results / "e6b" / condition / f"seed{SEED}"
        summary = ev.evaluate_run(run_dir, smoke=True, progress=False, base=base)
        assert summary["n_failed"] == 0, summary
        assert summary["n_written"] > 0, summary
        assert summary["base"] == base
        assert not summary["base_note"], (
            f"{condition}: the base failed to load, so drift is empty: "
            f"{summary['base_note']}")

        import pandas as pd
        frame = pd.read_csv(run_dir / "checkpoint_metrics.csv")
        assert (frame["status"] == "ok").all()
        assert (frame["experiment_id"] == "e6b").all()
        # The columns WP6 exists to fill.
        for column in ("fingerprint_drift_from_base", "topology_drift_from_base",
                       "carrier_drift_from_base"):
            assert frame[column].notna().all(), f"{condition}: {column} is empty"
        for column in ("task_accuracy", "task_nll", "ece_10bin"):
            assert frame[column].notna().all(), (
                f"{condition}: {column} did not travel from eval_log.jsonl")

        # Step 0 is saved *before* the first optimiser update, so it IS the base and its
        # drift must be exactly zero. A later step must have moved. Together these catch
        # the two ways this column can be fictitious: a base that is really the checkpoint
        # (everything zero) and a base that is some other model (step 0 non-zero).
        # `fingerprint_drift` is 1 - cosine, and a vector's cosine with itself is 1 +/- a
        # few ulp, so step 0 lands at machine epsilon rather than at literal zero. The bar
        # is "indistinguishable from the base", not "== 0.0".
        by_step = frame.groupby("checkpoint_step")["fingerprint_drift_from_base"].max()
        assert float(by_step.loc[0]) < 1e-12, (
            f"{condition}: step 0 is the untrained base, so its drift must be at machine "
            f"epsilon, not {float(by_step.loc[0]):.3e} — the base handle is loading a "
            "different model")
        final = int(by_step.index.max())
        if final > 0:
            assert float(by_step.loc[final]) > 1e-12, (
                f"{condition}: drift at step {final} is {float(by_step.loc[final]):.3e}, "
                "indistinguishable from the base; the two are the same weights")
        print(f"  [eval] {condition}: {len(frame)} rows, "
              f"drift(step {final})={float(by_step.loc[final]):.3e}, "
              f"acc={frame['task_accuracy'].iloc[-1]:.3f}")


def _aggregate(results: Path) -> None:
    """The E6B tables must now carry rows, not a recorded reason."""
    import pandas as pd

    rows = pd.concat([pd.read_csv(p)
                      for p in sorted(results.rglob("checkpoint_metrics.csv"))])
    corpus = sorted(rows["corpus_id"].dropna().unique())[0]
    report = ag.aggregate(results, corpus=corpus, figures=True, progress=False)
    out = results / "aggregate"

    assert report["n_e6b_rows"] > 0, report
    for name in ("e6b_drift.csv", "e6b_early_warning.csv", "e6b_factorial.csv",
                 "table3_clean_vs_corrupt.csv", "e6b_go_no_go.json"):
        assert (out / name).exists(), name

    drift = pd.read_csv(out / "e6b_drift.csv")
    assert len(drift) > 0, "e6b_drift.csv is still header-only; WP6 changed nothing"
    assert drift["fingerprint_drift_from_base"].notna().any()

    early = pd.read_csv(out / "e6b_early_warning.csv")
    assert len(early) > 0, "the §9.8 onset table needs drift trajectories to exist"
    assert set(early["drift_column"]) <= {"fingerprint_drift_from_base",
                                          "topology_drift_from_base",
                                          "carrier_drift_from_base"}

    factorial = pd.read_csv(out / "e6b_factorial.csv")
    assert len(factorial) > 0
    assert set(factorial["adaptation"]) & {"lora", "full"}

    # Under `e6_prereg_v4` decision D4 is made, so the E6B conjunction is unresolved for a
    # DATA reason rather than a decision one — and the two must stay distinguishable.
    #   e6b_1 is a fixed 0.02 bar: resolvable from whatever seeds exist.
    #   e6b_2 calibrates on F1's across-seed spread and refuses fewer than three seeds.
    #   e6b_3 is the §9.8 early-warning rule and needs `min_seeds: 2`.
    # A one-seed smoke therefore leaves e6b_2 and e6b_3 unresolved and nothing pending.
    decision = json.loads((out / "e6b_go_no_go.json").read_text("utf-8"))
    assert decision["combine"] == "all", decision["combine"]
    assert decision["combine_error"] == ""
    assert decision["decision"] == "incomplete"
    assert not [c for c in decision["criteria"]
                if str(c.get("status", "")).startswith("PENDING")], (
        "v4 decided D4; a PENDING sentinel here means one was reintroduced")

    unresolved = {c["id"] for c in decision["criteria"] if c["met"] is None}
    assert {"e6b_2", "e6b_3"} <= unresolved, unresolved
    by_id = {c["id"]: c for c in decision["criteria"]}
    assert by_id["e6b_1"]["threshold"] == 0.02, by_id["e6b_1"]
    assert "at least three" in str(
        by_id["e6b_2"].get("threshold_rule", {}).get("reason", "")), by_id["e6b_2"]
    for criterion in decision["criteria"]:
        if criterion["met"] is None:
            assert criterion.get("reason") or criterion.get("threshold_rule"), (
                f"{criterion['id']} is unresolved without saying why")

    print(f"  [aggregate] e6b rows={report['n_e6b_rows']} drift={len(drift)} "
          f"onset={len(early)} factorial={len(factorial)} "
          f"decision={decision['decision']}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e6b_",
                                     ignore_cleanup_errors=True) as temp:
        results = Path(temp) / "results"
        bases = _train(results)
        _evaluate(results, bases)
        _aggregate(results)
        gc.collect()
    print("E6B offline smoke test passed")


if __name__ == "__main__":
    main()
