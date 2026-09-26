"""Offline E6 evaluation smoke: train -> evaluate -> aggregate -> pilot gate (WP10).

Mirrors the E3/E4/E5/E6/E7 smoke programs: everything under a TemporaryDirectory, CPU,
fp32, no network. Unlike ``nnsight_e6_smoke.py`` (which exercises the tracing primitives)
this runs the whole E6 pipeline as an operator would:

1. ``train_distillation.py --smoke`` for **all three conditions** D0/D1/D2 at one seed,
   on random tiny GPT-Neo teacher (4 layers) and student (8 layers) — the exact layer pair
   that makes CLAUDE.md trap 1 reachable;
2. ``evaluate_transformation.py --smoke`` on each run, with ΔCE;
3. a second evaluation pass, asserting resumability recomputes nothing;
4. ``aggregate_transformation.py`` over the results tree;
5. ``check_pilot_gate.py``, asserting the criteria that had no inputs before WP10 now
   read real values.

This is the end-to-end proof that the pieces fit; it asserts plumbing and invariants, never
that a scientific quantity takes a particular value (``06`` §5). Three conditions trained
for five steps on random data say nothing about distillation and nothing here pretends they
do.

Run: ``./.venv/Scripts/python.exe tests/nnsight_e6_eval_smoke.py``.

On Windows the process may still exit non-zero at TemporaryDirectory teardown (the known
safetensors mmap-release flake shared with the e4/e5 smokes); all assertions run first.
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

CONDITIONS = ("D0", "D1", "D2")
CONFIGS = {"D0": "e6a_ce.yaml", "D1": "e6a_logit_kd.yaml",
           "D2": "e6a_logit_attention_kd.yaml"}
SEED = 0
STEPS = 5
CONFIG_DIR = REPO / "transformation_inheritance" / "configs"


def _train(results: Path) -> None:
    for condition, config_name in CONFIGS.items():
        config = td.load_config(CONFIG_DIR / config_name)
        assert config["condition"] == condition, (config["condition"], condition)
        setup = td.prepare_run(config, seed=SEED, output_dir=results,
                               max_steps=STEPS, smoke=True)
        td.train(setup, resume="none")
        del setup
        gc.collect()
    print(f"  [train] {len(CONFIGS)} conditions x {STEPS} steps OK")


def _evaluate(results: Path) -> None:
    for condition in CONDITIONS:
        run_dir = results / "e6a" / condition / f"seed{SEED}"

        first = ev.evaluate_run(run_dir, smoke=True, with_delta_ce=True,
                                bootstrap_n=200, progress=False)
        assert first["n_written"] > 0, (condition, first)
        assert first["n_failed"] == 0, (condition, first)
        assert not first["teacher_note"], (
            f"{condition}: the smoke teacher must load, otherwise the whole "
            f"teacher-comparison path goes untested — {first['teacher_note']}")

        # Resumability: the second pass must skip every unit, not merely rewrite it.
        second = ev.evaluate_run(run_dir, smoke=True, with_delta_ce=True,
                                 bootstrap_n=200, progress=False)
        assert second["n_written"] == 0, (condition, second)
        assert second["n_skipped"] == first["n_written"], (condition, second)

        _check_metrics(run_dir, condition)
        print(f"  [eval] {condition}: {first['n_written']} units, "
              f"{second['n_skipped']} skipped on resume")


def _check_metrics(run_dir: Path, condition: str) -> None:
    import pandas as pd

    frame = pd.read_csv(run_dir / "checkpoint_metrics.csv")
    missing = [c for c in ev.METRIC_COLUMNS if c not in frame.columns]
    assert not missing, f"{condition}: checkpoint_metrics.csv is missing {missing}"
    assert (frame["status"] == "ok").all(), frame["status"].tolist()
    assert frame["condition"].eq(condition).all()
    assert not frame.duplicated(subset=["run_id", "checkpoint_step",
                                        "corpus_id"]).any()

    # Every row is a real measurement against a real teacher.
    assert frame["n_keys_used"].gt(0).all()
    assert frame["fingerprint_cosine_to_teacher"].notna().all()
    assert frame["carrier_jaccard_to_teacher"].notna().all()
    assert frame["baseline_sink"].notna().all()
    assert frame["git_sha"].notna().all()

    # The band came from depth_band, not the legacy compute_band: an 8-layer student
    # yields [2,8), where compute_band(8,"scaled") would give (3,7) (CLAUDE.md trap 1).
    assert frame["layer_band"].eq("[2,8)").all(), frame["layer_band"].unique()
    assert frame["band_version"].eq("depth_band_v1").all()

    # JSON payloads are parseable and carry the fingerprint's key set explicitly.
    row = frame.iloc[0]
    fingerprint = json.loads(row["fingerprint_json"])
    assert "int_a" in fingerprint
    assert len(json.loads(row["depth_profile_16_json"])) == 16
    ci = json.loads(row["delta_ce_ci_json"])
    assert ci["uncertainty_kind"] == "sampling_over_corpus_items"


def _aggregate(results: Path) -> None:
    import pandas as pd

    # The shipped v3 pre-registration names the real corpora (tinystories_validation_sink_300
    # / e1_100x40); a smoke run builds synthetic ones, so the sink corpus it actually
    # measured is passed as the operator override `--corpus`. Discovered from the rows
    # rather than hard-coded, so a change to `build_corpora` cannot silently empty this.
    rows = pd.concat([pd.read_csv(p)
                      for p in sorted(results.rglob("checkpoint_metrics.csv"))])
    corpus = sorted(rows["corpus_id"].dropna().unique())[0]

    report = ag.aggregate(results, corpus=corpus, figures=True, progress=False)
    assert report["n_rows_used"] > 0, report
    assert report["n_e6a_rows"] > 0, report

    out = results / "aggregate"
    for name in ("e6a_contrasts.csv", "e6a_matched_loss.csv",
                 "table2_inheritance_components.csv", "e6b_drift.csv",
                 "e6b_early_warning.csv", "e6b_factorial.csv",
                 "table3_clean_vs_corrupt.csv", "go_no_go.json",
                 "e6b_go_no_go.json", "aggregate_summary.json"):
        assert (out / name).exists(), name

    contrasts = pd.read_csv(out / "e6a_contrasts.csv")
    # With all three conditions trained, the derivable contrasts have a value at one seed.
    computed = contrasts[contrasts["status"] == "ok"]
    assert len(computed) >= 3, contrasts[["id", "status"]].to_dict("records")
    assert computed["n_seeds"].eq(1).all()
    assert computed["min_attainable_p"].eq(1.0).all(), (
        "one seed can only attain p = 1.0; a smaller value would be a bug in the "
        "permutation test's bookkeeping")
    # e6a_c4a/c4b need P8M rows, which a smoke run does not produce; they must report
    # `no_data` with a reason rather than a number.
    for contrast_id in ("e6a_c4a", "e6a_c4b"):
        row = contrasts[contrasts["id"] == contrast_id]
        assert len(row) == 1 and row.iloc[0]["status"] == "no_data", contrast_id
        assert pd.isna(row.iloc[0]["mean_diff"])

    decision = json.loads((out / "go_no_go.json").read_text(encoding="utf-8"))
    # `e6_prereg_v4` resolves D1-D4, so nothing is undecided any more. The smoke is still
    # `incomplete`, and the CHANGE OF REASON is the point worth asserting: it is no longer
    # "a threshold was never chosen" but "the chosen rule cannot resolve on one seed".
    assert not decision["pending_preregistration"], (
        f"v4 decided every criterion; {decision['pending_preregistration']} suggests a "
        "PENDING sentinel was reintroduced")
    assert not decision["pending_decisions"], decision["pending_decisions"]
    assert decision["preregistration_version"] == "e6_prereg_v4", (
        decision["preregistration_version"])

    # D2/D3 took framing (B): the threshold is calibrated against D0's across-seed spread,
    # and `resolve_threshold_rule` refuses a spread from fewer than three seeds. A smoke
    # run has one seed, so e6a_3/e6a_4 must report met: null WITH THE CALIBRATION'S OWN
    # REASON — an unresolvable rule and an unmade decision must not look alike.
    by_id = {c["id"]: c for c in decision["criteria"]}
    for criterion_id in ("e6a_3", "e6a_4"):
        criterion = by_id[criterion_id]
        assert criterion["met"] is None, criterion_id
        limbs = criterion.get("sub_criteria") or [criterion]
        reasons = [str(limb.get("threshold_rule", {}).get("reason", "")) for limb in limbs]
        assert any("at least three" in reason for reason in reasons), (
            f"{criterion_id} must say the calibration was refused for want of seeds, "
            f"not merely that it had no threshold; got {reasons}")
        for limb in limbs:
            assert limb.get("threshold") is None, (
                f"{criterion_id}: an unresolved rule must not yield a number")

    assert decision["decision"] == "incomplete", decision["decision"]
    assert decision["n_unknown"] >= 2, decision["n_unknown"]

    # E6B has no runs yet, so its own conjunction is unresolved rather than negative.
    e6b = json.loads((out / "e6b_go_no_go.json").read_text(encoding="utf-8"))
    assert e6b["combine"] == "all" and e6b["decision"] == "incomplete"

    print(f"  [aggregate] corpus={corpus}, {len(computed)} contrasts computed, "
          f"{len(decision['pending_preregistration'])} pending criteria, "
          f"{len(decision['pending_decisions'])} open decisions, "
          f"decision={decision['decision']}")
    return corpus


def _pilot_gate(results: Path, corpus: str) -> None:
    """The gate's criteria 2 and 3 could not be evaluated at all before WP10.

    ``corpus`` is passed for the same reason ``--corpus`` is passed to the aggregator: the
    gate now names the corpus it scores on (the pre-registered sink corpus), and a smoke run
    builds a synthetic one. Without it criteria 2 and 3 correctly report ``met: null``.
    """
    report = gate.evaluate_gate(results, seed=SEED, pilot_step=STEPS,
                                parity_report=None, corpus_id=corpus)
    assert report["corpus_id"] == corpus, report["corpus_id"]
    by_id = {c["id"]: c for c in report["criteria"]}

    for criterion_id in ("e6a_pilot_2", "e6a_pilot_3"):
        criterion = by_id[criterion_id]
        assert criterion["met"] is not None, (
            f"{criterion_id} is still unevaluable after WP10 wrote checkpoint_metrics.csv"
            f": {criterion.get('reason')}")
        assert criterion["observed"], criterion

    # Criterion 4 has no parity report here, so the gate must still refuse to proceed —
    # a gate that passed on three of four criteria would be the worst possible outcome.
    assert by_id["e6a_pilot_4"]["met"] is None
    assert report["proceed"] is False
    print(f"  [gate] criteria 2/3 now evaluable "
          f"(sink={by_id['e6a_pilot_2']['observed']}), proceed={report['proceed']}")


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="sinks_e6_eval_",
                                     ignore_cleanup_errors=True) as temp:
        results = Path(temp) / "results"
        _train(results)
        _evaluate(results)
        corpus = _aggregate(results)
        _pilot_gate(results, corpus)
        gc.collect()
    print("E6 evaluation offline smoke test passed")


if __name__ == "__main__":
    main()
