# -*- coding: utf-8 -*-
"""test_pilot_gate.py — the E6A pilot gate reports honestly (WP5).

``03_MODULE_SPEC_e6_transformation.md`` §2. This does not test the *science* — the gate's
thresholds are pre-registered and whether a real run clears them is the experiment. It
tests the three properties that would let a bad pilot look like a good one:

* missing inputs report ``met: null`` with a reason, never a fabricated value, and
  ``proceed`` stays false (CLAUDE.md rule 4);
* criterion 2 failing prints the design's prescribed remedy — extend seed 0 to 5,000 steps
  — rather than reading as a refuted hypothesis;
* ``proceed`` is true only when **all four** criteria are met;
* **the criteria read the rows they say they read.** ``checkpoint_metrics.csv`` carries one
  row per ``(checkpoint_step, corpus_id)``, and criteria 2 and 3 used to name neither: 2 took
  the maximum over both corpora and every step (returning the *step-0 cross-domain* value on
  the first real pilot) and 3 took ``.iloc[0]``. The fixtures below are therefore written in
  the **producer's** shape — every step, both corpora — because the previous one-row-no-corpus
  fixture is exactly why a green suite did not see it (CLAUDE.md traps 16 and 23).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "transformation_inheritance"))

import check_pilot_gate as gate  # noqa: E402

PILOT_STEP = 2000

#: The pre-registered sink corpus and the cross-domain one the evaluator also writes. Both
#: appear in every fixture, because both appear in every real ``checkpoint_metrics.csv``.
PRIMARY = gate.PRIMARY_CORPUS_ID
CROSS_DOMAIN = "e1_100x40"

#: Every step the pilot evaluates, so "by step 2000" has a trajectory to be read out of.
EVAL_STEPS = (0, 250, PILOT_STEP)


def _write_eval_log(root: Path, condition: str, seed: int, ce_start, ce_end):
    path = gate.run_dir(root, condition, seed) / "eval_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"step": 0, "validation_ce": ce_start},
            {"step": PILOT_STEP, "validation_ce": ce_end}]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


def _write_metrics(root: Path, condition: str, seed: int, *, sink, mechanistic=None,
                   cross_sink=0.001, cross_mechanistic=None, sink_at_step0=0.001,
                   sink_early=0.001, cross_first=False, duplicate_primary=False,
                   omit_corpus_column=False, steps=EVAL_STEPS):
    """``checkpoint_metrics.csv`` in the shape ``evaluate_transformation.py`` writes it.

    One row per ``(checkpoint_step, corpus_id)``. ``sink`` is the primary-corpus value **at
    the pilot step**; ``sink_at_step0`` is the pre-training measurement, which is a property
    of the initialisation and must never satisfy a criterion about emergence.
    """
    import pandas as pd

    path = gate.run_dir(root, condition, seed) / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)

    def _row(step, corpus, sink_value, mech):
        row = {"checkpoint_step": step, "corpus_id": corpus, "baseline_sink": sink_value}
        for column in gate.MECHANISTIC_COLUMNS:
            row[column] = (mech or {}).get(column, 0.0)
        return row

    rows = []
    for step in steps:
        primary_sink = (sink_at_step0 if step == 0
                        else sink if step == PILOT_STEP else sink_early)
        primary = _row(step, PRIMARY, primary_sink, mechanistic)
        cross = _row(step, CROSS_DOMAIN, cross_sink, cross_mechanistic or mechanistic)
        rows.extend([cross, primary] if cross_first else [primary, cross])
        if duplicate_primary and step == PILOT_STEP:
            rows.append(dict(primary))
    frame = pd.DataFrame(rows)
    if omit_corpus_column:
        frame = frame.drop(columns=["corpus_id"])
    frame.to_csv(path, index=False)


def _parity(root: Path, passed: bool) -> Path:
    path = root / "parity.json"
    path.write_text(json.dumps({"passed": passed, "max_abs_deviation": 3e-7,
                                "n_examples": 5, "atol": 1e-5, "rtol": 1e-4}),
                    encoding="utf-8")
    return path


def _parity_frozen_shape(root: Path, all_rows_pass: bool, *, name="parity_report.json",
                         n_sentences=gate.PARITY_EXAMPLES_REQUIRED):
    """A parity report in the shape the FROZEN ``run_parity_check`` actually writes.

    ``_parity`` above writes ``{"passed": ...}`` — the shape the reader happened to look
    for. That is why a real defect survived: the test built its own input in the code's
    shape instead of the producer's, so ``criterion_4`` returning ``met: False`` on a
    passing report was invisible. This helper mirrors ``verify_parity``'s report:
    ``all_rows_pass``, ``n_sentences`` and one row per intervention.
    """
    interventions = ("int_a", "int_c", "int_d", "int_f", "int_g", "int_h", "int_i", "int_j")
    rows = [{"intervention": key,
             "max_abs_attention_difference": 4.1e-3,
             "max_abs_metric_deviation": 2.7e-7,
             "max_rel_metric_deviation": 5.0e-7,
             "status": "pass"} for key in interventions]
    if not all_rows_pass:
        rows[3] = {**rows[3], "max_abs_metric_deviation": 3.3e-2, "status": "fail"}
    path = root / name
    payload = {"reference": "manual", "atol": 1e-5, "rtol": 1e-4,
               "rows": rows, "all_rows_pass": all_rows_pass}
    if n_sentences is not None:
        payload["n_sentences"] = n_sentences
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_criterion_4_reads_the_frozen_drivers_report_shape(tmp_path):
    """A PASSING report from ``run_parity_check`` must read as met, not as failed.

    The regression this pins: ``criterion_4`` looked for ``passed`` and then
    ``all_within_tolerance``, neither of which that driver emits, so it fell through to
    ``bool(None)`` and reported ``met: False`` on a clean parity run. The gate could
    therefore never reach ``proceed: true`` and design §8.5 forbids launching Phase 2
    without it.
    """
    ok = gate.criterion_4(_parity_frozen_shape(tmp_path, True))
    assert ok["met"] is True, ok
    assert ok["observed"]["n_rows"] == 8
    assert ok["observed"]["max_abs_metric_deviation"] == pytest.approx(2.7e-7)
    assert "failed_interventions" not in ok["observed"]

    bad = gate.criterion_4(_parity_frozen_shape(tmp_path, False, name="bad.json"))
    assert bad["met"] is False, bad
    assert bad["observed"]["failed_interventions"] == ["int_f"]
    assert bad["observed"]["max_abs_metric_deviation"] == pytest.approx(3.3e-2)


def test_the_verdict_key_matches_the_frozen_driver():
    """The key this gate reads must be the key the frozen engine writes.

    Asserted against the frozen source rather than against a fixture, so drift on either
    side fails here instead of silently producing a vacuous verdict. ``nnsight_engine.py``
    is additive-only, so the key cannot be renamed out from under this.
    """
    engine_src = (REPO / "common" / "nnsight_engine.py").read_text(encoding="utf-8")
    assert '"all_rows_pass"' in engine_src, (
        "the frozen parity driver no longer writes 'all_rows_pass'; check_pilot_gate's "
        "criterion_4 reads that key and must be updated together with it")
    gate_src = (REPO / "transformation_inheritance"
                / "check_pilot_gate.py").read_text(encoding="utf-8")
    assert '"all_rows_pass"' in gate_src


def test_an_unrecognised_report_shape_is_refused_not_failed(tmp_path):
    """No recognised verdict key ⇒ ``met: null`` with a reason — never a defaulted False.

    A missing verdict is not a failed one (CLAUDE.md rule 4). Defaulting it to False is
    precisely how the original defect turned an absent key into a scientific-looking
    negative.
    """
    path = tmp_path / "alien.json"
    path.write_text(json.dumps({"verdict": "fine", "rows": []}), encoding="utf-8")
    out = gate.criterion_4(path)
    assert out["met"] is None
    assert "all_rows_pass" in out["reason"]
    assert out["observed"] == {}


def test_missing_inputs_are_unknown_not_failed_and_not_fabricated(tmp_path):
    report = gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP,
                                parity_report=None)
    assert report["n_unknown"] == 4
    assert report["n_met"] == 0
    assert report["proceed"] is False
    assert report["decision"] == "incomplete"
    for criterion in report["criteria"]:
        assert criterion["met"] is None
        assert criterion["reason"]
        assert criterion["observed"] in ({}, [])


def test_all_four_criteria_met_gives_proceed(tmp_path):
    for condition in gate.CONDITIONS:
        _write_eval_log(tmp_path, condition, 0, ce_start=4.0, ce_end=2.5)
    _write_metrics(tmp_path, "D0", 0, sink=0.05)
    _write_metrics(tmp_path, "D1", 0, sink=0.10)
    _write_metrics(tmp_path, "D2", 0, sink=0.30,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.42})

    report = gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP,
                                parity_report=_parity(tmp_path, True))
    assert [c["met"] for c in report["criteria"]] == [True, True, True, True]
    assert report["n_met"] == 4
    assert report["proceed"] is True
    assert report["decision"] == "continue"
    assert report["git_sha"]


def test_criterion_2_failure_prints_the_prescribed_remedy(tmp_path):
    for condition in gate.CONDITIONS:
        _write_eval_log(tmp_path, condition, 0, ce_start=4.0, ce_end=2.5)
        _write_metrics(tmp_path, condition, 0, sink=0.02)

    report = gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP,
                                parity_report=_parity(tmp_path, True))
    criterion = next(c for c in report["criteria"] if c["id"] == "e6a_pilot_2")
    assert criterion["met"] is False
    assert "5,000 steps" in criterion["remedy"]
    assert "not evidence that it will not emerge" in criterion["remedy"]
    assert report["proceed"] is False


def test_a_condition_whose_ce_rose_fails_criterion_1(tmp_path):
    _write_eval_log(tmp_path, "D0", 0, ce_start=4.0, ce_end=2.5)
    _write_eval_log(tmp_path, "D1", 0, ce_start=4.0, ce_end=2.5)
    _write_eval_log(tmp_path, "D2", 0, ce_start=4.0, ce_end=4.7)   # got worse

    criterion = gate.criterion_1(tmp_path, seed=0, pilot_step=PILOT_STEP)
    assert criterion["met"] is False
    assert criterion["observed"]["D2"]["reduced"] is False
    assert criterion["observed"]["D0"]["reduced"] is True


def test_criterion_3_uses_the_pre_registered_threshold(tmp_path):
    _write_metrics(tmp_path, "D0", 0, sink=0.2,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.50})
    _write_metrics(tmp_path, "D2", 0, sink=0.2,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.59})
    assert gate.criterion_3(tmp_path, 0, PILOT_STEP)["met"] is False   # delta 0.09

    _write_metrics(tmp_path, "D2", 0, sink=0.2,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.61})
    criterion = gate.criterion_3(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is True                                     # delta 0.11
    assert criterion["threshold"] == gate.MECHANISTIC_DELTA_THRESHOLD


def test_criterion_2_uses_the_primary_corpus_only(tmp_path):
    """A large cross-domain sink must not satisfy a criterion scored in-domain.

    The regression: ``criterion_2`` filtered on ``checkpoint_step`` alone and took
    ``.max()``, so the largest value anywhere in the file won — including the cross-domain
    corpus, whose sink runs an order of magnitude higher on this model family.
    """
    for condition in gate.CONDITIONS:
        _write_metrics(tmp_path, condition, 0, sink=0.01, cross_sink=0.90)

    criterion = gate.criterion_2(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is False, criterion
    assert criterion["corpus_id"] == PRIMARY
    for values in criterion["observed"].values():
        assert values["max_through_pilot_step"] == pytest.approx(0.01)
        assert values["at_pilot_step"] == pytest.approx(0.01)


def test_criterion_2_ignores_the_step_0_measurement(tmp_path):
    """Step 0 is saved *before* the first optimiser update, so it is not an outcome.

    A criterion about a sink emerging during training must not be satisfiable by the random
    initialisation — and the more permissive reading stays visible in the artefact.
    """
    for condition in gate.CONDITIONS:
        _write_metrics(tmp_path, condition, 0, sink=0.01, sink_at_step0=0.90)

    criterion = gate.criterion_2(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is False, criterion
    assert criterion["step_selection"] == "max over trained checkpoints (step > 0)"
    observed = criterion["observed"]["D0"]
    assert observed["max_through_pilot_step"] == pytest.approx(0.01)
    assert observed["max_including_step_0"] == pytest.approx(0.90)
    assert observed["argmax_step"] > 0


def test_criterion_2_reports_every_reading_of_by_step_2000(tmp_path):
    """All three readings are recorded, so the wording is auditable from the artefact."""
    _write_metrics(tmp_path, "D0", 0, sink=0.02, sink_early=0.05, sink_at_step0=0.07)
    observed = gate.criterion_2(tmp_path, 0, PILOT_STEP)["observed"]["D0"]
    assert observed == {"at_pilot_step": pytest.approx(0.02),
                        "max_through_pilot_step": pytest.approx(0.05),
                        "argmax_step": 250,
                        "max_including_step_0": pytest.approx(0.07)}


def test_criterion_2_refuses_a_file_without_a_corpus_column(tmp_path):
    """`05` §2 makes ``corpus_id`` mandatory; without it the criterion cannot name its row."""
    _write_metrics(tmp_path, "D0", 0, sink=0.9, omit_corpus_column=True)
    criterion = gate.criterion_2(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is None
    assert "corpus_id" in criterion["reason"]


def test_criterion_3_follows_the_primary_corpus_not_row_order(tmp_path):
    """``.iloc[0]`` read whichever corpus the writer ordered first.

    Here the cross-domain row comes first and carries a difference above the threshold while
    the in-domain row is below it. The verdict must come from the corpus the criterion is
    registered on, so row order cannot decide it.
    """
    _write_metrics(tmp_path, "D0", 0, sink=0.01, cross_first=True,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.50},
                   cross_mechanistic={"fingerprint_cosine_to_teacher": 0.10})
    _write_metrics(tmp_path, "D2", 0, sink=0.01, cross_first=True,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.55},
                   cross_mechanistic={"fingerprint_cosine_to_teacher": 0.90})

    criterion = gate.criterion_3(tmp_path, 0, PILOT_STEP)
    assert criterion["corpus_id"] == PRIMARY
    assert criterion["observed"]["fingerprint_cosine_to_teacher"] == pytest.approx(0.05)
    assert criterion["met"] is False, (
        "the cross-domain difference (0.80) decided the verdict — the corpus filter is "
        "not being applied")


def test_criterion_3_refuses_duplicate_rows_for_one_corpus_and_step(tmp_path):
    """Two rows for one ``(corpus, step)`` means duplicated runs; picking one is arbitrary."""
    _write_metrics(tmp_path, "D0", 0, sink=0.01, duplicate_primary=True)
    _write_metrics(tmp_path, "D2", 0, sink=0.01)
    criterion = gate.criterion_3(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is None
    assert "ambiguous" in criterion["reason"]


def test_criterion_3_records_the_denominator_it_normalised_by(tmp_path):
    """Every fingerprint entry is a ratio against ``baseline_sink``; a reader needs it.

    Diagnostic only — the criterion's threshold is pre-registered and unchanged.
    """
    _write_metrics(tmp_path, "D0", 0, sink=0.0028,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.50})
    _write_metrics(tmp_path, "D2", 0, sink=0.0019,
                   mechanistic={"fingerprint_cosine_to_teacher": 0.70})
    criterion = gate.criterion_3(tmp_path, 0, PILOT_STEP)
    assert criterion["met"] is True
    assert criterion["baseline_sink_denominator"] == {"D0": pytest.approx(0.0028),
                                                     "D2": pytest.approx(0.0019)}


def test_criterion_4_requires_the_registered_number_of_examples(tmp_path):
    """"on 5 examples" is part of the criterion, so the count is checked, not assumed.

    The frozen ``PARITY_SENTENCES`` default is three and the frozen CLI cannot be given more,
    so a report produced by it alone does not satisfy the registered wording. Its ``rows``
    list holds one row per *intervention*, which is why ten rows from three sentences read
    like a large sample and passed.
    """
    short = gate.criterion_4(_parity_frozen_shape(tmp_path, True, name="short.json",
                                                  n_sentences=3))
    assert short["met"] is None, short
    assert "3 example(s)" in short["reason"]
    assert "run_pilot_parity" in short["remedy"]
    assert short["observed"]["all_rows_pass_on_short_sample"] is True
    assert short["examples_required"] == 5

    enough = gate.criterion_4(_parity_frozen_shape(tmp_path, True, name="five.json",
                                                   n_sentences=5))
    assert enough["met"] is True, enough
    assert enough["observed"]["n_sentences"] == 5


def test_criterion_4_refuses_a_report_with_no_example_count(tmp_path):
    """An absent count is an unrecognised shape, not zero and not five."""
    criterion = gate.criterion_4(_parity_frozen_shape(tmp_path, True, name="nocount.json",
                                                      n_sentences=None))
    assert criterion["met"] is None
    assert "n_sentences" in criterion["reason"]
    assert "rows' counts interventions" in criterion["reason"]


def test_the_gate_scores_the_corpus_the_preregistration_names():
    """The gate's corpus must be the one every e6a criterion in the YAML is scored on.

    Read from the shipped pre-registration rather than duplicated as a literal, so the two
    cannot drift apart (the discipline CLAUDE.md trap 11 asks for).
    """
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((REPO / "transformation_inheritance" / "configs"
                           / "e6_preregistration.yaml").read_text(encoding="utf-8"))
    named = set()
    for entry in spec.get("contrasts", []) or []:
        if str(entry.get("family", "")).startswith("e6a") and entry.get("corpus_id"):
            named.add(str(entry["corpus_id"]))
    assert named, "no e6a contrast in the pre-registration names a corpus_id"
    assert named == {gate.PRIMARY_CORPUS_ID}, (
        f"the pre-registration scores e6a on {sorted(named)} but the pilot gate uses "
        f"{gate.PRIMARY_CORPUS_ID!r}")


def test_the_corpus_is_selectable_but_the_thresholds_are_not():
    """``--corpus`` exists for smoke runs; it is a row selector, never a threshold."""
    parser = gate.build_parser()
    defaults = {action.dest: action.default for action in parser._actions}
    # ``None`` means "the corpus this arm's pre-registration names", resolved in
    # ``evaluate_gate`` from ``ARMS`` — not a literal that could drift per arm.
    assert defaults["corpus"] is None
    assert defaults["experiment"] == gate.EXPERIMENT_ID
    assert gate.ARMS[gate.EXPERIMENT_ID]["corpus_id"] == gate.PRIMARY_CORPUS_ID


# ── the second arm: e6a_gpt2 ────────────────────────────────────────────────────


def _teacher_fingerprint(root: Path, condition: str, seed: int, corpus: str,
                         baseline_sink: float, *, experiment="e6a_gpt2",
                         legacy_layout=False):
    """A teacher fingerprint record in the shape ``fingerprint_runner`` actually writes.

    Producer's shape (trap 16), including the per-corpus directory level the trap-24 fix
    added — which is the only reason criterion 2b has anything to read.
    """
    base = (gate.run_dir(root, condition, seed, experiment) / "fingerprints"
            / "teacher_gpt2" / "step_na")
    path = base / "fingerprint.json" if legacy_layout else base / corpus / "fingerprint.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "_cache_key": {"model_name": "gpt2", "manifest_sha256": "deadbeef"},
        "record": {"model_name": "gpt2", "corpus_id": corpus,
                   "baseline_sink": baseline_sink, "num_layers": 12,
                   "frac_cells_above_0_2": 0.7},
    }), encoding="utf-8")
    return path


def _gpt2_metrics(root: Path, condition: str, seed: int, *, sink, corpus=None):
    import pandas as pd

    corpus = corpus or gate.ARMS["e6a_gpt2"]["corpus_id"]
    path = (gate.run_dir(root, condition, seed, "e6a_gpt2") / "checkpoint_metrics.csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for step in EVAL_STEPS:
        value = sink if step == PILOT_STEP else 0.001
        for corpus_id, sink_value in ((corpus, value), (CROSS_DOMAIN, 0.02)):
            row = {"checkpoint_step": step, "corpus_id": corpus_id,
                   "baseline_sink": sink_value}
            for column in gate.MECHANISTIC_COLUMNS:
                row[column] = 0.0
            rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_the_gpt2_arm_reads_its_own_directories_and_corpus(tmp_path):
    for condition in ("G0", "G1", "G2"):
        _write_eval_log(tmp_path, condition, 0, ce_start=4.0, ce_end=2.5)
        _gpt2_metrics(tmp_path, condition, 0, sink=0.20)
    _teacher_fingerprint(tmp_path, "G0", 0, gate.ARMS["e6a_gpt2"]["corpus_id"], 0.30)

    # `_write_eval_log` wrote to results/e6a/...; the gpt2 arm must not see it.
    report = gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP,
                                parity_report=None, experiment_id="e6a_gpt2")
    assert report["experiment"] == "e6a_gpt2"
    assert report["conditions"] == ["G0", "G1", "G2"]
    assert report["corpus_id"] == "openwebtext_validation_sink_300"

    by_id = {c["id"]: c for c in report["criteria"]}
    assert "e6a_pilot_2b" in by_id, "the gpt2 arm registers a fifth criterion"
    assert by_id["e6a_pilot_2"]["met"] is True          # 0.20 > 0.15
    assert by_id["e6a_pilot_2b"]["met"] is True         # 0.20 >= 0.50 * 0.30
    assert len(report["criteria"]) == 5


def test_criterion_2b_is_not_emitted_for_the_tinystories_arm(tmp_path):
    """Adding it there would amend a pre-registration that already produced a verdict."""
    report = gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP,
                                parity_report=None)
    assert [c["id"] for c in report["criteria"]] == [
        "e6a_pilot_1", "e6a_pilot_2", "e6a_pilot_3", "e6a_pilot_4"]


def test_criterion_2b_fails_when_the_student_holds_too_little_of_the_teachers_sink(tmp_path):
    """The criterion the TinyStories arm lacked: relative to what the teacher has."""
    for condition in ("G0", "G1", "G2"):
        _gpt2_metrics(tmp_path, condition, 0, sink=0.10)
    _teacher_fingerprint(tmp_path, "G0", 0, gate.ARMS["e6a_gpt2"]["corpus_id"], 0.40)

    criterion = gate.criterion_2b(tmp_path, 0, PILOT_STEP,
                                  gate.ARMS["e6a_gpt2"]["corpus_id"],
                                  ("G0", "G1", "G2"), "e6a_gpt2")
    assert criterion["met"] is False, criterion
    assert criterion["observed"]["teacher_sink"] == pytest.approx(0.40)
    assert criterion["observed"]["required"] == pytest.approx(0.20)
    assert criterion["observed"]["G1"]["fraction_of_teacher"] == pytest.approx(0.25)
    assert "remedy" in criterion


def test_criterion_2b_is_unknown_when_the_teacher_was_never_fingerprinted(tmp_path):
    """A missing input is `met: null` with a reason, never a defaulted False (trap 16)."""
    for condition in ("G0", "G1", "G2"):
        _gpt2_metrics(tmp_path, condition, 0, sink=0.10)

    criterion = gate.criterion_2b(tmp_path, 0, PILOT_STEP,
                                  gate.ARMS["e6a_gpt2"]["corpus_id"],
                                  ("G0", "G1", "G2"), "e6a_gpt2")
    assert criterion["met"] is None
    assert "teacher" in criterion["reason"]


def test_criterion_2b_refuses_a_pre_trap_24_teacher_record(tmp_path):
    """A record with no corpus in its path cannot be shown to be this corpus's measurement."""
    for condition in ("G0", "G1", "G2"):
        _gpt2_metrics(tmp_path, condition, 0, sink=0.10)
    for condition in ("G0", "G1", "G2"):
        _teacher_fingerprint(tmp_path, condition, 0,
                             gate.ARMS["e6a_gpt2"]["corpus_id"], 0.40, legacy_layout=True)

    criterion = gate.criterion_2b(tmp_path, 0, PILOT_STEP,
                                  gate.ARMS["e6a_gpt2"]["corpus_id"],
                                  ("G0", "G1", "G2"), "e6a_gpt2")
    assert criterion["met"] is None
    assert "trap-24" in criterion["reason"] or "corpus in" in criterion["reason"]


def test_criterion_2b_ignores_the_step_0_measurement(tmp_path):
    """Same rule as criterion 2: the initialisation is not a training outcome."""
    import pandas as pd

    corpus = gate.ARMS["e6a_gpt2"]["corpus_id"]
    path = gate.run_dir(tmp_path, "G0", 0, "e6a_gpt2") / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = [{"checkpoint_step": 0, "corpus_id": corpus, "baseline_sink": 0.90},
            {"checkpoint_step": PILOT_STEP, "corpus_id": corpus, "baseline_sink": 0.01}]
    for row in rows:
        for column in gate.MECHANISTIC_COLUMNS:
            row[column] = 0.0
    pd.DataFrame(rows).to_csv(path, index=False)
    _teacher_fingerprint(tmp_path, "G0", 0, corpus, 0.40)

    criterion = gate.criterion_2b(tmp_path, 0, PILOT_STEP, corpus, ("G0",), "e6a_gpt2")
    assert criterion["met"] is False
    assert criterion["observed"]["G0"]["max_through_pilot_step"] == pytest.approx(0.01)


def test_an_unknown_arm_is_refused(tmp_path):
    with pytest.raises(SystemExit, match="unknown experiment"):
        gate.evaluate_gate(tmp_path, seed=0, pilot_step=PILOT_STEP, parity_report=None,
                           experiment_id="e6a_llama")


def test_the_gpt2_arms_corpus_matches_its_own_preregistration():
    """Same drift guard as the TinyStories arm, against the second registration file."""
    yaml = pytest.importorskip("yaml")
    spec = yaml.safe_load((REPO / "transformation_inheritance" / "configs"
                           / "e6a_gpt2_preregistration.yaml").read_text(encoding="utf-8"))
    named = {str(e["corpus_id"]) for e in spec.get("contrasts", [])
             if str(e.get("family", "")).startswith("e6a") and e.get("corpus_id")}
    assert named == {gate.ARMS["e6a_gpt2"]["corpus_id"]}
    assert spec["pilot_gate"]["corpus_id"] == gate.ARMS["e6a_gpt2"]["corpus_id"]


def test_thresholds_are_constants_not_arguments():
    """Pre-registration (``05`` §6): a threshold must not be tunable from the CLI."""
    parser = gate.build_parser()
    options = {action.dest for action in parser._actions}
    assert "sink_threshold" not in options
    assert "mechanistic_delta_threshold" not in options
    assert gate.SINK_THRESHOLD == 0.15
    assert gate.MECHANISTIC_DELTA_THRESHOLD == 0.10


def test_cli_exit_code_tracks_proceed(tmp_path, capsys):
    code = gate.main(["--results", str(tmp_path), "--seed", "0",
                      "--pilot-step", str(PILOT_STEP),
                      "--out", str(tmp_path / "pilot_gate.json")])
    assert code == 1
    payload = json.loads((tmp_path / "pilot_gate.json").read_text(encoding="utf-8"))
    assert payload["proceed"] is False
    assert "????" in capsys.readouterr().out
