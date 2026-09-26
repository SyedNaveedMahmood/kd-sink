# -*- coding: utf-8 -*-
"""check_pilot_gate.py — the E6A pilot gate (WP5).

``03_MODULE_SPEC_e6_transformation.md`` §2 / design §8.5. After the seed-0 pilot
(``--max-steps 2000`` for D0/D1/D2) and ``evaluate_transformation.py`` at steps
0/250/500/1000/2000, this evaluates four criteria and writes ``pilot_gate.json`` with an
explicit ``proceed: true|false``.

    python transformation_inheritance/check_pilot_gate.py \\
      --results transformation_inheritance/results --seed 0 --pilot-step 2000

The criteria are transcribed from the design **before** any results exist — that is the
pre-registration (``05`` §6). Changing a threshold after seeing results requires a recorded
amendment with a timestamp and a reason, so the thresholds are constants here rather than
CLI arguments.

1. all conditions reduce TinyStories validation CE versus step 0;
2. at least one condition has baseline sink > 0.15 by the pilot step;
3. at least one mechanistic metric differs between D0 and D2 by >= 0.10;
4. manual vs NNsight attention agree within the Neo tolerance on 5 examples.

On criterion-2 failure the script prints the design's prescribed remedy — extend seed 0 to
5,000 steps — rather than concluding failure. A pilot that has not yet grown a sink is not
the same result as a pilot that cannot.

**Never fabricate.** Criteria whose inputs are absent are reported ``met: null`` with a
``reason``, and ``proceed`` is false. This script reads artefacts; it does not estimate.

Row selection is part of the criterion (CLAUDE.md trap 23)
----------------------------------------------------------
``checkpoint_metrics.csv`` carries one row per ``(checkpoint_step, corpus_id)``, so a
criterion that does not name both reads whichever row the file happens to order first.
Criteria 2 and 3 used to do exactly that: criterion 2 took ``baseline_sink.max()`` over
*every* row at or before the pilot step, which on the first real pilot returned the
**step-0 cross-domain** value — a random-initialisation number standing in for a trained
in-domain one — and criterion 3 took ``.iloc[0]``. Both now select
:data:`PRIMARY_CORPUS_ID` explicitly and refuse an ambiguous selection. No threshold moved.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import provenance as prov  # noqa: E402

EXPERIMENT_ID = "e6a"
CONDITIONS = ("D0", "D1", "D2")

#: Every arm of E6A: its conditions and the corpus its criteria are scored on. A second arm
#: exists because the TinyStories teacher has no sink on its own in-domain corpus
#: (`baseline_sink` 0.009309 against a 0.15 bar — CLAUDE.md trap 26), so `e6a_gpt2` distils
#: `gpt2`, whose sink is 16.5x the uniform floor on `e1_100x40`. The arms are kept apart by
#: `experiment_id` because both aggregators rglob the results tree without de-duplicating.
ARMS: Dict[str, Dict[str, Any]] = {
    "e6a": {"conditions": ("D0", "D1", "D2"),
            "corpus_id": "tinystories_validation_sink_300",
            "teacher_relative_gate": False},
    "e6a_gpt2": {"conditions": ("G0", "G1", "G2"),
                 "corpus_id": "openwebtext_validation_sink_300",
                 "validation_label": "OpenWebText validation",
                 "teacher_relative_gate": True,
                 "teacher_sink_fraction": 0.50},
    "e6a_gpt2_large_medium": {
        "conditions": ("G0", "G1", "G2"),
        "corpus_id": "openwebtext_validation_sink_300",
        "validation_label": "OpenWebText validation",
        "teacher_relative_gate": True,
        "teacher_sink_fraction": 0.50,
    },
    "e6a_gpt2_medium_small": {
        "conditions": ("G0", "G1", "G2-aligned"),
        "corpus_id": "openwebtext_validation_sink_300",
        "validation_label": "OpenWebText validation",
        "teacher_relative_gate": True,
        "teacher_sink_fraction": 0.50,
    },
    "e6a_gpt2_alignment_bridge": {
        "conditions": ("G0", "G1", "G2-legacy", "G2-aligned"),
        "corpus_id": "openwebtext_validation_sink_300",
        "validation_label": "OpenWebText validation",
        "teacher_relative_gate": True,
        "teacher_sink_fraction": 0.50,
    },
}

# --- pre-registered thresholds (design §8.5). Do not tune. ---
SINK_THRESHOLD = 0.15
MECHANISTIC_DELTA_THRESHOLD = 0.10
#: The Neo parity tolerance, imported rather than restated where available.
DEFAULT_PARITY_ATOL = 1e-5
#: Criterion 4's own text says "on 5 examples" (design §8.5 / ``03`` §2). The frozen
#: ``PARITY_SENTENCES`` default is three, so a report produced by the frozen CLI alone does
#: not satisfy the registered wording — and the count must be *checked*, not assumed, because
#: the report's ``rows`` list is per intervention and reads like a large sample when it is not.
PARITY_EXAMPLES_REQUIRED = 5

#: The corpus every criterion is scored on. Design §8.5 criterion 1 is the *TinyStories
#: validation* measurement and every e6a entry in ``configs/e6_preregistration.yaml`` names
#: this same corpus id; ``tests/test_pilot_gate.py`` asserts the two agree so this constant
#: cannot drift from the pre-registration. ``--corpus`` overrides it for a smoke run, which
#: builds a synthetic corpus under a different id.
PRIMARY_CORPUS_ID = "tinystories_validation_sink_300"

#: Criterion 2b (arms whose specification opts in): the fraction of the **teacher's own**
#: in-domain sink a student must reach. First registered for ``e6a_gpt2`` in
#: `configs/e6a_gpt2_preregistration.yaml` on 2026-08-01, before any gpt2 run existed, and
#: inherited prospectively by the isolated scale/alignment extensions. The teacher's value
#: is a property of the instrument — it is measurable without training a single student —
#: so fixing a fraction then was not fitting to a result. It is the criterion the
#: TinyStories arm lacked: 0.15 was 16.1x that teacher's own value, so no amount of
#: inheritance could ever have met it (CLAUDE.md trap 26).
TEACHER_SINK_FRACTION = 0.50


def arm(experiment_id: str) -> Dict[str, Any]:
    """The registered conditions and corpus for ``experiment_id``, or a refusal.

    An unknown arm is refused rather than defaulted: silently scoring a new arm on the
    TinyStories corpus would produce `no_data` everywhere and look like a missing run.
    """
    spec = ARMS.get(str(experiment_id))
    if spec is None:
        raise SystemExit(
            f"unknown experiment {experiment_id!r}; known arms are {sorted(ARMS)}. Add one "
            "to check_pilot_gate.ARMS together with its pre-registration file.")
    return spec

#: Mechanistic metrics compared between D0 and D2 for criterion 3.
MECHANISTIC_COLUMNS = (
    "fingerprint_cosine_to_teacher",
    "fingerprint_spearman_to_teacher",
    "fingerprint_l1_to_teacher",
    "category_agreement_to_teacher",
    "topology_wasserstein_to_teacher",
    "carrier_jaccard_to_teacher",
)


def run_dir(results: Path, condition: str, seed: int,
            experiment_id: str = EXPERIMENT_ID) -> Path:
    return Path(results) / experiment_id / condition / f"seed{seed}"


def read_eval_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def read_checkpoint_metrics(path: Path):
    """``checkpoint_metrics.csv`` from ``evaluate_transformation.py`` (WP10), if present."""
    if not path.exists():
        return None
    import pandas as pd
    return pd.read_csv(path)


def _at_step(rows: List[Dict[str, Any]], step: int) -> Optional[Dict[str, Any]]:
    for row in rows:
        if int(row.get("step", -1)) == step:
            return row
    return None


def criterion_1(results: Path, seed: int, pilot_step: int,
                conditions: Sequence[str] = CONDITIONS,
                experiment_id: str = EXPERIMENT_ID,
                validation_label: str = "TinyStories validation") -> Dict[str, Any]:
    """All conditions reduce validation CE versus step 0."""
    text = f"all conditions reduce {validation_label} CE vs step 0"
    per_condition: Dict[str, Any] = {}
    missing: List[str] = []
    for condition in conditions:
        rows = read_eval_log(
            run_dir(results, condition, seed, experiment_id) / "eval_log.jsonl")
        start = _at_step(rows, 0)
        end = _at_step(rows, pilot_step)
        if start is None or end is None:
            missing.append(condition)
            continue
        per_condition[condition] = {
            "ce_step0": start.get("validation_ce"),
            "ce_pilot": end.get("validation_ce"),
            "reduced": bool(end.get("validation_ce", float("inf"))
                            < start.get("validation_ce", float("-inf"))),
        }
    if missing:
        return {"id": "e6a_pilot_1",
                "text": text,
                "observed": per_condition, "met": None,
                "reason": f"no eval_log rows at step 0 and {pilot_step} for {missing}"}
    return {"id": "e6a_pilot_1",
            "text": text,
            "observed": per_condition,
            "met": all(v["reduced"] for v in per_condition.values())}


def select_corpus_rows(frame, corpus_id: str):
    """``(rows, None)`` for ``corpus_id``, or ``(None, reason)`` when that is not well defined.

    A criterion that reads ``checkpoint_metrics.csv`` without naming a corpus reads whichever
    row the writer happened to order first. ``corpus_id`` is mandatory in ``05`` §2, so its
    absence is a malformed input — reported as a reason, never defaulted to "the first one".
    """
    if "corpus_id" not in getattr(frame, "columns", ()):
        return None, ("checkpoint_metrics.csv carries no corpus_id column, which `05` §2 "
                      "makes mandatory; without it this criterion cannot state which "
                      "corpus it was measured on")
    rows = frame[frame["corpus_id"].astype(str) == str(corpus_id)]
    if not len(rows):
        present = sorted(set(frame["corpus_id"].astype(str)))
        return None, (f"no rows for corpus {corpus_id!r} (the file carries {present}); "
                      "pass --corpus if this run used a different corpus id")
    return rows, None


def criterion_2(results: Path, seed: int, pilot_step: int,
                corpus_id: str = PRIMARY_CORPUS_ID,
                conditions: Sequence[str] = CONDITIONS,
                experiment_id: str = EXPERIMENT_ID) -> Dict[str, Any]:
    """At least one condition has baseline sink > 0.15 by the pilot step.

    Three readings of "by the pilot step" are reported per condition so the wording is
    auditable rather than resolved silently:

    * ``at_pilot_step`` — the value exactly at ``pilot_step``;
    * ``max_through_pilot_step`` — the largest value over *trained* checkpoints, i.e. step
      > 0. **This is the verdict.** Step 0 is saved before the first optimiser update
      (``03`` §1.6), so it is not a training outcome, and a criterion about emergence must
      not be satisfiable by a randomly-initialised model;
    * ``max_including_step_0`` — the same maximum with step 0 included, recorded so the
      more permissive reading stays checkable from the artefact alone.
    """
    text = f"at least one condition has baseline sink > {SINK_THRESHOLD}"
    observed: Dict[str, Any] = {}
    unavailable: Dict[str, str] = {}
    for condition in conditions:
        frame = read_checkpoint_metrics(
            run_dir(results, condition, seed, experiment_id) / "checkpoint_metrics.csv")
        if frame is None:
            continue
        rows, reason = select_corpus_rows(frame, corpus_id)
        if rows is None:
            unavailable[condition] = reason
            continue
        rows = rows[rows["checkpoint_step"] <= pilot_step]
        trained = rows[rows["checkpoint_step"] > 0]
        if not len(trained):
            unavailable[condition] = (
                f"no trained checkpoint (step > 0) at or before step {pilot_step} on corpus "
                f"{corpus_id!r}; step 0 alone measures the initialisation, not training")
            continue
        peak = trained.loc[trained["baseline_sink"].idxmax()]
        at_step = rows[rows["checkpoint_step"] == pilot_step]
        observed[condition] = {
            "at_pilot_step": (float(at_step.iloc[0]["baseline_sink"])
                              if len(at_step) else None),
            "max_through_pilot_step": float(peak["baseline_sink"]),
            "argmax_step": int(peak["checkpoint_step"]),
            "max_including_step_0": float(rows["baseline_sink"].max()),
        }
    common = {"id": "e6a_pilot_2", "text": text, "threshold": SINK_THRESHOLD,
              "corpus_id": corpus_id,
              "step_selection": "max over trained checkpoints (step > 0)"}
    if not observed:
        return {**common, "observed": {}, "met": None,
                "unavailable": unavailable,
                "reason": (next(iter(unavailable.values()), None)
                           or "no checkpoint_metrics.csv; run evaluate_transformation.py "
                              "(WP10) on the pilot checkpoints first")}
    met = any(v["max_through_pilot_step"] > SINK_THRESHOLD for v in observed.values())
    out = {**common, "observed": observed, "met": met}
    if unavailable:
        out["unavailable"] = unavailable
    if not met:
        out["remedy"] = (
            "Design §8.5: extend seed 0 to 5,000 steps and re-check. A sink that has not "
            "emerged by step 2,000 is not evidence that it will not emerge; do not read "
            "this as a failed hypothesis.")
    return out


def criterion_3(results: Path, seed: int, pilot_step: int,
                corpus_id: str = PRIMARY_CORPUS_ID,
                conditions: Sequence[str] = CONDITIONS,
                experiment_id: str = EXPERIMENT_ID) -> Dict[str, Any]:
    """At least one mechanistic metric differs between D0 and D2 by >= 0.10.

    The pilot-step row is selected by ``(corpus_id, checkpoint_step)`` and an ambiguous
    selection is refused: two rows for one corpus at one step means the results tree holds
    duplicated runs, and picking either would be arbitrary.
    """
    frames = {}
    reasons: List[str] = []
    # The CE-only control and the attention-KD arm: first and last of the arm's conditions,
    # never the literal "D0"/"D2", so a second arm compares G0 against G2.
    control, attention_kd = conditions[0], conditions[-1]
    text = (f"at least one mechanistic metric differs between {control} and "
            f"{attention_kd} by >= {MECHANISTIC_DELTA_THRESHOLD}")
    common = {"id": "e6a_pilot_3", "text": text,
              "threshold": MECHANISTIC_DELTA_THRESHOLD, "corpus_id": corpus_id}
    for condition in (control, attention_kd):
        frame = read_checkpoint_metrics(
            run_dir(results, condition, seed, experiment_id) / "checkpoint_metrics.csv")
        if frame is None:
            reasons.append(f"{condition}: no checkpoint_metrics.csv")
            continue
        rows, reason = select_corpus_rows(frame, corpus_id)
        if rows is None:
            reasons.append(f"{condition}: {reason}")
            continue
        rows = rows[rows["checkpoint_step"] == pilot_step]
        if len(rows) == 1:
            frames[condition] = rows.iloc[0]
        elif len(rows) == 0:
            reasons.append(f"{condition}: no row at step {pilot_step} on corpus "
                           f"{corpus_id!r}")
        else:
            reasons.append(f"{condition}: {len(rows)} rows at step {pilot_step} on corpus "
                           f"{corpus_id!r} — the selection is ambiguous, so no value is "
                           "read; de-duplicate the results tree (`05` §7.1)")
    if len(frames) < 2:
        return {**common, "observed": {}, "met": None, "reason": "; ".join(reasons)}

    deltas: Dict[str, Any] = {}
    for column in MECHANISTIC_COLUMNS:
        if column in frames[control] and column in frames[attention_kd]:
            a, b = frames[control][column], frames[attention_kd][column]
            try:
                deltas[column] = abs(float(b) - float(a))
            except (TypeError, ValueError):
                continue
    if not deltas:
        return {**common, "observed": {}, "met": None,
                "reason": "no comparable mechanistic columns present"}
    out = {**common, "observed": deltas,
           "met": any(v >= MECHANISTIC_DELTA_THRESHOLD for v in deltas.values())}
    # Diagnostic, not a verdict: every fingerprint entry is a ratio against `baseline_sink`,
    # so a reader has to see the denominator these differences were normalised by. Reporting
    # it changes no threshold -- making the criterion conditional on it would be a post-hoc
    # amendment to a pre-registration (CLAUDE.md rule 4).
    sinks = {}
    for condition, row in frames.items():
        try:
            sinks[condition] = float(row["baseline_sink"])
        except (KeyError, TypeError, ValueError):
            sinks[condition] = None
    out["baseline_sink_denominator"] = sinks
    if reasons:
        out["notes"] = reasons
    return out


def read_teacher_sink(run_directory: Path, corpus_id: str) -> Tuple[Optional[float], str]:
    """The teacher's own ``baseline_sink`` on ``corpus_id``, from its fingerprint record.

    ``(value, note)``; ``value`` is ``None`` when it cannot be read, and ``note`` says why.

    The record lives at ``fingerprints/teacher_*/step_na/<corpus_id>/fingerprint.json``.
    That per-corpus directory level exists only because of the trap-24 fix — before it, the
    evaluator's sweep over corpora overwrote the teacher's in-domain record with its
    cross-domain one, and this criterion would have had nothing to read (or, worse, would
    have read the wrong corpus and not known).
    """
    roots = sorted((Path(run_directory) / "fingerprints").glob("teacher_*"))
    if not roots:
        return None, ("no teacher fingerprint under "
                      f"{Path(run_directory) / 'fingerprints'}; run "
                      "evaluate_transformation.py with a teacher first")
    candidates = [p for root in roots
                  for p in root.glob(f"step_na/{corpus_id}/fingerprint.json")]
    if not candidates:
        legacy = [p for root in roots for p in root.glob("step_na/fingerprint.json")]
        if legacy:
            return None, (
                "the teacher's fingerprint is in the pre-trap-24 layout with no corpus in "
                f"the path, so it cannot be shown to be the {corpus_id!r} measurement; "
                "re-run evaluate_transformation.py --force to regenerate it per corpus")
        return None, f"no teacher fingerprint for corpus {corpus_id!r}"
    if len(candidates) > 1:
        return None, (f"{len(candidates)} teacher fingerprints for corpus {corpus_id!r}; "
                      "the selection is ambiguous, so no value is read")
    record = json.loads(candidates[0].read_text(encoding="utf-8")).get("record") or {}
    if record.get("corpus_id") != corpus_id:
        return None, (f"teacher fingerprint at {candidates[0].name} records corpus "
                      f"{record.get('corpus_id')!r}, not {corpus_id!r}")
    value = record.get("baseline_sink")
    if value is None:
        return None, "teacher fingerprint carries no baseline_sink"
    return float(value), str(candidates[0])


def criterion_2b(results: Path, seed: int, pilot_step: int,
                 corpus_id: str = PRIMARY_CORPUS_ID,
                 conditions: Sequence[str] = CONDITIONS,
                 experiment_id: str = EXPERIMENT_ID,
                 fraction: float = TEACHER_SINK_FRACTION) -> Dict[str, Any]:
    """At least one condition reaches ``fraction`` of the **teacher's own** sink.

    First registered for ``e6a_gpt2`` on 2026-08-01, before any gpt2 run existed, then
    inherited by prospectively isolated arms through their explicit arm specifications.

    Criterion 2's 0.15 is an absolute bar transcribed from design §8.5; it says nothing
    about whether the teacher in front of it has a sink at all. On the TinyStories arm it
    turned out to be 16.1× that teacher's own in-domain value, so it was unreachable by
    inheritance and its failure carried no information about inheritance. This criterion is
    the relative reading: *did the student acquire a serious fraction of what the teacher
    actually has*. The two are reported side by side and both must be met.

    Same row selection as criterion 2 — the arm's corpus, trained checkpoints only.
    """
    text = (f"at least one condition reaches {fraction:.0%} of the teacher's own "
            "baseline sink")
    common = {"id": "e6a_pilot_2b", "text": text, "corpus_id": corpus_id,
              "fraction_required": fraction,
              "step_selection": "max over trained checkpoints (step > 0)"}

    teacher_sink = None
    teacher_note = ""
    for condition in conditions:
        teacher_sink, teacher_note = read_teacher_sink(
            run_dir(results, condition, seed, experiment_id), corpus_id)
        if teacher_sink is not None:
            break
    if teacher_sink is None:
        return {**common, "observed": {}, "met": None,
                "reason": f"the teacher's own sink could not be read: {teacher_note}"}
    if teacher_sink <= 0:
        return {**common, "observed": {"teacher_sink": teacher_sink}, "met": None,
                "reason": ("the teacher's own sink is non-positive, so a fraction of it is "
                           "not a threshold; report the teacher measurement instead")}

    required = fraction * teacher_sink
    observed: Dict[str, Any] = {}
    for condition in conditions:
        frame = read_checkpoint_metrics(
            run_dir(results, condition, seed, experiment_id) / "checkpoint_metrics.csv")
        if frame is None:
            continue
        rows, _reason = select_corpus_rows(frame, corpus_id)
        if rows is None:
            continue
        trained = rows[(rows["checkpoint_step"] <= pilot_step)
                       & (rows["checkpoint_step"] > 0)]
        if not len(trained):
            continue
        peak = float(trained["baseline_sink"].max())
        at_step = rows[rows["checkpoint_step"] == pilot_step]
        observed[condition] = {
            "max_through_pilot_step": peak,
            "fraction_of_teacher": peak / teacher_sink,
            "at_pilot_step": (float(at_step.iloc[0]["baseline_sink"])
                              if len(at_step) else None),
        }
    if not observed:
        return {**common, "observed": {"teacher_sink": teacher_sink}, "met": None,
                "reason": (f"no trained checkpoint on corpus {corpus_id!r}; run "
                           "evaluate_transformation.py on the pilot checkpoints first")}

    out = {**common,
           "observed": {"teacher_sink": teacher_sink, "required": required, **observed},
           "teacher_fingerprint": teacher_note,
           "met": any(v["max_through_pilot_step"] >= required for v in observed.values())}
    if not out["met"]:
        out["remedy"] = (
            "The students hold a small fraction of a sink the teacher does have — unlike "
            "the TinyStories arm, where the teacher had none. Extending training is a "
            "meaningful next step here; check criterion 2 and the carrier metrics together "
            "before concluding.")
    return out


def criterion_4(parity_report: Optional[Path]) -> Dict[str, Any]:
    """Manual vs NNsight attention agree within the Neo tolerance on 5 examples.

    Reads a parity report produced by the frozen ``run_parity_check`` rather than
    recomputing it here: the parity harness is the authority on what "agree" means, and a
    second implementation of the comparison would be exactly the kind of drift this gate
    exists to detect.

    The **example count is part of the criterion** and is therefore checked. The report's
    ``rows`` list is one row per *intervention*, so ten rows from three sentences reads like
    a larger sample than it is; ``n_sentences`` is the number of examples, and a report
    carrying fewer than :data:`PARITY_EXAMPLES_REQUIRED` leaves the criterion ``met: null``
    with a remedy rather than passing on evidence the design did not ask for.
    """
    if parity_report is None or not Path(parity_report).exists():
        return {"id": "e6a_pilot_4",
                "text": "manual vs NNsight attention agree within the Neo tolerance "
                        "on 5 examples",
                "observed": {}, "met": None,
                "reason": "no parity report supplied; pass --parity-report pointing at "
                          "the output of the frozen run_parity_check on the pilot "
                          "checkpoint"}
    payload = json.loads(Path(parity_report).read_text(encoding="utf-8"))

    # `run_parity_check` writes `all_rows_pass` (nnsight_engine.py, verify_parity's report).
    # Reading only `passed`/`all_within_tolerance` -- neither of which that driver emits --
    # made this criterion report met=False on a PASSING report, so the gate could never
    # reach proceed=true. The verdict now comes from the key the frozen driver actually
    # writes; the other two are kept as fallbacks for a hand-written or future report.
    verdict_keys = ("all_rows_pass", "passed", "all_within_tolerance")
    passed = next((payload[k] for k in verdict_keys if payload.get(k) is not None), None)
    if passed is None:
        return {"id": "e6a_pilot_4",
                "text": "manual vs NNsight attention agree within the Neo tolerance "
                        "on 5 examples",
                "observed": {}, "met": None,
                "reason": (f"parity report {Path(parity_report).name!r} carries none of "
                           f"{verdict_keys}; refusing to read a verdict out of a report "
                           "shape this gate does not recognise")}

    # The per-intervention rows are the evidence; summarise them rather than looking for
    # scalar keys the driver does not write.
    rows = payload.get("rows") or []
    observed = {k: payload[k] for k in ("atol", "rtol") if k in payload}
    observed["n_rows"] = len(rows)
    deviations = [row[key] for row in rows
                  for key in ("max_abs_metric_deviation",)
                  if isinstance(row, dict) and row.get(key) is not None]
    if deviations:
        observed["max_abs_metric_deviation"] = max(float(d) for d in deviations)
    failed = [row.get("intervention") for row in rows
              if isinstance(row, dict) and row.get("status") not in (None, "pass")]
    if failed:
        observed["failed_interventions"] = failed
    for key in ("max_abs_deviation", "n_examples", "n_sentences", "reference"):
        if key in payload:
            observed[key] = payload[key]

    out = {"id": "e6a_pilot_4",
           "text": "manual vs NNsight attention agree within the Neo tolerance "
                   "on 5 examples",
           "observed": observed,
           "examples_required": PARITY_EXAMPLES_REQUIRED,
           "met": bool(passed)}

    # `n_sentences` is what `verify_parity` writes; `n_examples` is accepted as a synonym for
    # a hand-written report. An absent count is not read as zero -- it is an unrecognised
    # report shape, and the gate says so instead of guessing (trap 16).
    n_examples = next((payload[k] for k in ("n_sentences", "n_examples")
                       if isinstance(payload.get(k), int)), None)
    if n_examples is None:
        out["met"] = None
        out["reason"] = ("parity report carries neither 'n_sentences' nor 'n_examples', so "
                         "the number of examples compared cannot be verified; note that "
                         "'rows' counts interventions, not examples")
    elif n_examples < PARITY_EXAMPLES_REQUIRED:
        out["met"] = None
        out["reason"] = (f"parity ran on {n_examples} example(s); design §8.5 criterion 4 "
                         f"asks for {PARITY_EXAMPLES_REQUIRED}. The comparison that ran "
                         f"{'passed' if passed else 'FAILED'}, but on fewer examples than "
                         "the criterion registers, so the verdict is unknown rather than "
                         "true")
        out["remedy"] = ("run transformation_inheritance/run_pilot_parity.py, which supplies "
                         f"{PARITY_EXAMPLES_REQUIRED} sentences; the frozen harness's own "
                         "PARITY_SENTENCES default is three")
        out["observed"]["all_rows_pass_on_short_sample"] = bool(passed)
    return out


def evaluate_gate(results: Path, seed: int, pilot_step: int,
                  parity_report: Optional[Path],
                  corpus_id: Optional[str] = None,
                  experiment_id: str = EXPERIMENT_ID,
                  conditions: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    spec = arm(experiment_id)
    conditions = tuple(conditions) if conditions else tuple(spec["conditions"])
    corpus_id = corpus_id or str(spec["corpus_id"])
    criteria = [
        criterion_1(results, seed, pilot_step, conditions, experiment_id,
                    str(spec.get("validation_label", "TinyStories validation"))),
        criterion_2(results, seed, pilot_step, corpus_id, conditions, experiment_id),
        criterion_3(results, seed, pilot_step, corpus_id, conditions, experiment_id),
        criterion_4(parity_report),
    ]
    # Criterion 2b belongs only to arms that explicitly register it. Emitting it for `e6a`
    # would amend an experiment that has already produced a verdict.
    teacher_sink_fraction = spec.get("teacher_sink_fraction")
    if teacher_sink_fraction is not None:
        criteria.insert(2, criterion_2b(results, seed, pilot_step, corpus_id,
                                        conditions, experiment_id,
                                        float(teacher_sink_fraction)))
    n_met = sum(1 for c in criteria if c["met"] is True)
    n_unknown = sum(1 for c in criteria if c["met"] is None)
    proceed = n_met == len(criteria)
    return {
        "experiment": experiment_id,
        "gate": "pilot",
        "seed": seed,
        "pilot_step": pilot_step,
        "corpus_id": corpus_id,
        "conditions": list(conditions),
        "criteria": criteria,
        "n_met": n_met,
        "n_unknown": n_unknown,
        "proceed": proceed,
        "decision": ("continue" if proceed else
                     ("incomplete" if n_unknown else "hold")),
        "note": ("Do not launch Phase 2 without proceed: true (03 §2). Criteria with "
                 "met: null are missing inputs, not failures."),
        **prov.provenance_block(),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results",
                        default=str(_REPO / "transformation_inheritance" / "results"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--pilot-step", type=int, default=2000)
    parser.add_argument("--experiment", default=EXPERIMENT_ID, choices=sorted(ARMS),
                        help="which E6A arm to gate (default: %(default)s)")
    parser.add_argument("--conditions", default=None,
                        help="comma-separated conditions, first = the CE-only control and "
                             "last = the attention-KD arm (default: the arm's registered "
                             "conditions)")
    parser.add_argument("--corpus", default=None,
                        help="corpus_id criteria 2/2b/3 are scored on (default: the corpus "
                             "the arm's pre-registration names; a smoke run builds a "
                             "synthetic corpus under a different id)")
    parser.add_argument("--parity-report", default=None,
                        help="JSON from the frozen run_parity_check (criterion 4)")
    parser.add_argument("--out", default=None,
                        help="defaults to <results>/<experiment>/pilot_gate.json")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = Path(args.results)
    conditions = ([c.strip() for c in args.conditions.split(",") if c.strip()]
                  if args.conditions else None)
    report = evaluate_gate(results, args.seed, args.pilot_step,
                           Path(args.parity_report) if args.parity_report else None,
                           corpus_id=args.corpus, experiment_id=args.experiment,
                           conditions=conditions)

    out = Path(args.out) if args.out else results / args.experiment / "pilot_gate.json"
    prov.write_json(out, report)

    print(f"{args.experiment} pilot gate (seed {args.seed}, step {args.pilot_step}, "
          f"corpus {report['corpus_id']}) -> {out}")
    for criterion in report["criteria"]:
        mark = {True: "PASS", False: "FAIL", None: "????"}[criterion["met"]]
        print(f"  [{mark}] {criterion['id']}: {criterion['text']}")
        if criterion.get("reason"):
            print(f"         reason: {criterion['reason']}")
        if criterion.get("remedy"):
            print(f"         remedy: {criterion['remedy']}")
    print(f"  n_met={report['n_met']}/{len(report['criteria'])} "
          f"unknown={report['n_unknown']} "
          f"decision={report['decision']} proceed={report['proceed']}")
    return 0 if report["proceed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
