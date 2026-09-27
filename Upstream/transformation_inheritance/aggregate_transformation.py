# -*- coding: utf-8 -*-
"""aggregate_transformation.py — E6 contrasts, matched-loss, drift, go/no-go (WP10).

``03_MODULE_SPEC_e6_transformation.md`` §7. Reads every ``checkpoint_metrics.csv`` written
by ``evaluate_transformation.py`` and produces the tables, figures and the mechanical
go/no-go decision.

    python transformation_inheritance/aggregate_transformation.py \\
      --results transformation_inheritance/results

Where the thresholds live
-------------------------
**Nowhere in this file.** Every contrast, criterion and threshold is read from
``configs/e6_preregistration.yaml``. That is deliberate: ``05`` §6 makes the criteria a
pre-registration, and a threshold embedded in analysis code is a threshold that can be
quietly tuned after seeing results. The YAML records its own sources and carries explicit
``PENDING_DESIGN_8_7`` / ``PENDING_DESIGN_18`` entries where the design document — which
is not in this repository — has not been transcribed. Those entries emit ``met: null``
with a reason and force ``decision: "incomplete"``; they are never estimated.

Statistics are dispatched to :mod:`inheritance_metrics`, never reimplemented:
:func:`paired_seed_contrast` (exact sign-flip permutation, carrying ``min_attainable_p``
and its descriptive-not-inferential note), :func:`bootstrap_ci`, :func:`bh_correct`.

Two refusals that are load-bearing
----------------------------------
* **Failure rate.** ``05`` §7.2: a condition whose rows exceed a 2% failure rate is
  excluded from contrasts unless ``--allow-high-failure``. Recorded either way.
* **Matched loss.** ``03`` §6.4: checkpoints are discrete, so the matched-loss checkpoint
  rarely lands on D0's final CE exactly. Above a 0.05-nat realised gap the comparison is
  written with ``comparable=False`` and excluded — flagged, never silently compared.

E6B tables select exactly one sentiment-adaptation arm. The historical default is
``experiment_id == "e6b"``; ``--e6b-experiment`` selects an isolated model-scale extension
without pooling it with that arm. With no matching rows, tables are emitted header-only
with a recorded reason; no E6B number is invented.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import inheritance_metrics as im  # noqa: E402
import provenance as prov  # noqa: E402

AGGREGATOR_VERSION = "aggregate_transformation_v3"

DEFAULT_PREREG = Path(__file__).resolve().parent / "configs" / "e6_preregistration.yaml"

#: The only two decision rules this aggregator honours. An absent key defaults to `any`
#: (E6A's §18 is an explicit "at least one of the following"); anything else present is
#: refused rather than defaulted, because a silently defaulted rule resolves a
#: pre-registration decision in code (`05` §6).
VALID_COMBINE: Tuple[str, ...] = ("any", "all")

#: The in-domain corpus a contrast is computed on unless the YAML entry says otherwise.
#: E6 contrasts are about the student's own domain; the cross-domain corpus is reported
#: separately in table 2 rather than pooled into a single number.
DEFAULT_CORPUS_PREFIXES = ("tinystories_", "openwebtext_", "synthetic_shuffled_natural")

#: The E6A arm this aggregation is about. A second arm (`e6a_gpt2`) distils gpt2 because the
#: TinyStories teacher has no in-domain sink to inherit; its rows must never enter this
#: arm's contrasts, matched-loss table or framing-(B) null spread. Isolation is by
#: `experiment_id`, because `load_all_metrics` rglobs and does not de-duplicate.
DEFAULT_EXPERIMENT = "e6a"
DEFAULT_E6B_EXPERIMENT = "e6b"

#: Column groups for `table2_inheritance_components.csv`. `02` §4 forbids a composite
#: "inheritance score", so these stay three separate families of columns, never summed.
TABLE2_COMPONENTS: Dict[str, Tuple[str, ...]] = {
    "topological": ("topology_wasserstein_to_teacher", "topology_spearman_to_teacher",
                    "carrier_jaccard_to_teacher", "carrier_concentration",
                    "frac_cells_above_0_2", "baseline_sink"),
    "mechanistic": ("fingerprint_cosine_to_teacher", "fingerprint_spearman_to_teacher",
                    "fingerprint_l1_to_teacher", "category_agreement_to_teacher"),
    "functional": ("functional_cosine_to_teacher", "validation_ce"),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Inputs
# ═══════════════════════════════════════════════════════════════════════════════


def load_preregistration(path: Path = DEFAULT_PREREG) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    payload.setdefault("contrasts", [])
    payload.setdefault("go_no_go", [])
    payload.setdefault("analysis", {})
    payload.setdefault("e6b", {})
    # `08` §5.2 — E6B carries its own criteria list and its own conjunction. Defaulted to
    # an empty block so a pre-registration that omits it behaves exactly as before.
    payload.setdefault("e6b_go_no_go", {})
    payload["_path"] = str(path)
    return payload


def load_all_metrics(results: Path):
    """Concatenate every ``checkpoint_metrics.csv`` under ``results``.

    The source path is recorded per row so a contradictory row can be traced back to its
    run directory rather than merely to a ``run_id``.
    """
    import pandas as pd

    frames = []
    for path in sorted(Path(results).rglob("checkpoint_metrics.csv")):
        frame = pd.read_csv(path)
        if not len(frame):
            continue
        frame["source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def ok_rows(frame, *, max_failure_rate: float, allow_high_failure: bool):
    """Rows usable in a contrast, plus the audit of what was excluded and why.

    ``05`` §7.2 — downstream stages refuse contrasts above a 2% failure rate without an
    explicit override. Exclusions are returned, not just dropped, so the audit reaches the
    output JSON.
    """
    import pandas as pd

    if not len(frame):
        return frame, []
    audit: List[Dict[str, Any]] = []
    keep = frame.copy()

    failed = keep[keep.get("status", "ok").astype(str) != "ok"]
    for _, row in failed.iterrows():
        audit.append({"reason": "status", "run_id": row.get("run_id"),
                      "checkpoint_step": row.get("checkpoint_step"),
                      "corpus_id": row.get("corpus_id"), "status": row.get("status")})
    keep = keep[keep.get("status", "ok").astype(str) == "ok"]

    if not allow_high_failure and len(keep) and "n_failed" in keep.columns:
        n_items = pd.to_numeric(keep["n_items"], errors="coerce")
        n_failed = pd.to_numeric(keep["n_failed"], errors="coerce").fillna(0.0)
        rate = n_failed / n_items.replace(0, np.nan)
        high = keep[rate > max_failure_rate]
        for index, row in high.iterrows():
            audit.append({"reason": "failure_rate", "run_id": row.get("run_id"),
                          "checkpoint_step": row.get("checkpoint_step"),
                          "corpus_id": row.get("corpus_id"),
                          "failure_rate": float(rate.loc[index]),
                          "threshold": max_failure_rate})
        keep = keep[~(rate > max_failure_rate).fillna(False)]
    return keep, audit


def assert_comparable(frame, columns=("intervention_registry_version",)) -> None:
    """``05`` §7.1: refuse to pool records from different registries or manifests.

    Raises rather than warning. A trajectory silently spanning two registry versions is
    precisely the contamination the version field exists to prevent.
    """
    for column in columns:
        if column not in frame.columns:
            continue
        values = sorted({str(v) for v in frame[column].dropna().unique()})
        if len(values) > 1:
            raise ValueError(
                f"{column} takes {len(values)} distinct values across the rows being "
                f"pooled: {values}. Records from different registries must not be "
                "compared (05 §1.1 / §7.1). Re-evaluate the older runs.")


def select_corpus(frame, corpus: Optional[str]):
    """Restrict to one corpus. Contrasts are always within a single corpus id."""
    if not len(frame):
        return frame
    if corpus:
        return frame[frame["corpus_id"] == corpus]
    mask = frame["corpus_id"].astype(str).str.startswith(DEFAULT_CORPUS_PREFIXES)
    return frame[mask] if mask.any() else frame


# ═══════════════════════════════════════════════════════════════════════════════
# Matched-loss selection (03 §6.4)
# ═══════════════════════════════════════════════════════════════════════════════


def matched_loss_checkpoint(condition_frame, target_ce: float, *, max_gap: float
                            ) -> Dict[str, Any]:
    """The **earliest** checkpoint whose validation CE is closest to ``target_ce``.

    "Earliest" and "closest" can disagree; the spec says closest, and ties break to the
    earliest step. The realised gap is always recorded, and a gap above ``max_gap``
    (0.05 nats) sets ``comparable=False`` — checkpoints are discrete, so refusing the
    comparison is the only honest option when nothing lands near the target.
    """
    import pandas as pd

    rows = condition_frame.dropna(subset=["validation_ce"])
    if not len(rows):
        return {"step": None, "validation_ce": None, "realised_gap": None,
                "comparable": False, "reason": "no checkpoint carries a validation_ce"}
    ce = pd.to_numeric(rows["validation_ce"], errors="coerce")
    steps = pd.to_numeric(rows["checkpoint_step"], errors="coerce")
    gaps = (ce - float(target_ce)).abs()
    best_gap = float(gaps.min())
    # closest first, earliest among equally-close
    tied = steps[np.isclose(gaps, best_gap, rtol=0.0, atol=1e-12)]
    step = int(tied.min())
    chosen_ce = float(ce[steps == step].iloc[0])
    comparable = best_gap <= max_gap
    out = {"step": step, "validation_ce": chosen_ce, "realised_gap": best_gap,
           "target_ce": float(target_ce), "max_gap": float(max_gap),
           "comparable": bool(comparable)}
    if not comparable:
        out["reason"] = (
            f"realised CE gap {best_gap:.4f} exceeds {max_gap} nats; the comparison is "
            "refused rather than made across mismatched losses (03 §6.4)")
    return out


def build_matched_loss(frame, *, reference_condition: str = "D0", max_gap: float = 0.05):
    """``e6a_matched_loss.csv`` — equal-step vs matched-loss, with realised CE gaps."""
    import pandas as pd

    rows: List[Dict[str, Any]] = []
    if not len(frame):
        return pd.DataFrame(columns=["seed", "condition", "reference_condition",
                                     "reference_final_step", "reference_final_ce",
                                     "equal_step", "equal_step_ce", "matched_step",
                                     "matched_ce", "realised_gap", "max_gap",
                                     "comparable", "reason"])
    for seed, seed_frame in frame.groupby("seed"):
        reference = seed_frame[seed_frame["condition"] == reference_condition]
        reference = reference.dropna(subset=["validation_ce"])
        if not len(reference):
            rows.append({"seed": int(seed), "condition": None,
                         "reference_condition": reference_condition,
                         "comparable": False,
                         "reason": f"{reference_condition} has no validation_ce rows"})
            continue
        final_step = int(pd.to_numeric(reference["checkpoint_step"]).max())
        final_ce = float(pd.to_numeric(
            reference[reference["checkpoint_step"] == final_step]["validation_ce"]
        ).iloc[0])

        for condition, condition_frame in seed_frame.groupby("condition"):
            selection = matched_loss_checkpoint(condition_frame, final_ce,
                                                max_gap=max_gap)
            equal = condition_frame[condition_frame["checkpoint_step"] == final_step]
            rows.append({
                "seed": int(seed), "condition": str(condition),
                "reference_condition": reference_condition,
                "reference_final_step": final_step, "reference_final_ce": final_ce,
                "equal_step": final_step,
                "equal_step_ce": (float(pd.to_numeric(equal["validation_ce"]).iloc[0])
                                  if len(equal) and equal["validation_ce"].notna().any()
                                  else None),
                "matched_step": selection["step"],
                "matched_ce": selection["validation_ce"],
                "realised_gap": selection["realised_gap"],
                "max_gap": max_gap,
                "comparable": selection["comparable"],
                "reason": selection.get("reason", ""),
            })
    return pd.DataFrame(rows)


def _matched_steps(matched_frame) -> Dict[Tuple[int, str], Optional[int]]:
    """``(seed, condition) -> matched step``, only where the comparison was allowed."""
    out: Dict[Tuple[int, str], Optional[int]] = {}
    for _, row in matched_frame.iterrows():
        if row.get("condition") is None:
            continue
        out[(int(row["seed"]), str(row["condition"]))] = (
            int(row["matched_step"]) if row.get("comparable")
            and row.get("matched_step") is not None else None)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Contrasts (03 §7, extended by 08 §2–§4)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Every field `08` adds is OPTIONAL. A pre-registration that names none of them produces
# byte-identical output to the pre-WP12 aggregator; that invariant is what
# `tests/test_aggregate_contracts.py` and `tests/test_matched_loss_selection.py` assert,
# and it is the reason each new form is a branch rather than a change to the old path.

#: `08` §4 — the checkpoint step written for a public reference model, which has no
#: training step at all. Used to tell a seedless reference condition from a student run
#: without matching on the condition name, so a rename cannot silently reclassify it.
REFERENCE_STEP = -1


def _value_at(frame, seed: int, condition: str, step: Optional[int],
              metric: str) -> Optional[float]:
    import pandas as pd

    rows = frame[(frame["seed"] == seed) & (frame["condition"] == condition)]
    if step is not None:
        rows = rows[rows["checkpoint_step"] == step]
    if not len(rows) or metric not in rows.columns:
        return None
    values = pd.to_numeric(rows[metric], errors="coerce").dropna()
    if not len(values):
        return None
    # At equal-step selection there is exactly one row; guard against a duplicated unit.
    return float(values.iloc[-1])


def _final_step(frame, seed: int, condition: str) -> Optional[int]:
    import pandas as pd

    rows = frame[(frame["seed"] == seed) & (frame["condition"] == condition)]
    if not len(rows):
        return None
    return int(pd.to_numeric(rows["checkpoint_step"]).max())


def is_reference_condition(frame, condition: Optional[str]) -> bool:
    """True when every row of ``condition`` is a public-reference row (``08`` §4).

    A public checkpoint contributes one measurement and no training seed, so it can never
    be an arm of a paired-seed contrast. Detected by ``checkpoint_step == -1`` rather than
    by the condition id: renaming ``P8M`` must not turn a fixed reference into a third
    replicate.
    """
    import pandas as pd

    if condition is None or not len(frame):
        return False
    rows = frame[frame["condition"] == condition]
    if not len(rows):
        return False
    steps = pd.to_numeric(rows["checkpoint_step"], errors="coerce").dropna()
    return bool(len(steps)) and bool((steps == REFERENCE_STEP).all())


def _reference_value(frame, condition: str, metric: str) -> Optional[float]:
    """The single measurement of a seedless reference condition."""
    import pandas as pd

    rows = frame[frame["condition"] == condition]
    if not len(rows) or metric not in rows.columns:
        return None
    values = pd.to_numeric(rows[metric], errors="coerce").dropna()
    return float(values.iloc[-1]) if len(values) else None


def student_seeds(frame) -> List[int]:
    """Training seeds present, excluding the sentinel a public reference row carries."""
    import pandas as pd

    if not len(frame):
        return []
    steps = pd.to_numeric(frame["checkpoint_step"], errors="coerce")
    students = frame[steps != REFERENCE_STEP]
    return sorted({int(s) for s in students["seed"].dropna().unique()})


def entry_corpus(frame, entry: Dict[str, Any], corpus_override: Optional[str]):
    """The rows one entry is computed on, and the corpus id actually used.

    `08` §2.1 needs a contrast to name its own corpus, because `e6a_c5` pairs the in-domain
    set against the cross-domain one and the default prefix filter drops the latter before
    contrasts run. Precedence: an explicit ``--corpus`` (an operator override, recorded in
    the row) beats the entry, which beats the default prefixes — the pre-WP12 behaviour for
    any entry that names no corpus.
    """
    wanted = corpus_override or entry.get("corpus_id")
    if not wanted:
        return select_corpus(frame, None), None
    return frame[frame["corpus_id"] == wanted] if len(frame) else frame, str(wanted)


def matched_steps_for(frame, reference: Optional[str], *, max_gap: float,
                      cache: Optional[Dict[Any, Any]] = None,
                      corpus_key: Optional[str] = None):
    """``(seed, condition) -> matched step`` against ``reference``'s final CE (``08`` §3).

    Design §8.7 contrast 4 compares D1's checkpoint nearest **P8M's** CE, which is a
    different reference per contrast; ``build_matched_loss`` matches everything against a
    single global reference. Results are cached because `e6a_c4a`, `e6a_c4b` and `e6a_4`'s
    two limbs would otherwise rebuild the same table four times.

    The cache key is ``(corpus_key, reference, max_gap)`` — never the frame's identity.
    ``entry_corpus`` returns a freshly filtered DataFrame on every call, so ``id(frame)``
    both misses every time *and* can be reused by a later object after garbage collection,
    which would hand one corpus's matched-loss table to another corpus's contrast.
    """
    key = (str(corpus_key), str(reference), float(max_gap))
    if cache is not None and key in cache:
        return cache[key]
    table = build_matched_loss(frame, reference_condition=str(reference), max_gap=max_gap)
    resolved = (_matched_steps(table), table)
    if cache is not None:
        cache[key] = resolved
    return resolved


def _matched_gap(table, seed: int, condition: str) -> Optional[float]:
    """The realised CE gap for one (seed, condition), for per-contrast reporting."""
    if table is None or not len(table):
        return None
    rows = table[(table["seed"] == seed) & (table["condition"] == condition)]
    if not len(rows) or rows["realised_gap"].isna().all():
        return None
    return float(rows["realised_gap"].iloc[0])


def _resolve_steps(frame, entry: Dict[str, Any], seed: int, matched_steps, *,
                   max_gap: float, cache, matched_table=None,
                   corpus_key: Optional[str] = None
                   ) -> Tuple[Optional[int], Optional[int],
                              Optional[str], Dict[str, Any]]:
    """``(step_a, step_b, skip_reason, extras)`` for one seed of one entry."""
    selection = entry.get("selection", "equal_step")
    condition_a = entry.get("condition_a")
    condition_b = entry.get("condition_b")
    extras: Dict[str, Any] = {}

    if selection == "matched_loss":
        reference = entry.get("matched_loss_reference")
        if reference:
            steps, table = matched_steps_for(frame, reference, max_gap=max_gap,
                                             cache=cache, corpus_key=corpus_key)
            extras["matched_loss_reference"] = str(reference)
        else:
            steps, table = matched_steps, matched_table
        step_a = (None if is_reference_condition(frame, condition_a)
                  else steps.get((seed, condition_a)))
        step_b = (None if condition_b is None
                  else steps.get((seed, condition_b)))
        if table is not None:
            extras["realised_gap_a"] = _matched_gap(table, seed, str(condition_a))
            if condition_b is not None:
                extras["realised_gap_b"] = _matched_gap(table, seed, str(condition_b))
        needed = [step_a] if condition_b is None else [step_a, step_b]
        if is_reference_condition(frame, condition_a):
            needed = [step_b] if condition_b is not None else []
        if any(step is None for step in needed):
            return None, None, ("matched-loss comparison refused for this seed"), extras
        return step_a, step_b, None, extras

    step_a = (None if is_reference_condition(frame, condition_a)
              else _final_step(frame, seed, str(condition_a)))
    step_b = None if condition_b is None else _final_step(frame, seed, str(condition_b))
    if condition_b is None:
        if step_a is None:
            return None, None, f"{condition_a} has no checkpoint at this seed", extras
        return step_a, None, None, extras
    if is_reference_condition(frame, condition_a):
        if step_b is None:
            return None, None, f"{condition_b} has no checkpoint at this seed", extras
        return None, step_b, None, extras
    if step_a is None or step_b is None or step_a != step_b:
        return None, None, f"no common final step ({step_a} vs {step_b})", extras
    return step_a, step_b, None, extras


def _collect_units(frame, entry: Dict[str, Any], matched_steps, *, max_gap: float,
                   cache, matched_table=None, corpus_key: Optional[str] = None
                   ) -> Tuple[List[float], List[float], List[Dict[str, Any]],
                              Dict[str, Any]]:
    """Per-unit ``(a, b)`` values plus the audit trail, for every contrast shape."""
    metric = entry["metric"]
    condition_a = entry.get("condition_a")
    condition_b = entry.get("condition_b")
    kind = entry.get("kind")
    a_values: List[float] = []
    b_values: List[float] = []
    units: List[Dict[str, Any]] = []
    meta: Dict[str, Any] = {}

    if kind == "corpus_pair":
        # `08` §2.1 — two CORPORA within each condition, not two conditions within one.
        conditions = [str(c) for c in (entry.get("conditions") or [])]
        corpus_a, corpus_b = entry.get("corpus_a"), entry.get("corpus_b")
        meta["n_conditions"] = len(conditions)
        per_condition: Dict[str, List[float]] = {}
        for seed in student_seeds(frame):
            for condition in conditions:
                step = (matched_steps.get((seed, condition))
                        if entry.get("selection") == "matched_loss"
                        else _final_step(frame, seed, condition))
                if step is None:
                    units.append({"seed": seed, "condition": condition,
                                  "skipped": "no checkpoint selected"})
                    continue
                rows_a = frame[frame["corpus_id"] == corpus_a]
                rows_b = frame[frame["corpus_id"] == corpus_b]
                va = _value_at(rows_a, seed, condition, step, metric)
                vb = _value_at(rows_b, seed, condition, step, metric)
                if va is None or vb is None:
                    units.append({"seed": seed, "condition": condition,
                                  "skipped": f"{metric} missing on "
                                             f"{corpus_a if va is None else corpus_b}"})
                    continue
                a_values.append(va)
                b_values.append(vb)
                per_condition.setdefault(condition, []).append(va - vb)
                units.append({"seed": seed, "condition": condition, "step": step,
                              "value_a": va, "value_b": vb, "diff": va - vb})
        # A single condition driving the pooled effect must be visible, not averaged away.
        meta["per_condition"] = {c: {"n": len(v), "mean_diff": float(np.mean(v))}
                                 for c, v in per_condition.items()}
        meta["n_seeds"] = len({u["seed"] for u in units if "diff" in u})
        return a_values, b_values, units, meta

    reference_arm = is_reference_condition(frame, condition_a)
    if reference_arm:
        meta["pairing"] = "reference_vs_seeds"
        meta["test"] = "descriptive_reference_contrast"

    for seed in student_seeds(frame):
        step_a, step_b, skipped, extras = _resolve_steps(
            frame, entry, seed, matched_steps, max_gap=max_gap, cache=cache,
            matched_table=matched_table, corpus_key=corpus_key)
        meta.update({k: v for k, v in extras.items() if k == "matched_loss_reference"})
        if skipped:
            units.append({"seed": seed, "skipped": skipped, **extras})
            continue
        if reference_arm:
            va = _reference_value(frame, str(condition_a), metric)
        else:
            va = _value_at(frame, seed, str(condition_a), step_a, metric)
        vb = (None if condition_b is None
              else _value_at(frame, seed, str(condition_b), step_b, metric))
        if va is None or (condition_b is not None and vb is None):
            units.append({"seed": seed, "skipped": f"{metric} missing for one arm",
                          **extras})
            continue
        a_values.append(va)
        b_values.append(0.0 if condition_b is None else float(vb))
        unit = {"seed": seed, "step_a": step_a, "step_b": step_b, "value_a": va,
                **extras}
        if condition_b is None:
            unit.update({"value_b": None, "diff": va})
        else:
            unit.update({"value_b": vb, "diff": va - vb})
        units.append(unit)

    meta["n_seeds"] = len(a_values)
    if condition_b is None:
        meta["pairing"] = meta.get("pairing", "single_arm")
    return a_values, b_values, units, meta


def evaluate_contrast(frame, entry: Dict[str, Any], matched_steps, *,
                      corpus_override: Optional[str] = None, max_gap: float = 0.05,
                      matched_cache: Optional[Dict[Any, Any]] = None,
                      matched_table=None) -> Dict[str, Any]:
    """One contrast entry from the pre-registration, evaluated across seeds.

    A ``PENDING_*`` entry short-circuits: it reports its reason and produces no number.
    """
    result: Dict[str, Any] = {"id": entry.get("id"), "text": entry.get("text", ""),
                              "kind": entry.get("kind"),
                              "metric": entry.get("metric"),
                              "condition_a": entry.get("condition_a"),
                              "condition_b": entry.get("condition_b"),
                              "selection": entry.get("selection", "equal_step"),
                              "family": entry.get("family", "unassigned")}
    status = entry.get("status")
    if status:
        result.update({"status": status, "reason": entry.get("reason", "").strip(),
                       "n_seeds": 0, "mean_diff": None, "mean_abs_diff": None,
                       "p_value": None,
                       "min_attainable_p": None, "ci_lo": None, "ci_hi": None,
                       "per_seed_json": None, "note": "pre-registration entry not "
                                                      "transcribed; no value computed"})
        return result

    scoped, corpus_used = entry_corpus(frame, entry, corpus_override)
    result["corpus_id"] = corpus_used
    result["corpus_override"] = corpus_override
    if entry.get("kind") == "corpus_pair":
        # A corpus-pair entry needs both corpora, so it reads the unfiltered frame.
        scoped = frame if corpus_override is None else scoped
        result["corpus_id"] = f"{entry.get('corpus_a')} vs {entry.get('corpus_b')}"

    absolute = bool(entry.get("absolute_difference"))
    result["absolute_difference"] = absolute
    a_values, b_values, units, meta = _collect_units(
        scoped, entry, matched_steps, max_gap=max_gap, cache=matched_cache,
        matched_table=matched_table,
        corpus_key=result["corpus_id"] or "__default__")

    result["per_seed_json"] = json.dumps(units, default=str)
    for key in ("n_conditions", "per_condition", "pairing", "test",
                "matched_loss_reference"):
        if key in meta:
            result[key] = (json.dumps(meta[key], default=str)
                           if key == "per_condition" else meta[key])
    result["n_seeds"] = int(meta.get("n_seeds", len(a_values)))
    result["n_units"] = len(a_values)
    # `08` §3: the realised CE gap belongs on the contrast, not only in the matched-loss
    # table, so a reader of one row can see how far from "matched" the comparison was.
    gaps = [unit[key] for unit in units for key in ("realised_gap_a", "realised_gap_b")
            if unit.get(key) is not None]
    result["max_realised_gap"] = float(max(gaps)) if gaps else None

    if not a_values:
        result.update({"status": "no_data", "mean_diff": None, "mean_abs_diff": None,
                       "p_value": None, "min_attainable_p": None, "ci_lo": None,
                       "ci_hi": None,
                       "reason": "no seed had both arms measured at a common step"})
        return result

    diffs = [a - b for a, b in zip(a_values, b_values)]
    # `08` §2.2 — the magnitude is taken BEFORE the threshold test and before the
    # permutation, because "differ by at least 0.10" is directionless. Both means are
    # recorded so the sign is never lost, only set aside.
    tested = [abs(d) for d in diffs] if absolute else list(diffs)
    contrast = im.paired_seed_contrast(tested, [0.0] * len(tested))
    lo, hi = im.bootstrap_ci(tested)
    result.update({
        "status": "ok",
        "mean_diff": float(np.mean(diffs)),
        "mean_abs_diff": float(np.mean([abs(d) for d in diffs])),
        "observed": contrast["mean_diff"],
        "p_value": contrast["p_value"],
        "min_attainable_p": contrast["min_attainable_p"],
        "note": contrast["note"],
        "ci_lo": lo, "ci_hi": hi,
        "ci_uncertainty_kind": "sampling_over_seeds",
        "mean_a": float(np.mean(a_values)), "mean_b": float(np.mean(b_values)),
        "per_seed_tested_json": json.dumps(tested),
    })
    if result.get("pairing") == "reference_vs_seeds":
        result["note"] = (
            "descriptive reference contrast: the uncertainty is over student seeds only "
            "and carries none from the public reference, which is one checkpoint (08 §4)")
    return result


def build_contrasts(frame, prereg: Dict[str, Any], matched_steps, *,
                    corpus_override: Optional[str] = None, matched_table=None):
    """``e6a_contrasts.csv`` with BH correction applied **within** each family."""
    import pandas as pd

    max_gap = float(prereg["analysis"].get("matched_loss_max_gap_nats", 0.05))
    cache: Dict[Any, Any] = {}
    rows = [evaluate_contrast(frame, entry, matched_steps,
                              corpus_override=corpus_override, max_gap=max_gap,
                              matched_cache=cache, matched_table=matched_table)
            for entry in prereg["contrasts"]]
    alpha = float(prereg["analysis"].get("bh_alpha", 0.05))
    for family in {row["family"] for row in rows}:
        members = [row for row in rows
                   if row["family"] == family and row.get("p_value") is not None]
        if not members:
            continue
        rejected, qvalues = im.bh_correct([row["p_value"] for row in members],
                                          alpha=alpha)
        for row, reject, q in zip(members, rejected, qvalues):
            row["q_value"] = float(q)
            row["bh_rejected"] = bool(reject)
            row["bh_family_size"] = len(members)
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# Table 2 — inheritance components kept separate (02 §4: no composite score)
# ═══════════════════════════════════════════════════════════════════════════════


def build_table2(frame):
    import pandas as pd

    if not len(frame):
        return pd.DataFrame(columns=["run_id", "condition", "seed", "checkpoint_step",
                                     "corpus_id", "component", "metric", "value"])
    rows: List[Dict[str, Any]] = []
    for _, row in frame.iterrows():
        for component, columns in TABLE2_COMPONENTS.items():
            for column in columns:
                if column not in frame.columns:
                    continue
                value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                if pd.isna(value):
                    continue
                rows.append({"run_id": row.get("run_id"),
                             "condition": row.get("condition"),
                             "seed": row.get("seed"),
                             "checkpoint_step": row.get("checkpoint_step"),
                             "corpus_id": row.get("corpus_id"),
                             "component": component, "metric": column,
                             "value": float(value)})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# E6B — built now, fed when WP6 lands
# ═══════════════════════════════════════════════════════════════════════════════


E6B_DRIFT_COLUMNS = ("run_id", "condition", "seed", "checkpoint_step", "corpus_id",
                     "fingerprint_drift_from_base", "topology_drift_from_base",
                     "carrier_drift_from_base")
E6B_ONSET_COLUMNS = ("run_id", "condition", "seed", "drift_column", "threshold",
                     "onset_step", "threshold_kind")
E6B_FACTORIAL_COLUMNS = ("metric", "adaptation", "label_quality", "mean", "n",
                         "interaction")


def build_e6b_drift(frame):
    """``e6b_drift.csv`` — the drift columns evaluate_transformation already writes."""
    import pandas as pd

    if not len(frame):
        return pd.DataFrame(columns=list(E6B_DRIFT_COLUMNS))
    columns = [c for c in E6B_DRIFT_COLUMNS if c in frame.columns]
    return frame[columns].copy()


def build_e6b_early_warning(drift_frame, prereg: Dict[str, Any]):
    """``e6b_early_warning.csv`` — onset step per drift column, per run.

    The threshold is ``mean + k*std`` over the **clean** runs' drift values
    (``inheritance_metrics.clean_run_threshold``), and the onset is the first step past it
    (``drift_onset``). Both are dispatched, not reimplemented.
    """
    import pandas as pd

    columns = list(prereg["e6b"].get("drift_columns")
                   or ("fingerprint_drift_from_base", "topology_drift_from_base",
                       "carrier_drift_from_base"))
    k = float(prereg["e6b"].get("onset_threshold_k", 2.0))
    if not len(drift_frame):
        return pd.DataFrame(columns=list(E6B_ONSET_COLUMNS))

    conditions = prereg["e6b"].get("factorial", {}).get("conditions", {})
    clean = {c for c, factors in conditions.items() if "clean" in factors}
    rows: List[Dict[str, Any]] = []
    for column in columns:
        if column not in drift_frame.columns:
            continue
        clean_trajs = []
        for (_run, _seed), sub in drift_frame[
                drift_frame["condition"].isin(clean)].groupby(["run_id", "seed"]):
            clean_trajs.append(sub.rename(columns={"checkpoint_step": "step"})
                               [["step", column]])
        threshold = (im.clean_run_threshold(clean_trajs, column, k=k)
                     if clean_trajs else float("nan"))
        for (run_id, condition, seed), sub in drift_frame.groupby(
                ["run_id", "condition", "seed"]):
            traj = sub.rename(columns={"checkpoint_step": "step"}).sort_values("step")
            onset = (im.drift_onset(traj, threshold, column)
                     if np.isfinite(threshold) else None)
            rows.append({"run_id": run_id, "condition": condition, "seed": seed,
                         "drift_column": column,
                         "threshold": (float(threshold) if np.isfinite(threshold)
                                       else None),
                         "onset_step": onset,
                         "threshold_kind": f"clean_run_mean_plus_{k}_std"})
    return pd.DataFrame(rows)


def build_e6b_factorial(frame, prereg: Dict[str, Any]):
    """``e6b_factorial.csv`` — 2×2 (LoRA vs full) × (clean vs corrupt) with interaction."""
    import pandas as pd

    conditions = prereg["e6b"].get("factorial", {}).get("conditions", {})
    if not len(frame) or not conditions:
        return pd.DataFrame(columns=list(E6B_FACTORIAL_COLUMNS))

    factor_of = {condition: tuple(factors) for condition, factors in conditions.items()}
    rows: List[Dict[str, Any]] = []
    metrics = [c for c in ("fingerprint_drift_from_base", "topology_drift_from_base",
                           "carrier_drift_from_base", "task_accuracy")
               if c in frame.columns]
    for metric in metrics:
        cells: Dict[Tuple[str, str], float] = {}
        for condition, (adaptation, quality) in factor_of.items():
            sub = frame[frame["condition"] == condition]
            values = pd.to_numeric(sub.get(metric), errors="coerce").dropna() \
                if len(sub) else pd.Series(dtype=float)
            if not len(values):
                continue
            cells[(adaptation, quality)] = float(values.mean())
            rows.append({"metric": metric, "adaptation": adaptation,
                         "label_quality": quality, "mean": float(values.mean()),
                         "n": int(len(values)), "interaction": None})
        needed = {("lora", "clean"), ("lora", "corrupt"),
                  ("full", "clean"), ("full", "corrupt")}
        if needed.issubset(cells):
            interaction = ((cells[("lora", "corrupt")] - cells[("lora", "clean")])
                           - (cells[("full", "corrupt")] - cells[("full", "clean")]))
            rows.append({"metric": metric, "adaptation": "interaction",
                         "label_quality": "interaction", "mean": None, "n": None,
                         "interaction": float(interaction)})
    return pd.DataFrame(rows)


def build_table3(frame, prereg: Dict[str, Any]):
    """``table3_clean_vs_corrupt.csv`` — clean vs corrupt means per metric."""
    import pandas as pd

    conditions = prereg["e6b"].get("factorial", {}).get("conditions", {})
    columns = ["metric", "label_quality", "mean", "n"]
    if not len(frame) or not conditions:
        return pd.DataFrame(columns=columns)
    quality_of = {c: factors[1] for c, factors in conditions.items() if len(factors) > 1}
    rows: List[Dict[str, Any]] = []
    for metric in [c for c in ("fingerprint_drift_from_base", "task_accuracy",
                               "ece_10bin") if c in frame.columns]:
        for quality in ("clean", "corrupt"):
            members = [c for c, q in quality_of.items() if q == quality]
            sub = frame[frame["condition"].isin(members)]
            values = pd.to_numeric(sub.get(metric), errors="coerce").dropna() \
                if len(sub) else pd.Series(dtype=float)
            if not len(values):
                continue
            rows.append({"metric": metric, "label_quality": quality,
                         "mean": float(values.mean()), "n": int(len(values))})
    return pd.DataFrame(rows)


# ═══════════════════════════════════════════════════════════════════════════════
# go_no_go.json (05 §6)
# ═══════════════════════════════════════════════════════════════════════════════


def resolve_threshold_rule(frame, rule: Dict[str, Any], entry: Dict[str, Any],
                           matched_steps, *, max_gap: float, cache,
                           corpus_override: Optional[str] = None) -> Dict[str, Any]:
    """``08`` §7 — a threshold calibrated against the metric's own null spread.

    The *rule* is the pre-registration; the number it produces is a result, so the realised
    value is written into ``go_no_go.json`` beside the rule rather than back into the YAML.
    The spread is measured across seeds **within** one condition (D0 vs D0), which does not
    involve the D1/D2 comparison the criterion tests, so calibrating cannot leak the
    outcome.

    Two guards, both refusals:

    * fewer than three seeds in the null condition — a spread from two points is a gap, not
      a spread, and would make the threshold an artefact of which two seeds ran;
    * the null condition absent from a frame that does carry the treatment arm — silently
      returning "no threshold" there is indistinguishable from a criterion that was never
      meant to have one.

    ``anchor_side`` selects which side of the anchor the boundary sits on:

    * ``below`` (the default, and the only behaviour before it existed) gives
      ``anchor − k·spread``. Correct where the observed quantity is a *similarity* with a
      natural ceiling — ``e6a_3``'s two limbs read that one boundary from either side of an
      anchor at 1.0.
    * ``above`` gives ``anchor + k·spread``. Required where the observed quantity is an
      *absolute difference* anchored at 0, as in ``e6a_4`` (P8M versus D2) and ``e6b_2``
      (F2 versus F1 drift). With ``below`` those would resolve to ``0 − k·spread``, a
      negative bar: "similar function" (``less_equal``) could never be satisfied and
      "measurably different" (``greater_equal``) would be satisfied by anything at all.
      Two criteria that silently cannot fail is worse than two that cannot be evaluated.

    An absent ``anchor_side`` defaults to ``below`` because that reproduces the
    pre-extension behaviour exactly; a *present but unrecognised* value is refused rather
    than folded into either side (CLAUDE.md trap 13).
    """
    out: Dict[str, Any] = {"kind": rule.get("kind"), "metric": rule.get("metric"),
                           "null_condition": rule.get("null_condition"),
                           "statistic": rule.get("statistic", "range"),
                           "k": rule.get("k"), "anchor": rule.get("anchor"),
                           "anchor_side": rule.get("anchor_side", "below"),
                           "threshold": None, "spread": None, "n_seeds": 0}
    if str(rule.get("kind")) != "null_seed_spread":
        out["reason"] = (f"unknown threshold_rule kind {rule.get('kind')!r}; the "
                         "aggregator refuses to guess a calibration it does not implement")
        return out
    if out["anchor_side"] not in ("below", "above"):
        out["reason"] = (f"unknown threshold_rule anchor_side {out['anchor_side']!r}; "
                         "expected 'below' or 'above'. Refused rather than defaulted — a "
                         "resolved threshold on the wrong side of the anchor is a criterion "
                         "that cannot fail (CLAUDE.md trap 13)")
        return out

    metric = rule.get("metric") or entry.get("metric")
    condition = str(rule.get("null_condition"))
    scoped, corpus_used = entry_corpus(frame, entry, corpus_override)
    out["corpus_id"] = corpus_used
    values: List[float] = []
    for seed in student_seeds(scoped):
        step = (matched_steps.get((seed, condition))
                if entry.get("selection") == "matched_loss"
                else _final_step(scoped, seed, condition))
        if step is None:
            continue
        value = _value_at(scoped, seed, condition, step, metric)
        if value is not None:
            values.append(value)

    out["n_seeds"] = len(values)
    if not values:
        treatment = entry.get("condition_a")
        has_treatment = bool(len(scoped)) and bool(
            (scoped["condition"] == treatment).any())
        out["reason"] = (
            f"null condition {condition!r} has no usable {metric!r} rows"
            + (f", but the treatment arm {treatment!r} is present — the calibration is "
               "refused rather than skipped (08 §7)" if has_treatment else ""))
        return out
    if len(values) < 3:
        out["reason"] = (f"null condition {condition!r} has {len(values)} seed(s); a "
                         "spread needs at least three (08 §7)")
        return out

    array = np.asarray(values, dtype=float)
    spread = (float(array.max() - array.min()) if out["statistic"] == "range"
              else float(array.std(ddof=1)))
    if out["statistic"] not in ("range", "std"):
        out["reason"] = f"unknown threshold_rule statistic {out['statistic']!r}"
        return out
    anchor = float(rule.get("anchor", 1.0))
    k = float(rule.get("k", 2.0))
    offset = (-k * spread) if out["anchor_side"] == "below" else (k * spread)
    out.update({
        "spread": spread,
        "null_seed_values": [float(v) for v in values],
        # "within k spreads of the anchor" for a `greater_equal` limb and "more than k
        # spreads below it" for a `less_equal` limb are the same boundary read from the
        # two sides; the limb's own `direction` decides which side is being asserted.
        # `anchor_side` decides where that single boundary sits relative to the anchor —
        # see the docstring for why an absolute-difference criterion needs `above`.
        "threshold": float(anchor + offset),
        "note": "threshold derived from the null condition's across-seed spread; the RULE "
                "is the pre-registration, this NUMBER is a result (08 §7)",
    })
    return out


def _test_threshold(observed: Optional[float], threshold: Optional[float],
                    direction: str) -> Optional[bool]:
    if observed is None or threshold is None:
        return None
    return bool(observed >= threshold if direction == "greater_equal"
                else observed <= threshold)


def _evaluate_simple_criterion(frame, entry: Dict[str, Any], matched_steps, *,
                               corpus_override: Optional[str], max_gap: float,
                               cache, matched_table=None) -> Dict[str, Any]:
    """One non-compound criterion body: a contrast, a threshold and a direction."""
    contrast = evaluate_contrast(frame, {**entry, "family": entry.get("family",
                                                                     "go_no_go")},
                                 matched_steps, corpus_override=corpus_override,
                                 max_gap=max_gap, matched_cache=cache,
                                 matched_table=matched_table)
    per_seed = json.loads(contrast["per_seed_json"]) if contrast.get("per_seed_json") \
        else []
    tested = (json.loads(contrast["per_seed_tested_json"])
              if contrast.get("per_seed_tested_json") else
              [row["diff"] for row in per_seed if "diff" in row])
    direction = entry.get("direction", "greater_equal")
    observed = contrast.get("observed", contrast.get("mean_diff"))

    out: Dict[str, Any] = {
        "metric": entry.get("metric"), "condition_a": entry.get("condition_a"),
        "condition_b": entry.get("condition_b"),
        "selection": entry.get("selection", "equal_step"),
        "corpus_id": contrast.get("corpus_id"),
        "observed": observed, "direction": direction,
        "all_seeds": tested, "n_seeds": contrast.get("n_seeds", 0),
        "absolute_difference": bool(entry.get("absolute_difference")),
        "contrast_status": contrast.get("status"),
    }
    if contrast.get("max_realised_gap") is not None:
        out["max_realised_gap"] = contrast["max_realised_gap"]
    if contrast.get("matched_loss_reference"):
        out["matched_loss_reference"] = contrast["matched_loss_reference"]

    threshold = entry.get("threshold")
    rule = entry.get("threshold_rule")
    if threshold is None and rule:
        # The SAME corpus override the observed value was measured on. Without this the
        # criterion compares a value measured on the overridden corpus against a threshold
        # calibrated on the entry's declared one — a threshold that looks calibrated and
        # was calibrated against different data.
        resolved = resolve_threshold_rule(frame, rule, entry, matched_steps,
                                          max_gap=max_gap, cache=cache,
                                          corpus_override=corpus_override)
        out["threshold_rule"] = resolved
        threshold = resolved.get("threshold")
    out["threshold"] = threshold

    if observed is None or threshold is None:
        out.update({"met": None,
                    "reason": (out.get("threshold_rule", {}).get("reason")
                               or contrast.get("reason")
                               or ("the pre-registration supplies no threshold for this "
                                   "criterion" if observed is not None
                                   else "no data for this criterion yet"))})
        return out

    # `08` §5.1 — the correctness fix. Design §18 says "across all three seeds"; a pooled
    # mean lets one strong seed carry two weak ones, so `seed_consistency: all` tests every
    # seed. Absent, the pooled-mean reading is preserved exactly.
    if str(entry.get("seed_consistency", "")).lower() == "all":
        per_seed_met = [_test_threshold(value, threshold, direction) for value in tested]
        out.update({"seed_consistency": "all",
                    "per_seed_met": per_seed_met,
                    "n_seeds_meeting": sum(1 for m in per_seed_met if m is True),
                    "pooled_met": _test_threshold(observed, threshold, direction)})
        out["met"] = bool(per_seed_met) and all(m is True for m in per_seed_met)
        return out

    out["met"] = _test_threshold(observed, threshold, direction)
    return out


def evaluate_criterion(frame, entry: Dict[str, Any], matched_steps, *,
                       corpus_override: Optional[str] = None, max_gap: float = 0.05,
                       matched_cache: Optional[Dict[Any, Any]] = None,
                       matched_table=None) -> Dict[str, Any]:
    """One design-§18 criterion. ``PENDING_*`` entries report ``met: null`` and a reason."""
    out: Dict[str, Any] = {"id": entry.get("id"), "text": entry.get("text", "")}
    status = entry.get("status")
    if status:
        out.update({"status": status, "observed": None, "threshold": None,
                    "all_seeds": [], "met": None,
                    "reason": entry.get("reason", "").strip()})
        return out

    cache = matched_cache if matched_cache is not None else {}
    compound = entry.get("compound")
    if compound:
        # `08` §2.3 — an AND over ordinary criterion bodies. Every limb reports its own
        # verdict so a failure names which one failed; a `None` limb makes the whole
        # criterion `None`, never `False`, because unknown is not the same as refuted.
        limbs = list(compound.get("all_of") or [])
        results = []
        for index, limb in enumerate(limbs):
            body = {**{k: v for k, v in entry.items()
                       if k in ("family", "seed_consistency")}, **limb}
            sub = _evaluate_simple_criterion(frame, body, matched_steps,
                                             corpus_override=corpus_override,
                                             max_gap=max_gap, cache=cache,
                                             matched_table=matched_table)
            sub["limb"] = limb.get("id") or f"all_of[{index}]"
            results.append(sub)
        met_values = [sub.get("met") for sub in results]
        if not results:
            met = None
        elif any(value is None for value in met_values):
            met = None
        else:
            met = all(bool(value) for value in met_values)
        out.update({"compound": "all_of", "sub_criteria": results, "met": met,
                    "observed": None, "threshold": None,
                    "all_seeds": [], "n_seeds": max((sub.get("n_seeds", 0)
                                                     for sub in results), default=0)})
        if met is None:
            failing = [sub["limb"] for sub in results if sub.get("met") is None]
            out["reason"] = ("compound criterion unresolved; limb(s) "
                             f"{failing} report no verdict: "
                             + "; ".join(str(sub.get("reason", "")) for sub in results
                                         if sub.get("met") is None))
        elif met is False:
            out["reason"] = ("compound criterion failed on limb(s) "
                             + str([sub["limb"] for sub in results
                                    if sub.get("met") is False]))
        return out

    out.update(_evaluate_simple_criterion(frame, entry, matched_steps,
                                          corpus_override=corpus_override,
                                          max_gap=max_gap, cache=cache,
                                          matched_table=matched_table))
    return out


def collect_pending_decisions(prereg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every ``PENDING_*`` sentinel outside the go/no-go criteria list.

    ``08`` §12 requires ``decision: "incomplete"`` while any ``PENDING_DECISION_*`` remains,
    and the v3 pre-registration puts three of them where ``build_go_no_go`` never looked:
    ``primary_topology_metric``, the E6B criteria, and PENDING contrasts. They are reported
    under their own key so ``pending_preregistration`` keeps meaning exactly what it meant
    before — the criteria that could not be evaluated.

    None of them carries ``blocks_decision``: an E6A criterion inlines its own contrast, so
    no sentinel outside the criteria list can change an E6A verdict. ``primary_topology_metric``
    chooses which of two *already computed* contrasts the paper calls primary, and the E6B
    entries are evaluated in their own ``e6b_go_no_go.json``, where they already report
    ``met: null``. They are reported anyway: an operator must know they are open before
    quoting a table, and a field that is read by nothing is the one that goes stale.
    """
    found: List[Dict[str, Any]] = []

    def _record(location: str, node: Any) -> None:
        if isinstance(node, dict) and str(node.get("status", "")).startswith("PENDING"):
            found.append({"location": location, "id": node.get("id"),
                          "status": node["status"], "blocks_decision": False,
                          "reason": str(node.get("reason", "")).strip()})

    for index, entry in enumerate(prereg.get("contrasts") or []):
        _record(f"contrasts[{index}]", entry)
    _record("primary_topology_metric", prereg.get("primary_topology_metric"))
    e6b = prereg.get("e6b_go_no_go") or {}
    for index, entry in enumerate(e6b.get("criteria") or []):
        _record(f"e6b_go_no_go.criteria[{index}]", entry)
    return found


def build_go_no_go(frame, prereg: Dict[str, Any], matched_steps,
                   experiment: str = "e6a", *, corpus_override: Optional[str] = None,
                   criteria_key: str = "go_no_go", matched_table=None) -> Dict[str, Any]:
    """``05`` §6 shape. Any unresolved criterion forces ``decision: "incomplete"``."""
    analysis = prereg.get("analysis", {})
    max_gap = float(analysis.get("matched_loss_max_gap_nats", 0.05))
    cache: Dict[Any, Any] = {}

    block = prereg.get(criteria_key) or []
    # An *absent* combine key defaults to `any` — E6A's §18 is an explicit "at least one of
    # the following", and that is today's rule. A *present but unrecognised* one is refused
    # rather than defaulted: reading `combine: "al"` as `any` would invert the decision
    # rule and nothing in the artefact would say so (CLAUDE.md trap 13).
    declared_combine = prereg.get("go_no_go_combine")
    if isinstance(block, dict):                      # the `e6b_go_no_go` shape (08 §5.2)
        if "combine" in block:
            declared_combine = block["combine"]
        entries = list(block.get("criteria") or [])
    else:
        entries = list(block)
    combine = ("any" if declared_combine is None
               else str(declared_combine).strip().lower())
    combine_error = ""
    if combine not in VALID_COMBINE:
        combine_error = (f"{criteria_key} combine rule is {declared_combine!r}; expected "
                         f"one of {list(VALID_COMBINE)}. The decision rule is refused "
                         "rather than defaulted, so no verdict is reached.")

    criteria = [evaluate_criterion(frame, entry, matched_steps,
                                   corpus_override=corpus_override, max_gap=max_gap,
                                   matched_cache=cache, matched_table=matched_table)
                for entry in entries]
    n_met = sum(1 for c in criteria if c["met"] is True)
    n_unknown = sum(1 for c in criteria if c["met"] is None)
    pending = [c["id"] for c in criteria if str(c.get("status", "")).startswith("PENDING")]
    pending_decisions = collect_pending_decisions(prereg)

    blocking = [d for d in pending_decisions if d["blocks_decision"]]
    if n_unknown or blocking or combine_error:
        decision = "incomplete"
    elif (all(c["met"] is True for c in criteria) if combine == "all" else bool(n_met)):
        decision = "continue"
    else:
        decision = "stop"
    return {
        "experiment": experiment,
        "criteria": criteria,
        "combine": combine,
        "combine_error": combine_error,
        "n_criteria": len(criteria),
        "n_met": n_met,
        "n_unknown": n_unknown,
        "pending_preregistration": pending,
        "pending_decisions": pending_decisions,
        "pending_decisions_blocking": [d["location"] for d in blocking],
        "decision": decision,
        "preregistration": prereg.get("_path"),
        "preregistration_version": prereg.get("version"),
        "note": ("Criteria are read from the pre-registration YAML, never embedded here. "
                 "PENDING_* entries are decisions the design document does not make (see "
                 "new_design_plans/DECISIONS_REQUIRED.md); they report met: null and force "
                 "decision: incomplete rather than being estimated (CLAUDE.md rule 4)."),
        "evaluated_utc": prov.utc_now(),
        "aggregator_version": AGGREGATOR_VERSION,
        **prov.provenance_block(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Figures (design §19)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `03` §7 says to import the style helpers from `emergence_dynamics_analysis.py`. That
# module exposes none — its plotting is a set of experiment-specific `plot_*` functions
# with the style inline — and importing it would pull torch and the HF stack into a pure
# analysis step. Its conventions are therefore matched literally: the Agg backend, and
# `dpi=200, bbox_inches="tight"` on every save.


def _figure_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_trajectories(frame, path: Path) -> Optional[Path]:
    """fig2 — sink and teacher-similarity trajectories per condition."""
    import pandas as pd

    if not len(frame):
        return None
    plt = _figure_setup()
    panels = [("baseline_sink", "baseline sink"),
              ("fingerprint_cosine_to_teacher", "fingerprint cosine to teacher"),
              ("validation_ce", "validation CE"),
              ("carrier_concentration", "carrier concentration (Gini)")]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, (column, label) in zip(axes.ravel(), panels):
        if column not in frame.columns:
            ax.set_visible(False)
            continue
        for condition, sub in frame.groupby("condition"):
            grouped = sub.groupby("checkpoint_step")[column].mean().dropna()
            if len(grouped):
                ax.plot(grouped.index, grouped.values, marker="o", label=str(condition))
        ax.set_xlabel("step")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
    fig.suptitle("E6A trajectories (mean over seeds)")
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_matched_loss(matched_frame, path: Path) -> Optional[Path]:
    """fig3 — realised CE gap per condition, with the 0.05-nat refusal line."""
    if not len(matched_frame) or matched_frame["realised_gap"].isna().all():
        return None
    plt = _figure_setup()
    fig, ax = plt.subplots(figsize=(8, 5))
    rows = matched_frame.dropna(subset=["realised_gap"])
    for index, (_, row) in enumerate(rows.iterrows()):
        ax.bar(index, row["realised_gap"],
               color="tab:blue" if row["comparable"] else "tab:red")
    ax.axhline(float(rows["max_gap"].iloc[0]) if "max_gap" in rows.columns else 0.05,
               color="gray", linestyle=":", label="refusal threshold")
    ax.set_xticks(range(len(rows)))
    ax.set_xticklabels([f"{r['condition']}/s{r['seed']}" for _, r in rows.iterrows()],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("realised CE gap (nats)")
    ax.set_title("Matched-loss selection: realised gaps (red = comparison refused)")
    ax.legend()
    out = _save(fig, path)
    plt.close(fig)
    return out


def plot_drift(drift_frame, path: Path) -> Optional[Path]:
    """fig4 — E6B drift trajectories. Returns None until WP6 produces rows."""
    if not len(drift_frame):
        return None
    plt = _figure_setup()
    columns = [c for c in ("fingerprint_drift_from_base", "topology_drift_from_base",
                           "carrier_drift_from_base") if c in drift_frame.columns]
    if not columns:
        return None
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.5),
                             squeeze=False)
    for ax, column in zip(axes[0], columns):
        for condition, sub in drift_frame.groupby("condition"):
            grouped = sub.groupby("checkpoint_step")[column].mean().dropna()
            if len(grouped):
                ax.plot(grouped.index, grouped.values, marker="o", label=str(condition))
        ax.set_xlabel("step")
        ax.set_ylabel(column)
        ax.legend(fontsize=8)
    fig.suptitle("E6B drift from base")
    out = _save(fig, path)
    plt.close(fig)
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════════


def _write(frame, path: Path, *, reason: str = "") -> Path:
    """Write a table, always with its header, plus a sidecar reason when it is empty.

    An empty table is a fact ("no e6b runs found"), not an absence — writing the header
    and the reason keeps that fact in the artefact tree instead of leaving a reader to
    guess whether the step ran.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    if not len(frame) and reason:
        prov.write_json(path.with_suffix(".reason.json"),
                        {"table": path.name, "n_rows": 0, "reason": reason,
                         **prov.provenance_block()})
    return path


def aggregate(results: Path, *, prereg_path: Path = DEFAULT_PREREG,
              corpus: Optional[str] = None, allow_high_failure: bool = False,
              reference_condition: str = "D0", figures: bool = True,
              progress: bool = True,
              experiment: str = DEFAULT_EXPERIMENT,
              e6b_experiment: str = DEFAULT_E6B_EXPERIMENT) -> Dict[str, Any]:
    """Read every ``checkpoint_metrics.csv`` under ``results`` and write ``aggregate/``."""
    import pandas as pd

    prereg = load_preregistration(prereg_path)
    analysis = prereg["analysis"]
    out_dir = Path(results) / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all_metrics(results)
    if not len(raw):
        # Still emit the contrast table and the decision, both empty with a reason: an
        # aggregation that found nothing is a recorded state, not a silent no-op.
        blank = pd.DataFrame(columns=["seed", "condition", "checkpoint_step",
                                      "corpus_id"])
        reason = f"no checkpoint_metrics.csv under {results}"
        contrasts = build_contrasts(blank, prereg, {}, corpus_override=corpus)
        outputs = {"e6a_contrasts.csv": _write(contrasts, out_dir / "e6a_contrasts.csv",
                                               reason=reason)}
        decision = build_go_no_go(blank, prereg, {}, corpus_override=corpus)
        prov.write_json(out_dir / "go_no_go.json", decision)
        outputs["go_no_go.json"] = out_dir / "go_no_go.json"
        e6b_decision = build_go_no_go(blank, prereg, {}, experiment=e6b_experiment,
                                      criteria_key="e6b_go_no_go")
        prov.write_json(out_dir / "e6b_go_no_go.json", e6b_decision)
        outputs["e6b_go_no_go.json"] = out_dir / "e6b_go_no_go.json"
        report = {"results_root": str(results), "n_rows_read": 0, "n_rows_used": 0,
                  "n_rows_excluded": 0, "reason": reason,
                  "decision": decision["decision"],
                  "pending_preregistration": decision["pending_preregistration"],
                  "pending_decisions": decision["pending_decisions"],
                  "e6b_experiment": e6b_experiment,
                  "outputs": {k: str(v) for k, v in outputs.items()},
                  "aggregator_version": AGGREGATOR_VERSION}
        prov.write_json(out_dir / "aggregate_summary.json",
                        {**report, **prov.provenance_block()})
        if progress:
            print(f"E6 aggregation -> {out_dir}\n  {reason}")
        return report

    usable, audit = ok_rows(raw, max_failure_rate=float(
        analysis.get("max_failure_rate", 0.02)), allow_high_failure=allow_high_failure)
    assert_comparable(usable)

    # `08` §2.1 — contrasts see the *unfiltered* e6a rows, because `e6a_c5` pairs the
    # in-domain corpus against the cross-domain one and the default prefix filter drops
    # `e1_100x40`. Everything else keeps reading the corpus-selected frame, so no existing
    # table or figure moves.
    e6a_all = usable[usable["experiment_id"] == experiment]
    e6a = select_corpus(e6a_all, corpus)
    e6b = usable[usable["experiment_id"] == e6b_experiment]

    matched = build_matched_loss(e6a, reference_condition=reference_condition,
                                 max_gap=float(analysis.get("matched_loss_max_gap_nats",
                                                            0.05)))
    matched_steps = _matched_steps(matched)
    contrasts = build_contrasts(e6a_all, prereg, matched_steps, corpus_override=corpus,
                                matched_table=matched)
    table2 = build_table2(e6a)
    drift = build_e6b_drift(e6b)
    early = build_e6b_early_warning(drift, prereg)
    factorial = build_e6b_factorial(e6b, prereg)
    table3 = build_table3(e6b, prereg)
    decision = build_go_no_go(e6a_all, prereg, matched_steps, corpus_override=corpus,
                              matched_table=matched)
    e6b_decision = build_go_no_go(e6b, prereg, {}, experiment=e6b_experiment,
                                  criteria_key="e6b_go_no_go")

    no_e6b = (
        "no e6b runs found; WP6 (train_sentiment_adaptation.py) is not implemented"
        if e6b_experiment == DEFAULT_E6B_EXPERIMENT else
        f"no {e6b_experiment} runs found; no sentiment-adaptation rows matched the "
        "selected --e6b-experiment")
    outputs = {
        "e6a_contrasts.csv": _write(contrasts, out_dir / "e6a_contrasts.csv",
                                    reason="no e6a rows"),
        "e6a_matched_loss.csv": _write(matched, out_dir / "e6a_matched_loss.csv",
                                       reason="no e6a rows with validation_ce"),
        "table2_inheritance_components.csv": _write(
            table2, out_dir / "table2_inheritance_components.csv", reason="no e6a rows"),
        "e6b_drift.csv": _write(drift, out_dir / "e6b_drift.csv", reason=no_e6b),
        "e6b_early_warning.csv": _write(early, out_dir / "e6b_early_warning.csv",
                                        reason=no_e6b),
        "e6b_factorial.csv": _write(factorial, out_dir / "e6b_factorial.csv",
                                    reason=no_e6b),
        "table3_clean_vs_corrupt.csv": _write(table3,
                                              out_dir / "table3_clean_vs_corrupt.csv",
                                              reason=no_e6b),
    }
    prov.write_json(out_dir / "go_no_go.json", decision)
    outputs["go_no_go.json"] = out_dir / "go_no_go.json"
    prov.write_json(out_dir / "e6b_go_no_go.json", e6b_decision)
    outputs["e6b_go_no_go.json"] = out_dir / "e6b_go_no_go.json"

    figure_paths: Dict[str, Optional[str]] = {}
    if figures:
        figure_dir = out_dir / "figures"
        for name, builder, argument in (
                ("fig2_trajectories.pdf", plot_trajectories, e6a),
                ("fig3_matched_loss.pdf", plot_matched_loss, matched),
                ("fig4_drift.pdf", plot_drift, drift)):
            try:
                path = builder(argument, figure_dir / name)
            except Exception as exc:  # a plotting failure must not lose the tables
                figure_paths[name] = f"failed: {type(exc).__name__}: {exc}"
                continue
            figure_paths[name] = str(path) if path else "skipped: no data"

    report = {
        "results_root": str(results),
        "n_rows_read": int(len(raw)),
        "n_rows_used": int(len(usable)),
        "n_rows_excluded": int(len(raw) - len(usable)),
        "exclusions": audit,
        "allow_high_failure": bool(allow_high_failure),
        "experiment": experiment,
        "e6b_experiment": e6b_experiment,
        "n_e6a_rows": int(len(e6a)),
        "n_e6b_rows": int(len(e6b)),
        "reference_condition": reference_condition,
        "preregistration": str(prereg_path),
        "preregistration_version": prereg.get("version"),
        "pending_preregistration": decision["pending_preregistration"],
        "pending_decisions": decision["pending_decisions"],
        "decision": decision["decision"],
        "e6b_decision": e6b_decision["decision"],
        "outputs": {k: str(v) for k, v in outputs.items()},
        "figures": figure_paths,
        "aggregator_version": AGGREGATOR_VERSION,
    }
    prov.write_json(out_dir / "aggregate_summary.json",
                    {**report, **prov.provenance_block()})
    if progress:
        print(f"E6 aggregation -> {out_dir}")
        print(f"  rows read={report['n_rows_read']} used={report['n_rows_used']} "
              f"excluded={report['n_rows_excluded']}")
        print(f"  e6a={report['n_e6a_rows']} e6b={report['n_e6b_rows']} "
              f"decision={report['decision']}")
        if decision["pending_preregistration"]:
            print(f"  pending criteria: {decision['pending_preregistration']}")
        if decision["pending_decisions"]:
            print("  pending decisions: "
                  + ", ".join(f"{d['location']}({d['status']})"
                              for d in decision["pending_decisions"])
                  + " — resolve them per new_design_plans/DECISIONS_REQUIRED.md before "
                    "reading go_no_go.json as a decision")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--results",
                        default=str(_REPO / "transformation_inheritance" / "results"))
    parser.add_argument("--preregistration", default=str(DEFAULT_PREREG))
    parser.add_argument("--corpus", default=None,
                        help="restrict contrasts to one corpus_id "
                             "(default: the in-domain corpus)")
    parser.add_argument("--reference-condition", default="D0",
                        help="the condition whose final CE matched-loss targets")
    parser.add_argument("--allow-high-failure", action="store_true",
                        help="include rows above the 2%% failure rate (05 §7.2)")
    parser.add_argument("--experiment", default=DEFAULT_EXPERIMENT,
                        help="the E6A arm to aggregate (default: %(default)s). Rows from "
                             "any other experiment_id are excluded from its contrasts, "
                             "matched loss and calibrated nulls")
    parser.add_argument("--e6b-experiment", default=DEFAULT_E6B_EXPERIMENT,
                        help="the E6B sentiment-adaptation arm to aggregate (default: "
                             "%(default)s). Use a distinct experiment_id for a model-scale "
                             "extension so it cannot pool with the original arm")
    parser.add_argument("--no-figures", dest="figures", action="store_false",
                        default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = aggregate(Path(args.results), prereg_path=Path(args.preregistration),
                       corpus=args.corpus, allow_high_failure=args.allow_high_failure,
                       reference_condition=args.reference_condition,
                       experiment=args.experiment,
                       e6b_experiment=args.e6b_experiment,
                       figures=args.figures)
    # `incomplete` is not a failure of this script — it is the honest state of a
    # pre-registration that has not been fully transcribed, or of a run still in progress.
    return 0 if report.get("n_rows_used") else 1


if __name__ == "__main__":
    raise SystemExit(main())
