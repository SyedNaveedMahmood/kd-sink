# -*- coding: utf-8 -*-
"""aggregate_crosslingual.py — E7 contrasts, claim gate and go/no-go (WP11).

``04_MODULE_SPEC_e7_crosslingual.md`` §6. Reads every ``patching_per_example.csv`` under a
results root (and, optionally, WP9's ``retrieval_by_object.csv``) and produces the §6.3
outputs.

    python crosslingual_semantics/aggregate_crosslingual.py \\
      --results crosslingual_semantics/results/patching \\
      --retrieval crosslingual_semantics/results/retrieval

**No threshold, criterion, language tier or interpretation row lives in this file.** They
are all read from ``configs/e7_preregistration.yaml``; entries the design document has not
supplied carry a ``PENDING_DESIGN_*`` status and are emitted with ``observed: null``,
``met: null`` and their reason, forcing ``supported: null`` and ``decision: "incomplete"``.
``tests/test_e7_aggregate_contracts.py`` asserts that changing a threshold in that YAML
changes the verdict, which is what proves nothing is hard-coded here.

Four properties that are easy to get wrong, and are therefore explicit
----------------------------------------------------------------------
* **Paired within target example** (``04`` §6.1). Treatment and control are joined on
  (model, language, semantic id, object, layer, norm) *before* differencing, so example
  difficulty cancels and an example missing either side contributes to neither.
* **Invalid units are excluded.** A run unit whose identity control moved the margin past
  the tolerance is not a noisy unit, it is a broken one (``04`` §5.4). Its rows are read,
  counted, and then dropped from every contrast.
* **Failure rate gates the contrast** (``05`` §7.2). Above 2% in any condition a contrast
  is refused, not silently computed, unless ``--allow-high-failure`` is passed — and the
  refusal is recorded in the row.
* **Pre-selected contrasts and exploratory sweeps go to different files** (``04`` §6.2).
  ``patching_contrasts.csv`` holds only what the pre-registration names;
  ``exploratory_layer_sweep.csv`` holds the layer sweep, and the two never mix.

Statistics follow ``04`` §6.2: hierarchical bootstrap resampling semantic ids then
languages, a two-sided p from *that same resampling* (never a second inference framework
next to the interval), and BH correction within each pre-registered family.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import inheritance_metrics as im  # noqa: E402
import provenance as prov  # noqa: E402

AGGREGATOR_VERSION = "aggregate_crosslingual_v2"

DEFAULT_PREREG = Path(__file__).resolve().parent / "configs" / "e7_preregistration.yaml"

#: The only two decision rules either aggregator honours. Anything else — a typo, a
#: missing key on an otherwise resolved entry — is refused rather than defaulted, because
#: a silently defaulted rule resolves a pre-registration decision in code (`05` §6).
VALID_COMBINE: Tuple[str, ...] = ("any", "all")

#: The join that makes a contrast paired within target example (``04`` §6.1).
PAIR_KEYS: Tuple[str, ...] = ("model_tag", "target_language", "semantic_id",
                              "patch_object", "patch_layer", "norm_condition")

CONTRAST_COLUMNS: Tuple[str, ...] = (
    "id", "text", "kind", "family", "status", "reason", "metric",
    "treatment_source", "control_source", "norm_condition", "scope",
    "n_pairs", "n_languages", "n_model_sizes", "n_semantic_ids",
    "value_a", "value_b", "absolute_difference", "relative_difference",
    "ci_lo", "ci_hi", "ci_uncertainty_kind", "p_value", "min_attainable_p",
    "q_value", "bh_rejected", "bh_family_size", "n_languages_same_direction",
    "n_model_sizes_same_direction", "failure_rate", "excluded_reason",
    "preregistration_source", "aggregator_version", "git_sha",
)

SWEEP_COLUMNS: Tuple[str, ...] = (
    "model_tag", "model_variant", "model_size", "patch_object", "patch_layer",
    "norm_condition", "treatment_source", "control_source", "stage", "partition",
    "n_pairs", "n_languages", "effect", "ci_lo", "ci_hi", "p_value",
    "note", "aggregator_version", "git_sha",
)

FAILURE_COLUMNS: Tuple[str, ...] = (
    "model_tag", "target_language", "patch_object", "patch_layer", "norm_condition",
    "source_condition", "stage", "partition", "status", "unit_status", "n_rows",
    "n_rows_in_condition", "failure_rate", "example_warning",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Inputs
# ═══════════════════════════════════════════════════════════════════════════════


def load_preregistration(path: Path = DEFAULT_PREREG) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    payload.setdefault("contrasts", [])
    payload.setdefault("go_no_go", [])
    payload.setdefault("claim_gate", {})
    payload.setdefault("interpretation_matrix", {"status": "PENDING", "rows": []})
    payload.setdefault("analysis", {})
    payload["_path"] = str(path)
    return payload


def model_size_of(model_name: str) -> str:
    """Parameter scale parsed from the model id — ``"Qwen/Qwen2.5-1.5B"`` -> ``"1.5B"``.

    The §10.11 claim gate counts *model sizes*, not checkpoints, so base and instruct at the
    same scale are one size. An unparseable name returns the name itself rather than a
    guess, which makes it visible in the artefact instead of quietly collapsing two sizes.
    """
    match = re.search(r"(\d+(?:\.\d+)?)\s*B\b", str(model_name), flags=re.IGNORECASE)
    return f"{match.group(1)}B" if match else str(model_name)


def load_all_rows(results: Path):
    """Concatenate every ``patching_per_example.csv`` under ``results``."""
    import pandas as pd

    frames = []
    for path in sorted(Path(results).rglob("patching_per_example.csv")):
        frame = pd.read_csv(path, encoding="utf-8")
        frame["source_csv"] = str(path)
        frames.append(frame)
    if not frames:
        return pd.DataFrame()
    frame = pd.concat(frames, ignore_index=True)
    return normalise_rows(frame)


def normalise_rows(frame):
    """Fill the optional columns a hand-made or older CSV may lack, and derive model size.

    ``05`` §5 permits adding columns, which means a CSV written before a column existed is
    still a valid input. Defaulting the optional ones here — rather than requiring them —
    keeps an old run readable instead of turning a schema addition into a crash.
    """
    if frame.empty:
        return frame
    if "unit_status" not in frame.columns:
        frame["unit_status"] = "ok"
    for column, default in (("warning", ""), ("stage", ""), ("partition", ""),
                            ("model_variant", ""), ("model_revision", None)):
        if column not in frame.columns:
            frame[column] = default
    frame["model_size"] = frame["model"].map(model_size_of)
    return frame


def load_retrieval(path: Optional[Path]):
    """WP9's ``retrieval_by_object.csv`` — the input for go/no-go criteria e7_1 and e7_2."""
    import pandas as pd

    if path is None:
        return None
    path = Path(path)
    candidates = ([path] if path.is_file()
                  else sorted(path.rglob("retrieval_by_object.csv")))
    frames = [pd.read_csv(candidate, encoding="utf-8") for candidate in candidates]
    return pd.concat(frames, ignore_index=True) if frames else None


# ═══════════════════════════════════════════════════════════════════════════════
# Failures and exclusions
# ═══════════════════════════════════════════════════════════════════════════════


def failure_report(frame):
    """One row per (condition, non-ok status): ``failures.csv`` (``04`` §6.3)."""
    import pandas as pd

    if frame.empty:
        return pd.DataFrame(columns=list(FAILURE_COLUMNS)), {}

    frame = normalise_rows(frame.copy())
    condition = ["model_tag", "target_language", "patch_object", "patch_layer",
                 "norm_condition", "source_condition", "stage", "partition"]
    totals = frame.groupby(condition, dropna=False).size().rename("n_rows_in_condition")

    bad = frame[(frame["status"] != "ok") | (frame["unit_status"] != "ok")]
    if bad.empty:
        rates = {key: 0.0 for key in totals.index}
        return pd.DataFrame(columns=list(FAILURE_COLUMNS)), rates

    counted = (bad.groupby(condition + ["status", "unit_status"], dropna=False)
               .agg(n_rows=("semantic_id", "size"),
                    example_warning=("warning", "first"))
               .reset_index()
               .join(totals, on=condition))
    counted["failure_rate"] = counted["n_rows"] / counted["n_rows_in_condition"]

    bad_totals = bad.groupby(condition, dropna=False).size()
    rates = {key: float(bad_totals.get(key, 0)) / float(totals[key]) for key in
             totals.index}
    return counted.reindex(columns=list(FAILURE_COLUMNS)), rates


def usable_rows(frame):
    """Rows that may enter a contrast: ``status == "ok"`` **and** a valid run unit."""
    if frame.empty:
        return frame
    return frame[(frame["status"] == "ok") & (frame["unit_status"] == "ok")].copy()


def contrast_failure_rate(raw, *, objects: Sequence[str], norm_condition: str,
                          sources: Sequence[str]) -> float:
    """Failure rate over exactly the rows a contrast would consume (``05`` §7.2).

    Scoped rather than global on purpose: one broken language-object-layer unit must not
    refuse a contrast that never touches it, and equally must not be diluted away by the
    thousands of healthy rows belonging to other conditions.
    """
    if raw is None or raw.empty:
        return 0.0
    subset = raw[raw["patch_object"].isin(list(objects))
                 & (raw["norm_condition"] == norm_condition)
                 & raw["source_condition"].isin(list(sources))]
    if subset.empty:
        return 0.0
    bad = (subset["status"] != "ok") | (subset["unit_status"] != "ok")
    return float(bad.mean())


# ═══════════════════════════════════════════════════════════════════════════════
# Pairing
# ═══════════════════════════════════════════════════════════════════════════════


def paired_effects(frame, *, treatment: str, control: str, objects: Sequence[str],
                   norm_condition: str, metric: str = "margin_delta"):
    """The parallel-vs-control effect, paired within target example.

    Returns one row per (model, language, semantic id, object, layer) with the difference
    ``treatment[metric] - control[metric]``. The inner join is what makes it paired: an
    example scored under only one of the two conditions drops out of both sides rather than
    contributing an unmatched value to a mean.
    """
    import pandas as pd

    if frame.empty:
        return pd.DataFrame(columns=list(PAIR_KEYS) + ["value"])
    subset = frame[frame["patch_object"].isin(list(objects))
                   & (frame["norm_condition"] == norm_condition)]
    if subset.empty:
        return pd.DataFrame(columns=list(PAIR_KEYS) + ["value"])

    keys = list(PAIR_KEYS)
    extra = [c for c in ("model_variant", "model_size", "stage", "partition")
             if c in subset.columns]
    a = (subset[subset["source_condition"] == treatment]
         .set_index(keys)[[metric] + extra])
    b = subset[subset["source_condition"] == control].set_index(keys)[[metric]]
    joined = a.join(b, how="inner", lsuffix="_a", rsuffix="_b")
    if joined.empty:
        return pd.DataFrame(columns=keys + ["value"])
    joined["value"] = joined[f"{metric}_a"] - joined[f"{metric}_b"]
    return joined.reset_index().dropna(subset=["value"])


def _bootstrap(values, analysis: Dict[str, Any], *, group_cols: Sequence[str]
               ) -> Dict[str, Any]:
    """Hierarchical bootstrap + a two-sided p from the same replicates (``04`` §6.2)."""
    import pandas as pd

    usable = [c for c in group_cols if c in values.columns]
    if len(usable) < 2:
        # Fall back to a flat bootstrap rather than silently pretending the data is
        # clustered when only one clustering column survives.
        lo, hi = im.bootstrap_ci(values["value"].to_numpy(dtype=float),
                                 n_boot=int(analysis.get("bootstrap_n", 10000)),
                                 alpha=float(analysis.get("bootstrap_alpha", 0.05)),
                                 seed=0)
        return {"mean": float(values["value"].mean()), "ci_lo": float(lo),
                "ci_hi": float(hi), "p_value": float("nan"),
                "min_attainable_p": float("nan"),
                "ci_uncertainty_kind": "sampling_over_examples_flat_bootstrap"}

    result = im.hierarchical_bootstrap(
        values, usable[:2], "value",
        n_boot=int(analysis.get("bootstrap_n", 10000)), seed=0, return_draws=True)
    p = im.bootstrap_two_sided_p(result["draws"])
    return {"mean": result["mean"], "ci_lo": result["ci_lo"], "ci_hi": result["ci_hi"],
            "p_value": p["p_value"], "min_attainable_p": p["min_attainable_p"],
            "ci_uncertainty_kind": "hierarchical_bootstrap_over_"
                                   + "_then_".join(usable[:2])}


def _direction_counts(values, *, mean: float) -> Dict[str, int]:
    """How many languages / model sizes move the same way as the pooled mean (§10.11)."""
    counts = {"n_languages_same_direction": 0, "n_model_sizes_same_direction": 0}
    if not np.isfinite(mean) or mean == 0.0:
        return counts
    sign = np.sign(mean)
    for column, key in (("target_language", "n_languages_same_direction"),
                        ("model_size", "n_model_sizes_same_direction")):
        if column not in values.columns:
            continue
        per_group = values.groupby(column)["value"].mean()
        counts[key] = int((np.sign(per_group) == sign).sum())
    return counts


# ═══════════════════════════════════════════════════════════════════════════════
# Contrasts
# ═══════════════════════════════════════════════════════════════════════════════


def _pending(entry: Dict[str, Any], reason: str = "") -> Dict[str, Any]:
    """A contrast the pre-registration has not supplied: reported, never estimated."""
    return {
        "id": entry.get("id"), "text": entry.get("text", ""),
        "kind": entry.get("kind", ""), "family": entry.get("family", ""),
        "status": entry.get("status", "pending"),
        "reason": reason or str(entry.get("reason", "")).strip(),
        "metric": entry.get("metric", ""),
        "treatment_source": entry.get("treatment_source"),
        "control_source": entry.get("control_source"),
        "norm_condition": entry.get("norm_condition"),
        "scope": "", "n_pairs": 0, "n_languages": 0, "n_model_sizes": 0,
        "n_semantic_ids": 0, "value_a": None, "value_b": None,
        "absolute_difference": None, "relative_difference": None,
        "ci_lo": None, "ci_hi": None, "ci_uncertainty_kind": "",
        "p_value": None, "min_attainable_p": None,
        "n_languages_same_direction": 0, "n_model_sizes_same_direction": 0,
        "failure_rate": None, "excluded_reason": "",
        "preregistration_source": entry.get("source", ""),
        "aggregator_version": AGGREGATOR_VERSION, "git_sha": prov.git_sha(),
    }


def _summarise(entry: Dict[str, Any], values, analysis: Dict[str, Any], *,
               scope: str, value_a: Optional[float] = None,
               value_b: Optional[float] = None,
               excluded_reason: str = "") -> Dict[str, Any]:
    row = _pending(entry, reason="")
    row["status"] = "ok"
    row["scope"] = scope
    row["excluded_reason"] = excluded_reason
    if excluded_reason or values.empty:
        row["status"] = "excluded" if excluded_reason else "no_data"
        row["reason"] = excluded_reason or "no paired rows survived the join"
        return row

    stats = _bootstrap(values, analysis,
                       group_cols=analysis.get("bootstrap_group_cols",
                                               ["semantic_id", "target_language"]))
    mean = stats["mean"]
    row.update({
        "n_pairs": int(len(values)),
        "n_languages": int(values["target_language"].nunique())
        if "target_language" in values else 0,
        "n_model_sizes": int(values["model_size"].nunique())
        if "model_size" in values else 0,
        "n_semantic_ids": int(values["semantic_id"].nunique())
        if "semantic_id" in values else 0,
        "value_a": value_a, "value_b": value_b,
        "absolute_difference": float(mean),
        # A relative difference is only meaningful against a non-trivial baseline; below
        # this it is reported as absent rather than as a huge ratio.
        "relative_difference": (float(mean / abs(value_b))
                                if value_b not in (None, 0)
                                and abs(value_b) > 1e-12 else None),
        "ci_lo": stats["ci_lo"], "ci_hi": stats["ci_hi"],
        "ci_uncertainty_kind": stats["ci_uncertainty_kind"],
        "p_value": stats["p_value"], "min_attainable_p": stats["min_attainable_p"],
        **_direction_counts(values, mean=mean),
    })
    return row


def evaluate_contrast(frame, entry: Dict[str, Any], analysis: Dict[str, Any], *,
                      raw=None, allow_high_failure: bool = False) -> Dict[str, Any]:
    """One pre-registered contrast. ``PENDING_*`` entries short-circuit, never estimate."""
    if str(entry.get("status", "")).startswith("PENDING"):
        return _pending(entry)

    kind = entry.get("kind")
    treatment = entry.get("treatment_source", "parallel_en")
    control = entry.get("control_source", "same_label_en")
    norm = entry.get("norm_condition", "direct")
    metric = entry.get("metric", "margin_delta")
    max_rate = float(analysis.get("max_failure_rate", 0.02))

    def _effects(objects):
        return paired_effects(frame, treatment=treatment, control=control,
                              objects=objects, norm_condition=norm, metric=metric)

    scope_objects = (list(entry.get("objects") or [])
                     + list(entry.get("objects_a") or [])
                     + list(entry.get("objects_b") or []))
    # A one-group contrast consumes only its own source's rows; scoping the failure rate to
    # the treatment/control pair would measure a condition it never reads (`05` §7.2).
    scope_sources = ([str(entry.get("source_condition"))] if kind == "source_level"
                     else [treatment, control])
    worst = contrast_failure_rate(raw, objects=scope_objects, norm_condition=norm,
                                  sources=scope_sources)
    excluded = ""
    if worst > max_rate and not allow_high_failure:
        excluded = (f"this contrast's rows have a {worst:.3%} failure rate, above the "
                    f"{max_rate:.0%} bar in `05` §7.2; pass --allow-high-failure to "
                    "compute it anyway")

    if kind == "source_level":
        # `08` §8.1 — a one-group mean. `value_b` and `relative_difference` stay null:
        # there is no second arm, and inventing a denominator would make an absolute
        # effect look like a ratio.
        source = entry.get("source_condition")
        if not source:
            return _pending(entry, reason="a source_level contrast names no "
                                          "source_condition")
        values = source_level_effects(frame, source_condition=str(source),
                                      objects=entry.get("objects") or [],
                                      norm_condition=norm, metric=metric)
        row = _summarise(entry, values, analysis,
                         scope=f"source={source} objects={entry.get('objects')}",
                         excluded_reason=excluded)
        row["failure_rate"] = worst
        row["treatment_source"] = str(source)
        row["control_source"] = None
        row["relative_difference"] = None
        return row

    if kind == "source_condition":
        values = _effects(entry.get("objects") or [])
        row = _summarise(entry, values, analysis,
                         scope=f"objects={entry.get('objects')}",
                         excluded_reason=excluded)
        row["failure_rate"] = worst
        return row

    if kind == "object_pair":
        a = _effects(entry.get("objects_a") or [])
        b = _effects(entry.get("objects_b") or [])
        values = _difference_of_effects(a, b, on=["model_tag", "target_language",
                                                  "semantic_id", "patch_layer"])
        row = _summarise(entry, values, analysis,
                         scope=f"{entry.get('objects_a')} minus {entry.get('objects_b')}",
                         value_a=float(a["value"].mean()) if not a.empty else None,
                         value_b=float(b["value"].mean()) if not b.empty else None,
                         excluded_reason=excluded)
        row["failure_rate"] = worst
        return row

    if kind in ("model_pair", "language_group"):
        column = "model_variant" if kind == "model_pair" else "target_language"
        group_a = ([entry.get("group_a")] if kind == "model_pair"
                   else list(entry.get("languages_a") or []))
        group_b = ([entry.get("group_b")] if kind == "model_pair"
                   else list(entry.get("languages_b") or []))
        if not group_a or not group_b or group_a == [None]:
            return _pending(entry, reason=f"{kind} contrast names no groups to compare")

        effects = _effects(entry.get("objects") or [])
        if effects.empty or column not in effects.columns:
            row = _summarise(entry, effects.iloc[0:0], analysis, scope=f"{kind}")
            row["failure_rate"] = worst
            return row
        a = effects[effects[column].isin(group_a)]
        b = effects[effects[column].isin(group_b)]
        # Pair on everything except the grouping column itself.
        on = [c for c in ("semantic_id", "patch_object", "patch_layer") if c in
              effects.columns]
        values = _difference_of_effects(a, b, on=on)
        row = _summarise(entry, values, analysis,
                         scope=f"{group_a} minus {group_b}",
                         value_a=float(a["value"].mean()) if not a.empty else None,
                         value_b=float(b["value"].mean()) if not b.empty else None,
                         excluded_reason=excluded)
        row["failure_rate"] = worst
        return row

    return _pending(entry, reason=f"unknown contrast kind {kind!r}")


def source_level_effects(frame, *, source_condition: str, objects: Sequence[str],
                         norm_condition: str, metric: str = "margin_delta"):
    """One group's own causal effect against the unpatched baseline (``08`` §8.1).

    Not a paired difference: ``margin_delta`` is already ``patched − baseline``, so the
    mean of that column for one source condition *is* "what this source does compared to
    no patch at all". §10.12 rows 1 and 3 turn on whether K0 transfers for an **unrelated**
    sentence, which is not expressible as any parallel-versus-control contrast — there is
    no second arm to subtract.
    """
    import pandas as pd

    keys = list(PAIR_KEYS)
    if frame.empty:
        return pd.DataFrame(columns=keys + ["value"])
    subset = frame[frame["patch_object"].isin(list(objects))
                   & (frame["norm_condition"] == norm_condition)
                   & (frame["source_condition"] == source_condition)]
    if subset.empty:
        return pd.DataFrame(columns=keys + ["value"])
    carry = [c for c in ("model_variant", "model_size", "stage", "partition")
             if c in subset.columns]
    values = subset[keys + carry + [metric]].copy()
    values["value"] = values[metric]
    return values.dropna(subset=["value"])


def _difference_of_effects(a, b, *, on: Sequence[str]):
    """Mean effect of ``a`` minus mean effect of ``b``, joined on ``on``.

    Both sides are collapsed to one value per join key first, so a group with more objects
    or more languages does not get more weight than the other side.
    """
    import pandas as pd

    on = [c for c in on if c in a.columns and c in b.columns]
    if a.empty or b.empty or not on:
        return pd.DataFrame(columns=list(on) + ["value"])
    carry = [c for c in ("model_tag", "model_variant", "model_size", "target_language",
                         "semantic_id") if c in a.columns and c not in on]
    left = a.groupby(on, dropna=False).agg(
        value_a=("value", "mean"), **{c: (c, "first") for c in carry}).reset_index()
    right = b.groupby(on, dropna=False)["value"].mean().rename("value_b").reset_index()
    joined = left.merge(right, on=on, how="inner")
    if joined.empty:
        return pd.DataFrame(columns=list(on) + ["value"])
    joined["value"] = joined["value_a"] - joined["value_b"]
    return joined.dropna(subset=["value"])


def build_contrasts(frame, prereg: Dict[str, Any], *, raw=None,
                    allow_high_failure: bool = False):
    """Every pre-registered contrast, BH-corrected **within family** (``04`` §6.2)."""
    import pandas as pd

    analysis = prereg.get("analysis", {})
    rows = [evaluate_contrast(frame, entry, analysis, raw=raw,
                              allow_high_failure=allow_high_failure)
            for entry in prereg.get("contrasts", [])]
    result = pd.DataFrame(rows).reindex(columns=list(CONTRAST_COLUMNS))
    if result.empty:
        return result

    result["q_value"] = None
    result["bh_rejected"] = None
    result["bh_family_size"] = 0
    alpha = float(analysis.get("bh_alpha", 0.05))
    for family, group in result.groupby("family", dropna=False):
        testable = group[group["p_value"].notna()]
        if testable.empty:
            continue
        rejected, qvalues = im.bh_correct(testable["p_value"].tolist(), alpha=alpha)
        result.loc[testable.index, "q_value"] = list(qvalues)
        result.loc[testable.index, "bh_rejected"] = list(rejected)
        result.loc[group.index, "bh_family_size"] = int(len(testable))
    return result


def build_layer_sweep(frame, prereg: Dict[str, Any]):
    """Every layer's effect — **exploratory**, and therefore its own file (``04`` §6.2)."""
    import pandas as pd

    analysis = prereg.get("analysis", {})
    if frame.empty:
        return pd.DataFrame(columns=list(SWEEP_COLUMNS))

    treatment, control, norm = "parallel_en", "same_label_en", "direct"
    rows: List[Dict[str, Any]] = []
    for obj in sorted(frame["patch_object"].dropna().unique()):
        effects = paired_effects(frame, treatment=treatment, control=control,
                                 objects=[obj], norm_condition=norm)
        if effects.empty:
            continue
        for (tag, layer), group in effects.groupby(["model_tag", "patch_layer"]):
            stats = _bootstrap(group, analysis,
                               group_cols=analysis.get("bootstrap_group_cols",
                                                       ["semantic_id",
                                                        "target_language"]))
            rows.append({
                "model_tag": tag,
                "model_variant": group["model_variant"].iloc[0]
                if "model_variant" in group else "",
                "model_size": group["model_size"].iloc[0]
                if "model_size" in group else "",
                "patch_object": obj, "patch_layer": int(layer),
                "norm_condition": norm, "treatment_source": treatment,
                "control_source": control,
                "stage": group["stage"].iloc[0] if "stage" in group else "",
                "partition": group["partition"].iloc[0] if "partition" in group else "",
                "n_pairs": int(len(group)),
                "n_languages": int(group["target_language"].nunique())
                if "target_language" in group else 0,
                "effect": stats["mean"], "ci_lo": stats["ci_lo"],
                "ci_hi": stats["ci_hi"], "p_value": stats["p_value"],
                "note": "EXPLORATORY — not a pre-registered contrast; no BH correction "
                        "is applied across this sweep and no claim rests on it",
                "aggregator_version": AGGREGATOR_VERSION, "git_sha": prov.git_sha(),
            })
    return pd.DataFrame(rows).reindex(columns=list(SWEEP_COLUMNS))


def build_table4(contrast_frame):
    """Design §19 Table 4: the pre-registered effects, one line each."""
    import pandas as pd

    if contrast_frame.empty:
        return pd.DataFrame(columns=["id", "text", "status", "absolute_difference",
                                     "ci_lo", "ci_hi", "p_value", "q_value",
                                     "n_languages_same_direction",
                                     "n_model_sizes_same_direction", "n_pairs"])
    return contrast_frame[["id", "text", "status", "absolute_difference", "ci_lo",
                           "ci_hi", "p_value", "q_value",
                           "n_languages_same_direction",
                           "n_model_sizes_same_direction", "n_pairs"]].copy()


# ═══════════════════════════════════════════════════════════════════════════════
# Claim gate, go/no-go, interpretation matrix
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_claim_gate(contrast_frame, prereg: Dict[str, Any]) -> Dict[str, Any]:
    """``04`` §6.4 / design §10.11 — quoted in full in the spec pack, so it is mechanical."""
    gate = prereg.get("claim_gate") or {}
    if not gate:
        return {"id": "e7_claim_gate", "status": "PENDING", "met": None,
                "reason": "the pre-registration declares no claim gate"}

    ids = list(gate.get("contrasts") or [])
    rows = contrast_frame[contrast_frame["id"].isin(ids)] if not contrast_frame.empty \
        else contrast_frame
    per_contrast = []
    met_all = True
    for contrast_id in ids:
        row = rows[rows["id"] == contrast_id]
        if row.empty or row.iloc[0]["status"] != "ok":
            per_contrast.append({"id": contrast_id, "met": None,
                                 "reason": "contrast not computed"})
            met_all = False
            continue
        record = row.iloc[0]
        languages = int(record["n_languages_same_direction"] or 0)
        sizes = int(record["n_model_sizes_same_direction"] or 0)
        met = (sizes >= int(gate.get("min_model_sizes", 2))
               and languages >= int(gate.get("min_languages", 5)))
        met_all = met_all and met
        per_contrast.append({
            "id": contrast_id, "met": bool(met),
            "n_languages_same_direction": languages,
            "n_model_sizes_same_direction": sizes,
            "min_languages": int(gate.get("min_languages", 5)),
            "min_model_sizes": int(gate.get("min_model_sizes", 2)),
        })
    # `08` §9 — the gate's two halves are also §18 criteria 4 and 5. They are decomposed
    # here so `evaluate_criterion` can *read* a verdict instead of recomputing one; two
    # independent computations of the same quantity is how a criterion and its own gate end
    # up contradicting each other.
    components: Dict[str, Any] = {}
    for field, column, label in (
            ("min_model_sizes", "n_model_sizes_same_direction",
             "the direction replicates in at least this many model sizes"),
            ("min_languages", "n_languages_same_direction",
             "the result holds in at least this many target languages")):
        threshold = int(gate.get(field, 0))
        observed = [entry.get(column) for entry in per_contrast
                    if entry.get(column) is not None]
        components[field] = {
            "text": label, "threshold": threshold,
            "observed": min(observed) if observed else None,
            "per_contrast": {entry["id"]: entry.get(column) for entry in per_contrast},
            "met": (bool(min(observed) >= threshold) if observed else None),
        }

    return {"id": gate.get("id", "e7_claim_gate"), "text": gate.get("text", ""),
            "status": "ok", "met": bool(met_all) if per_contrast else None,
            "per_contrast": per_contrast, "components": components,
            "satisfies_criteria": list(gate.get("satisfies_criteria") or []),
            "source": gate.get("source", "")}


def retrieval_observation(retrieval, *, objects: Sequence[str],
                          alignment: Optional[str], metric: str
                          ) -> Tuple[Optional[float], str]:
    """``max`` over layers of ``metric`` for ``objects`` at ``alignment`` (``04`` §4).

    Extracted so criterion ``e7_1`` and interpretation row 5 read the *same* number
    (``08`` §11). If they disagreed, the go/no-go gate could pass on a retrieval level the
    matrix simultaneously called low, and nothing in the artefact would reveal it.
    Returns ``(value, reason)``; the value is ``None`` whenever the reason is non-empty.
    """
    if retrieval is None or getattr(retrieval, "empty", True):
        return None, ("no retrieval_by_object.csv supplied; run WP9's "
                      "evaluate_crosslingual_retrieval.py and pass --retrieval")
    subset = retrieval[retrieval["object"].isin(list(objects))]
    if "alignment" in subset.columns and alignment:
        subset = subset[subset["alignment"] == alignment]
    if metric not in subset.columns or subset.empty:
        return None, (f"retrieval table has no usable {metric!r} rows for objects "
                      f"{list(objects)}")
    values = subset[metric].dropna()
    if values.empty:
        return None, (f"every {metric!r} value is missing (a position-0 object with no "
                      "matched middle-token control produces this)")
    return float(values.max()), ""


def evaluate_criterion(entry: Dict[str, Any], *, retrieval, contrast_frame,
                       claim_gate: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One design §18 criterion. ``PENDING_*`` reports ``met: null`` and its reason."""
    base = {"id": entry.get("id"), "text": entry.get("text", ""),
            "observed": None, "threshold": entry.get("threshold"),
            "direction": entry.get("direction"), "met": None,
            "source": entry.get("source", "")}
    if str(entry.get("status", "")).startswith("PENDING"):
        return {**base, "status": entry.get("status"),
                "reason": str(entry.get("reason", "")).strip()}

    if entry.get("evaluation") == "claim_gate_component":
        # `08` §9 — read the gate's verdict, never recompute it.
        field = str(entry.get("claim_gate_field", ""))
        component = ((claim_gate or {}).get("components") or {}).get(field)
        if component is None:
            return {**base, "status": "no_input",
                    "reason": f"the claim gate exposes no {field!r} component; the "
                              "criterion is a view onto the gate and cannot be computed "
                              "independently (08 §9)"}
        return {**base, "status": "ok" if component["met"] is not None else "no_input",
                "evaluation": "claim_gate_component", "claim_gate_field": field,
                "observed": component["observed"], "threshold": component["threshold"],
                "direction": "greater_equal", "met": component["met"],
                "reason": ("" if component["met"] is not None
                           else "the claim gate's contrasts were not computed"),
                "note": "verdict read from the claim gate; excluded from n_met so it is "
                        "not counted twice (08 §9)"}

    source = entry.get("input", "patching")
    if source == "retrieval":
        observed, reason = retrieval_observation(
            retrieval, objects=entry.get("objects") or [],
            alignment=entry.get("alignment"), metric=entry.get("metric"))
        if observed is None:
            return {**base, "status": "no_input", "reason": reason}
    else:
        rows = contrast_frame[contrast_frame["id"] == entry.get("contrast")] \
            if not contrast_frame.empty else contrast_frame
        if rows.empty or rows.iloc[0]["status"] != "ok":
            return {**base, "status": "no_input",
                    "reason": f"contrast {entry.get('contrast')!r} was not computed"}
        observed = float(rows.iloc[0][entry.get("metric", "absolute_difference")])

    threshold = entry.get("threshold")
    if threshold is None:
        return {**base, "observed": observed, "status": "no_threshold",
                "reason": "the pre-registration supplies no threshold for this criterion"}
    met = (observed >= float(threshold)
           if entry.get("direction", "greater_equal") == "greater_equal"
           else observed <= float(threshold))
    return {**base, "observed": observed, "met": bool(met), "status": "ok"}


def collect_pending_decisions(prereg: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Every ``PENDING_*`` sentinel outside the go/no-go criteria list.

    ``08`` §12 requires ``decision: "incomplete"`` while any ``PENDING_DECISION_*`` remains.
    The v3 pre-registration puts three of them where ``build_go_no_go`` never looked —
    contrast ``e7_c6``'s language tiers, ``criterion_3_combine`` and the interpretation
    matrix's two thresholds. ``pending_preregistration`` keeps its existing meaning
    (criteria that could not be evaluated), so these are reported under their own key.

    Each carries ``blocks_decision``. Only a sentinel that can change a criterion's verdict
    blocks: ``criterion_3_combine`` decides how §18 criterion 3 folds, and a PENDING
    contrast blocks if some criterion or the claim gate reads it. The interpretation matrix
    does **not** — it is a separate ``04`` §6.3 artefact whose own ``matched_row`` is
    already ``null``, and letting it veto the go/no-go would mean an undecided *reading* of
    a result could suppress the result. A non-blocking sentinel is still reported, because
    the operator must know it is open before quoting any table.
    """
    found: List[Dict[str, Any]] = []
    read_by_criteria = {str(entry.get("contrast")) for entry in
                        (prereg.get("go_no_go") or []) if entry.get("contrast")}
    read_by_gate = {str(i) for i in
                    ((prereg.get("claim_gate") or {}).get("contrasts") or [])}
    consumed = read_by_criteria | read_by_gate

    def _record(location: str, node: Any, *, blocks: bool) -> None:
        if isinstance(node, dict) and str(node.get("status", "")).startswith("PENDING"):
            found.append({"location": location, "id": node.get("id"),
                          "status": node["status"], "blocks_decision": bool(blocks),
                          "reason": str(node.get("reason", "")).strip()})

    for index, entry in enumerate(prereg.get("contrasts") or []):
        _record(f"contrasts[{index}]", entry,
                blocks=str(entry.get("id")) in consumed)
    _record("criterion_3_combine", prereg.get("criterion_3_combine"), blocks=True)
    _record("interpretation_matrix", prereg.get("interpretation_matrix"), blocks=False)
    _record("claim_gate", prereg.get("claim_gate"), blocks=True)
    return found


def _combine_criterion_3(prereg: Dict[str, Any], criteria: List[Dict[str, Any]]
                         ) -> Optional[Dict[str, Any]]:
    """Fold ``e7_3k``/``e7_3v`` into one verdict when ``criterion_3_combine`` is resolved.

    Design §18 criterion 3 names no patch object, so the pre-registration registers it
    twice — once per object — and a separate key says how the two combine. `08` specifies
    no handling for that key; ignoring it would silently count criterion 3 twice under
    ``go_no_go_combine: all``, turning a disjunction the design may have intended into a
    conjunction. While the key carries a ``PENDING_*`` status the fold is refused and the
    two limbs are reported unresolved, which the pending sweep already forces to
    ``incomplete``.
    """
    spec = prereg.get("criterion_3_combine") or {}
    ids = [str(i) for i in (spec.get("applies_to") or [])]
    if not ids:
        return None
    limbs = [c for c in criteria if c.get("id") in ids]
    if not limbs:
        return None
    if str(spec.get("status", "")).startswith("PENDING"):
        return {"id": "e7_3", "text": "design §18 criterion 3, combined over its objects",
                "status": spec["status"], "met": None, "combine": None,
                "limbs": {c["id"]: c.get("met") for c in limbs},
                "reason": str(spec.get("reason", "")).strip()
                or "criterion_3_combine is undecided; the two objects are not folded"}

    # An entry with no `PENDING_*` status *looks* resolved, so the value it names must
    # actually be honoured. Defaulting a missing key to `any` would silently resolve
    # decision D6 in code, and falling through an unrecognised value to `any` is worse
    # still: the row would record a `combine` the fold did not use, which reads as an
    # audit trail and is not one. Both are refused (CLAUDE.md trap 13).
    declared = spec.get("combine", spec.get("value"))
    combine = str(declared).strip().lower() if declared is not None else None
    if combine not in VALID_COMBINE:
        return {"id": "e7_3", "text": "design §18 criterion 3, combined over its objects",
                "status": "unresolved_combine", "met": None, "combine": None,
                "declared_combine": declared,
                "limbs": {c["id"]: c.get("met") for c in limbs},
                "reason": (
                    f"criterion_3_combine names no usable rule (`combine: "
                    f"{declared!r}`); expected one of {sorted(VALID_COMBINE)}. Decision D6 "
                    "in new_design_plans/DECISIONS_REQUIRED.md decides how §18 criterion "
                    "3's two objects fold, and it is not resolved by deleting `status:` "
                    "alone — add an explicit `combine:` key.")}

    values = [c.get("met") for c in limbs]
    if combine == "all":
        met = None if any(v is None for v in values) else all(bool(v) for v in values)
    else:
        # `any`: one limb clearing the bar settles it, even if another is unresolved.
        met = True if any(v is True for v in values) else (
            None if any(v is None for v in values) else False)
    return {"id": "e7_3", "text": "design §18 criterion 3, combined over its objects",
            "status": "ok", "met": met, "combine": combine,
            "limbs": {c["id"]: c.get("met") for c in limbs}}


def build_go_no_go(prereg: Dict[str, Any], contrast_frame, retrieval) -> Dict[str, Any]:
    """``e7_go_no_go.json``. Any unresolved criterion forces ``decision: "incomplete"``."""
    gate = evaluate_claim_gate(contrast_frame, prereg)
    criteria = [evaluate_criterion(entry, retrieval=retrieval,
                                   contrast_frame=contrast_frame, claim_gate=gate)
                for entry in prereg.get("go_no_go", [])]

    # `08` §9 — a criterion that *is* the gate must not also be counted beside it.
    gate_component_ids = {str(i) for i in
                          ((prereg.get("claim_gate") or {}).get("satisfies_criteria")
                           or [])}
    composite = _combine_criterion_3(prereg, criteria)
    folded_ids = set(composite["limbs"]) if composite else set()

    counted = [c for c in criteria
               if c.get("id") not in gate_component_ids and c.get("id") not in folded_ids]
    if composite is not None:
        counted = counted + [composite]

    n_met = sum(1 for c in counted if c.get("met") is True)
    n_unknown = sum(1 for c in counted if c.get("met") is None)
    pending = [c["id"] for c in criteria
               if str(c.get("status", "")).startswith("PENDING")]
    pending_decisions = collect_pending_decisions(prereg)
    if gate.get("met") is None:
        n_unknown += 1

    # `08` §5.2 — E7's §18 list is a numbered conjunction, unlike E6A's explicit
    # "at least one of the following". An *absent* key defaults to `any`, which is today's
    # rule and what §5.2 specifies; a *present but unrecognised* one is refused, because
    # reading `combine: "al"` as `any` would invert the decision rule silently (trap 13).
    declared_combine = prereg.get("go_no_go_combine")
    combine = ("any" if declared_combine is None
               else str(declared_combine).strip().lower())
    combine_error = ""
    if combine not in VALID_COMBINE:
        combine_error = (f"go_no_go_combine is {declared_combine!r}; expected one of "
                         f"{list(VALID_COMBINE)}. The decision rule is refused rather "
                         "than defaulted, so no verdict is reached.")
    all_met = bool(counted) and all(c.get("met") is True for c in counted)

    blocking = [d for d in pending_decisions if d["blocks_decision"]]
    if n_unknown or blocking or combine_error:
        decision = "incomplete"
        supported = None
    elif (all_met if combine == "all" else bool(n_met)) and gate.get("met"):
        decision = "continue"
        supported = True
    else:
        decision = "stop"
        supported = False

    return {
        "experiment": "e7",
        "criteria": criteria,
        "criterion_3": composite,
        "claim_gate": gate,
        "combine": combine,
        "combine_error": combine_error,
        "n_criteria": len(criteria),
        "n_counted": len(counted),
        "n_met": n_met,
        "n_met_excluding_gate_components": n_met,
        "n_met_all_criteria": sum(1 for c in criteria if c.get("met") is True),
        "n_unknown": n_unknown,
        "pending_preregistration": pending,
        "pending_decisions": pending_decisions,
        "pending_decisions_blocking": [d["location"] for d in blocking],
        "supported": supported,
        "decision": decision,
        "preregistration": prereg.get("_path"),
        "preregistration_version": prereg.get("version"),
        "note": "`supported` is null while any criterion or pre-registration decision is "
                "unresolved: a claim gate cannot pass on a partially decided "
                "pre-registration (05 §6). Criteria listed in "
                "claim_gate.satisfies_criteria are reported but not counted in n_met, and "
                "the two objects of §18 criterion 3 are folded per criterion_3_combine.",
        "evaluated_utc": prov.utc_now(),
        "aggregator_version": AGGREGATOR_VERSION,
    }


def _retrieval_gate_spec(prereg: Dict[str, Any]) -> Dict[str, Any]:
    """Criterion ``e7_1``'s own object/alignment/metric triple, read from the YAML.

    `08` §11 requires row 5 to compute retrieval "the same way criterion e7_1 does". Taken
    from e7_1's entry rather than restated here, so the two cannot drift apart when the
    pre-registration is edited.
    """
    for entry in prereg.get("go_no_go") or []:
        if entry.get("input") == "retrieval" and entry.get("metric") == "top1_over_chance":
            return {"objects": list(entry.get("objects") or []),
                    "alignment": entry.get("alignment"),
                    "metric": entry.get("metric"), "id": entry.get("id")}
    return {"objects": [], "alignment": None, "metric": "top1_over_chance", "id": None}


def observed_pattern(contrast_frame, *, retrieval=None,
                     prereg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The §10.12 inputs, computed from real rows whether or not the rows exist yet."""
    def _value(contrast_id, column="absolute_difference"):
        if contrast_frame.empty:
            return None
        row = contrast_frame[contrast_frame["id"] == contrast_id]
        if row.empty or row.iloc[0]["status"] != "ok":
            return None
        value = row.iloc[0][column]
        return None if value is None or (isinstance(value, float)
                                         and not np.isfinite(value)) else float(value)

    spec = _retrieval_gate_spec(prereg or {})
    top1_over_chance, _reason = retrieval_observation(
        retrieval, objects=spec["objects"], alignment=spec["alignment"],
        metric=spec["metric"]) if spec["objects"] else (None, "no e7_1 entry")

    return {
        "k0_effect": _value("e7_c1"),
        "v0_effect": _value("e7_c2"),
        # `08` §8.2 — without these four, rows 1, 3, 4, 5 and 6 evaluate to None and
        # `matched_row` is null whatever the data says.
        "r0_effect": _value("e7_c7"),
        "k0_unrelated_effect": _value("e7_c8k"),
        "v0_unrelated_effect": _value("e7_c8v"),
        "retrieval_top1_over_chance": top1_over_chance,
        "k0_minus_v0": _value("e7_c3"),
        "position0_minus_mid": _value("e7_c4"),
        "n_languages_same_direction": _value("e7_c1", "n_languages_same_direction"),
        "n_model_sizes_same_direction": _value("e7_c1", "n_model_sizes_same_direction"),
    }


def build_interpretation_matrix(prereg: Dict[str, Any], contrast_frame,
                                retrieval=None) -> Dict[str, Any]:
    """Map the observed pattern onto exactly one design §10.12 row, or say it matches none.

    Each row is ``{id, text, conditions: {key: {op, value}}}``. With no rows transcribed the
    observed pattern is still computed and recorded — those numbers are real — and
    ``matched_row`` is ``null`` with the PENDING status. Writing the mapping as data means
    pasting the six rows in needs no code change.
    """
    matrix = prereg.get("interpretation_matrix") or {}
    observed = observed_pattern(contrast_frame, retrieval=retrieval, prereg=prereg)
    rows = list(matrix.get("rows") or [])

    declaration = check_threshold_declaration(matrix)

    if str(matrix.get("status", "")).startswith("PENDING") or not rows:
        return {
            "experiment": "e7",
            "status": matrix.get("status", "PENDING"),
            "observed": observed,
            "matched_row": None,
            "threshold_declaration_consistent": declaration["consistent"],
            "threshold_declaration": declaration,
            "candidate_rows": [],
            "reason": str(matrix.get("reason", "")).strip()
            or "the pre-registration transcribes no interpretation rows",
            "evaluated_utc": prov.utc_now(),
            "aggregator_version": AGGREGATOR_VERSION,
        }

    matched: List[str] = []
    evaluated: List[Dict[str, Any]] = []
    for row in rows:
        results = {}
        holds = True
        for key, condition in (row.get("conditions") or {}).items():
            value = observed.get(key)
            ok = _condition_holds(value, condition)
            results[key] = ok
            holds = holds and bool(ok)
        evaluated.append({"id": row.get("id"), "text": row.get("text", ""),
                          "conditions": results, "matched": holds})
        if holds:
            matched.append(str(row.get("id")))

    return {
        "experiment": "e7",
        "status": "ok",
        "observed": observed,
        # `04` §6.3 wants *exactly one* row. Two matches is a defective matrix, not a
        # result, and is reported as such rather than resolved by picking the first.
        "matched_row": matched[0] if len(matched) == 1 else None,
        "n_matched": len(matched),
        "threshold_declaration_consistent": declaration["consistent"],
        "threshold_declaration": declaration,
        "candidate_rows": evaluated,
        "reason": ("matches none of the pre-registered rows" if not matched else
                   ("more than one row matched; the interpretation matrix is not "
                    "mutually exclusive" if len(matched) > 1 else "")),
        "evaluated_utc": prov.utc_now(),
        "aggregator_version": AGGREGATOR_VERSION,
    }


def check_threshold_declaration(matrix: Dict[str, Any]) -> Dict[str, Any]:
    """Does the matrix's ``thresholds:`` block agree with the literals its rows use?

    ``08`` §10 has the six rows carry literal numbers, and ``_condition_holds`` reads those
    — the ``thresholds:`` block above them is a *declaration*, not an input. So editing the
    block alone changes nothing, silently: every test passes and the matrix keeps comparing
    against the old value. Nothing detected that, which is what this closes.

    Compared as sets rather than per key, because the YAML declares no mapping from a
    threshold name to the row keys it governs (``semantic_sensitivity`` drives three keys,
    ``unrelated_transfer`` two). Set equality still catches both drift directions, and
    ``test_each_condition_key_uses_exactly_one_threshold`` pins the per-key half.
    """
    declared = {float(v) for v in (matrix.get("thresholds") or {}).values()}
    used = {float(condition["value"])
            for row in (matrix.get("rows") or [])
            for condition in (row.get("conditions") or {}).values()
            if condition.get("value") is not None}
    if not declared or not used:
        return {"consistent": None, "declared": sorted(declared), "used": sorted(used),
                "reason": "the matrix declares no thresholds or transcribes no rows"}
    if declared == used:
        return {"consistent": True, "declared": sorted(declared), "used": sorted(used),
                "reason": ""}
    return {
        "consistent": False, "declared": sorted(declared), "used": sorted(used),
        "reason": (
            f"interpretation_matrix.thresholds declares {sorted(declared)} but the rows "
            f"compare against {sorted(used)}. The rows are what is evaluated; editing the "
            "`thresholds:` block alone changes nothing. Replace the value in ALL six rows "
            "(08 §10) and re-run tests/test_e7_interpretation_exclusivity.py."),
    }


def _condition_holds(value, condition) -> Optional[bool]:
    if value is None:
        return None
    op = str(condition.get("op", "greater_equal"))
    target = float(condition.get("value", 0.0))
    if op in ("greater_equal", "ge"):
        return value >= target
    if op in ("less_equal", "le"):
        return value <= target
    if op in ("greater", "gt"):
        return value > target
    if op in ("less", "lt"):
        return value < target
    raise ValueError(f"unknown interpretation-matrix operator {op!r}")


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════


def _figure_setup():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_patch_decomposition(frame, path: Path) -> Optional[Path]:
    """Parallel vs unrelated margin change for K0/V0/R0 (``04`` §6.3, fig 6)."""
    if frame.empty:
        return None
    objects = [o for o in ("K0_prerope", "V0", "R0")
               if o in set(frame["patch_object"].dropna())]
    subset = frame[(frame["patch_object"].isin(objects))
                   & (frame["norm_condition"] == "direct")]
    if subset.empty:
        return None

    plt = _figure_setup()
    conditions = ["parallel_en", "same_label_en", "different_label_en", "random_en"]
    fig, ax = plt.subplots(figsize=(1.9 * len(objects) + 2.5, 3.6))
    width = 0.8 / max(len(conditions), 1)
    for index, condition in enumerate(conditions):
        means, errors = [], []
        for obj in objects:
            values = subset[(subset["patch_object"] == obj)
                            & (subset["source_condition"] == condition)]["margin_delta"]
            values = values.dropna()
            means.append(float(values.mean()) if len(values) else np.nan)
            errors.append(float(values.std() / np.sqrt(len(values)))
                          if len(values) > 1 else 0.0)
        offsets = np.arange(len(objects)) + index * width - 0.4 + width / 2
        ax.bar(offsets, means, width=width, yerr=errors, capsize=2, label=condition)
    ax.axhline(0.0, color="black", linewidth=0.8)
    ax.set_xticks(np.arange(len(objects)))
    ax.set_xticklabels(objects)
    ax.set_ylabel("margin change (patched - baseline)")
    ax.set_title("E7 patch decomposition — parallel vs control sources")
    ax.legend(fontsize=8)
    saved = _save(fig, Path(path))
    plt.close(fig)
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════════


def _write(frame, path: Path, *, reason: str = "") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    if reason:
        prov.write_json(path.with_suffix(".reason.json"),
                        {"path": str(path), "reason": reason, **prov.provenance_block()})
    return path


def aggregate(results: Path, *, prereg_path: Path = DEFAULT_PREREG,
              retrieval_path: Optional[Path] = None, out_dir: Optional[Path] = None,
              allow_high_failure: bool = False, figures: bool = True,
              progress: bool = True) -> Dict[str, Any]:
    """Produce every ``04`` §6.3 artefact. Returns the report dict."""
    import pandas as pd

    started = time.time()
    results = Path(results)
    out_dir = Path(out_dir) if out_dir else results / "aggregate"
    out_dir.mkdir(parents=True, exist_ok=True)
    prereg = load_preregistration(prereg_path)
    retrieval = load_retrieval(retrieval_path)

    raw = load_all_rows(results)
    failures, rates = failure_report(raw)
    _write(failures, out_dir / "failures.csv")

    if raw.empty:
        # Still emit the pre-registration-shaped artefacts, with the reason recorded — a
        # missing input must not look like an unsupported claim.
        empty = build_contrasts(pd.DataFrame(), prereg,
                                allow_high_failure=allow_high_failure)
        _write(empty, out_dir / "patching_contrasts.csv",
               reason=f"no patching_per_example.csv under {results}")
        _write(pd.DataFrame(columns=list(SWEEP_COLUMNS)),
               out_dir / "exploratory_layer_sweep.csv",
               reason="no input rows")
        _write(build_table4(empty), out_dir / "table4_patch_effects.csv",
               reason="no input rows")
        decision = build_go_no_go(prereg, empty, retrieval)
        prov.write_json(out_dir / "e7_go_no_go.json",
                        {**decision, **prov.provenance_block()})
        prov.write_json(out_dir / "interpretation_matrix.json",
                        {**build_interpretation_matrix(prereg, empty, retrieval),
                         **prov.provenance_block()})
        report = {"results": str(results), "out_dir": str(out_dir), "n_rows": 0,
                  "n_rows_used": 0, "reason": f"no patching_per_example.csv under "
                                              f"{results}",
                  "decision": decision["decision"], "supported": decision["supported"],
                  "pending_preregistration": decision["pending_preregistration"],
                  "pending_decisions": decision["pending_decisions"],
                  "pending_decisions_blocking":
                      decision["pending_decisions_blocking"],
                  "interpretation_matrix_status":
                      (prereg.get("interpretation_matrix") or {}).get("status",
                                                                     "PENDING"),
                  "aggregator_version": AGGREGATOR_VERSION,
                  "wallclock_s": round(time.time() - started, 3)}
        prov.write_json(out_dir / "aggregate_summary.json",
                        {**report, **prov.provenance_block()})
        return report

    usable = usable_rows(raw)
    n_invalid = int((raw["unit_status"] != "ok").sum())
    if progress:
        print(f"  [aggregate] {len(raw)} rows, {len(usable)} usable, "
              f"{n_invalid} from invalid units")

    contrasts = build_contrasts(usable, prereg, raw=raw,
                                allow_high_failure=allow_high_failure)
    _write(contrasts, out_dir / "patching_contrasts.csv")
    sweep = build_layer_sweep(usable, prereg)
    _write(sweep, out_dir / "exploratory_layer_sweep.csv",
           reason="EXPLORATORY — 04 §6.2 keeps the layer sweep out of the "
                  "pre-registered contrast file; no claim rests on these rows")
    _write(build_table4(contrasts), out_dir / "table4_patch_effects.csv")

    decision = build_go_no_go(prereg, contrasts, retrieval)
    prov.write_json(out_dir / "e7_go_no_go.json", {**decision, **prov.provenance_block()})
    matrix = build_interpretation_matrix(prereg, contrasts, retrieval)
    prov.write_json(out_dir / "interpretation_matrix.json",
                    {**matrix, **prov.provenance_block()})

    figure_note = "skipped"
    if figures:
        try:
            figure = plot_patch_decomposition(
                usable, out_dir / "figures" / "fig6_patch_decomposition.pdf")
            figure_note = str(figure) if figure else "skipped: no data"
        except Exception as exc:  # a plotting failure must never lose a table
            figure_note = f"failed: {type(exc).__name__}: {exc}"

    report = {
        "results": str(results),
        "out_dir": str(out_dir),
        "n_rows": int(len(raw)),
        "n_rows_used": int(len(usable)),
        "n_rows_from_invalid_units": n_invalid,
        "n_failed_rows": int((raw["status"] != "ok").sum()),
        "max_failure_rate_observed": float(max(rates.values())) if rates else 0.0,
        "max_failure_rate_allowed": float(prereg.get("analysis", {})
                                          .get("max_failure_rate", 0.02)),
        "allow_high_failure": bool(allow_high_failure),
        "models": sorted(raw["model_tag"].dropna().unique().tolist()),
        "model_sizes": sorted(raw["model_size"].dropna().unique().tolist()),
        "languages": sorted(raw["target_language"].dropna().unique().tolist()),
        "n_contrasts": int(len(contrasts)),
        "n_contrasts_pending": int((contrasts["status"].astype(str)
                                    .str.startswith("PENDING")).sum())
        if not contrasts.empty else 0,
        "decision": decision["decision"],
        "supported": decision["supported"],
        "pending_preregistration": decision["pending_preregistration"],
        "pending_decisions": decision["pending_decisions"],
        "pending_decisions_blocking": decision["pending_decisions_blocking"],
        "interpretation_matrix_status": matrix["status"],
        "figures": {"fig6_patch_decomposition": figure_note},
        "preregistration": prereg.get("_path"),
        "aggregator_version": AGGREGATOR_VERSION,
        "wallclock_s": round(time.time() - started, 3),
    }
    prov.write_json(out_dir / "aggregate_summary.json",
                    {**report, **prov.provenance_block()})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Aggregate E7 patching results into the 04 §6.3 artefacts.")
    parser.add_argument("--results", default=None,
                        help="root to search for patching_per_example.csv "
                             "(default crosslingual_semantics/results/patching)")
    parser.add_argument("--retrieval", default=None,
                        help="retrieval_by_object.csv or a directory containing one; "
                             "supplies go/no-go criteria e7_1 and e7_2")
    parser.add_argument("--preregistration", default=str(DEFAULT_PREREG))
    parser.add_argument("--out", default=None)
    parser.add_argument("--allow-high-failure", action="store_true",
                        help="compute contrasts whose failure rate exceeds the 05 §7.2 bar")
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = Path(args.results) if args.results else (
        _REPO / "crosslingual_semantics" / "results" / "patching")

    report = aggregate(results, prereg_path=Path(args.preregistration),
                       retrieval_path=Path(args.retrieval) if args.retrieval else None,
                       out_dir=Path(args.out) if args.out else None,
                       allow_high_failure=args.allow_high_failure,
                       figures=args.figures, progress=args.progress)

    print(f"E7 aggregation: {report['n_rows_used']}/{report['n_rows']} rows used, "
          f"decision={report['decision']}, supported={report['supported']}")
    if report.get("pending_preregistration"):
        print(f"  PENDING pre-registration: {report['pending_preregistration']} — "
              "paste design §18/§10.12 into "
              f"{report.get('preregistration')} before reading a verdict.")
    print(f"  -> {report['out_dir']}")
    return 0 if report["n_rows_used"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
