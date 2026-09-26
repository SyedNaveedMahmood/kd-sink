# -*- coding: utf-8 -*-
"""test_aggregate_contracts.py — the aggregator's refusals and its pre-registration (WP10).

``05_SCHEMAS_AND_CONTRACTS.md`` §7 and ``03`` §7. These test the things that make an E6
table trustworthy rather than merely present:

* an unresolved pre-registration criterion reports ``met: null`` and forces
  ``decision: "incomplete"`` — it never becomes a number (CLAUDE.md rule 4);
* thresholds live in the YAML, not in the code, so a threshold cannot be tuned after
  seeing results without leaving a diff in the pre-registration;
* rows above the 2% failure rate are excluded unless explicitly overridden (§7.2);
* records from different intervention registries raise rather than pool (§7.1);
* BH correction is applied **within** a family, never across all contrasts at once;
* there is no composite "inheritance score" column anywhere (``02`` §4 forbids one).
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402


def _rows(n_failed=0, n_items=100, status="ok", registry="v1", condition="D2",
          seed=0, step=1000, cos=0.5):
    return {"experiment_id": "e6a", "run_id": f"e6a_{condition}_seed{seed}",
            "condition": condition, "seed": seed, "checkpoint_step": step,
            "corpus_id": "tinystories_validation_sink_300", "status": status,
            "n_items": n_items, "n_failed": n_failed,
            "intervention_registry_version": registry, "validation_ce": 2.5,
            "fingerprint_cosine_to_teacher": cos, "baseline_sink": 0.2,
            "topology_wasserstein_to_teacher": 0.01, "n_keys_used": 9}


def _write_run(root: Path, rows) -> Path:
    """Write rows as a run's ``checkpoint_metrics.csv`` so `aggregate` discovers them."""
    path = root / "e6a" / "D2" / "seed0" / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


# ── the pre-registration is honest about what it does not know ──────────────────


def test_shipped_preregistration_has_every_decision_made_and_dated():
    """``e6_prereg_v4``: all four open decisions (D1-D4) are resolved and recorded.

    This replaces the v3 assertion that the shipped file *carries* sentinels. v3's job was
    to refuse to guess; v4's is to have decided, before results existed, with the reasoning
    on the record. Both directions matter, so the sentinel mechanism itself is still
    asserted — by ``test_pending_criteria_are_null_and_force_incomplete`` on an injected
    sentinel, and by ``test_injecting_a_sentinel_into_the_shipped_file_still_blocks``
    below on the real file.

    What is checked here is the property a reviewer would check: nothing is left open, and
    nothing was resolved silently.
    """
    prereg = ag.load_preregistration()
    entries = prereg["contrasts"] + prereg["go_no_go"]
    entries += list((prereg.get("e6b_go_no_go") or {}).get("criteria") or [])
    entries.append(prereg["primary_topology_metric"])

    still_open = [e.get("id", "primary_topology_metric") for e in entries
                  if str(e.get("status", "")).startswith("PENDING")]
    assert not still_open, (
        f"these entries are still undecided: {still_open}. A threshold written after "
        "results exist is not a pre-registration (05 §6).")

    # Every entry that carried a decision records when it was made and why.
    decided = [e for e in entries if e.get("decided")]
    assert len(decided) >= 5, "expected D1-D4 to touch at least five entries"
    for entry in decided:
        name = entry.get("id", "primary_topology_metric")
        assert entry.get("decision_id"), name
        assert entry.get("decision_taken") or entry.get("rationale"), (
            f"{name} was decided without recording the argument")

    # And the amendment log carries one dated `kind: decision` entry per decision.
    decisions = {a.get("decision_id") for a in prereg["amendments"]
                 if a.get("kind") == "decision"}
    assert decisions == {"D1", "D2", "D3", "D4"}, decisions
    for amendment in prereg["amendments"]:
        assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", str(amendment["date"])), (
            f"amendment date {amendment['date']!r} is not a real date — an undated "
            "amendment log reads as one written after the fact (05 §6)")


def test_no_criterion_is_left_unresolvable():
    """A criterion with neither a threshold nor a rule can never be evaluated.

    That is the failure mode v4's edits could most easily introduce: deleting a sentinel
    while leaving ``threshold: null`` behind produces a criterion that is silently
    permanently ``met: null`` and no longer says why.
    """
    prereg = ag.load_preregistration()
    criteria = list(prereg["go_no_go"])
    criteria += list((prereg.get("e6b_go_no_go") or {}).get("criteria") or [])
    for entry in criteria:
        limbs = (entry.get("compound") or {}).get("all_of") or [entry]
        for index, limb in enumerate(limbs):
            if limb.get("evaluation"):        # early_warning etc. carry their own rule
                continue
            resolvable = (limb.get("threshold") is not None
                          or limb.get("threshold_rule") is not None)
            assert resolvable, (
                f"{entry['id']} limb {index} has neither `threshold` nor `threshold_rule`; "
                "it can never be evaluated and would report met: null forever")


def test_injecting_a_sentinel_into_the_shipped_file_still_blocks(tmp_path):
    """The refusal mechanism must survive the file being fully decided.

    v4 removed every sentinel, so no shipped entry exercises this path any more. If the
    reader were deleted tomorrow the suite would still pass on the shipped file — which is
    exactly the "field the aggregator does not read" failure of CLAUDE.md trap 11. So the
    sentinel is put back on a copy and the block is asserted on the real schema.
    """
    import yaml

    prereg = ag.load_preregistration()
    prereg["go_no_go"] = list(prereg["go_no_go"]) + [
        {"id": "reopened", "status": "PENDING_DECISION_THRESHOLD", "text": "t",
         "reason": "put back by the test"}]
    path = tmp_path / "reopened.yaml"
    path.write_text(yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")

    frame = pd.DataFrame([_rows(condition="D2"), _rows(condition="D0", cos=0.1)])
    decision = ag.build_go_no_go(frame, yaml.safe_load(path.read_text(encoding="utf-8")),
                                 {})
    assert decision["decision"] == "incomplete"
    assert "reopened" in decision["pending_preregistration"]


def test_pending_criteria_are_null_and_force_incomplete():
    frame = pd.DataFrame([_rows(condition="D2"), _rows(condition="D0", cos=0.1)])
    prereg = {"contrasts": [], "analysis": {}, "e6b": {}, "version": "t",
              "go_no_go": [
                  {"id": "ok_one", "text": "t", "metric": "fingerprint_cosine_to_teacher",
                   "condition_a": "D2", "condition_b": "D0", "threshold": 0.15,
                   "direction": "greater_equal"},
                  {"id": "pending_one", "status": "PENDING_DESIGN_18", "text": "t",
                   "reason": "design §18 absent"}]}
    decision = ag.build_go_no_go(frame, prereg, {})
    met = {c["id"]: c["met"] for c in decision["criteria"]}
    assert met["ok_one"] is True                       # 0.5 - 0.1 = 0.4 >= 0.15
    assert met["pending_one"] is None
    assert decision["decision"] == "incomplete"
    assert decision["pending_preregistration"] == ["pending_one"]
    pending = next(c for c in decision["criteria"] if c["id"] == "pending_one")
    assert pending["observed"] is None and pending["threshold"] is None
    assert pending["reason"]


def test_criteria_are_read_from_the_yaml_not_hard_coded():
    """Changing the YAML must change the verdict; a code-embedded threshold would not."""
    frame = pd.DataFrame([_rows(condition="D2", cos=0.50),
                          _rows(condition="D0", cos=0.40)])   # observed diff = 0.10
    def _prereg(threshold):
        return {"contrasts": [], "analysis": {}, "e6b": {}, "version": "t",
                "go_no_go": [{"id": "c", "text": "t", "threshold": threshold,
                              "metric": "fingerprint_cosine_to_teacher",
                              "condition_a": "D2", "condition_b": "D0",
                              "direction": "greater_equal"}]}
    assert ag.build_go_no_go(frame, _prereg(0.05), {})["criteria"][0]["met"] is True
    assert ag.build_go_no_go(frame, _prereg(0.15), {})["criteria"][0]["met"] is False


def test_no_composite_inheritance_score_is_produced():
    """`02` §4: the three components stay separate and are never summed into one score."""
    frame = pd.DataFrame([_rows()])
    table2 = ag.build_table2(frame)
    assert set(table2["component"]) <= {"topological", "mechanistic", "functional"}
    for column in list(table2.columns):
        assert "inheritance_score" not in column
        assert "composite" not in column


# ── refusals ────────────────────────────────────────────────────────────────────


def test_rows_above_the_failure_rate_are_excluded_unless_overridden():
    frame = pd.DataFrame([_rows(n_failed=1, n_items=100),          # 1% -> kept
                          _rows(n_failed=3, n_items=100, step=500)])  # 3% -> excluded
    kept, audit = ag.ok_rows(frame, max_failure_rate=0.02, allow_high_failure=False)
    assert len(kept) == 1
    assert audit and audit[0]["reason"] == "failure_rate"
    assert audit[0]["failure_rate"] == pytest.approx(0.03)

    kept_all, _ = ag.ok_rows(frame, max_failure_rate=0.02, allow_high_failure=True)
    assert len(kept_all) == 2


def test_failed_status_rows_are_excluded_and_audited():
    frame = pd.DataFrame([_rows(), _rows(status="fingerprint_failed", step=500)])
    kept, audit = ag.ok_rows(frame, max_failure_rate=0.02, allow_high_failure=False)
    assert len(kept) == 1
    assert any(entry["reason"] == "status" for entry in audit)


def test_mixed_registry_versions_raise_rather_than_pool():
    frame = pd.DataFrame([_rows(registry="v1"), _rows(registry="v2", step=500)])
    with pytest.raises(ValueError, match="intervention_registry_version"):
        ag.assert_comparable(frame)


def test_a_single_registry_version_is_fine():
    ag.assert_comparable(pd.DataFrame([_rows(), _rows(step=500)]))


# ── statistics ──────────────────────────────────────────────────────────────────


def test_bh_correction_is_applied_within_a_family_not_across_all_contrasts():
    frame = pd.DataFrame(
        [_rows(condition="D2", seed=s, cos=0.8) for s in (0, 1, 2)]
        + [_rows(condition="D1", seed=s, cos=0.6) for s in (0, 1, 2)]
        + [_rows(condition="D0", seed=s, cos=0.4) for s in (0, 1, 2)])
    prereg = {"analysis": {"bh_alpha": 0.05}, "e6b": {}, "go_no_go": [], "version": "t",
              "contrasts": [
                  {"id": "a", "metric": "fingerprint_cosine_to_teacher",
                   "condition_a": "D2", "condition_b": "D0", "family": "f1"},
                  {"id": "b", "metric": "fingerprint_cosine_to_teacher",
                   "condition_a": "D1", "condition_b": "D0", "family": "f1"},
                  {"id": "c", "metric": "fingerprint_cosine_to_teacher",
                   "condition_a": "D2", "condition_b": "D1", "family": "f2"}]}
    table = ag.build_contrasts(frame, prereg, {})
    sizes = dict(zip(table["id"], table["bh_family_size"]))
    assert sizes["a"] == sizes["b"] == 2      # corrected within f1 only
    assert sizes["c"] == 1                    # f2 is its own family


def test_a_pending_contrast_produces_no_number():
    frame = pd.DataFrame([_rows()])
    prereg = {"analysis": {}, "e6b": {}, "go_no_go": [], "version": "t",
              "contrasts": [{"id": "p", "status": "PENDING_DESIGN_8_7", "text": "t",
                             "reason": "design §8.7 absent"}]}
    row = ag.build_contrasts(frame, prereg, {}).iloc[0]
    assert row["status"] == "PENDING_DESIGN_8_7"
    assert pd.isna(row["mean_diff"]) and pd.isna(row["p_value"])
    assert row["reason"]


# ── end to end over a results tree ──────────────────────────────────────────────


def test_aggregate_writes_every_table_and_a_reason_for_the_empty_ones(tmp_path):
    _write_run(tmp_path, [_rows(condition="D2", seed=s, cos=0.8) for s in (0, 1, 2)]
               + [_rows(condition="D0", seed=s, cos=0.4) for s in (0, 1, 2)])
    report = ag.aggregate(tmp_path, figures=False, progress=False)

    out = tmp_path / "aggregate"
    for name in ("e6a_contrasts.csv", "e6a_matched_loss.csv",
                 "table2_inheritance_components.csv", "e6b_drift.csv",
                 "e6b_early_warning.csv", "e6b_factorial.csv",
                 "table3_clean_vs_corrupt.csv", "go_no_go.json"):
        assert (out / name).exists(), name
    # E6B has no runs yet: header-only tables, each with a recorded reason (never rows)
    assert len(pd.read_csv(out / "e6b_drift.csv")) == 0
    reason = json.loads((out / "e6b_drift.reason.json").read_text(encoding="utf-8"))
    assert "WP6" in reason["reason"] and reason["n_rows"] == 0
    assert report["n_e6b_rows"] == 0
    assert report["decision"] == "incomplete"   # the shipped YAML still has PENDING rows
    assert json.loads((out / "go_no_go.json").read_text(encoding="utf-8"))["git_sha"]


def test_aggregate_on_an_empty_tree_records_the_fact(tmp_path):
    report = ag.aggregate(tmp_path, figures=False, progress=False)
    assert report["n_rows_read"] == 0
    assert "no checkpoint_metrics.csv" in report["reason"]
    assert (tmp_path / "aggregate" / "go_no_go.json").exists()


# ── arm isolation: the gpt2 arm's rows must never enter the TinyStories arm's tables ──


def _arm_rows(experiment, condition, corpus, *, cos=0.5, seed=0, step=1000):
    row = _rows(condition=condition, seed=seed, step=step, cos=cos)
    row.update({"experiment_id": experiment, "corpus_id": corpus,
                "run_id": f"{experiment}_{condition}_seed{seed}"})
    return row


def _write_arm(root: Path, experiment, condition, rows, seed=0):
    path = root / experiment / condition / f"seed{seed}" / "checkpoint_metrics.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def test_an_experiment_filter_keeps_the_two_arms_apart(tmp_path):
    """Both aggregators rglob and do not de-duplicate, so `experiment_id` is the isolation.

    Without the filter the gpt2 arm's rows would enter the TinyStories arm's contrasts,
    its matched-loss reference and — worst of all — the framing-(B) null spread that
    calibrates e6a_3/e6a_4, silently widening a threshold with a different teacher's runs.
    """
    tiny = "tinystories_validation_sink_300"
    owt = "openwebtext_validation_sink_300"
    for condition in ("D0", "D2"):
        _write_arm(tmp_path, "e6a", condition,
                   [_arm_rows("e6a", condition, tiny, cos=0.5)])
    for condition in ("G0", "G2"):
        _write_arm(tmp_path, "e6a_gpt2", condition,
                   [_arm_rows("e6a_gpt2", condition, owt, cos=0.9)])

    default = ag.aggregate(tmp_path, figures=False, progress=False)
    assert default["experiment"] == "e6a"
    assert default["n_rows_read"] == 4, "discovery still sees the whole tree"
    assert default["n_e6a_rows"] == 2, "but only this arm's rows are used"

    gpt2 = ag.aggregate(tmp_path, figures=False, progress=False,
                        experiment="e6a_gpt2", corpus=owt,
                        reference_condition="G0",
                        prereg_path=(REPO / "transformation_inheritance" / "configs"
                                     / "e6a_gpt2_preregistration.yaml"))
    assert gpt2["experiment"] == "e6a_gpt2"
    assert gpt2["n_e6a_rows"] == 2
    assert gpt2["preregistration_version"] == "e6a_gpt2_prereg_v1"


def test_an_e6b_experiment_filter_keeps_model_scale_arms_apart(tmp_path):
    """A larger-model extension must never pool with the original DistilGPT-2 rows."""
    corpus = "tinystories_validation_sink_300"
    _write_arm(tmp_path, "e6b", "F1",
               [_arm_rows("e6b", "F1", corpus)])
    _write_arm(tmp_path, "e6b_gpt2_medium", "F1",
               [_arm_rows("e6b_gpt2_medium", "F1", corpus)])

    original = ag.aggregate(tmp_path, figures=False, progress=False)
    assert original["e6b_experiment"] == "e6b"
    assert original["n_e6b_rows"] == 1

    extension = ag.aggregate(
        tmp_path, figures=False, progress=False,
        e6b_experiment="e6b_gpt2_medium",
        prereg_path=(REPO / "transformation_inheritance" / "configs"
                     / "e6b_gpt2_medium_preregistration.yaml"))
    assert extension["e6b_experiment"] == "e6b_gpt2_medium"
    assert extension["n_e6b_rows"] == 1
    decision = json.loads(
        (tmp_path / "aggregate" / "e6b_go_no_go.json").read_text(encoding="utf-8"))
    assert decision["experiment"] == "e6b_gpt2_medium"
    assert decision["preregistration_version"] == "e6b_gpt2_medium_prereg_v1"


def test_the_gpt2_arms_preregistration_loads_and_names_its_own_conditions(tmp_path):
    """The second registration file must be readable by the same aggregator, unmodified."""
    prereg = ag.load_preregistration(
        REPO / "transformation_inheritance" / "configs"
        / "e6a_gpt2_preregistration.yaml")
    assert prereg["version"] == "e6a_gpt2_prereg_v1"
    conditions = {e.get("condition_a") for e in prereg["contrasts"] + prereg["go_no_go"]}
    conditions |= {e.get("condition_b") for e in prereg["contrasts"] + prereg["go_no_go"]}
    assert conditions - {None} <= {"G0", "G1", "G2", "PDG2"}
    assert not any("D0" in str(e) or "D1" in str(e) for e in conditions if e)


def test_the_openwebtext_corpus_prefix_is_selected_by_default(tmp_path):
    """`select_corpus` must not drop the new arm's corpus when no --corpus is passed."""
    frame = pd.DataFrame([
        _arm_rows("e6a_gpt2", "G0", "openwebtext_validation_sink_300"),
        _arm_rows("e6a_gpt2", "G0", "e1_100x40"),
    ])
    selected = ag.select_corpus(frame, None)
    assert set(selected["corpus_id"]) == {"openwebtext_validation_sink_300"}
