# -*- coding: utf-8 -*-
"""test_e7_aggregate_contracts.py — WP11 aggregation contracts.

The mirror of ``tests/test_aggregate_contracts.py`` for E7. What it proves:

* an untranscribed design section (``PENDING_DESIGN_*``) reports ``met: null`` with its
  reason and forces ``decision: "incomplete"`` / ``supported: null`` — **whatever the
  results say**;
* **a threshold change in the YAML changes the verdict**, which is the only real proof
  that no threshold is hard-coded in the aggregator;
* the ``05`` §7.2 failure-rate refusal fires, and its override works;
* pre-registered contrasts and the exploratory layer sweep land in *different files*;
* the interpretation matrix maps onto exactly one row, or reports that it matched none —
  and refuses to pick one when the matrix is not mutually exclusive.

No thresholds are asserted by value: each test writes its own tiny pre-registration and
asserts the *relationship* between it and the verdict.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))

import aggregate_crosslingual as ag  # noqa: E402

SHIPPED_PREREG = REPO / "crosslingual_semantics" / "configs" / "e7_preregistration.yaml"


# ── fixtures ──────────────────────────────────────────────────────────────────


def _rows(*, effect=0.05, languages=("de", "zh", "tr"), models=(("t05", "base", "0.5B"),),
          objects=("K0_prerope", "V0", "R0", "Kmid_prerope", "Vmid", "Rmid"),
          layers=(3,), n_examples=6, status="ok", unit_status="ok"):
    """Synthetic patch rows whose parallel-minus-control effect is exactly ``effect``."""
    rows = []
    for tag, variant, size in models:
        for obj in objects:
            magnitude = effect if obj.endswith("0") or "0_" in obj else effect / 5.0
            for layer in layers:
                for lang in languages:
                    for i in range(n_examples):
                        for condition, delta in (("parallel_en", magnitude),
                                                 ("same_label_en", 0.0),
                                                 ("different_label_en", 0.0),
                                                 ("random_en", 0.0)):
                            rows.append({
                                "model": f"Qwen/Qwen2.5-{size}", "model_tag": tag,
                                "model_variant": variant, "dtype": "float32",
                                "target_language": lang, "semantic_id": f"s{i}",
                                "partition": "test", "patch_object": obj,
                                "patch_layer": layer, "norm_condition": "direct",
                                "source_condition": condition,
                                "margin_delta": delta + 0.0005 * i,
                                "status": status, "unit_status": unit_status,
                                "stage": "window",
                            })
    return pd.DataFrame(rows)


def _write_rows(root: Path, frame) -> Path:
    path = root / "patching" / "t" / "patching_per_example.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return root / "patching"


def _prereg(tmp_path: Path, payload) -> Path:
    path = Path(tmp_path) / "prereg.yaml"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return path


def _minimal(threshold=2.0, *, pending=True):
    """One retrieval criterion plus one contrast, so a threshold change is isolatable."""
    payload = {
        "version": "test_v1",
        "contrasts": [{
            "id": "e7_c1", "text": "K0 parallel vs control", "kind": "source_condition",
            "metric": "margin_delta", "treatment_source": "parallel_en",
            "control_source": "same_label_en", "objects": ["K0_prerope"],
            "norm_condition": "direct", "family": "e7_primary", "source": "test",
        }],
        "go_no_go": [{
            "id": "e7_1", "text": "retrieval at least N x chance", "input": "retrieval",
            "metric": "top1_over_chance", "objects": ["K0_prerope"],
            "alignment": "cosine", "threshold": threshold,
            "direction": "greater_equal", "family": "e7_go_no_go", "source": "test",
        }],
        "claim_gate": {"id": "e7_claim_gate", "text": "gate",
                       "contrasts": ["e7_c1"], "min_model_sizes": 1,
                       "min_languages": 2, "n_languages_total": 7, "source": "test"},
        "interpretation_matrix": {"status": "PENDING", "rows": []},
        "analysis": {"max_failure_rate": 0.02, "bootstrap_n": 200,
                     "bootstrap_alpha": 0.05, "bh_alpha": 0.05,
                     "bootstrap_group_cols": ["semantic_id", "target_language"]},
        "amendments": [],
    }
    if pending:
        payload["go_no_go"].append({"id": "e7_x", "status": "PENDING_DESIGN_18",
                                    "text": "not in this repository",
                                    "reason": "design §18 absent"})
    return payload


def _retrieval(tmp_path: Path, top1_over_chance=6.0, delta_points=9.0) -> Path:
    path = tmp_path / "retrieval_by_object.csv"
    pd.DataFrame([{
        "model_tag": "t05", "object": "K0_prerope", "layer": 3, "alignment": "cosine",
        "partition": "all", "top1": 0.02, "top1_over_chance": top1_over_chance,
        "control_object": "Kmid_prerope", "control_top1": 0.005,
        "position0_minus_mid_top1_points": delta_points,
    }]).to_csv(path, index=False, encoding="utf-8")
    return path


# ── PENDING forces incomplete ─────────────────────────────────────────────────


def test_a_pending_criterion_forces_incomplete_whatever_the_results_say(tmp_path):
    results = _write_rows(tmp_path, _rows(effect=0.5))
    report = ag.aggregate(results, prereg_path=_prereg(tmp_path, _minimal(pending=True)),
                          retrieval_path=_retrieval(tmp_path),
                          out_dir=tmp_path / "agg", figures=False, progress=False)

    assert report["decision"] == "incomplete"
    assert report["supported"] is None
    assert "e7_x" in report["pending_preregistration"]

    decision = json.loads((tmp_path / "agg" / "e7_go_no_go.json").read_text("utf-8"))
    pending = [c for c in decision["criteria"] if c["id"] == "e7_x"][0]
    assert pending["met"] is None
    assert pending["observed"] is None
    assert "design" in pending["reason"]


def test_the_shipped_preregistration_is_fully_decided_and_reaches_a_verdict(tmp_path):
    """``e7_prereg_v4``: D5/D6/D7 are made, so the data decides rather than a sentinel.

    This replaces the v3 assertion that the shipped file yields ``incomplete``. That was
    the correct invariant while decisions were open; v4 resolves them, and the property
    worth protecting becomes the opposite one — no sentinel remains to suppress a result,
    and ``supported`` is now driven by the numbers.

    The blocking mechanism is *not* thereby untested: it is asserted on an injected
    sentinel in ``test_reopening_a_decision_blocks_the_verdict_again`` below, so deleting
    the reader would still fail the suite (CLAUDE.md trap 11).
    """
    results = _write_rows(tmp_path, _rows(effect=0.5))
    report = ag.aggregate(results, prereg_path=SHIPPED_PREREG,
                          retrieval_path=_retrieval(tmp_path),
                          out_dir=tmp_path / "agg", figures=False, progress=False)

    assert report["decision"] != "incomplete", (
        "with D5-D7 resolved the verdict must come from the data; 'incomplete' now means "
        "a sentinel crept back in")
    assert report["supported"] is not None
    assert report["pending_decisions_blocking"] == []
    assert report["pending_decisions"] == []
    assert report["interpretation_matrix_status"] != "PENDING_DECISION_THRESHOLD"

    # D6 resolved to `any`, so criterion 3's two objects fold into one reported verdict —
    # written to its own `criterion_3` key so `n_met` cannot count the limbs twice.
    decision = json.loads((tmp_path / "agg" / "e7_go_no_go.json").read_text("utf-8"))
    combined = decision["criterion_3"]
    assert combined is not None and combined["id"] == "e7_3", combined
    assert combined["combine"] == "any", combined
    assert combined["met"] is not None, "a resolved fold must produce a verdict"
    assert set(combined["limbs"]) == {"e7_3k", "e7_3v"}, combined["limbs"]

    # D5 took route C: e7_c6 stays registered, untiered, and computes no grouped value.
    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv")
    c6 = contrasts[contrasts["id"] == "e7_c6"]
    assert len(c6) == 1, "the contrast must still be registered, not deleted"
    assert pd.isna(c6.iloc[0]["absolute_difference"]), (
        "an untiered language_group contrast must report no grouped value")


def test_reopening_a_decision_blocks_the_verdict_again(tmp_path):
    """Putting a sentinel back must still force ``incomplete`` and ``supported: null``.

    v4 leaves no sentinel in the shipped file, so nothing there exercises this path any
    more. Without this test the pending-decision reader could be deleted and the suite
    would stay green — the exact "field the aggregator does not read" failure of trap 11.
    """
    import yaml

    prereg = yaml.safe_load(SHIPPED_PREREG.read_text(encoding="utf-8"))
    prereg["criterion_3_combine"] = {
        **prereg["criterion_3_combine"],
        "status": "PENDING_DECISION_COMBINE",
        "reason": "reopened by the test",
    }
    path = tmp_path / "reopened.yaml"
    path.write_text(yaml.safe_dump(prereg, sort_keys=False, allow_unicode=True),
                    encoding="utf-8")

    results = _write_rows(tmp_path, _rows(effect=0.5))
    report = ag.aggregate(results, prereg_path=path,
                          retrieval_path=_retrieval(tmp_path),
                          out_dir=tmp_path / "agg2", figures=False, progress=False)

    assert report["decision"] == "incomplete"
    assert report["supported"] is None
    assert "criterion_3_combine" in set(report["pending_decisions_blocking"])


# ── the threshold really lives in the YAML ────────────────────────────────────


def test_changing_a_threshold_in_the_yaml_changes_the_verdict(tmp_path):
    """The proof that no threshold is hard-coded in the aggregator."""
    results = _write_rows(tmp_path, _rows(effect=0.05))
    retrieval = _retrieval(tmp_path, top1_over_chance=6.0)

    lenient = ag.aggregate(results,
                           prereg_path=_prereg(tmp_path / "a",
                                               _minimal(threshold=2.0, pending=False)),
                           retrieval_path=retrieval, out_dir=tmp_path / "agg_a",
                           figures=False, progress=False)
    strict = ag.aggregate(results,
                          prereg_path=_prereg(tmp_path / "b",
                                              _minimal(threshold=99.0, pending=False)),
                          retrieval_path=retrieval, out_dir=tmp_path / "agg_b",
                          figures=False, progress=False)

    assert lenient["decision"] == "continue" and lenient["supported"] is True
    assert strict["decision"] == "stop" and strict["supported"] is False

    a = json.loads((tmp_path / "agg_a" / "e7_go_no_go.json").read_text("utf-8"))
    b = json.loads((tmp_path / "agg_b" / "e7_go_no_go.json").read_text("utf-8"))
    assert a["criteria"][0]["met"] is True
    assert b["criteria"][0]["met"] is False
    # Same data, same observed value — only the threshold moved.
    assert a["criteria"][0]["observed"] == b["criteria"][0]["observed"]


def test_a_criterion_with_no_retrieval_input_reports_why_rather_than_failing(tmp_path):
    results = _write_rows(tmp_path, _rows())
    report = ag.aggregate(results,
                          prereg_path=_prereg(tmp_path, _minimal(pending=False)),
                          retrieval_path=None, out_dir=tmp_path / "agg",
                          figures=False, progress=False)

    assert report["decision"] == "incomplete"
    decision = json.loads((tmp_path / "agg" / "e7_go_no_go.json").read_text("utf-8"))
    criterion = decision["criteria"][0]
    assert criterion["status"] == "no_input"
    assert criterion["met"] is None
    assert "retrieval" in criterion["reason"]


# ── the failure-rate gate ─────────────────────────────────────────────────────


def test_a_contrast_above_the_failure_rate_is_refused_and_says_so(tmp_path):
    clean = _rows(languages=("zh", "tr"))
    broken = _rows(languages=("de",), status="failed")
    results = _write_rows(tmp_path, pd.concat([clean, broken], ignore_index=True))

    report = ag.aggregate(results, prereg_path=_prereg(tmp_path, _minimal(pending=False)),
                          retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                          figures=False, progress=False)

    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv")
    row = contrasts.iloc[0]
    assert row["status"] == "excluded"
    assert row["failure_rate"] > 0.02
    assert "05" in row["excluded_reason"] and "failure rate" in row["excluded_reason"]
    assert report["max_failure_rate_allowed"] == 0.02


def test_the_override_computes_the_refused_contrast(tmp_path):
    clean = _rows(languages=("zh", "tr"))
    broken = _rows(languages=("de",), status="failed")
    results = _write_rows(tmp_path, pd.concat([clean, broken], ignore_index=True))

    ag.aggregate(results, prereg_path=_prereg(tmp_path, _minimal(pending=False)),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False, allow_high_failure=True)

    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv")
    assert contrasts.iloc[0]["status"] == "ok"
    assert contrasts.iloc[0]["n_pairs"] > 0


def test_a_healthy_condition_is_not_refused_by_an_unrelated_broken_one(tmp_path):
    """The gate is scoped to the rows a contrast consumes, not to the whole run."""
    clean = _rows(objects=("K0_prerope",))
    broken = _rows(objects=("Vmid",), status="failed")
    results = _write_rows(tmp_path, pd.concat([clean, broken], ignore_index=True))

    ag.aggregate(results, prereg_path=_prereg(tmp_path, _minimal(pending=False)),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv")
    assert contrasts.iloc[0]["status"] == "ok"
    assert contrasts.iloc[0]["failure_rate"] == 0.0


# ── pre-registered vs exploratory ─────────────────────────────────────────────


def test_the_layer_sweep_is_a_separate_file_and_says_it_is_exploratory(tmp_path):
    results = _write_rows(tmp_path, _rows(layers=(1, 2, 3)))
    ag.aggregate(results, prereg_path=_prereg(tmp_path, _minimal(pending=False)),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv")
    sweep = pd.read_csv(tmp_path / "agg" / "exploratory_layer_sweep.csv")

    assert set(contrasts["id"]) == {"e7_c1"}
    assert len(sweep) > len(contrasts)
    assert sweep["note"].str.startswith("EXPLORATORY").all()
    assert "q_value" not in sweep.columns, "no BH correction across an exploratory sweep"
    assert (tmp_path / "agg" / "exploratory_layer_sweep.reason.json").exists()


def test_bh_correction_is_applied_within_a_family_only(tmp_path):
    payload = _minimal(pending=False)
    payload["contrasts"] = [
        {**payload["contrasts"][0], "id": "e7_c1", "family": "e7_primary"},
        {**payload["contrasts"][0], "id": "e7_c2", "objects": ["V0"],
         "family": "e7_primary"},
        {**payload["contrasts"][0], "id": "e7_c5", "objects": ["R0"],
         "family": "e7_secondary"},
    ]
    results = _write_rows(tmp_path, _rows())
    ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    contrasts = pd.read_csv(tmp_path / "agg" / "patching_contrasts.csv").set_index("id")
    assert contrasts.loc["e7_c1", "bh_family_size"] == 2
    assert contrasts.loc["e7_c2", "bh_family_size"] == 2
    assert contrasts.loc["e7_c5", "bh_family_size"] == 1


# ── the claim gate and the interpretation matrix ──────────────────────────────


def test_the_claim_gate_counts_languages_and_model_sizes(tmp_path):
    payload = _minimal(pending=False)
    payload["claim_gate"].update({"min_model_sizes": 2, "min_languages": 3})
    two_sizes = _rows(models=(("t05", "base", "0.5B"), ("t15", "base", "1.5B")))
    results = _write_rows(tmp_path, two_sizes)

    ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)
    gate = json.loads((tmp_path / "agg" /
                       "e7_go_no_go.json").read_text("utf-8"))["claim_gate"]

    assert gate["met"] is True
    entry = gate["per_contrast"][0]
    assert entry["n_model_sizes_same_direction"] == 2
    assert entry["n_languages_same_direction"] == 3


def test_the_claim_gate_fails_with_only_one_model_size(tmp_path):
    payload = _minimal(pending=False)
    payload["claim_gate"].update({"min_model_sizes": 2, "min_languages": 3})
    results = _write_rows(tmp_path, _rows())

    report = ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                          retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                          figures=False, progress=False)
    gate = json.loads((tmp_path / "agg" /
                       "e7_go_no_go.json").read_text("utf-8"))["claim_gate"]

    assert gate["met"] is False
    assert report["supported"] is False


def test_the_interpretation_matrix_matches_exactly_one_row_when_it_can(tmp_path):
    payload = _minimal(pending=False)
    payload["interpretation_matrix"] = {"status": "ok", "rows": [
        {"id": "row_a", "text": "K0 carries it",
         "conditions": {"k0_effect": {"op": "greater_equal", "value": 0.01}}},
        {"id": "row_b", "text": "K0 carries nothing",
         "conditions": {"k0_effect": {"op": "less", "value": 0.01}}},
    ]}
    results = _write_rows(tmp_path, _rows(effect=0.5))
    ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    matrix = json.loads((tmp_path / "agg" /
                         "interpretation_matrix.json").read_text("utf-8"))
    assert matrix["status"] == "ok"
    assert matrix["matched_row"] == "row_a"
    assert matrix["n_matched"] == 1
    assert matrix["observed"]["k0_effect"] is not None


def test_an_ambiguous_interpretation_matrix_refuses_to_pick_a_row(tmp_path):
    """`04` §6.3 wants exactly one row; two matches is a defective matrix, not a result."""
    payload = _minimal(pending=False)
    payload["interpretation_matrix"] = {"status": "ok", "rows": [
        {"id": "row_a", "conditions": {"k0_effect": {"op": "greater_equal",
                                                     "value": -99.0}}},
        {"id": "row_b", "conditions": {"k0_effect": {"op": "less", "value": 99.0}}},
    ]}
    results = _write_rows(tmp_path, _rows())
    ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    matrix = json.loads((tmp_path / "agg" /
                         "interpretation_matrix.json").read_text("utf-8"))
    assert matrix["n_matched"] == 2
    assert matrix["matched_row"] is None
    assert "mutually exclusive" in matrix["reason"]


def test_a_matrix_that_matches_nothing_says_so(tmp_path):
    payload = _minimal(pending=False)
    payload["interpretation_matrix"] = {"status": "ok", "rows": [
        {"id": "row_a", "conditions": {"k0_effect": {"op": "greater_equal",
                                                     "value": 99.0}}},
    ]}
    results = _write_rows(tmp_path, _rows())
    ag.aggregate(results, prereg_path=_prereg(tmp_path, payload),
                 retrieval_path=_retrieval(tmp_path), out_dir=tmp_path / "agg",
                 figures=False, progress=False)

    matrix = json.loads((tmp_path / "agg" /
                         "interpretation_matrix.json").read_text("utf-8"))
    assert matrix["matched_row"] is None
    assert "matches none" in matrix["reason"]


# ── missing input is not an unsupported claim ─────────────────────────────────


def test_no_input_still_writes_the_artefacts_with_a_recorded_reason(tmp_path):
    report = ag.aggregate(tmp_path / "empty", prereg_path=SHIPPED_PREREG,
                          out_dir=tmp_path / "agg", figures=False, progress=False)

    assert report["n_rows"] == 0
    assert report["decision"] == "incomplete"
    assert report["supported"] is None
    assert "no patching_per_example.csv" in report["reason"]
    for name in ("patching_contrasts.csv", "e7_go_no_go.json",
                 "interpretation_matrix.json", "aggregate_summary.json"):
        assert (tmp_path / "agg" / name).exists(), name


def test_model_size_is_parsed_from_the_model_id_not_guessed():
    assert ag.model_size_of("Qwen/Qwen2.5-0.5B") == "0.5B"
    assert ag.model_size_of("Qwen/Qwen2.5-1.5B-Instruct") == "1.5B"
    assert ag.model_size_of("Qwen/Qwen2.5-3B") == "3B"
    # Unparseable names stay visible rather than collapsing two sizes into one bucket.
    assert ag.model_size_of("local/checkpoint") == "local/checkpoint"


@pytest.mark.parametrize("op,value,observed,expected", [
    ("greater_equal", 1.0, 2.0, True),
    ("less_equal", 1.0, 2.0, False),
    ("greater", 2.0, 2.0, False),
    ("less", 3.0, 2.0, True),
])
def test_interpretation_operators(op, value, observed, expected):
    assert ag._condition_holds(observed, {"op": op, "value": value}) is expected


def test_an_unknown_interpretation_operator_raises():
    with pytest.raises(ValueError, match="unknown interpretation-matrix operator"):
        ag._condition_holds(1.0, {"op": "approximately", "value": 1.0})
