import pytest

from sinklab.analysis import (AnalysisError, matched_ce_pair, missing_cadence,
                              paired_seed_differences)


def _row(condition, seed, device="rtx3090"):
    return {"study": "S1", "condition": condition, "seed": seed, "device_role": device,
            "step": 10000, "status": "complete", "sink": seed + (condition == "C2"),
            "protocol_sha256": "p", "initialization_sha256": f"i{seed}", "data_sha256": "d"}


def test_paired_seeds_are_not_gpu_replicas_or_checkpoints():
    rows = [_row(c, s) for s in (0, 1, 2) for c in ("C1", "C2")]
    rows += [_row("C1", 0, "rtx4080super")]
    paired = paired_seed_differences(rows, condition_a="C1", condition_b="C2",
                                     device_role="rtx3090", step=10000, metric="sink")
    assert [row["seed"] for row in paired] == [0, 1, 2]
    assert [row["difference_b_minus_a"] for row in paired] == [1, 1, 1]
    with pytest.raises(AnalysisError, match="duplicate"):
        paired_seed_differences(rows + [_row("C1", 0)], condition_a="C1",
                                condition_b="C2", device_role="rtx3090", step=10000, metric="sink")
    with pytest.raises(AnalysisError, match="incomplete"):
        paired_seed_differences(rows[:-2], condition_a="C1", condition_b="C2",
                                device_role="rtx3090", step=10000, metric="sink")


def test_matched_ce_selection_uses_only_ce_and_step():
    common = {"seed": 0, "device_role": "rtx3090", "protocol_sha256": "p", "status": "complete"}
    a = [{**common, "step": 100, "ce_nats": 2.00, "sink": 999},
         {**common, "step": 200, "ce_nats": 1.95, "sink": -999}]
    b = [{**common, "step": 100, "ce_nats": 2.04, "sink": -999},
         {**common, "step": 250, "ce_nats": 1.96, "sink": 999}]
    chosen = matched_ce_pair(a, b)
    assert (chosen["a_step"], chosen["b_step"]) == (200, 250)
    for row in a + b:
        row["sink"] *= -1
    assert matched_ce_pair(a, b) == chosen
    assert matched_ce_pair(a, b, max_gap=.001) is None


def test_missing_cadence_never_interpolates():
    assert missing_cadence([0, 100, 300], [0, 100, 200, 300]) == {
        "missing_steps": [200], "unexpected_steps": [], "status": "incomplete"}
    with pytest.raises(AnalysisError, match="duplicate"):
        missing_cadence([0, 100, 100], [0, 100])
