"""Independent arithmetic and corruption checks for recorded S1 context."""
import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("context_report", Path(__file__).parents[2] / "scripts/report_mechanistic_context.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def fixture():
    values = [{"behavior": {"valid_targets": 127, "clean_nll_sum_nats": 254, "edited_nll_sum_nats": 381,
               "self_kl_sum_nats": 12.7, "absolute_target_logprob_change_sum_nats": 127, "flip_count": 127},
               "clean_structure": {"native_layer_mean": .25}} for _ in range(300)]
    metrics = {"clean_ce_nats": 2, "edited_ce_nats": 3, "delta_ce_nats": 1, "self_kl_nats": .1,
               "absolute_target_logprob_change_nats": 1, "prediction_flip_fraction": 1}
    return values, metrics


def test_exact_reduction():
    assert module.reaggregate(*fixture()) == .25


@pytest.mark.parametrize("corruption", ["targets", "coverage", "metric", "signed_delta"])
def test_rejects_corruption(corruption):
    values, metrics = fixture()
    if corruption == "targets": values[0]["behavior"]["valid_targets"] = 126
    elif corruption == "coverage": values.pop()
    elif corruption == "metric": metrics["self_kl_nats"] = .2
    else: metrics["delta_ce_nats"] = -1
    with pytest.raises(ValueError): module.reaggregate(values, metrics)
