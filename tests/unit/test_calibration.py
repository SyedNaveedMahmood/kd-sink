import math

import pytest
import torch
from torch import nn

from sinklab.calibration import CalibrationError, calibrate_initial_gradients


class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([0.4, -0.7]))
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        return self.dropout(self.weight) @ x


def _loss(scale):
    def calculate(model, micro):
        x, y = micro
        value = (model(x) - y).square()
        return scale * value.sum(), value.numel()
    return calculate


def test_full_batch_gradient_norm_ratios_modes_rng_and_architecture():
    model = Tiny()
    model.train()
    model.weight.grad = torch.tensor([8.0, 9.0])
    batch = [
        (torch.tensor([[1., 2.], [3., 4.]]).T, torch.tensor([0., 1.])),
        (torch.tensor([[5., -2.]]).T, torch.tensor([2.])),
    ]
    # The denominator is three examples. The two microbatch gradients oppose
    # partly, so a mean of separate gradient norms would be different.
    with torch.no_grad():
        pass
    flat_x = torch.cat([m[0] for m in batch], 1)
    flat_y = torch.cat([m[1] for m in batch])
    manual = nn.Parameter(model.weight.detach().clone())
    expected = torch.autograd.grad(((manual @ flat_x - flat_y).square()).mean(), manual)[0].norm().item()
    rng = torch.get_rng_state().clone()
    evidence = calibrate_initial_gradients(
        model, [batch, batch], {"jsd": _loss(1.0), "mse": _loss(2.0), "rel": _loss(0.5)},
        architecture="s1_large_medium", panel_manifest="fixture-sha", panel_role="training_calibration",
        expected_batches=2,
    )
    assert evidence.raw_norms["jsd"] == pytest.approx((expected, expected), rel=1e-6)
    assert evidence.factors == pytest.approx({"mse": 0.5, "rel": 2.0})
    assert evidence.architecture == "s1_large_medium"
    assert model.training and model.dropout.training
    assert torch.equal(model.weight.grad, torch.tensor([8.0, 9.0]))
    assert torch.equal(torch.get_rng_state(), rng)


def test_calibration_rejects_panel_and_degenerate_gradients():
    model = Tiny()
    batches = [[(torch.ones(2, 1), torch.zeros(1))]]
    losses = {"jsd": _loss(1), "mse": _loss(2), "rel": _loss(3)}
    with pytest.raises(CalibrationError, match="training-only"):
        calibrate_initial_gradients(model, batches, losses, architecture="s1", panel_manifest="x",
                                    panel_role="final_evaluation", expected_batches=1)
    with pytest.raises(CalibrationError, match="degenerate"):
        calibrate_initial_gradients(model, batches, {**losses, "mse": _loss(0)},
                                    architecture="s1", panel_manifest="x",
                                    panel_role="training_calibration", expected_batches=1)
    assert model.training


def test_calibration_rejects_mutation_and_restores_model():
    model = Tiny()
    original = model.weight.detach().clone()
    batches = [[(torch.ones(2, 1), torch.zeros(1))]]

    def bad(model, micro):
        with torch.no_grad():
            model.weight.add_(1)
        return _loss(1)(model, micro)

    with pytest.raises(CalibrationError, match="modified"):
        calibrate_initial_gradients(model, batches, {"jsd": bad, "mse": _loss(1), "rel": _loss(1)},
                                    architecture="s3_small_distil", panel_manifest="x",
                                    panel_role="training_calibration", expected_batches=1)
    assert torch.equal(model.weight, original)
