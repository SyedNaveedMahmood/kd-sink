"""Training-panel-only initial gradient calibration; no optimizer updates."""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median
from typing import Any, Callable, Iterable, Mapping, Sequence

import torch
from torch import nn


class CalibrationError(ValueError):
    pass


# Callback returns (unreduced loss numerator, denominator). It may use an
# attention/example denominator or shifted-target denominator as appropriate.
LossTerm = Callable[[nn.Module, Any], tuple[torch.Tensor, int | float | torch.Tensor]]


@dataclass(frozen=True)
class CalibrationEvidence:
    architecture: str
    panel_manifest: str
    seed: int
    raw_norms: Mapping[str, tuple[float, ...]]
    ratios: Mapping[str, tuple[float, ...]]
    factors: Mapping[str, float]


def calibrate_initial_gradients(
    model: nn.Module,
    effective_batches: Sequence[Sequence[Any]],
    losses: Mapping[str, LossTerm],
    *, architecture: str,
    panel_manifest: str,
    panel_role: str,
    seed: int = 1729,
) -> CalibrationEvidence:
    """Measure each full-effective-batch gradient at one unchanged initialization.

    ``losses`` needs ``jsd``, ``mse`` and ``rel``. The callbacks return sums
    and their exact denominators, so unequal microbatches and masks accumulate
    correctly. All model modes, parameters, gradients and RNG are restored.
    """
    if seed != 1729 or panel_role != "training_calibration" or not panel_manifest or not architecture:
        raise CalibrationError("requires seed 1729 and identified training-only calibration panel")
    if len(effective_batches) != 16 or any(not b for b in effective_batches):
        raise CalibrationError("requires 16 nonempty effective batches")
    if set(losses) != {"jsd", "mse", "rel"}:
        raise CalibrationError("requires raw jsd, mse and rel objectives")
    params = tuple(p for p in model.parameters() if p.requires_grad)
    if not params or len({id(p) for p in params}) != len(params):
        raise CalibrationError("requires unique trainable parameters")
    snapshots = [p.detach().clone() for p in params]
    old_grads = [None if p.grad is None else p.grad.detach().clone() for p in params]
    modes = [(m, m.training) for m in model.modules()]
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    norms: dict[str, list[float]] = {k: [] for k in losses}
    try:
        model.eval()
        for batch in effective_batches:
            for name, loss_fn in losses.items():
                summed = [torch.zeros_like(p, dtype=torch.float32) for p in params]
                denominator = 0.0
                for microbatch in batch:
                    numerator, count = loss_fn(model, microbatch)
                    count = float(count.detach().item() if isinstance(count, torch.Tensor) else count)
                    if count <= 0 or not torch.isfinite(torch.tensor(count)) or numerator.ndim != 0 or not torch.isfinite(numerator):
                        raise CalibrationError("nonfinite loss or invalid denominator")
                    if not numerator.requires_grad:
                        raise CalibrationError(f"degenerate {name} loss has no gradient")
                    gradients = torch.autograd.grad(numerator, params, allow_unused=True)
                    denominator += count
                    for result, grad in zip(summed, gradients):
                        if grad is not None:
                            result.add_(grad.detach().float())
                squared = torch.zeros((), dtype=torch.float32, device=params[0].device)
                for gradient in summed:
                    squared += (gradient / denominator).square().sum(dtype=torch.float32)
                norm = float(squared.sqrt().item())
                if not (0 < norm < float("inf")):
                    raise CalibrationError(f"degenerate {name} full-batch gradient")
                norms[name].append(norm)
        ratios = {name: tuple(a / b for a, b in zip(norms["jsd"], norms[name]))
                  for name in ("mse", "rel")}
        factors = {name: median(values) for name, values in ratios.items()}
        if any(not (0 < x < float("inf")) for x in factors.values()):
            raise CalibrationError("nonfinite calibration factor")
        return CalibrationEvidence(architecture, panel_manifest, seed,
                                   {k: tuple(v) for k, v in norms.items()}, ratios, factors)
    finally:
        changed = any(not torch.equal(p.detach(), original) for p, original in zip(params, snapshots))
        for p, original, grad in zip(params, snapshots, old_grads):
            if not torch.equal(p.detach(), original):
                p.data.copy_(original)
            p.grad = grad
        for module, training in modes:
            module.training = training
        torch.set_rng_state(cpu_rng)
        if cuda_rng is not None:
            torch.cuda.set_rng_state_all(cuda_rng)
        if changed:
            raise CalibrationError("calibration callback modified model parameters")
