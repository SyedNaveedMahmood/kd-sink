"""Pure probability interventions and exact integer layer scopes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Sequence

import torch


class InterventionError(ValueError):
    """Raised when an intervention or layer scope violates its contract."""


@dataclass(frozen=True)
class AttentionIntervention:
    """A pre-value-aggregation edit of one attention key.

    ``delete`` redistributes source mass according to a fresh conditional
    softmax of the original scores. ``relocate`` moves it to
    ``destination_key``. ``none`` is an explicit causal no-op.
    """

    operation: Literal["none", "delete", "relocate"]
    strength: float = 1.0
    source_key: int = 0
    destination_key: int = 1

    def __post_init__(self) -> None:
        if self.operation not in {"none", "delete", "relocate"}:
            raise InterventionError(f"unknown intervention operation: {self.operation!r}")
        if isinstance(self.strength, bool) or not isinstance(self.strength, (int, float)):
            raise InterventionError("intervention strength must be a real scalar")
        if not 0.0 <= float(self.strength) <= 1.0:
            raise InterventionError("intervention strength must be in [0, 1]")
        for name, value in (
            ("source_key", self.source_key),
            ("destination_key", self.destination_key),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InterventionError(f"{name} must be a non-negative integer")
        if self.operation == "relocate" and self.source_key == self.destination_key:
            raise InterventionError("relocation source and destination must differ")


def normalize_layer_scope(scope: Sequence[int] | None, layer_count: int) -> tuple[int, ...]:
    """Return a duplicate-free integer scope without silently coercing values."""

    if isinstance(layer_count, bool) or not isinstance(layer_count, int) or layer_count <= 0:
        raise InterventionError("layer_count must be a positive integer")
    if scope is None:
        return tuple(range(layer_count))
    normalized: list[int] = []
    seen: set[int] = set()
    for layer in scope:
        if isinstance(layer, bool) or not isinstance(layer, int):
            raise InterventionError("layer scope entries must be integers")
        if layer < 0 or layer >= layer_count:
            raise InterventionError(f"layer {layer} is outside [0, {layer_count})")
        if layer in seen:
            raise InterventionError(f"duplicate layer in scope: {layer}")
        normalized.append(layer)
        seen.add(layer)
    if not normalized:
        raise InterventionError("layer scope cannot be empty")
    return tuple(normalized)


def mapped_teacher_scope(student_scope: Sequence[int], teacher_map: Sequence[int]) -> tuple[int, ...]:
    """Map an exact student-layer scope through an explicit student-to-teacher map."""

    scope = normalize_layer_scope(student_scope, len(teacher_map))
    mapped = tuple(teacher_map[index] for index in scope)
    if any(isinstance(layer, bool) or not isinstance(layer, int) or layer < 0 for layer in mapped):
        raise InterventionError("teacher map entries must be non-negative integers")
    if len(set(mapped)) != len(mapped):
        raise InterventionError("teacher map must be one-to-one over the requested scope")
    return mapped


def _conditional_softmax(scores: torch.Tensor, support: torch.Tensor) -> torch.Tensor:
    """Stable masked softmax which returns zero for rows with no supported key."""

    has_support = support.any(dim=-1, keepdim=True)
    safe_scores = scores.float().masked_fill(~support, -torch.inf)
    safe_scores = torch.where(has_support, safe_scores, torch.zeros_like(safe_scores))
    probabilities = torch.softmax(safe_scores, dim=-1)
    return torch.where(support & has_support, probabilities, torch.zeros_like(probabilities))


def apply_attention_intervention(
    probabilities: torch.Tensor,
    scores: torch.Tensor,
    valid_edges: torch.Tensor,
    intervention: AttentionIntervention,
) -> torch.Tensor:
    """Edit normalized FP32 probabilities before value aggregation.

    Args:
        probabilities: ``[B, H, Q, K]`` normalized pre-dropout probabilities.
        scores: scaled, unmasked attention scores with the same shape.
        valid_edges: boolean tensor broadcastable to the same shape. It includes
            real-query, real-key, and causal validity.
        intervention: validated edit specification.

    Rows without the requested source/destination or without a deletion
    alternative are returned unchanged. This is the explicit q0/length-1 rule.
    """

    if probabilities.ndim != 4 or scores.shape != probabilities.shape:
        raise InterventionError("probabilities and scores must have equal [B,H,Q,K] shape")
    if probabilities.dtype != torch.float32:
        raise InterventionError("intervention probabilities must be FP32")
    if valid_edges.dtype != torch.bool:
        raise InterventionError("valid_edges must be boolean")
    try:
        support = torch.broadcast_to(valid_edges, probabilities.shape)
    except RuntimeError as exc:
        raise InterventionError("valid_edges is not broadcastable to probabilities") from exc
    if not torch.isfinite(probabilities).all() or not torch.isfinite(scores).all():
        raise InterventionError("attention probabilities and scores must be finite")
    if intervention.operation == "none" or float(intervention.strength) == 0.0:
        return probabilities

    key_count = probabilities.shape[-1]
    source = intervention.source_key
    if source >= key_count:
        return probabilities
    source_valid = support[..., source]
    source_mass = probabilities[..., source]
    strength = float(intervention.strength)

    if intervention.operation == "relocate":
        destination = intervention.destination_key
        if destination >= key_count:
            return probabilities
        active = source_valid & support[..., destination]
        candidate = probabilities.clone()
        candidate[..., source] = (1.0 - strength) * source_mass
        candidate[..., destination] = probabilities[..., destination] + strength * source_mass
        return torch.where(active.unsqueeze(-1), candidate, probabilities)

    alternatives = support.clone()
    alternatives[..., source] = False
    active = source_valid & alternatives.any(dim=-1)
    conditional = _conditional_softmax(scores, alternatives)
    candidate = probabilities + strength * source_mass.unsqueeze(-1) * conditional
    candidate[..., source] = (1.0 - strength) * source_mass
    return torch.where(active.unsqueeze(-1), candidate, probabilities)

