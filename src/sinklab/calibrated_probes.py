"""E1 continuous GPT-2 probes, explicit effective-dose controls; S4 is unchanged."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import math
from typing import Iterator

import torch
from transformers import GPT2LMHeadModel

from .interventions import normalize_layer_scope, mapped_teacher_scope
from .probes import epe_directions, transport_epe_batch, coordinate_controls

VERSION = "mechanistic-e1-routes-v1"
ROUTES = {"none", "reapply_q", "q_bias", "k_input", "epe_transport", "position0_to1"}
ALPHAS = (0., .25, .5, .75, 1.)  # Declared proposal, not a fitted dose grid.


def require(condition, message):
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True)
class RouteProbe:
    route: str
    alpha: float
    layers: tuple[int, ...]
    coordinates: tuple[int, ...] = ()

    def __post_init__(self):
        require(self.route in ROUTES, "unknown E1 route")
        require(type(self.alpha) in (int, float) and math.isfinite(self.alpha) and 0 <= self.alpha <= 1,
                "finite alpha in [0,1] required")
        require(isinstance(self.layers, tuple) and isinstance(self.coordinates, tuple), "immutable route tuples required")
        if self.route == "epe_transport":
            require(self.layers == (0,), "legacy EPE is layer0 only")
        elif self.route in {"none", "position0_to1"}:
            require(not self.layers, "global/no-op route has no layer scope")
        else:
            require(bool(self.layers), "explicit route scope required")
        if self.route == "k_input":
            require(len(self.coordinates) == 3 and len(set(self.coordinates)) == 3 and
                    all(type(c) is int and c >= 0 for c in self.coordinates), "three model-local K coordinates required")
        else:
            require(not self.coordinates, "coordinates apply only to K-input probes")


def route_scope(*, role: str, kind: str, layer_count: int, student_scope: tuple[int, ...],
                teacher_map: tuple[int, ...]) -> tuple[int, ...]:
    """Explicit map supplied by the protocol; no head/coordinate homology."""
    require(role in {"student", "teacher"}, "explicit model role required")
    if kind == "native":
        return tuple(range(layer_count))
    require(role == "teacher" and kind == "mapped_teacher", "mapped scope is teacher-only")
    scope = mapped_teacher_scope(student_scope, teacher_map)
    return normalize_layer_scope(scope, layer_count)


def transport(output, u0, u1, mask, alpha):
    require(type(alpha) in (int, float) and math.isfinite(alpha) and 0 <= alpha <= 1, "invalid EPE alpha")
    if alpha == 0:
        return output
    full = transport_epe_batch(output, u0, u1, mask)
    return full if alpha == 1 else output + alpha * (full - output)


@contextmanager
def apply_route(model: GPT2LMHeadModel, probe: RouteProbe, mask: torch.Tensor) -> Iterator[dict]:
    """Edit only named slices, not whole QKV tensors; restore even on exceptions."""
    require(isinstance(model, GPT2LMHeadModel) and not model.training and
            model.config._attn_implementation == "eager", "E1 requires eval GPT-2 eager attention")
    require(mask.ndim == 2 and mask.shape[1] >= 2 and mask[:, :2].bool().all(), "two real positions required")
    width = model.config.n_embd
    if probe.layers:
        normalize_layer_scope(probe.layers, model.config.n_layer)
    require(all(c < width for c in probe.coordinates), "K coordinates exceed native width")
    info = {"version": VERSION, "route": probe.route, "alpha": probe.alpha,
            "parameter_delta_l2": 0., "parameter_norm_kind": "edited_slices_combined_l2",
            "layers": list(probe.layers), "coordinates": list(probe.coordinates)}
    if probe.route == "none" or probe.alpha == 0:
        yield info
        return
    if probe.route == "epe_transport":
        direction = epe_directions(model)
        # Legacy directions are inference tensors; own clones can be used in a live grad graph.
        u0, u1 = direction["u"][0].clone(), direction["u"][1].clone()
        info.update(parameter_delta_l2=None, parameter_norm_kind="activation_edit_not_parameter_edit",
                    epe_definition=direction["definition"])
        handle = model.transformer.h[0].mlp.register_forward_hook(
            lambda module, inputs, output: transport(output, u0, u1, mask, probe.alpha))
        try:
            yield info
        finally:
            handle.remove()
        return
    selected = []
    if probe.route == "position0_to1":
        selected = [(model.transformer.wpe.weight, 0)]
    else:
        for layer in probe.layers:
            attn = model.transformer.h[layer].attn
            selected.append((attn.c_attn.weight, (list(probe.coordinates), slice(width, 2 * width)))
                            if probe.route == "k_input" else (attn.c_attn.bias, slice(0, width)))
    originals = [tensor[index].detach().clone() for tensor, index in selected]
    squared = 0.
    try:
        with torch.no_grad():
            for (tensor, index), original in zip(selected, originals, strict=True):
                changed = (model.transformer.wpe.weight[1] if probe.route == "position0_to1" and probe.alpha == 1 else
                           original + probe.alpha * (model.transformer.wpe.weight[1] - original)
                           if probe.route == "position0_to1" else original if probe.route == "reapply_q"
                           else (1 - probe.alpha) * original)
                squared += float((changed.double() - original.double()).square().sum())
                tensor[index] = changed
        info["parameter_delta_l2"] = math.sqrt(squared)
        yield info
    finally:
        with torch.no_grad():
            for (tensor, index), original in zip(selected, originals, strict=True):
                tensor[index] = original
                require(torch.equal(tensor[index], original), "route failed exact parameter restoration")


def observed_dose_match(left: list[dict], right: list[dict], *, dose_metric: str, target: float) -> dict:
    """Adjacent observed brackets only; never choose a branch or extrapolate."""
    require(dose_metric in {"sink_removed_fraction", "attention_output_delta_rms"}, "matching on output loss is forbidden")
    require(type(target) in (float, int) and math.isfinite(target), "finite dose target required")
    def locate(curve):
        require(len(curve) >= 2 and all(type(r["alpha"]) in (int, float) and math.isfinite(r["alpha"]) and
                type(r[dose_metric]) in (int, float) and math.isfinite(r[dose_metric]) for r in curve), "finite observed dose curve required")
        require(all(a["alpha"] < b["alpha"] for a,b in zip(curve,curve[1:])), "strict alpha order required")
        hits = []
        # Exact observations retain their alpha and are not duplicated as brackets.
        for row in curve:
            if row[dose_metric] == target:
                hits.append({"alpha": row["alpha"], "bracket": [row["alpha"], row["alpha"]], "weight": 0.})
        for a,b in zip(curve,curve[1:]):
            if min(a[dose_metric], b[dose_metric]) < target < max(a[dose_metric], b[dose_metric]):
                w = (target - a[dose_metric]) / (b[dose_metric] - a[dose_metric])
                hits.append({"alpha": a["alpha"] + w * (b["alpha"] - a["alpha"]), "bracket": [a["alpha"],b["alpha"]], "weight": w})
        return hits
    l, r = locate(left), locate(right)
    return {"status": "dose_unmatched" if not l or not r else "ambiguous_nonmonotonic" if len(l) != 1 or len(r) != 1 else "observed_overlap",
            "dose_metric": dose_metric, "target": target, "left_candidates": l, "right_candidates": r,
            "interpolation_only": True, "interpretation": "linear descriptive interpolation within observed brackets; no equivalence claim"}


def validate_panel_split(discovery: list[dict], confirmation: list[dict]) -> dict:
    """Block and underlying source-document identities must both be disjoint."""
    def identities(items):
        require(bool(items) and len({i["id"] for i in items}) == len(items), "unique nonempty panel required")
        for item in items:
            require(isinstance(item["id"], str) and item["id"] and item.get("document_ids") and
                    item.get("document_hashes") and len(item["document_ids"]) == len(item["document_hashes"]),
                    "explicit underlying document IDs and text hashes required")
            require(all(isinstance(d, str) and d for d in item["document_ids"]) and
                    all(isinstance(h, str) and len(h) == 64 and all(c in '0123456789abcdef' for c in h)
                        for h in item["document_hashes"]), "valid source document identities required")
        return {i["id"] for i in items}, {d for i in items for d in i["document_ids"]}, {h for i in items for h in i["document_hashes"]}
    d, c = identities(discovery), identities(confirmation)
    require(all(not a.intersection(b) for a,b in zip(d,c)), "discovery/confirmation item or document overlap")
    return {"status": "document_disjoint", "discovery_items": len(discovery), "confirmation_items": len(confirmation),
            "discovery_documents": len(d[1]), "confirmation_documents": len(c[1])}


def route_controls(model, control_seed: int) -> dict:
    return coordinate_controls(model, control_seed=control_seed)


def observed_forward(adapter, ids, mask):
    """Observe actual Q/K/MLP/output computation, including installed route hooks."""
    observed, handles = {}, []
    for layer, block in enumerate(adapter.model.transformer.h):
        observed[layer] = {}
        def qkv_hook(module, inputs, output, layer=layer):
            q, k, _ = output.split(adapter.model.config.n_embd, dim=-1)
            observed[layer].update(q=q.detach(), k=k.detach())
        def out_hook(module, inputs, output, layer=layer):
            observed[layer]["attention_output"] = output[0].detach()
        handles.extend([block.attn.c_attn.register_forward_hook(qkv_hook), block.attn.register_forward_hook(out_hook)])
        if layer == 0:
            handles.append(block.mlp.register_forward_hook(
                lambda module, inputs, output: observed[0].update(epe_output=output.detach())))
    try:
        # Native attention matrices from this same live forward, not recomputed features.
        result = adapter.forward(input_ids=ids, attention_mask=mask, output_attentions=True, use_cache=False)
        return result, observed
    finally:
        for handle in reversed(handles):
            handle.remove()


def activation_effect(clean: dict, edited: dict, mask: torch.Tensor) -> dict:
    require(set(clean) == set(edited), "activation layer coverage mismatch")
    result = []
    for layer in clean:
        require(set(clean[layer]) == set(edited[layer]), "activation locus mismatch")
        row = {"layer": layer, "valid_positions": int(mask.bool().sum())}
        for locus in clean[layer]:
            a, b = clean[layer][locus], edited[layer][locus]
            require(a.shape == b.shape and a.shape[:2] == mask.shape and torch.isfinite(a).all() and
                    torch.isfinite(b).all(), "activation shape/finite mismatch")
            norms = (b.double() - a.double()).norm(dim=-1)[mask.bool()]
            row[locus + "_delta_rms"] = float(norms.square().mean().sqrt())
            row[locus + "_delta_max"] = float(norms.max())
        result.append(row)
    return {"support": "all_real_positions", "layers": result,
            "attention_output_delta_rms": math.sqrt(sum(r["attention_output_delta_rms"] ** 2 for r in result) / len(result))}
