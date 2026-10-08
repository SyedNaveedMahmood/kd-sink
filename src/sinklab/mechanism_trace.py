"""E2 live GPT-2 attention junctions and exact isolated-deletion accounting."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping
import math

import torch

from .models import GPT2Adapter, _gpt2_scores, _valid_edges
from .interventions import AttentionIntervention, normalize_layer_scope, _conditional_softmax
from .calibrated_probes import require

VERSION = "mechanistic-e2-live-trace-v1"


@dataclass(frozen=True)
class OutputEdit:
    operation: str
    tensor: torch.Tensor

    def __post_init__(self):
        require(self.operation in {"add", "replace"}, "unknown output-junction edit")


@dataclass(frozen=True)
class LayerTrace:
    layer: int
    residual_input: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    scores: torch.Tensor
    probabilities: torch.Tensor
    valid_edges: torch.Tensor
    pre_projection: torch.Tensor
    post_projection: torch.Tensor
    attention_output: torch.Tensor
    output_weight: torch.Tensor


@dataclass(frozen=True)
class TraceForward:
    outputs: Any
    traces: dict[int, LayerTrace]


@dataclass(frozen=True)
class ParityTolerance:
    atol: float
    rtol: float

    def __post_init__(self):
        require(all(type(x) in (int,float) and math.isfinite(x) and x >= 0 for x in (self.atol,self.rtol)),
                "explicit finite nonnegative parity tolerances required")

    def check(self, actual, reference, label):
        require(actual.shape == reference.shape and torch.isfinite(actual).all() and
                torch.isfinite(reference).all(), f"nonfinite/shape parity: {label}")
        error = (actual.double() - reference.double()).abs()
        require(bool((error <= self.atol + self.rtol * reference.double().abs()).all()), f"parity failed: {label}")
        return {"label": label, "max_absolute_error": float(error.max()), "atol": self.atol, "rtol": self.rtol}


class MechanisticGPT2Adapter(GPT2Adapter):
    """Native clean forwards plus transient observation/output hooks; no S4 edits."""

    def traced_forward(self, *, input_ids, attention_mask, trace_layers: tuple[int,...],
                       delete_layers: tuple[int,...] = (), output_edits: Mapping[int, OutputEdit] | None = None) -> TraceForward:
        require(not any(m.training for m in self.model.modules()), "trace requires eval mode/dropout disabled")
        require(self.model.config._attn_implementation == "eager" and
                all(p.dtype == torch.float32 for p in self.model.parameters()), "trace requires FP32 eager model")
        require(not torch.is_autocast_enabled(input_ids.device.type), "trace forbids autocast")
        require(input_ids.ndim == 2 and attention_mask.dtype == torch.bool and
                attention_mask.shape == input_ids.shape and attention_mask[:,0].all() and
                not ((~attention_mask[:,:-1]) & attention_mask[:,1:]).any(), "boolean right-padded mask required")
        scope = normalize_layer_scope(trace_layers, self.layer_count)
        if delete_layers:
            normalize_layer_scope(delete_layers, self.layer_count)
        edits = dict(output_edits or {})
        if edits:
            normalize_layer_scope(tuple(edits), self.layer_count)
        valid_edges = _valid_edges(attention_mask)
        eligible = attention_mask.clone(); eligible[:,0] = False
        for edit in edits.values():
            require(edit.tensor.shape == (*input_ids.shape, self.model.config.n_embd) and
                    edit.tensor.dtype == torch.float32 and edit.tensor.device == input_ids.device and
                    torch.isfinite(edit.tensor).all(), "output edit shape/device/FP32/finite mismatch")
            if edit.operation == "add":
                require(not edit.tensor[~eligible].any(), "additive edit must be zero at q0 and padding")
        captures, residuals, handles = {i:{} for i in scope}, {}, []
        needed = set(scope) | set(edits)
        try:
            for i in needed:
                block = self.model.transformer.h[i]
                def residual_hook(module, inputs, i=i):
                    require(i not in residuals, "trace layer executed twice")
                    residuals[i] = inputs[0]  # Actual pre-ln_1 residual stream, no detach.
                handles.append(block.register_forward_pre_hook(residual_hook))
                if i in edits:
                    def edit_hook(module, inputs, output, i=i):
                        edit = edits[i]
                        changed = output[0] + edit.tensor if edit.operation == "add" else edit.tensor
                        return (torch.where(eligible[...,None], changed, output[0]), *output[1:])
                    handles.append(block.attn.register_forward_hook(edit_hook))
                if i not in captures:
                    continue
                def qkv_hook(module, inputs, output, i=i):
                    q,k,v = output.split(self.model.config.n_embd, dim=-1)
                    def heads(x):
                        return x.view(*x.shape[:2], self.model.config.n_head, -1).transpose(1,2)
                    captures[i].update(query=heads(q), key=heads(k), value=heads(v))
                def projection_in(module, inputs, i=i):
                    captures[i]["pre_projection"] = inputs[0]
                def projection_out(module, inputs, output, i=i):
                    captures[i]["post_projection"] = output
                def output_hook(module, inputs, output, i=i):
                    captures[i].update(attention_output=output[0], probabilities=output[1])
                handles.extend([block.attn.c_attn.register_forward_hook(qkv_hook),
                    block.attn.c_proj.register_forward_pre_hook(projection_in),
                    block.attn.c_proj.register_forward_hook(projection_out),
                    block.attn.register_forward_hook(output_hook)])
            kwargs = {"input_ids":input_ids, "attention_mask":attention_mask, "use_cache":False}
            outputs = self.forward(intervention=AttentionIntervention("delete") if delete_layers else None,
                                   layer_scope=delete_layers if delete_layers else None, **kwargs)
            traces = {}
            for i, captured in captures.items():
                require(set(captured) == {"query","key","value","probabilities","pre_projection","post_projection","attention_output"},
                        "incomplete live trace")
                traces[i] = LayerTrace(i, residuals[i], **captured,
                    scores=_gpt2_scores(self.model.transformer.h[i].attn, captured["query"],captured["key"]),
                    valid_edges=valid_edges, output_weight=self.model.transformer.h[i].attn.c_proj.weight)
            return TraceForward(outputs,traces)
        finally:
            for handle in reversed(handles):
                handle.remove()


def deletion_factors(trace: LayerTrace) -> dict[str, torch.Tensor]:
    """Stable conditional softmax, never dividing by 1-a0 (even if a0==1)."""
    a, v, edges = trace.probabilities, trace.value, trace.valid_edges
    require(a.ndim == 4 and v.ndim == 4 and a.shape[:2] == v.shape[:2] and
            a.shape[-1] == v.shape[-2], "native head/value layout mismatch")
    support = edges.expand_as(a).clone(); support[...,0] = False
    active = edges.expand_as(a)[...,0] & support.any(-1)
    conditional = _conditional_softmax(trace.scores, support)
    vbar = conditional.to(v.dtype) @ v
    contrast = vbar - v[:,:,0:1,:]
    head_delta = torch.where(active[...,None], a[...,0,None] * contrast, torch.zeros_like(contrast))
    concatenated = head_delta.transpose(1,2).reshape(*trace.pre_projection.shape)
    projected = concatenated @ trace.output_weight  # Conv1D [input, output], no bias.
    return {"a0":a[...,0], "v0":v[:,:,0:1,:], "conditional_value":vbar,
            "value_contrast":contrast, "head_delta":head_delta,
            "pre_projection_delta":concatenated, "projected_delta":projected, "active_queries":active}


def factor_summary(trace: LayerTrace, mask: torch.Tensor, *, denominator_floor: float) -> dict:
    require(type(denominator_floor) in (int,float) and math.isfinite(denominator_floor) and denominator_floor > 0,
            "explicit positive denominator floor required")
    factors = deletion_factors(trace)
    counts = mask.sum(-1)
    q = torch.arange(mask.shape[1],device=mask.device)[None]
    second_half = mask & (q >= ((counts+1)//2)[:,None])
    all_queries = mask.clone(); all_queries[:,0] = False
    require(all_queries.any() and second_half.any(), "two real tokens required")
    def summary(support):
        rows = support[:,None].expand_as(factors["a0"])
        head_norms = factors["head_delta"].double().norm(dim=-1)
        # Each head's projected contribution, then norm of the sum (orientation/cancellation).
        heads,dim = trace.value.shape[1],trace.value.shape[-1]
        weight = trace.output_weight.reshape(heads,dim,-1).double()
        contributions = torch.einsum("bhqd,hdo->bhqo",factors["head_delta"].double(),weight)
        contribution_norms = contributions.norm(dim=-1)
        norm_sum = contribution_norms.sum(1)
        # Orientation uses a double-precision sum of the same projected heads.
        # Undefined zero vectors stay unavailable rather than acquiring an angle.
        total = contributions.sum(1)
        total_norm = total.norm(dim=-1)
        cosine_support = support[:,None] & (contribution_norms >= denominator_floor) & (total_norm[:,None] >= denominator_floor)
        cosines = (contributions * total[:,None]).sum(-1) / (contribution_norms * total_norm[:,None]).clamp_min(denominator_floor**2)
        projected = factors["projected_delta"].double().norm(dim=-1)
        clean = trace.attention_output.double().norm(dim=-1)
        ratios = projected / clean.clamp_min(denominator_floor)
        defined = support & (norm_sum >= denominator_floor)
        return {"real_query_count":int(support.sum()),
            "head_sink_mass_mean":factors["a0"][rows].mean().item(),
            "per_head_sink_value_norm":[float(factors["v0"][:,h].double().norm(dim=-1).mean()) for h in range(heads)],
            "per_head_conditional_value_norm_mean":[float(factors["conditional_value"][:,h].double().norm(dim=-1)[support].mean()) for h in range(heads)],
            "per_head_value_contrast_norm_mean":[float(factors["value_contrast"][:,h].double().norm(dim=-1)[support].mean()) for h in range(heads)],
            "per_head_local_delta_norm_mean":[float(head_norms[:,h][support].mean()) for h in range(heads)],
            "per_head_projected_delta_norm_mean":[float(contribution_norms[:,h][support].mean()) for h in range(heads)],
            "per_head_cosine_with_projected_sum_mean":[float(cosines[:,h][cosine_support[:,h]].mean()) if cosine_support[:,h].any() else None for h in range(heads)],
            "per_head_cosine_defined_queries":[int(cosine_support[:,h].sum()) for h in range(heads)],
            "projected_delta_norm_mean":float(projected[support].mean()),
            "relative_projected_delta_mean":float(ratios[support].mean()),
            "cancellation_ratio_mean":float((projected/norm_sum.clamp_min(denominator_floor))[defined].mean()) if defined.any() else None,
            "cancellation_ratio_defined_queries":int(defined.sum()),
            "clean_output_below_floor_positions":int((clean[support] < denominator_floor).sum()),
            "zero_projected_head_sum_positions":int((norm_sum[support] < denominator_floor).sum())}
    return {"version":VERSION, "layer":trace.layer, "denominator_floor":denominator_floor,
            "all_q_ge1":summary(all_queries), "second_half_queries":summary(second_half),
            "interpretation":"isolated clean-layer algebra, not an additive all-layer decomposition"}


def isolated_parity(adapter, ids, mask, layer: int, tolerance: ParityTolerance) -> tuple[TraceForward, dict]:
    """Actual deletion versus fixed clean-delta injection at exactly one junction."""
    clean = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,))
    trace = clean.traces[layer]
    factors = deletion_factors(trace)
    # Verify captured A,V were used in the actual native projection.
    native_pre = (trace.probabilities.to(trace.value.dtype) @ trace.value).transpose(1,2).reshape_as(trace.pre_projection)
    checks = [tolerance.check(native_pre,trace.pre_projection,"live_A_times_V")]
    direct = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),delete_layers=(layer,))
    injection = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),
        output_edits={layer:OutputEdit("add",factors["projected_delta"])})
    checks += [tolerance.check(direct.traces[layer].pre_projection - trace.pre_projection,
                             factors["pre_projection_delta"],"local_head_delta"),
        tolerance.check(direct.traces[layer].post_projection - trace.post_projection,
                        factors["projected_delta"],"Conv1D_projected_delta"),
        tolerance.check(injection.outputs.logits,direct.outputs.logits,"direct_vs_injected_logits")]
    return clean, {"checks":checks, "direct":direct, "injected":injection, "factors":factors}
