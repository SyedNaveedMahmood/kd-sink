"""E3 equal-norm controls, live rescue/order contrasts and exact loss geometry."""
from __future__ import annotations

from contextlib import contextmanager
import random
import math
import numpy as np
import torch

from .calibrated_probes import require
from .mechanism_trace import OutputEdit, ParityTolerance, deletion_factors
from .metrics import behavioral_item

VERSION = "mechanistic-e3-injection-geometry-v1"


@contextmanager
def evaluation_mode(model):
    """Restore every child mode and Python/NumPy/Torch RNG, including exceptions."""
    modes = [(m,m.training) for m in model.modules()]
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        model.eval()
        with torch.inference_mode():
            yield
    finally:
        for module, mode in modes:
            module.training = mode
        random.setstate(py); np.random.set_state(np_state); torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


def equal_norm_injections(trace, mask, *, eta: float, norm_floor: float, control_seed: int,
                          query_min: int, nonsink_keys: tuple[int,int]) -> tuple[dict, dict]:
    """Entering-residual reference fixed by researcher; causal common control support."""
    require(type(eta) in (int,float) and math.isfinite(eta) and eta >= 0, "explicit nonnegative relative dose required")
    require(type(norm_floor) in (int,float) and math.isfinite(norm_floor) and norm_floor > 0, "positive norm floor required")
    require(type(control_seed) is int and 0 <= control_seed < 2**63, "separate control seed required")
    require(len(nonsink_keys) == 2 and len(set(nonsink_keys)) == 2 and
            all(type(k) is int and 0 < k < mask.shape[1] for k in nonsink_keys), "two preselected non-sink keys required")
    require(type(query_min) is int and max(nonsink_keys) <= query_min < mask.shape[1], "causal common q support required")
    eligible = mask & (torch.arange(mask.shape[1],device=mask.device)[None] >= query_min)
    delta = deletion_factors(trace)["projected_delta"].double()
    norms = delta.norm(dim=-1)
    unit = delta/norms[...,None].clamp_min(norm_floor)
    generator = torch.Generator(device="cpu").manual_seed(control_seed)
    noise = torch.randn(delta.shape,generator=generator,dtype=torch.float32).to(delta.device).double()
    orthogonal = noise - (noise*unit).sum(-1,keepdim=True)*unit
    vdiff = trace.value[:,:,nonsink_keys[1]]-trace.value[:,:,nonsink_keys[0]]
    nonsink = (vdiff.reshape(mask.shape[0],-1) @ trace.output_weight).double()[:,None].expand_as(delta)
    directions = {"sink":delta,"random":noise,"orthogonal":orthogonal,"non_sink":nonsink}
    available = {name:eligible & (vector.norm(dim=-1) >= norm_floor) for name,vector in directions.items()}
    # All comparisons edit exactly the same positions; unavailable counts remain explicit.
    common = eligible.clone()
    for support in available.values():
        common &= support
    reference_norm = trace.residual_input.double().norm(dim=-1)
    common &= reference_norm >= norm_floor
    requested_norm = eta*reference_norm
    edits, rows = {}, {}
    for name,vector in directions.items():
        norm = vector.norm(dim=-1)
        change = vector/norm[...,None].clamp_min(norm_floor)*requested_norm[...,None]
        change = torch.where(common[...,None],change,torch.zeros_like(change)).float()
        require(torch.isfinite(change).all(), "nonfinite normalized injection")
        edits[name] = change
        actual = change.double().norm(dim=-1)
        rows[name] = {"unavailable_direction_positions":int((eligible & ~available[name]).sum()),
            "injected_positions":int(common.sum()), "actual_norm_sum":float(actual[common].sum()),
            "requested_norm_sum":float(requested_norm[common].sum()),
            "max_norm_error":float((actual-requested_norm).abs()[common].max()) if common.any() else None}
    return edits, {"version":VERSION,"eta":eta,"norm_floor":norm_floor,"control_seed":control_seed,
        "reference":"clean_residual_input_before_ln_1","query_min":query_min,"nonsink_keys":list(nonsink_keys),
        "eligible_positions":int(eligible.sum()),"common_available_positions":int(common.sum()),
        "natural_sink_degenerate_positions":int((eligible & ~available["sink"]).sum()),
        "natural_projected_norm_sum":float(norms[eligible].sum()),
        "reference_below_floor_positions":int((eligible & (reference_norm < norm_floor)).sum()),
        "reference_residual_norm_sum":float(reference_norm[eligible].sum()),
        "common_support_mask":common.int().cpu().tolist(),"directions":rows,
        "interpretation":"within-checkpoint equal relative norm; directions/bases differ across checkpoints"}


def loss_geometry(clean, edited, ids, mask, *, tolerance: ParityTolerance, token_chunk: int) -> dict:
    """Exact per-target identity, stable double logsumexp; no full logits persisted."""
    require(clean.ndim == 3 and clean.shape == edited.shape and ids.shape == mask.shape == clean.shape[:2], "geometry shapes mismatch")
    require(mask.dtype == torch.bool and mask[:,0].all() and
            not ((~mask[:,:-1]) & mask[:,1:]).any(), "geometry requires boolean right padding")
    require(type(token_chunk) is int and token_chunk > 0, "positive token chunk required")
    require(torch.isfinite(clean).all() and torch.isfinite(edited).all(), "nonfinite logits")
    valid = mask[:,:-1] & mask[:,1:]
    require(valid.any(), "no valid next-token targets")
    targets = ids[:,1:][valid].long()
    c = clean[:,:-1][valid]; e = edited[:,:-1][valid]
    rows, errors = [], []
    for start in range(0,len(targets),token_chunk):
        z, ze, y = c[start:start+token_chunk].double(),e[start:start+token_chunk].double(),targets[start:start+token_chunk]
        delta = ze-z
        logp, logpe = z.log_softmax(-1),ze.log_softmax(-1)
        cy, ey = logp.gather(1,y[:,None]).squeeze(1),logpe.gather(1,y[:,None]).squeeze(1)
        dy = delta.gather(1,y[:,None]).squeeze(1)
        normalizer = torch.logsumexp(logp+delta,dim=-1)
        predicted, actual = -dy+normalizer, cy-ey
        errors.append(tolerance.check(predicted,actual,"exact_target_loss_identity"))
        self_kl = (logp.exp()*(logp-logpe)).sum(-1)
        raw_norm = delta.norm(dim=-1)
        centered = (delta-delta.mean(-1,keepdim=True)).norm(dim=-1)
        for i in range(len(y)):
            rows.append({"target_probability":float(cy[i].exp()),"clean_target_nll_nats":float(-cy[i]),
                "edited_target_nll_nats":float(-ey[i]),"target_logit_change":float(dy[i]),
                "log_normalizer_change":float(normalizer[i]),"delta_nll_nats":float(actual[i]),
                "self_kl_nats":float(self_kl[i]),"raw_logit_displacement_l2":float(raw_norm[i]),
                "centered_logit_displacement_l2":float(centered[i])})
    positions = torch.nonzero(valid,as_tuple=False).cpu().tolist()
    for row,(batch,query) in zip(rows,positions,strict=True):
        row.update(batch=batch,query=query,target_position=query+1)
    delta_values = [r["delta_nll_nats"] for r in rows]
    return {"version":VERSION,"valid_targets":len(rows),"target_records":rows,
        "delta_nll_sum_nats":math.fsum(delta_values),
        "positive_delta_nll_sum_nats":math.fsum(x for x in delta_values if x>0),
        "negative_delta_nll_sum_nats":math.fsum(x for x in delta_values if x<0),
        "identity_max_error":max(r["max_absolute_error"] for r in errors),
        "centered_logit_displacement_sum":math.fsum(r["centered_logit_displacement_l2"] for r in rows)}


def rescue_for_layer(adapter, ids, mask, layer, *, tolerance: ParityTolerance):
    clean = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,))
    donor = clean.traces[layer].attention_output
    direct = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),delete_layers=(layer,))
    restored = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),delete_layers=(layer,),
        output_edits={layer:OutputEdit("replace",donor)})
    check = tolerance.check(restored.outputs.logits,clean.outputs.logits,"single_layer_clean_output_rescue")
    all_layers = tuple(range(adapter.layer_count))
    all_deleted = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),delete_layers=all_layers)
    conditional = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(layer,),delete_layers=all_layers,
        output_edits={layer:OutputEdit("replace",donor)})
    return {"parity":check,"clean":clean,"single_deleted":direct,"single_rescued":restored,
        "all_deleted":all_deleted,"conditional_rescue":conditional,
        "interpretation":"all-layer plus one clean-output rescue is conditional, nonlinear and not a mediated fraction"}


def telescoping(adapter, ids, mask, *, order: tuple[int,...], tolerance: ParityTolerance) -> dict:
    require(len(order)==adapter.layer_count and set(order)==set(range(adapter.layer_count)) and
            all(type(i) is int for i in order), "explicit full layer permutation required")
    clean = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(order[0],))
    previous = 0.; increments=[]
    for k in range(1,len(order)+1):
        edited = adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(order[k-1],),delete_layers=order[:k])
        b = behavioral_item(clean.outputs.logits,edited.outputs.logits,ids,mask)
        delta = (b["edited_nll_sum_nats"]-b["clean_nll_sum_nats"])/b["valid_targets"]
        increments.append({"layer_added":order[k-1],"prefix":list(order[:k]),"delta_ce_from_clean":delta,
                           "increment_ce_nats":delta-previous})
        previous = delta
    summed = math.fsum(i["increment_ce_nats"] for i in increments)
    check = tolerance.check(torch.tensor(summed,dtype=torch.float64),torch.tensor(previous,dtype=torch.float64),"ordered_telescope")
    return {"version":VERSION,"order":list(order),"increments":increments,"all_layer_delta_ce_nats":previous,
        "summed_increment_ce_nats":summed,"closure":check,
        "interpretation":"increments depend on explicit deletion order; not independent layer contributions"}
