import copy
from dataclasses import replace
import pytest
import torch
from test_calibrated_probes import model_fixture,inputs
from sinklab.mechanism_trace import (MechanisticGPT2Adapter,OutputEdit,ParityTolerance,
    deletion_factors,factor_summary,isolated_parity)


@pytest.mark.parametrize("layer",[0,1])
@torch.no_grad()
def test_live_trace_and_direct_injection_parity(layer):
    model=model_fixture(); adapter=MechanisticGPT2Adapter(model); ids,mask=inputs()
    before=copy.deepcopy(model.state_dict()); native=adapter.forward(input_ids=ids,attention_mask=mask).logits
    native_hooks={id(m):dict(m._forward_hooks) for m in model.modules()}
    clean,parity=isolated_parity(adapter,ids,mask,layer,ParityTolerance(2e-6,2e-5))
    assert torch.equal(clean.outputs.logits,native)
    trace=clean.traces[layer];factor=parity["factors"]
    assert not factor["projected_delta"][~mask].any() and not factor["projected_delta"][:,0].any()
    assert not torch.equal(parity["direct"].outputs.logits,native)
    summary=factor_summary(trace,mask,denominator_floor=1e-8)
    assert summary["all_q_ge1"]["real_query_count"]==6 and summary["second_half_queries"]["real_query_count"]==3
    assert len(summary["all_q_ge1"]["per_head_local_delta_norm_mean"])==3
    for k,v in model.state_dict().items():assert torch.equal(v,before[k])
    assert {id(m):dict(m._forward_hooks) for m in model.modules()}==native_hooks


def test_trace_does_not_detach_live_graph_and_residual_reference_is_pre_layernorm():
    model=model_fixture(); adapter=MechanisticGPT2Adapter(model);ids,mask=inputs()
    result=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,))
    trace=result.traces[0]
    embedding=model.transformer.wte(ids)+model.transformer.wpe(torch.arange(ids.shape[1]))
    torch.testing.assert_close(trace.residual_input,embedding,atol=0,rtol=0)
    assert trace.pre_projection.requires_grad and trace.post_projection.requires_grad
    (deletion_factors(trace)["projected_delta"].square().sum()+result.outputs.logits.square().sum()).backward()
    assert model.transformer.h[0].attn.c_attn.weight.grad is not None


@torch.no_grad()
def test_conditional_deletion_is_stable_at_rounded_unit_sink_mass():
    model=model_fixture(); adapter=MechanisticGPT2Adapter(model);ids,mask=inputs()
    trace=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,)).traces[0]
    scores=torch.zeros_like(trace.scores);scores[...,0]=1000
    probabilities=torch.softmax(scores.masked_fill(~trace.valid_edges,-torch.inf),-1)
    # Ignore padded queries exactly as the native forward; their values are excluded by support.
    probabilities=torch.nan_to_num(probabilities)
    saturated=replace(trace,scores=scores,probabilities=probabilities)
    factors=deletion_factors(saturated)
    assert probabilities[0,:,1:,0].eq(1).all()
    assert torch.isfinite(factors["projected_delta"]).all()
    expected=trace.value[0,:,1,:]-trace.value[0,:,0,:]
    torch.testing.assert_close(factors["head_delta"][0,:,1,:],expected,atol=0,rtol=0)


def test_trace_error_restores_forward_methods_and_hooks():
    model=model_fixture();adapter=MechanisticGPT2Adapter(model);ids,mask=inputs()
    adapter.forward(input_ids=ids,attention_mask=mask)
    hook_snapshot={id(m):dict(m._forward_hooks) for m in model.modules()}
    forward_snapshot=[b.attn.__dict__.get('forward') for b in model.transformer.h]
    handle=model.transformer.h[0].attn.c_proj.register_forward_hook(lambda *_: (_ for _ in ()).throw(RuntimeError('fault')))
    try:
        with pytest.raises(RuntimeError,match='fault'):
            adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,),delete_layers=(0,))
    finally:handle.remove()
    assert [b.attn.__dict__.get('forward') for b in model.transformer.h]==forward_snapshot
    assert {id(m):dict(m._forward_hooks) for m in model.modules()}==hook_snapshot


def test_parity_failure_and_unsupported_inputs_fail_closed():
    with pytest.raises(ValueError,match='parity failed'):ParityTolerance(0,0).check(torch.ones(1),torch.zeros(1),'bad')
    model=model_fixture();adapter=MechanisticGPT2Adapter(model);ids,mask=inputs()
    bad=torch.ones(2,5,24)
    with pytest.raises(ValueError,match='q0'):adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,),output_edits={0:OutputEdit('add',bad)})
    with torch.autocast('cpu',dtype=torch.bfloat16):
        with pytest.raises(ValueError,match='autocast'):adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,))
    model.train()
    with pytest.raises(ValueError,match='eval'):adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,))
    model.eval().half()
    with pytest.raises(ValueError,match='FP32'):adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,))
