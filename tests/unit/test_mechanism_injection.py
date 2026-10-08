from dataclasses import replace
import random
import numpy as np
import pytest
import torch
from test_calibrated_probes import model_fixture,inputs
from sinklab.mechanism_trace import MechanisticGPT2Adapter,OutputEdit,ParityTolerance
from sinklab.mechanism_injection import (equal_norm_injections,loss_geometry,rescue_for_layer,telescoping,evaluation_mode)
from sinklab.metrics import behavioral_item

TOL=ParityTolerance(2e-6,2e-5)


@torch.no_grad()
def test_equal_relative_norm_controls_support_determinism_and_orthogonality():
    model=model_fixture();ids,mask=inputs();adapter=MechanisticGPT2Adapter(model)
    clean=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,));trace=clean.traces[0]
    kwargs=dict(eta=.03,norm_floor=1e-10,control_seed=811,query_min=2,nonsink_keys=(1,2))
    rng=torch.get_rng_state().clone()
    edits,meta=equal_norm_injections(trace,mask,**kwargs)
    again,_=equal_norm_injections(trace,mask,**kwargs)
    assert torch.equal(rng,torch.get_rng_state())
    common=torch.tensor(meta['common_support_mask'],dtype=torch.bool)
    assert meta['reference']=='clean_residual_input_before_ln_1' and meta['eligible_positions']==4
    expected=.03*trace.residual_input.double().norm(dim=-1)[common]
    for name,delta in edits.items():
        assert torch.equal(delta,again[name]) and not delta[~common].any()
        torch.testing.assert_close(delta.double().norm(dim=-1)[common],expected,atol=1e-9,rtol=1e-6)
        assert not delta[:,0].any() and not delta[~mask].any()
    inner=(edits['sink'].double()*edits['orthogonal'].double()).sum(-1)[common]
    assert inner.abs().max()<1e-10
    for name,delta in edits.items():
        result=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,),output_edits={0:OutputEdit('add',delta)})
        assert not torch.equal(result.outputs.logits,clean.outputs.logits)


@torch.no_grad()
def test_zero_dose_and_degenerate_natural_directions_are_explicit():
    model=model_fixture();ids,mask=inputs();adapter=MechanisticGPT2Adapter(model)
    clean=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,));trace=clean.traces[0]
    kwargs=dict(norm_floor=1e-8,control_seed=4,query_min=2,nonsink_keys=(1,2))
    edits,meta=equal_norm_injections(trace,mask,eta=0.,**kwargs)
    assert all(not delta.any() for delta in edits.values())
    result=adapter.traced_forward(input_ids=ids,attention_mask=mask,trace_layers=(0,),output_edits={0:OutputEdit('add',edits['sink'])})
    assert torch.equal(result.outputs.logits,clean.outputs.logits)
    trace=replace(trace,output_weight=torch.zeros_like(trace.output_weight))
    edits,meta=equal_norm_injections(trace,mask,eta=.1,**kwargs)
    assert meta['natural_sink_degenerate_positions']==meta['eligible_positions'] and meta['common_available_positions']==0
    assert all(not delta.any() for delta in edits.values())
    assert meta['directions']['sink']['max_norm_error'] is None


@torch.no_grad()
def test_rescue_and_alternate_orders_are_live_and_close():
    model=model_fixture();ids,mask=inputs();adapter=MechanisticGPT2Adapter(model)
    rescued=rescue_for_layer(adapter,ids,mask,1,tolerance=TOL)
    assert not torch.equal(rescued['single_deleted'].outputs.logits,rescued['clean'].outputs.logits)
    torch.testing.assert_close(rescued['single_rescued'].outputs.logits,rescued['clean'].outputs.logits,atol=2e-6,rtol=2e-5)
    for order in [(0,1),(1,0)]:
        result=telescoping(adapter,ids,mask,order=order,tolerance=TOL)
        assert result['summed_increment_ce_nats']==pytest.approx(result['all_layer_delta_ce_nats'],abs=1e-12)
    first=telescoping(adapter,ids,mask,order=(0,1),tolerance=TOL)
    second=telescoping(adapter,ids,mask,order=(1,0),tolerance=TOL)
    assert first['all_layer_delta_ce_nats']==second['all_layer_delta_ce_nats']
    assert first['increments'][0]['increment_ce_nats']!=second['increments'][1]['increment_ce_nats']


def test_exact_geometry_matches_independent_ce_and_preserves_signed_cancellation():
    ids=torch.tensor([[0,1,2,0],[1,0,1,0]])
    mask=torch.tensor([[1,1,1,1],[1,1,1,0]],dtype=torch.bool)
    with torch.random.fork_rng():
        torch.manual_seed(22);clean=torch.randn(2,4,3);edited=clean+torch.randn(2,4,3)*4
    result=loss_geometry(clean,edited,ids,mask,tolerance=ParityTolerance(1e-10,1e-10),token_chunk=2)
    valid=mask[:,:-1]&mask[:,1:]
    independent=torch.nn.functional.cross_entropy(edited[:,:-1][valid].double(),ids[:,1:][valid],reduction='none')-torch.nn.functional.cross_entropy(clean[:,:-1][valid].double(),ids[:,1:][valid],reduction='none')
    assert result['valid_targets']==5
    torch.testing.assert_close(torch.tensor([r['delta_nll_nats'] for r in result['target_records']],dtype=torch.float64),independent,atol=1e-12,rtol=1e-12)
    assert result['positive_delta_nll_sum_nats']>0 and result['negative_delta_nll_sum_nats']<0
    assert result['delta_nll_sum_nats']==pytest.approx(float(independent.sum()))
    assert [(r['query'],r['target_position']) for r in result['target_records']]==[(0,1),(1,2),(2,3),(0,1),(1,2)]
    assert loss_geometry(clean,edited,ids,mask,tolerance=ParityTolerance(1e-10,1e-10),token_chunk=1)['target_records']==result['target_records']


def test_geometry_stable_large_logit_shifts_and_gauge_invariance():
    clean=torch.tensor([[[1000.,0.,-1000.],[.1,.2,.3]]],dtype=torch.float64)
    edited=clean+800.
    ids=torch.tensor([[0,1]]);mask=torch.ones(1,2,dtype=torch.bool)
    result=loss_geometry(clean,edited,ids,mask,tolerance=ParityTolerance(1e-10,1e-10),token_chunk=1)
    row=result['target_records'][0]
    assert row['delta_nll_nats']==pytest.approx(0.,abs=1e-10) and row['self_kl_nats']==pytest.approx(0.,abs=1e-10)
    assert row['centered_logit_displacement_l2']==0 and row['raw_logit_displacement_l2']>0


def test_rng_and_every_child_mode_restore_after_failure():
    model=model_fixture();model.train();model.transformer.h[0].mlp.eval()
    modes=[m.training for m in model.modules()];state=torch.get_rng_state().clone();py=random.getstate();nps=np.random.get_state()
    with pytest.raises(RuntimeError):
        with evaluation_mode(model):
            assert not any(m.training for m in model.modules())
            torch.rand(4);np.random.rand();random.random();raise RuntimeError('fault')
    assert [m.training for m in model.modules()]==modes and torch.equal(torch.get_rng_state(),state)
    assert random.getstate()==py and np.array_equal(np.random.get_state()[1],nps[1])
