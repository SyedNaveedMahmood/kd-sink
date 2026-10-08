import copy
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel
from sinklab.models import GPT2Adapter
from sinklab.probes import apply_probe, probe_plan
from sinklab.calibrated_probes import (RouteProbe, apply_route, transport, observed_dose_match,
    validate_panel_split, route_scope, observed_forward, activation_effect)


def model_fixture(layers=2, length=12):
    with torch.random.fork_rng():
        torch.manual_seed(921)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=31, n_embd=24, n_head=3, n_layer=layers,
            n_positions=length, n_ctx=length, resid_pdrop=0., embd_pdrop=0., attn_pdrop=0.,
            _attn_implementation="eager" )).eval()
        for block in model.transformer.h:
            with torch.no_grad(): block.attn.c_attn.bias.normal_(0,.13)
    return model


def inputs():
    return torch.tensor([[1,2,3,4,5],[6,7,8,0,0]]), torch.tensor([[1,1,1,1,1],[1,1,1,0,0]],dtype=torch.bool)


@pytest.mark.parametrize("route,layers,coords", [("q_bias",(0,1),()),("k_input",(0,1),(1,2,3)),
    ("epe_transport",(0,),()),("position0_to1",(),()),("none",(),()),("reapply_q",(0,),())])
def test_zero_noop_and_exception_restore(route,layers,coords):
    model = model_fixture(); adapter=GPT2Adapter(model); ids,mask=inputs()
    before=copy.deepcopy(model.state_dict()); clean=adapter.forward(input_ids=ids,attention_mask=mask).logits
    with apply_route(model,RouteProbe(route,0.,layers,coords),mask):
        assert torch.equal(adapter.forward(input_ids=ids,attention_mask=mask).logits,clean)
    with pytest.raises(RuntimeError):
        with apply_route(model,RouteProbe(route,.5,layers,coords),mask):
            raise RuntimeError("injected failure")
    for k,v in model.state_dict().items(): assert torch.equal(v,before[k])
    assert not model.transformer.h[0].mlp._forward_hooks


@pytest.mark.parametrize("route,legacy,layers,coords", [("q_bias","q_bias_all",(0,1),()),
    ("k_input","k_top3_all",(0,1),(1,2,3)),("epe_transport","epe_transport_layer0",(0,),()),
    ("position0_to1","position0_to1",(),())])
@torch.no_grad()
def test_alpha1_matches_unchanged_s4(route,legacy,layers,coords):
    model=model_fixture(); ids,mask=inputs(); adapter=GPT2Adapter(model)
    plan=probe_plan(model,control_seed=18)
    if coords: plan[legacy]["coordinates"]=list(coords)
    with apply_probe(model,legacy,plan=plan,mask=mask):
        old=adapter.forward(input_ids=ids,attention_mask=mask).logits
    with apply_route(model,RouteProbe(route,1.,layers,coords),mask):
        new=adapter.forward(input_ids=ids,attention_mask=mask).logits
    torch.testing.assert_close(old,new,atol=0,rtol=0)


def test_q_and_k_touch_only_their_named_slices_and_live_activations():
    model=model_fixture(); ids,mask=inputs(); adapter=GPT2Adapter(model)
    with torch.no_grad(): clean,observations=observed_forward(adapter,ids,mask)
    native_hooks=dict(model.transformer.h[0].attn._forward_hooks)
    weight=model.transformer.h[0].attn.c_attn.weight.detach().clone()
    bias=model.transformer.h[0].attn.c_attn.bias.detach().clone()
    for probe in [RouteProbe("q_bias",.5,(0,)),RouteProbe("k_input",.5,(0,),(1,2,3))]:
        with apply_route(model,probe,mask) as info:
            with torch.no_grad(): edited,obs=observed_forward(adapter,ids,mask)
            assert not torch.equal(edited.logits,clean.logits)
            assert info["parameter_delta_l2"]>0
            if probe.route=="q_bias":
                assert torch.equal(model.transformer.h[0].attn.c_attn.bias[24:],bias[24:])
                assert torch.equal(model.transformer.h[0].attn.c_attn.weight,weight)
                assert torch.equal(obs[0]["k"],observations[0]["k"])
            else:
                difference=model.transformer.h[0].attn.c_attn.weight-weight
                allowed=torch.zeros_like(difference,dtype=torch.bool);allowed[[1,2,3],24:48]=True
                assert not difference[~allowed].any()
                assert torch.equal(obs[0]["q"],observations[0]["q"])
            effects=activation_effect(observations,obs,mask)
            assert effects["attention_output_delta_rms"]>0
    assert dict(model.transformer.h[0].attn._forward_hooks)==native_hooks


def test_epe_batch_reference_sum_conservation_and_gradients():
    with torch.random.fork_rng():
        torch.manual_seed(1);output=torch.randn(2,5,24,requires_grad=True);u0=torch.randn(24);u1=torch.randn(24)
    mask=torch.ones(2,5,dtype=torch.bool)
    edited=transport(output,u0,u1,mask,.5)
    for b in range(2):
        change=.5*(output[b,0]*u0).sum()*(u1-u0)
        torch.testing.assert_close(edited[b,0],output[b,0]+change)
    torch.testing.assert_close(edited[:,:2].sum(1),output[:,:2].sum(1),atol=1e-5,rtol=1e-5)
    assert torch.equal(edited[:,2:],output[:,2:]); edited.square().sum().backward()
    assert torch.isfinite(output.grad).all()


def test_effective_dose_matching_never_extrapolates_or_selects_outcomes():
    a=[{"alpha":x,"sink_removed_fraction":x} for x in [0.,.5,1.]]
    b=[{"alpha":x,"sink_removed_fraction":x/2} for x in [0.,.5,1.]]
    result=observed_dose_match(a,b,dose_metric="sink_removed_fraction",target=.25)
    assert result["status"]=="observed_overlap" and result["right_candidates"][0]["alpha"]==.5
    assert observed_dose_match(a,b,dose_metric="sink_removed_fraction",target=.75)["status"]=="dose_unmatched"
    a[-1]["sink_removed_fraction"]=0
    assert observed_dose_match(a,b,dose_metric="sink_removed_fraction",target=.25)["status"]=="ambiguous_nonmonotonic"
    with pytest.raises(ValueError): observed_dose_match(a,b,dose_metric="delta_ce_nats",target=.25)


def test_document_split_and_explicit_teacher_mapping():
    a=[{"id":"a","document_ids":["doc-a"],"document_hashes":["a"*64]}]
    b=[{"id":"b","document_ids":["doc-b"],"document_hashes":["b"*64]}]
    assert validate_panel_split(a,b)["status"]=="document_disjoint"
    b[0]["document_hashes"]=["a"*64]
    with pytest.raises(ValueError,match="overlap"):validate_panel_split(a,b)
    with pytest.raises(ValueError,match="document"):validate_panel_split([{"id":"a"}],b)
    assert route_scope(role="teacher",kind="mapped_teacher",layer_count=4,student_scope=(0,1),teacher_map=(0,3))==(0,3)
    with pytest.raises(ValueError):route_scope(role="student",kind="mapped_teacher",layer_count=4,student_scope=(0,1),teacher_map=(0,3))


@pytest.mark.parametrize("alpha",[-1,2,True,float('nan'),float('inf')])
def test_invalid_doses(alpha):
    with pytest.raises(ValueError):RouteProbe("q_bias",alpha,(0,))
