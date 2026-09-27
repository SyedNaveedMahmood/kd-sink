"""T02/T07 tiny CPU coverage for the narrow model adapters."""

from __future__ import annotations

import copy

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

from sinklab.interventions import AttentionIntervention
from sinklab.models import (
    GPT2Adapter,
    GPTNeoXAdapter,
    ModelAdapterError,
    ModelShape,
    adapt_causal_lm,
)


def _gpt2(*, reorder: bool = False) -> GPT2LMHeadModel:
    torch.manual_seed(117)
    return GPT2LMHeadModel(GPT2Config(
        vocab_size=41,
        n_positions=12,
        n_ctx=12,
        n_embd=24,
        n_layer=2,
        n_head=4,
        resid_pdrop=0.0,
        embd_pdrop=0.0,
        attn_pdrop=0.0,
        scale_attn_weights=True,
        scale_attn_by_inverse_layer_idx=True,
        reorder_and_upcast_attn=reorder,
        _attn_implementation="eager",
    )).cpu().eval()


def _neox() -> GPTNeoXForCausalLM:
    torch.manual_seed(117)
    return GPTNeoXForCausalLM(GPTNeoXConfig(
        vocab_size=41,
        max_position_embeddings=12,
        hidden_size=24,
        num_hidden_layers=2,
        num_attention_heads=4,
        intermediate_size=48,
        hidden_dropout=0.0,
        attention_dropout=0.0,
        rotary_pct=0.5,
        use_parallel_residual=True,
        _attn_implementation="eager",
    )).cpu().eval()


def _inputs() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]], dtype=torch.long),
        torch.tensor([[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]], dtype=torch.long),
    )


@pytest.mark.parametrize("factory,adapter_type", [(_gpt2, GPT2Adapter), (_neox, GPTNeoXAdapter)])
def test_native_and_clean_adapter_forward_are_exact_and_parameters_are_unchanged(factory, adapter_type):
    model = factory()
    ids, mask = _inputs()
    before = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    native = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    adapter = adapt_causal_lm(model, ModelShape(2, 4, 24))
    assert isinstance(adapter, adapter_type)
    adapted = adapter.forward(input_ids=ids, attention_mask=mask).logits
    torch.testing.assert_close(adapted, native, atol=0, rtol=0)
    for name, tensor in model.state_dict().items():
        torch.testing.assert_close(tensor, before[name], atol=0, rtol=0)


def test_gpt2_tied_weights_projection_layout_and_shape_contract_are_preserved():
    model = _gpt2()
    GPT2Adapter(model, ModelShape(2, 4, 24))
    assert model.transformer.wte.weight.data_ptr() == model.lm_head.weight.data_ptr()
    assert model.transformer.h[0].attn.c_attn.weight.shape == (24, 72)
    with pytest.raises(ModelAdapterError, match="shape mismatch"):
        GPT2Adapter(model, ModelShape(3, 4, 24))


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_pre_dropout_probabilities_match_native_eager_attentions_with_masks(factory):
    model = factory()
    ids, mask = _inputs()
    adapter = adapt_causal_lm(model)
    featured = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
    native = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        output_attentions=True,
    )
    assert len(featured.attention) == 2
    for features, native_probabilities in zip(featured.attention, native.attentions, strict=True):
        assert features.probabilities.dtype == torch.float32
        torch.testing.assert_close(features.probabilities, native_probabilities, atol=0, rtol=0)
        assert features.query.shape == (2, 5, 24)
        assert features.key.shape == (2, 5, 24)
        assert features.value.shape == (2, 5, 24)
        assert (features.probabilities.masked_select(
            features.valid_edges.expand_as(features.probabilities)
        ) >= 0).all()
        assert not features.valid_edges[1, :, 3:].any()  # padded query rows excluded
        assert not features.valid_edges[1, :, :, 3:].any()  # padded keys excluded
        future = torch.ones((5, 5), dtype=torch.bool).triu(1)[None, None]
        assert (features.probabilities.masked_select(future.expand_as(features.probabilities)) == 0).all()


def test_gpt2_qkv_matches_independent_fused_projection_and_scale_flags():
    model = _gpt2(reorder=True)
    ids, mask = _inputs()
    featured = GPT2Adapter(model).forward_with_features(input_ids=ids, attention_mask=mask)
    hidden = featured.outputs.hidden_states[0]
    manual = model.transformer.h[0].attn.c_attn(model.transformer.h[0].ln_1(hidden))
    manual_q, manual_k, manual_v = manual.split(24, dim=-1)
    torch.testing.assert_close(featured.attention[0].query, manual_q, atol=0, rtol=0)
    torch.testing.assert_close(featured.attention[0].key, manual_k, atol=0, rtol=0)
    torch.testing.assert_close(featured.attention[0].value, manual_v, atol=0, rtol=0)
    native = model(input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=True)
    for actual, expected in zip(featured.attention, native.attentions, strict=True):
        torch.testing.assert_close(actual.probabilities, expected, atol=0, rtol=0)


def test_neox_qk_include_native_rotary_positions_and_value_does_not():
    model = _neox()
    ids, mask = _inputs()
    featured = GPTNeoXAdapter(model).forward_with_features(input_ids=ids, attention_mask=mask)
    hidden = featured.outputs.hidden_states[0]
    layer = model.gpt_neox.layers[0]
    normalized = layer.input_layernorm(hidden)
    raw = layer.attention.query_key_value(normalized).view(2, 5, 4, 18).transpose(1, 2)
    raw_q, raw_k, raw_v = raw.chunk(3, dim=-1)
    actual_q = featured.attention[0].query.view(2, 5, 4, 6).transpose(1, 2)
    actual_k = featured.attention[0].key.view(2, 5, 4, 6).transpose(1, 2)
    actual_v = featured.attention[0].value.view(2, 5, 4, 6).transpose(1, 2)
    torch.testing.assert_close(actual_q[:, :, 0], raw_q[:, :, 0], atol=0, rtol=0)
    torch.testing.assert_close(actual_k[:, :, 0], raw_k[:, :, 0], atol=0, rtol=0)
    assert not torch.equal(actual_q[:, :, 1:], raw_q[:, :, 1:])
    assert not torch.equal(actual_k[:, :, 1:], raw_k[:, :, 1:])
    torch.testing.assert_close(actual_v, raw_v, atol=0, rtol=0)


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_explicit_noop_matches_native_forward_and_backward(factory):
    native_model = factory().train()
    edited_model = copy.deepcopy(native_model).train()
    ids, mask = _inputs()

    native_logits = native_model(input_ids=ids, attention_mask=mask, use_cache=False).logits
    native_loss = native_logits.square().mean()
    native_loss.backward()

    edited_logits = adapt_causal_lm(edited_model).forward(
        input_ids=ids,
        attention_mask=mask,
        intervention=AttentionIntervention("none"),
    ).logits
    edited_loss = edited_logits.square().mean()
    edited_loss.backward()

    torch.testing.assert_close(edited_logits, native_logits, atol=0, rtol=0)
    for (native_name, native_parameter), (edited_name, edited_parameter) in zip(
        native_model.named_parameters(), edited_model.named_parameters(), strict=True
    ):
        assert native_name == edited_name
        torch.testing.assert_close(edited_parameter.grad, native_parameter.grad, atol=0, rtol=0)


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_pure_feature_path_retains_qkv_and_probability_gradients(factory):
    model = factory().train()
    ids, mask = _inputs()
    featured = adapt_causal_lm(model).forward_with_features(input_ids=ids, attention_mask=mask)
    first = featured.attention[0]
    weights = torch.arange(first.probabilities.shape[-1], dtype=torch.float32)
    loss = (first.probabilities * weights).sum() + 1e-3 * (
        first.query.square().sum() + first.key.square().sum() + first.value.square().sum()
    )
    loss.backward()
    projection = (
        model.transformer.h[0].attn.c_attn.weight
        if isinstance(model, GPT2LMHeadModel)
        else model.gpt_neox.layers[0].attention.query_key_value.weight
    )
    assert projection.grad is not None
    assert torch.isfinite(projection.grad).all()
    assert projection.grad.abs().sum() > 0


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_pure_features_have_identical_gradients_under_activation_recomputation(factory):
    direct = factory().train()
    checkpointed = copy.deepcopy(direct).train()
    checkpointed.gradient_checkpointing_enable()
    ids, mask = _inputs()

    for model in (direct, checkpointed):
        featured = adapt_causal_lm(model).forward_with_features(input_ids=ids, attention_mask=mask)
        last = featured.attention[-1]
        loss = last.query.square().mean() + last.probabilities[..., 0].mean()
        loss.backward()

    for (direct_name, direct_parameter), (checkpointed_name, checkpointed_parameter) in zip(
        direct.named_parameters(), checkpointed.named_parameters(), strict=True
    ):
        assert direct_name == checkpointed_name
        torch.testing.assert_close(
            checkpointed_parameter.grad, direct_parameter.grad, atol=0, rtol=0
        )


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_feature_probabilities_are_normalized_before_attention_dropout(factory):
    model = factory().train()
    if isinstance(model, GPT2LMHeadModel):
        for block in model.transformer.h:
            block.attn.attn_dropout.p = 0.5
    else:
        for layer in model.gpt_neox.layers:
            layer.attention.attention_dropout = 0.5
    ids, mask = _inputs()
    featured = adapt_causal_lm(model).forward_with_features(input_ids=ids, attention_mask=mask)
    for layer in featured.attention:
        real_queries = mask[:, None, :].bool().expand(2, 4, 5)
        row_sums = layer.probabilities.sum(-1)
        torch.testing.assert_close(
            row_sums.masked_select(real_queries),
            torch.ones_like(row_sums.masked_select(real_queries)),
            atol=2e-6,
            rtol=0,
        )


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_editing_returned_attention_is_noncausal_but_pre_v_edit_changes_logits(factory):
    model = factory()
    ids, mask = _inputs()
    displayed = model(
        input_ids=ids,
        attention_mask=mask,
        use_cache=False,
        output_attentions=True,
    )
    fixed_logits = displayed.logits.detach().clone()
    with torch.no_grad():
        displayed.attentions[0].zero_()
    torch.testing.assert_close(displayed.logits, fixed_logits, atol=0, rtol=0)

    adapter = adapt_causal_lm(model)
    changed = adapter.forward(
        input_ids=ids,
        attention_mask=mask,
        intervention=AttentionIntervention("relocate"),
    ).logits
    assert (changed - fixed_logits).abs().max() > 1e-6


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_selected_layer_scope_q0_padding_and_probability_support(factory):
    model = factory()
    ids, mask = _inputs()
    adapter = adapt_causal_lm(model)
    clean = model(
        input_ids=ids, attention_mask=mask, use_cache=False, output_attentions=True
    )
    edited = adapter.forward(
        input_ids=ids,
        attention_mask=mask,
        output_attentions=True,
        intervention=AttentionIntervention("delete"),
        layer_scope=[1],
    )
    torch.testing.assert_close(edited.attentions[0], clean.attentions[0], atol=0, rtol=0)
    assert not torch.equal(edited.attentions[1][..., 1:, :], clean.attentions[1][..., 1:, :])
    torch.testing.assert_close(
        edited.attentions[1][..., 0, :], clean.attentions[1][..., 0, :], atol=0, rtol=0
    )
    torch.testing.assert_close(
        edited.attentions[1][1, :, 3:, :],
        clean.attentions[1][1, :, 3:, :],
        atol=0,
        rtol=0,
    )
    real_rows = mask[:, None, :].bool().expand(2, 4, 5)
    sums = edited.attentions[1].sum(-1).masked_select(real_rows)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=2e-6, rtol=0)
    future = torch.ones((5, 5), dtype=torch.bool).triu(1)[None, None]
    assert (edited.attentions[1].masked_select(future.expand_as(edited.attentions[1])) == 0).all()


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_transaction_restores_forward_modes_hooks_and_parameters_after_exception(factory):
    model = factory().train()
    adapter = adapt_causal_lm(model)
    modules = adapter._attention_modules()
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    before_training = {name: module.training for name, module in model.named_modules()}
    hook = modules[0].register_forward_hook(lambda module, args, output: None)
    hook_ids = tuple(modules[0]._forward_hooks)
    assert all("forward" not in module.__dict__ for module in modules)

    token_mask = torch.ones((1, 3), dtype=torch.bool)
    with pytest.raises(RuntimeError, match="forced"):
        with adapter.intervention_context(AttentionIntervention("delete"), token_mask):
            assert all("forward" in module.__dict__ for module in modules)
            raise RuntimeError("forced restoration check")

    assert all("forward" not in module.__dict__ for module in modules)
    assert tuple(modules[0]._forward_hooks) == hook_ids
    assert {name: module.training for name, module in model.named_modules()} == before_training
    for name, value in model.state_dict().items():
        torch.testing.assert_close(value, before[name], atol=0, rtol=0)
    hook.remove()


@pytest.mark.parametrize("factory", [_gpt2, _neox])
def test_model_forward_exception_also_restores_transaction(factory):
    model = factory()
    adapter = adapt_causal_lm(model)
    ids = torch.tensor([[999, 2]], dtype=torch.long)
    with pytest.raises((IndexError, RuntimeError)):
        adapter.forward(input_ids=ids, intervention=AttentionIntervention("delete"))
    assert all("forward" not in module.__dict__ for module in adapter._attention_modules())


def test_invalid_padding_and_cache_calls_fail_explicitly():
    adapter = GPT2Adapter(_gpt2())
    ids = torch.tensor([[1, 2, 3]], dtype=torch.long)
    with pytest.raises(ModelAdapterError, match="right-padded"):
        adapter.forward_with_features(input_ids=ids, attention_mask=torch.tensor([[1, 0, 1]]))
    with pytest.raises(ModelAdapterError, match="use_cache=False"):
        adapter.forward(input_ids=ids, use_cache=True)

