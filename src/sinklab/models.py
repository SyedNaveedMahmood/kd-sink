"""Narrow eager-attention adapters for GPT-2 and GPT-NeoX.

Clean forwards delegate to the pinned Transformers implementation. Feature
collection is a pure recomputation from returned layer inputs. Interventions
transactionally replace only attention-module forwards and edit normalized
probabilities before value aggregation; parameters and module structure are
never replaced.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from types import MethodType
from typing import Any, Iterator, Mapping, Sequence

import torch
from torch import nn
from transformers import GPT2LMHeadModel, GPTNeoXForCausalLM
from transformers.masking_utils import create_causal_mask
from transformers.models.gpt_neox.modeling_gpt_neox import apply_rotary_pos_emb

from .interventions import (
    AttentionIntervention,
    InterventionError,
    apply_attention_intervention,
    normalize_layer_scope,
)


class ModelAdapterError(ValueError):
    """Raised when a model or call cannot satisfy the adapter contract."""


@dataclass(frozen=True)
class ModelShape:
    layers: int
    heads: int
    width: int


@dataclass(frozen=True)
class AttentionFeatures:
    """One layer's native projected features and pre-dropout probabilities."""

    layer: int
    scores: torch.Tensor
    probabilities: torch.Tensor
    query: torch.Tensor
    key: torch.Tensor
    value: torch.Tensor
    valid_edges: torch.Tensor


@dataclass(frozen=True)
class FeatureForward:
    outputs: Any
    attention: tuple[AttentionFeatures, ...]


def _right_padding_mask(
    attention_mask: torch.Tensor | None, batch: int, length: int, device: torch.device
) -> torch.Tensor:
    if attention_mask is None:
        return torch.ones((batch, length), dtype=torch.bool, device=device)
    if attention_mask.ndim != 2 or tuple(attention_mask.shape) != (batch, length):
        raise ModelAdapterError("attention_mask must have shape [batch, sequence]")
    mask = attention_mask.to(device=device, dtype=torch.bool)
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ModelAdapterError("only right-padded attention masks are supported")
    if (~mask[:, 0]).any():
        raise ModelAdapterError("each sequence must contain at least one real token")
    return mask


def _valid_edges(token_mask: torch.Tensor) -> torch.Tensor:
    length = token_mask.shape[1]
    causal = torch.ones((length, length), dtype=torch.bool, device=token_mask.device).tril()
    return token_mask[:, None, :, None] & token_mask[:, None, None, :] & causal[None, None]


def _native_causal_mask(
    config: Any,
    hidden: torch.Tensor,
    token_mask: torch.Tensor,
    position_ids: torch.Tensor,
) -> torch.Tensor | None:
    length = hidden.shape[1]
    cache_position = torch.arange(length, device=hidden.device)
    return create_causal_mask(
        config=config,
        inputs_embeds=hidden,
        attention_mask=token_mask,
        cache_position=cache_position,
        past_key_values=None,
        position_ids=position_ids,
    )


def _masked_probabilities(scores: torch.Tensor, additive_mask: torch.Tensor | None) -> torch.Tensor:
    masked = scores.float()
    if additive_mask is not None:
        masked = masked + additive_mask.float()
    probabilities = torch.softmax(masked, dim=-1)
    if not torch.isfinite(probabilities).all():
        raise InterventionError("native masked softmax produced non-finite probabilities")
    return probabilities


def _gpt2_scores(attention: nn.Module, query: torch.Tensor, key: torch.Tensor) -> torch.Tensor:
    if attention.reorder_and_upcast_attn:
        batch, heads, query_length, head_dim = query.shape
        key_length = key.shape[-2]
        scale = 1.0
        if attention.scale_attn_weights:
            scale /= float(head_dim) ** 0.5
        if attention.scale_attn_by_inverse_layer_idx:
            scale /= float(attention.layer_idx + 1)
        output = torch.empty(
            batch * heads,
            query_length,
            key_length,
            dtype=torch.float32,
            device=query.device,
        )
        q = query.reshape(-1, query_length, head_dim)
        k = key.transpose(-1, -2).reshape(-1, head_dim, key_length)
        return torch.baddbmm(output, q.float(), k.float(), beta=0, alpha=scale).reshape(
            batch, heads, query_length, key_length
        )
    scores = torch.matmul(query, key.transpose(-1, -2))
    if attention.scale_attn_weights:
        scores = scores / torch.full(
            [], query.shape[-1] ** 0.5, dtype=scores.dtype, device=scores.device
        )
    if attention.scale_attn_by_inverse_layer_idx:
        scores = scores / float(attention.layer_idx + 1)
    return scores.float()


class _BaseAdapter:
    model: nn.Module

    def __init__(self, model: nn.Module, expected_shape: ModelShape | None = None) -> None:
        self.model = model
        self._validate_model(expected_shape)

    @property
    def layer_count(self) -> int:
        return len(self._attention_modules())

    def _validate_model(self, expected_shape: ModelShape | None) -> None:
        raise NotImplementedError

    def _attention_modules(self) -> tuple[nn.Module, ...]:
        raise NotImplementedError

    def _patched_forward(self, module: nn.Module, spec: AttentionIntervention, valid_edges: torch.Tensor):
        raise NotImplementedError

    def _features_from_hidden(
        self,
        hidden_states: Sequence[torch.Tensor],
        token_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[AttentionFeatures, ...]:
        raise NotImplementedError

    def forward(self, *, intervention: AttentionIntervention | None = None,
                layer_scope: Sequence[int] | None = None, **model_inputs: Any) -> Any:
        """Run a clean or causally edited full-context forward."""

        if model_inputs.get("use_cache") is True:
            raise ModelAdapterError("adapter forwards require use_cache=False")
        model_inputs["use_cache"] = False
        if intervention is None:
            return self.model(**model_inputs)

        reference = model_inputs.get("input_ids")
        if reference is None:
            reference = model_inputs.get("inputs_embeds")
        if reference is None or reference.ndim < 2:
            raise ModelAdapterError("input_ids or inputs_embeds is required")
        batch, length = reference.shape[:2]
        token_mask = _right_padding_mask(
            model_inputs.get("attention_mask"), batch, length, reference.device
        )
        scope = normalize_layer_scope(layer_scope, self.layer_count)
        with self.intervention_context(intervention, token_mask, scope):
            return self.model(**model_inputs)

    def forward_with_features(self, **model_inputs: Any) -> FeatureForward:
        """Return native outputs plus pure, differentiable attention features."""

        if model_inputs.get("use_cache") is True:
            raise ModelAdapterError("feature forwards require use_cache=False")
        if "past_key_values" in model_inputs and model_inputs["past_key_values"] is not None:
            raise ModelAdapterError("feature forwards do not accept a KV cache")
        reference = model_inputs.get("input_ids")
        if reference is None:
            reference = model_inputs.get("inputs_embeds")
        if reference is None or reference.ndim < 2:
            raise ModelAdapterError("input_ids or inputs_embeds is required")
        batch, length = reference.shape[:2]
        token_mask = _right_padding_mask(
            model_inputs.get("attention_mask"), batch, length, reference.device
        )
        position_ids = model_inputs.get("position_ids")
        if position_ids is None:
            position_ids = torch.arange(length, device=reference.device).unsqueeze(0)
        if (
            position_ids.ndim != 2
            or position_ids.shape[-1] != length
            or position_ids.shape[0] not in {1, batch}
        ):
            raise ModelAdapterError("position_ids must have shape [1|batch, sequence]")

        call_inputs = dict(model_inputs)
        call_inputs["use_cache"] = False
        call_inputs["output_hidden_states"] = True
        outputs = self.model(**call_inputs)
        hidden_states = outputs.hidden_states
        if hidden_states is None or len(hidden_states) != self.layer_count + 1:
            raise ModelAdapterError("Transformers did not return the expected layer hidden states")
        features = self._features_from_hidden(hidden_states[:-1], token_mask, position_ids)
        return FeatureForward(outputs=outputs, attention=features)

    @contextmanager
    def intervention_context(
        self,
        intervention: AttentionIntervention,
        token_mask: torch.Tensor,
        layer_scope: Sequence[int] | None = None,
    ) -> Iterator[None]:
        """Transactionally install causal attention forwards and always restore them."""

        modules = self._attention_modules()
        scope = normalize_layer_scope(layer_scope, len(modules))
        if token_mask.dtype != torch.bool or token_mask.ndim != 2:
            raise ModelAdapterError("transaction token_mask must be boolean [batch, sequence]")
        if token_mask.shape[1] == 0 or (~token_mask[:, 0]).any():
            raise ModelAdapterError("transaction token_mask requires at least one real token per row")
        valid_edges = _valid_edges(token_mask)
        saved: list[tuple[nn.Module, bool, Any]] = []
        try:
            for index in scope:
                module = modules[index]
                had_instance_forward = "forward" in module.__dict__
                original = module.__dict__.get("forward")
                saved.append((module, had_instance_forward, original))
                module.forward = MethodType(
                    self._patched_forward(module, intervention, valid_edges), module
                )
            yield
        finally:
            for module, had_instance_forward, original in reversed(saved):
                if had_instance_forward:
                    module.forward = original
                elif "forward" in module.__dict__:
                    delattr(module, "forward")


class GPT2Adapter(_BaseAdapter):
    def __init__(self, model: GPT2LMHeadModel, expected_shape: ModelShape | None = None) -> None:
        if not isinstance(model, GPT2LMHeadModel):
            raise ModelAdapterError("GPT2Adapter requires GPT2LMHeadModel")
        super().__init__(model, expected_shape)

    def _attention_modules(self) -> tuple[nn.Module, ...]:
        return tuple(block.attn for block in self.model.transformer.h)

    def _validate_model(self, expected_shape: ModelShape | None) -> None:
        config = self.model.config
        actual = ModelShape(config.n_layer, config.n_head, config.n_embd)
        if expected_shape is not None and actual != expected_shape:
            raise ModelAdapterError(f"GPT-2 shape mismatch: expected {expected_shape}, got {actual}")
        if config._attn_implementation != "eager":
            raise ModelAdapterError("GPT-2 adapter requires eager attention")
        if config.add_cross_attention:
            raise ModelAdapterError("GPT-2 cross-attention is outside this adapter")
        if len(self.model.transformer.h) != config.n_layer:
            raise ModelAdapterError("GPT-2 layer count does not match its config")
        for block in self.model.transformer.h:
            attention = block.attn
            if attention.num_heads != config.n_head or attention.embed_dim != config.n_embd:
                raise ModelAdapterError("GPT-2 attention shape does not match its config")
            if tuple(attention.c_attn.weight.shape) != (config.n_embd, 3 * config.n_embd):
                raise ModelAdapterError("unexpected GPT-2 fused QKV projection layout")
        if config.tie_word_embeddings and (
            self.model.get_input_embeddings().weight.data_ptr()
            != self.model.get_output_embeddings().weight.data_ptr()
        ):
            raise ModelAdapterError("GPT-2 input/output embeddings are not tied")

    def _patched_forward(self, module: nn.Module, spec: AttentionIntervention, valid_edges: torch.Tensor):
        def forward(
            attention: nn.Module,
            hidden_states: torch.Tensor,
            past_key_values: Any = None,
            cache_position: torch.Tensor | None = None,
            attention_mask: torch.Tensor | None = None,
            encoder_hidden_states: torch.Tensor | None = None,
            encoder_attention_mask: torch.Tensor | None = None,
            output_attentions: bool | None = False,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_position, output_attentions, kwargs
            if past_key_values is not None or encoder_hidden_states is not None or encoder_attention_mask is not None:
                raise ModelAdapterError("causal GPT-2 edits require cache-free self-attention")
            query, key, value = attention.c_attn(hidden_states).split(attention.split_size, dim=2)
            shape = (*query.shape[:-1], -1, attention.head_dim)
            query = query.view(shape).transpose(1, 2)
            key = key.view(shape).transpose(1, 2)
            value = value.view(shape).transpose(1, 2)
            scores = _gpt2_scores(attention, query, key)
            probabilities = _masked_probabilities(scores, attention_mask)
            edited = apply_attention_intervention(probabilities, scores, valid_edges, spec)
            aggregate_weights = attention.attn_dropout(edited.to(value.dtype))
            output = torch.matmul(aggregate_weights, value).transpose(1, 2)
            output = output.reshape(*output.shape[:-2], -1).contiguous()
            output = attention.c_proj(output)
            output = attention.resid_dropout(output)
            return output, aggregate_weights

        return forward

    def _features_from_hidden(
        self,
        hidden_states: Sequence[torch.Tensor],
        token_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[AttentionFeatures, ...]:
        additive_mask = _native_causal_mask(
            self.model.config, hidden_states[0], token_mask, position_ids
        )
        valid_edges = _valid_edges(token_mask)
        result: list[AttentionFeatures] = []
        for index, (block, hidden) in enumerate(zip(self.model.transformer.h, hidden_states, strict=True)):
            attention = block.attn
            projected = attention.c_attn(block.ln_1(hidden))
            query_flat, key_flat, value_flat = projected.split(attention.split_size, dim=2)
            shape = (*query_flat.shape[:-1], -1, attention.head_dim)
            query = query_flat.view(shape).transpose(1, 2)
            key = key_flat.view(shape).transpose(1, 2)
            scores = _gpt2_scores(attention, query, key)
            probabilities = _masked_probabilities(scores, additive_mask)
            result.append(AttentionFeatures(
                layer=index,
                scores=scores,
                probabilities=probabilities,
                query=query_flat,
                key=key_flat,
                value=value_flat,
                valid_edges=valid_edges,
            ))
        return tuple(result)


class GPTNeoXAdapter(_BaseAdapter):
    def __init__(self, model: GPTNeoXForCausalLM, expected_shape: ModelShape | None = None) -> None:
        if not isinstance(model, GPTNeoXForCausalLM):
            raise ModelAdapterError("GPTNeoXAdapter requires GPTNeoXForCausalLM")
        super().__init__(model, expected_shape)

    def _attention_modules(self) -> tuple[nn.Module, ...]:
        return tuple(layer.attention for layer in self.model.gpt_neox.layers)

    def _validate_model(self, expected_shape: ModelShape | None) -> None:
        config = self.model.config
        actual = ModelShape(config.num_hidden_layers, config.num_attention_heads, config.hidden_size)
        if expected_shape is not None and actual != expected_shape:
            raise ModelAdapterError(f"GPT-NeoX shape mismatch: expected {expected_shape}, got {actual}")
        if config._attn_implementation != "eager":
            raise ModelAdapterError("GPT-NeoX adapter requires eager attention")
        if len(self.model.gpt_neox.layers) != config.num_hidden_layers:
            raise ModelAdapterError("GPT-NeoX layer count does not match its config")
        for layer in self.model.gpt_neox.layers:
            attention = layer.attention
            if attention.head_size * config.num_attention_heads != config.hidden_size:
                raise ModelAdapterError("GPT-NeoX attention shape does not match its config")
            if tuple(attention.query_key_value.weight.shape) != (
                3 * config.hidden_size,
                config.hidden_size,
            ):
                raise ModelAdapterError("unexpected GPT-NeoX fused QKV projection layout")
        if config.tie_word_embeddings and (
            self.model.get_input_embeddings().weight.data_ptr()
            != self.model.get_output_embeddings().weight.data_ptr()
        ):
            raise ModelAdapterError("GPT-NeoX input/output embeddings are not tied")

    def _patched_forward(self, module: nn.Module, spec: AttentionIntervention, valid_edges: torch.Tensor):
        def forward(
            attention: nn.Module,
            hidden_states: torch.Tensor,
            attention_mask: torch.Tensor | None,
            layer_past: Any = None,
            cache_position: torch.Tensor | None = None,
            position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            del cache_position, kwargs
            if layer_past is not None:
                raise ModelAdapterError("causal GPT-NeoX edits require cache-free self-attention")
            if position_embeddings is None:
                raise ModelAdapterError("GPT-NeoX position embeddings are required")
            input_shape = hidden_states.shape[:-1]
            shape = (*input_shape, -1, 3 * attention.head_size)
            qkv = attention.query_key_value(hidden_states).view(shape).transpose(1, 2)
            query, key, value = qkv.chunk(3, dim=-1)
            cos, sin = position_embeddings
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            scores = (torch.matmul(query, key.transpose(2, 3)) * attention.scaling).float()
            probabilities = _masked_probabilities(scores, attention_mask)
            edited = apply_attention_intervention(probabilities, scores, valid_edges, spec)
            aggregate_weights = nn.functional.dropout(
                edited.to(query.dtype),
                p=attention.attention_dropout,
                training=attention.training,
            )
            output = torch.matmul(aggregate_weights, value).transpose(1, 2).contiguous()
            output = output.reshape(*input_shape, -1).contiguous()
            return attention.dense(output), aggregate_weights

        return forward

    def _features_from_hidden(
        self,
        hidden_states: Sequence[torch.Tensor],
        token_mask: torch.Tensor,
        position_ids: torch.Tensor,
    ) -> tuple[AttentionFeatures, ...]:
        additive_mask = _native_causal_mask(
            self.model.config, hidden_states[0], token_mask, position_ids
        )
        valid_edges = _valid_edges(token_mask)
        cos, sin = self.model.gpt_neox.rotary_emb(hidden_states[0], position_ids=position_ids)
        result: list[AttentionFeatures] = []
        for index, (layer, hidden) in enumerate(zip(self.model.gpt_neox.layers, hidden_states, strict=True)):
            attention = layer.attention
            normalized = layer.input_layernorm(hidden)
            input_shape = normalized.shape[:-1]
            shape = (*input_shape, -1, 3 * attention.head_size)
            qkv = attention.query_key_value(normalized).view(shape).transpose(1, 2)
            query, key, value = qkv.chunk(3, dim=-1)
            query, key = apply_rotary_pos_emb(query, key, cos, sin)
            scores = (torch.matmul(query, key.transpose(2, 3)) * attention.scaling).float()
            probabilities = _masked_probabilities(scores, additive_mask)
            result.append(AttentionFeatures(
                layer=index,
                scores=scores,
                probabilities=probabilities,
                query=query.transpose(1, 2).reshape(*input_shape, -1),
                key=key.transpose(1, 2).reshape(*input_shape, -1),
                value=value.transpose(1, 2).reshape(*input_shape, -1),
                valid_edges=valid_edges,
            ))
        return tuple(result)


def adapt_causal_lm(model: nn.Module, expected_shape: ModelShape | None = None) -> _BaseAdapter:
    """Construct the architecture-specific narrow adapter."""

    if isinstance(model, GPT2LMHeadModel):
        return GPT2Adapter(model, expected_shape)
    if isinstance(model, GPTNeoXForCausalLM):
        return GPTNeoXAdapter(model, expected_shape)
    raise ModelAdapterError(f"unsupported causal LM type: {type(model).__name__}")

