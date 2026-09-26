"""Offline-only fixtures for the E3/E4/E5 NNsight smoke programs.

Nothing in this module is imported by production experiment entry points.  The smoke
programs create random GPT-2-compatible repositories and a tiny local tokenizer under a
TemporaryDirectory, so NNsight exercises its real local-model loading and tracing path
without a model/dataset download.
"""

from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import (
    GPT2Config,
    GPT2LMHeadModel,
    GPTNeoConfig,
    GPTNeoForCausalLM,
    OPTConfig,
    OPTForCausalLM,
    PreTrainedTokenizerFast,
    Qwen2Config,
    Qwen2ForCausalLM,
)


def _tiny_tokenizer(path: Path, vocab_size: int) -> PreTrainedTokenizerFast:
    """Write the repository-style WordLevel tokenizer used by every random checkpoint.

    Identical vocab construction to the original inline GPT-2 helper, factored out so the
    Neo/Qwen/OPT builders share exactly the same token space (ids ``t3..`` -> ``index+3``).
    """
    path.mkdir(parents=True, exist_ok=True)
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    vocab.update({f"t{index}": index + 3 for index in range(vocab_size - 3)})
    backend = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]")
    tokenizer.save_pretrained(path)
    return tokenizer


def create_tiny_checkpoint(path: Path, seed: int, *, n_positions: int = 32,
                           n_layer: int = 4, n_head: int = 4,
                           n_embd: int = 32, vocab_size: int = 64):
    """Write one deterministic random GPT-2 checkpoint plus repository-style tokenizer."""
    tokenizer = _tiny_tokenizer(path, vocab_size)

    torch.manual_seed(seed)
    config = GPT2Config(
        vocab_size=len(tokenizer), n_positions=n_positions, n_ctx=n_positions,
        n_embd=n_embd, n_layer=n_layer, n_head=n_head,
        resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0,
        attn_implementation="eager")
    model = GPT2LMHeadModel(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    del model
    return tokenizer


def create_tiny_neo_checkpoint(path: Path, seed: int, *, n_positions: int = 32,
                               n_layer: int = 4, n_head: int = 4,
                               n_embd: int = 32, vocab_size: int = 64):
    """Write a deterministic random GPT-Neo checkpoint (alternating global/local attention).

    ``attention_types=[[["global","local"], n_layer//2]]`` reproduces the real GPT-Neo
    global/local alternation, so ``config.attention_layers`` is populated and the local
    layers exercise the windowed-attention path. GPT-Neo has no q/k/v biases, so
    intervention (b) is a structural no-op here — exactly as on the real 125M model.
    """
    tokenizer = _tiny_tokenizer(path, vocab_size)
    if n_layer % 2 != 0:
        raise ValueError("GPT-Neo attention_types alternation needs an even n_layer")

    torch.manual_seed(seed)
    config = GPTNeoConfig(
        vocab_size=len(tokenizer), max_position_embeddings=n_positions,
        hidden_size=n_embd, num_layers=n_layer, num_heads=n_head,
        attention_types=[[["global", "local"], n_layer // 2]],
        intermediate_size=4 * n_embd, window_size=n_positions,
        resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0,
        attn_implementation="eager")
    model = GPTNeoForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    del model
    return tokenizer


def create_tiny_qwen_checkpoint(path: Path, seed: int, *, n_positions: int = 32,
                                n_layer: int = 4, n_head: int = 8,
                                n_kv_head: int = 2, head_dim: int = 8,
                                vocab_size: int = 64):
    """Write a deterministic random Qwen2 checkpoint with grouped-query attention.

    Defaults ``num_attention_heads=8, num_key_value_heads=2`` so the KV-grouped
    (``kv_grouped=True``) code path and the ``num_kv_heads * head_dim`` trailing-dim guard
    are both exercised. RoPE is applied inside ``Qwen2Attention.forward``.
    """
    tokenizer = _tiny_tokenizer(path, vocab_size)
    hidden = n_head * head_dim

    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=len(tokenizer), hidden_size=hidden, num_hidden_layers=n_layer,
        num_attention_heads=n_head, num_key_value_heads=n_kv_head,
        intermediate_size=2 * hidden, max_position_embeddings=n_positions,
        max_window_layers=n_layer, use_sliding_window=False,
        attention_dropout=0.0, tie_word_embeddings=True,
        attn_implementation="eager")
    model = Qwen2ForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    del model
    return tokenizer


def create_tiny_opt_checkpoint(path: Path, seed: int, *, n_positions: int = 32,
                               n_layer: int = 4, n_head: int = 4,
                               n_embd: int = 32, vocab_size: int = 64):
    """Write a deterministic random pre-LayerNorm OPT checkpoint.

    ``do_layer_norm_before=True`` and ``word_embed_proj_dim == hidden_size`` satisfy
    ``guard_opt`` (post-LN and projected-embedding variants are rejected by the harness).
    OPT nulls ``attn_weights`` unless ``output_attentions=True``, which the engine threads.
    """
    tokenizer = _tiny_tokenizer(path, vocab_size)

    torch.manual_seed(seed)
    config = OPTConfig(
        vocab_size=len(tokenizer), max_position_embeddings=n_positions,
        hidden_size=n_embd, word_embed_proj_dim=n_embd, ffn_dim=4 * n_embd,
        num_hidden_layers=n_layer, num_attention_heads=n_head,
        do_layer_norm_before=True, dropout=0.0, attention_dropout=0.0,
        attn_implementation="eager")
    model = OPTForCausalLM(config).eval()
    model.save_pretrained(path, safe_serialization=True)
    del model
    return tokenizer


def synthetic_text(length: int, offset: int = 0) -> str:
    return " ".join(f"t{3 + ((offset + index) % 50)}" for index in range(length))


def synthetic_records(length: int = 16, examples_per_domain: int = 1):
    records = {}
    for domain_index, domain in enumerate(("sst2", "gsm8k", "humaneval")):
        rows = []
        for example_id in range(examples_per_domain):
            ids = [3 + ((domain_index * 11 + example_id * 7 + pos) % 50)
                   for pos in range(length)]
            rows.append({
                "dataset": domain,
                "example_id": example_id,
                "input_ids": ids,
                "text": synthetic_text(length, domain_index * 11 + example_id * 7),
                "source_indices": [example_id],
            })
        records[domain] = rows
    return records


def assert_nnsight_metadata(config: dict) -> None:
    assert config["engine_name"] == "nnsight"
    engine = config["engine"]
    required = {
        "name", "nnsight_version", "transformers_version", "torch_version",
        "model_name", "model_revision", "dtype", "device", "execution_location",
        "attn_implementation", "attention_probability_source", "layer_band",
        "intervention_registry_version",
    }
    missing = sorted(required.difference(engine))
    assert not missing, f"missing NNsight provenance keys: {missing}"
    assert engine["name"] == "nnsight"
    assert engine["attn_implementation"] == "eager"
