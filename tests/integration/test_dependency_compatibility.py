"""Stage 00 CPU smoke for the locked model stack; no network or weights."""

import importlib.metadata

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel


def test_locked_torch_transformers_tiny_gpt2_cpu():
    assert torch.__version__ == "2.10.0+cu128"
    assert torch.version.cuda == "12.8"
    assert transformers.__version__ == "5.3.0"
    config = GPT2Config(
        vocab_size=67, n_positions=16, n_ctx=16, n_embd=32,
        n_layer=2, n_head=2, resid_pdrop=0.0, embd_pdrop=0.0,
        attn_pdrop=0.0,
    )
    model = GPT2LMHeadModel(config).cpu()
    assert model.get_input_embeddings().weight.data_ptr() == model.get_output_embeddings().weight.data_ptr()
    ids = torch.tensor([[1, 2, 3, 4], [4, 3, 2, 1]], dtype=torch.long)
    result = model(input_ids=ids, labels=ids, use_cache=False)
    assert result.logits.shape == (2, 4, 67)
    assert torch.isfinite(result.loss)
    result.loss.backward()
    assert model.transformer.wte.weight.grad is not None
    assert torch.isfinite(model.transformer.wte.weight.grad).all()


def test_upstream_parity_dependencies_import():
    # Legacy parity is a requested environment constraint. The new package
    # does not import these optional research integrations at import time.
    import datasets
    import huggingface_hub
    import nnsight
    import peft

    assert datasets.__version__ == "4.8.4"
    assert huggingface_hub.__version__ == "1.33.0"
    assert importlib.metadata.version("nnsight") == "0.7.0"
    assert importlib.metadata.version("peft") == "0.21.0"
    assert nnsight is not None and peft is not None
