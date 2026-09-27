import torch
import pytest
from transformers import GPT2Config

from sinklab.initialization import (InitializationError, create_initialization,
                                    load_initialization, tensor_content_hash)


def tiny_config():
    return GPT2Config(vocab_size=67, n_positions=128, n_ctx=128, n_embd=32,
                      n_layer=2, n_head=2)


def test_shared_seed_artifact_and_rng_isolation(tmp_path):
    config = tiny_config()
    before = torch.random.get_rng_state().clone()
    weights, metadata, digest = create_initialization(config, 7, tmp_path)
    assert torch.equal(before, torch.random.get_rng_state())
    again = create_initialization(config, 7, tmp_path)
    assert again == (weights, metadata, digest)
    for condition in ("C0", "C1", "C2"):
        loaded = load_initialization(metadata, config=config, seed=7)
        assert tensor_content_hash(loaded) == digest, condition
        assert all(t.device.type == "cpu" and t.dtype == torch.float32 for t in loaded.values())
    assert create_initialization(config, 8, tmp_path)[2] != digest
    with pytest.raises(InitializationError, match="architecture"):
        load_initialization(metadata, config=GPT2Config(vocab_size=68, n_positions=128,
            n_ctx=128, n_embd=32, n_layer=2, n_head=2), seed=7)
    with pytest.raises(InitializationError, match="identity"):
        load_initialization(metadata, config=config, seed=8)


def test_tensor_corruption_detected(tmp_path):
    config = tiny_config()
    weights, metadata, _ = create_initialization(config, 1, tmp_path)
    data = bytearray(weights.read_bytes())
    data[-1] ^= 1
    weights.write_bytes(data)
    with pytest.raises(InitializationError, match="content"):
        load_initialization(metadata, config=config, seed=1)
