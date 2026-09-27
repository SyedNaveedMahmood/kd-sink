"""Shared, random CPU-FP32 GPT-2 initialization artifacts."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file, save_file
from transformers import GPT2Config, GPT2LMHeadModel

from .provenance import _no_duplicate_keys, canonical_json_bytes, seal_payload, verify_envelope


class InitializationError(ValueError):
    pass


def tensor_content_hash(tensors: dict[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name]
        if tensor.device.type != "cpu" or tensor.dtype != torch.float32:
            raise InitializationError("initialization tensors must be CPU FP32")
        raw = tensor.detach().contiguous().view(torch.uint8).numpy().tobytes()
        header = canonical_json_bytes({"name": name, "shape": list(tensor.shape),
                                       "dtype": "float32", "nbytes": len(raw)})
        h.update(len(header).to_bytes(8, "big") + header + raw)
    return h.hexdigest()


def create_initialization(config: GPT2Config, seed: int, directory: str | Path) -> tuple[Path, Path, str]:
    if type(seed) is not int or seed < 0:
        raise InitializationError("explicit nonnegative seed required")
    if not isinstance(config, GPT2Config):
        raise InitializationError("GPT2Config required; pretrained loading is forbidden")
    prior_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        with torch.device("cpu"), torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            model = GPT2LMHeadModel(config)
            tensors = {key: value.detach().contiguous().clone()
                       for key, value in model.state_dict().items()}
    finally:
        torch.set_default_dtype(prior_dtype)
    if any(t.dtype != torch.float32 for t in tensors.values()):
        raise InitializationError("model is not wholly FP32")
    digest = tensor_content_hash(tensors)
    config_digest = hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()
    payload = {"kind": "gpt2-random-cpu-fp32-v1", "seed": seed,
               "config_sha256": config_digest, "tensor_content_sha256": digest,
               "tensor_count": len(tensors), "tensor_shapes": {k: list(v.shape) for k, v in tensors.items()}}
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    weights = root / f"init-seed{seed}-{digest}.safetensors"
    metadata = root / f"init-seed{seed}-{digest}.json"
    if weights.exists() or metadata.exists():
        if not weights.exists() or not metadata.exists():
            raise InitializationError("incomplete existing initialization")
        load_initialization(metadata, config=config, seed=seed)
        return weights, metadata, digest
    save_file(tensors, str(weights))
    metadata.write_bytes(canonical_json_bytes(seal_payload(payload)) + b"\n")
    load_initialization(metadata, config=config, seed=seed)
    return weights, metadata, digest


def load_initialization(metadata: str | Path, *, config: GPT2Config, seed: int) -> dict[str, torch.Tensor]:
    metadata = Path(metadata)
    try:
        document = json.loads(metadata.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        payload, _ = verify_envelope(document)
        if payload["kind"] != "gpt2-random-cpu-fp32-v1" or payload["seed"] != seed:
            raise InitializationError("initialization identity mismatch")
        if payload["config_sha256"] != hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest():
            raise InitializationError("architecture config mismatch")
        digest = payload["tensor_content_sha256"]
        if metadata.name != f"init-seed{seed}-{digest}.json":
            raise InitializationError("initialization metadata filename mismatch")
        weights = metadata.with_suffix(".safetensors")
        tensors = load_file(str(weights), device="cpu")
        if (len(tensors) != payload["tensor_count"] or
                {k: list(v.shape) for k, v in tensors.items()} != payload["tensor_shapes"] or
                tensor_content_hash(tensors) != digest):
            raise InitializationError("initialization tensor content mismatch")
        return tensors
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, InitializationError):
            raise
        raise InitializationError(f"cannot verify initialization: {exc}") from exc
