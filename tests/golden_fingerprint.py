"""Shared, deterministic ``run_intervention`` fingerprint driver for WP3 parity.

Both the golden-capture step (``python -m tests.golden_fingerprint``, run *before* the
``_apply_edits`` refactor) and ``tests/test_run_intervention_parity.py`` (run *after*)
import ``build_arch_engine`` and ``reference_fingerprint`` from here, so the exact same
random model, inputs, massive coords and swap directions drive both sides. The only thing
that differs between capture and check is the ``run_intervention`` implementation itself —
which is precisely what the parity test is proving unchanged.

Offline / CPU / fp32 only; no network. Neo is the required non-fused arch for the parity
gate (`01` §8), GPT-2 the fused reference.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # tests/ for nnsight_smoke_utils

from intervention_analysis import compute_bos_attention_metric  # noqa: E402
from nnsight_engine import ARCH_SPECS, NNsightEngine, load_nnsight_model  # noqa: E402

from nnsight_smoke_utils import (  # noqa: E402
    create_tiny_checkpoint,
    create_tiny_neo_checkpoint,
    create_tiny_opt_checkpoint,
    create_tiny_qwen_checkpoint,
)

# The two archs the parity gate covers (fused GPT-2 + separate-projection Neo).
PARITY_ARCHS = ("gpt2", "neo")

FIXTURE_PATH = REPO / "tests" / "fixtures" / "run_intervention_golden.pt"

# Deterministic per-arch model seed. Same seed on capture and check -> identical weights.
_ARCH_SEED = {"gpt2": 4001, "neo": 4002, "qwen": 4003, "opt": 4004}

_BUILDERS = {
    "gpt2": create_tiny_checkpoint,
    "neo": create_tiny_neo_checkpoint,
    "qwen": create_tiny_qwen_checkpoint,
    "opt": create_tiny_opt_checkpoint,
}

# Fixed hidden width of the random gpt2/neo/opt checkpoints (n_embd default).
_HIDDEN = 32
# Fixed massive-coord set for interventions (i)/(j). Values < hidden, chosen once so the
# Wk-column ablation is deterministic and reproducible across capture and check.
MASSIVE_COORDS: Tuple[int, ...] = (4, 17)


def build_arch_engine(arch: str, checkpoint_dir: Path,
                      seed: int | None = None) -> Tuple[NNsightEngine, object]:
    """Build a deterministic random ``arch`` checkpoint and wrap it in an NNsightEngine.

    ``guard=None``: the random configs are constructed to satisfy every harness guard, and
    the parity/CE tests only need the module tree, not the harness's HF-id load path.
    """
    if seed is None:
        seed = _ARCH_SEED[arch]
    tokenizer = _BUILDERS[arch](checkpoint_dir, seed)
    spec = ARCH_SPECS[arch]
    lm = load_nnsight_model(
        spec, str(checkpoint_dir), dtype=torch.float32, tokenizer=tokenizer,
        device=torch.device("cpu"), local_files_only=True)
    return NNsightEngine(lm, spec), tokenizer


def fixed_inputs(n_sentences: int = 3, length: int = 16) -> List[dict]:
    """Deterministic input-id batches inside the tiny vocab and context window."""
    inputs = []
    for s in range(n_sentences):
        ids = [3 + ((s * 7 + pos * 3) % 50) for pos in range(length)]
        tensor = torch.tensor([ids], dtype=torch.long)
        inputs.append({"input_ids": tensor,
                       "attention_mask": torch.ones_like(tensor)})
    return inputs


def swap_directions(hidden: int) -> Dict[str, Tuple[torch.Tensor, torch.Tensor]]:
    """Deterministic unit vectors for the (d)/(e) component swap, so that branch runs."""
    gen = torch.Generator().manual_seed(20260728)
    out = {}
    for key in ("int_d", "int_e"):
        v0 = torch.randn(hidden, generator=gen, dtype=torch.float32)
        v1 = torch.randn(hidden, generator=gen, dtype=torch.float32)
        out[key] = (v0 / v0.norm(), v1 / v1.norm())
    return out


def reference_fingerprint(engine: NNsightEngine,
                          inputs_list: Sequence[dict]) -> List[Dict[str, dict]]:
    """Run every intervention on each input and record raw maps + the BOS metric.

    Returns a list (one entry per input) of ``{intervention_key: {"attn", "bos"}}`` where
    ``attn`` is the list of per-layer ``[heads, seq, seq]`` CPU tensors and ``bos`` is the
    Table-1 BOS metric over the full-depth band — the number that reaches the CSV.
    """
    hidden = engine.hidden
    swap = swap_directions(hidden)
    num_layers = engine.num_layers
    records: List[Dict[str, dict]] = []
    for inputs in inputs_list:
        per_key = engine.run_all(
            inputs, massive_coords=list(MASSIVE_COORDS), swap_dirs=swap, verbose=False)
        row: Dict[str, dict] = {}
        for key, maps in per_key.items():
            bos = compute_bos_attention_metric(
                maps, num_layers, "mid", layer_start=0, layer_end=num_layers)
            row[key] = {"attn": [m.clone() for m in maps], "bos": float(bos)}
        records.append(row)
    return records


def capture_golden(fixture_path: Path = FIXTURE_PATH) -> None:
    """Capture the pre-refactor golden fingerprints for gpt2 + neo and persist them."""
    import gc
    import tempfile

    golden: Dict[str, object] = {"archs": list(PARITY_ARCHS),
                                 "massive_coords": list(MASSIVE_COORDS),
                                 "hidden": _HIDDEN}
    with tempfile.TemporaryDirectory(
            prefix="golden_fp_", ignore_cleanup_errors=True) as temp:
        base = Path(temp)
        inputs_list = fixed_inputs()
        for arch in PARITY_ARCHS:
            engine, _ = build_arch_engine(arch, base / arch)
            golden[arch] = {
                "num_layers": engine.num_layers,
                "records": reference_fingerprint(engine, inputs_list),
            }
            del engine
            gc.collect()
    fixture_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(golden, fixture_path)
    print(f"Golden run_intervention fingerprints written to {fixture_path}")


if __name__ == "__main__":
    capture_golden()
