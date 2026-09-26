# -*- coding: utf-8 -*-
"""cross_example_patching.py — source-capture → target-inject activation patching (WP7, G4).

Every causal claim in E7 rests on this module. Nothing like it existed in the repo:
:class:`~nnsight_engine.NNsightEngine` applies edits computed from *static weights* inside a
single trace, with no path from one example's activations into another example's forward.

Design (``02_MODULE_SPEC_common.md`` §6)
---------------------------------------
Two traces. :func:`capture_sources` reads the requested objects out of each source example
and caches them fp32 on CPU. :func:`run_patched` runs the target twice — once clean, once
with the cached tensors written into the corresponding sites — and returns both sets of
measurements plus their difference.

What is reused rather than reimplemented (CLAUDE.md rule 3)
-----------------------------------------------------------
* module-tree navigation, the ``ArchSpec`` surface and the payload construction from
  ``nnsight_engine`` (``_resolve``, ``spec.k_path``/``v_path``/``o_path``,
  ``NNsightEngine._payload`` — the explicit ``attention_mask`` and RoPE ``position_ids``
  are load-bearing, see that module's header);
* the frozen ``compute_bos_attention_metric`` for every sink number;
* the frozen Qwen RoPE/GQA primitives ``build_rope_cos_sin`` / ``_rotate_half`` /
  ``_apply_rope`` from ``intervention_analysis_qwen``.

``NNsightEngine.run_arch_trace(capture_position0=...)`` still raises ``NotImplementedError``
and is *not* extended: a write path cannot be expressed through it (it captures, it does not
inject), and ``tests/test_arch_trace_ce.py`` asserts that error. This module supersedes that
placeholder. ``nnsight_engine.py`` is not modified by WP7 at all.

Deliberate scope decisions
--------------------------
1. **``Kmid`` is split into ``Kmid_prerope`` / ``Kmid_postrope``.** ``05`` §3 lists a bare
   ``Kmid`` patch object, but ``02`` §6.4 requires both RoPE variants to be reported and
   never mixed. Bare ``K0``/``Kmid`` are accepted only on architectures where RoPE is *not*
   applied inside the attention module (there the distinction does not exist); on Qwen they
   raise, naming the two variants. This adds *values*, never a renamed column.
2. **``K*_postrope`` writes an inverse rotation.** To make the target's post-RoPE key at
   position ``p`` equal the source's post-RoPE key, the value written at ``k_proj.output``
   is ``R(p)^-1 · k_post_source``. At ``p = 0`` the rotation is the identity, so pre- and
   post-RoPE coincide — :func:`rope_position0_deviation` *measures* that rather than
   assuming it, per §6.4, and it does not hold at ``Kmid``.
3. **K/V patching requires ``qkv_layout == "separate"``** (neo/opt/qwen). GPT-2's fused
   ``c_attn`` raises: E7 is Qwen-only, and a fused-slice write path would be untested risk.
   ``R*`` objects work on all four architectures.
4. **``*mid`` positions resolve from the prompt length, not prompt+candidate.** Label
   scoring appends candidate tokens to the prompt; resolving ``seq_len // 2`` on the
   concatenated sequence would move the patch site for every candidate. Resolved positions
   are recorded on every returned row.
5. **Batched patching asserts equal lengths.** Padding would move position 0, which is the
   object under study. Production runs use ``batch_size=1``; the batched path exists so
   invariant 7 exercises the same writer as production.
6. **Capture-only objects (WP9, additive).** ``04`` §3.1 asks extraction for three objects
   that are *reductions over a span* rather than single-position tensors — ``Qmean`` (mean
   query over the second-half positions), ``Rlast`` and ``Rmean``. They are listed in
   :data:`CAPTURE_ONLY_OBJECTS`, never in :data:`PATCH_OBJECTS`, and both
   :class:`PatchSpec` and the writer refuse them: a mean over a span has no well-defined
   single site to write back to, and silently accepting one would produce a plausible
   effect size for a patch that never happened. They exist so WP9 extraction runs through
   *this* trace path instead of opening a second one.
"""

from __future__ import annotations

import gc
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

try:  # dual-import idiom: scripts put common/ on sys.path directly
    from . import provenance as _prov
except ImportError:  # pragma: no cover - exercised via direct-path imports
    import provenance as _prov

from intervention_analysis import compute_bos_attention_metric
from nnsight_engine import EditPlan, _resolve

_REPO = Path(__file__).resolve().parents[1]

#: Bumped whenever the meaning of a patch object or norm condition changes (``05`` §1.1).
PATCH_REGISTRY_VERSION = "patch_v1"

#: Canonical patch objects. ``K*`` carries an explicit RoPE variant (decision 1 above).
PATCH_OBJECTS: Tuple[str, ...] = (
    "R0", "K0_prerope", "K0_postrope", "V0",
    "Rmid", "Kmid_prerope", "Kmid_postrope", "Vmid",
)

#: Objects accepted only when RoPE is not applied inside the attention module.
UNQUALIFIED_K_OBJECTS: Tuple[str, ...] = ("K0", "Kmid")

#: ``04`` §3.1 measurement objects. Reductions over a span, so they can be *captured* but
#: never *patched* (decision 6). Kept out of ``PATCH_OBJECTS`` on purpose.
CAPTURE_ONLY_OBJECTS: Tuple[str, ...] = ("Qmean", "Rlast", "Rmean")

#: object -> (surface, span kind, reduction). ``span_kind`` resolves against ``seq_len`` in
#: :func:`capture_span`; ``"last"`` is a single position and therefore has no reduction.
_CAPTURE_ONLY_SPEC: Dict[str, Tuple[str, str, str]] = {
    "Qmean": ("q", "second_half", "mean"),
    "Rlast": ("r", "last", "none"),
    "Rmean": ("r", "all", "mean"),
}

#: Everything :func:`capture_sources` accepts.
CAPTURE_OBJECTS: Tuple[str, ...] = PATCH_OBJECTS + CAPTURE_ONLY_OBJECTS

NORM_CONDITIONS: Tuple[str, ...] = ("direct", "rescaled", "random", "identity")

#: ``04`` §5.4 — an identity patch whose margin moves by more than this invalidates the run.
IDENTITY_TOLERANCE = 1e-5

#: Invariants 1/2/5 are asserted at this tolerance in fp32 (``02`` §6.2).
EXACTNESS_TOLERANCE = 1e-6


# ═══════════════════════════════════════════════════════════════════════════════
# Object taxonomy
# ═══════════════════════════════════════════════════════════════════════════════


def _object_parts(obj: str, *, rope_inside: bool) -> Tuple[str, str, str]:
    """Split a patch-object name into ``(surface, position_kind, rope_variant)``.

    ``surface`` is ``"q"``/``"k"``/``"v"``/``"r"``; ``position_kind`` is
    ``"zero"``/``"mid"`` for patch objects and the span kind for capture-only ones;
    ``rope_variant`` is ``"prerope"``/``"postrope"``/``"na"``.
    """
    if obj in _CAPTURE_ONLY_SPEC:
        surface, span_kind, _reduction = _CAPTURE_ONLY_SPEC[obj]
        return surface, span_kind, "na"

    if obj in UNQUALIFIED_K_OBJECTS:
        if rope_inside:
            raise ValueError(
                f"{obj!r} is ambiguous on an architecture that applies RoPE inside the "
                "attention module: 'the key' is a different tensor before and after the "
                "rotation. Use K0_prerope/K0_postrope (or Kmid_prerope/Kmid_postrope); "
                "02 §6.4 requires both variants to be reported and never mixed.")
        return "k", ("zero" if obj == "K0" else "mid"), "prerope"

    if obj not in PATCH_OBJECTS:
        raise ValueError(f"Unknown patch object {obj!r}; choose from {PATCH_OBJECTS} "
                         f"(or {UNQUALIFIED_K_OBJECTS} on non-RoPE architectures)")

    surface = obj[0].lower()
    rope_variant = "na"
    body = obj[1:]
    if surface == "k":
        body, _, rope_variant = body.partition("_")
    position_kind = "zero" if body == "0" else "mid"
    if surface == "k" and rope_variant not in ("prerope", "postrope"):
        raise ValueError(f"Malformed key object {obj!r}")
    return surface, position_kind, rope_variant


def resolve_position(obj: str, seq_len: int, *, rope_inside: bool,
                     override: Optional[int] = None) -> int:
    """Resolve a patch object's position for a sequence of length ``seq_len``.

    ``*0`` objects resolve to 0; ``*mid`` objects to ``seq_len // 2`` **per example**
    (invariant 6). ``override`` wins when given, which is how a caller pins a site that was
    resolved against a different (e.g. prompt-only) length.

    For a capture-only object this returns the *start* of its span — see
    :func:`capture_span` for the span itself.
    """
    if override is not None:
        position = int(override)
    elif obj in _CAPTURE_ONLY_SPEC:
        _surface, span_kind, _reduction = _CAPTURE_ONLY_SPEC[obj]
        position = {"all": 0, "second_half": seq_len // 2, "last": seq_len - 1}[span_kind]
    else:
        _surface, position_kind, _variant = _object_parts(obj, rope_inside=rope_inside)
        position = 0 if position_kind == "zero" else seq_len // 2
    if not 0 <= position < seq_len:
        raise ValueError(f"Resolved position {position} out of range for length {seq_len}")
    return position


def capture_span(obj: str, seq_len: int, *, rope_inside: bool,
                 override: Optional[int] = None) -> Tuple[int, Optional[int], str]:
    """``(start, stop, reduction)`` for a capture. ``stop is None`` means one position.

    Every patch object is a single position, so it returns ``(position, None, "none")`` and
    the capture path is byte-identical to what it was before capture-only objects existed.
    ``Qmean`` spans ``[seq_len // 2, seq_len)`` and ``Rmean`` spans ``[0, seq_len)``; both
    are reduced **inside** the trace so only the reduced vector ever leaves it (``04`` §3.2
    forbids persisting full hidden states).
    """
    start = resolve_position(obj, seq_len, rope_inside=rope_inside, override=override)
    if obj not in _CAPTURE_ONLY_SPEC:
        return start, None, "none"
    _surface, span_kind, reduction = _CAPTURE_ONLY_SPEC[obj]
    if span_kind == "last" or override is not None:
        # An override pins a single position; a span cannot be meaningfully re-anchored.
        return start, None, "none"
    return start, seq_len, reduction


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class PatchSite:
    """One (object, layer, position) location. ``position=None`` resolves per example.

    ``kv_head`` restricts a K/V write to a single key/value head — needed to demonstrate
    the pre-``_repeat_kv`` property (invariant 4) and unused in production, where the whole
    position vector is patched.
    """

    object: str
    layer: int
    position: Optional[int] = None
    kv_head: Optional[int] = None

    def key(self) -> str:
        """Cache key, matching ``02`` §6.1's ``"K0@L5"`` form."""
        suffix = "" if self.kv_head is None else f"#h{self.kv_head}"
        return f"{self.object}@L{self.layer}{suffix}"


@dataclass(frozen=True)
class PatchSpec:
    """One patch to apply: where, from whom, and how the norm is treated."""

    site: PatchSite
    norm_condition: str
    source_item_id: Optional[str] = None
    seed: int = 0

    def __post_init__(self) -> None:
        if self.norm_condition not in NORM_CONDITIONS:
            raise ValueError(f"Unknown norm condition {self.norm_condition!r}; "
                             f"choose from {NORM_CONDITIONS}")
        # Decision 6: a reduction over a span has no single site to write back to. Refusing
        # at construction is the earliest possible point; the writer refuses again.
        if self.site.object in CAPTURE_ONLY_OBJECTS:
            raise ValueError(
                f"{self.site.object!r} is a capture-only object (04 §3.1): it is a "
                "reduction over a span of positions, so there is no single site to patch. "
                f"Patchable objects are {PATCH_OBJECTS}.")
        # Only the two conditions whose *value* comes from the source need one. 'identity'
        # restores the target's own vector and 'random' draws a Gaussian matched to the
        # target's norm — neither reads the source tensor. E7 still records a
        # source_item_id on 'random' rows when one is being crossed with a source
        # condition; it is metadata there, not an input, so it stays optional.
        if self.norm_condition in ("direct", "rescaled") and self.source_item_id is None:
            raise ValueError(
                f"norm_condition={self.norm_condition!r} derives its value from the source "
                "and so needs a source_item_id.")


@dataclass
class SourceCache:
    """Captured activations for one source example, fp32 on CPU.

    ``resolved_positions`` is additive to ``02`` §6.1's sketch: ``*mid`` sites resolve
    per example, and the position they resolved to has to travel with the tensor or a
    downstream row cannot record it (invariant 6).
    """

    model_fingerprint: str
    item_id: str
    tensors: Dict[str, torch.Tensor]
    resolved_positions: Dict[str, int] = field(default_factory=dict)
    seq_len: int = 0
    provenance: Dict[str, Any] = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════════════════════
# Model identity (invariant 8)
# ═══════════════════════════════════════════════════════════════════════════════


def model_fingerprint(handle) -> str:
    """Identity string for a :class:`~fingerprint_runner.ModelHandle`.

    ``02`` §6.1 specifies "arch + model_name + revision + dtype + weight sha". No such
    helper existed in the repo, so this defines one: canonical JSON of the declared identity
    plus a sampled weight digest (``provenance.sha256_state_dict(sample=8)``).

    The weight component is a *fingerprint*, not a full hash — it samples each parameter's
    ends rather than hashing hundreds of megabytes on every patch call. It is used to make
    cross-model patching **fail loudly** (invariant 8), which only requires that different
    models differ, not that identical models are proven identical byte-for-byte.
    """
    model = handle.model
    geom = geometry(handle)
    payload = {
        "arch": handle.arch,
        "model_name": handle.model_name,
        "model_revision": handle.model_revision,
        "checkpoint_step": handle.checkpoint_step,
        "checkpoint_sha256": handle.checkpoint_sha256,
        "dtype": handle.dtype,
        "num_layers": geom["num_layers"],
        "num_heads": geom["num_heads"],
        "num_kv_heads": geom["num_kv_heads"],
        "head_dim": geom["head_dim"],
        "hidden": geom["hidden"],
        "weights": _prov.sha256_state_dict(model.state_dict(), sample=8),
        "patch_registry_version": PATCH_REGISTRY_VERSION,
    }
    return _prov.sha256_json(payload)


def geometry(handle) -> Dict[str, Any]:
    """Head geometry read from the config, with the repo's standard fallbacks."""
    cfg = handle.model.config
    engine = handle.nn_engine
    num_heads = int(getattr(cfg, "num_attention_heads", None)
                    or getattr(cfg, "n_head"))
    hidden = int(getattr(cfg, "hidden_size", None) or getattr(cfg, "n_embd"))
    head_dim = int(getattr(cfg, "head_dim", 0) or (hidden // num_heads))
    num_kv_heads = int(getattr(cfg, "num_key_value_heads", 0) or num_heads)
    return {
        "num_layers": engine.num_layers,
        "num_heads": num_heads,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "hidden": hidden,
        "rope_theta": float(getattr(cfg, "rope_theta", 10000.0)),
    }


def _model_dtype(handle) -> torch.dtype:
    """The model's real parameter dtype. ``ModelHandle.dtype`` is only a declared string."""
    return next(handle.model.parameters()).dtype


def _model_device(handle) -> torch.device:
    return next(handle.model.parameters()).device


# ═══════════════════════════════════════════════════════════════════════════════
# Qwen RoPE primitives — imported from the frozen harness, never reimplemented
# ═══════════════════════════════════════════════════════════════════════════════


def _qwen_rope_helpers():
    """Import ``build_rope_cos_sin`` / ``_rotate_half`` from the frozen Qwen harness.

    Lazy ``sys.path`` insertion, matching ``fingerprint_runner._swap_directions``.
    """
    qwen_dir = _REPO / "cross_scale_and_architecture" / "qwen"
    if str(qwen_dir) not in sys.path:
        sys.path.insert(0, str(qwen_dir))
    from intervention_analysis_qwen import _rotate_half, build_rope_cos_sin
    return build_rope_cos_sin, _rotate_half


def _rope_cos_sin_at(position: int, head_dim: int, rope_theta: float,
                     device) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(cos, sin)`` of shape ``[head_dim]`` for a single absolute position."""
    build_rope_cos_sin, _ = _qwen_rope_helpers()
    pos = torch.tensor([float(position)], device=device)
    cos, sin = build_rope_cos_sin(pos, head_dim, rope_theta, device)
    return cos[0], sin[0]


def apply_rope_forward(k: torch.Tensor, position: int, *, head_dim: int,
                       rope_theta: float) -> torch.Tensor:
    """Rotate ``k`` ``[n_kv, head_dim]`` by RoPE at one absolute ``position``."""
    _, rotate_half = _qwen_rope_helpers()
    cos, sin = _rope_cos_sin_at(position, head_dim, rope_theta, k.device)
    cos = cos.to(k.dtype)
    sin = sin.to(k.dtype)
    return k * cos + rotate_half(k) * sin


def apply_rope_inverse(k_post: torch.Tensor, position: int, *, head_dim: int,
                       rope_theta: float) -> torch.Tensor:
    """Invert :func:`apply_rope_forward`.

    RoPE at a position is an orthogonal rotation in each ``(i, i + d/2)`` plane, so its
    inverse is the rotation by the negated angle: ``k = k_post * cos - rotate_half(k_post)
    * sin``. Writing this value at ``k_proj.output`` makes the model's own in-attention RoPE
    reproduce ``k_post`` exactly at that position — which is what "patch the post-RoPE key"
    means when there is no envoy on the post-RoPE tensor (``02`` §6.4).
    """
    _, rotate_half = _qwen_rope_helpers()
    cos, sin = _rope_cos_sin_at(position, head_dim, rope_theta, k_post.device)
    cos = cos.to(k_post.dtype)
    sin = sin.to(k_post.dtype)
    return k_post * cos - rotate_half(k_post) * sin


def rope_position0_deviation(handle, k_pre: torch.Tensor) -> float:
    """Max abs difference between a pre-RoPE key and its rotation at position 0.

    ``02`` §6.4 says to *verify* the position-0 identity numerically rather than assume it
    (a non-zero position offset would break it). Returned as a number so tests and
    production runs can both assert on it.
    """
    geom = geometry(handle)
    rotated = apply_rope_forward(k_pre, 0, head_dim=geom["head_dim"],
                                 rope_theta=geom["rope_theta"])
    return float((rotated - k_pre).abs().max().item())


# ═══════════════════════════════════════════════════════════════════════════════
# Label scoring (§6.3) — pure, model-free, shared by baseline and patched
# ═══════════════════════════════════════════════════════════════════════════════


def sequence_logprob(logits: torch.Tensor, prompt_len: int,
                     candidate_ids: Sequence[int]) -> Tuple[float, float]:
    """Teacher-forced log-probability of ``candidate_ids`` appended to a prompt.

    ``logits`` is ``[seq, vocab]`` for the concatenated ``prompt + candidate`` sequence.
    Token ``j`` of the candidate is predicted from position ``prompt_len - 1 + j``.

    Returns ``(total_logprob, length_normalised_logprob)``. Design §10.4 scores candidates
    on the length-normalised scale, and the unnormalised value is kept so both margins can
    be reported (``05`` §3 has columns for each).

    Implemented once, here, and used for the baseline and the patched forward alike, so a
    scoring bug cannot affect one and not the other.
    """
    if prompt_len < 1:
        raise ValueError("prompt_len must be >= 1")
    candidate_ids = [int(t) for t in candidate_ids]
    if not candidate_ids:
        raise ValueError("candidate_ids must be non-empty")
    if logits.ndim != 2:
        raise ValueError(f"logits must be [seq, vocab]; got shape {tuple(logits.shape)}")
    needed = prompt_len + len(candidate_ids)
    if logits.shape[0] < needed:
        raise ValueError(
            f"logits has {logits.shape[0]} positions but scoring needs {needed} "
            f"(prompt_len={prompt_len} + {len(candidate_ids)} candidate tokens)")

    logprobs = torch.log_softmax(logits.float(), dim=-1)
    total = 0.0
    for j, token in enumerate(candidate_ids):
        total += float(logprobs[prompt_len - 1 + j, token].item())
    return total, total / len(candidate_ids)


def correct_label_margin(scores: Sequence[float], gold_index: int) -> float:
    """``logP(gold) - max_{non-gold} logP(label)`` on whichever scale ``scores`` is on.

    Positive means the gold label is preferred. With a single candidate the margin is
    undefined and ``nan`` is returned rather than a fabricated value.
    """
    if not 0 <= gold_index < len(scores):
        raise IndexError(f"gold_index {gold_index} out of range for {len(scores)} scores")
    others = [s for i, s in enumerate(scores) if i != gold_index]
    if not others:
        return float("nan")
    return float(scores[gold_index]) - float(max(others))


#: JSD between two distributions is bounded by ``log 2`` nats (disjoint point masses).
JSD_MAX = float(np.log(2.0))


def jensen_shannon_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """JSD in nats between two next-token distributions given as logits vectors.

    Clamped to ``[0, log 2]``, which are the quantity's mathematical bounds. The clamp
    only absorbs float error: computed in fp32 over a vocabulary, the difference of two
    nearly equal sums lands around ``-2e-8`` for a near-no-op patch, and a negative
    divergence written into a results CSV would be indistinguishable from a real (if
    impossible) measurement. Anything large enough to be a genuine effect is untouched.
    """
    p = torch.softmax(p_logits.float(), dim=-1)
    q = torch.softmax(q_logits.float(), dim=-1)
    m = 0.5 * (p + q)
    eps = 1e-12

    def _kl(a, b):
        return float((a * ((a + eps).log() - (b + eps).log())).sum().item())

    value = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
    if not np.isfinite(value):
        return float("nan")
    return float(min(max(value, 0.0), JSD_MAX))


# ═══════════════════════════════════════════════════════════════════════════════
# Trace plumbing
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class _Write:
    """One resolved write, in model dtype/device, ready for the trace body."""

    layer: int
    surface: str            # "k" | "v" | "r"
    batch_index: int
    position: int
    value: torch.Tensor     # [n_kv*head_dim] for k/v, [hidden] for r; or [head_dim]
    col_start: int          # slice start into the projection output (0 for r)
    col_stop: int


@dataclass(frozen=True)
class _Capture:
    """One read out of the trace.

    ``stop is None`` reads the single row at ``position`` — the only shape that existed
    before WP9. ``stop`` set reads ``[position, stop)`` and reduces it inside the trace.
    """

    key: str
    layer: int
    surface: str
    batch_index: int
    position: int
    stop: Optional[int] = None
    reduction: str = "none"


def _block_output_is_tuple(engine) -> bool:
    """Whether decoder blocks return ``(hidden, ...)`` or a bare tensor.

    ``transformers`` changed this per-architecture across versions (GPT-2 blocks return a
    tuple; several v5 decoder layers return the hidden state directly). Probing once with a
    real forward hook is cheaper and far more robust than version sniffing, and the answer
    is cached on the engine.
    """
    cached = getattr(engine, "_cep_block_output_is_tuple", None)
    if cached is not None:
        return bool(cached)

    seen: Dict[str, bool] = {}

    def _hook(_module, _inputs, output):
        seen["tuple"] = isinstance(output, (tuple, list))

    block = engine.blocks[0]
    handle = block.register_forward_hook(_hook)
    try:
        device = next(engine.hf.parameters()).device
        ids = torch.zeros((1, 2), dtype=torch.long, device=device)
        with torch.no_grad():
            engine.hf(input_ids=ids, attention_mask=torch.ones_like(ids))
    finally:
        handle.remove()
    if "tuple" not in seen:  # pragma: no cover - the hook always fires on a real forward
        raise RuntimeError("Could not determine the decoder block output type")
    engine._cep_block_output_is_tuple = seen["tuple"]
    return seen["tuple"]


def _kv_columns(handle, surface: str, kv_head: Optional[int]) -> Tuple[int, int]:
    """Column slice of ``k_proj``/``v_proj`` output covered by ``kv_head`` (all if None)."""
    geom = geometry(handle)
    width = geom["num_kv_heads"] * geom["head_dim"]
    if kv_head is None:
        return 0, width
    if not 0 <= kv_head < geom["num_kv_heads"]:
        raise ValueError(f"kv_head {kv_head} out of range for "
                         f"{geom['num_kv_heads']} key/value heads")
    start = kv_head * geom["head_dim"]
    return start, start + geom["head_dim"]


def _surface_path(handle, surface: str) -> Optional[str]:
    spec = handle.nn_engine.spec
    if surface == "q":
        return spec.q_path
    if surface == "k":
        return spec.k_path
    if surface == "v":
        return spec.v_path
    return None


def _require_q_surface(handle) -> str:
    """Guard the ``Qmean`` capture surface (WP9). Same shape as :func:`_require_kv_surface`."""
    spec = handle.nn_engine.spec
    if spec.qkv_layout != "separate" or spec.q_path is None:
        raise NotImplementedError(
            f"Qmean needs a separate q projection; arch {spec.name!r} declares none "
            "(a fused c_attn would need an untested slice offset). E7 runs on Qwen.")
    return spec.q_path


def _require_kv_surface(handle, surface: str) -> str:
    """Guard decision 3: K/V patching needs separate projections."""
    spec = handle.nn_engine.spec
    if spec.qkv_layout != "separate":
        raise NotImplementedError(
            f"K/V patching needs separate q/k/v projections; arch {spec.name!r} fuses them "
            "into one Conv1D (c_attn). E7 runs on Qwen only, so the fused write path is "
            "deliberately not implemented rather than shipped untested. R* objects work on "
            "every architecture.")
    path = _surface_path(handle, surface)
    if path is None:
        raise NotImplementedError(
            f"ArchSpec for {spec.name!r} declares no {surface}_path; cannot patch "
            f"{surface.upper()} on this architecture.")
    return path


def _traced_forward(engine, payload: Dict[str, Any], *,
                    writes: Sequence[_Write] = (),
                    captures: Sequence[_Capture] = (),
                    attention_layers: Sequence[int] = (),
                    attn_output_layers: Sequence[int] = (),
                    block_output_layers: Sequence[int] = (),
                    capture_logits: bool = False,
                    block_is_tuple: bool = True) -> Dict[str, Any]:
    """One NNsight trace that may capture and/or write at ``q/k/v_proj``/block output.

    The per-layer envoy access order is ``q_proj -> k_proj -> v_proj -> attn ->
    block.output``, which is the model's own execution order; reordering raises
    ``MissedProviderError`` (see ``nnsight_engine``'s header). Within a surface, captures
    run **before** writes so a caller that does both at one site records the original value.
    ``q_proj`` is capture-only (``Qmean``, WP9) and is skipped entirely when no capture asks
    for it, so nothing that predates WP9 touches a new envoy.

    Deliberately not wrapped in ``torch.no_grad()``: NNsight defers the body to ``__exit__``
    and manages grad itself; an outer ``no_grad`` makes the slice views and their in-place
    mutation disagree about grad mode. Saved tensors are detached on the way out instead —
    the same reasoning as ``NNsightEngine.run_all``.
    """
    spec = engine.spec
    num_layers = engine.num_layers

    writes_by_layer: Dict[int, List[_Write]] = {}
    for write in writes:
        writes_by_layer.setdefault(write.layer, []).append(write)
    captures_by_layer: Dict[int, List[_Capture]] = {}
    for capture in captures:
        captures_by_layer.setdefault(capture.layer, []).append(capture)

    attention_set = set(int(i) for i in attention_layers)
    attn_out_set = set(int(i) for i in attn_output_layers)
    block_out_set = set(int(i) for i in block_output_layers)

    saved_captures: Dict[str, Any] = {}
    saved_attention: Dict[int, Any] = {}
    saved_attn_output: Dict[int, Any] = {}
    saved_block_output: Dict[int, Any] = {}
    saved_logits = None

    q_path = spec.q_path
    k_path = spec.k_path
    v_path = spec.v_path

    def _read(tensor, capture: _Capture):
        """Index (and reduce) one capture inside the trace, so spans never leave it."""
        if capture.stop is None:
            return tensor[capture.batch_index, capture.position, :]
        span = tensor[capture.batch_index, capture.position:capture.stop, :]
        if capture.reduction == "mean":
            return span.mean(dim=0)
        raise ValueError(f"Unknown capture reduction {capture.reduction!r}")

    with engine.lm.trace(payload, remote=engine.remote):
        for i in range(num_layers):
            block = engine._block(i)
            attn = engine._attn(block)
            layer_captures = captures_by_layer.get(i, [])
            layer_writes = writes_by_layer.get(i, [])

            for surface, path in (("q", q_path), ("k", k_path), ("v", v_path)):
                surface_captures = [c for c in layer_captures if c.surface == surface]
                surface_writes = [w for w in layer_writes if w.surface == surface]
                if not surface_captures and not surface_writes:
                    continue
                proj = _resolve(attn, path)
                for capture in surface_captures:
                    saved_captures[capture.key] = _read(proj.output, capture).save()
                for write in surface_writes:
                    proj.output[write.batch_index, write.position,
                                write.col_start:write.col_stop] = write.value

            if i in attention_set:
                saved_attention[i] = attn.output[1].save()
            if i in attn_out_set:
                saved_attn_output[i] = attn.output[0].save()

            block_captures = [c for c in layer_captures if c.surface == "r"]
            block_writes = [w for w in layer_writes if w.surface == "r"]
            if block_captures or block_writes or i in block_out_set:
                hidden = block.output[0] if block_is_tuple else block.output
                for capture in block_captures:
                    saved_captures[capture.key] = _read(hidden, capture).save()
                for write in block_writes:
                    hidden[write.batch_index, write.position, :] = write.value
                if i in block_out_set:
                    saved_block_output[i] = hidden.save()

        if capture_logits:
            # lm_head is touched only after the final block: accessing it earlier raises
            # MissedProviderError (NNsight execution order).
            saved_logits = engine.lm.lm_head.output.save()

    def _cpu(value):
        if value is None:
            return None
        return value.detach().float().cpu()

    attention = None
    if attention_set:
        attention = {}
        for i in sorted(attention_set):
            probs = saved_attention.get(i)
            if probs is None:
                raise RuntimeError(
                    "Attention probabilities came back None. The model is not running "
                    "eager attention, or this architecture needs output_attentions=True.")
            attention[i] = _cpu(probs)

    return {
        "captures": {k: _cpu(v) for k, v in saved_captures.items()},
        "attention": attention,
        "attn_output": {i: _cpu(v) for i, v in saved_attn_output.items()},
        "block_output": {i: _cpu(v) for i, v in saved_block_output.items()},
        "logits": _cpu(saved_logits),
    }


def _payload(engine, input_ids: torch.Tensor) -> Dict[str, Any]:
    """Build the trace payload through the engine's own ``_payload``.

    Reused rather than rebuilt because the explicit ``attention_mask`` and the RoPE
    ``position_ids`` are correctness-critical, not boilerplate.
    """
    return engine._payload(EditPlan("int_a"), {"input_ids": input_ids})


# ═══════════════════════════════════════════════════════════════════════════════
# Capture
# ═══════════════════════════════════════════════════════════════════════════════


def _sites_to_captures(handle, sites: Sequence[PatchSite], seq_len: int,
                       batch_index: int = 0,
                       position_overrides: Optional[Dict[str, int]] = None
                       ) -> Tuple[List[_Capture], Dict[str, int]]:
    """Turn sites into trace captures, resolving ``*mid`` positions against ``seq_len``."""
    spec = handle.nn_engine.spec
    rope_inside = bool(spec.rope_applied_inside_attn)
    overrides = position_overrides or {}
    captures: List[_Capture] = []
    positions: Dict[str, int] = {}
    for site in sites:
        surface, _kind, _variant = _object_parts(site.object, rope_inside=rope_inside)
        if surface in ("k", "v"):
            _require_kv_surface(handle, surface)
        elif surface == "q":
            _require_q_surface(handle)
        if not 0 <= site.layer < handle.nn_engine.num_layers:
            raise ValueError(f"Layer {site.layer} out of range for "
                             f"{handle.nn_engine.num_layers} layers")
        key = site.key()
        override = site.position if site.position is not None else overrides.get(key)
        position, stop, reduction = capture_span(
            site.object, seq_len, rope_inside=rope_inside, override=override)
        captures.append(_Capture(key=key, layer=site.layer, surface=surface,
                                 batch_index=batch_index, position=position,
                                 stop=stop, reduction=reduction))
        positions[key] = position
    return captures, positions


def _shape_capture(handle, obj: str, flat: torch.Tensor) -> torch.Tensor:
    """Reshape a captured row to the declared object shape and assert it (invariant 9)."""
    geom = geometry(handle)
    spec = handle.nn_engine.spec
    surface, _kind, _variant = _object_parts(obj, rope_inside=spec.rope_applied_inside_attn)
    if surface == "r":
        expected = (geom["hidden"],)
        if tuple(flat.shape) != expected:
            raise ValueError(f"{obj}: expected residual shape {expected}, "
                             f"got {tuple(flat.shape)}")
        return flat
    if surface == "q":
        # q_proj is one row per *attention* head, not per KV head — Qmean is a query object
        # and must not be reshaped against num_kv_heads.
        expected_flat = geom["num_heads"] * geom["head_dim"]
        if tuple(flat.shape) != (expected_flat,):
            raise ValueError(
                f"{obj}: expected a [{expected_flat}] query row "
                f"(num_heads {geom['num_heads']} * head_dim {geom['head_dim']}), "
                f"got {tuple(flat.shape)}")
        return flat.view(geom["num_heads"], geom["head_dim"])
    expected_flat = geom["num_kv_heads"] * geom["head_dim"]
    if tuple(flat.shape) != (expected_flat,):
        raise ValueError(
            f"{obj}: expected a [{expected_flat}] projection row "
            f"(num_kv_heads {geom['num_kv_heads']} * head_dim {geom['head_dim']}), "
            f"got {tuple(flat.shape)}")
    return flat.view(geom["num_kv_heads"], geom["head_dim"])


def capture_sources(handle, corpus, sites: Sequence[PatchSite], *,
                    batch_size: int = 1, to_cpu: bool = True,
                    dtype: torch.dtype = torch.float32,
                    on_item: Optional[Any] = None,
                    keep: bool = True) -> Dict[str, SourceCache]:
    """Capture every requested site for every item in ``corpus``.

    Returns ``{item_id: SourceCache}``. Tensors are ``[n_kv_heads, head_dim]`` for K/V,
    ``[num_heads, head_dim]`` for ``Qmean`` and ``[hidden]`` for R, in ``dtype`` (fp32 by
    default) on CPU.

    ``on_item`` / ``keep`` (WP9, additive; defaults reproduce the original behaviour
    exactly). ``on_item(item_id, cache)`` is called as each example finishes, and
    ``keep=False`` drops the cache afterwards instead of accumulating it. WP9 extraction is
    300 sentences × 8 languages × every layer × ten objects — it streams each example
    straight into a ``.npy`` memmap rather than holding the set in RAM (``04`` §3.2). The
    streaming form still pays ``model_fingerprint``'s weight digest once per *call*, which
    is why it is a callback here rather than a one-item-corpus loop in the caller.

    ``batch_size`` is accepted for API compatibility with ``02`` §6.1 and is asserted to be
    1: capture is cheap relative to patching, and batching would require padding, which
    moves position 0 — the object under study. Raising is preferable to silently capturing
    from a padded sequence.

    ``K*_postrope`` sites are captured pre-RoPE at ``k_proj.output`` (the only available
    envoy) and rotated **outside** the trace with the frozen Qwen RoPE helpers, so the
    stored tensor really is the post-RoPE key at the source's own position.
    """
    if batch_size != 1:
        raise NotImplementedError(
            "capture_sources runs at batch_size=1. Batching would require padding, and "
            "padding moves position 0 — the object this experiment measures.")
    engine = handle.nn_engine
    if engine is None:
        raise ValueError("cross-example patching requires an nnsight ModelHandle")

    geom = geometry(handle)
    spec = engine.spec
    rope_inside = bool(spec.rope_applied_inside_attn)
    fingerprint = model_fingerprint(handle)
    device = _model_device(handle)
    block_is_tuple = _block_output_is_tuple(engine)

    caches: Dict[str, SourceCache] = {}
    for item in corpus.items:
        input_ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
        seq_len = int(input_ids.shape[-1])
        captures, positions = _sites_to_captures(handle, sites, seq_len)
        result = _traced_forward(
            engine, _payload(engine, input_ids),
            captures=captures, block_is_tuple=block_is_tuple)

        tensors: Dict[str, torch.Tensor] = {}
        for site in sites:
            key = site.key()
            flat = result["captures"][key]
            shaped = _shape_capture(handle, site.object, flat).to(dtype)
            _surface, _kind, variant = _object_parts(site.object, rope_inside=rope_inside)
            if variant == "postrope":
                shaped = apply_rope_forward(shaped, positions[key],
                                            head_dim=geom["head_dim"],
                                            rope_theta=geom["rope_theta"])
            tensors[key] = shaped.cpu() if to_cpu else shaped

        cache = SourceCache(
            model_fingerprint=fingerprint,
            item_id=item.item_id,
            tensors=tensors,
            resolved_positions=dict(positions),
            seq_len=seq_len,
            provenance=_prov.provenance_block(extra={
                "corpus_id": corpus.corpus_id,
                "manifest_sha256": corpus.manifest_sha256,
                "arch": handle.arch,
                "model_name": handle.model_name,
                "dtype": str(dtype).replace("torch.", ""),
                "device": str(device),
                "sites": [site.key() for site in sites],
                "patch_registry_version": PATCH_REGISTRY_VERSION,
            }),
        )
        if on_item is not None:
            on_item(item.item_id, cache)
        if keep:
            caches[item.item_id] = cache
        else:
            del cache
        del result
    return caches


# ═══════════════════════════════════════════════════════════════════════════════
# Patch value construction (invariant 5)
# ═══════════════════════════════════════════════════════════════════════════════


def _norm(tensor: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(tensor.reshape(-1).float()).item())


def build_patch_value(spec: PatchSpec, target_value: torch.Tensor,
                      source_value: Optional[torch.Tensor]
                      ) -> Tuple[torch.Tensor, Dict[str, float], str]:
    """The tensor to write, per ``norm_condition``, its norms, and any warning.

    * ``identity`` — the target's own captured value (an exact no-op; invariant 1);
    * ``direct``   — the source value unchanged;
    * ``rescaled`` — the source rescaled to ``||target||`` (invariant 5);
    * ``random``   — a Gaussian vector rescaled to ``||target||``, drawn from a generator
      seeded with ``spec.seed`` so the control is reproducible.

    ``identity`` and ``random`` never read ``source_value``; only ``direct`` and
    ``rescaled`` do.

    Norms are Euclidean over the flattened tensor. A zero-norm source cannot be rescaled;
    the source is returned unchanged and a warning is emitted rather than dividing by zero.
    """
    target_norm = _norm(target_value)
    warning = ""
    if spec.norm_condition == "identity":
        value = target_value.clone()
    elif source_value is None and spec.norm_condition in ("direct", "rescaled"):
        raise ValueError(f"norm_condition={spec.norm_condition!r} requires a source value")
    elif spec.norm_condition == "direct":
        value = source_value.clone()
    elif spec.norm_condition == "rescaled":
        source_norm = _norm(source_value)
        if source_norm <= 0.0:
            value = source_value.clone()
            warning = "source norm is zero; rescaling skipped"
        else:
            value = source_value * (target_norm / source_norm)
    else:  # "random"
        generator = torch.Generator(device="cpu").manual_seed(int(spec.seed))
        noise = torch.randn(target_value.shape, generator=generator, dtype=torch.float32)
        noise_norm = _norm(noise)
        value = (noise * (target_norm / noise_norm)) if noise_norm > 0 else noise
        value = value.to(target_value.dtype)

    stats = {
        "target_norm": target_norm,
        "source_norm": _norm(source_value) if source_value is not None else float("nan"),
        "patched_norm": _norm(value),
    }
    return value, stats, warning


# ═══════════════════════════════════════════════════════════════════════════════
# Measurement helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _attention_list(attention: Dict[int, torch.Tensor], num_layers: int
                    ) -> List[torch.Tensor]:
    """``{layer: [1, H, S, S]}`` -> the ``[H, S, S]`` per-layer list the frozen metric takes."""
    return [attention[i][0] for i in range(num_layers)]


def _sink_measurements(attention: Dict[int, torch.Tensor], num_layers: int,
                       band: Tuple[int, int],
                       carrier_heads: Optional[Sequence[Tuple[int, int]]]
                       ) -> Dict[str, Any]:
    """Sink strength, attention to positions 1–4, and per-carrier-head sink.

    Every scalar here comes from the frozen ``compute_bos_attention_metric`` (CLAUDE.md
    rule 3); only the carrier-head slice is taken directly, because the frozen reducer
    averages over heads by construction.
    """
    maps = _attention_list(attention, num_layers)
    seq_len = int(maps[0].shape[-1])
    ls, le = band
    sink = compute_bos_attention_metric(maps, num_layers, "mid", target_pos=0,
                                        layer_start=ls, layer_end=le)
    near = []
    for pos in (1, 2, 3, 4):
        if pos < seq_len:
            near.append(compute_bos_attention_metric(maps, num_layers, "mid",
                                                     target_pos=pos,
                                                     layer_start=ls, layer_end=le))
    attn_pos1_4 = float(np.mean(near)) if near else float("nan")

    carrier = float("nan")
    if carrier_heads:
        second_half = seq_len // 2
        values = []
        for layer, head in carrier_heads:
            values.append(float(maps[layer][head, second_half:, 0].mean().item()))
        carrier = float(np.mean(values)) if values else float("nan")

    return {"sink": float(sink), "attn_pos1_4": attn_pos1_4, "carrier_sink": carrier}


# ═══════════════════════════════════════════════════════════════════════════════
# The entry point
# ═══════════════════════════════════════════════════════════════════════════════


def run_patched(handle, target_item, patch_specs: Sequence[PatchSpec],
                source_caches: Dict[str, SourceCache], *,
                score_candidates: Optional[Sequence[Sequence[int]]] = None,
                gold_index: Optional[int] = None,
                capture_sink: bool = True,
                band: Optional[Tuple[int, int]] = None,
                carrier_heads: Optional[Sequence[Tuple[int, int]]] = None,
                capture_block_outputs: Sequence[int] = (),
                capture_attn_outputs: Sequence[int] = ()) -> Dict[str, Any]:
    """Run ``target_item`` clean and once per patch spec, and return both measurements.

    Returns ``{"baseline": {...}, "patches": [row, ...], ...}`` where each row carries the
    ``05`` §3 fields this module can know: log-probs, margins (normalised and raw),
    prediction, position-0 attention, attention to positions 1–4, per-carrier-head sink,
    output JSD, the resolved patch positions, the three norms, ``status`` and ``warning``.

    Failures — non-finite logits, OOM, a shape mismatch — set ``status`` and ``warning`` on
    the row and never drop it (``02`` §6.5). A model-fingerprint mismatch is the one thing
    that raises: cross-model patching is not a supported operation and must fail loudly
    rather than produce plausible garbage (invariant 8).

    ``capture_block_outputs`` / ``capture_attn_outputs`` return per-layer tensors for the
    locality and pre-``_repeat_kv`` checks; they are unused in production (the returned
    tensors are large) and default to empty.
    """
    engine = handle.nn_engine
    if engine is None:
        raise ValueError("cross-example patching requires an nnsight ModelHandle")
    num_layers = engine.num_layers
    spec = engine.spec
    rope_inside = bool(spec.rope_applied_inside_attn)
    geom = geometry(handle)
    device = _model_device(handle)
    model_dtype = _model_dtype(handle)
    block_is_tuple = _block_output_is_tuple(engine)

    fingerprint = model_fingerprint(handle)
    for cache in source_caches.values():
        if cache.model_fingerprint != fingerprint:
            raise ValueError(
                "Source cache was captured from a different model "
                f"(cache={cache.model_fingerprint[:12]}…, handle={fingerprint[:12]}…). "
                "Cross-model patching is not supported and is refused rather than run.")

    prompt_ids = [int(t) for t in target_item.input_ids]
    prompt_len = len(prompt_ids)
    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)

    sites = [spec_i.site for spec_i in patch_specs]
    unique_sites: List[PatchSite] = []
    seen_keys = set()
    for site in sites:
        if site.key() not in seen_keys:
            seen_keys.add(site.key())
            unique_sites.append(site)

    # Positions resolve against the PROMPT length, never prompt+candidate (decision 4).
    captures, positions = _sites_to_captures(handle, unique_sites, prompt_len)

    band = tuple(band) if band is not None else (0, num_layers)
    attention_layers = list(range(num_layers)) if capture_sink else []

    clean = _traced_forward(
        engine, _payload(engine, input_ids),
        captures=captures,
        attention_layers=attention_layers,
        attn_output_layers=list(capture_attn_outputs),
        block_output_layers=list(capture_block_outputs),
        capture_logits=True,
        block_is_tuple=block_is_tuple)

    target_values: Dict[str, torch.Tensor] = {}
    for site in unique_sites:
        key = site.key()
        shaped = _shape_capture(handle, site.object, clean["captures"][key])
        _surface, _kind, variant = _object_parts(site.object, rope_inside=rope_inside)
        if variant == "postrope":
            shaped = apply_rope_forward(shaped, positions[key],
                                        head_dim=geom["head_dim"],
                                        rope_theta=geom["rope_theta"])
        target_values[key] = shaped.float()

    baseline: Dict[str, Any] = {"prompt_len": prompt_len}
    if capture_sink:
        baseline.update(_sink_measurements(clean["attention"], num_layers, band,
                                           carrier_heads))
    baseline_last_logits = clean["logits"][0, prompt_len - 1, :]
    if capture_block_outputs:
        baseline["block_outputs"] = clean["block_output"]
    if capture_attn_outputs:
        baseline["attn_outputs"] = clean["attn_output"]

    baseline_scores = None
    candidate_logits: Dict[int, torch.Tensor] = {}
    if score_candidates:
        baseline_scores = _score_candidate_set(
            engine, handle, prompt_ids, score_candidates, gold_index,
            writes=(), block_is_tuple=block_is_tuple, store_logits=candidate_logits)
        baseline.update(baseline_scores)

    rows: List[Dict[str, Any]] = []
    for patch_spec in patch_specs:
        rows.append(_run_one_patch(
            engine=engine, handle=handle, patch_spec=patch_spec,
            prompt_ids=prompt_ids, positions=positions, target_values=target_values,
            source_caches=source_caches, geom=geom, device=device,
            model_dtype=model_dtype, rope_inside=rope_inside, band=band,
            num_layers=num_layers, capture_sink=capture_sink,
            carrier_heads=carrier_heads, baseline=baseline,
            baseline_last_logits=baseline_last_logits,
            score_candidates=score_candidates, gold_index=gold_index,
            capture_block_outputs=capture_block_outputs,
            capture_attn_outputs=capture_attn_outputs,
            block_is_tuple=block_is_tuple))

    del clean
    gc.collect()

    return {
        "target_item_id": getattr(target_item, "item_id", None),
        "target_seq_len": prompt_len,
        "baseline": baseline,
        "patches": rows,
        "resolved_positions": dict(positions),
        "band": [int(band[0]), int(band[1])],
        "provenance": _prov.provenance_block(extra={
            "arch": handle.arch,
            "model_name": handle.model_name,
            "model_revision": handle.model_revision,
            "dtype": handle.dtype,
            "device": str(device),
            "engine": handle.engine,
            "model_fingerprint": fingerprint,
            "patch_registry_version": PATCH_REGISTRY_VERSION,
        }),
    }


def _score_candidate_set(engine, handle, prompt_ids: Sequence[int],
                         candidates: Sequence[Sequence[int]],
                         gold_index: Optional[int], *,
                         writes: Sequence[_Write],
                         block_is_tuple: bool,
                         store_logits: Optional[Dict[int, torch.Tensor]] = None
                         ) -> Dict[str, Any]:
    """Score every candidate with one forward each, under the same writes.

    One forward per candidate because teacher forcing needs the candidate tokens *in* the
    input and the candidates have different lengths; batching them would need padding, and
    padding moves position 0.
    """
    device = _model_device(handle)
    prompt_len = len(prompt_ids)
    per_candidate_logits: List[torch.Tensor] = []
    for index, candidate in enumerate(candidates):
        ids = torch.tensor([list(prompt_ids) + [int(t) for t in candidate]],
                           dtype=torch.long, device=device)
        result = _traced_forward(engine, _payload(engine, ids), writes=writes,
                                 capture_logits=True, block_is_tuple=block_is_tuple)
        logits = result["logits"][0]
        per_candidate_logits.append(logits)
        if store_logits is not None:
            store_logits[index] = logits

    normalised: List[float] = []
    unnormalised: List[float] = []
    for logits, candidate in zip(per_candidate_logits, candidates):
        total, norm = sequence_logprob(logits, prompt_len, candidate)
        unnormalised.append(total)
        normalised.append(norm)

    prediction = int(np.argmax(normalised))
    return {
        "candidate_logprobs": normalised,
        "candidate_logprobs_unnorm": unnormalised,
        "gold_logprob": (float(normalised[gold_index])
                         if gold_index is not None else float("nan")),
        "margin": (correct_label_margin(normalised, gold_index)
                   if gold_index is not None else float("nan")),
        "margin_unnorm": (correct_label_margin(unnormalised, gold_index)
                          if gold_index is not None else float("nan")),
        "prediction": prediction,
    }


def _run_one_patch(*, engine, handle, patch_spec: PatchSpec, prompt_ids, positions,
                   target_values, source_caches, geom, device, model_dtype,
                   rope_inside, band, num_layers, capture_sink, carrier_heads,
                   baseline, baseline_last_logits, score_candidates, gold_index,
                   capture_block_outputs, capture_attn_outputs,
                   block_is_tuple) -> Dict[str, Any]:
    """Apply one :class:`PatchSpec` and measure. Never raises for data reasons."""
    site = patch_spec.site
    key = site.key()
    surface, _kind, variant = _object_parts(site.object, rope_inside=rope_inside)
    position = positions[key]
    prompt_len = len(prompt_ids)

    row: Dict[str, Any] = {
        "patch_object": site.object,
        "patch_layer": int(site.layer),
        "patch_position": int(position),
        "kv_head": site.kv_head,
        "norm_condition": patch_spec.norm_condition,
        "source_item_id": patch_spec.source_item_id,
        "source_patch_position": None,
        "source_seq_len": None,
        "target_seq_len": prompt_len,
        "seed": int(patch_spec.seed),
        "status": "ok",
        "warning": "",
        "patch_registry_version": PATCH_REGISTRY_VERSION,
    }

    try:
        source_value = None
        if patch_spec.source_item_id is not None:
            cache = source_caches.get(patch_spec.source_item_id)
            if cache is None:
                raise KeyError(f"no SourceCache for item {patch_spec.source_item_id!r}")
            if key not in cache.tensors:
                raise KeyError(f"SourceCache for {cache.item_id!r} has no site {key!r}")
            source_value = cache.tensors[key].float()
            row["source_patch_position"] = int(cache.resolved_positions.get(key, -1))
            row["source_seq_len"] = int(cache.seq_len)

        target_value = target_values[key]
        if source_value is not None and source_value.shape != target_value.shape:
            raise ValueError(f"{key}: source shape {tuple(source_value.shape)} != target "
                             f"shape {tuple(target_value.shape)}")

        value, norms, warning = build_patch_value(patch_spec, target_value, source_value)
        row.update(norms)
        if warning:
            row["warning"] = warning

        # postrope: write the inverse rotation so the model's own RoPE reproduces `value`
        # at the target position (decision 2).
        if variant == "postrope":
            value = apply_rope_inverse(value, position, head_dim=geom["head_dim"],
                                       rope_theta=geom["rope_theta"])

        write = _make_write(handle, site, surface, position, value, device, model_dtype)
        writes = (write,)

        attention_layers = list(range(num_layers)) if capture_sink else []
        patched = _traced_forward(
            engine, _payload(engine, torch.tensor([list(prompt_ids)], dtype=torch.long,
                                                  device=device)),
            writes=writes, attention_layers=attention_layers,
            attn_output_layers=list(capture_attn_outputs),
            block_output_layers=list(capture_block_outputs),
            capture_logits=True, block_is_tuple=block_is_tuple)

        logits = patched["logits"]
        if not torch.isfinite(logits).all():
            row["status"] = "nonfinite"
            row["warning"] = (row["warning"] + "; " if row["warning"] else "") + \
                "patched logits contain non-finite values"
            return row

        if capture_sink:
            measured = _sink_measurements(patched["attention"], num_layers, band,
                                          carrier_heads)
            row["patched_sink"] = measured["sink"]
            row["patched_attn_pos1_4"] = measured["attn_pos1_4"]
            row["patched_carrier_sink"] = measured["carrier_sink"]
            row["baseline_sink"] = baseline.get("sink", float("nan"))
            row["baseline_attn_pos1_4"] = baseline.get("attn_pos1_4", float("nan"))
            row["carrier_sink_delta"] = (row["patched_carrier_sink"]
                                         - baseline.get("carrier_sink", float("nan")))

        row["jsd_output"] = jensen_shannon_divergence(
            baseline_last_logits, logits[0, prompt_len - 1, :])

        if capture_block_outputs:
            row["block_outputs"] = patched["block_output"]
        if capture_attn_outputs:
            row["attn_outputs"] = patched["attn_output"]

        if score_candidates:
            scored = _score_candidate_set(
                engine, handle, prompt_ids, score_candidates, gold_index,
                writes=writes, block_is_tuple=block_is_tuple)
            row["patched_candidate_logprobs"] = scored["candidate_logprobs"]
            row["patched_gold_logprob"] = scored["gold_logprob"]
            row["patched_margin"] = scored["margin"]
            row["patched_margin_unnorm"] = scored["margin_unnorm"]
            row["patched_prediction"] = scored["prediction"]
            row["baseline_gold_logprob"] = baseline.get("gold_logprob", float("nan"))
            row["baseline_margin"] = baseline.get("margin", float("nan"))
            row["baseline_margin_unnorm"] = baseline.get("margin_unnorm", float("nan"))
            row["baseline_prediction"] = baseline.get("prediction")
            row["margin_delta"] = row["patched_margin"] - row["baseline_margin"]

            if patch_spec.norm_condition == "identity" and \
                    abs(row["margin_delta"]) > IDENTITY_TOLERANCE:
                row["status"] = "identity_violation"
                row["warning"] = (
                    f"identity patch moved the margin by {row['margin_delta']:.3e} "
                    f"(> {IDENTITY_TOLERANCE:.0e}); 04 §5.4 says the run unit is invalid")

        del patched
    except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - GPU-only path
        row["status"] = "oom"
        row["warning"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        row["status"] = "failed"
        row["warning"] = f"{type(exc).__name__}: {exc}"
    return row


def _make_write(handle, site: PatchSite, surface: str, position: int,
                value: torch.Tensor, device, model_dtype,
                batch_index: int = 0) -> _Write:
    """Cast, flatten and shape-check one write (invariant 9)."""
    geom = geometry(handle)
    # Decision 6, second guard. PatchSpec already refuses these at construction; a caller
    # that reaches the writer by another route must still not get a silent write.
    if site.object in CAPTURE_ONLY_OBJECTS:
        raise ValueError(
            f"{site.object!r} is capture-only (04 §3.1) and cannot be written: it is a "
            "reduction over a span of positions.")
    if surface == "r":
        expected = (geom["hidden"],)
        if tuple(value.shape) != expected:
            raise ValueError(f"{site.object}: expected {expected}, got {tuple(value.shape)}")
        flat = value.reshape(-1)
        col_start, col_stop = 0, geom["hidden"]
    else:
        _require_kv_surface(handle, surface)
        expected = (geom["num_kv_heads"], geom["head_dim"])
        if tuple(value.shape) != expected:
            raise ValueError(f"{site.object}: expected {expected}, got {tuple(value.shape)}")
        col_start, col_stop = _kv_columns(handle, surface, site.kv_head)
        flat = value.reshape(-1) if site.kv_head is None else value[site.kv_head]
    return _Write(layer=int(site.layer), surface=surface, batch_index=batch_index,
                  position=int(position), value=flat.to(device=device, dtype=model_dtype),
                  col_start=col_start, col_stop=col_stop)


# ═══════════════════════════════════════════════════════════════════════════════
# Batched path — exists so invariant 7 exercises the production writer
# ═══════════════════════════════════════════════════════════════════════════════


def run_batched_logits(handle, items_input_ids: Sequence[Sequence[int]],
                       writes_by_item: Dict[int, List[Tuple[PatchSite, torch.Tensor]]]
                       ) -> torch.Tensor:
    """Forward a batch, patching only the items named in ``writes_by_item``.

    Used by ``tests/test_cross_example_patching.py`` invariant 7 (no batch leakage) and by
    any future batched production path. All items must have the same length: padding would
    move position 0, which is the object under study, so a length mismatch raises rather
    than silently padding.

    Returns ``[batch, seq, vocab]`` logits on CPU in fp32.
    """
    engine = handle.nn_engine
    lengths = {len(ids) for ids in items_input_ids}
    if len(lengths) != 1:
        raise ValueError(
            "Batched patching requires equal-length items; padding would move position 0. "
            f"Got lengths {sorted(lengths)}.")
    device = _model_device(handle)
    model_dtype = _model_dtype(handle)
    input_ids = torch.tensor([list(ids) for ids in items_input_ids],
                             dtype=torch.long, device=device)
    seq_len = int(input_ids.shape[-1])
    rope_inside = bool(engine.spec.rope_applied_inside_attn)

    writes: List[_Write] = []
    for batch_index, site_values in writes_by_item.items():
        for site, value in site_values:
            surface, _kind, _variant = _object_parts(site.object, rope_inside=rope_inside)
            position = resolve_position(site.object, seq_len, rope_inside=rope_inside,
                                        override=site.position)
            writes.append(_make_write(handle, site, surface, position, value,
                                      device, model_dtype, batch_index=int(batch_index)))

    result = _traced_forward(engine, _payload(engine, input_ids), writes=writes,
                             capture_logits=True,
                             block_is_tuple=_block_output_is_tuple(engine))
    return result["logits"]
