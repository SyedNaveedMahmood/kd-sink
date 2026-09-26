# -*- coding: utf-8 -*-
"""train_distillation.py — E6A: distil TinyStories-33M into a randomly-initialised 8M (WP5).

``03_MODULE_SPEC_e6_transformation.md`` §1–3, closing G7 (the repo had no training code at
all). Three objectives share one loop and differ only in loss weights:

===  ==================================================
D0   ``L_CE``
D1   ``0.50*L_CE + 0.50*L_KD``                  (T=2.0)
D2   ``0.45*L_CE + 0.45*L_KD + 0.10*L_ATTN``    (T=2.0)
===  ==================================================

Every component is logged every step **even when its weight is zero** — ``L_ATTN`` is an
evaluation metric for D0 and D1, not only a training term (design §8.6).

    python transformation_inheritance/train_distillation.py \\
      --config transformation_inheritance/configs/e6a_logit_attention_kd.yaml \\
      --seed 0 --output-dir transformation_inheritance/results \\
      [--max-steps 2000] [--stop-after 2000] [--resume auto] [--smoke]

Things that are easy to get wrong here, and are therefore explicit
------------------------------------------------------------------
* **The student is randomly initialised from the TinyStories-8M *config*.** The public 8M
  checkpoint is an independent-convergence *reference*, never a starting point. Three
  conditions at one seed must have byte-identical initial weights, and
  ``tests/test_distillation_init.py`` asserts the sha256 of the initial state dict.
* **``L_ATTN`` masks by exclusion, never renormalisation.** GPT-Neo alternates global and
  local attention; a mapped teacher/student pair may have different attention types, and
  renormalising over invalid positions would optimise a quantity that does not exist in the
  teacher. ``06_TEST_PLAN.md`` §3 singles this out.
* **``L_ATTN`` sums over keys and averages over queries** — the reduction axes are layer,
  head and query (``03`` §1.5), because a JSD is a divergence between two *distributions*.
  Averaging over ``(query, key)`` cells instead silently divides the registered objective by
  the mean number of valid keys per query (64.5 at S=128); the first pilot ran that way.
  ``run_config.json`` records :data:`ATTN_REDUCTION` so a corrected run is distinguishable
  from that one (CLAUDE.md trap 22).
* **``T**2`` is applied exactly once**, inside :func:`kd_loss`.
* **The step-0 checkpoint is saved by an explicit call before the loop**, not by a branch
  inside it — a branch would run after the first optimiser update on resume.
* **The band for a 4-layer teacher.** Evaluation uses ``depth_band.normalised_depth_band``,
  never the legacy ``compute_band``, which returns a single layer at L=4 (CLAUDE.md trap 1).

``--smoke`` runs 5 steps on tiny random GPT-Neo configs, CPU, fp32, with no downloads.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn.functional as F

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import block_corpus_cache as bcc  # noqa: E402
import datasets_loader as dl  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import normalised_depth_band  # noqa: E402

EXPERIMENT_ID = "e6a"

#: HF architecture class -> the arch key `run_config.json` records and the fingerprint
#: runner resolves. Mirrors ``evaluate_transformation.ARCH_BY_CLASS``; kept here so the
#: trainer does not import the evaluator.
ARCH_BY_CLASS: Dict[str, str] = {
    "GPTNeoForCausalLM": "neo",
    "GPT2LMHeadModel": "gpt2",
    "Qwen2ForCausalLM": "qwen",
    "OPTForCausalLM": "opt",
}

#: Teacher layer -> student layer (design §8.3).
DEFAULT_LAYER_MAP: Dict[int, int] = {0: 1, 1: 3, 2: 5, 3: 7}

#: Checkpoints, including step 0 *before* the first optimiser update (§1.6).
DEFAULT_CHECKPOINTS: Tuple[int, ...] = (0, 100, 250, 500, 1000, 2000, 5000, 10000)

#: How :func:`attn_js_loss` reduces, recorded in ``run_config.json``. ``03`` §1.5's axes are
#: layer, head and query; the E6A pilot of 2026-07-30/31 divided by valid ``(query, key)``
#: cells instead, which scaled ``L_ATTN`` down by 64.5× at S=128. A run directory that does
#: not carry this field predates the fix and its D2 arm did not optimise the registered
#: objective — do not pool the two (CLAUDE.md trap 22).
ATTN_REDUCTION = "sum_over_keys_mean_over_queries"

#: Head-alignment methods accepted by :func:`resolve_head_alignment`.  Existing configs
#: omit the field and therefore stay on ``index`` byte-for-byte; the unequal-head E6A
#: extension opts into ``amad_jsd`` explicitly.  The latter is an AMAD-style alignment,
#: not the published AMAD objective: cosine-soft head weights are retained, while the
#: divergence and reduction remain E6A's registered JSD.
LEGACY_HEAD_ALIGNMENT = "index"
MEAN_HEAD_ALIGNMENT = "mean"
AMAD_JSD_HEAD_ALIGNMENT = "amad_jsd"
HEAD_ALIGNMENT_VERSION = "amad_style_teacher_to_soft_student_jsd_v1"


def resolve_head_alignment(loss_config: Dict[str, Any]) -> Dict[str, Any]:
    """Return a complete, validated head-alignment specification.

    ``loss.head_alignment`` may be absent (the historical one-to-one/index objective), a
    method string, or a mapping.  The mapping is deliberately narrow: a misspelled
    divergence or alignment direction must abort instead of silently producing a new
    scientific intervention.

    ``amad_jsd`` follows Zhao et al.'s AMAD *alignment* construction: for every teacher
    head, cosine similarities to all student heads are softmaxed and the student attention
    maps are mixed with those weights.  It then uses E6A's JSD rather than AMAD's primary
    L2-normalised MSE formulation or its KL variants.  The alignment weights remain attached
    to autograd, as in the differentiable AMAD construction.
    """
    legacy_raw = loss_config.get("attention_head_alignment")
    if "head_alignment" in loss_config and legacy_raw is not None:
        raise ValueError(
            "loss may declare only one of head_alignment or attention_head_alignment")
    if legacy_raw is not None:
        legacy_method = str(legacy_raw).strip().lower()
        if legacy_method not in {"strict", MEAN_HEAD_ALIGNMENT}:
            raise ValueError(
                "loss.attention_head_alignment must be 'strict' or 'mean', got "
                f"{legacy_method!r}")
        raw: Any = (LEGACY_HEAD_ALIGNMENT if legacy_method == "strict"
                    else MEAN_HEAD_ALIGNMENT)
    else:
        raw = loss_config.get("head_alignment", LEGACY_HEAD_ALIGNMENT)
    if isinstance(raw, str):
        spec: Dict[str, Any] = {"method": raw}
    elif isinstance(raw, dict):
        spec = dict(raw)
    else:
        raise ValueError(
            "loss.head_alignment must be a method string or mapping, got "
            f"{type(raw).__name__}")

    method = str(spec.get("method", "")).strip().lower()
    if method == "strict":
        method = LEGACY_HEAD_ALIGNMENT
    if method in {LEGACY_HEAD_ALIGNMENT, MEAN_HEAD_ALIGNMENT}:
        unknown = set(spec) - {"method"}
        if unknown:
            raise ValueError(
                f"{method} head alignment takes no options; unexpected keys "
                f"{sorted(unknown)}")
        if method == MEAN_HEAD_ALIGNMENT:
            return {
                "method": MEAN_HEAD_ALIGNMENT,
                "version": "mean_postsoftmax_jsd_v1",
                "similarity": None,
                "softmax_temperature": None,
                "direction": "mean_to_mean",
                "divergence": "jsd",
                "alignment_weights_gradient": None,
            }
        return {
            "method": LEGACY_HEAD_ALIGNMENT,
            "version": "legacy_index_jsd_v1",
            "similarity": None,
            "softmax_temperature": None,
            "direction": "index_to_index",
            "divergence": "jsd",
            "alignment_weights_gradient": None,
        }

    if method != AMAD_JSD_HEAD_ALIGNMENT:
        raise ValueError(
            f"unknown loss.head_alignment method {method!r}; expected "
            f"{LEGACY_HEAD_ALIGNMENT!r}, {MEAN_HEAD_ALIGNMENT!r}, or "
            f"{AMAD_JSD_HEAD_ALIGNMENT!r}")

    allowed = {
        "method", "similarity", "softmax_temperature", "direction", "divergence",
        "alignment_weights_gradient",
    }
    unknown = set(spec) - allowed
    if unknown:
        raise ValueError(
            f"amad_jsd head alignment has unexpected keys {sorted(unknown)}")

    similarity = str(spec.get("similarity", "cosine")).strip().lower()
    direction = str(spec.get("direction", "teacher_to_student")).strip().lower()
    divergence = str(spec.get("divergence", "jsd")).strip().lower()
    gradient = str(spec.get("alignment_weights_gradient", "attached")).strip().lower()
    temperature = float(spec.get("softmax_temperature", 1.0))
    required = {
        "similarity": (similarity, "cosine"),
        "direction": (direction, "teacher_to_student"),
        "divergence": (divergence, "jsd"),
        "alignment_weights_gradient": (gradient, "attached"),
    }
    bad = {name: observed for name, (observed, expected) in required.items()
           if observed != expected}
    if bad:
        raise ValueError(
            "amad_jsd is a fixed scientific method; unsupported option values "
            f"{bad}. Expected cosine / teacher_to_student / jsd / attached.")
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError(
            "loss.head_alignment.softmax_temperature must be finite and > 0, got "
            f"{temperature!r}")

    return {
        "method": AMAD_JSD_HEAD_ALIGNMENT,
        "version": HEAD_ALIGNMENT_VERSION,
        "similarity": similarity,
        "softmax_temperature": temperature,
        "direction": direction,
        "divergence": divergence,
        "alignment_weights_gradient": gradient,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(path) -> Dict[str, Any]:
    """Load a YAML experiment config. The repo's first YAML reader."""
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    # YAML turns `{0: 1}` keys into ints already, but a JSON-ish config may use strings.
    loss = payload.get("loss", {})
    if "layer_map" in loss:
        loss["layer_map"] = {int(k): int(v) for k, v in loss["layer_map"].items()}
    # Recorded into run_config.json so a later evaluation can find the config that
    # produced the run. Underscore-prefixed: it is loader-supplied, not a YAML field.
    payload["_config_path"] = str(Path(path))
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# Losses (§1.5)
# ═══════════════════════════════════════════════════════════════════════════════


def ce_loss(student_logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Next-token cross-entropy, mean over non-ignored positions.

    ``labels`` is the input sequence; the shift is done here so callers never have to
    remember it. ``-100`` positions are ignored by ``F.cross_entropy``.
    """
    shifted_logits = student_logits[:, :-1, :].contiguous()
    shifted_labels = labels[:, 1:].contiguous()
    return F.cross_entropy(shifted_logits.reshape(-1, shifted_logits.size(-1)),
                           shifted_labels.reshape(-1), ignore_index=-100)


def kd_loss(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
            T: float = 2.0, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """``KL(teacher_T || student_T)`` per token, mean over valid tokens, times ``T**2``.

    The ``T**2`` rescaling (which restores the gradient magnitude that softening removes)
    is applied **once**, here. Teacher logits are detached; the caller is additionally
    expected to run the teacher forward under ``torch.no_grad()``.
    """
    student_log_probs = F.log_softmax(student_logits / T, dim=-1)
    teacher_probs = F.softmax(teacher_logits.detach() / T, dim=-1)
    teacher_log_probs = F.log_softmax(teacher_logits.detach() / T, dim=-1)

    per_token = (teacher_probs * (teacher_log_probs - student_log_probs)).sum(dim=-1)
    if mask is not None:
        mask = mask.to(per_token.dtype)
        denom = mask.sum().clamp_min(1.0)
        value = (per_token * mask).sum() / denom
    else:
        value = per_token.mean()
    return value * (T ** 2)


def causal_mask(seq_len: int, device) -> torch.Tensor:
    """``[S, S]`` boolean: query *i* may attend key *j*."""
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def local_window_mask(seq_len: int, window: int, device) -> torch.Tensor:
    """``[S, S]`` boolean causal mask restricted to a ``window``-wide local attention band.

    Mirrors GPT-Neo's local attention: query *i* may attend key *j* when
    ``i - window < j <= i``.
    """
    idx = torch.arange(seq_len, device=device)
    delta = idx[:, None] - idx[None, :]
    return (delta >= 0) & (delta < window)


def attention_type_mask(attention_type: str, seq_len: int, window: int,
                        device) -> torch.Tensor:
    """The valid-position mask for one layer, from its declared attention type."""
    if attention_type == "local":
        return local_window_mask(seq_len, window, device)
    return causal_mask(seq_len, device)


def attn_js_loss(student_attn: Sequence[torch.Tensor],
                 teacher_attn: Sequence[torch.Tensor],
                 layer_map: Dict[int, int],
                 valid_mask: Dict[Tuple[int, int], torch.Tensor],
                 pair_types: Optional[Dict[Tuple[int, int], str]] = None,
                 *, collect_breakdown: bool = True,
                 head_alignment: str = "strict"
                 ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Mean Jensen-Shannon divergence between mapped attention distributions.

    ``student_attn`` / ``teacher_attn`` are per-layer ``[B, H, S, S]`` post-softmax maps.
    ``valid_mask[(t, s)]`` is the elementwise AND of both layers' causal/window masks.
    The strict default requires equal head counts and preserves the original instrument.
    ``head_alignment="mean"`` is an explicit cross-head-count instrument: each model is
    averaged over its own heads before the same per-query JSD is computed.

    **The reduction axes are layer, head and query — never key** (``03`` §1.5). A JSD is a
    divergence between two *distributions*: for one (batch, head, query) triple the
    contributions are **summed over key positions**, and only the resulting per-query
    divergences are averaged. Dividing by the number of valid ``(query, key)`` cells instead
    computes a mean *pointwise contribution*, which is the same quantity scaled down by the
    mean number of valid keys per query — ``(S + 1) / 2`` for a full causal mask, i.e. **64.5×
    at S = 128**. That is what the first E6A pilot trained under: D2's registered
    ``0.10 * L_ATTN`` term was 0.022% of its loss instead of ~1.4%, so its attention
    objective was effectively inert (CLAUDE.md trap 22). The weight is unchanged; only the
    denominator is, because the weight is pre-registered and the denominator was a defect.

    **Masked entries are excluded, not renormalised.** The two distributions are compared
    only where both are defined. Renormalising over the intersection would make the student
    match a distribution the teacher never produced, and ``06_TEST_PLAN.md`` §3 flags that as
    the failure mode that would test H2 against an artefact. A query row with no valid key
    contributes nothing *and* is not counted in the denominator — counting it would divide a
    correct numerator by an inflated row count, reintroducing a scale error of the same kind.

    Returns ``(loss, breakdown)`` where the breakdown reports each mapped pair separately,
    tagged ``global_global`` / ``local_local`` / ``mixed``, so design §16.1's global- and
    local-mapped comparison can be reported without a second pass. Each pair carries both
    ``n_valid`` (valid cells — a diagnostic, and what the pilot divided by) and
    ``n_query_rows`` (the denominator actually used), so the two readings can never be
    confused in an artefact again.

    ``collect_breakdown=False`` computes the **same loss** and skips only the reporting: the
    per-pair dictionaries and the ``n_valid`` cell count, which is a diagnostic and enters no
    denominator. It exists because reading those scalars costs three GPU→CPU
    synchronisations per mapped pair — eighteen per micro-batch on the six-pair gpt2 arm —
    and :func:`_average_components` keeps only the *last* micro-batch's breakdown while
    :func:`evaluate` discards it entirely. The breakdown then reports
    ``collected: False`` rather than an empty ``per_pair`` that could be misread as "no
    pairs were mapped" (the trap-16 rule: an absent measurement must never render as a
    measured one).

    When it *is* collected, the per-pair scalars are read back in a single stacked
    ``.tolist()`` rather than one ``.item()`` each. Widening a float32 scalar to a Python
    float is the same operation either way; only the number of pipeline stalls changes.
    """
    if head_alignment not in {"strict", "mean"}:
        raise ValueError(
            f"unknown attention head alignment {head_alignment!r}; expected 'strict' or "
            "'mean'")

    pair_types = pair_types or {}
    ordered_pairs = sorted(layer_map.items())
    pair_scalars: List[torch.Tensor] = []
    total = None
    n_pairs = 0

    for teacher_layer, student_layer in ordered_pairs:
        # Cast one mapped pair at a time.  Casting every layer eagerly keeps a second
        # float32 copy of all teacher attention maps alive; that is unnecessary and is
        # material for 36-layer teachers.  The arithmetic is unchanged from the former
        # caller-side ``[a.float() for a in ...]`` conversion.
        t_map = teacher_attn[teacher_layer].float()
        s_map = student_attn[student_layer].float()
        if (t_map.ndim != 4 or s_map.ndim != 4
                or t_map.shape[0] != s_map.shape[0]
                or t_map.shape[2:] != s_map.shape[2:]):
            raise ValueError(
                f"attention shape mismatch for teacher layer {teacher_layer} -> student "
                f"layer {student_layer}: {tuple(t_map.shape)} vs {tuple(s_map.shape)}. "
                "L_ATTN requires equal batch and sequence dimensions.")
        if head_alignment == "strict" and t_map.shape[1] != s_map.shape[1]:
            raise ValueError(
                f"attention shape mismatch for teacher layer {teacher_layer} -> student "
                f"layer {student_layer}: {tuple(t_map.shape)} vs {tuple(s_map.shape)}. "
                "Strict L_ATTN requires equal head counts; use the explicitly registered "
                "'mean' alignment only for a cross-head-count experiment.")
        if head_alignment == "mean":
            # A mean of post-softmax head maps is still a probability distribution over
            # keys.  This is the only head-count-invariant reduction offered: grouping or
            # truncating heads would invent an arbitrary correspondence between independently
            # parameterised heads.  The strict default preserves every existing experiment.
            t_map = t_map.mean(dim=1, keepdim=True)
            s_map = s_map.mean(dim=1, keepdim=True)
        mask = valid_mask[(teacher_layer, student_layer)]
        while mask.dim() < t_map.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(t_map.dtype)

        m = 0.5 * (t_map + s_map)
        eps = 1e-12
        kl_t = t_map * ((t_map + eps).log() - (m + eps).log())
        kl_s = s_map * ((s_map + eps).log() - (m + eps).log())
        jsd = 0.5 * kl_t + 0.5 * kl_s

        expanded = mask.expand_as(jsd)
        # Sum over KEY positions -> one JSD per (batch, head, query); then average over the
        # query rows that have at least one valid key.
        per_query = (jsd * mask).sum(dim=-1)
        query_rows = expanded.to(torch.bool).any(dim=-1)
        n_rows = query_rows.sum().to(per_query.dtype).clamp_min(1.0)
        pair_loss = (per_query * query_rows.to(per_query.dtype)).sum() / n_rows

        total = pair_loss if total is None else total + pair_loss
        n_pairs += 1
        if collect_breakdown:
            # `n_valid` is a diagnostic — it is what the mis-scaled pilot divided by — and
            # is computed only when it is going to be reported.
            pair_scalars.append(
                torch.stack([pair_loss.detach(), expanded.sum(), n_rows]))

    if total is None:
        device = student_attn[0].device if len(student_attn) else torch.device("cpu")
        return torch.zeros((), device=device), {"per_pair": [], "n_pairs": 0}

    loss = total / n_pairs
    if not collect_breakdown:
        # `collected` appears ONLY here. A collected breakdown keeps exactly the keys it
        # has always had, because this dict is written verbatim into `train_log.jsonl` and
        # adding a field to it would move the artefact — the one thing this whole change
        # is not allowed to do. Its absence therefore means "collected", and its presence
        # (always `False`) means the empty `per_pair` beside it is a skipped measurement
        # rather than a mapping with no pairs.
        return loss, {"per_pair": [], "n_pairs": n_pairs, "collected": False}

    # One synchronisation for every pair's three scalars, instead of three per pair.
    read_back = torch.stack(pair_scalars).tolist()
    per_pair: List[Dict[str, Any]] = [
        {
            "teacher_layer": int(teacher_layer),
            "student_layer": int(student_layer),
            "pair_type": pair_types.get((teacher_layer, student_layer), "unknown"),
            "jsd": float(jsd_value),
            "n_valid": float(n_valid),
            "n_query_rows": float(n_query_rows),
        }
        for (teacher_layer, student_layer), (jsd_value, n_valid, n_query_rows)
        in zip(ordered_pairs, read_back)
    ]
    breakdown = {
        "per_pair": per_pair,
        "n_pairs": n_pairs,
        "global_global": _mean_of([p["jsd"] for p in per_pair
                                   if p["pair_type"] == "global_global"]),
        "local_local": _mean_of([p["jsd"] for p in per_pair
                                 if p["pair_type"] == "local_local"]),
        "mixed": _mean_of([p["jsd"] for p in per_pair if p["pair_type"] == "mixed"]),
        "n_mixed": sum(1 for p in per_pair if p["pair_type"] == "mixed"),
    }
    if head_alignment != "strict":
        breakdown["head_alignment"] = head_alignment
    return loss, breakdown


def amad_js_attn_loss(student_attn: Sequence[torch.Tensor],
                      teacher_attn: Sequence[torch.Tensor],
                      layer_map: Dict[int, int],
                      valid_mask: Dict[Tuple[int, int], torch.Tensor],
                      pair_types: Optional[Dict[Tuple[int, int], str]] = None,
                      *, softmax_temperature: float = 1.0,
                      collect_breakdown: bool = True
                      ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """AMAD-style soft head alignment followed by E6A's registered JSD.

    For each mapped layer and batch item, the teacher and student attention maps are
    flattened over their common valid ``(query, key)`` cells and L2-normalised.  Their
    cosine-similarity matrix has shape ``[B, H_teacher, H_student]``.  A softmax over the
    student-head axis gives one convex mixture of student attention maps for every teacher
    head.  The mixture is compared with that teacher head using the *same* per-query JSD as
    :func:`attn_js_loss`: sum over keys, mean over batch/head/valid-query rows, then mean
    over mapped layers.

    This is intentionally named an **AMAD-style JSD extension**, not AMAD.  AMAD's primary
    Variant 1 applies L2-normalised MSE after the cosine-soft alignment, and the paper also
    studies L1-normalised KL variants.  This method instead forms the mixture from the exact
    attention tensors consumed by the original E6A JSD term (including the model's existing
    training-time dropout semantics), so the registered ``0.10`` coefficient and reduction
    are not redefined.  On normalised attention rows the score has the usual JSD bound.  The
    similarity weights are differentiable (not detached).

    The construction is invariant to student-head permutation and accepts unequal head
    counts.  It still requires equal batch and sequence dimensions; a mismatch there is
    not a head-alignment problem and is refused.
    """
    if not math.isfinite(float(softmax_temperature)) or softmax_temperature <= 0:
        raise ValueError(
            f"softmax_temperature must be finite and > 0, got {softmax_temperature!r}")

    pair_types = pair_types or {}
    ordered_pairs = sorted(layer_map.items())
    pair_scalars: List[torch.Tensor] = []
    pair_head_counts: List[Tuple[int, int]] = []
    total = None
    n_pairs = 0

    for teacher_layer, student_layer in ordered_pairs:
        # Match the legacy/mean path's fp32 arithmetic without retaining float copies of
        # every layer at once; this matters for the 24/36-layer teachers.
        t_map = teacher_attn[teacher_layer].float()
        s_map = student_attn[student_layer].float()
        if t_map.dim() != 4 or s_map.dim() != 4:
            raise ValueError(
                "AMAD-style JSD expects [batch, head, query, key] attention maps; "
                f"teacher layer {teacher_layer} has {tuple(t_map.shape)} and student "
                f"layer {student_layer} has {tuple(s_map.shape)}")
        if (t_map.shape[0] != s_map.shape[0]
                or t_map.shape[2:] != s_map.shape[2:]):
            raise ValueError(
                f"attention batch/sequence mismatch for teacher layer {teacher_layer} -> "
                f"student layer {student_layer}: {tuple(t_map.shape)} vs "
                f"{tuple(s_map.shape)}. AMAD-style JSD aligns heads only.")

        mask = valid_mask[(teacher_layer, student_layer)]
        while mask.dim() < t_map.dim():
            mask = mask.unsqueeze(0)
        mask = mask.to(t_map.dtype)

        # AMAD's alignment: cosine similarities between flattened, L2-normalised maps,
        # computed only over positions both mapped layers define. Invalid positions are
        # excluded rather than renormalised, preserving E6A's mask contract.
        t_flat = (t_map * mask).flatten(start_dim=-2)
        s_flat = (s_map * mask).flatten(start_dim=-2)
        t_unit = F.normalize(t_flat, p=2, dim=-1, eps=1e-12)
        s_unit = F.normalize(s_flat, p=2, dim=-1, eps=1e-12)
        similarity = torch.matmul(t_unit, s_unit.transpose(-1, -2))
        weights = torch.softmax(similarity / float(softmax_temperature), dim=-1)

        # [B, Ht, Hs] x [B, Hs, Q, K] -> [B, Ht, Q, K]. The weights are positive and sum
        # to one, so this is a convex mixture of the exact student tensors the legacy E6A
        # objective would consume. We deliberately do not add a renormalisation step.
        aligned_student = torch.einsum("bts,bsqk->btqk", weights, s_map)

        m = 0.5 * (t_map + aligned_student)
        eps = 1e-12
        kl_t = t_map * ((t_map + eps).log() - (m + eps).log())
        kl_s = aligned_student * (
            (aligned_student + eps).log() - (m + eps).log())
        jsd = 0.5 * kl_t + 0.5 * kl_s

        expanded = mask.expand_as(jsd)
        per_query = (jsd * mask).sum(dim=-1)
        query_rows = expanded.to(torch.bool).any(dim=-1)
        n_rows = query_rows.sum().to(per_query.dtype).clamp_min(1.0)
        pair_loss = (per_query * query_rows.to(per_query.dtype)).sum() / n_rows

        total = pair_loss if total is None else total + pair_loss
        n_pairs += 1
        if collect_breakdown:
            entropy = -(weights * (weights + eps).log()).sum(dim=-1).mean()
            mean_max_weight = weights.max(dim=-1).values.mean()
            pair_scalars.append(torch.stack([
                pair_loss.detach(), expanded.sum(), n_rows,
                entropy.detach(), mean_max_weight.detach(),
            ]))
            pair_head_counts.append((int(t_map.shape[1]), int(s_map.shape[1])))

    if total is None:
        device = student_attn[0].device if len(student_attn) else torch.device("cpu")
        return torch.zeros((), device=device), {
            "per_pair": [], "n_pairs": 0,
            "alignment_method": AMAD_JSD_HEAD_ALIGNMENT,
        }

    loss = total / n_pairs
    if not collect_breakdown:
        return loss, {
            "per_pair": [], "n_pairs": n_pairs, "collected": False,
            "alignment_method": AMAD_JSD_HEAD_ALIGNMENT,
        }

    read_back = torch.stack(pair_scalars).tolist()
    per_pair: List[Dict[str, Any]] = []
    for ((teacher_layer, student_layer), values, (teacher_heads, student_heads)) in zip(
            ordered_pairs, read_back, pair_head_counts):
        jsd_value, n_valid, n_query_rows, entropy, mean_max_weight = values
        per_pair.append({
            "teacher_layer": int(teacher_layer),
            "student_layer": int(student_layer),
            "pair_type": pair_types.get((teacher_layer, student_layer), "unknown"),
            "jsd": float(jsd_value),
            "n_valid": float(n_valid),
            "n_query_rows": float(n_query_rows),
            "teacher_heads": teacher_heads,
            "student_heads": student_heads,
            "alignment_entropy_nats": float(entropy),
            "alignment_mean_max_weight": float(mean_max_weight),
        })
    breakdown = {
        "per_pair": per_pair,
        "n_pairs": n_pairs,
        "alignment_method": AMAD_JSD_HEAD_ALIGNMENT,
        "alignment_version": HEAD_ALIGNMENT_VERSION,
        "softmax_temperature": float(softmax_temperature),
        "global_global": _mean_of([p["jsd"] for p in per_pair
                                   if p["pair_type"] == "global_global"]),
        "local_local": _mean_of([p["jsd"] for p in per_pair
                                 if p["pair_type"] == "local_local"]),
        "mixed": _mean_of([p["jsd"] for p in per_pair if p["pair_type"] == "mixed"]),
        "n_mixed": sum(1 for p in per_pair if p["pair_type"] == "mixed"),
    }
    return loss, breakdown


def attention_distillation_loss(student_attn: Sequence[torch.Tensor],
                                teacher_attn: Sequence[torch.Tensor],
                                layer_map: Dict[int, int],
                                valid_mask: Dict[Tuple[int, int], torch.Tensor],
                                pair_types: Optional[Dict[Tuple[int, int], str]],
                                alignment: Dict[str, Any], *,
                                collect_breakdown: bool = True
                                ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Dispatch to the explicitly configured attention objective."""
    method = alignment["method"]
    if method == LEGACY_HEAD_ALIGNMENT:
        return attn_js_loss(
            student_attn, teacher_attn, layer_map, valid_mask, pair_types,
            collect_breakdown=collect_breakdown)
    if method == MEAN_HEAD_ALIGNMENT:
        return attn_js_loss(
            student_attn, teacher_attn, layer_map, valid_mask, pair_types,
            collect_breakdown=collect_breakdown, head_alignment="mean")
    if method == AMAD_JSD_HEAD_ALIGNMENT:
        return amad_js_attn_loss(
            student_attn, teacher_attn, layer_map, valid_mask, pair_types,
            softmax_temperature=float(alignment["softmax_temperature"]),
            collect_breakdown=collect_breakdown)
    raise ValueError(f"unsupported resolved head alignment {method!r}")


def _mean_of(values: Sequence[float]) -> Optional[float]:
    """Mean, or ``None`` when the category is empty.

    ``None`` rather than ``nan`` so the JSONL log stays strict JSON — ``json.dumps``
    emits a bare ``NaN`` token that non-Python readers reject.
    """
    return float(np.mean(values)) if len(values) else None


# ═══════════════════════════════════════════════════════════════════════════════
# Attention-type bookkeeping
# ═══════════════════════════════════════════════════════════════════════════════


def attention_types(config) -> List[str]:
    """Per-layer ``"global"``/``"local"`` pattern, read from the config, never assumed.

    ``GPTNeoConfig`` exposes ``attention_layers`` directly; when it only carries the
    compressed ``attention_types`` form it is expanded here. Architectures without the
    concept report every layer as global.
    """
    layers = getattr(config, "attention_layers", None)
    if layers:
        return [str(x) for x in layers]

    compressed = getattr(config, "attention_types", None)
    if compressed:
        expanded: List[str] = []
        for entry in compressed:
            kinds, repeats = entry
            expanded.extend(list(kinds) * int(repeats))
        return [str(x) for x in expanded]

    n_layers = int(getattr(config, "num_layers", None)
                   or getattr(config, "num_hidden_layers", 0))
    return ["global"] * n_layers


def build_valid_masks(teacher_config, student_config, layer_map: Dict[int, int],
                      seq_len: int, device
                      ) -> Tuple[Dict[Tuple[int, int], torch.Tensor],
                                 Dict[Tuple[int, int], str]]:
    """``valid_mask[(t, s)]`` = teacher-layer mask AND student-layer mask, plus pair types.

    Where a mapped pair has different attention types the loss is computed on the
    intersection and the pair is tagged ``mixed``, so design §16.1 can report global-mapped
    and local-mapped pairs separately rather than pooling incomparable quantities.
    """
    t_types = attention_types(teacher_config)
    s_types = attention_types(student_config)
    t_window = int(getattr(teacher_config, "window_size", seq_len) or seq_len)
    s_window = int(getattr(student_config, "window_size", seq_len) or seq_len)

    masks: Dict[Tuple[int, int], torch.Tensor] = {}
    types: Dict[Tuple[int, int], str] = {}
    for teacher_layer, student_layer in layer_map.items():
        t_kind = t_types[teacher_layer]
        s_kind = s_types[student_layer]
        t_mask = attention_type_mask(t_kind, seq_len, t_window, device)
        s_mask = attention_type_mask(s_kind, seq_len, s_window, device)
        masks[(teacher_layer, student_layer)] = t_mask & s_mask
        types[(teacher_layer, student_layer)] = (
            f"{t_kind}_{s_kind}" if t_kind == s_kind else "mixed")
    return masks, types


# ═══════════════════════════════════════════════════════════════════════════════
# Startup assertions (design-delta D6)
# ═══════════════════════════════════════════════════════════════════════════════


def vocab_assertions(tokenizer, teacher_config, public_config, student_config,
                     head_alignment: Optional[Dict[str, Any]] = None, *,
                     attention_head_alignment: Optional[str] = None
                     ) -> Dict[str, Any]:
    """Assert tokenizer/config agreement across teacher, public 8M and student (§1.2).

    Returns the per-check booleans for ``run_config.json``; the caller aborts on any
    ``False``. A silent vocab mismatch would make the KD term compare distributions over
    different symbol sets — plausible-looking numbers, no crash.
    """
    if head_alignment is not None and attention_head_alignment is not None:
        raise ValueError(
            "provide head_alignment or attention_head_alignment, not both")
    alignment = (resolve_head_alignment(
        {"attention_head_alignment": attention_head_alignment})
        if attention_head_alignment is not None
        else (head_alignment or resolve_head_alignment({})))
    teacher_heads = _num_heads(teacher_config)
    student_heads = _num_heads(student_config)
    head_counts_equal = teacher_heads == student_heads
    heads_compatible = (
        head_counts_equal
        if alignment["method"] == LEGACY_HEAD_ALIGNMENT
        else teacher_heads > 0 and student_heads > 0)
    checks = {
        "vocab_size": (tokenizer.vocab_size == teacher_config.vocab_size),
        "len_tokenizer": (len(tokenizer) == public_config.vocab_size
                          == student_config.vocab_size),
        "bos": (tokenizer.bos_token_id == teacher_config.bos_token_id),
        "eos": (tokenizer.eos_token_id == teacher_config.eos_token_id),
        # Historical key retained for artefact compatibility. Under index alignment it
        # still means equality; explicit mean/amad_jsd methods accept valid unequal axes.
        "heads": heads_compatible,
    }
    observed = {
        "tokenizer_vocab_size": int(tokenizer.vocab_size),
        "len_tokenizer": int(len(tokenizer)),
        "teacher_vocab_size": int(teacher_config.vocab_size),
        "public_vocab_size": int(public_config.vocab_size),
        "student_vocab_size": int(student_config.vocab_size),
        "tokenizer_bos": tokenizer.bos_token_id,
        "tokenizer_eos": tokenizer.eos_token_id,
        "teacher_bos": teacher_config.bos_token_id,
        "teacher_eos": teacher_config.eos_token_id,
        "teacher_heads": teacher_heads,
        "student_heads": student_heads,
        "head_counts_equal": head_counts_equal,
        "attention_head_alignment": (
            "strict" if alignment["method"] == LEGACY_HEAD_ALIGNMENT
            else alignment["method"]),
        "head_alignment_method": alignment["method"],
        "teacher_attention_layers": attention_types(teacher_config),
        "student_attention_layers": attention_types(student_config),
        "teacher_window_size": getattr(teacher_config, "window_size", None),
        "student_window_size": getattr(student_config, "window_size", None),
    }
    return {"checks": checks, "observed": observed, "passed": all(checks.values())}


def model_contract_assertions(teacher_config, public_config, student_config,
                              contract: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Verify pinned model geometry before a real run starts.

    Model ids are not geometry.  The new scale arm records the expected layer/head/width
    triples prospectively and checks the loaded configs, including the public reference,
    so a moved tag or wrong config cannot silently change the experiment.
    """
    if not contract:
        return {"checks": {}, "observed": {}, "passed": True}
    if not isinstance(contract, dict):
        raise ValueError("model_contract must be a mapping")

    role_configs = {
        "teacher": teacher_config,
        "student": student_config,
        "public_reference": public_config,
    }
    allowed_fields = {"num_layers", "num_heads", "hidden_size"}
    missing_roles = set(role_configs) - set(contract)
    if missing_roles:
        raise ValueError(
            f"model_contract is incomplete; missing roles {sorted(missing_roles)}")
    checks: Dict[str, bool] = {}
    observed: Dict[str, Any] = {}
    for role, expected in contract.items():
        if role not in role_configs:
            raise ValueError(
                f"model_contract has unknown role {role!r}; expected "
                f"{sorted(role_configs)}")
        if not isinstance(expected, dict):
            raise ValueError(f"model_contract.{role} must be a mapping")
        unknown = set(expected) - allowed_fields
        if unknown:
            raise ValueError(
                f"model_contract.{role} has unknown fields {sorted(unknown)}")
        missing_fields = allowed_fields - set(expected)
        if missing_fields:
            raise ValueError(
                f"model_contract.{role} is incomplete; missing fields "
                f"{sorted(missing_fields)}")
        config = role_configs[role]
        actual = {
            "num_layers": _num_layers(config),
            "num_heads": _num_heads(config),
            "hidden_size": int(getattr(config, "hidden_size", 0)),
        }
        observed[role] = actual
        for field, value in expected.items():
            checks[f"{role}.{field}"] = actual[field] == int(value)
    return {"checks": checks, "observed": observed, "passed": all(checks.values())}


def validate_layer_map(layer_map: Dict[int, int], teacher_config,
                       student_config) -> None:
    """Refuse out-of-range or duplicate layer mappings at startup."""
    teacher_layers = _num_layers(teacher_config)
    student_layers = _num_layers(student_config)
    if not layer_map:
        raise ValueError("loss.layer_map must contain at least one mapped layer")
    bad_teacher = sorted(i for i in layer_map if i < 0 or i >= teacher_layers)
    bad_student = sorted(i for i in layer_map.values() if i < 0 or i >= student_layers)
    if bad_teacher or bad_student:
        raise ValueError(
            f"loss.layer_map is outside teacher/student depths "
            f"({teacher_layers}/{student_layers}); bad teacher={bad_teacher}, "
            f"bad student={bad_student}")
    values = list(layer_map.values())
    if len(values) != len(set(values)):
        raise ValueError(
            "loss.layer_map maps more than one teacher layer to the same student layer; "
            "the per-layer mean would silently double-weight that student layer")


def _num_heads(config) -> int:
    return int(getattr(config, "num_heads", None)
               or getattr(config, "num_attention_heads", 0))


def _num_layers(config) -> int:
    return int(getattr(config, "num_layers", None)
               or getattr(config, "num_hidden_layers", 0))


# ═══════════════════════════════════════════════════════════════════════════════
# Determinism
# ═══════════════════════════════════════════════════════════════════════════════


def seed_everything(seed: int) -> None:
    """Seed every RNG that can affect initialisation or data order."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }


def load_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class BlockDataset:
    """Packed TinyStories blocks plus the manifest that proves two runs saw the same data."""

    #: Either a list of ``block_size``-long int lists, or — as ``build_block_dataset``
    #: produces — a ``(n_blocks, block_size)`` ``int32`` array. Same values either way;
    #: the array form is what makes the full TinyStories train split fit in host RAM.
    blocks: Any
    manifest: List[dict]
    block_size: int
    manifest_sha256: str = ""

    def __len__(self) -> int:
        return len(self.blocks)

    def batch(self, indices: Sequence[int], device) -> torch.Tensor:
        if isinstance(self.blocks, np.ndarray):
            # Fancy-index once and convert once: the list comprehension below would
            # rebuild a Python list of 128-int rows per micro-batch, which is the cost
            # this representation exists to avoid.
            rows = self.blocks[np.asarray(indices, dtype=np.int64)]
            return torch.from_numpy(np.ascontiguousarray(rows)).to(
                device=device, dtype=torch.long)
        return torch.tensor([self.blocks[i] for i in indices], dtype=torch.long,
                            device=device)


#: ``data.dataset`` -> the block loader that packs it. Both loaders take and return exactly
#: the same shapes, so the trainer is arm-agnostic below this line. Adding an entry here is
#: the only change a new corpus needs.
BLOCK_LOADERS = {
    "roneneldan/TinyStories": dl.load_tinystories_blocks,
    "Skylion007/openwebtext": dl.load_openwebtext_blocks,
}

#: Extra loader kwargs by dataset, for options not every loader accepts.
#: ``tokenizer_batch_size`` is offered on the OpenWebText loader only — the TinyStories
#: arm's packed records are already on disk and its packing path is deliberately frozen
#: (CLAUDE.md trap 19), so batching is available exactly where the E6A-GPT2 arm needs it.
BLOCK_LOADER_EXTRA_KWARGS: Dict[str, Dict[str, Any]] = {
    "Skylion007/openwebtext": {"tokenizer_batch_size": dl.DEFAULT_TOKENIZER_BATCH},
}

#: Where packed corpora are cached when ``--corpus-cache`` is left at its default. Packing
#: is deterministic in the key :mod:`block_corpus_cache` builds, so paying it once per
#: (config, seed) instead of once per run and per resume costs nothing scientific.
DEFAULT_CORPUS_CACHE = _REPO / ".corpus_cache"

#: What an *absent* ``data.dataset`` means. Every config written before the E6A-GPT2 arm
#: omits nothing — it names TinyStories — so this only covers the smoke path and old run
#: directories. A key that is *present but unrecognised* is refused, never folded into this
#: default (CLAUDE.md trap 13): a typo must not silently train the wrong arm's corpus.
DEFAULT_DATASET = "roneneldan/TinyStories"


def resolve_block_loader(dataset_name: Optional[str]):
    """``(loader, resolved_name)`` for ``dataset_name``, or a refusal naming the registry."""
    resolved = str(dataset_name) if dataset_name else DEFAULT_DATASET
    loader = BLOCK_LOADERS.get(resolved)
    if loader is None:
        raise ValueError(
            f"data.dataset {resolved!r} has no block loader. Known datasets: "
            f"{sorted(BLOCK_LOADERS)}. Add one to train_distillation.BLOCK_LOADERS rather "
            "than letting an unrecognised id fall back to a default corpus.")
    return loader, resolved


def build_block_dataset(tokenizer, split: str, block_size: int, seed: int,
                        n_blocks: Optional[int] = None,
                        eos_between: bool = True,
                        dataset_name: Optional[str] = None,
                        loader_kwargs: Optional[Dict[str, Any]] = None,
                        cache_root=None,
                        cache_info_out: Optional[Dict[str, Any]] = None) -> BlockDataset:
    """Pack a split into blocks via the configured loader and hash the manifest.

    ``dataset_name`` selects the loader through :func:`resolve_block_loader`; the default
    reproduces the TinyStories behaviour every existing config and run directory means.
    ``loader_kwargs`` passes corpus-specific options straight through (the OpenWebText
    loader's ``train_documents`` / ``validation_documents`` document windows).

    Memory, not speed, is the constraint here, and it is why this is not the obvious
    three-liner. The TinyStories *train* split packs to ~3.57M blocks; held as Python
    lists, duplicated into the manifest, and hashed through ``prov.sha256_json``, that
    peaked near 29 GB and could not start on a 16 GB host. Three changes, none of which
    moves a token or a digest:

    * ``as_array=True`` — blocks as ``int32``, ~1.8 GB rather than ~12.8 GB;
    * ``manifest_input_ids=False`` — drop the second copy of every block; the ids are
      still in ``blocks`` and still reach the ``05`` §4 parquet;
    * ``prov.sha256_int_rows`` — the same digest, streamed, without building a ~2.7 GB
      JSON string and a full list-of-lists copy to feed it.

    ``cache_root`` (``None`` = no caching, the original behaviour) routes the pack through
    :mod:`block_corpus_cache`, which stores the result under a key covering everything that
    can change it and **re-verifies the digest on every hit**. ``cache_info_out``, when
    given, receives that module's ``info`` dict so ``run_config.json`` can record whether
    the run packed its corpus or read it.
    """
    loader, resolved = resolve_block_loader(dataset_name)
    kwargs = dict(BLOCK_LOADER_EXTRA_KWARGS.get(resolved, {}))
    kwargs.update(loader_kwargs or {})
    blocks, manifest, digest, info = bcc.packed_blocks(
        loader, tokenizer, split, dataset=resolved, block_size=block_size, seed=seed,
        eos_between=eos_between, n_blocks=n_blocks, loader_kwargs=kwargs,
        cache_root=cache_root)
    if cache_info_out is not None:
        cache_info_out.update(info)
    return BlockDataset(blocks=blocks, manifest=manifest, block_size=block_size,
                        manifest_sha256=digest)


#: Rows per parquet row-group / CSV append when streaming the block manifest. Chosen so a
#: chunk's JSON-encoded ``input_ids`` stays well under a hundred MB at block_size 128.
MANIFEST_CHUNK_ROWS = 50_000


def _manifest_rows(dataset: BlockDataset, start: int, stop: int):
    """One chunk of ``05`` §4 rows, reading ``input_ids`` from ``blocks``.

    The manifest no longer carries its own copy of the tokens (that duplication cost ~12.8
    GB on the train split), so the ids come from the block array. The parquet keeps the
    column either way — the schema is unchanged, only the in-memory duplication is gone.
    """
    for row in dataset.manifest[start:stop]:
        ids = row.get("input_ids")
        if ids is None:
            ids = [int(t) for t in dataset.blocks[row["block_index"]]]
        yield {
            "block_index": row["block_index"],
            "input_ids": json.dumps(list(ids)),
            "source_story_indices": json.dumps(row["source_story_indices"]),
            "n_tokens": row["n_tokens"],
            "n_eos": row["n_eos"],
        }


def write_block_manifest(dataset: BlockDataset, path: Path) -> Path:
    """Write ``block_manifest.parquet`` (``05`` §4). Falls back to CSV if pyarrow is absent.

    The fallback is recorded in the filename rather than silently substituted, so a run
    directory always shows which format its manifest is in.

    Written in row-group chunks. A single ``pd.DataFrame`` over the train split's ~3.57M
    blocks — each with its ``input_ids`` re-encoded as a JSON string — is several GB before
    parquet ever sees it, on top of the corpus already resident. Chunking bounds that to
    ``MANIFEST_CHUNK_ROWS`` at a time; the file's contents and column order are unchanged.
    """
    import pandas as pd

    path.parent.mkdir(parents=True, exist_ok=True)
    total = len(dataset.manifest)

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        try:
            for start in range(0, total, MANIFEST_CHUNK_ROWS):
                chunk = pd.DataFrame(list(_manifest_rows(
                    dataset, start, min(start + MANIFEST_CHUNK_ROWS, total))))
                table = pa.Table.from_pandas(chunk, preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
            if writer is None:            # an empty manifest still gets a valid file
                pd.DataFrame(columns=["block_index", "input_ids",
                                      "source_story_indices", "n_tokens",
                                      "n_eos"]).to_parquet(path, index=False)
        finally:
            if writer is not None:
                writer.close()
        return path
    except Exception:  # pragma: no cover - only when pyarrow is unavailable
        fallback = path.with_suffix(".csv")
        first = True
        for start in range(0, total, MANIFEST_CHUNK_ROWS):
            chunk = pd.DataFrame(list(_manifest_rows(
                dataset, start, min(start + MANIFEST_CHUNK_ROWS, total))))
            chunk.to_csv(fallback, index=False, encoding="utf-8",
                         mode="w" if first else "a", header=first)
            first = False
        if first:
            pd.DataFrame(columns=["block_index", "input_ids", "source_story_indices",
                                  "n_tokens", "n_eos"]).to_csv(
                fallback, index=False, encoding="utf-8")
        return fallback


def epoch_order(n_blocks: int, seed: int, epoch: int) -> List[int]:
    """Deterministic block order for one pass, independent of global RNG state."""
    order = list(range(n_blocks))
    random.Random((seed + 1) * 100003 + epoch).shuffle(order)
    return order


@lru_cache(maxsize=2)
def epoch_order_array(n_blocks: int, seed: int, epoch: int) -> np.ndarray:
    """:func:`epoch_order` as a cached, read-only ``int64`` array — same integers, same order.

    This exists because ``epoch_order`` is a **pure function of its three arguments** that
    the training loop used to call once per gradient-accumulation micro-batch, then read
    sixteen elements of and discard. At E6A scale that is a Python-level Fisher-Yates
    shuffle over ~3.1M elements (OpenWebText) or ~3.57M (TinyStories) — **measured 1.09 s
    and 1.30 s** on this hardware — repeated four times per optimiser step and 40,000 times
    per 10,000-step run, entirely on one CPU core with the GPU idle. It was ~80% of the
    measured 4.85 s/step in the first pilot and is why the E6A-GPT2 runs showed 12% GPU
    utilisation on a 4090 and 3% on a 5090 (CLAUDE.md trap 27).

    ``epoch`` advances only every ``n_blocks / batch_size`` micro-batches — ~194,000 for the
    gpt2 arm — so within a 10,000-step run (40,000 micro-batches) it never advances at all
    and ``maxsize=2`` is enough to make the shuffle a once-per-run cost. Two is not one, so
    an epoch rollover mid-run still keeps the previous order rather than thrashing.

    The array is ``int64`` (25 MB at 3.1M blocks, against ~112 MB for the equivalent list of
    Python ints) and marked non-writeable, because an ``lru_cache`` handing out a shared
    mutable sequence is a defect waiting to happen: a caller that sorted it in place would
    silently change the data order of every subsequent step.

    ``epoch_order`` itself is deliberately left untouched — it is the reference this is
    checked against in ``tests/test_epoch_order_cache.py``.
    """
    order = np.asarray(epoch_order(n_blocks, seed, epoch), dtype=np.int64)
    order.flags.writeable = False
    return order


# ═══════════════════════════════════════════════════════════════════════════════
# Schedule
# ═══════════════════════════════════════════════════════════════════════════════


def lr_lambda(step: int, warmup_steps: int, max_steps: int,
              min_lr_ratio: float) -> float:
    """Linear warmup then cosine decay to ``min_lr_ratio`` of peak (§1.6)."""
    if warmup_steps > 0 and step < warmup_steps:
        return (step + 1) / warmup_steps
    if max_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_ratio + (1.0 - min_lr_ratio) * cosine


# ═══════════════════════════════════════════════════════════════════════════════
# Checkpointing
# ═══════════════════════════════════════════════════════════════════════════════


def save_checkpoint(run_dir: Path, step: int, model, optimizer, scheduler,
                    *, extra: Optional[Dict[str, Any]] = None) -> Path:
    """Save model + optimiser + scheduler + RNG so ``--resume auto`` is bit-exact (§1.6)."""
    path = run_dir / "checkpoints" / f"step_{step}"
    path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(path, safe_serialization=True)
    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(rng_state(), path / "rng_state.pt")
    if extra:
        prov.write_json(path / "trainer_state.json", extra)
    digest = prov.sha256_tree(path, patterns=("*.safetensors", "*.json"))
    (path / "checkpoint_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    return path


def tied_weight_keys(model) -> set:
    """Parameter names ``save_pretrained`` omits because they are tied to another tensor.

    GPT-Neo ties ``lm_head.weight`` to ``transformer.wte.weight``, so the saved
    ``model.safetensors`` legitimately has one fewer key than ``state_dict()``. Read from
    the model rather than hard-coded, because which keys are tied is an architecture and
    config property (``tie_word_embeddings``).
    """
    declared = getattr(model, "_tied_weights_keys", None) or []
    keys = set()
    for key in declared:
        keys.add(key)
        keys.add(key[:-len(".weight")] + ".weight" if key.endswith(".weight")
                 else key + ".weight")
    return keys


def assert_resumable_schedule(setup) -> None:
    """Refuse a resume that would silently change the learning-rate schedule.

    ``torch.optim.lr_scheduler.LambdaLR.state_dict()`` deliberately excludes the lambda, so
    ``load_checkpoint`` restores ``last_epoch`` but **not** the schedule it was produced
    under. The lambda here closes over ``setup.max_steps``, which means resuming a run under
    a different ``--max-steps`` silently splices two different cosines together: the LR
    jumps at the resume point and the realised trajectory is neither of the two schedules.

    This is reachable through the *documented* workflow, not an exotic one. The pilot runs
    ``--max-steps 2000`` and Phase 2 re-runs the same config, same run directory, with the
    default 10,000 — so without this guard the production run would be 2,000 steps of a
    2,000-step cosine followed by 8,000 steps of a 10,000-step cosine. Design §1.6
    pre-registers the schedule and says the hyperparameters are not to be improved; a
    spliced schedule is not the pre-registered one, and nothing in the artefacts would say
    so.

    Resolution is the operator's, not ours: continue in a **fresh run directory**, or pass
    ``--resume none`` to start over, or re-run at the recorded ``max_steps``.
    """
    config_path = setup.run_dir / "run_config.json"
    if not config_path.exists():
        return
    try:
        recorded = json.loads(config_path.read_text(encoding="utf-8")).get("max_steps")
    except (json.JSONDecodeError, OSError):
        return
    if recorded is None or int(recorded) == int(setup.max_steps):
        return
    raise SystemExit(
        f"Refusing to resume {setup.run_dir}: its checkpoints were produced with "
        f"max_steps={int(recorded)} and this invocation uses max_steps="
        f"{int(setup.max_steps)}.\n"
        "  LambdaLR does not serialise its lambda, so the restored scheduler would keep "
        "the old step count while following the NEW cosine — the learning rate jumps at "
        "the resume point and the run follows neither pre-registered schedule (design "
        "§1.6).\n"
        "  Fix: train the longer run in a FRESH run directory, or pass --resume none to "
        f"start over, or re-run with --max-steps {int(recorded)}."
    )


def load_checkpoint(path: Path, model, optimizer, scheduler) -> Dict[str, Any]:
    """Restore weights, optimiser, scheduler and RNG from a checkpoint directory.

    Loads non-strictly and then *verifies* that the only absent keys are tied ones — a
    strict load would fail on every tied-embedding architecture, and blanket
    ``strict=False`` would let a genuinely truncated checkpoint resume silently with
    randomly-initialised layers.
    """
    from safetensors.torch import load_file

    state = load_file(str(path / "model.safetensors"))
    missing, unexpected = model.load_state_dict(state, strict=False)
    tied = tied_weight_keys(model)
    unaccounted = [k for k in missing if k not in tied]
    if unaccounted or unexpected:
        raise RuntimeError(
            f"Checkpoint {path} does not match the model: missing {unaccounted}, "
            f"unexpected {list(unexpected)}. Resuming would train a partially "
            "randomly-initialised model.")
    if missing:
        # Re-establish the sharing that save_pretrained relied on when it dropped them.
        model.tie_weights()
    optimizer.load_state_dict(torch.load(path / "optimizer.pt", weights_only=False))
    scheduler.load_state_dict(torch.load(path / "scheduler.pt", weights_only=False))
    load_rng_state(torch.load(path / "rng_state.pt", weights_only=False))
    trainer_state = path / "trainer_state.json"
    return (json.loads(trainer_state.read_text(encoding="utf-8"))
            if trainer_state.exists() else {})


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    root = run_dir / "checkpoints"
    if not root.exists():
        return None
    steps = []
    for child in root.iterdir():
        if child.is_dir() and child.name.startswith("step_"):
            try:
                steps.append((int(child.name.split("_")[1]), child))
            except ValueError:
                continue
    if not steps:
        return None
    return max(steps)[1]


# ═══════════════════════════════════════════════════════════════════════════════
# Model construction
# ═══════════════════════════════════════════════════════════════════════════════


def build_student(student_config, seed: int):
    """Randomly initialise the student from a config — **never** ``from_pretrained``.

    Seeding immediately before construction is what makes three conditions at one seed
    byte-identical at step 0; the public 8M checkpoint is an independent-convergence
    reference and is never a starting point.
    """
    from transformers import AutoModelForCausalLM

    seed_everything(seed)
    return AutoModelForCausalLM.from_config(student_config)


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke assets
# ═══════════════════════════════════════════════════════════════════════════════


def build_smoke_assets(root: Path, *, vocab_size: int = 64, block_size: int = 32,
                       arch: str = "neo", teacher_layers: Optional[int] = None,
                       student_layers: Optional[int] = None,
                       teacher_heads: Optional[int] = None,
                       student_heads: Optional[int] = None,
                       teacher_hidden_size: Optional[int] = None,
                       student_hidden_size: Optional[int] = None):
    """Tiny random teacher/student + a word-level tokenizer, entirely offline.

    Re-expressed here rather than imported from ``tests/`` — production code must not
    depend on the test package — but deliberately the same recipe as
    ``tests/nnsight_smoke_utils``, so a smoke run and a smoke test exercise the same shapes.

    ``arch`` selects the family, defaulting to ``"neo"`` so every existing caller gets
    byte-identical assets. The default depths mirror each arm's real pair, because the
    depth relationship is what makes the trap-1 band question reachable in smoke mode:

    * ``neo``  — 4-layer teacher, 8-layer student (the 33M/8M TinyStories pair): the
      student is *deeper* than its teacher, and GPT-Neo alternates global/local attention
      so mapped pairs can be ``mixed``;
    * ``gpt2`` — 12-layer teacher, 6-layer student by default (the gpt2/distilgpt2 pair),
      with optional depths supplied by a config; every layer is global.
    """
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPTNeoConfig, PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    for i in range(vocab_size - 3):
        vocab[f"t{i}"] = i + 3
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]", bos_token="[EOS]")
    tokenizer.name_or_path = "smoke_tokenizer"
    tokenizer.save_pretrained(str(root / "tokenizer"))

    def _neo(n_layer, n_head=4, hidden_size=32):
        return GPTNeoConfig(
            vocab_size=len(tokenizer), hidden_size=hidden_size, num_layers=n_layer,
            num_heads=n_head,
            max_position_embeddings=block_size, window_size=block_size,
            attention_types=[[["global", "local"], n_layer // 2]],
            intermediate_size=2 * hidden_size, bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            resid_dropout=0.0, embed_dropout=0.0, attention_dropout=0.0,
            classifier_dropout=0.0)

    def _gpt2(n_layer, n_head=4, hidden_size=32):
        return GPT2Config(
            vocab_size=len(tokenizer), n_embd=hidden_size, n_layer=n_layer, n_head=n_head,
            n_positions=block_size, n_inner=2 * hidden_size,
            bos_token_id=tokenizer.bos_token_id, eos_token_id=tokenizer.eos_token_id,
            resid_pdrop=0.0, embd_pdrop=0.0, attn_pdrop=0.0, summary_first_dropout=0.0)

    builders = {"neo": (_neo, 4, 8), "gpt2": (_gpt2, 12, 6)}
    if arch not in builders:
        raise ValueError(f"no smoke assets for arch {arch!r}; known: {sorted(builders)}")
    build, default_teacher, default_student = builders[arch]
    resolved_teacher_heads = int(teacher_heads or 4)
    resolved_student_heads = int(student_heads or 4)
    resolved_teacher_hidden = int(teacher_hidden_size or 32)
    resolved_student_hidden = int(student_hidden_size or 32)
    if resolved_teacher_hidden % resolved_teacher_heads:
        raise ValueError(
            f"smoke teacher hidden size {resolved_teacher_hidden} is not divisible by "
            f"{resolved_teacher_heads} heads")
    if resolved_student_hidden % resolved_student_heads:
        raise ValueError(
            f"smoke student hidden size {resolved_student_hidden} is not divisible by "
            f"{resolved_student_heads} heads")
    return (tokenizer,
            build(teacher_layers or default_teacher, resolved_teacher_heads,
                  resolved_teacher_hidden),
            build(student_layers or default_student, resolved_student_heads,
                  resolved_student_hidden))


def smoke_blocks(tokenizer, n_blocks: int, block_size: int, seed: int) -> BlockDataset:
    """Deterministic random token blocks, so ``--smoke`` never touches the network."""
    rng = random.Random(seed)
    lo, hi = 3, len(tokenizer) - 1
    blocks, manifest = [], []
    for index in range(n_blocks):
        ids = [rng.randint(lo, hi) for _ in range(block_size)]
        blocks.append(ids)
        manifest.append({"block_index": index, "input_ids": ids,
                         "source_story_indices": [index], "n_tokens": block_size,
                         "n_eos": 0})
    return BlockDataset(blocks=blocks, manifest=manifest, block_size=block_size,
                        manifest_sha256=prov.sha256_json(blocks))


# ═══════════════════════════════════════════════════════════════════════════════
# The trainer
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class TrainerSetup:
    """Everything a training run needs, assembled by :func:`prepare_run`."""

    config: Dict[str, Any]
    condition: str
    seed: int
    run_dir: Path
    device: torch.device
    dtype: torch.dtype
    tokenizer: Any
    teacher: Any
    student: Any
    teacher_config: Any
    student_config: Any
    train_data: BlockDataset
    eval_data: BlockDataset
    layer_map: Dict[int, int]
    head_alignment: Dict[str, Any]
    weights: Dict[str, float]
    temperature: float
    optim: Dict[str, Any]
    checkpoints: Tuple[int, ...]
    max_steps: int
    initial_state_sha256: str
    vocab_report: Dict[str, Any]
    model_contract_report: Dict[str, Any]
    smoke: bool = False
    provenance: Dict[str, Any] = field(default_factory=dict)
    #: ``(seq_len, device) -> (valid_masks, pair_types)``. :func:`build_valid_masks` is a
    #: deterministic function of the two configs, the layer map and the sequence length, all
    #: of which are fixed for a run — but it was being rebuilt on every micro-batch, once
    #: per mapped pair. Cached here rather than in a module-level ``lru_cache`` because the
    #: configs are unhashable and keying on ``id()`` would survive their garbage collection.
    mask_cache: Dict[Tuple[int, str], Tuple[Dict[Tuple[int, int], torch.Tensor],
                                            Dict[Tuple[int, int], str]]] = field(
        default_factory=dict)


def prepare_run(config: Dict[str, Any], *, seed: int, output_dir: Path,
                max_steps: Optional[int] = None, smoke: bool = False,
                device: Optional[str] = None, corpus_cache=None) -> TrainerSetup:
    """Load or synthesise every artefact, assert the startup invariants, and abort on failure."""
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    condition = str(config.get("condition", "D0"))
    loss_cfg = config.get("loss", {})
    head_alignment = resolve_head_alignment(loss_cfg)
    optim_cfg = dict(config.get("optim", {}))
    data_cfg = config.get("data", {})
    block_size = int(data_cfg.get("block_size", 128))
    layer_map = {int(k): int(v) for k, v in
                 (loss_cfg.get("layer_map") or DEFAULT_LAYER_MAP).items()}

    if device is not None:
        dev = torch.device(device)
    elif smoke:
        dev = torch.device("cpu")
    else:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # bf16 on CUDA, fp32 on CPU (§1.6). A CPU bf16 run would be both slow and
    # numerically different from the pre-registered setup, so it is not offered.
    dtype = (torch.bfloat16 if (dev.type == "cuda"
                                and str(optim_cfg.get("precision", "bfloat16"))
                                == "bfloat16")
             else torch.float32)

    # The arm's own id, so `e6a_gpt2` writes to results/e6a_gpt2/ and can never be rglob'd
    # into the TinyStories arm's aggregation. Defaults to `e6a` for every existing config.
    experiment_id = str(config.get("experiment_id", EXPERIMENT_ID))
    run_dir = Path(output_dir) / experiment_id / condition / f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # Filled by build_block_dataset and recorded in run_config.json. A smoke run
    # synthesises its blocks and never touches the cache, so both halves stay empty.
    corpus_cache_info: Dict[str, Dict[str, Any]] = {"train": {}, "eval": {}}

    if smoke:
        # The smoke family follows the config's own teacher, so the gpt2 arm's smoke run
        # exercises GPT-2 shapes and a 12->6 layer map rather than GPT-Neo's 4->8.
        teacher_leaf = str(config.get("teacher", "")).rstrip("/\\").split("/")[-1]
        smoke_arch = "gpt2" if teacher_leaf.lower().startswith("gpt2") else "neo"
        contract = config.get("model_contract") or {}
        teacher_contract = contract.get("teacher") or {}
        student_contract = contract.get("student") or {}
        # Scale-arm smokes preserve the registered depths and head counts exactly while
        # shrinking each head to four dimensions. The real run separately asserts the
        # canonical widths before loading data.
        smoke_teacher_heads = teacher_contract.get("num_heads")
        smoke_student_heads = student_contract.get("num_heads")
        smoke_teacher_hidden = (4 * int(smoke_teacher_heads)
                                if smoke_teacher_heads is not None else None)
        smoke_student_hidden = (4 * int(smoke_student_heads)
                                if smoke_student_heads is not None else None)
        tokenizer, teacher_config, student_config = build_smoke_assets(
            run_dir / "_smoke", block_size=block_size, arch=smoke_arch,
            teacher_layers=(teacher_contract.get("num_layers")
                            or max(layer_map) + 1),
            student_layers=(student_contract.get("num_layers")
                            or max(layer_map.values()) + 1),
            teacher_heads=smoke_teacher_heads, student_heads=smoke_student_heads,
            teacher_hidden_size=smoke_teacher_hidden,
            student_hidden_size=smoke_student_hidden)
        public_config = student_config
        # A DIFFERENT seed for the smoke teacher, deliberately. Seeding both from `seed`
        # draws their weights from the same RNG stream, which for matching layer shapes
        # makes teacher and student near-identical — L_KD and L_ATTN then collapse to
        # ~1e-5 and a smoke run could not reveal a broken distillation term.
        seed_everything(seed + 977)
        teacher = AutoModelForCausalLM.from_config(teacher_config)
        # Persist the synthetic teacher and point `teacher` at it. A smoke run's teacher
        # is created in-process and would otherwise be unrecoverable, so
        # evaluate_transformation.py would fall back to the config's HF id — unreachable
        # offline — and silently produce student-only rows. Saving it means the smoke path
        # exercises the whole teacher-comparison branch (mutual_interventions, the band
        # guard, every *_to_teacher column) instead of skipping it.
        teacher_path = run_dir / "_smoke" / "teacher"
        teacher.save_pretrained(teacher_path, safe_serialization=True)
        config = {**config, "teacher": str(teacher_path), "teacher_revision": None}
        train_data = smoke_blocks(tokenizer, 64, block_size, seed)
        eval_data = smoke_blocks(tokenizer, 8, block_size, seed + 1000)
    else:
        tokenizer = AutoTokenizer.from_pretrained(
            config["tokenizer"], revision=config.get("tokenizer_revision"))
        teacher_config = AutoConfig.from_pretrained(
            config["teacher"], revision=config.get("teacher_revision"))
        public_config = AutoConfig.from_pretrained(
            config["public_reference"], revision=config.get("public_reference_revision"))
        student_config = AutoConfig.from_pretrained(
            config["student_config_from"], revision=config.get("student_config_revision"))
        teacher = AutoModelForCausalLM.from_pretrained(
            config["teacher"], revision=config.get("teacher_revision"),
            dtype=dtype, attn_implementation="eager")
        # The corpus follows `data.dataset`; an unrecognised id raises here rather than
        # training the wrong arm's data under the right arm's name.
        dataset_name = data_cfg.get("dataset")
        loader_kwargs = {
            key: int(data_cfg[key]) for key in ("train_documents", "validation_documents")
            if data_cfg.get(key) is not None}
        train_data = build_block_dataset(
            tokenizer, data_cfg.get("train_split", "train"), block_size, seed,
            eos_between=bool(data_cfg.get("eos_between", True)),
            dataset_name=dataset_name, loader_kwargs=loader_kwargs,
            cache_root=corpus_cache, cache_info_out=corpus_cache_info["train"])
        eval_blocks = int(config.get("eval", {}).get("sink_corpus", {})
                          .get("n_blocks", 300))
        eval_data = build_block_dataset(
            tokenizer, data_cfg.get("eval_split", "validation"), block_size, seed,
            n_blocks=eval_blocks,
            eos_between=bool(data_cfg.get("eos_between", True)),
            dataset_name=dataset_name, loader_kwargs=loader_kwargs,
            cache_root=corpus_cache, cache_info_out=corpus_cache_info["eval"])

    # Eager attention is required to read attention maps for L_ATTN.
    student_config._attn_implementation = "eager"
    teacher_config._attn_implementation = "eager"

    validate_layer_map(layer_map, teacher_config, student_config)
    report = vocab_assertions(
        tokenizer, teacher_config, public_config, student_config, head_alignment)
    if not report["passed"]:
        failed = [k for k, v in report["checks"].items() if not v]
        raise SystemExit(
            "Startup assertions failed (03 §1.2 / design-delta D6): "
            f"{failed}. Observed: {json.dumps(report['observed'], default=str)}. "
            "A vocabulary mismatch invalidates KD, and head geometry unsupported by the "
            "declared alignment invalidates L_ATTN; aborting rather than training on it.")

    if smoke:
        model_report = {
            "checks": {},
            "observed": {
                "teacher": {"num_layers": _num_layers(teacher_config),
                            "num_heads": _num_heads(teacher_config),
                            "hidden_size": int(getattr(teacher_config, "hidden_size", 0))},
                "student": {"num_layers": _num_layers(student_config),
                            "num_heads": _num_heads(student_config),
                            "hidden_size": int(getattr(student_config, "hidden_size", 0))},
            },
            "passed": True,
            "skipped": "smoke uses scaled widths while preserving registered depths/heads",
        }
    else:
        model_report = model_contract_assertions(
            teacher_config, public_config, student_config,
            config.get("model_contract"))
        if not model_report["passed"]:
            failed = [k for k, v in model_report["checks"].items() if not v]
            raise SystemExit(
                "Model contract assertions failed: "
                f"{failed}. Observed: {json.dumps(model_report['observed'])}. "
                "The pinned model geometry differs from the prospectively registered arm.")

    student = build_student(student_config, seed)
    initial_sha = prov.sha256_state_dict(student.state_dict())

    teacher.to(device=dev, dtype=dtype).eval()
    teacher.requires_grad_(False)
    student.to(device=dev, dtype=torch.float32)

    weights = {
        "ce": float(loss_cfg.get("ce_weight", 1.0)),
        "kd": float(loss_cfg.get("kd_weight", 0.0)),
        "attn": float(loss_cfg.get("attn_weight", 0.0)),
    }
    checkpoints = tuple(int(s) for s in config.get("checkpoints", DEFAULT_CHECKPOINTS))
    resolved_max_steps = int(max_steps if max_steps is not None
                             else optim_cfg.get("max_steps", 10000))

    if smoke:
        optim_cfg.setdefault("per_device_batch_size", 2)
        optim_cfg["per_device_batch_size"] = 2
        optim_cfg["grad_accum"] = 1
        optim_cfg["warmup_steps"] = 1

    return TrainerSetup(
        config=config, condition=condition, seed=seed, run_dir=run_dir, device=dev,
        dtype=dtype, tokenizer=tokenizer, teacher=teacher, student=student,
        teacher_config=teacher_config, student_config=student_config,
        train_data=train_data, eval_data=eval_data, layer_map=layer_map,
        head_alignment=head_alignment,
        weights=weights, temperature=float(loss_cfg.get("temperature", 2.0)),
        optim=optim_cfg, checkpoints=checkpoints, max_steps=resolved_max_steps,
        initial_state_sha256=initial_sha, vocab_report=report,
        model_contract_report=model_report, smoke=smoke,
        provenance={"corpus_cache": corpus_cache_info})


def write_run_config(setup: TrainerSetup, block_manifest_path: Path) -> Path:
    """``run_config.json`` per ``05`` §1, with the fields the gap analysis added."""
    student_layers = _num_layers(setup.student_config)
    band_start, band_end, band_meta = normalised_depth_band(student_layers)
    teacher_layers = _num_layers(setup.teacher_config)
    t_start, t_end, t_meta = normalised_depth_band(teacher_layers)

    experiment_id = str(setup.config.get("experiment_id", EXPERIMENT_ID))
    # From the student's own class, not from a literal: the gpt2 arm's student is a
    # GPT2LMHeadModel and `evaluate_transformation` resolves the fingerprint runner's arch
    # from this field. Hard-coding "neo" would send a GPT-2 student at the Neo harness.
    arch_key = ARCH_BY_CLASS.get(type(setup.student).__name__, "neo")

    payload = {
        "experiment_id": experiment_id,
        "experiment_family": setup.config.get("experiment_family", "e6a"),
        "preregistration": setup.config.get("preregistration"),
        "condition_id": setup.condition,
        "run_id": f"{experiment_id}_{setup.condition}_seed{setup.seed}",
        "model_name": str(setup.run_dir),
        "model_revision": None,
        "architecture": type(setup.student).__name__,
        "arch_key": arch_key,
        "parameter_count": int(sum(p.numel() for p in setup.student.parameters())),
        "num_layers": student_layers,
        "num_heads": _num_heads(setup.student_config),
        "hidden_size": int(getattr(setup.student_config, "hidden_size", 0)),
        "training_seed": setup.seed,
        "data_seed": setup.seed,
        "evaluation_seed": 42,
        "tokenizer_name": getattr(setup.tokenizer, "name_or_path", "unknown"),
        "tokenizer_revision": setup.config.get("tokenizer_revision"),
        "vocab_assertions": setup.vocab_report["checks"],
        "vocab_observed": setup.vocab_report["observed"],
        "model_contract": setup.config.get("model_contract"),
        "model_contract_assertions": setup.model_contract_report,
        "dtype": str(setup.dtype).replace("torch.", ""),
        "device": str(setup.device),
        "engine": "train",
        "attn_implementation": "eager",
        "dataset_name": setup.config.get("data", {}).get("dataset", "smoke"),
        "dataset_revision": None,
        # WP10 additive: evaluate_transformation.py must compare the student against the
        # teacher it was distilled from, and nothing in the run directory recorded which
        # model that was. Added rather than renamed (05 §5); `null` for a smoke run, which
        # synthesises its teacher and has no HF id to record.
        "teacher": setup.config.get("teacher"),
        "teacher_revision": setup.config.get("teacher_revision"),
        "public_reference": setup.config.get("public_reference"),
        "public_reference_revision": setup.config.get("public_reference_revision"),
        "student_config_from": setup.config.get("student_config_from"),
        "student_config_revision": setup.config.get("student_config_revision"),
        "config_path": setup.config.get("_config_path"),
        "corpus_id": setup.config.get("data", {}).get(
            "corpus_id",
            f"tinystories_{setup.config.get('data', {}).get('train_split', 'train')}"),
        "manifest_sha256": setup.train_data.manifest_sha256,
        "block_manifest_sha256": prov.sha256_file(block_manifest_path),
        "block_manifest_path": block_manifest_path.name,
        # Whether each corpus was packed by this run or read from the shared cache, and
        # under which key. Additive (`05` §5). A cache whose use leaves no trace in the
        # artefacts is not auditable, and `manifest_sha256` above is re-derived from the
        # blocks on every hit, so the two can never disagree silently.
        "corpus_cache": setup.provenance.get("corpus_cache", {"train": {}, "eval": {}}),
        "initial_state_sha256": setup.initial_state_sha256,
        "checkpoint_step": None,
        "checkpoint_sha256": None,
        "layer_band": [band_start, band_end],
        "layer_band_depth_interval": list(band_meta["depth_interval"]),
        "layer_band_version": band_meta["depth_band_version"],
        "teacher_layer_band": [t_start, t_end],
        "teacher_layer_band_depth_interval": list(t_meta["depth_interval"]),
        "intervention_registry_version": None,
        "available_interventions": None,
        "patch_registry_version": None,
        "loss_weights": setup.weights,
        "temperature": setup.temperature,
        # Which reduction L_ATTN used. A run directory without this key predates the trap-22
        # fix, so its D2 arm trained on a 64.5x under-scaled attention term; the two are not
        # comparable and must not be pooled.
        "attn_reduction": ATTN_REDUCTION,
        # Keep the scalar field introduced by the large->medium arm while also recording
        # the complete method specification needed by AMAD-style alignment.
        "attention_head_alignment": (
            "strict" if setup.head_alignment["method"] == LEGACY_HEAD_ALIGNMENT
            else setup.head_alignment["method"]),
        "head_alignment": setup.head_alignment,
        "layer_map": {str(k): v for k, v in setup.layer_map.items()},
        "optim": setup.optim,
        "checkpoints": list(setup.checkpoints),
        "max_steps": setup.max_steps,
        "smoke": setup.smoke,
    }
    payload.update(prov.provenance_block())
    return prov.write_json(setup.run_dir / "run_config.json", payload)


def _forward_with_attention(model, input_ids, *, need_attention: bool):
    kwargs = {"input_ids": input_ids, "attention_mask": torch.ones_like(input_ids)}
    if need_attention:
        kwargs["output_attentions"] = True
    return model(**kwargs)


def cached_valid_masks(setup: TrainerSetup, seq_len: int, device
                       ) -> Tuple[Dict[Tuple[int, int], torch.Tensor],
                                  Dict[Tuple[int, int], str]]:
    """:func:`build_valid_masks` memoised on ``setup`` for one ``(seq_len, device)``.

    The masks are a deterministic function of the two configs, the layer map and the
    sequence length — none of which changes within a run — so rebuilding them on every
    micro-batch allocated the same tensors tens of thousands of times. Same tensors, built
    once.
    """
    key = (int(seq_len), str(device))
    cached = setup.mask_cache.get(key)
    if cached is None:
        cached = build_valid_masks(setup.teacher_config, setup.student_config,
                                   setup.layer_map, seq_len, device)
        setup.mask_cache[key] = cached
    return cached


def compute_losses(setup: TrainerSetup, input_ids: torch.Tensor, *,
                   need_attention: bool, want_breakdown: bool = True,
                   tensors_out: Optional[Dict[str, torch.Tensor]] = None
                   ) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """One forward pass of student and teacher, returning the weighted loss and all components.

    Every component is computed regardless of its weight: ``L_ATTN`` is an evaluation
    metric for D0 and D1 (design §8.6), and ``L_KD`` is logged for D0. Only the *weighted*
    sum enters the backward pass.

    Three throughput properties, none of which moves a number:

    * **The four component scalars are read back in one synchronisation**, not four. Each
      ``.item()`` drains the CUDA queue, so with the per-pair reads in
      :func:`attn_js_loss` this function used to stall the pipeline twenty-two times per
      micro-batch — eighty-eight times per optimiser step at ``grad_accum=4``. Widening a
      float32 scalar to a Python float is identical whether it goes through ``.item()`` or a
      stacked ``.tolist()``.
    * ``want_breakdown=False`` skips the per-pair report (see :func:`attn_js_loss`). The
      caller passes it wherever the breakdown is discarded: every micro-batch but the last
      (``_average_components`` keeps ``rows[-1]``), and all of :func:`evaluate`.
    * **``L_ATTN`` is computed under ``no_grad`` when its weight is zero.** For D0/D1/G0/G1
      it is a logged metric only (design §8.6) and never reaches ``backward``, but it was
      still being recorded into the autograd graph — measured **11%** of step time on a real
      gpt2→distilgpt2 pair. ``nullcontext``, never ``enable_grad``: :func:`evaluate` runs
      under an ambient ``no_grad`` and must keep it.

    ``tensors_out``, when given, receives the student and teacher logits this function has
    already computed, so :func:`evaluate` need not run both models a second time to get
    them. Default ``None`` leaves behaviour and the ``components`` dict untouched.
    """
    student_out = _forward_with_attention(setup.student, input_ids,
                                          need_attention=need_attention)
    student_logits = student_out.logits.float()
    l_ce = ce_loss(student_logits, input_ids)

    with torch.no_grad():
        teacher_out = _forward_with_attention(setup.teacher, input_ids,
                                              need_attention=need_attention)
    teacher_logits = teacher_out.logits.float()

    # KD is measured on the same shifted next-token positions as CE.
    l_kd = kd_loss(student_logits[:, :-1, :], teacher_logits[:, :-1, :],
                   T=setup.temperature)

    l_attn = torch.zeros((), device=student_logits.device)
    breakdown: Dict[str, Any] = {"per_pair": [], "n_pairs": 0}
    if need_attention:
        seq_len = int(input_ids.shape[-1])
        masks, pair_types = cached_valid_masks(setup, seq_len, student_logits.device)
        # `nullcontext`, never `enable_grad`: evaluate() calls this under an ambient
        # no_grad and re-enabling the graph there would be a behaviour change, not an
        # optimisation.
        attn_ctx = (contextlib.nullcontext() if setup.weights["attn"] > 0
                    else torch.no_grad())
        with attn_ctx:
            student_attn = student_out.attentions
            teacher_attn = teacher_out.attentions
            l_attn, breakdown = attention_distillation_loss(
                student_attn, teacher_attn, setup.layer_map, masks, pair_types,
                setup.head_alignment, collect_breakdown=want_breakdown)

    total = setup.weights["ce"] * l_ce + setup.weights["kd"] * l_kd
    if setup.weights["attn"] > 0:
        total = total + setup.weights["attn"] * l_attn

    # The single synchronisation. All four are 0-dim float32, so one stacked `.tolist()`
    # yields exactly the Python floats four separate `.item()` calls did.
    l_ce_v, l_kd_v, l_attn_v, loss_v = torch.stack(
        [l_ce.detach(), l_kd.detach(), l_attn.detach(), total.detach()]).tolist()

    components: Dict[str, Any] = {
        "l_ce": float(l_ce_v),
        "l_kd": float(l_kd_v),
        # `nan`, not the 0.0 the placeholder tensor holds: L_ATTN was not measured on this
        # pass, and `_average_components` uses `nanmean` precisely so an unmeasured
        # micro-batch cannot be averaged in as a zero.
        "l_attn": float(l_attn_v) if need_attention else float("nan"),
        "attn_breakdown": breakdown,
        "loss": float(loss_v),
    }
    if tensors_out is not None:
        tensors_out["student_logits"] = student_logits
        tensors_out["teacher_logits"] = teacher_logits
    return total, components


@torch.no_grad()
def evaluate(setup: TrainerSetup, max_batches: Optional[int] = None) -> Dict[str, Any]:
    """Validation CE, teacher KL, top-1/top-5 agreement and ``L_ATTN`` (§1.8 eval_log).

    The top-1/top-5 agreement is computed from the logits :func:`compute_losses` has
    **already produced**, rather than from a second pair of forward passes. This halves the
    forwards per evaluation batch and is exact rather than approximate: both models are
    constructed with ``attn_implementation="eager"``, so ``output_attentions`` only decides
    whether the attention probabilities are *returned* — it never changes the computation.
    Verified numerically on this project's own pair rather than assumed: real gpt2 in bf16
    and a randomly-initialised distilgpt2 in fp32 both give ``torch.equal(logits_with,
    logits_without) is True``, max absolute difference exactly ``0.0``.
    """
    setup.student.eval()
    batch_size = int(setup.optim.get("per_device_batch_size", 16))
    n = len(setup.eval_data)
    totals = {"ce": 0.0, "kd": 0.0, "attn": 0.0, "top1": 0.0, "top5": 0.0}
    n_batches = 0
    n_tokens = 0

    for start in range(0, n, batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        indices = list(range(start, min(start + batch_size, n)))
        input_ids = setup.eval_data.batch(indices, setup.device)
        logits: Dict[str, torch.Tensor] = {}
        # The breakdown is discarded here, so it is not collected — see attn_js_loss.
        _loss, components = compute_losses(setup, input_ids, need_attention=True,
                                           want_breakdown=False, tensors_out=logits)

        s_pred = logits["student_logits"][:, :-1, :]
        t_pred = logits["teacher_logits"][:, :-1, :]
        top5 = t_pred.topk(5, dim=-1).indices
        s_top1 = s_pred.argmax(dim=-1)
        top1_hit = (s_top1 == top5[..., 0]).float().mean()
        top5_hit = (s_top1.unsqueeze(-1) == top5).any(-1).float().mean()
        top1_v, top5_v = torch.stack([top1_hit, top5_hit]).tolist()
        totals["top1"] += float(top1_v)
        totals["top5"] += float(top5_v)

        totals["ce"] += components["l_ce"]
        totals["kd"] += components["l_kd"]
        if not math.isnan(components["l_attn"]):
            totals["attn"] += components["l_attn"]
        n_batches += 1
        n_tokens += int(input_ids.numel())

    setup.student.train()
    if n_batches == 0:
        return {"n_batches": 0}
    return {
        "validation_ce": totals["ce"] / n_batches,
        "validation_ppl": float(math.exp(min(totals["ce"] / n_batches, 20.0))),
        "teacher_kl": totals["kd"] / n_batches,
        "l_attn": totals["attn"] / n_batches,
        "top1_agreement": totals["top1"] / n_batches,
        "top5_agreement": totals["top5"] / n_batches,
        "n_batches": n_batches,
        "n_tokens": n_tokens,
    }


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def write_runtime_estimate(setup: TrainerSetup, measured_units: int,
                           elapsed_s: float) -> Path:
    """``runtime_estimate.json`` per ``05`` §5 — no unverified runtime enters the paper."""
    rate = measured_units / elapsed_s if elapsed_s > 0 else 0.0
    projected = setup.max_steps / rate if rate > 0 else float("inf")
    from datetime import datetime, timedelta, timezone
    finish = (datetime.now(timezone.utc) + timedelta(seconds=min(projected, 3e8))
              ).replace(microsecond=0).isoformat() if rate > 0 else ""
    payload = {
        "unit": "optimizer_step",
        "measured_units": measured_units,
        "elapsed_s": elapsed_s,
        "units_per_second": rate,
        "projected_total_units": setup.max_steps,
        "projected_total_s": projected,
        "projected_finish_utc": finish,
        "device": (torch.cuda.get_device_name(0) if setup.device.type == "cuda"
                   else "cpu"),
        "peak_vram_bytes": (int(torch.cuda.max_memory_allocated())
                            if setup.device.type == "cuda" else 0),
        "dtype": str(setup.dtype).replace("torch.", ""),
        "measured_utc": prov.utc_now(),
    }
    return prov.write_json(setup.run_dir / "runtime_estimate.json", payload)


def train(setup: TrainerSetup, *, resume: str = "auto",
          stop_after: Optional[int] = None) -> Dict[str, Any]:
    """Run the training loop. Returns a summary; all detail goes to the run directory."""
    optim_cfg = setup.optim
    batch_size = int(optim_cfg.get("per_device_batch_size", 16))
    grad_accum = int(optim_cfg.get("grad_accum", 1))
    warmup = int(optim_cfg.get("warmup_steps", 500))
    min_lr_ratio = float(optim_cfg.get("min_lr_ratio", 0.10))
    grad_clip = float(optim_cfg.get("grad_clip", 1.0))

    optimizer = torch.optim.AdamW(
        setup.student.parameters(),
        lr=float(optim_cfg.get("lr", 5e-4)),
        betas=tuple(optim_cfg.get("betas", (0.9, 0.95))),
        eps=float(optim_cfg.get("eps", 1e-8)),
        weight_decay=float(optim_cfg.get("weight_decay", 0.1)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda step: lr_lambda(step, warmup, setup.max_steps, min_lr_ratio))

    # BEFORE write_run_config, which is about to overwrite the very field this reads.
    if resume == "auto" and latest_checkpoint(setup.run_dir) is not None:
        assert_resumable_schedule(setup)

    block_manifest_path = write_block_manifest(
        setup.train_data, setup.run_dir / "block_manifest.parquet")
    write_run_config(setup, block_manifest_path)

    train_log = setup.run_dir / "train_log.jsonl"
    eval_log = setup.run_dir / "eval_log.jsonl"
    need_attention = setup.weights["attn"] > 0 or True   # always: L_ATTN is a metric

    start_step = 0
    if resume == "auto":
        checkpoint = latest_checkpoint(setup.run_dir)
        if checkpoint is not None:
            state = load_checkpoint(checkpoint, setup.student, optimizer, scheduler)
            start_step = int(state.get("step", int(checkpoint.name.split("_")[1])))
            print(f"  resumed from {checkpoint} at step {start_step}")

    if start_step == 0:
        # Step 0 is saved BEFORE the first optimiser update — an explicit call, not a
        # branch inside the loop, which on resume would run post-update (§1.6).
        save_checkpoint(setup.run_dir, 0, setup.student, optimizer, scheduler,
                        extra={"step": 0, "pre_first_update": True,
                               "initial_state_sha256": setup.initial_state_sha256})
        metrics = evaluate(setup, max_batches=2 if setup.smoke else None)
        append_jsonl(eval_log, {"step": 0, **metrics})

    setup.student.train()
    n_blocks = len(setup.train_data)
    micro_index = start_step * grad_accum
    started = time.time()
    runtime_written = start_step >= 100
    last_components: Dict[str, Any] = {}

    end_step = min(setup.max_steps, int(stop_after)) if stop_after is not None \
        else setup.max_steps
    if end_step < start_step:
        raise ValueError(
            f"stop_after={end_step} is behind resumed checkpoint step {start_step}")

    for step in range(start_step, end_step):
        optimizer.zero_grad(set_to_none=True)
        step_components: List[Dict[str, Any]] = []
        for micro in range(grad_accum):
            epoch = (micro_index * batch_size) // max(1, n_blocks)
            # Cached: the same three arguments give the same order, and `epoch` does not
            # advance inside a run at E6A scale. See `epoch_order_array` — this line used
            # to reshuffle ~3.1M elements in Python on every micro-batch.
            order = epoch_order_array(n_blocks, setup.seed, epoch)
            offset = (micro_index * batch_size) % max(1, n_blocks)
            indices = order[(offset + np.arange(batch_size)) % n_blocks].tolist()
            input_ids = setup.train_data.batch(indices, setup.device)

            # Only the last micro-batch's breakdown is logged (`_average_components` keeps
            # `rows[-1]`), so only the last one pays to read it back off the GPU.
            loss, components = compute_losses(
                setup, input_ids, need_attention=need_attention,
                want_breakdown=(micro == grad_accum - 1))
            (loss / grad_accum).backward()
            step_components.append(components)
            micro_index += 1

        grad_norm = torch.nn.utils.clip_grad_norm_(setup.student.parameters(), grad_clip)
        optimizer.step()
        scheduler.step()

        last_components = _average_components(step_components)
        append_jsonl(train_log, {
            "step": step + 1,
            "lr": float(scheduler.get_last_lr()[0]),
            "grad_norm": float(grad_norm),
            "tokens": int((step + 1) * grad_accum * batch_size * setup.train_data.block_size),
            **{k: v for k, v in last_components.items() if k != "attn_breakdown"},
            "attn_breakdown": last_components.get("attn_breakdown"),
        })

        completed = step + 1
        if not runtime_written and completed >= 100:
            write_runtime_estimate(setup, completed - start_step, time.time() - started)
            runtime_written = True

        if completed in setup.checkpoints:
            save_checkpoint(setup.run_dir, completed, setup.student, optimizer, scheduler,
                            extra={"step": completed})
            metrics = evaluate(setup, max_batches=2 if setup.smoke else None)
            append_jsonl(eval_log, {"step": completed, **metrics})

    if not runtime_written:
        write_runtime_estimate(setup, end_step - start_step,
                               max(time.time() - started, 1e-9))

    # A requested endpoint that is not one of the configured checkpoints still gets a
    # checkpoint AND an eval row, so every saved checkpoint has matching metrics and the
    # gate never has to interpolate. This covers both a short horizon and --stop-after.
    final_step = end_step
    if final_step not in setup.checkpoints and final_step > start_step:
        save_checkpoint(setup.run_dir, final_step, setup.student, optimizer, scheduler,
                        extra={"step": final_step})
        metrics = evaluate(setup, max_batches=2 if setup.smoke else None)
        append_jsonl(eval_log, {"step": final_step, **metrics})

    return {"run_dir": str(setup.run_dir), "steps": end_step,
            "target_steps": setup.max_steps,
            "initial_state_sha256": setup.initial_state_sha256,
            "block_manifest_sha256": setup.train_data.manifest_sha256,
            "final_components": last_components}


def _average_components(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Mean of the scalar components across grad-accumulation micro-batches."""
    if not rows:
        return {}
    out: Dict[str, Any] = {}
    for key in ("loss", "l_ce", "l_kd", "l_attn"):
        values = [r[key] for r in rows if key in r]
        out[key] = float(np.nanmean(values)) if values else float("nan")
    out["attn_breakdown"] = rows[-1].get("attn_breakdown")
    return out


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir",
                        default=str(_REPO / "transformation_inheritance" / "results"))
    parser.add_argument("--max-steps", type=int, default=None,
                        help=("override the scheduler horizon; do not use this for a pilot "
                              "that must later resume—use --stop-after instead"))
    parser.add_argument(
        "--stop-after", type=int, default=None,
        help=("stop at this checkpoint while retaining the config/--max-steps scheduler "
              "horizon; use for a resumable pilot, e.g. --stop-after 2000 with a "
              "10,000-step config"))
    parser.add_argument("--resume", default="auto", choices=("auto", "none"))
    parser.add_argument("--smoke", action="store_true",
                        help="5 steps on tiny random configs, CPU, fp32, no downloads")
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--corpus-cache", default=str(DEFAULT_CORPUS_CACHE),
        help=("directory for the packed-corpus cache, or 'none' to pack every time. "
              "Packing the E6A-GPT2 window is ~17-30 min of single-threaded work per "
              "split, repeated on every run and every resume; the cache pays it once and "
              "re-verifies the block digest on every hit."))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    max_steps = args.max_steps
    if args.smoke and max_steps is None:
        max_steps = 5

    # An explicit 'none' is the opt-out; anything else is a path. A smoke run synthesises
    # its blocks and never reaches the cache either way.
    corpus_cache = (None if str(args.corpus_cache).lower() == "none"
                    else Path(args.corpus_cache))

    setup = prepare_run(config, seed=args.seed, output_dir=Path(args.output_dir),
                        max_steps=max_steps, smoke=args.smoke, device=args.device,
                        corpus_cache=corpus_cache)
    if args.stop_after is not None and not (0 < args.stop_after <= setup.max_steps):
        raise SystemExit(
            f"--stop-after must be in [1, {setup.max_steps}], got {args.stop_after}")

    print(f"E6A {setup.condition} seed={setup.seed} -> {setup.run_dir}")
    print(f"  weights={setup.weights} T={setup.temperature} layer_map={setup.layer_map}")
    print(f"  teacher layers={_num_layers(setup.teacher_config)} "
          f"({attention_types(setup.teacher_config)})")
    print(f"  student layers={_num_layers(setup.student_config)} "
          f"({attention_types(setup.student_config)})")
    print(f"  device={setup.device} dtype={setup.dtype} steps={setup.max_steps}")
    print(f"  initial_state_sha256={setup.initial_state_sha256[:16]}…")
    print(f"  block_manifest_sha256={setup.train_data.manifest_sha256[:16]}…")

    summary = train(setup, resume=args.resume, stop_after=args.stop_after)
    # `:.4g`, not `round(v, 5)`: L_ATTN is legitimately ~1e-5 on a small model and fixed
    # rounding would print it as 0.0, which reads as "the term is inert".
    components = ", ".join(f"{k}={v:.4g}"
                           for k, v in summary["final_components"].items()
                           if isinstance(v, float))
    stopped = (f" (paused; target {summary['target_steps']})"
               if summary["steps"] < summary["target_steps"] else "")
    print(f"  done: {summary['steps']} steps{stopped}, final components: {components}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
