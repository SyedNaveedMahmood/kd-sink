"""Exact token-weighted behavior and descriptive sink measurements."""

from __future__ import annotations

import math
import sys
from typing import Sequence

import numpy as np
import torch
from scipy.stats import spearmanr


class MetricError(ValueError):
    pass


METRIC_VERSION = "e6a-v2-metrics-1"
S7_DECOMPOSITION_VERSION = "s7-head-mean-jsd-decomposition-v1"


def behavioral_item(clean: torch.Tensor, edited: torch.Tensor, ids: torch.Tensor,
                    mask: torch.Tensor, *, teacher: torch.Tensor | None = None,
                    token_chunk: int = 32) -> dict:
    """All vocabulary entries are used for each valid shifted target."""
    if clean.ndim != 3 or edited.shape != clean.shape or ids.shape != clean.shape[:2] or mask.shape != ids.shape:
        raise MetricError("logit/target/mask shapes disagree")
    if teacher is not None and teacher.shape != clean.shape:
        raise MetricError("teacher logits shape mismatch")
    if token_chunk < 1:
        raise MetricError("positive token chunk required")
    valid = (mask[:, :-1].bool() & mask[:, 1:].bool()).reshape(-1)
    if not valid.any():
        raise MetricError("no valid shifted target")
    targets = ids[:, 1:].reshape(-1)[valid].long()
    c = clean[:, :-1].reshape(-1, clean.shape[-1])[valid]
    e = edited[:, :-1].reshape(-1, clean.shape[-1])[valid]
    t = teacher[:, :-1].reshape(-1, clean.shape[-1])[valid] if teacher is not None else None
    result = {"valid_targets": int(targets.numel()), "clean_nll_sum_nats": 0.,
              "edited_nll_sum_nats": 0., "self_kl_sum_nats": 0.,
              "absolute_target_logprob_change_sum_nats": 0., "flip_count": 0,
              "clean_correct_count": 0, "edited_correct_count": 0,
              "teacher_kl_sum_nats": None if t is None else 0.,
              "teacher_top1_agreement_count": None if t is None else 0}
    for start in range(0, len(targets), token_chunk):
        stop = start + token_chunk
        y = targets[start:stop]
        cl = torch.log_softmax(c[start:stop].float(), -1)
        el = torch.log_softmax(e[start:stop].float(), -1)
        cy = cl.gather(-1, y[:, None]).squeeze(-1)
        ey = el.gather(-1, y[:, None]).squeeze(-1)
        self_kl = (cl.exp() * (cl - el)).sum(-1)
        if (self_kl < -2e-5).any():
            raise MetricError("materially negative self-KL")
        result["clean_nll_sum_nats"] += float((-cy).sum())
        result["edited_nll_sum_nats"] += float((-ey).sum())
        result["self_kl_sum_nats"] += float(self_kl.sum())
        result["absolute_target_logprob_change_sum_nats"] += float((ey - cy).abs().sum())
        cp, ep = cl.argmax(-1), el.argmax(-1)
        result["flip_count"] += int((cp != ep).sum())
        result["clean_correct_count"] += int((cp == y).sum())
        result["edited_correct_count"] += int((ep == y).sum())
        if t is not None:
            tl = torch.log_softmax(t[start:stop].float(), -1)
            result["teacher_kl_sum_nats"] += float((tl.exp() * (tl - el)).sum())
            result["teacher_top1_agreement_count"] += int((tl.argmax(-1) == ep).sum())
    return result


def _exp(value: float) -> tuple[float | None, str | None]:
    if value > math.log(sys.float_info.max):
        return None, "exponent_overflow"
    return math.exp(value), None


def aggregate_behavior(items: Sequence[dict]) -> dict:
    if not items or any(item.get("valid_targets", 0) <= 0 for item in items):
        raise MetricError("nonempty valid item records required")
    n = sum(item["valid_targets"] for item in items)
    def total(field):
        return sum(item[field] for item in items)
    clean_ce = total("clean_nll_sum_nats") / n
    edited_ce = total("edited_nll_sum_nats") / n
    delta = edited_ce - clean_ce
    clean_ppl, clean_overflow = _exp(clean_ce)
    edited_ppl, edited_overflow = _exp(edited_ce)
    ratio, ratio_overflow = _exp(delta)
    absolute_items = sorted(item["absolute_target_logprob_change_sum_nats"] / item["valid_targets"] for item in items)
    def quantile(q):
        return float(np.quantile(absolute_items, q))
    teacher = all(item["teacher_kl_sum_nats"] is not None for item in items)
    return {"schema_version": METRIC_VERSION, "units": {"ce": "nats_per_target", "ppl": "ratio",
             "flip": "fraction", "accuracy": "fraction", "kl": "nats_per_target"},
            "item_count": len(items), "valid_targets": n,
            "clean_ce_nats": clean_ce, "edited_ce_nats": edited_ce,
            "clean_log_ppl": clean_ce, "edited_log_ppl": edited_ce,
            "clean_ppl": clean_ppl, "edited_ppl": edited_ppl,
            "ppl_overflow_reason": clean_overflow or edited_overflow,
            "delta_ce_nats": delta,
            "relative_delta_ce_percent": 100 * delta / clean_ce if clean_ce > 1e-12 else None,
            "relative_delta_ce_unavailable_reason": None if clean_ce > 1e-12 else "nonpositive_or_tiny_clean_ce",
            "ppl_ratio": ratio, "ppl_ratio_unavailable_reason": ratio_overflow,
            "relative_ppl_change": math.expm1(delta) if ratio is not None else None,
            "self_kl_nats": total("self_kl_sum_nats") / n,
            "absolute_target_logprob_change_nats": total("absolute_target_logprob_change_sum_nats") / n,
            "median_item_absolute_change_nats": quantile(.5),
            "p90_item_absolute_change_nats": quantile(.9),
            "prediction_flip_fraction": total("flip_count") / n,
            "clean_accuracy_fraction": total("clean_correct_count") / n,
            "edited_accuracy_fraction": total("edited_correct_count") / n,
            "teacher_kl_nats": total("teacher_kl_sum_nats") / n if teacher else None,
            "teacher_top1_agreement_fraction": total("teacher_top1_agreement_count") / n if teacher else None}


def sink_head_table(probabilities: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Return [B,H] item means over each item's second-half real queries."""
    if probabilities.ndim != 4 or probabilities.shape[0] != mask.shape[0] or probabilities.shape[-2:] != (mask.shape[1], mask.shape[1]):
        raise MetricError("attention/mask shape mismatch")
    values = []
    for b in range(mask.shape[0]):
        length = int(mask[b].bool().sum())
        if length < 2 or not mask[b, :length].all() or mask[b, length:].any():
            raise MetricError("sink profile needs right-padded real length >=2")
        values.append(probabilities[b, :, (length + 1) // 2:length, 0].float().mean(-1))
    return torch.stack(values)


def sink_profile(layers: Sequence[torch.Tensor], mask: torch.Tensor) -> dict:
    if not layers:
        raise MetricError("at least one native layer required")
    tables = [sink_head_table(layer, mask).mean(0).detach().cpu().tolist() for layer in layers]
    amplitudes = [sum(heads) / len(heads) for heads in tables]
    return {"layer_head_sink": tables, "layer_amplitude": amplitudes,
            "native_layer_mean": sum(amplitudes) / len(amplitudes),
            "query_rule": "ceil(real_length/2)..real_length-1"}


def depth_support(profile: Sequence[float], points: int = 16) -> list[float]:
    if not profile or points < 2 or any(not math.isfinite(x) or x < 0 for x in profile):
        raise MetricError("finite nonnegative depth profile required")
    return np.interp(np.linspace(0, 1, points), np.linspace(0, 1, len(profile)), profile).tolist()


def weighted_depth_wasserstein(a: Sequence[float], b: Sequence[float], points: int = 16) -> float | None:
    x, y = np.asarray(depth_support(a, points)), np.asarray(depth_support(b, points))
    if x.sum() <= 0 or y.sum() <= 0:
        return None
    return float(np.abs(np.cumsum(x / x.sum() - y / y.sum())[:-1]).sum() / (points - 1))


def guarded_spearman(a: Sequence[float], b: Sequence[float]) -> float | None:
    if len(a) != len(b) or len(a) < 2:
        raise MetricError("matched depth profiles required")
    if len(set(a)) == 1 or len(set(b)) == 1:
        return None
    return float(spearmanr(a, b).statistic)


def fingerprint(baseline: float, probed: float, *, denominator_floor: float) -> dict:
    if not all(math.isfinite(v) for v in (baseline, probed, denominator_floor)) or denominator_floor <= 0:
        raise MetricError("finite fingerprint and positive locked floor required")
    return {"baseline_sink": baseline, "probed_sink": probed,
            "absolute_delta_sink": probed - baseline,
            "ratio": probed / baseline if abs(baseline) >= denominator_floor else None,
            "ratio_unavailable_reason": None if abs(baseline) >= denominator_floor else "negligible_baseline",
            "denominator_floor": denominator_floor}


def attention_similarity(teacher: torch.Tensor, student: torch.Tensor,
                         mask: torch.Tensor, *, exclude_sink: bool = False) -> dict:
    """Head-mean descriptive JSD/MSE over the same causal rows and keys."""
    if teacher.ndim != 4 or student.ndim != 4 or teacher.shape[0] != student.shape[0] or teacher.shape[-2:] != student.shape[-2:] or mask.shape != teacher.shape[:1] + teacher.shape[-1:]:
        raise MetricError("mapped attention shapes mismatch")
    t, s = teacher.float().mean(1), student.float().mean(1)
    length = mask.shape[1]
    edge = torch.ones((length, length), device=mask.device, dtype=torch.bool).tril()
    edge = edge[None] & mask[:, :, None].bool() & mask[:, None, :].bool()
    edge[:, 0] = False
    if exclude_sink:
        edge[:, :, 0] = False
        t = t * edge
        s = s * edge
        if (t.sum(-1)[edge.any(-1)] <= 0).any() or (s.sum(-1)[edge.any(-1)] <= 0).any():
            raise MetricError("sink-excluded map has zero conditional mass")
        t = t / t.sum(-1, keepdim=True).clamp_min(1e-30)
        s = s / s.sum(-1, keepdim=True).clamp_min(1e-30)
    mid = (t + s) / 2
    tlog = torch.log(t.clamp_min(1e-30)) - torch.log(mid.clamp_min(1e-30))
    slog = torch.log(s.clamp_min(1e-30)) - torch.log(mid.clamp_min(1e-30))
    jsd = .5 * (t * tlog + s * slog).masked_fill(~edge, 0).sum(-1)
    rows = edge.any(-1)
    return {"jsd_nats": float(jsd[rows].mean()),
            "mse_probability_cells": float(((t - s).square() * edge).sum() / edge.sum()),
            "valid_query_count": int(rows.sum()), "valid_edge_count": int(edge.sum())}


def attention_jsd_decomposition(teacher: torch.Tensor, student: torch.Tensor,
                                mask: torch.Tensor) -> dict:
    """S7 descriptive head-mean JSD split into sink mass and residual shape.

    Each example is reduced over its valid q>=1 rows before examples are
    averaged. Native heads are averaged separately for teacher and student;
    this is deliberately distinct from the C2/C5/C6 training alignments.
    Undefined conditional comparisons remain unavailable, never zero-filled.
    """
    if (teacher.ndim != 4 or student.ndim != 4 or
            teacher.shape[0] != student.shape[0] or
            teacher.shape[-2:] != student.shape[-2:] or
            mask.shape != teacher.shape[:1] + teacher.shape[-1:]):
        raise MetricError("S7 attention/mask shapes disagree")
    if mask.dtype != torch.bool:
        raise MetricError("S7 requires a boolean right-padding mask")
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise MetricError("S7 attention contains nonfinite values")
    if (teacher < 0).any() or (student < 0).any():
        raise MetricError("S7 attention contains negative probabilities")
    length = mask.shape[1]
    if length < 2:
        raise MetricError("S7 needs at least two positions")
    for item in mask:
        count = int(item.sum())
        if count < 2 or not item[:count].all() or item[count:].any():
            raise MetricError("S7 requires right-padded real length >=2")
    edge = torch.ones((length, length), dtype=torch.bool, device=mask.device).tril()
    edge = edge[None] & mask[:, :, None] & mask[:, None, :]
    rows = edge.any(-1)
    rows[:, 0] = False
    edge[:, 0] = False
    for source in (teacher, student):
        invalid = source.double().masked_fill(edge[:, None], 0)
        # Query 0 is intentionally ignored, including its forced self-edge.
        invalid[:, :, 0] = 0
        if invalid.abs().max() > 2e-6:
            raise MetricError("S7 attention has probability outside causal support")
        sums = (source.double() * edge[:, None]).sum(-1)
        if ((sums[rows[:, None].expand_as(sums)] - 1).abs() > 2e-6).any():
            raise MetricError("S7 attention rows are not normalized")
    t = teacher.double().mean(1) * edge
    s = student.double().mean(1) * edge
    # Correct sub-FP32 summation noise only after enforcing native row sums.
    t = t / t.sum(-1, keepdim=True).clamp_min(1e-300)
    s = s / s.sum(-1, keepdim=True).clamp_min(1e-300)
    midpoint = (t + s) / 2
    def kl_terms(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        return torch.where(p > 0, p * (p.clamp_min(1e-300).log() -
                                        q.clamp_min(1e-300).log()), 0.)
    columns = .5 * (kl_terms(t, midpoint) + kl_terms(s, midpoint))
    full = columns.sum(-1)
    key0 = columns[..., 0]
    other = columns[..., 1:].sum(-1)
    rt, rs = t[..., 1:].sum(-1), s[..., 1:].sum(-1)
    binary_mid = (rt + rs) / 2
    mass = .5 * (kl_terms(t[..., 0], midpoint[..., 0]) +
                 kl_terms(s[..., 0], midpoint[..., 0]) +
                 kl_terms(rt, binary_mid) + kl_terms(rs, binary_mid))
    ut = t[..., 1:] / rt.unsqueeze(-1).clamp_min(1e-300)
    us = s[..., 1:] / rs.unsqueeze(-1).clamp_min(1e-300)
    weighted_mid = (t[..., 1:] + s[..., 1:]) / (rt + rs).unsqueeze(-1).clamp_min(1e-300)
    shape = .5 * (rt * kl_terms(ut, weighted_mid).sum(-1) +
                  rs * kl_terms(us, weighted_mid).sum(-1))
    both = (rt > 0) & (rs > 0) & rows
    conditional_mid = (ut + us) / 2
    conditional = .5 * (kl_terms(ut, conditional_mid) +
                        kl_terms(us, conditional_mid)).sum(-1)
    closure = torch.maximum((full - mass - shape).abs(),
                            (full - key0 - other).abs())
    values = {"full_jsd_nats": full, "mass_jsd_nats": mass,
              "shape_jsd_nats": shape, "key0_jsd_nats": key0,
              "other_columns_jsd_nats": other}
    result = {"version": S7_DECOMPOSITION_VERSION,
              "reduction": "head_mean_then_equal_queries_then_equal_items",
              "valid_query_count": int(rows.sum()),
              "conditional_valid_query_count": int(both.sum()),
              "max_closure_error_nats": float(closure[rows].max())}
    for name, tensor in values.items():
        result[name] = float(torch.stack([tensor[b][rows[b]].mean()
                                          for b in range(len(rows))]).mean())
    result["conditional_jsd_nats"] = (
        float(torch.stack([conditional[b][rows[b]].mean()
                           for b in range(len(rows))]).mean())
        if bool(torch.equal(both, rows)) else None)
    result["conditional_unavailable_reason"] = (
        None if result["conditional_jsd_nats"] is not None else
        "zero_non_sink_mass_in_at_least_one_distribution")
    if (result["max_closure_error_nats"] > 1e-8 or
            any(not math.isfinite(value) or value < -1e-8 for value in
                (result[name] for name in values))):
        raise MetricError("S7 JSD decomposition failed numerical closure")
    return result
