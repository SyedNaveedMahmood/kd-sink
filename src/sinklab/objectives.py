"""Pure, FP32 attention and behavior objectives for E6A v2.

Inputs are already mapped teacher/student layers. Teacher tensors are detached
at this boundary. No objective owns a model, optimizer, or mutable collector.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F


class ObjectiveError(ValueError):
    pass


METHOD_IDS = {
    "S1": {"C0": "ce_only", "C1": "logit_kd", "C2": "cosine_soft_jsd_v2",
           "C3": "head_mean_probability_mse_v1", "C4": "causal_qq_kk_vv_v1",
           "C5": "cosine_soft_conditional_nosink_jsd_v1", "C6": "cosine_soft_binary_sink_jsd_v1"},
    "S3": {"C0": "ce_only", "C1": "logit_kd", "C2": "index_jsd_v1",
           "C3": "index_probability_mse_v1", "C4": "causal_qq_kk_vv_v1"},
}


def _token_mask(mask: torch.Tensor) -> torch.Tensor:
    if mask.ndim != 2 or mask.dtype != torch.bool or mask.shape[1] < 2:
        raise ObjectiveError("token mask must be bool [batch, length>=2]")
    if ((~mask[:, :-1]) & mask[:, 1:]).any():
        raise ObjectiveError("only right padding is supported")
    return mask


def _edges(mask: torch.Tensor, *, no_sink: bool = False) -> torch.Tensor:
    mask = _token_mask(mask)
    length = mask.shape[1]
    q = torch.arange(length, device=mask.device)
    edge = mask[:, :, None] & mask[:, None, :] & (q[:, None] >= q[None, :])
    edge[:, 0, :] = False
    if no_sink:
        edge[:, :, 0] = False
    if not edge[:, 1:, :].any(dim=(-1, -2)).all():
        raise ObjectiveError("every example needs at least two real tokens")
    return edge


def _mean_examples_layers(values: Sequence[torch.Tensor]) -> torch.Tensor:
    if not values:
        raise ObjectiveError("at least one mapped layer is required")
    return torch.stack([v.mean() for v in values]).mean()


def _row_average(row_values: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
    # row_values: [B,H,Q]; each example and head has equal weight.
    count = rows.sum(-1)
    if (count == 0).any():
        raise ObjectiveError("an example has no valid attention query")
    return (row_values * rows[:, None, :]).sum(-1) / count[:, None]


def shifted_ce(student_logits: torch.Tensor, input_ids: torch.Tensor,
               token_mask: torch.Tensor) -> torch.Tensor:
    mask = _token_mask(token_mask)
    if student_logits.ndim != 3 or student_logits.shape[:2] != input_ids.shape or input_ids.shape != mask.shape:
        raise ObjectiveError("CE shapes must be logits [B,L,V], ids/mask [B,L]")
    valid = mask[:, :-1] & mask[:, 1:]
    if not valid.any():
        raise ObjectiveError("no valid shifted targets")
    logp = F.log_softmax(student_logits[:, :-1].float(), dim=-1)
    picked = logp.gather(-1, input_ids[:, 1:].long().unsqueeze(-1)).squeeze(-1)
    return -picked[valid].mean()


def full_vocab_kd(student_logits: torch.Tensor, teacher_logits: torch.Tensor,
                  token_mask: torch.Tensor, temperature: float = 2.0) -> torch.Tensor:
    mask = _token_mask(token_mask)
    if student_logits.shape != teacher_logits.shape or student_logits.ndim != 3 or student_logits.shape[:2] != mask.shape:
        raise ObjectiveError("KD logits must share [B,L,V] shape")
    if temperature <= 0 or not torch.isfinite(torch.tensor(temperature)):
        raise ObjectiveError("temperature must be positive and finite")
    valid = mask[:, :-1] & mask[:, 1:]
    if not valid.any():
        raise ObjectiveError("no valid shifted targets")
    teacher_log = F.log_softmax(teacher_logits.detach()[:, :-1].float() / temperature, dim=-1)
    student_log = F.log_softmax(student_logits[:, :-1].float() / temperature, dim=-1)
    return (teacher_log.exp() * (teacher_log - student_log)).sum(-1)[valid].mean() * temperature**2


def _validate_maps(teacher: torch.Tensor, student: torch.Tensor, edge: torch.Tensor) -> None:
    if teacher.ndim != 4 or student.ndim != 4 or teacher.shape[0] != student.shape[0] or teacher.shape[2:] != student.shape[2:] or teacher.shape[2:] != edge.shape[1:]:
        raise ObjectiveError("attention maps must be [B,H,L,L] with matched batch/length")
    if teacher.shape[1] < 1 or student.shape[1] < 1:
        raise ObjectiveError("at least one head is required")
    if not torch.isfinite(teacher).all() or not torch.isfinite(student).all():
        raise ObjectiveError("attention probabilities must be finite")


def _jsd_rows(p: torch.Tensor, r: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    # p,r [B,H,Q,K]; zero contribution outside the exact key support.
    m = (p + r) * 0.5
    term_p = p * (p.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())
    term_r = r * (r.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())
    return (0.5 * (term_p + term_r)).masked_fill(~edge[:, None], 0).sum(-1)


def primitive_jsd(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """JSD over the last axis, with no reduction of leading axes."""
    m = (p.float() + q.float()) * 0.5
    a, b = p.float(), q.float()
    return 0.5 * (a * (a.clamp_min(1e-30).log() - m.clamp_min(1e-30).log()) +
                  b * (b.clamp_min(1e-30).log() - m.clamp_min(1e-30).log())).sum(-1)


def _cosine_mix(teacher: torch.Tensor, student: torch.Tensor,
                edge: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    t = teacher.masked_fill(~edge[:, None], 0).flatten(-2)
    s = student.masked_fill(~edge[:, None], 0).flatten(-2)
    t = F.normalize(t, p=2, dim=-1, eps=eps)
    s = F.normalize(s, p=2, dim=-1, eps=eps)
    weights = torch.softmax(torch.bmm(t, s.transpose(1, 2)), dim=-1)
    return torch.einsum("bht,btqk->bhqk", weights, student)


def _conditional(scores: torch.Tensor, edge: torch.Tensor) -> torch.Tensor:
    # Query 0 has no non-sink key; give softmax a temporary finite entry and
    # remove the row afterward. The original sink score never enters this graph.
    safe = edge.clone()
    safe[:, :, 0] |= ~safe.any(-1)
    logits = scores.float().masked_fill(~safe[:, None], -torch.inf)
    probs = torch.softmax(logits, dim=-1)
    return probs.masked_fill(~edge[:, None], 0)


def attention_auxiliary(teacher: torch.Tensor, student: torch.Tensor,
                        token_mask: torch.Tensor, method: str,
                        *, teacher_scores: torch.Tensor | None = None,
                        student_scores: torch.Tensor | None = None) -> torch.Tensor:
    """One mapped-layer C2/C3/C5/C6 or S3 index loss."""
    edge = _edges(token_mask, no_sink=method == "cosine_soft_conditional_nosink_jsd_v1")
    _validate_maps(teacher, student, edge)
    t, s = teacher.detach().float(), student.float()
    rows = edge.any(-1)
    if method == "cosine_soft_conditional_nosink_jsd_v1":
        if teacher_scores is None or student_scores is None or teacher_scores.shape != teacher.shape or student_scores.shape != student.shape:
            raise ObjectiveError("NoSink requires matched original attention scores")
        t = _conditional(teacher_scores.detach(), edge)
        s = _conditional(student_scores, edge)
    elif method == "cosine_soft_binary_sink_jsd_v1":
        # Ignore all non-sink shape, including during alignment.
        t0 = t[..., :1]
        s0 = s[..., :1]
        t = torch.cat((t0, 1 - t0), dim=-1)
        s = torch.cat((s0, 1 - s0), dim=-1)
        edge = rows[:, :, None].expand(-1, -1, 2)
    if method in {"cosine_soft_jsd_v2", "cosine_soft_conditional_nosink_jsd_v1", "cosine_soft_binary_sink_jsd_v1"}:
        mixed = _cosine_mix(t, s, edge)
        return _row_average(_jsd_rows(t, mixed, edge), rows).mean()
    if method == "head_mean_probability_mse_v1":
        diff = t.mean(1) - s.mean(1)
        return ((diff.square() * edge).sum((-1, -2)) / edge.sum((-1, -2))).mean()
    if method == "index_jsd_v1":
        if t.shape[1] != s.shape[1]:
            raise ObjectiveError("fixed-index JSD requires equal native head counts")
        return _row_average(_jsd_rows(t, s, edge), rows).mean()
    if method == "index_probability_mse_v1":
        if t.shape[1] != s.shape[1]:
            raise ObjectiveError("fixed-index MSE requires equal native head counts")
        return ((t - s).square() * edge[:, None]).sum((-1, -2)).div(edge.sum((-1, -2))[:, None]).mean()
    raise ObjectiveError(f"unknown attention method: {method}")


def _relation(x: torch.Tensor, edge: torch.Tensor, relation_heads: int) -> torch.Tensor:
    batch, length, width = x.shape
    if relation_heads < 1 or width % relation_heads:
        raise ObjectiveError("width must be divisible by relation head count")
    dim = width // relation_heads
    x = x.float().reshape(batch, length, relation_heads, dim).transpose(1, 2)
    scores = torch.matmul(x, x.transpose(-1, -2)) / sqrt(dim)
    # Query 0 remains valid for softmax, then is excluded from the reduction.
    causal = edge.clone()
    causal[:, :, 0] |= ~causal.any(-1)
    return torch.log_softmax(scores.masked_fill(~causal[:, None], -torch.inf), dim=-1)


def relation_auxiliary(teacher_layers: Sequence[Mapping[str, torch.Tensor]],
                       student_layers: Sequence[Mapping[str, torch.Tensor]],
                       token_mask: torch.Tensor, *, relation_heads: int = 64,
                       chunk_size: int | None = None) -> torch.Tensor:
    """Causal QQ/KK/VV KL, streamed over mapped layers/types/relation heads."""
    edge = _edges(token_mask)
    if len(teacher_layers) != len(student_layers) or not teacher_layers:
        raise ObjectiveError("mapped relation layers must have equal nonzero length")
    if chunk_size is None:
        chunk_size = relation_heads
    if chunk_size < 1:
        raise ObjectiveError("chunk_size must be positive")
    rows = edge.any(-1)
    terms: list[torch.Tensor] = []
    for t_layer, s_layer in zip(teacher_layers, student_layers):
        for kind in ("query", "key", "value"):
            if kind not in t_layer or kind not in s_layer:
                raise ObjectiveError("relation layer needs query, key and value")
            t, s = t_layer[kind].detach(), s_layer[kind]
            if t.ndim != 3 or s.ndim != 3 or t.shape[:2] != token_mask.shape or s.shape[:2] != token_mask.shape:
                raise ObjectiveError("relation projections must be [B,L,D]")
            if t.shape[-1] % relation_heads or s.shape[-1] % relation_heads:
                raise ObjectiveError("relation widths must divide relation_heads")
            td, sd = t.shape[-1] // relation_heads, s.shape[-1] // relation_heads
            t_view = t.reshape(*token_mask.shape, relation_heads, td)
            s_view = s.reshape(*token_mask.shape, relation_heads, sd)
            for start in range(0, relation_heads, chunk_size):
                stop = min(start + chunk_size, relation_heads)
                t_log = _relation(t_view[:, :, start:stop].reshape(*token_mask.shape, -1), edge, stop-start)
                s_log = _relation(s_view[:, :, start:stop].reshape(*token_mask.shape, -1), edge, stop-start)
                p = t_log.exp()
                kl = torch.where(edge[:, None], p * (t_log - s_log), 0).sum(-1)
                # Sum head means; fixed denominator below makes chunking exact.
                terms.append(_row_average(kl, rows).sum())
    return torch.stack(terms).sum() / (len(teacher_layers) * 3 * relation_heads * token_mask.shape[0])


@dataclass(frozen=True)
class ObjectiveResult:
    total: torch.Tensor
    active: Mapping[str, torch.Tensor]
    inactive: tuple[str, ...]
    method_id: str


def compose_objective(study: str, condition: str, student_logits: torch.Tensor,
                      input_ids: torch.Tensor, token_mask: torch.Tensor,
                      *, teacher_logits: torch.Tensor | None = None,
                      attention_pairs: Sequence[tuple[torch.Tensor, torch.Tensor,
                                                      torch.Tensor | None, torch.Tensor | None]] = (),
                      relation_pairs: tuple[Sequence[Mapping[str, torch.Tensor]],
                                            Sequence[Mapping[str, torch.Tensor]]] | None = None,
                      mse_scale: float | None = None, rel_scale: float | None = None) -> ObjectiveResult:
    if study not in METHOD_IDS or condition not in METHOD_IDS[study]:
        raise ObjectiveError("unsupported study/condition")
    method = METHOD_IDS[study][condition]
    ce = shifted_ce(student_logits, input_ids, token_mask)
    active: dict[str, torch.Tensor] = {"ce": ce}
    inactive = ["kd", "attention", "relation"]
    if condition == "C0":
        return ObjectiveResult(ce, active, tuple(inactive), method)
    if teacher_logits is None:
        raise ObjectiveError("teacher logits required")
    kd = full_vocab_kd(student_logits, teacher_logits, token_mask)
    active["kd"] = kd
    inactive.remove("kd")
    total = (ce + kd) * 0.5
    if condition in {"C2", "C3", "C5", "C6"}:
        if not attention_pairs:
            raise ObjectiveError("mapped attention pairs required")
        aux = _mean_examples_layers([attention_auxiliary(t, s, token_mask, method,
                                teacher_scores=ts, student_scores=ss)
                         for t, s, ts, ss in attention_pairs])
        if condition == "C3":
            if mse_scale is None or not (0 < mse_scale < float("inf")):
                raise ObjectiveError("positive frozen MSE calibration required")
            aux = aux * mse_scale
        active["attention"] = aux
        inactive.remove("attention")
        total = total + aux / 9
    elif condition == "C4":
        if relation_pairs is None or rel_scale is None or not (0 < rel_scale < float("inf")):
            raise ObjectiveError("relation pairs and positive frozen REL calibration required")
        aux = relation_auxiliary(*relation_pairs, token_mask)
        active["relation"] = aux * rel_scale
        inactive.remove("relation")
        total = total + active["relation"] / 9
    return ObjectiveResult(total, active, tuple(inactive), method)
