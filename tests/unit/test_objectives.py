"""Independent scalar-loop references for Stage 03 objectives."""

import math

import pytest
import torch
import torch.nn.functional as F

from sinklab.config import VARIANTS
from sinklab.objectives import (
    METHOD_IDS, ObjectiveError, attention_auxiliary, compose_objective,
    full_vocab_kd, primitive_jsd, relation_auxiliary, shifted_ce,
)


def _mask():
    return torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0]], dtype=torch.bool)


def _maps(heads_t=3, heads_s=2):
    torch.manual_seed(13)
    a = torch.randn(2, heads_t, 4, 4, requires_grad=True)
    b = torch.randn(2, heads_s, 4, 4, requires_grad=True)
    causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
    t = torch.softmax(a.masked_fill(~causal, -1e4), -1)
    s = torch.softmax(b.masked_fill(~causal, -1e4), -1)
    return a, b, t, s


def _scalar_jsd(p, q):
    out = p.new_zeros(())
    for k in range(p.numel()):
        m = (p[k] + q[k]) / 2
        out = out + 0.5 * (p[k] * (p[k].clamp_min(1e-30).log() - m.clamp_min(1e-30).log()) +
                           q[k] * (q[k].clamp_min(1e-30).log() - m.clamp_min(1e-30).log()))
    return out


def _reference_attention(t, s, mask, method, ta=None, sa=None):
    values = []
    for b in range(mask.shape[0]):
        length = int(mask[b].sum())
        teacher = t[b]
        student = s[b]
        if method == "cosine_soft_conditional_nosink_jsd_v1":
            # Build full rows from non-sink scores, independently of the original softmax.
            def conditional(scores):
                rows = []
                for h in range(scores.shape[0]):
                    hr = []
                    for q in range(4):
                        if q < 1 or q >= length:
                            hr.append(torch.zeros(4, device=scores.device))
                        else:
                            probs = F.softmax(scores[h, q, 1:q+1], -1)
                            hr.append(F.pad(probs, (1, 3-q)))
                    rows.append(torch.stack(hr))
                return torch.stack(rows)
            teacher = conditional(ta[b].detach())
            student = conditional(sa[b])
        if method == "cosine_soft_binary_sink_jsd_v1":
            teacher = torch.stack((teacher[..., 0], 1-teacher[..., 0]), -1)
            student = torch.stack((student[..., 0], 1-student[..., 0]), -1)
        if method in {"cosine_soft_jsd_v2", "cosine_soft_conditional_nosink_jsd_v1", "cosine_soft_binary_sink_jsd_v1"}:
            head_values = []
            for h in range(teacher.shape[0]):
                sims = []
                for j in range(student.shape[0]):
                    tv, sv = [], []
                    for q in range(1, length):
                        keys = 2 if method == "cosine_soft_binary_sink_jsd_v1" else q+1
                        begin = 1 if method == "cosine_soft_conditional_nosink_jsd_v1" else 0
                        tv.extend(teacher[h, q, begin:keys].unbind())
                        sv.extend(student[j, q, begin:keys].unbind())
                    u, v = torch.stack(tv), torch.stack(sv)
                    sims.append((u @ v) / (u.norm().clamp_min(1e-12) * v.norm().clamp_min(1e-12)))
                weights = torch.softmax(torch.stack(sims), 0)
                rows = []
                for q in range(1, length):
                    keys = 2 if method == "cosine_soft_binary_sink_jsd_v1" else q+1
                    begin = 1 if method == "cosine_soft_conditional_nosink_jsd_v1" else 0
                    mixed = sum(weights[j] * student[j, q, begin:keys] for j in range(student.shape[0]))
                    rows.append(_scalar_jsd(teacher[h, q, begin:keys], mixed))
                head_values.append(torch.stack(rows).mean())
            values.append(torch.stack(head_values).mean())
        elif method == "head_mean_probability_mse_v1":
            cells = [(teacher[:, q, k].mean() - student[:, q, k].mean()).square()
                     for q in range(1, length) for k in range(q+1)]
            values.append(torch.stack(cells).mean())
        elif method == "index_probability_mse_v1":
            cells = [(teacher[h, q, k] - student[h, q, k]).square()
                     for h in range(teacher.shape[0]) for q in range(1, length) for k in range(q+1)]
            values.append(torch.stack(cells).mean())
        else:
            heads = []
            for h in range(teacher.shape[0]):
                heads.append(torch.stack([_scalar_jsd(teacher[h, q, :q+1], student[h, q, :q+1])
                                         for q in range(1, length)]).mean())
            values.append(torch.stack(heads).mean())
    return torch.stack(values).mean()


def _reference_relation(tlayers, slayers, mask, heads):
    layers = []
    for tl, sl in zip(tlayers, slayers):
        types = []
        for kind in ("query", "key", "value"):
            items = []
            for b in range(mask.shape[0]):
                length = int(mask[b].sum())
                for h in range(heads):
                    a = tl[kind][b, :, h*(tl[kind].shape[-1]//heads):(h+1)*(tl[kind].shape[-1]//heads)].detach()
                    c = sl[kind][b, :, h*(sl[kind].shape[-1]//heads):(h+1)*(sl[kind].shape[-1]//heads)]
                    rows = []
                    for q in range(1, length):
                        pa = torch.stack([a[q] @ a[k] for k in range(q+1)]) / math.sqrt(a.shape[-1])
                        pc = torch.stack([c[q] @ c[k] for k in range(q+1)]) / math.sqrt(c.shape[-1])
                        loga, logc = F.log_softmax(pa, 0), F.log_softmax(pc, 0)
                        rows.append((loga.exp() * (loga-logc)).sum())
                    items.append(torch.stack(rows).mean())
            types.append(torch.stack(items).mean())
        layers.append(torch.stack(types).mean())
    return torch.stack(layers).mean()


def _value_gradient_parity(actual, reference, leaves, atol=2e-6):
    assert torch.allclose(actual, reference, atol=atol, rtol=atol)
    ga = torch.autograd.grad(actual, leaves, retain_graph=True, allow_unused=True)
    gr = torch.autograd.grad(reference, leaves, retain_graph=True, allow_unused=True)
    for x, y in zip(ga, gr):
        if x is None or y is None:
            assert x is None and y is None
        else:
            assert torch.isfinite(x).all()
            assert torch.allclose(x, y, atol=atol, rtol=atol)


def test_shifted_ce_kd_independent_reference_and_detachment():
    torch.manual_seed(5)
    s = torch.randn(2, 4, 7, requires_grad=True)
    t = torch.randn(2, 4, 7, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    mask = _mask()
    valid = [(b, q) for b in range(2) for q in range(3) if mask[b, q+1]]
    ce_ref = torch.stack([-F.log_softmax(s[b, q].float(), 0)[ids[b, q+1]] for b, q in valid]).mean()
    kd_ref = torch.stack([4 * (F.softmax(t[b, q].detach().float()/2, 0) *
                  (F.log_softmax(t[b, q].detach().float()/2, 0)-F.log_softmax(s[b, q].float()/2, 0))).sum()
                  for b, q in valid]).mean()
    _value_gradient_parity(shifted_ce(s, ids, mask), ce_ref, (s,))
    _value_gradient_parity(full_vocab_kd(s, t, mask), kd_ref, (s, t))
    result = compose_objective("S1", "C0", s, ids, mask, teacher_logits=t)
    assert result.inactive == ("kd", "attention", "relation") and set(result.active) == {"ce"}
    assert torch.autograd.grad(result.total, t, allow_unused=True)[0] is None
    base = compose_objective("S1", "C1", s, ids, mask, teacher_logits=t)
    assert torch.allclose(base.total, (ce_ref + kd_ref)/2)
    assert set(base.active) == {"ce", "kd"}


@pytest.mark.parametrize("method,heads_t,heads_s", [
    ("cosine_soft_jsd_v2", 3, 2), ("head_mean_probability_mse_v1", 3, 2),
    ("cosine_soft_conditional_nosink_jsd_v1", 3, 2), ("cosine_soft_binary_sink_jsd_v1", 3, 2),
    ("index_jsd_v1", 2, 2), ("index_probability_mse_v1", 2, 2),
])
def test_attention_forward_and_backward_independent_reference(method, heads_t, heads_s):
    ta, sa, t, s = _maps(heads_t, heads_s)
    actual = attention_auxiliary(t, s, _mask(), method, teacher_scores=ta, student_scores=sa)
    reference = _reference_attention(t.detach(), s, _mask(), method, ta, sa)
    _value_gradient_parity(actual, reference, (ta, sa), atol=3e-6)
    assert torch.autograd.grad(actual, ta, allow_unused=True)[0] is None


def test_jsd_primitive_and_reductions():
    p = torch.tensor([0.75, 0.25]); q = torch.tensor([0.25, 0.75])
    assert torch.allclose(primitive_jsd(p, q), primitive_jsd(q, p))
    assert primitive_jsd(p, p) == 0
    assert 0 < primitive_jsd(p, q) < math.log(2)
    _, _, t, s = _maps(1, 1)
    j = attention_auxiliary(t, s, _mask(), "index_jsd_v1")
    mse = attention_auxiliary(t, s, _mask(), "index_probability_mse_v1")
    assert torch.allclose(j, _reference_attention(t.detach(), s, _mask(), "index_jsd_v1"))
    assert torch.allclose(mse, _reference_attention(t.detach(), s, _mask(), "index_probability_mse_v1"))


def test_cosine_permutation_and_index_sensitivity():
    _, _, t, s = _maps(2, 2)
    a = attention_auxiliary(t, s, _mask(), "cosine_soft_jsd_v2")
    b = attention_auxiliary(t, s[:, [1, 0]], _mask(), "cosine_soft_jsd_v2")
    assert torch.allclose(a, b, atol=1e-7)
    assert not torch.allclose(attention_auxiliary(t, s, _mask(), "index_jsd_v1"),
                              attention_auxiliary(t, s[:, [1, 0]], _mask(), "index_jsd_v1"))
    assert METHOD_IDS["S3"]["C2"] != METHOD_IDS["S1"]["C2"]
    assert METHOD_IDS == VARIANTS


def test_nosink_zero_direct_score_gradient_saturation_and_pre_alignment():
    ta, sa, t, s = _maps()
    loss = attention_auxiliary(t, s, _mask(), "cosine_soft_conditional_nosink_jsd_v1", teacher_scores=ta, student_scores=sa)
    grad = torch.autograd.grad(loss, sa)[0]
    assert torch.equal(grad[..., 0], torch.zeros_like(grad[..., 0]))
    moved = sa.detach().clone()
    moved[..., 0] += 1000
    changed = attention_auxiliary(t, s, _mask(), "cosine_soft_conditional_nosink_jsd_v1", teacher_scores=ta, student_scores=moved)
    assert torch.allclose(loss, changed, atol=1e-7)
    assert loss.isfinite()


def test_sinkonly_binary_nonvacuous_and_redistribution_invariant():
    _, _, t, s = _maps()
    first = attention_auxiliary(t, s, _mask(), "cosine_soft_binary_sink_jsd_v1")
    altered = s.detach().clone()
    altered[..., 1:] = torch.flip(altered[..., 1:], (-1,))
    assert torch.allclose(first, attention_auxiliary(t, altered, _mask(), "cosine_soft_binary_sink_jsd_v1"))
    changed = s.detach().clone()
    changed[..., 0] = 0.99
    assert not torch.allclose(first, attention_auxiliary(t, changed, _mask(), "cosine_soft_binary_sink_jsd_v1"))
    assert first > 0
    leaf = s.detach().clone().requires_grad_()
    loss = attention_auxiliary(t, leaf, _mask(), "cosine_soft_binary_sink_jsd_v1")
    gradient = torch.autograd.grad(loss, leaf)[0]
    assert torch.equal(gradient[..., 1:], torch.zeros_like(gradient[..., 1:]))


def test_relation_reference_chunks_axes_and_teacher_detachment():
    torch.manual_seed(27)
    mask = _mask()
    tlayers = [{k: torch.randn(2, 4, 12, requires_grad=True) for k in ("query", "key", "value")}
               for _ in range(2)]
    slayers = [{k: torch.randn(2, 4, 8, requires_grad=True) for k in ("query", "key", "value")}
               for _ in range(2)]
    leaves = tuple(x for layer in tlayers+slayers for x in layer.values())
    reference = _reference_relation(tlayers, slayers, mask, 4)
    full = relation_auxiliary(tlayers, slayers, mask, relation_heads=4)
    chunked = relation_auxiliary(tlayers, slayers, mask, relation_heads=4, chunk_size=1)
    _value_gradient_parity(full, reference, leaves, atol=3e-6)
    _value_gradient_parity(chunked, reference, leaves, atol=3e-6)
    assert torch.allclose(full, chunked, atol=2e-7)
    for kind in ("query", "key", "value"):
        assert torch.autograd.grad(full, tlayers[0][kind], retain_graph=True, allow_unused=True)[0] is None
        assert torch.autograd.grad(full, slayers[0][kind], retain_graph=True)[0].abs().sum() > 0
    with pytest.raises(ObjectiveError, match="divide"):
        relation_auxiliary(tlayers, slayers, mask, relation_heads=5)


def test_objective_composition_scales_and_inactive_fields():
    ta, sa, t, s = _maps()
    logits = torch.randn(2, 4, 7, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    teacher = torch.randn_like(logits)
    pairs = [(t, s, ta, sa)]
    base = compose_objective("S1", "C1", logits, ids, _mask(), teacher_logits=teacher).total
    for condition in ("C2", "C3", "C5", "C6"):
        result = compose_objective("S1", condition, logits, ids, _mask(), teacher_logits=teacher,
                                   attention_pairs=pairs, mse_scale=3.0)
        assert result.active["attention"].isfinite()
        assert torch.allclose(result.total, base + result.active["attention"]/9)
        assert result.inactive == ("relation",)
    with pytest.raises(ObjectiveError, match="calibration"):
        compose_objective("S1", "C3", logits, ids, _mask(), teacher_logits=teacher, attention_pairs=pairs)


def test_s3_index_composition_and_c4_scale():
    _, _, t, s = _maps(2, 2)
    logits = torch.randn(2, 4, 7, requires_grad=True)
    ids = torch.tensor([[1, 2, 3, 4], [3, 2, 1, 0]])
    teacher = torch.randn_like(logits)
    pairs = [(t, s, None, None)]
    base = compose_objective("S3", "C1", logits, ids, _mask(), teacher_logits=teacher).total
    for condition in ("C2", "C3"):
        result = compose_objective("S3", condition, logits, ids, _mask(), teacher_logits=teacher,
                                   attention_pairs=pairs, mse_scale=2.5)
        raw = attention_auxiliary(t, s, _mask(), METHOD_IDS["S3"][condition])
        expected = base + raw * (2.5 if condition == "C3" else 1) / 9
        assert torch.allclose(result.total, expected)
    torch.manual_seed(41)
    tl = [{k: torch.randn(2, 4, 64) for k in ("query", "key", "value")}]
    sl = [{k: torch.randn(2, 4, 64, requires_grad=True) for k in ("query", "key", "value")}]
    result = compose_objective("S3", "C4", logits, ids, _mask(), teacher_logits=teacher,
                               relation_pairs=(tl, sl), rel_scale=1.75)
    raw = relation_auxiliary(tl, sl, _mask())
    assert torch.allclose(result.total, base + raw * 1.75 / 9)
    assert result.method_id == METHOD_IDS["S3"]["C4"]
    assert all(torch.autograd.grad(result.total, sl[0][k], retain_graph=True)[0].abs().sum() > 0
               for k in ("query", "key", "value"))


def test_fp32_finite_difference_on_attention_and_relation():
    ta, sa, t, s = _maps()
    position = (0, 0, 2, 1)
    def attention_at(delta):
        scores = sa.detach().clone()
        scores[position] += delta
        causal = torch.tril(torch.ones(4, 4, dtype=torch.bool))[None, None]
        probs = torch.softmax(scores.masked_fill(~causal, -1e4), -1)
        return attention_auxiliary(t, probs, _mask(), "cosine_soft_jsd_v2")
    analytic = torch.autograd.grad(attention_auxiliary(t, s, _mask(), "cosine_soft_jsd_v2"), sa)[0][position]
    numerical = (attention_at(0.005)-attention_at(-0.005))/0.01
    assert torch.allclose(analytic, numerical, atol=1e-4, rtol=1e-3)
    torch.manual_seed(43)
    tl = [{k: torch.randn(2, 4, 8) for k in ("query", "key", "value")}]
    sl = [{k: torch.randn(2, 4, 8, requires_grad=True) for k in ("query", "key", "value")}]
    analytic = torch.autograd.grad(relation_auxiliary(tl, sl, _mask(), relation_heads=4), sl[0]["value"])[0][0, 2, 1]
    def relation_at(delta):
        copy = {k: v.detach().clone() for k, v in sl[0].items()}
        copy["value"][0, 2, 1] += delta
        return relation_auxiliary(tl, [copy], _mask(), relation_heads=4)
    numerical = (relation_at(0.005)-relation_at(-0.005))/0.01
    assert torch.allclose(analytic, numerical, atol=1e-4, rtol=1e-3)
