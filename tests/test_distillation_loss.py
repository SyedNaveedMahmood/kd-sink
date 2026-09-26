# -*- coding: utf-8 -*-
"""test_distillation_loss.py — the E6A loss terms (WP5).

``06_TEST_PLAN.md`` row 25. Five properties:

* ``L_KD`` is zero when student and teacher logits are identical;
* ``L_ATTN`` is zero when the mapped maps are identical;
* **masked local positions are excluded, not renormalised** — the one ``06`` §3 calls out
  as carrying unusual weight, because a renormalising implementation optimises a quantity
  that does not exist in the teacher and would test H2 against an artefact;
* ``T**2`` is applied exactly once;
* teacher tensors carry no gradient and no grad reaches teacher parameters.

Pure tensors and one tiny random GPT-Neo pair. No network.
"""

from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "transformation_inheritance"))

import train_distillation as td  # noqa: E402


def _rand_logits(shape, seed):
    return torch.randn(shape, generator=torch.Generator().manual_seed(seed))


def _rows_to_distribution(rows):
    return torch.softmax(torch.tensor(rows, dtype=torch.float32), dim=-1)


# ═══════════════════════════════════════════════════════════════════════════════
# L_KD
# ═══════════════════════════════════════════════════════════════════════════════


def test_kd_is_zero_for_identical_logits():
    logits = _rand_logits((2, 7, 11), seed=1)
    assert td.kd_loss(logits, logits.clone(), T=2.0).item() == pytest.approx(0.0, abs=1e-6)
    assert td.kd_loss(logits, logits.clone(), T=1.0).item() == pytest.approx(0.0, abs=1e-6)


def test_kd_applies_t_squared_exactly_once():
    """``KD(T) / T**2`` must equal the plain KL of the softened distributions.

    Applying ``T**2`` twice (or not at all) changes the loss by a factor of ``T**2`` and
    silently rescales the KD term against CE — the whole D1/D2 contrast.
    """
    student = _rand_logits((3, 5, 9), seed=2)
    teacher = _rand_logits((3, 5, 9), seed=3)
    T = 2.0

    reference = torch.nn.functional.kl_div(
        torch.log_softmax(student / T, dim=-1),
        torch.softmax(teacher / T, dim=-1),
        reduction="none").sum(-1).mean()

    observed = td.kd_loss(student, teacher, T=T)
    assert observed.item() == pytest.approx(float(reference) * T ** 2, rel=1e-5)
    # And the T=1 case must not be rescaled at all.
    assert td.kd_loss(student, teacher, T=1.0).item() == pytest.approx(
        float(torch.nn.functional.kl_div(
            torch.log_softmax(student, dim=-1), torch.softmax(teacher, dim=-1),
            reduction="none").sum(-1).mean()), rel=1e-5)


def test_kd_mask_excludes_positions():
    student = _rand_logits((1, 4, 6), seed=4)
    teacher = _rand_logits((1, 4, 6), seed=5)
    mask = torch.tensor([[1.0, 1.0, 0.0, 0.0]])

    masked = td.kd_loss(student, teacher, T=2.0, mask=mask)
    truncated = td.kd_loss(student[:, :2], teacher[:, :2], T=2.0)
    assert masked.item() == pytest.approx(truncated.item(), rel=1e-6)


def test_kd_does_not_propagate_gradient_into_the_teacher():
    student = _rand_logits((1, 3, 5), seed=6).requires_grad_(True)
    teacher = _rand_logits((1, 3, 5), seed=7).requires_grad_(True)
    td.kd_loss(student, teacher, T=2.0).backward()
    assert student.grad is not None
    assert teacher.grad is None, "teacher logits must be detached inside kd_loss"


# ═══════════════════════════════════════════════════════════════════════════════
# L_ATTN — the masking property
# ═══════════════════════════════════════════════════════════════════════════════


def test_attn_loss_is_zero_for_identical_maps():
    maps = torch.softmax(_rand_logits((2, 4, 6, 6), seed=8), dim=-1)
    student = [maps.clone() for _ in range(8)]
    teacher = [maps.clone() for _ in range(4)]
    masks = {(t, s): torch.ones(6, 6, dtype=torch.bool)
             for t, s in td.DEFAULT_LAYER_MAP.items()}
    loss, breakdown = td.attn_js_loss(student, teacher, td.DEFAULT_LAYER_MAP, masks)
    assert loss.item() == pytest.approx(0.0, abs=1e-9)
    assert breakdown["n_pairs"] == 4


def test_masked_positions_are_excluded_not_renormalised():
    """``06`` §3's decisive construction.

    Teacher and student agree exactly on the **valid** positions of every mapped pair and
    differ on the invalid ones. Exclusion ⇒ the loss is exactly zero. A renormalising
    implementation would rescale the two distributions over the valid support by different
    constants (their masked-out mass differs) and report a non-zero loss.
    """
    seq = 6
    window = 3
    valid = td.local_window_mask(seq, window, torch.device("cpu"))

    agreed = torch.softmax(_rand_logits((1, 2, seq, seq), seed=9), dim=-1)
    teacher_map = agreed.clone()
    student_map = agreed.clone()
    # Differ only where the mask is False.
    noise = torch.rand((1, 2, seq, seq), generator=torch.Generator().manual_seed(10))
    invalid = ~valid
    student_map[..., invalid] = noise[..., invalid]

    layer_map = {0: 0}
    masks = {(0, 0): valid}
    loss, breakdown = td.attn_js_loss([student_map], [teacher_map], layer_map, masks)
    assert loss.item() == pytest.approx(0.0, abs=1e-9), (
        "non-zero loss means invalid positions leaked in, or the distributions were "
        "renormalised over the valid support")
    assert breakdown["per_pair"][0]["n_valid"] == float(valid.sum() * 1 * 2)

    # The construction has teeth: without the mask the same pair scores well above zero.
    unmasked = td.attn_js_loss([student_map], [teacher_map], layer_map,
                               {(0, 0): torch.ones(seq, seq, dtype=torch.bool)})[0]
    assert unmasked.item() > 1e-3


def test_attn_js_sums_over_keys_then_averages_queries():
    """``03`` §1.5 reduces over layer, head and query — **never** over key.

    An analytic construction with a known answer. Query 0 compares ``(1, 0)`` against
    ``(0, 1)``: two distributions with disjoint support, whose JSD is exactly ``ln 2``.
    Query 1 compares ``(0.5, 0.5)`` against itself, contributing 0. Averaged over the two
    query rows the loss is ``ln 2 / 2 = 0.3465736``.

    The pilot's implementation divided by the four valid ``(query, key)`` cells instead,
    giving ``ln 2 / 4 = 0.1732868`` — the same quantity scaled down by the mean number of
    valid keys per query. That is not a tolerance question: the two differ by a factor of
    exactly 2 here and 64.5 on a 128-token causal batch.
    """
    teacher = torch.tensor([[[[1.0, 0.0], [0.5, 0.5]]]])
    student = torch.tensor([[[[0.0, 1.0], [0.5, 0.5]]]])
    masks = {(0, 0): torch.ones(2, 2, dtype=torch.bool)}

    loss, breakdown = td.attn_js_loss([student], [teacher], {0: 0}, masks)
    assert loss.item() == pytest.approx(math.log(2.0) / 2.0, rel=1e-6)
    assert loss.item() != pytest.approx(math.log(2.0) / 4.0, rel=1e-3)
    assert breakdown["per_pair"][0]["n_query_rows"] == 2.0
    assert breakdown["per_pair"][0]["n_valid"] == 4.0


def test_the_denominator_is_query_rows_not_cells():
    """The two counts are both reported, and the loss is the one divided by query rows.

    An artefact that records only the cell count cannot be checked against the objective the
    run actually optimised, which is how the pilot's 64.5x under-scaling went unnoticed.
    """
    seq = 8
    teacher = torch.softmax(_rand_logits((2, 3, seq, seq), seed=41), dim=-1)
    student = torch.softmax(_rand_logits((2, 3, seq, seq), seed=42), dim=-1)
    mask = td.causal_mask(seq, torch.device("cpu"))

    loss, breakdown = td.attn_js_loss([student], [teacher], {0: 0}, {(0, 0): mask})
    pair = breakdown["per_pair"][0]
    assert pair["n_query_rows"] == float(2 * 3 * seq)
    assert pair["n_valid"] == float(2 * 3 * int(mask.sum()))

    # Recompute from the definition: sum over keys, then mean over query rows.
    m = 0.5 * (teacher + student)
    cells = 0.5 * (teacher * ((teacher + 1e-12).log() - (m + 1e-12).log())
                   + student * ((student + 1e-12).log() - (m + 1e-12).log()))
    expected = (cells * mask).sum(dim=-1).mean()
    assert loss.item() == pytest.approx(float(expected), rel=1e-6)


def test_a_query_row_with_no_valid_key_is_not_counted():
    """An all-masked query contributes nothing *and* does not inflate the denominator.

    Counting it would divide a correct numerator by too many rows — a scale error of exactly
    the kind this reduction was fixed for.
    """
    seq = 4
    teacher = torch.softmax(_rand_logits((1, 1, seq, seq), seed=43), dim=-1)
    student = torch.softmax(_rand_logits((1, 1, seq, seq), seed=44), dim=-1)
    full = td.causal_mask(seq, torch.device("cpu"))
    holed = full.clone()
    holed[1, :] = False                                # query 1 has no valid key at all

    _, breakdown = td.attn_js_loss([student], [teacher], {0: 0}, {(0, 0): holed})
    assert breakdown["per_pair"][0]["n_query_rows"] == float(seq - 1)


def test_the_pilot_reduction_was_lower_by_the_mean_valid_key_count():
    """Pins the size of the correction: (S + 1) / 2 on a full causal mask, 64.5 at S = 128.

    Recorded as a test rather than only as prose because the first E6A pilot trained D2 on
    the smaller quantity, and the two runs must never be pooled (CLAUDE.md trap 22).
    """
    seq = 128
    teacher = torch.softmax(_rand_logits((1, 2, seq, seq), seed=45), dim=-1)
    student = torch.softmax(_rand_logits((1, 2, seq, seq), seed=46), dim=-1)
    mask = td.causal_mask(seq, torch.device("cpu"))

    loss, breakdown = td.attn_js_loss([student], [teacher], {0: 0}, {(0, 0): mask})
    pair = breakdown["per_pair"][0]
    ratio = pair["n_valid"] / pair["n_query_rows"]
    assert ratio == pytest.approx((seq + 1) / 2.0)     # 64.5

    # What the pilot's per-cell reduction would have reported for the same pair.
    per_cell = loss.item() * pair["n_query_rows"] / pair["n_valid"]
    assert per_cell == pytest.approx(loss.item() / 64.5, rel=1e-6)


def test_the_recorded_reduction_names_the_axes_it_reduces_over():
    """``run_config.json`` must say which reduction a run used.

    Without it a pre-fix run directory and a corrected one are indistinguishable, and the
    only difference between them is a 64.5x change in the weight of a pre-registered loss
    term.
    """
    assert td.ATTN_REDUCTION == "sum_over_keys_mean_over_queries"
    source = (REPO / "transformation_inheritance"
              / "train_distillation.py").read_text(encoding="utf-8")
    assert '"attn_reduction": ATTN_REDUCTION' in source, (
        "write_run_config no longer records the reduction; a corrected run would be "
        "indistinguishable from the under-scaled pilot")


def test_valid_mask_is_the_intersection_of_both_layer_types():
    """A global teacher layer mapped to a local student layer uses the intersection."""
    seq = 8
    window = 3

    class _Cfg:
        def __init__(self, types, window_size):
            self.attention_layers = types
            self.window_size = window_size

    teacher_cfg = _Cfg(["global", "local"], seq)
    student_cfg = _Cfg(["local", "global"], window)
    masks, types = td.build_valid_masks(teacher_cfg, student_cfg, {0: 0, 1: 1}, seq,
                                        torch.device("cpu"))

    assert types[(0, 0)] == "mixed"
    assert types[(1, 1)] == "mixed"
    # global (causal) AND local(window=3) == local(window=3)
    assert torch.equal(masks[(0, 0)], td.local_window_mask(seq, window,
                                                           torch.device("cpu")))
    # A pair of equal types is tagged as such and is not "mixed".
    same = _Cfg(["global", "global"], seq)
    _m, same_types = td.build_valid_masks(same, same, {0: 0}, seq, torch.device("cpu"))
    assert same_types[(0, 0)] == "global_global"


def test_attention_types_are_read_from_the_config_not_assumed():
    class _Compressed:
        attention_layers = None
        attention_types = [[["global", "local"], 2]]

    assert td.attention_types(_Compressed()) == ["global", "local", "global", "local"]

    class _Plain:
        attention_layers = ["global", "global", "local"]

    assert td.attention_types(_Plain()) == ["global", "global", "local"]

    class _NoConcept:
        attention_layers = None
        attention_types = None
        num_hidden_layers = 3

    assert td.attention_types(_NoConcept()) == ["global"] * 3


def test_attn_loss_reports_pairs_separately(monkeypatch):
    """Design §16.1: global-mapped and local-mapped pairs must be reportable apart."""
    maps_t = [torch.softmax(_rand_logits((1, 2, 5, 5), seed=11 + i), dim=-1)
              for i in range(2)]
    maps_s = [torch.softmax(_rand_logits((1, 2, 5, 5), seed=21 + i), dim=-1)
              for i in range(2)]
    masks = {(0, 0): torch.ones(5, 5, dtype=torch.bool),
             (1, 1): torch.ones(5, 5, dtype=torch.bool)}
    types = {(0, 0): "global_global", (1, 1): "mixed"}
    loss, breakdown = td.attn_js_loss(maps_s, maps_t, {0: 0, 1: 1}, masks, types)

    assert breakdown["n_mixed"] == 1
    assert breakdown["global_global"] is not None
    assert breakdown["local_local"] is None            # no such pair -> None, not NaN
    assert loss.item() == pytest.approx(
        0.5 * (breakdown["per_pair"][0]["jsd"] + breakdown["per_pair"][1]["jsd"]),
        rel=1e-6)


def test_attn_loss_refuses_mismatched_shapes():
    student = [torch.softmax(_rand_logits((1, 4, 5, 5), seed=30), dim=-1)]
    teacher = [torch.softmax(_rand_logits((1, 2, 5, 5), seed=31), dim=-1)]
    with pytest.raises(ValueError, match="equal head counts"):
        td.attn_js_loss(student, teacher, {0: 0},
                        {(0, 0): torch.ones(5, 5, dtype=torch.bool)})


def test_mean_head_alignment_supports_unequal_heads_without_changing_the_jsd():
    """The large->medium opt-in compares layer means, never truncated head pairs."""
    seq = 5
    teacher = torch.softmax(_rand_logits((1, 20, seq, seq), seed=130), dim=-1)
    student_logits = _rand_logits((1, 16, seq, seq), seed=131).requires_grad_()
    student = torch.softmax(student_logits, dim=-1)
    mask = {(0, 0): td.causal_mask(seq, torch.device("cpu"))}

    actual, breakdown = td.attn_js_loss(
        [student], [teacher], {0: 0}, mask, head_alignment="mean")
    expected, _ = td.attn_js_loss(
        [student.mean(dim=1, keepdim=True)],
        [teacher.mean(dim=1, keepdim=True)], {0: 0}, mask)

    assert actual.item() == pytest.approx(expected.item(), rel=1e-7, abs=1e-9)
    assert breakdown["head_alignment"] == "mean"
    assert breakdown["per_pair"][0]["n_query_rows"] == float(seq)
    actual.backward()
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()


def test_attn_loss_refuses_an_unknown_head_alignment():
    maps = [torch.ones(1, 1, 2, 2) / 2]
    with pytest.raises(ValueError, match="unknown attention head alignment"):
        td.attn_js_loss(maps, maps, {0: 0},
                        {(0, 0): torch.ones(2, 2, dtype=torch.bool)},
                        head_alignment="truncate")
def test_amad_jsd_accepts_the_registered_16_to_12_geometry_and_is_bounded():
    """The new method solves only the head-axis mismatch and retains bounded JSD."""
    student = [torch.softmax(_rand_logits((2, 12, 5, 5), seed=130), dim=-1)]
    teacher = [torch.softmax(_rand_logits((2, 16, 5, 5), seed=131), dim=-1)]
    loss, breakdown = td.amad_js_attn_loss(
        student, teacher, {0: 0},
        {(0, 0): torch.tril(torch.ones(5, 5, dtype=torch.bool))},
        {(0, 0): "global_global"})

    assert 0.0 <= loss.item() <= math.log(2.0) + 1e-6
    assert breakdown["alignment_method"] == td.AMAD_JSD_HEAD_ALIGNMENT
    pair = breakdown["per_pair"][0]
    assert (pair["teacher_heads"], pair["student_heads"]) == (16, 12)
    assert 0.0 < pair["alignment_mean_max_weight"] < 1.0
    assert pair["alignment_entropy_nats"] > 0.0


def test_amad_jsd_is_invariant_to_student_head_permutation():
    """Head order has no semantics; changing it must not change the aligned objective."""
    student_map = torch.softmax(_rand_logits((1, 5, 4, 4), seed=132), dim=-1)
    teacher = [torch.softmax(_rand_logits((1, 7, 4, 4), seed=133), dim=-1)]
    mask = {(0, 0): torch.ones(4, 4, dtype=torch.bool)}
    first, _ = td.amad_js_attn_loss([student_map], teacher, {0: 0}, mask)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    second, _ = td.amad_js_attn_loss(
        [student_map[:, permutation]], teacher, {0: 0}, mask)
    assert second.item() == pytest.approx(first.item(), rel=1e-6, abs=1e-7)


def test_amad_jsd_reduces_to_original_jsd_when_there_is_one_student_head():
    """With one possible student head the AMAD softmax is exactly one."""
    student_map = torch.softmax(_rand_logits((1, 1, 4, 4), seed=134), dim=-1)
    teacher_map = torch.softmax(_rand_logits((1, 3, 4, 4), seed=135), dim=-1)
    mask = {(0, 0): torch.tril(torch.ones(4, 4, dtype=torch.bool))}
    aligned, _ = td.amad_js_attn_loss(
        [student_map], [teacher_map], {0: 0}, mask)
    legacy, _ = td.attn_js_loss(
        [student_map.expand(-1, 3, -1, -1)], [teacher_map], {0: 0}, mask)
    assert aligned.item() == pytest.approx(legacy.item(), rel=1e-6, abs=1e-7)


def test_amad_jsd_keeps_alignment_weights_attached_and_reaches_every_student_head():
    student_logits = _rand_logits((1, 4, 4, 4), seed=136).requires_grad_(True)
    student = [torch.softmax(student_logits, dim=-1)]
    teacher = [torch.softmax(_rand_logits((1, 6, 4, 4), seed=137), dim=-1)]
    loss, _ = td.amad_js_attn_loss(
        student, teacher, {0: 0},
        {(0, 0): torch.ones(4, 4, dtype=torch.bool)})
    loss.backward()
    per_head_gradient = student_logits.grad.abs().sum(dim=(0, 2, 3))
    assert torch.all(per_head_gradient > 0), per_head_gradient


def test_amad_jsd_refuses_non_head_shape_mismatches():
    student = [torch.softmax(_rand_logits((1, 12, 4, 4), seed=138), dim=-1)]
    teacher = [torch.softmax(_rand_logits((1, 16, 5, 5), seed=139), dim=-1)]
    with pytest.raises(ValueError, match="aligns heads only"):
        td.amad_js_attn_loss(
            student, teacher, {0: 0},
            {(0, 0): torch.ones(5, 5, dtype=torch.bool)})


# ═══════════════════════════════════════════════════════════════════════════════
# L_CE and the schedule
# ═══════════════════════════════════════════════════════════════════════════════


def test_ce_matches_a_manual_shifted_cross_entropy():
    logits = _rand_logits((2, 5, 7), seed=12)
    labels = torch.randint(0, 7, (2, 5), generator=torch.Generator().manual_seed(13))
    expected = torch.nn.functional.cross_entropy(
        logits[:, :-1, :].reshape(-1, 7), labels[:, 1:].reshape(-1))
    assert td.ce_loss(logits, labels).item() == pytest.approx(float(expected), rel=1e-6)


def test_lr_schedule_warms_up_then_decays_to_the_floor():
    warmup, max_steps, floor = 10, 100, 0.10
    assert td.lr_lambda(0, warmup, max_steps, floor) == pytest.approx(0.1)
    assert td.lr_lambda(warmup - 1, warmup, max_steps, floor) == pytest.approx(1.0)
    assert td.lr_lambda(warmup, warmup, max_steps, floor) == pytest.approx(1.0)
    assert td.lr_lambda(max_steps, warmup, max_steps, floor) == pytest.approx(floor)
    mid = td.lr_lambda((warmup + max_steps) // 2, warmup, max_steps, floor)
    assert floor < mid < 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# End to end: the teacher is frozen
# ═══════════════════════════════════════════════════════════════════════════════


def test_no_gradient_reaches_the_teacher_in_a_real_step():
    with tempfile.TemporaryDirectory(prefix="e6a_loss_", ignore_cleanup_errors=True) as t:
        root = Path(t)
        setup = td.prepare_run(
            {"condition": "D2",
             "loss": {"ce_weight": 0.45, "kd_weight": 0.45, "attn_weight": 0.10,
                      "temperature": 2.0, "layer_map": td.DEFAULT_LAYER_MAP},
             "data": {"block_size": 32},
             "optim": {"max_steps": 1}},
            seed=0, output_dir=root, max_steps=1, smoke=True)

        assert all(not p.requires_grad for p in setup.teacher.parameters())
        assert not setup.teacher.training

        input_ids = setup.train_data.batch([0, 1], setup.device)
        loss, components = td.compute_losses(setup, input_ids, need_attention=True)
        loss.backward()

        assert all(p.grad is None for p in setup.teacher.parameters()), \
            "gradient reached the frozen teacher"
        assert any(p.grad is not None for p in setup.student.parameters())
        # Every component is logged even when its weight is zero (design §8.6).
        for key in ("l_ce", "l_kd", "l_attn"):
            assert key in components and not math.isnan(components[key])
