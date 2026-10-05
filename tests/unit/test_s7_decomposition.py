import math

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.metrics import MetricError, attention_jsd_decomposition
from sinklab.models import GPT2Adapter
from sinklab.s7_utility import S7Error, evaluate_clean_decomposition


def _tensor(row, heads=1):
    values = torch.zeros((1, heads, 3, 3), dtype=torch.float64)
    values[:, :, 0, 0] = 1
    values[:, :, 1, :2] = torch.tensor([.5, .5])
    values[:, :, 2] = torch.tensor(row)
    return values


def test_exact_mass_shape_and_column_closure_with_unequal_residuals():
    t = _tensor([.7, .2, .1])
    s = _tensor([.2, .3, .5])
    result = attention_jsd_decomposition(t, s, torch.ones((1, 3), dtype=torch.bool))
    assert result["full_jsd_nats"] == pytest.approx(result["mass_jsd_nats"] + result["shape_jsd_nats"], abs=1e-13)
    assert result["full_jsd_nats"] == pytest.approx(result["key0_jsd_nats"] + result["other_columns_jsd_nats"], abs=1e-13)
    assert result["mass_jsd_nats"] > 0 and result["shape_jsd_nats"] > 0
    assert result["conditional_jsd_nats"] is not None
    assert result["shape_jsd_nats"] != pytest.approx(result["conditional_jsd_nats"])
    assert result["valid_query_count"] == 2
    assert result["max_closure_error_nats"] < 1e-13


def test_identical_maps_zero_and_distinct_native_head_counts():
    t = _tensor([.4, .3, .3], heads=2)
    s = _tensor([.4, .3, .3], heads=3)
    result = attention_jsd_decomposition(t, s, torch.ones((1, 3), dtype=torch.bool))
    for field in ("full_jsd_nats", "mass_jsd_nats", "shape_jsd_nats", "conditional_jsd_nats"):
        assert result[field] == pytest.approx(0, abs=1e-14)


def test_one_pure_sink_preserves_undefined_conditional():
    t = _tensor([1., 0., 0.])
    s = _tensor([.5, .3, .2])
    # q1 has non-sink mass on both sides to isolate q2's undefined case.
    t[:, :, 1, :2] = torch.tensor([.5, .5])
    s[:, :, 1, :2] = torch.tensor([.5, .5])
    result = attention_jsd_decomposition(t, s, torch.ones((1, 3), dtype=torch.bool))
    assert result["conditional_jsd_nats"] is None
    assert result["conditional_valid_query_count"] == 1
    assert result["full_jsd_nats"] == pytest.approx(result["mass_jsd_nats"] + result["shape_jsd_nats"], abs=1e-13)
    assert result["shape_jsd_nats"] == pytest.approx(0, abs=1e-14)


def test_independent_bernoulli_reference_and_equal_mass_scaling():
    mask = torch.ones((1, 3), dtype=torch.bool)
    teacher = _tensor([.8, .1, .1])
    student = _tensor([.2, .4, .4])
    result = attention_jsd_decomposition(teacher, student, mask)
    def kl_binary(p, m):
        return p * math.log(p / m) + (1 - p) * math.log((1 - p) / (1 - m))
    reference = (kl_binary(.8, .5) + kl_binary(.2, .5)) / 4  # q1 is identical
    assert result["full_jsd_nats"] == pytest.approx(reference, abs=1e-13)
    assert result["mass_jsd_nats"] == pytest.approx(reference, abs=1e-13)
    assert result["shape_jsd_nats"] == pytest.approx(0, abs=1e-13)
    assert result["conditional_jsd_nats"] == pytest.approx(0, abs=1e-13)
    assert result["key0_jsd_nats"] != pytest.approx(result["mass_jsd_nats"])

    student = _tensor([.8, .2, 0.])
    result = attention_jsd_decomposition(teacher, student, mask)
    # q2 conditional comparison is JSD([.5,.5], [1,0]), weighted by 0.2;
    # q1 is identical. This gives a known independent equal-mass identity.
    conditional_q2 = .5 * (.5 * math.log(.5 / .75) + .5 * math.log(.5 / .25) +
                            math.log(1 / .75))
    assert result["mass_jsd_nats"] == pytest.approx(0, abs=1e-13)
    assert result["conditional_jsd_nats"] == pytest.approx(conditional_q2 / 2, abs=1e-13)
    assert result["full_jsd_nats"] == pytest.approx(.2 * conditional_q2 / 2, abs=1e-13)


def test_head_order_and_right_padding_do_not_change_valid_rows():
    teacher = torch.cat([_tensor([.7, .2, .1]), _tensor([.3, .2, .5])], dim=1)
    student = torch.cat([_tensor([.4, .4, .2]), _tensor([.2, .3, .5])], dim=1)
    mask = torch.ones((1, 3), dtype=torch.bool)
    baseline = attention_jsd_decomposition(teacher, student, mask)
    shuffled = attention_jsd_decomposition(teacher[:, [1, 0]], student[:, [1, 0]], mask)
    assert shuffled["full_jsd_nats"] == pytest.approx(baseline["full_jsd_nats"], abs=1e-14)
    padded_t = torch.zeros((1, 2, 4, 4), dtype=torch.float64)
    padded_s = torch.zeros_like(padded_t)
    padded_t[:, :, :3, :3] = teacher
    padded_s[:, :, :3, :3] = student
    padded = attention_jsd_decomposition(padded_t, padded_s,
                                          torch.tensor([[1, 1, 1, 0]], dtype=torch.bool))
    assert padded["full_jsd_nats"] == pytest.approx(baseline["full_jsd_nats"], abs=1e-14)


@pytest.mark.parametrize("mutation", ["negative", "future", "unnormalized", "mask", "nan"])
def test_bad_attention_fails_closed(mutation):
    t, s = _tensor([.5, .3, .2]), _tensor([.5, .3, .2])
    mask = torch.ones((1, 3), dtype=torch.bool)
    if mutation == "negative":
        s[0, 0, 2, 0] = -.1
    elif mutation == "future":
        s[0, 0, 1, 2] = .1
    elif mutation == "unnormalized":
        s[0, 0, 2, 0] = .7
    elif mutation == "mask":
        mask[0, 1] = False
    else:
        s[0, 0, 2, 0] = math.nan
    with pytest.raises(MetricError):
        attention_jsd_decomposition(t, s, mask)


def test_clean_supplement_is_rng_neutral_and_exactly_mapped():
    config = GPT2Config(vocab_size=23, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=2, n_head=4, _attn_implementation="eager")
    teacher = GPT2Adapter(GPT2LMHeadModel(config))
    student = GPT2Adapter(GPT2LMHeadModel(config))
    teacher.model.train()
    student.model.train()
    state = torch.get_rng_state().clone()
    items = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
             {"id": "b", "input_ids": [4, 3, 2, 1], "attention_mask": [1, 1, 1, 1]}]
    result = evaluate_clean_decomposition(adapter=student, teacher_adapter=teacher,
                                          teacher_map=[1, 0], items=items)
    assert result["status"] == "complete" and result["item_count"] == 2
    assert result["items"][0]["layers"][0]["teacher_layer"] == 1
    assert result["aggregate"]["full_jsd_nats"] == pytest.approx(
        result["aggregate"]["mass_jsd_nats"] + result["aggregate"]["shape_jsd_nats"], abs=1e-8)
    assert teacher.model.training and student.model.training
    assert torch.equal(torch.get_rng_state(), state)
    with pytest.raises(S7Error, match="scope"):
        evaluate_clean_decomposition(adapter=student, teacher_adapter=teacher,
                                     teacher_map=[0, 0], items=items)
