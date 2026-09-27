"""Independent manual references for Stage 02 causal probability edits."""

import math

import pytest
import torch

from sinklab.interventions import (
    AttentionIntervention,
    InterventionError,
    apply_attention_intervention,
    mapped_teacher_scope,
    normalize_layer_scope,
)


def _fixture() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    scores = torch.tensor(
        [[[[2.0, -1.0, 0.5], [3.0, 1.0, -2.0], [4.0, 0.0, 2.0]]]],
        dtype=torch.float32,
    )
    support = torch.ones((1, 1, 3, 3), dtype=torch.bool).tril()
    masked = scores.masked_fill(~support, -torch.inf)
    probabilities = torch.softmax(masked, dim=-1)
    return scores, support, probabilities


def _manual_delete_row(scores: torch.Tensor, probabilities: torch.Tensor, strength: float) -> torch.Tensor:
    """Test-only scalar reference, intentionally independent of production code."""

    p0, p1, p2 = (float(value) for value in probabilities)
    denominator = math.exp(float(scores[1])) + math.exp(float(scores[2]))
    r1 = math.exp(float(scores[1])) / denominator
    r2 = math.exp(float(scores[2])) / denominator
    return torch.tensor(
        [(1.0 - strength) * p0, p1 + strength * p0 * r1, p2 + strength * p0 * r2]
    )


def test_delete_and_relocate_match_independent_manual_reference_before_v_aggregation():
    scores, support, probabilities = _fixture()
    value = torch.tensor([[[[2.0], [5.0], [11.0]]]])

    deleted = apply_attention_intervention(
        probabilities, scores, support, AttentionIntervention("delete", strength=0.25)
    )
    expected_row = _manual_delete_row(scores[0, 0, 2], probabilities[0, 0, 2], 0.25)
    torch.testing.assert_close(deleted[0, 0, 2], expected_row, atol=1e-7, rtol=1e-7)

    relocated = apply_attention_intervention(
        probabilities, scores, support, AttentionIntervention("relocate", strength=1.0)
    )
    expected_relocated = probabilities.clone()
    expected_relocated[0, 0, 1:, 1] += probabilities[0, 0, 1:, 0]
    expected_relocated[0, 0, 1:, 0] = 0.0
    torch.testing.assert_close(relocated, expected_relocated, atol=0.0, rtol=0.0)

    # Independent downstream readout makes the causal change analytic all the
    # way to toy logits, rather than stopping at the returned attention tuple.
    unembedding = torch.tensor([[[1.0, -2.0]]])
    clean_context = probabilities @ value
    delete_context = deleted @ value
    relocate_context = relocated @ value
    clean_logits = clean_context @ unembedding
    delete_logits = delete_context @ unembedding
    relocate_logits = relocate_context @ unembedding
    assert not torch.equal(clean_context[:, :, 1:], delete_context[:, :, 1:])
    assert not torch.equal(clean_context[:, :, 1:], relocate_context[:, :, 1:])
    assert not torch.equal(clean_logits[:, :, 1:], delete_logits[:, :, 1:])
    assert not torch.equal(clean_logits[:, :, 1:], relocate_logits[:, :, 1:])


@pytest.mark.parametrize("operation", ["delete", "relocate"])
@pytest.mark.parametrize("strength", [0.0, 0.25, 0.5, 1.0])
def test_intervention_doses_preserve_rows_support_nonnegativity_and_q0(operation, strength):
    scores, support, probabilities = _fixture()
    edited = apply_attention_intervention(
        probabilities, scores, support, AttentionIntervention(operation, strength=strength)
    )
    torch.testing.assert_close(edited.sum(-1), torch.ones_like(edited.sum(-1)), atol=2e-6, rtol=0)
    assert (edited >= 0).all()
    assert (edited.masked_select(~support) == 0).all()
    torch.testing.assert_close(edited[..., 0, :], probabilities[..., 0, :], atol=0, rtol=0)
    if strength == 0.0:
        assert edited is probabilities


def test_delete_uses_stable_conditional_softmax_when_sink_rounds_to_one():
    scores = torch.tensor([[[[1000.0, 0.0, 1.0]]]], dtype=torch.float32)
    probabilities = torch.tensor([[[[1.0, 0.0, 0.0]]]], dtype=torch.float32)
    support = torch.ones_like(probabilities, dtype=torch.bool)
    edited = apply_attention_intervention(
        probabilities, scores, support, AttentionIntervention("delete")
    )
    expected = torch.tensor([0.0, 1.0 / (1.0 + math.e), math.e / (1.0 + math.e)])
    torch.testing.assert_close(edited[0, 0, 0], expected, atol=1e-7, rtol=1e-7)
    assert torch.isfinite(edited).all()


def test_padding_and_length_one_rows_are_explicitly_unchanged():
    scores, support, probabilities = _fixture()
    support[:, :, 2, :] = False  # padded query
    for operation in ("delete", "relocate"):
        edited = apply_attention_intervention(
            probabilities, scores, support, AttentionIntervention(operation)
        )
        torch.testing.assert_close(edited[..., 0, :], probabilities[..., 0, :], atol=0, rtol=0)
        torch.testing.assert_close(edited[..., 2, :], probabilities[..., 2, :], atol=0, rtol=0)

    one_scores = torch.tensor([[[[7.0]]]], dtype=torch.float32)
    one_probabilities = torch.ones_like(one_scores)
    one_support = torch.ones_like(one_scores, dtype=torch.bool)
    for operation in ("delete", "relocate"):
        actual = apply_attention_intervention(
            one_probabilities, one_scores, one_support, AttentionIntervention(operation)
        )
        assert actual.item() == 1.0


def test_key_one_delete_is_supported_as_a_positional_control():
    scores, support, probabilities = _fixture()
    edited = apply_attention_intervention(
        probabilities,
        scores,
        support,
        AttentionIntervention("delete", source_key=1),
    )
    torch.testing.assert_close(edited[..., 0, :], probabilities[..., 0, :], atol=0, rtol=0)
    assert edited[0, 0, 1, 1] == 0
    assert edited[0, 0, 2, 1] == 0
    torch.testing.assert_close(edited.sum(-1), torch.ones_like(edited.sum(-1)), atol=2e-6, rtol=0)


def test_nonfinite_inputs_are_rejected_without_uniform_fallback():
    scores, support, probabilities = _fixture()
    scores[..., 1, 0] = torch.nan
    with pytest.raises(InterventionError, match="finite"):
        apply_attention_intervention(
            probabilities, scores, support, AttentionIntervention("delete")
        )


def test_fp32_row_policy_is_separate_from_bfloat16_rounding():
    scores, support, probabilities = _fixture()
    edited = apply_attention_intervention(
        probabilities, scores, support, AttentionIntervention("delete", strength=0.25)
    )
    fp32_error = (edited.sum(-1) - 1.0).abs().max()
    bf16_roundtrip_error = (edited.to(torch.bfloat16).float().sum(-1) - 1.0).abs().max()
    assert fp32_error <= 2e-6
    assert torch.isfinite(bf16_roundtrip_error)


def test_integer_layer_scopes_and_main_teacher_subset_are_exact():
    teacher_map = [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17,
                   19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35]
    assert normalize_layer_scope(None, 3) == (0, 1, 2)
    assert normalize_layer_scope([2, 0], 3) == (2, 0)
    assert mapped_teacher_scope(range(8, 16), teacher_map) == tuple(teacher_map[8:16])
    with pytest.raises(InterventionError, match="integers"):
        normalize_layer_scope([1.0], 3)
    with pytest.raises(InterventionError, match="duplicate"):
        normalize_layer_scope([1, 1], 3)
    with pytest.raises(InterventionError, match="outside"):
        normalize_layer_scope([3], 3)


def test_intervention_validation_rejects_invalid_values():
    with pytest.raises(InterventionError, match="unknown"):
        AttentionIntervention("zero")  # type: ignore[arg-type]
    with pytest.raises(InterventionError, match=r"\[0, 1\]"):
        AttentionIntervention("delete", strength=1.01)
    with pytest.raises(InterventionError, match="must differ"):
        AttentionIntervention("relocate", source_key=1, destination_key=1)

