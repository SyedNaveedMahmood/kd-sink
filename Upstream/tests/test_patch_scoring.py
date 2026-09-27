# -*- coding: utf-8 -*-
"""test_patch_scoring.py — the label scorer of ``02_MODULE_SPEC_common.md`` §6.3.

WP7. The scorer is length-normalised sequence log-probability over the candidate's full
token sequence, computed by teacher forcing. It is implemented **once** and used for the
baseline and the patched forward alike, so a scoring bug cannot differentially affect them;
these tests pin it against hand-computed values on a toy vocabulary and check the margin's
sign convention.

Model-free: only numpy/torch, no checkpoints, no network.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))

import cross_example_patching as cep  # noqa: E402

# The scorer computes log-softmax in fp32 (``logits.float()``), matching the repo's
# capture convention, while the hand computation below is fp64. 1e-6 is the agreement
# fp32 can deliver; a tighter bound would be testing float64 arithmetic that the scorer
# deliberately does not do.
TOL = 1e-6


def _handmade_logits() -> torch.Tensor:
    """A ``[seq=5, vocab=4]`` logits tensor whose log-softmax is exactly computable.

    Every row is a one-hot-ish pattern ``[a, 0, 0, 0]`` so ``log softmax`` at index 0 is
    ``a - log(e^a + 3)`` and at any other index is ``-log(e^a + 3)``.
    """
    logits = torch.zeros(5, 4)
    logits[:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    return logits


def _row_logprob(a: float, index: int) -> float:
    denom = math.log(math.exp(a) + 3.0)
    return (a - denom) if index == 0 else -denom


def test_sequence_logprob_matches_a_hand_computation():
    """Two candidate tokens scored from positions ``prompt_len-1`` and ``prompt_len``."""
    logits = _handmade_logits()
    prompt_len = 3                      # candidate is predicted from rows 2 and 3
    candidate = [0, 2]

    total, normalised = cep.sequence_logprob(logits, prompt_len, candidate)
    expected = _row_logprob(3.0, 0) + _row_logprob(4.0, 2)
    assert abs(total - expected) < TOL
    assert abs(normalised - expected / 2) < TOL


def test_single_token_candidate_normalises_to_itself():
    logits = _handmade_logits()
    total, normalised = cep.sequence_logprob(logits, prompt_len=1, candidate_ids=[1])
    assert abs(total - _row_logprob(1.0, 1)) < TOL
    assert abs(total - normalised) < TOL


def test_length_normalisation_is_the_mean_not_the_sum():
    """A long low-probability candidate must not be penalised purely for being long."""
    logits = torch.zeros(6, 4)
    logits[:, 0] = 5.0                       # every row strongly prefers token 0
    short_total, short_norm = cep.sequence_logprob(logits, 2, [0])
    long_total, long_norm = cep.sequence_logprob(logits, 2, [0, 0, 0])
    assert long_total < short_total                      # sums accumulate
    assert abs(long_norm - short_norm) < 1e-6            # means do not


def test_scorer_rejects_impossible_requests():
    logits = _handmade_logits()
    with pytest.raises(ValueError, match="prompt_len"):
        cep.sequence_logprob(logits, 0, [1])
    with pytest.raises(ValueError, match="non-empty"):
        cep.sequence_logprob(logits, 2, [])
    with pytest.raises(ValueError, match="positions"):
        cep.sequence_logprob(logits, 4, [1, 2, 3])       # needs 7 rows, has 5
    with pytest.raises(ValueError, match=r"\[seq, vocab\]"):
        cep.sequence_logprob(logits.unsqueeze(0), 2, [1])


def test_margin_sign_convention():
    """``margin = logP(gold) - max_{non-gold} logP``: positive iff gold is preferred."""
    scores = [-0.5, -1.5, -3.0]
    assert cep.correct_label_margin(scores, 0) == pytest.approx(1.0)     # gold wins
    assert cep.correct_label_margin(scores, 1) == pytest.approx(-1.0)    # gold loses
    assert cep.correct_label_margin(scores, 2) == pytest.approx(-2.5)
    # A tie gives exactly zero, not a nudge in either direction.
    assert cep.correct_label_margin([-1.0, -1.0], 0) == pytest.approx(0.0)


def test_margin_is_nan_with_a_single_candidate():
    """One candidate makes the margin undefined; nan is written, never a fabricated 0."""
    assert math.isnan(cep.correct_label_margin([-1.0], 0))
    with pytest.raises(IndexError):
        cep.correct_label_margin([-1.0, -2.0], 5)


def test_jsd_is_zero_for_identical_distributions_and_bounded_by_log2():
    a = torch.tensor([2.0, 1.0, 0.0, -1.0])
    b = torch.tensor([-1.0, 0.0, 1.0, 2.0])
    assert cep.jensen_shannon_divergence(a, a) == pytest.approx(0.0, abs=1e-9)
    assert cep.jensen_shannon_divergence(a, b) > 0.0
    assert cep.jensen_shannon_divergence(a, b) == pytest.approx(
        cep.jensen_shannon_divergence(b, a))

    # Two disjoint point masses are the maximum: log 2 nats.
    far_a = torch.tensor([50.0, -50.0])
    far_b = torch.tensor([-50.0, 50.0])
    assert cep.jensen_shannon_divergence(far_a, far_b) == pytest.approx(
        math.log(2.0), abs=1e-6)


def test_the_same_scorer_serves_baseline_and_patched():
    """A guard against the failure mode of §6.3: two scorers that could diverge.

    ``run_patched`` reaches the scorer through exactly one private helper. If a future
    change adds a second scoring implementation, this test's assumption is what breaks.
    """
    import inspect

    source = inspect.getsource(cep)
    assert source.count("def sequence_logprob(") == 1
    scoring_calls = source.count("sequence_logprob(")
    # one definition + one call inside _score_candidate_set + the docstring-free helper use
    assert scoring_calls >= 2
    assert source.count("def _score_candidate_set(") == 1
