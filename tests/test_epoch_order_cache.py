# -*- coding: utf-8 -*-
"""test_epoch_order_cache.py — the memoised block order IS the block order.

``epoch_order`` is a pure function of ``(n_blocks, seed, epoch)`` that the training loop
used to call once per gradient-accumulation micro-batch, then read ``batch_size`` elements
of and discard. At E6A scale that is a Python shuffle over ~3.1M elements — **measured
1.09 s** — repeated four times per optimiser step and 40,000 times per 10,000-step run,
single-threaded, with the GPU idle. It was the reason the E6A-GPT2 runs showed 12% GPU
utilisation on a 4090 and 3% on a 5090 (CLAUDE.md trap 27).

``epoch_order_array`` caches it. The whole safety of that rests on one claim — that it
returns the same integers in the same order — so this file asserts exactly that, against
the untouched original, plus the two things a shared cached array can get wrong: handing
out something mutable, and confusing two keys.

These are invariant tests, not value tests: none of them asserts a *particular* shuffle,
only that the two implementations agree.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_distillation as td  # noqa: E402

# Small enough to run fast, varied enough that an off-by-one in the key would show.
CASES = [
    (1, 0, 0), (2, 0, 0), (7, 0, 0), (64, 0, 0), (997, 0, 0), (5_000, 0, 0),
    (997, 1, 0), (997, 2, 0), (997, 0, 1), (997, 0, 2), (997, 1, 1), (997, 2, 3),
]


@pytest.mark.parametrize("n_blocks,seed,epoch", CASES)
def test_cached_array_equals_the_original_function(n_blocks, seed, epoch):
    """The load-bearing assertion: same integers, same order, same length."""
    reference = td.epoch_order(n_blocks, seed, epoch)
    cached = td.epoch_order_array(n_blocks, seed, epoch)
    assert cached.dtype == np.int64
    assert cached.tolist() == reference


def test_it_is_a_permutation_not_merely_the_same_length():
    """A cache that returned `arange` would pass a length check and destroy the shuffle."""
    n = 5_000
    cached = td.epoch_order_array(n, 0, 0)
    assert sorted(cached.tolist()) == list(range(n))
    # ...and it is genuinely shuffled, or the whole seeded-order property is vacuous.
    assert cached.tolist() != list(range(n))


def test_different_keys_do_not_collide():
    """seed and epoch must both be part of the key, not just n_blocks."""
    base = td.epoch_order_array(997, 0, 0).tolist()
    assert td.epoch_order_array(997, 1, 0).tolist() != base
    assert td.epoch_order_array(997, 0, 1).tolist() != base


def test_repeated_calls_return_the_identical_object():
    """The point of the cache. `is`, not `==`: a fresh 3.1M-element shuffle per call is
    exactly the defect being fixed."""
    first = td.epoch_order_array(5_000, 0, 0)
    second = td.epoch_order_array(5_000, 0, 0)
    assert first is second


def test_the_cached_array_cannot_be_mutated_by_a_caller():
    """An lru_cache handing out a shared *mutable* sequence is a defect waiting to happen:
    one caller sorting it in place would silently change the data order of every later
    step of every later run in the process."""
    cached = td.epoch_order_array(997, 0, 0)
    assert cached.flags.writeable is False
    with pytest.raises(ValueError):
        cached[0] = 12345


def test_the_batch_indices_match_the_uncached_expression():
    """The exact expression the training loop uses, against the exact one it replaced.

    The loop reads a `batch_size` window at a rotating offset with wraparound; this pins
    the numpy form against the original list comprehension for offsets that do and do not
    wrap, because wraparound is where an index expression usually breaks.
    """
    n_blocks, batch_size = 997, 16
    reference = td.epoch_order(n_blocks, 0, 0)
    cached = td.epoch_order_array(n_blocks, 0, 0)
    for offset in (0, 1, 500, n_blocks - 1, n_blocks - batch_size, n_blocks - 3):
        expected = [reference[(offset + k) % n_blocks] for k in range(batch_size)]
        actual = cached[(offset + np.arange(batch_size)) % n_blocks].tolist()
        assert actual == expected, f"offset {offset} diverges"
