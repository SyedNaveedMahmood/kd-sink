# -*- coding: utf-8 -*-
"""test_provenance_streaming.py — the streamed corpus digest equals the one-shot one.

``build_block_dataset`` used to hash the packed corpus with ``prov.sha256_json``, which
materialises the whole object as one JSON string. On the real TinyStories train split that
is 3.57M blocks of 128 tokens: measured on a 50,000-block sample and scaled, an extra
~8.8 GB of peak RSS and ~11.7 minutes on top of the ~20 GB the blocks already hold. That is
what made E6A training unrunnable on a 16 GB host, and it had never been hit because only
the 5-step ``--smoke`` path ever ran.

``prov.sha256_int_rows`` streams the identical byte sequence instead. The **value** must not
move: a run hashed by either helper has to stay comparable under ``05`` §7.1, and a digest
that is "different but equally good" would silently partition the artefacts into two
incomparable sets. So equality is asserted here on random inputs rather than argued in a
docstring.
"""

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import provenance as prov  # noqa: E402


def _rows(n_rows, width, rng, hi=50257):
    return [[rng.randrange(0, hi) for _ in range(width)] for _ in range(n_rows)]


@pytest.mark.parametrize("n_rows,width", [(0, 0), (1, 1), (1, 128), (7, 128), (64, 13)])
def test_streamed_digest_equals_sha256_json(n_rows, width):
    rng = random.Random(n_rows * 1000 + width)
    rows = _rows(n_rows, width, rng)
    assert prov.sha256_int_rows(rows) == prov.sha256_json(rows)


def test_equality_holds_on_the_token_id_range_and_edges():
    """Token ids span 0..vocab, and 0/1/-1 are where a hand-rolled serialiser goes wrong."""
    rows = [[0], [1], [255], [256], [50256], [0, 0, 0], [1, 22, 333, 4444, 55555]]
    assert prov.sha256_int_rows(rows) == prov.sha256_json(rows)


def test_numpy_rows_hash_identically_to_python_lists():
    """The corpus is stored as an int32 array; hashing it must not change the digest.

    This is the property that lets the blocks drop from ~20 GB of Python ints to ~1.8 GB
    of numpy without touching `manifest_sha256`.
    """
    rng = random.Random(0)
    rows = _rows(37, 128, rng)
    array = np.asarray(rows, dtype=np.int32)
    assert prov.sha256_int_rows(array) == prov.sha256_json(rows)
    assert prov.sha256_int_rows(array) == prov.sha256_int_rows(rows)


def test_the_digest_still_separates_different_corpora():
    """A streaming digest that collapsed rows would be fast, wrong, and silent."""
    a = [[1, 2], [3, 4]]
    b = [[1, 2, 3, 4]]
    c = [[1, 2], [4, 3]]
    assert prov.sha256_int_rows(a) != prov.sha256_int_rows(b), "row boundaries must matter"
    assert prov.sha256_int_rows(a) != prov.sha256_int_rows(c), "order within a row matters"


def test_it_accepts_a_generator_and_never_needs_the_whole_corpus():
    """The point of the helper: one row in memory at a time."""
    rng = random.Random(7)
    rows = _rows(50, 128, rng)
    assert prov.sha256_int_rows(iter(rows)) == prov.sha256_json(rows)
