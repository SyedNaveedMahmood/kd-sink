# -*- coding: utf-8 -*-
"""wp8_fake_data.py — offline FLORES/XNLI fixtures for the WP8 manifest tests.

Not a test module. Provides a deterministic fake ``load_dataset`` that the WP8 tests
monkeypatch into ``datasets_loader`` — the same pattern ``test_tinystories_packing.py``
and ``test_sst2_prompt_format.py`` already use, so no test needs the network.

The XNLI fixture is deliberately hostile: each language's shard is emitted in a
**different order**, with an explicit ``promptID``. A positional join therefore mismatches
premises and gold labels immediately, which is exactly what
``test_parallel_manifest_alignment.py`` is there to catch.
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional, Sequence

from tokenizers import Tokenizer, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

FLORES_LANGS = ("eng_Latn", "ben_Beng", "zho_Hans", "arb_Arab",
                "deu_Latn", "hin_Deva", "swh_Latn", "tur_Latn")
XNLI_LANGS = ("en", "zh", "ar", "de", "hi", "sw", "tr")
XNLI_LABELS = (0, 1, 2)


#: Wide enough that every fixture sentence can carry a unique leading token — the fixture
#: sentences must be distinct, or a test that uses text as a proxy for identity (e.g. the
#: partition-disjointness check) fails on fixture collisions rather than on real defects.
FAKE_VOCAB_SIZE = 24000
_UID_SPACE = 16000      # leading-token ids live in [0, _UID_SPACE); body tokens above it
_FIELD_STRIDE = 8000    # premise/hypothesis occupy disjoint halves of the uid space


def tiny_tokenizer(path, vocab_size: int = FAKE_VOCAB_SIZE):
    """A whitespace WordLevel tokenizer, mirroring ``nnsight_smoke_utils._tiny_tokenizer``.

    Word-level rather than byte-level so a fixture sentence's token count is exactly its
    word count — which makes the length filters in the tests readable.
    """
    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2}
    for i in range(vocab_size - 3):
        vocab[f"t{i}"] = i + 3
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="[UNK]", pad_token="[PAD]",
        eos_token="[EOS]", bos_token="[EOS]")
    tokenizer.name_or_path = "wp8_fake_tokenizer"
    if path is not None:
        tokenizer.save_pretrained(str(path))
    return tokenizer


class _FakeSplit(list):
    """A list of dicts that quacks like a ``datasets.Dataset`` for our loaders' needs."""


def _uid(row_index: int, lang_index: int, field: int = 0) -> int:
    """A unique id per (row, language, field), inside the leading-token space."""
    uid = (row_index * 8 + lang_index) + field * _FIELD_STRIDE
    if uid >= _UID_SPACE:
        raise ValueError(f"fixture too large for the fake vocabulary: uid {uid}")
    return uid


def _sentence(n_words: int, salt: int) -> str:
    """A sentence of exactly ``n_words`` word-level tokens, unique for a unique ``salt``.

    The first token encodes ``salt`` directly, so two sentences are equal only if their
    salts are equal. The body is deterministic filler drawn from a disjoint id range.
    """
    if n_words < 1:
        raise ValueError("n_words must be at least 1")
    body = [f"t{_UID_SPACE + (salt * 7 + i * 13) % 4000}" for i in range(n_words - 1)]
    return " ".join([f"t{salt}"] + body)


def make_flores(n_rows: int = 40, *, short_rows: Sequence[int] = (0,),
                long_rows: Sequence[int] = (1,),
                skew_every: Optional[int] = None) -> Dict[str, _FakeSplit]:
    """FLORES-shaped fixture: line-aligned across languages, by construction.

    ``short_rows``/``long_rows`` fall outside a 12–160 token filter in **one** language
    each, so the "keep only if every language passes" rule has something to reject.

    ``skew_every`` makes every *k*-th row far longer in ``deu_Latn`` while staying inside
    the filter, so the length-matching subset has rows it must genuinely drop.
    """
    data: Dict[str, _FakeSplit] = {}
    for lang_index, lang in enumerate(FLORES_LANGS):
        rows = _FakeSplit()
        for i in range(n_rows):
            n_words = 20 + (i % 5)
            if lang_index == 1 and i in short_rows:
                n_words = 4                     # below min_tokens in ben_Beng only
            if lang_index == 2 and i in long_rows:
                n_words = 400                   # above max_tokens in zho_Hans only
            if skew_every and lang == "deu_Latn" and i % skew_every == 0:
                n_words = 90                    # inside the filter, far off the reference
            rows.append({"sentence": _sentence(n_words, salt=_uid(i, lang_index))})
        data[lang] = rows
    return data


def make_xnli(n_rows: int = 36, *, with_prompt_id: bool = True,
              shuffle_seed: int = 5) -> Dict[str, _FakeSplit]:
    """XNLI-shaped fixture with per-language shards emitted in *different* orders.

    Every language's shard contains the same ``promptID`` set and the same gold label per
    ``promptID``, but the row order differs per language. Any implementation that joins by
    position will mismatch labels; joining on ``promptID`` recovers the alignment exactly.
    """
    data: Dict[str, _FakeSplit] = {}
    labels = {i: XNLI_LABELS[i % len(XNLI_LABELS)] for i in range(n_rows)}
    for lang_index, lang in enumerate(XNLI_LANGS):
        rows: List[dict] = []
        for i in range(n_rows):
            row = {
                "premise": _sentence(8 + (i % 3), salt=_uid(i, lang_index, field=0)),
                "hypothesis": _sentence(6 + (i % 2), salt=_uid(i, lang_index, field=1)),
                "label": labels[i],
            }
            if with_prompt_id:
                row["promptID"] = 1000 + i
            rows.append(row)
        random.Random(shuffle_seed + lang_index).shuffle(rows)
        data[lang] = _FakeSplit(rows)
    return data


def make_xnli_all_languages(n_rows: int = 36) -> _FakeSplit:
    """The ``all_languages`` config shape: one row *is* the translation set."""
    rows = _FakeSplit()
    for i in range(n_rows):
        premise = {lang: _sentence(8 + (i % 3), salt=_uid(i, j, field=0))
                   for j, lang in enumerate(XNLI_LANGS)}
        hypothesis = {
            "language": list(XNLI_LANGS),
            "translation": [_sentence(6 + (i % 2), salt=_uid(i, j, field=1))
                            for j in range(len(XNLI_LANGS))],
        }
        rows.append({"premise": premise, "hypothesis": hypothesis,
                     "label": XNLI_LABELS[i % len(XNLI_LABELS)]})
    return rows


def fake_load_dataset(flores=None, xnli=None, xnli_all=None,
                      xnli_has_prompt_id: bool = True):
    """Build a ``load_dataset`` replacement closing over the given fixtures."""

    def _load(path, name=None, split=None, **kwargs):
        if path == "facebook/flores":
            if flores is None:
                raise ValueError("no FLORES fixture configured")
            return flores[name]
        if path == "facebook/xnli":
            if name == "all_languages":
                if xnli_all is None:
                    raise ValueError("no XNLI all_languages fixture configured")
                return xnli_all
            if xnli is None:
                raise ValueError("no XNLI fixture configured")
            return xnli[name]
        raise ValueError(f"unexpected dataset {path!r} in an offline test")

    return _load
