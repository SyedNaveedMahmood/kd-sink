# -*- coding: utf-8 -*-
"""test_openwebtext_corpus.py — the E6A-GPT2 arm's in-domain corpus.

OpenWebText ships a single ``train`` split, so ``train`` and ``validation`` here are
**disjoint document windows** carved out of it. That is one more level of disjointness than
the TinyStories arm needs, and it is the level that matters most: if the windows overlapped,
the arm would measure its sink on blocks the student had trained on and nothing downstream
would say so. These tests pin the disjointness, the determinism and the corpus id the
pre-registration names. No downloads.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import datasets_loader as dl  # noqa: E402


class _Tok:
    eos_token_id = 2
    name_or_path = "stub-gpt2"

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [(sum(ord(c) for c in w) % 5000) + 3 for w in text.split()]}

    def decode(self, ids):
        return " ".join(str(int(i)) for i in ids)


#: One distinctive token per document, so a block can be traced to the window it came from.
def _documents(n=2000, tag="d"):
    return [{"text": " ".join(f"{tag}{i}_{j}" for j in range(60))} for i in range(n)]


@pytest.fixture
def stub_openwebtext(monkeypatch):
    """A stand-in for the streamed prefix that honours the real window arithmetic."""
    docs = _documents()

    def _load(split, *, train_documents=None, validation_documents=None):
        n_train = int(train_documents or dl.OPENWEBTEXT_TRAIN_DOCUMENTS)
        n_val = int(validation_documents or dl.OPENWEBTEXT_VALIDATION_DOCUMENTS)
        windows = {"train": (0, n_train), "validation": (n_train, n_train + n_val)}
        if split not in windows:
            raise ValueError(f"unknown window {split!r}")
        start, end = windows[split]
        rows = docs[start:end]
        if not rows:
            raise ValueError(f"window {split!r} yielded nothing")
        return rows

    monkeypatch.setattr(dl, "_load_openwebtext", _load)
    return docs


def test_the_corpus_id_is_the_one_the_preregistration_names(stub_openwebtext):
    corpus = cp.openwebtext_corpus(_Tok(), "validation", 8, block_size=16, seed=0,
                                   train_documents=100, validation_documents=40)
    assert corpus.corpus_id == "openwebtext_validation_sink_8"

    import yaml

    prereg = yaml.safe_load((REPO / "transformation_inheritance" / "configs"
                             / "e6a_gpt2_preregistration.yaml").read_text(encoding="utf-8"))
    # The literal the criteria are scored on, at the production block count.
    assert prereg["corpora"]["in_domain"] == "openwebtext_validation_sink_300"
    assert corpus.corpus_id.rsplit("_", 1)[0] == "openwebtext_validation_sink"


def test_train_and_validation_windows_share_no_document(stub_openwebtext):
    """The guarantee the arm rests on: no validation block contains a training token."""
    train = dl._load_openwebtext("train", train_documents=100, validation_documents=40)
    validation = dl._load_openwebtext("validation", train_documents=100,
                                      validation_documents=40)
    assert len(train) == 100 and len(validation) == 40
    assert not ({d["text"] for d in train} & {d["text"] for d in validation})


def test_the_window_arithmetic_is_recorded_in_provenance(stub_openwebtext):
    corpus = cp.openwebtext_corpus(_Tok(), "train", 6, block_size=16, seed=0)
    window = corpus.provenance["document_window"]
    assert window == list(dl.OPENWEBTEXT_SPLIT_WINDOWS["train"])
    assert corpus.provenance["provider"] == "openwebtext_corpus"


def test_sink_and_ppl_windows_are_disjoint(stub_openwebtext):
    """`02` §2.3, exactly as the TinyStories provider guarantees it."""
    sink = cp.openwebtext_corpus(_Tok(), "validation", 5, block_size=16, seed=0,
                                 purpose="sink", train_documents=100,
                                 validation_documents=400)
    ppl = cp.openwebtext_corpus(_Tok(), "validation", 5, block_size=16, seed=0,
                                purpose="ppl", train_documents=100,
                                validation_documents=400)
    sink_blocks = {tuple(item.input_ids) for item in sink.items}
    ppl_blocks = {tuple(item.input_ids) for item in ppl.items}
    assert not (sink_blocks & ppl_blocks)
    assert {i.meta["block_index"] for i in sink.items} == set(range(5))
    assert min(i.meta["block_index"] for i in ppl.items) == cp.SINK_BLOCKS_RESERVED


def test_a_sink_corpus_larger_than_the_reserved_window_is_refused(stub_openwebtext):
    with pytest.raises(ValueError, match="disjointness"):
        cp.openwebtext_corpus(_Tok(), "validation", cp.SINK_BLOCKS_RESERVED + 1,
                              block_size=16, seed=0, purpose="sink")


def test_the_corpus_is_deterministic_under_a_fixed_seed(stub_openwebtext):
    kwargs = dict(block_size=16, seed=0, train_documents=100, validation_documents=40)
    first = cp.openwebtext_corpus(_Tok(), "validation", 6, **kwargs)
    second = cp.openwebtext_corpus(_Tok(), "validation", 6, **kwargs)
    assert first.manifest_sha256 == second.manifest_sha256

    other = cp.openwebtext_corpus(_Tok(), "validation", 6,
                                  **{**kwargs, "seed": 1})
    assert other.manifest_sha256 != first.manifest_sha256


def test_an_unknown_window_is_refused_not_silently_emptied(stub_openwebtext):
    with pytest.raises(ValueError, match="window"):
        dl._load_openwebtext("test")


#: A deliberately pessimistic tokens-per-document figure for OpenWebText. The real average
#: is several times this; the point of the low bound is that the cap must hold even if the
#: prefix happens to be short documents.
PESSIMISTIC_TOKENS_PER_DOCUMENT = 250


def test_the_document_cap_covers_the_configs_own_step_horizon():
    """One epoch at ``optim.max_steps``, not merely at the 2,000-step pilot.

    Sizing for the pilot alone would be a trap rather than a saving: ``_pack_documents``
    shuffles ``range(len(documents))``, so ``train_documents`` is an input to the block
    order. A pilot packed from a smaller cap would see a *different* block sequence from
    Phase 2, and ``--max-steps 2000`` would stop being a truncation of the same run.

    Read from the shipped configs so the constant and the schedule cannot drift apart.
    """
    import yaml

    configs = sorted((REPO / "transformation_inheritance" / "configs")
                     .glob("e6a_gpt2_*.yaml"))
    configs = [p for p in configs if "preregistration" not in p.name]
    # Every registered GPT-2 training config is checked by the horizon assertions below.
    # Require the established original and large->medium arms while allowing separately
    # registered scale/alignment extensions to expand this filename family.
    required = {
        "e6a_gpt2_ce.yaml",
        "e6a_gpt2_logit_kd.yaml",
        "e6a_gpt2_logit_attention_kd.yaml",
        "e6a_gpt2_large_medium_ce.yaml",
        "e6a_gpt2_large_medium_logit_kd.yaml",
        "e6a_gpt2_large_medium_logit_attention_kd.yaml",
    }
    assert required.issubset({p.name for p in configs}), [p.name for p in configs]

    for path in configs:
        cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
        data = cfg["data"]
        optim = cfg["optim"]
        needed = (int(optim["max_steps"]) * int(optim["per_device_batch_size"])
                  * int(optim["grad_accum"]) * int(data["block_size"]))
        available = int(data["train_documents"]) * PESSIMISTIC_TOKENS_PER_DOCUMENT
        assert available > needed, (
            f"{path.name}: {data['train_documents']} documents give at most {available} "
            f"tokens at {PESSIMISTIC_TOKENS_PER_DOCUMENT}/doc, but max_steps="
            f"{optim['max_steps']} consumes {needed}. A second epoch would silently repeat "
            "data the registered schedule assumes is fresh.")

        # The ΔCE endpoint corpus is the largest validation demand: 2,000 blocks + 300 sink.
        val_needed = (2000 + 300) * int(data["block_size"])
        val_available = int(data["validation_documents"]) * PESSIMISTIC_TOKENS_PER_DOCUMENT
        assert val_available > val_needed, path.name


def test_the_windows_abut_and_never_overlap():
    train_end = dl.OPENWEBTEXT_SPLIT_WINDOWS["train"][1]
    validation_start = dl.OPENWEBTEXT_SPLIT_WINDOWS["validation"][0]
    assert train_end == validation_start, "the windows must abut, never overlap"
    assert dl.OPENWEBTEXT_SPLIT_WINDOWS["train"][0] == 0


def test_the_configs_and_the_module_defaults_agree():
    """A config that silently disagreed with the module default would change the shuffle."""
    import yaml

    for path in (REPO / "transformation_inheritance" / "configs").glob("e6a_gpt2_*.yaml"):
        if "preregistration" in path.name:
            continue
        data = yaml.safe_load(path.read_text(encoding="utf-8"))["data"]
        assert data["train_documents"] == dl.OPENWEBTEXT_TRAIN_DOCUMENTS, path.name
        assert data["validation_documents"] == dl.OPENWEBTEXT_VALIDATION_DOCUMENTS, path.name
