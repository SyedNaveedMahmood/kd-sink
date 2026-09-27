import copy

import pytest

from sinklab.data import DataError, document_hash, document_split, prepare_corpus
from sinklab.panels import (prepare_domain_panels, prepare_owt_panels,
                            validate_domain_panels, validate_owt_panels)
from sinklab.provenance import seal_payload


class TinyTokenizer:
    eos_token_id = 999

    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return [ord(char) for char in text]


def selected_text(split, size):
    for n in range(1000):
        text = f"item-{n}-" + "x" * size
        if document_split(document_hash(text)) == split:
            return text
    raise AssertionError("missing split fixture")


def test_frozen_nested_panels_and_calibration_guard():
    corpus = prepare_corpus([
        {"source_index": 0, "document_id": "eval", "text": selected_text("evaluation", 2000 * 128)},
        {"source_index": 1, "document_id": "cal", "text": selected_text("calibration", 1024 * 128)},
    ], TinyTokenizer(), dataset_id="synthetic", dataset_revision="v1", license_id="fixture",
        tokenizer_id="tiny", tokenizer_revision="v1", tokenizer_sha256="a" * 64)
    panels = prepare_owt_panels(corpus)
    payload = panels["payload"]
    assert payload["owt_dense64"] == payload["owt_full300"][:64]
    assert payload["owt_full300"] == payload["owt_lm2000"][:300]
    assert len(payload["calibration16x64"]) == 16
    assert len({x for batch in payload["calibration16x64"] for x in batch}) == 1024
    assert not set(payload["owt_lm2000"]) & {x for b in payload["calibration16x64"] for x in b}
    assert validate_owt_panels(panels, corpus) == panels["sha256"]
    altered = copy.deepcopy(payload)
    altered["owt_dense64"].reverse()
    with pytest.raises(DataError, match="membership"):
        validate_owt_panels(seal_payload(altered), corpus)
    with pytest.raises(DataError, match="insufficient"):
        prepare_owt_panels(prepare_corpus([], TinyTokenizer(), dataset_id="fixture",
            dataset_revision="v1", license_id="fixture", tokenizer_id="tiny",
            tokenizer_revision="v1", tokenizer_sha256="a" * 64))


def test_domain_exact_fields_padding_prefix_and_insufficient():
    sources = {
        "sst2": [{"split": "validation", "sentence": f"sent {i}", "label": "SECRET"} for i in range(101)],
        "gsm8k": [{"split": "test", "question": f"question {i}", "answer": "SECRET"} for i in range(101)],
        "humaneval": [{"split": "test", "prompt": f"def function_{i}():", "canonical_solution": "SECRET",
                       "test": "SECRET"} for i in range(101)],
    }
    panel = prepare_domain_panels(sources, TinyTokenizer(), tokenizer_sha256="a" * 64,
                                  revisions={name: "pinned-fixture-v1" for name in sources})
    assert validate_domain_panels(panel, tokenizer_sha256="a" * 64) == panel["sha256"]
    assert "SECRET" not in str(panel)
    assert sum(len(x["items"]) for x in panel["payload"]["domains"].values()) == 300
    item = panel["payload"]["domains"]["sst2"]["items"][0]
    assert item["renderings"]["40"]["attention_mask"][-1] == 0
    with pytest.raises(DataError, match="tokenizer"):
        validate_domain_panels(panel, tokenizer_sha256="b" * 64)
    sources["sst2"] = sources["sst2"][:99]
    with pytest.raises(DataError, match="insufficient"):
        prepare_domain_panels(sources, TinyTokenizer(), tokenizer_sha256="a" * 64,
                              revisions={name: "pinned-fixture-v1" for name in sources})
