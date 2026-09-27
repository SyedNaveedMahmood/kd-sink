"""Prepare explicit synthetic sequence-128 inputs for GPU memory profiling only.

These artifacts are deliberately ineligible for scientific training or
calibration. Keep them outside the repository and report their hashes.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import GPT2Config

from sinklab.data import document_hash, document_split, prepare_corpus, save_manifest, validate_corpus
from sinklab.initialization import create_initialization
from sinklab.provenance import seal_payload


class ShapeTokenizer:
    eos_token_id = 50256

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            raise ValueError("shape fixture forbids implicit special tokens")
        return [ord(char) for char in text]


def _row(split: str, source_index: int) -> dict:
    base = "synthetic stage06 memory shape only " * 280
    for number in range(100000):
        candidate = base + f"{split}-{number:05d}"
        if document_split(document_hash(candidate)) == split:
            return {"source_index": source_index, "document_id": f"fixture-{split}", "text": candidate}
    raise RuntimeError(f"cannot construct {split} shape document")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.out_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    config = GPT2Config(n_layer=24, n_head=16, n_embd=1024,
                        _attn_implementation="eager")
    config_path = root / "student_config.json"
    config_path.write_text(json.dumps(config.to_dict(), sort_keys=True, indent=2) + "\n", encoding="utf-8")
    _, initialization, init_hash = create_initialization(config, 0, root / "initialization")
    corpus = prepare_corpus([_row("training", 0), _row("calibration", 1),
                             _row("evaluation", 2)], ShapeTokenizer(),
                            dataset_id="synthetic-stage06-shape-only", dataset_revision="v1",
                            license_id="generated-fixture", tokenizer_id="ascii-shape-fixture",
                            tokenizer_revision="v1", tokenizer_sha256="0" * 64)
    validate_corpus(corpus, tokenizer_sha256="0" * 64)
    corpus_path = save_manifest(corpus, root / "data", "corpus")
    evaluation = corpus["payload"]["partitions"]["evaluation"]["blocks"]
    if len(evaluation) < 64:
        raise RuntimeError("shape fixture has fewer than 64 evaluation blocks")
    panel = seal_payload({"kind": "stage06-synthetic-dense64-shape-only",
                          "corpus_sha256": corpus["sha256"],
                          "owt_dense64": [block["id"] for block in evaluation[:64]]})
    panel_path = save_manifest(panel, root / "data", "shape-panel")
    summary = {"status": "synthetic_shape_only_not_production_data",
               "student_config": str(config_path), "initialization": str(initialization),
               "initialization_tensor_sha256": init_hash,
               "corpus": str(corpus_path), "corpus_sha256": corpus["sha256"],
               "panel": str(panel_path), "panel_sha256": panel["sha256"],
               "sequence_length": 128, "dense_items": 64,
               "training_blocks": len(corpus["payload"]["partitions"]["training"]["blocks"])}
    (root / "fixture.json").write_text(json.dumps(summary, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
