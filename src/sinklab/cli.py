"""Explicit validation and offline Stage01 artifact preparation entry point."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, resolve_config
from .provenance import LockError, _no_duplicate_keys


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as source:
        for line in source:
            if line.strip():
                yield json.loads(line, object_pairs_hook=_no_duplicate_keys)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sinklab")
    command = parser.add_subparsers(dest="command", required=True)
    validate = command.add_parser("validate", help="validate one explicit run; no training")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--seed", type=int, required=True)
    validate.add_argument("--protocol-lock", type=Path)
    validate.add_argument("--production", action="store_true")
    corpus = command.add_parser("prepare-corpus", help="prepare a pinned local JSONL corpus; no downloads")
    corpus.add_argument("--input-jsonl", type=Path, required=True)
    corpus.add_argument("--tokenizer-dir", type=Path, required=True)
    corpus.add_argument("--dataset-id", required=True)
    corpus.add_argument("--dataset-revision", required=True)
    corpus.add_argument("--license-id", required=True)
    corpus.add_argument("--tokenizer-id", required=True)
    corpus.add_argument("--tokenizer-revision", required=True)
    corpus.add_argument("--out-dir", type=Path, required=True)
    init = command.add_parser("prepare-init", help="save one random CPU-FP32 GPT-2 state")
    init.add_argument("--config", type=Path, required=True)
    init.add_argument("--seed", type=int, required=True)
    target = init.add_mutually_exclusive_group(required=True)
    target.add_argument("--study", choices=("S1", "S3"))
    target.add_argument("--fixture", action="store_true")
    init.add_argument("--out-dir", type=Path, required=True)
    panels = command.add_parser("prepare-panels", help="freeze OWT panels from a local manifest")
    panels.add_argument("--corpus", type=Path, required=True)
    panels.add_argument("--out-dir", type=Path, required=True)
    domains = command.add_parser("prepare-domains", help="freeze three local domain JSONL sources")
    domains.add_argument("--tokenizer-dir", type=Path, required=True)
    for name in ("sst2", "gsm8k", "humaneval"):
        domains.add_argument(f"--{name}-jsonl", type=Path, required=True)
        domains.add_argument(f"--{name}-revision", required=True)
    domains.add_argument("--out-dir", type=Path, required=True)
    order = command.add_parser("prepare-order", help="persist a seeded training-order cursor")
    order.add_argument("--corpus", type=Path, required=True)
    order.add_argument("--seed", type=int, required=True)
    order.add_argument("--updates", type=int, required=True)
    order.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "validate":
            raw = _json(args.config)
            lock = _json(args.protocol_lock) if args.protocol_lock else None
            spec = resolve_config(raw, seed=args.seed, protocol_lock=lock, production=args.production)
            output = {"study": spec.study, "condition": spec.condition, "variant": spec.variant,
                "seed": spec.seed, "device_role": spec.device_role,
                "protocol_digest": spec.protocol_digest,
                "production_requested": args.production, "action": "validated_only"}
        elif args.command == "prepare-corpus":
            from .data import local_tokenizer, prepare_corpus, save_manifest, tokenizer_files_hash, validate_corpus
            tokenizer = local_tokenizer(args.tokenizer_dir)
            digest = tokenizer_files_hash(args.tokenizer_dir)
            artifact = prepare_corpus(_jsonl(args.input_jsonl), tokenizer,
                dataset_id=args.dataset_id, dataset_revision=args.dataset_revision,
                license_id=args.license_id, tokenizer_id=args.tokenizer_id,
                tokenizer_revision=args.tokenizer_revision, tokenizer_sha256=digest)
            validate_corpus(artifact, tokenizer_sha256=digest)
            path = save_manifest(artifact, args.out_dir, "corpus")
            output = {"action": "prepared_corpus", "path": str(path), "sha256": artifact["sha256"]}
        elif args.command == "prepare-init":
            from transformers import GPT2Config
            from .config import MODELS
            from .initialization import InitializationError, create_initialization
            raw = _json(args.config)
            if not isinstance(raw, dict):
                raise InitializationError("model config must be a JSON object")
            config = GPT2Config(**raw)
            if args.study:
                expected = MODELS[args.study][1]
                if (config.n_layer, config.n_head, config.n_embd) != (expected["layers"], expected["heads"], expected["width"]):
                    raise InitializationError("student dimensions differ from study contract")
            weights, metadata, digest = create_initialization(config, args.seed, args.out_dir)
            output = {"action": "prepared_initialization", "weights": str(weights),
                      "metadata": str(metadata), "tensor_content_sha256": digest}
        elif args.command in ("prepare-panels", "prepare-order"):
            from .data import load_manifest, save_manifest, validate_corpus
            from .provenance import seal_payload
            from .panels import prepare_owt_panels
            from .order import UpdateOrder
            corpus_doc = seal_payload(load_manifest(args.corpus, stem="corpus")[0])
            corpus_payload, _ = validate_corpus(corpus_doc)
            if args.command == "prepare-panels":
                artifact = prepare_owt_panels(corpus_doc)
                stem, action = "owt-panels", "prepared_panels"
            else:
                if args.updates < 0:
                    raise ValueError("updates must be nonnegative")
                ids = [b["id"] for b in corpus_payload["partitions"]["training"]["blocks"]]
                stream = UpdateOrder(ids, seed=args.seed)
                for _ in range(args.updates):
                    stream.take_update()
                artifact = stream.snapshot()
                stem, action = "update-order", "prepared_order"
            path = save_manifest(artifact, args.out_dir, stem)
            output = {"action": action, "path": str(path), "sha256": artifact["sha256"]}
        else:
            from .data import local_tokenizer, save_manifest, tokenizer_files_hash
            from .panels import prepare_domain_panels, validate_domain_panels
            tokenizer = local_tokenizer(args.tokenizer_dir)
            digest = tokenizer_files_hash(args.tokenizer_dir)
            sources = {name: _jsonl(getattr(args, f"{name}_jsonl"))
                       for name in ("sst2", "gsm8k", "humaneval")}
            revisions = {name: getattr(args, f"{name}_revision")
                         for name in ("sst2", "gsm8k", "humaneval")}
            artifact = prepare_domain_panels(sources, tokenizer, tokenizer_sha256=digest, revisions=revisions)
            validate_domain_panels(artifact, tokenizer_sha256=digest)
            path = save_manifest(artifact, args.out_dir, "xdomain300")
            output = {"action": "prepared_domains", "path": str(path), "sha256": artifact["sha256"]}
    except (OSError, UnicodeError, json.JSONDecodeError, ConfigError, LockError, ValueError) as exc:
        print(f"sinklab: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
