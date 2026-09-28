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
    jobs = command.add_parser("validate-job-plan", help="inspect explicit jobs; never launch")
    jobs.add_argument("--plan", type=Path, required=True)
    jobs.add_argument("--config-root", type=Path, required=True)
    one_job = command.add_parser("inspect-job", help="inspect one named job; never launch")
    one_job.add_argument("--plan", type=Path, required=True)
    one_job.add_argument("--config-root", type=Path, required=True)
    one_job.add_argument("--run-id", required=True)
    corpus = command.add_parser("prepare-corpus", help="prepare a pinned local JSONL corpus; no downloads")
    corpus.add_argument("--input-jsonl", type=Path, required=True)
    corpus.add_argument("--tokenizer-dir", type=Path, required=True)
    corpus.add_argument("--dataset-id", required=True)
    corpus.add_argument("--dataset-revision", required=True)
    corpus.add_argument("--license-id", required=True)
    corpus.add_argument("--tokenizer-id", required=True)
    corpus.add_argument("--tokenizer-revision", required=True)
    corpus.add_argument("--out-dir", type=Path, required=True)
    owt = command.add_parser("prepare-owt-compat", help="pack pinned local OWT train stream; no download")
    owt.add_argument("--input-jsonl", type=Path, required=True)
    owt.add_argument("--tokenizer-dir", type=Path, required=True)
    owt.add_argument("--dataset-revision", required=True)
    owt.add_argument("--tokenizer-revision", required=True)
    owt.add_argument("--seed", type=int, required=True)
    owt.add_argument("--out-dir", type=Path, required=True)
    owt_panels = command.add_parser("prepare-owt-compat-panels", help="freeze historical OWT windows")
    owt_panels.add_argument("--corpus", type=Path, required=True)
    owt_panels.add_argument("--tokenizer-sha256", required=True)
    owt_panels.add_argument("--out-dir", type=Path, required=True)
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
    train = command.add_parser("train", help="one approved condition/seed; no sweep")
    for flag in ("config", "protocol-lock", "hardware-plan", "corpus", "panels", "student-config",
                 "initialization", "teacher-dir", "run-dir"):
        train.add_argument(f"--{flag}", type=Path, required=True)
    train.add_argument("--seed", type=int, required=True)
    train.add_argument("--stop-after", type=int, default=10000)
    train.add_argument("--extension-id")
    train.add_argument("--mse-scale", type=float)
    train.add_argument("--rel-scale", type=float)
    profile = command.add_parser("profile-candidate", help="one isolated real CUDA batch candidate")
    for flag in ("config", "corpus", "panels", "student-config", "initialization",
                 "teacher-dir", "profile-dir", "output"):
        profile.add_argument(f"--{flag}", type=Path, required=True)
    profile.add_argument("--seed", type=int, required=True)
    profile.add_argument("--microbatch", type=int, required=True)
    profile.add_argument("--mse-scale", type=float)
    profile.add_argument("--rel-scale", type=float)
    solve = command.add_parser("solve-batch-plan", help="solve one complete approved profile matrix")
    solve.add_argument("--required", type=Path, required=True)
    solve.add_argument("--profiles", type=Path, nargs="+", required=True)
    solve.add_argument("--out", type=Path, required=True)
    solve.add_argument("--production", action="store_true")
    inventory = command.add_parser("pythia-inventory-add", help="append one explicit immutable checkpoint entry")
    inventory.add_argument("--entry", type=Path, required=True)
    inventory.add_argument("--inventory", type=Path)
    inventory.add_argument("--out-dir", type=Path, required=True)
    resolve = command.add_parser("pythia-resolve-one", help="opt-in network resolution of one branch")
    resolve.add_argument("--size", choices=("160m", "410m"), required=True)
    resolve.add_argument("--training-seed", type=int, required=True)
    resolve.add_argument("--native-step", type=int, required=True)
    resolve.add_argument("--allow-network", action="store_true", required=True)
    resolve.add_argument("--out", type=Path, required=True)
    ppanel = command.add_parser("pythia-prepare-panel", help="retokenize selected raw documents locally")
    ppanel.add_argument("--input-jsonl", type=Path, required=True)
    ppanel.add_argument("--selected-document-hashes", type=Path, required=True)
    ppanel.add_argument("--source-panel-sha256", required=True)
    ppanel.add_argument("--tokenizer-dir", type=Path, required=True)
    ppanel.add_argument("--tokenizer-id", required=True)
    ppanel.add_argument("--tokenizer-revision", required=True)
    ppanel.add_argument("--out-dir", type=Path, required=True)
    pone = command.add_parser("pythia-evaluate-one", help="evaluate exactly one local pinned state")
    trajectory = command.add_parser("pythia-trajectory", help="opt-in bounded trajectory over explicit steps")
    for p in (pone, trajectory):
        p.add_argument("--inventory", type=Path, required=True)
        p.add_argument("--size", choices=("160m", "410m"), required=True)
        p.add_argument("--training-seed", type=int, required=True)
        p.add_argument("--panel", type=Path, required=True)
        p.add_argument("--tokenizer-sha256", required=True)
        p.add_argument("--store-root", type=Path, required=True)
        p.add_argument("--evaluator-seed", type=int, required=True)
        p.add_argument("--device", choices=("cpu", "cuda"), required=True)
        p.add_argument("--precision", choices=("fp32", "bf16"), required=True)
    pone.add_argument("--native-step", type=int, required=True)
    pone.add_argument("--snapshot", type=Path, required=True)
    trajectory.add_argument("--native-steps", type=int, nargs="+", required=True)
    trajectory.add_argument("--cache-root", type=Path, required=True)
    trajectory.add_argument("--byte-cap", type=int, required=True)
    trajectory.add_argument("--allow-download", action="store_true", required=True)
    trajectory.add_argument("--evict-after-verified", action="store_true",
                            help="delete only this command's verified checkpoint directory after each result")
    args = parser.parse_args(argv)
    try:
        if args.command.startswith("pythia-"):
            from .pythia import (create_inventory, load_inventory, resolve_one_hub_branch,
                prepare_panel, evaluate_one, select_checkpoint, run_trajectory)
            from .data import local_tokenizer, save_manifest, tokenizer_files_hash
            if args.command == "pythia-inventory-add":
                prior = load_inventory(args.inventory)[0]["entries"] if args.inventory else []
                document = create_inventory([*prior, _json(args.entry)])
                path = save_manifest(document, args.out_dir, "pythia-inventory")
                output = {"action": "inventory_added", "path": str(path), "sha256": document["sha256"]}
            elif args.command == "pythia-resolve-one":
                entry = resolve_one_hub_branch(size=args.size, training_seed=args.training_seed,
                    native_step=args.native_step)
                args.out.write_text(json.dumps(entry, sort_keys=True, indent=2) + "\n", encoding="utf-8")
                output = {"action": "resolved_one", "path": str(args.out), "revision": entry["revision"]}
            elif args.command == "pythia-prepare-panel":
                tokenizer = local_tokenizer(args.tokenizer_dir)
                document = prepare_panel(_jsonl(args.input_jsonl), tokenizer,
                    tokenizer_id=args.tokenizer_id, tokenizer_revision=args.tokenizer_revision,
                    tokenizer_sha256=tokenizer_files_hash(args.tokenizer_dir),
                    source_panel_sha256=args.source_panel_sha256,
                    selected_document_hashes=_json(args.selected_document_hashes))
                path = save_manifest(document, args.out_dir, "pythia-panel")
                output = {"action": "prepared_pythia_panel", "path": str(path), "sha256": document["sha256"]}
            else:
                inventory_payload, inventory_hash = load_inventory(args.inventory)
                if args.command == "pythia-evaluate-one":
                    entry = select_checkpoint(inventory_payload, size=args.size,
                        training_seed=args.training_seed, native_step=args.native_step)
                    output = evaluate_one(entry=entry, snapshot=args.snapshot,
                        panel_path=args.panel, tokenizer_sha256=args.tokenizer_sha256,
                        store_root=args.store_root, evaluator_seed=args.evaluator_seed,
                        device=args.device, precision=args.precision)
                else:
                    output = run_trajectory(inventory=inventory_payload, size=args.size,
                        training_seed=args.training_seed, native_steps=args.native_steps,
                        cache_root=args.cache_root, byte_cap=args.byte_cap, panel_path=args.panel,
                        tokenizer_sha256=args.tokenizer_sha256, store_root=args.store_root,
                        evaluator_seed=args.evaluator_seed, device=args.device,
                        precision=args.precision,
                        evict_after_verified=args.evict_after_verified)
                output["inventory_sha256"] = inventory_hash
        elif args.command == "solve-batch-plan":
            from .hardware import Profile, build_batch_plan
            profiles = [Profile(**_json(path)["profile"]) for path in args.profiles]
            required = {tuple(row) for row in _json(args.required)["required"]}
            output = build_batch_plan(profiles, required, production=args.production)
            args.out.write_text(json.dumps(output, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        elif args.command == "inspect-job":
            from .job_plan import inspect_job
            output = inspect_job(_json(args.plan), args.config_root, args.run_id)
        elif args.command in {"train", "profile-candidate"}:
            from .training_entry import run_approved_training, run_profile_candidate
            output = run_approved_training(args) if args.command == "train" else run_profile_candidate(args)
        elif args.command == "validate-job-plan":
            from .job_plan import validate_job_plan
            output = validate_job_plan(_json(args.plan), args.config_root)
        elif args.command == "validate":
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
        elif args.command == "prepare-owt-compat":
            from .data import local_tokenizer, save_manifest, tokenizer_files_hash
            from .owt_compat import prepare_owt_corpus, validate_owt_corpus

            tokenizer = local_tokenizer(args.tokenizer_dir)
            digest = tokenizer_files_hash(args.tokenizer_dir)
            artifact = prepare_owt_corpus(_jsonl(args.input_jsonl), tokenizer, seed=args.seed,
                dataset_revision=args.dataset_revision, tokenizer_revision=args.tokenizer_revision,
                tokenizer_sha256=digest)
            validate_owt_corpus(artifact, tokenizer_sha256=digest)
            path = save_manifest(artifact, args.out_dir, "owt-corpus")
            output = {"action": "prepared_owt_corpus", "path": str(path),
                      "sha256": artifact["sha256"]}
        elif args.command == "prepare-owt-compat-panels":
            from .data import save_manifest
            from .owt_compat import load_owt_corpus, prepare_owt_panels, validate_owt_panels
            from .provenance import seal_payload

            payload, _ = load_owt_corpus(args.corpus, tokenizer_sha256=args.tokenizer_sha256)
            corpus = seal_payload(payload)
            artifact = prepare_owt_panels(corpus)
            validate_owt_panels(artifact, corpus)
            path = save_manifest(artifact, args.out_dir, "owt-panels")
            output = {"action": "prepared_owt_panels", "path": str(path),
                      "sha256": artifact["sha256"]}
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
