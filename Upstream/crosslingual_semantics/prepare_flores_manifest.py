# -*- coding: utf-8 -*-
"""prepare_flores_manifest.py — build the E7 FLORES parallel manifest (WP8).

``04_MODULE_SPEC_e7_crosslingual.md`` §1.

    python crosslingual_semantics/prepare_flores_manifest.py \\
      --tokenizer Qwen/Qwen2.5-0.5B --n 300 --seed 42 \\
      --out crosslingual_semantics/results/manifests/flores_devtest.json

Joins ``facebook/flores`` ``devtest`` by source index — legitimate here because FLORES
devtest is line-aligned across language configs, and the loader asserts the configs really
do have equal row counts before relying on it. Keeps a row only when **every** language
falls in ``[--min-tokens, --max-tokens]``, samples ``--n`` with ``--seed``, and records the
per-language covariates the analysis needs later (first-token id and its frequency over
the full split, a punctuation-at-position-0 indicator, Unicode script, language family).

Also emits the length-matched secondary subset (``design §10.3``) and the grouped
Procrustes partition.

**Validity.** Fewer than 200 surviving rows ⇒ ``provenance["valid"] = False`` and a
non-zero exit. Do not proceed on an invalid manifest.

**Tokenizer coupling.** The manifest is tokenizer-specific. Qwen2.5-0.5B and -1.5B share a
tokenizer; ``--assert-tokenizer-matches`` checks that against a second model id rather than
assuming it, and a differing model needs its own manifest — never a join across them.

Requires network/HF cache access: run it on the compute PC. The alignment, filtering and
partition logic is unit-tested offline in ``tests/test_parallel_manifest_alignment.py``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import paired_manifests as pm  # noqa: E402
import provenance as prov  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B",
                        help="HF tokenizer id; the manifest is specific to it")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--assert-tokenizer-matches", default=None,
                        help="second model id whose tokenizer must be identical "
                             "(04 §1 tokenizer coupling)")
    parser.add_argument("--langs", nargs="+", default=list(pm.FLORES_LANGS))
    parser.add_argument("--split", default="devtest")
    parser.add_argument("--n", type=int, default=300)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--max-rel-diff", type=float, default=0.20,
                        help="length-matching bound for the secondary subset")
    parser.add_argument("--procrustes-fraction", type=float, default=1 / 3,
                        help="share of semantic ids used to FIT Procrustes; the rest is "
                             "the disjoint evaluation set (04 §4)")
    parser.add_argument("--out", default=str(
        _REPO / "crosslingual_semantics" / "results" / "manifests" / "flores_devtest.json"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def _load_tokenizer(name, revision, local_files_only):
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(name, revision=revision,
                                         local_files_only=local_files_only)


def _assert_same_tokenizer(tokenizer, other_name, local_files_only) -> dict:
    """Compare two tokenizers on the properties that make a manifest transferable."""
    other = _load_tokenizer(other_name, None, local_files_only)
    probe = "The quick brown fox jumps over the lazy dog."
    same = {
        "vocab_size": tokenizer.vocab_size == other.vocab_size,
        "len": len(tokenizer) == len(other),
        "encoding": (tokenizer(probe, add_special_tokens=False)["input_ids"]
                     == other(probe, add_special_tokens=False)["input_ids"]),
    }
    return {"other": other_name, "checks": same, "identical": all(same.values())}


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tokenizer = _load_tokenizer(args.tokenizer, args.tokenizer_revision,
                                args.local_files_only)

    tokenizer_check = None
    if args.assert_tokenizer_matches:
        tokenizer_check = _assert_same_tokenizer(
            tokenizer, args.assert_tokenizer_matches, args.local_files_only)
        if not tokenizer_check["identical"]:
            print(f"ERROR: {args.tokenizer} and {args.assert_tokenizer_matches} do not "
                  f"share a tokenizer: {tokenizer_check['checks']}. Build a separate "
                  "manifest for each and never join across them (04 §1).", file=sys.stderr)
            return 2

    manifest = pm.build_flores_manifest(
        tokenizer, args.langs, split=args.split, min_tokens=args.min_tokens,
        max_tokens=args.max_tokens, n=args.n, seed=args.seed)

    # Procrustes fit/eval split, by semantic id (04 §4).
    partitions = pm.grouped_partition(
        manifest, fractions=(0.0, args.procrustes_fraction, 1.0),
        names=("procrustes_train", "procrustes_test"), seed=args.seed)
    manifest = pm.with_partitions(manifest, partitions)
    if tokenizer_check is not None:
        manifest.provenance["tokenizer_check"] = tokenizer_check

    out = Path(args.out)
    report = pm.verify_manifest(manifest)
    manifest.save(out)

    subset = pm.build_length_matched_subset(
        manifest, reference_lang=args.langs[0], max_rel_diff=args.max_rel_diff)
    subset_path = out.with_name(out.stem + "_lenmatched" + out.suffix)
    subset.save(subset_path)

    prov.write_json(out.with_name(out.stem + "_report.json"), {
        "manifest": report,
        "length_matched": subset.provenance["length_matched"],
        "provenance": prov.provenance_block(extra={
            "script": "prepare_flores_manifest.py",
            "args": vars(args),
            "manifest_sha256": manifest.sha256,
            "length_matched_sha256": subset.sha256,
        }),
    })

    print(f"FLORES manifest: {len(manifest)} semantic ids x {len(manifest.languages)} "
          f"languages -> {out}")
    print(f"  candidates={manifest.provenance['n_candidates']} "
          f"survivors={manifest.provenance['n_survivors']} "
          f"realised={manifest.provenance['realised_n']}")
    print(f"  sha256={manifest.sha256}")
    print(f"  procrustes_train={len(partitions['procrustes_train'])} "
          f"procrustes_test={len(partitions['procrustes_test'])}")
    print(f"  length-matched subset: {len(subset)} ids -> {subset_path}")
    for lang, realised in subset.provenance["length_matched"][
            "realised_max_rel_diff"].items():
        print(f"    {lang}: realised max relative length difference {realised:.3f}")

    if not manifest.provenance["valid"]:
        print(f"ERROR: only {manifest.provenance['realised_n']} rows survived the "
              f"[{args.min_tokens}, {args.max_tokens}] filter in all "
              f"{len(manifest.languages)} languages; 04 §1 requires at least 200. "
              "provenance['valid'] is False — do not proceed on this manifest.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
