# -*- coding: utf-8 -*-
"""prepare_xnli_manifest.py — build the E7 XNLI manifest and patch controls (WP8).

``04_MODULE_SPEC_e7_crosslingual.md`` §2.

    python crosslingual_semantics/prepare_xnli_manifest.py \\
      --tokenizer Qwen/Qwen2.5-0.5B --n 600 --seed 42 \\
      --out crosslingual_semantics/results/manifests/xnli_test.json

Languages ``en zh ar de hi sw tr``; 600 aligned semantic ids balanced across entailment /
neutral / contradiction, after tokenisation and length filtering.

**The alignment trap (CLAUDE.md trap 2).** XNLI is joined on an explicit id — never on row
index. ``datasets_loader.load_xnli_aligned`` resolves the strategy at runtime (an explicit
``promptID``/``pairID`` column when the shards have one; otherwise the structurally aligned
``all_languages`` config, where one row *is* the translation set) and records which was
used. There is no positional fallback: a positional join yields a manifest that looks fine
and silently destroys every cross-language claim. ``verify_manifest`` then asserts gold
labels agree across all seven languages for every semantic id, which is what a positional
bug shows up as.

**Prompt.** The premise is at position 0 — no instruction prefix, because a constant prefix
would manufacture a shared first-token anchor and invalidate every position-0 measurement
(design §16.2). Candidates ``" entailment" / " neutral" / " contradiction"`` are scored by
length-normalised sequence log-probability; there is no free-form generation.

**Prompt-language confound.** The task wording is English in all conditions by design
(§16.3). ``--translated-instruction LANG=TEMPLATE`` is the prescribed robustness switch;
run it on at least two languages. Which languages were translated is recorded in the
manifest provenance.

Emits the manifest, ``patch_controls.csv``, ``partitions.json`` (200 dev / 400 test, split
by semantic id) and a verification report. Refuses to write a manifest that fails
``verify_manifest``.

Requires network/HF cache access: run it on the compute PC. The join, control-assignment
and partition logic is unit-tested offline in ``tests/test_parallel_manifest_alignment.py``,
``tests/test_patch_controls.py`` and ``tests/test_grouped_partition.py``.
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
    parser.add_argument("--tokenizer", default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--tokenizer-revision", default=None)
    parser.add_argument("--langs", nargs="+", default=list(pm.XNLI_LANGS))
    parser.add_argument("--split", default="test")
    parser.add_argument("--n", type=int, default=600)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-tokens", type=int, default=12)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--dev-fraction", type=float, default=1 / 3,
                        help="200 of 600 by default (04 §5.3)")
    parser.add_argument("--unbalanced", action="store_true",
                        help="skip the entailment/neutral/contradiction balancing")
    parser.add_argument("--translated-instruction", nargs="*", default=[],
                        metavar="LANG=TEMPLATE",
                        help="robustness switch (04 §2): replace the English task wording "
                             "for the named languages. TEMPLATE must contain {premise} "
                             "and {hypothesis}.")
    parser.add_argument("--out", default=str(
        _REPO / "crosslingual_semantics" / "results" / "manifests" / "xnli_test.json"))
    parser.add_argument("--local-files-only", action="store_true")
    return parser


def parse_translated_instructions(values) -> dict:
    """``LANG=TEMPLATE`` pairs -> ``{lang: template}``, validated."""
    out = {}
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"--translated-instruction expects LANG=TEMPLATE, got {value!r}")
        lang, template = value.split("=", 1)
        if "{premise}" not in template or "{hypothesis}" not in template:
            raise ValueError(
                f"translated template for {lang!r} must contain {{premise}} and "
                "{hypothesis}; the premise must still begin at position 0 (design §16.2)")
        out[lang.strip()] = template
    return out


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)

    try:
        translated = parse_translated_instructions(args.translated_instruction)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer, revision=args.tokenizer_revision,
        local_files_only=args.local_files_only)

    manifest = pm.build_xnli_manifest(
        tokenizer, args.langs, split=args.split, n=args.n, seed=args.seed,
        balanced=not args.unbalanced, min_tokens=args.min_tokens,
        max_tokens=args.max_tokens, instruction_by_lang=translated)

    partitions = pm.grouped_partition(
        manifest, fractions=(0.0, args.dev_fraction, 1.0),
        names=("dev", "test"), seed=args.seed)
    manifest = pm.with_partitions(manifest, partitions)

    # Verify BEFORE writing: 04 §2 refuses to write on failure.
    report = pm.verify_manifest(manifest)

    out = Path(args.out)
    manifest.save(out)

    controls = pm.assign_patch_controls(manifest, seed=args.seed)
    controls_path = out.with_name("patch_controls.csv")
    controls_path.parent.mkdir(parents=True, exist_ok=True)
    controls.to_csv(controls_path, index=False, encoding="utf-8")

    prov.write_json(out.with_name("partitions.json"), {
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "partitions": {k: list(v) for k, v in partitions.items()},
        "sizes": {k: len(v) for k, v in partitions.items()},
        "provenance": prov.provenance_block(extra={
            "script": "prepare_xnli_manifest.py", "seed": args.seed}),
    })

    unassignable = int((controls["status"] != "ok").sum())
    prov.write_json(out.with_name(out.stem + "_report.json"), {
        "manifest": report,
        "join_strategy": manifest.provenance["loader"]["join_strategy"],
        "label_counts": manifest.provenance["label_counts"],
        "translated_instruction_langs": manifest.provenance[
            "translated_instruction_langs"],
        "patch_controls": {"rows": int(len(controls)),
                           "unassignable": unassignable},
        "provenance": prov.provenance_block(extra={
            "script": "prepare_xnli_manifest.py",
            "args": vars(args),
            "manifest_sha256": manifest.sha256,
        }),
    })

    print(f"XNLI manifest: {len(manifest)} semantic ids x {len(manifest.languages)} "
          f"languages -> {out}")
    print(f"  join_strategy={manifest.provenance['loader']['join_strategy']} "
          f"({manifest.provenance['loader'].get('join_key_column')})")
    print(f"  label_counts={manifest.provenance['label_counts']}")
    print(f"  sha256={manifest.sha256}")
    print(f"  dev={len(partitions['dev'])} test={len(partitions['test'])} "
          f"-> {out.with_name('partitions.json')}")
    print(f"  patch controls: {len(controls)} rows -> {controls_path}")
    if translated:
        print(f"  translated instruction for: {sorted(translated)}")
    if unassignable:
        print(f"  WARNING: {unassignable} control rows are unassignable and are written "
              "with status='unassignable' (failures are data, never dropped).")

    if not manifest.provenance["valid"]:
        print(f"ERROR: only {manifest.provenance['realised_n']} semantic ids survived "
              "the join and length filter; provenance['valid'] is False. Do not proceed.",
              file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
