# -*- coding: utf-8 -*-
"""run_multilingual_fingerprint.py — the baseline multilingual fingerprint (WP9).

``04_MODULE_SPEC_e7_crosslingual.md`` §3.4: the full intervention battery run *per language*
on the FLORES manifest, alongside the design §10.5 controls. It answers the prior question
E7's patching results are interpreted against — *is the sink mechanism itself the same in
every language?* — and it is by far the cheapest evidence in E7.

    python crosslingual_semantics/run_multilingual_fingerprint.py \\
      --config crosslingual_semantics/configs/e7_qwen05_base.yaml \\
      --manifest crosslingual_semantics/results/manifests/flores_devtest.json \\
      --variant matched --out crosslingual_semantics/results/fingerprints

This is a **driver, not an instrument**. Every number comes from
:func:`fingerprint_runner.compute_fingerprint`, which dispatches into the frozen harness;
this file chooses corpora and writes files (CLAUDE.md rule 3).

The three §10.5 corpus controls
-------------------------------
``--variant matched``        the same semantic ids in every language (the treatment: any
                             cross-language difference cannot be a difference of content);
``--variant unmatched``      an independently drawn id set per language, seeded, same size
                             (if ``matched`` and ``unmatched`` agree, the fingerprint is a
                             property of the language, not of the particular sentences);
``--variant length_matched`` the ``*_lenmatched`` manifest from ``prepare_flores_manifest``
                             (token count is the most obvious confound for a position-0
                             measurement, and this is the subset where it is controlled).

The remaining §10.5 covariates — first-token frequency, punctuation at position 0, Unicode
script, language family — are already computed per row by ``paired_manifests`` and are
summarised into each record's provenance here rather than recomputed. Base vs instruct is a
*config* choice, not a flag: run this twice with the two configs and join on ``lang``.
"""

from __future__ import annotations

import argparse
import gc
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import paired_manifests as pm  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import normalised_depth_band  # noqa: E402

sys.path.insert(0, str(_REPO / "crosslingual_semantics"))
from extract_sink_representations import arch_of, load_config  # noqa: E402

FINGERPRINT_DRIVER_VERSION = "run_multilingual_fingerprint_v1"

VARIANTS = ("matched", "unmatched", "length_matched")

SUMMARY_COLUMNS = (
    "model_tag", "model", "variant", "corpus_variant", "dtype", "language", "corpus_id",
    "manifest_sha256", "baseline_sink", "frac_cells_above_0_2", "carrier_concentration",
    "n_items", "n_failed", "layer_band", "intervention_registry_version",
    "first_token_frequency_mean", "punctuation_at_position_0_frac", "script",
    "language_family", "n_tokens_mean", "status", "warning", "driver_version", "git_sha",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Corpus variants (design §10.5)
# ═══════════════════════════════════════════════════════════════════════════════


def semantic_ids_for(manifest: pm.ParallelManifest, lang: str, variant: str, *,
                     seed: int) -> List[str]:
    """Which ids this language is measured on.

    ``matched`` and ``length_matched`` use the manifest's own id order, so row *i* is the
    same sentence in every language. ``unmatched`` deliberately breaks that: it draws an
    independent permutation per language from a seed that includes the language name, so
    two languages see different sentences of the same corpus. The *sizes* stay equal —
    otherwise the comparison would confound content with sample size.
    """
    ids = list(manifest.semantic_ids)
    if variant in ("matched", "length_matched"):
        return ids
    if variant != "unmatched":
        raise ValueError(f"unknown variant {variant!r}; choose from {VARIANTS}")
    rng = random.Random(f"{seed}|{manifest.manifest_id}|{lang}")
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return shuffled


def covariates(manifest: pm.ParallelManifest, lang: str,
               semantic_ids: Sequence[str]) -> Dict[str, Any]:
    """The §10.5 per-language covariates, read off the manifest rows (never recomputed)."""
    rows = [manifest.row(sid, lang) for sid in semantic_ids]
    if not rows:
        return {}
    frequencies = [float(r.get("first_token_frequency", 0) or 0) for r in rows]
    punctuation = [bool(r.get("punctuation_at_position_0", False)) for r in rows]
    tokens = [int(r.get("n_tokens", 0) or 0) for r in rows]
    scripts = {str(r.get("script", "")) for r in rows}
    families = {str(r.get("language_family", "")) for r in rows}
    return {
        "first_token_frequency_mean": sum(frequencies) / len(frequencies),
        "punctuation_at_position_0_frac": sum(punctuation) / len(punctuation),
        "n_tokens_mean": sum(tokens) / len(tokens),
        # A language should have one dominant script and one family; if the manifest says
        # otherwise, say so in the artefact rather than silently picking the first.
        "script": sorted(scripts)[0] if len(scripts) == 1 else "|".join(sorted(scripts)),
        "language_family": (sorted(families)[0] if len(families) == 1
                            else "|".join(sorted(families))),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════════


def run(config, manifest: pm.ParallelManifest, out_root: Path, *,
        corpus_variant: str = "matched", handle=None,
        langs: Optional[Sequence[str]] = None, resume: bool = True,
        device: Optional[str] = None, with_delta_ce: bool = False,
        cache_dir: Optional[Path] = None, progress: bool = True) -> Dict[str, Any]:
    """One ``FingerprintRecord`` per language, written to ``<out>/<model_tag>/<lang>.json``."""
    import pandas as pd

    started = time.time()
    out_dir = Path(out_root) / config.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    owns_handle = handle is None
    if owns_handle:
        handle = fr.load_handle(arch_of(config.model), config.model,
                                engine=config.engine, dtype=config.dtype,
                                revision=config.model_revision, device=device,
                                tokenizer_name=config.raw.get("tokenizer"))
    try:
        num_layers = handle.nn_engine.num_layers
        band = normalised_depth_band(num_layers, frac=config.band_frac)
        requested = list(langs) if langs else list(manifest.languages)
        unknown = [lang for lang in requested if lang not in manifest.languages]
        if unknown:
            raise ValueError(f"languages {unknown} are not in the manifest")

        rows: List[Dict[str, Any]] = []
        n_written = 0
        n_skipped = 0
        n_failed = 0
        for lang in requested:
            path = out_dir / f"{lang}.json"
            if resume and path.exists():
                n_skipped += 1
                if progress:
                    print(f"  [fingerprint] {lang}: cached, skipped")
                rows.append(_row_from_record(
                    fr.FingerprintRecord.from_json(path.read_text(encoding="utf-8")),
                    config, corpus_variant, lang, manifest, status="cached"))
                continue

            ids = semantic_ids_for(manifest, lang, corpus_variant, seed=config.seed)
            try:
                corpus = cp.flores_corpus(handle.tokenizer, lang, ids,
                                          split=manifest.split, manifest=manifest)
                record = fr.compute_fingerprint(
                    handle, corpus, band=(band[0], band[1]),
                    with_delta_ce=with_delta_ce, progress=False,
                    cache_dir=str(cache_dir) if cache_dir else None,
                    run_id=f"{config.tag}_{corpus_variant}_{lang}",
                    experiment_id=config.experiment_id, condition=lang,
                    seed=config.seed)
            except Exception as exc:
                n_failed += 1
                rows.append({
                    "model_tag": config.tag, "model": config.model,
                    "variant": config.variant, "corpus_variant": corpus_variant,
                    "dtype": config.dtype, "language": lang, "corpus_id": "",
                    "manifest_sha256": manifest.sha256, "status": "failed",
                    "warning": f"{type(exc).__name__}: {exc}",
                    "driver_version": FINGERPRINT_DRIVER_VERSION,
                    "git_sha": prov.git_sha(),
                    **covariates(manifest, lang, ids)})
                if progress:
                    print(f"  [fingerprint] {lang}: FAILED {type(exc).__name__}: {exc}")
                continue

            path.write_text(record.to_json(), encoding="utf-8")
            n_written += 1
            rows.append(_row_from_record(record, config, corpus_variant, lang, manifest))
            if progress:
                print(f"  [fingerprint] {lang}: sink={record.baseline_sink:.4f} "
                      f"({record.n_items} items, {record.n_failed} failed)")

        frame = pd.DataFrame(rows).reindex(columns=list(SUMMARY_COLUMNS))
        summary_csv = out_dir / f"multilingual_fingerprints_{corpus_variant}.csv"
        frame.to_csv(summary_csv, index=False, encoding="utf-8")

        summary = {
            "model_tag": config.tag, "model": config.model, "variant": config.variant,
            "corpus_variant": corpus_variant, "dtype": config.dtype,
            "manifest_id": manifest.manifest_id, "manifest_sha256": manifest.sha256,
            "languages": requested, "layer_band": [int(band[0]), int(band[1])],
            "layer_band_version": band[2]["depth_band_version"],
            "n_written": n_written, "n_skipped": n_skipped, "n_failed": n_failed,
            "summary_csv": str(summary_csv), "with_delta_ce": bool(with_delta_ce),
            "driver_version": FINGERPRINT_DRIVER_VERSION,
            "wallclock_s": round(time.time() - started, 3),
        }
        prov.write_json(out_dir / f"summary_{corpus_variant}.json",
                        {**summary, **prov.provenance_block()})
        return summary
    finally:
        if owns_handle:
            del handle
            gc.collect()


def _row_from_record(record, config, corpus_variant: str, lang: str,
                     manifest: pm.ParallelManifest, *, status: str = "ok"
                     ) -> Dict[str, Any]:
    return {
        "model_tag": config.tag,
        "model": config.model,
        "variant": config.variant,
        "corpus_variant": corpus_variant,
        "dtype": record.dtype,
        "language": lang,
        "corpus_id": record.corpus_id,
        "manifest_sha256": record.manifest_sha256,
        "baseline_sink": record.baseline_sink,
        "frac_cells_above_0_2": record.frac_cells_above_0_2,
        "carrier_concentration": record.carrier_concentration,
        "n_items": record.n_items,
        "n_failed": record.n_failed,
        "layer_band": f"[{record.band[0]},{record.band[1]})",
        "intervention_registry_version": record.intervention_registry_version,
        "status": status,
        "warning": "",
        "driver_version": FINGERPRINT_DRIVER_VERSION,
        "git_sha": prov.git_sha(),
        **covariates(manifest, lang,
                     semantic_ids_for(manifest, lang, corpus_variant, seed=config.seed)),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Per-language intervention fingerprints for E7 (04 §3.4).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None,
                        help="FLORES manifest JSON; defaults to the config's flores.manifest")
    parser.add_argument("--variant", choices=VARIANTS, default="matched",
                        help="which design §10.5 corpus control to run")
    parser.add_argument("--langs", nargs="+", default=["all"])
    parser.add_argument("--out", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--with-delta-ce", action="store_true")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--force", dest="resume", action="store_false")
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    manifest_path = args.manifest
    if manifest_path is None:
        declared = (config.raw.get("flores") or {}).get("manifest")
        if declared is None:
            print("No --manifest given and the config declares no flores.manifest.")
            return 2
        manifest_path = Path(args.config).resolve().parents[1] / declared
    manifest_path = Path(manifest_path)
    if args.variant == "length_matched":
        # The length-matched subset is a *separate manifest* written by
        # prepare_flores_manifest.py, not a filter applied here.
        manifest_path = manifest_path.with_name(
            manifest_path.stem + "_lenmatched" + manifest_path.suffix)
        if not manifest_path.exists():
            print(f"length-matched manifest {manifest_path} not found; run "
                  "prepare_flores_manifest.py first.")
            return 2
    manifest = pm.ParallelManifest.load(manifest_path)

    out = Path(args.out) if args.out else (_REPO / "crosslingual_semantics" /
                                           "results" / "fingerprints")
    langs = None if args.langs == ["all"] else args.langs

    summary = run(config, manifest, out, corpus_variant=args.variant, langs=langs,
                  resume=args.resume, device=args.device,
                  with_delta_ce=args.with_delta_ce,
                  cache_dir=Path(args.cache_dir) if args.cache_dir else None,
                  progress=args.progress)
    print(f"Fingerprinted {summary['model_tag']} ({args.variant}): "
          f"{summary['n_written']} written, {summary['n_skipped']} cached, "
          f"{summary['n_failed']} failed")
    print(f"  -> {summary['summary_csv']}")
    return 1 if summary["n_failed"] and not (summary["n_written"] or
                                             summary["n_skipped"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
