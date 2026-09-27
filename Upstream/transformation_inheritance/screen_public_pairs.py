# -*- coding: utf-8 -*-
"""screen_public_pairs.py — fingerprint public model pairs, no training (WP6).

``03_MODULE_SPEC_e6_transformation.md`` §5 / design §9.9. The cheapest evidence in the
whole project: every pair here already exists on the hub, so this answers "do a model and
its distilled or instruction-tuned sibling share a sink fingerprint?" without a single
optimiser step. `07` P6 says it should run **first** in Phase 1.

    python transformation_inheritance/screen_public_pairs.py --pairs gpt2_distilgpt2

Three properties worth stating
------------------------------
* **Nothing is computed here.** Every number comes from
  :func:`fingerprint_runner.compute_fingerprint`, which dispatches into the frozen
  instrument. This file selects models, orders the loop and writes a CSV.
* **A pair is fingerprinted on its mutual intervention set.**
  :func:`fingerprint_runner.mutual_interventions` is computed per pair at runtime
  (design-delta D2), so a comparison never spans an intervention only one member supports —
  ``int_b`` is absent on GPT-Neo, ``int_e`` on Qwen, and a union would silently compare a
  measured value against a missing one.
* **The 3B pair is not run by default.** Design §9.9 gates it on 0.5B and 1.5B agreeing in
  direction, and that is a judgement made after reading this table, not a flag flipped
  ahead of it.

Failures are rows (``05`` §7.2): a model that will not load or a fingerprint that raises
writes a row with ``status`` set and the loop continues.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import inheritance_metrics as im  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import band_agreement_report, normalised_depth_band  # noqa: E402

SCREENING_VERSION = "screen_public_pairs_v1"

#: design §9.9's pairs. ``a`` is the reference and ``b`` the transformed sibling, so a
#: positive ``fingerprint_cosine_a_to_b`` reads the same way across every row.
PUBLIC_PAIRS: Dict[str, Dict[str, Any]] = {
    "gpt2_distilgpt2": {
        "arch": "gpt2", "a": "gpt2", "b": "distilbert/distilgpt2",
        "relation": "distillation",
        "note": "the E6A question asked of two checkpoints that already exist",
    },
    "qwen05_base_instruct": {
        "arch": "qwen", "a": "Qwen/Qwen2.5-0.5B", "b": "Qwen/Qwen2.5-0.5B-Instruct",
        "relation": "instruction_tuning",
        "note": "the E6B question — adaptation, not distillation — at 0.5B",
    },
    "qwen15_base_instruct": {
        "arch": "qwen", "a": "Qwen/Qwen2.5-1.5B", "b": "Qwen/Qwen2.5-1.5B-Instruct",
        "relation": "instruction_tuning",
        "note": "the same question at 1.5B; two sizes is what design §10.11 calls a "
                "replication",
    },
    # Run only if the two sizes above agree in direction (design §9.9). Named here so the
    # decision is "did the smaller pair agree?" rather than "which model should I add?".
    "qwen3_base_instruct": {
        "arch": "qwen", "a": "Qwen/Qwen2.5-3B", "b": "Qwen/Qwen2.5-3B-Instruct",
        "relation": "instruction_tuning", "conditional": True,
        "note": "CONDITIONAL — design §9.9 gates this on 0.5B and 1.5B agreeing",
    },
}

DEFAULT_PAIRS: Tuple[str, ...] = ("gpt2_distilgpt2", "qwen05_base_instruct",
                                  "qwen15_base_instruct")

ROW_COLUMNS: Tuple[str, ...] = (
    "pair_id", "relation", "role", "model", "arch", "corpus_id", "manifest_sha256",
    "status", "warning", "num_layers", "num_heads", "layer_band", "band_depth_interval",
    "baseline_sink", "frac_cells_above_0_2", "carrier_concentration", "n_keys_used",
    "fingerprint_json", "depth_profile_16_json", "top_carriers_json",
    "fingerprint_cosine_a_to_b", "fingerprint_spearman_a_to_b", "fingerprint_l1_a_to_b",
    "category_agreement_a_to_b", "topology_wasserstein_a_to_b",
    "topology_spearman_a_to_b", "carrier_jaccard_a_to_b", "band_depth_mismatch",
    "n_items", "n_failed", "wallclock_s", "engine", "dtype",
    "intervention_registry_version", "screening_version", "measured_utc", "git_sha",
)


def build_corpus(tokenizer, *, n_blocks: int, seed: int, smoke: bool):
    """The corpus every pair is screened on.

    ``frozen_e1_corpus`` deliberately: it is the same input distribution E1–E5 measured, so
    a screening fingerprint is directly readable against the published tables rather than
    against a set drawn only for this script.
    """
    if smoke:
        return cp.synthetic_corpus(tokenizer, "shuffled_natural", n_blocks, seed,
                                   cut_length=16)
    return cp.frozen_e1_corpus(tokenizer)


def _row(pair_id: str, spec: Dict[str, Any], role: str, model_id: str, *,
         status: str, warning: str = "", record=None, comparison=None,
         corpus_id: str = "", engine: str = "nnsight", dtype: str = "float32"
         ) -> Dict[str, Any]:
    import json

    row = {column: None for column in ROW_COLUMNS}
    row.update({
        "pair_id": pair_id, "relation": spec.get("relation"), "role": role,
        "model": model_id, "arch": spec.get("arch"), "status": status,
        "warning": warning, "corpus_id": corpus_id, "engine": engine, "dtype": dtype,
        "screening_version": SCREENING_VERSION, "measured_utc": prov.utc_now(),
        "git_sha": prov.git_sha(),
        "intervention_registry_version": fr.INTERVENTION_REGISTRY_VERSION,
    })
    if record is not None:
        row.update({
            "corpus_id": record.corpus_id, "manifest_sha256": record.manifest_sha256,
            "num_layers": record.num_layers, "num_heads": record.num_heads,
            "layer_band": f"[{record.band[0]},{record.band[1]})",
            "band_depth_interval": json.dumps(list(record.band_depth)),
            "baseline_sink": record.baseline_sink,
            "frac_cells_above_0_2": record.frac_cells_above_0_2,
            "carrier_concentration": record.carrier_concentration,
            "fingerprint_json": json.dumps(record.fingerprint),
            "depth_profile_16_json": json.dumps(record.depth_profile_16),
            "top_carriers_json": json.dumps(record.top_carrier_heads),
            "n_items": record.n_items, "n_failed": record.n_failed,
            "wallclock_s": record.wallclock_s, "engine": record.engine,
            "dtype": record.dtype,
            "intervention_registry_version": record.intervention_registry_version,
        })
    if comparison:
        row.update(comparison)
    return row


def screen_pair(pair_id: str, spec: Dict[str, Any], *, corpus, cache_dir: Path,
                engine: str = "nnsight", dtype: str = "float32",
                device: Optional[str] = None, smoke: bool = False,
                progress: bool = True) -> List[Dict[str, Any]]:
    """Fingerprint both members of one pair and record their distances.

    The comparison columns are attached to the ``b`` row, so one row per pair carries the
    answer and the ``a`` row stays a plain fingerprint of the reference.
    """
    import numpy as np

    rows: List[Dict[str, Any]] = []
    handles, records = {}, {}
    for role in ("a", "b"):
        model_id = spec[role]
        try:
            handles[role] = fr.load_handle(
                spec["arch"], model_id, engine=engine, dtype=dtype, device=device,
                local_files_only=smoke)
        except Exception as exc:
            rows.append(_row(pair_id, spec, role, model_id, status="model_load_failed",
                             warning=f"{type(exc).__name__}: {exc}", engine=engine,
                             dtype=dtype))
            return rows

    keys = fr.mutual_interventions(handles["a"], handles["b"])
    for role in ("a", "b"):
        model_id = spec[role]
        handle = handles[role]
        band_start, band_end, _meta = normalised_depth_band(handle.nn_engine.num_layers)
        try:
            records[role] = fr.compute_fingerprint(
                handle, corpus, band=(band_start, band_end), interventions=keys,
                cache_dir=cache_dir, run_id=f"screen:{pair_id}:{role}",
                experiment_id="screening", condition=role, seed=0, progress=False)
        except Exception as exc:
            rows.append(_row(pair_id, spec, role, model_id, status="fingerprint_failed",
                             warning=f"{type(exc).__name__}: {exc}", engine=engine,
                             dtype=dtype, corpus_id=corpus.corpus_id))
            return rows

    a, b = records["a"], records["b"]
    fp = im.fingerprint_report(a, b, keys)
    topo = im.topology_report(a.depth_profile_16, b.depth_profile_16)
    band = band_agreement_report(a.num_layers, b.num_layers)
    try:
        jaccard = im.weighted_jaccard(np.asarray(a.per_head_sink),
                                      np.asarray(b.per_head_sink))
        jaccard_warning = ""
    except ValueError as exc:
        # Unequal head counts is a real incomparability, not a number to coerce (02 §4.3).
        jaccard, jaccard_warning = None, f"carrier_jaccard: {exc}"

    comparison = {
        "fingerprint_cosine_a_to_b": fp["cosine"],
        "fingerprint_spearman_a_to_b": fp["spearman"],
        "fingerprint_l1_a_to_b": fp["l1_normalised"],
        "category_agreement_a_to_b": fp["category_agreement"],
        "topology_wasserstein_a_to_b": topo["wasserstein"],
        "topology_spearman_a_to_b": topo["spearman"],
        "carrier_jaccard_a_to_b": jaccard,
        "band_depth_mismatch": band["max_mismatch"],
        "n_keys_used": fp["n_keys_used"],
    }
    rows.append(_row(pair_id, spec, "a", spec["a"], status="ok", record=a,
                     engine=engine, dtype=dtype))
    rows.append(_row(pair_id, spec, "b", spec["b"], status="ok", record=b,
                     comparison=comparison, warning=jaccard_warning, engine=engine,
                     dtype=dtype))
    if progress:
        print(f"  {pair_id}: cos={fp['cosine']:.4f} "
              f"wasserstein={topo['wasserstein']:.4f} "
              f"band_mismatch={band['max_mismatch']:.3f} n_keys={fp['n_keys_used']}")
    for handle in handles.values():
        del handle
    return rows


def screen(pairs: Sequence[str], out_dir: Path, *, engine: str = "nnsight",
           dtype: str = "float32", device: Optional[str] = None, smoke: bool = False,
           n_blocks: int = 8, seed: int = 0, tokenizer_name: Optional[str] = None,
           progress: bool = True) -> Dict[str, Any]:
    """Screen every named pair and write ``screening_fingerprints.csv``."""
    import pandas as pd
    from transformers import AutoTokenizer

    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    unknown = [p for p in pairs if p not in PUBLIC_PAIRS]
    if unknown:
        raise SystemExit(f"unknown pair id(s) {unknown}; known: {sorted(PUBLIC_PAIRS)}")

    # One corpus for every pair, so cross-pair rows are joinable on manifest_sha256.
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_name or ("gpt2" if not smoke else tokenizer_name or "gpt2"))
    corpus = build_corpus(tokenizer, n_blocks=n_blocks, seed=seed, smoke=smoke)

    rows: List[Dict[str, Any]] = []
    for pair_id in pairs:
        rows.extend(screen_pair(pair_id, PUBLIC_PAIRS[pair_id], corpus=corpus,
                                cache_dir=out_dir / "fingerprints", engine=engine,
                                dtype=dtype, device=device, smoke=smoke,
                                progress=progress))

    frame = pd.DataFrame(rows).reindex(columns=list(ROW_COLUMNS))
    path = out_dir / "screening_fingerprints.csv"
    frame.to_csv(path, index=False, encoding="utf-8")

    report = {
        "pairs": list(pairs), "n_rows": int(len(frame)),
        "n_failed": int((frame["status"] != "ok").sum()),
        "csv": str(path), "corpus_id": corpus.corpus_id,
        "manifest_sha256": corpus.manifest_sha256,
        "engine": engine, "dtype": dtype, "smoke": smoke,
        "screening_version": SCREENING_VERSION,
        "wallclock_s": round(time.time() - started, 3),
        "conditional_pairs_not_run": [p for p, s in PUBLIC_PAIRS.items()
                                      if s.get("conditional") and p not in pairs],
        "note": "No training. Every number comes from compute_fingerprint, which "
                "dispatches into the frozen instrument. Design §9.9 gates the 3B pair on "
                "0.5B and 1.5B agreeing in direction — that is a judgement made after "
                "reading this table.",
    }
    prov.write_json(out_dir / "screening_summary.json",
                    {**report, **prov.provenance_block()})
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--pairs", nargs="*", default=list(DEFAULT_PAIRS),
                        help=f"pair ids to screen; known: {sorted(PUBLIC_PAIRS)}")
    parser.add_argument("--out", default=str(_REPO / "transformation_inheritance" /
                                             "results" / "screening"))
    parser.add_argument("--engine", default="nnsight", choices=("nnsight",))
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--device", default=None)
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer for the shared corpus (default gpt2)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--blocks", type=int, default=8,
                        help="synthetic corpus size under --smoke; ignored otherwise")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic corpus and local models only, no downloads")
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    report = screen(args.pairs, Path(args.out), engine=args.engine, dtype=args.dtype,
                    device=args.device, smoke=args.smoke, n_blocks=args.blocks,
                    seed=args.seed, tokenizer_name=args.tokenizer,
                    progress=args.progress)
    print(f"Public-pair screening -> {report['csv']}")
    print(f"  pairs={len(report['pairs'])} rows={report['n_rows']} "
          f"failed={report['n_failed']} corpus={report['corpus_id']}")
    if report["conditional_pairs_not_run"]:
        print(f"  not run (conditional, design §9.9): "
              f"{report['conditional_pairs_not_run']}")
    return 1 if report["n_failed"] and not (report["n_rows"] - report["n_failed"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
