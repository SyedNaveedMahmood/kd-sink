# -*- coding: utf-8 -*-
"""extract_sink_representations.py — per-language sink representations for E7 (WP9).

``04_MODULE_SPEC_e7_crosslingual.md`` §3. Reads a FLORES :class:`ParallelManifest`, runs one
NNsight capture pass per language, and writes
``reps/<model_tag>/<lang>/<object>.npy`` of shape ``[n_sentences, n_layers, dim]`` plus an
``index.json`` and a ``run_config.json``.

    python crosslingual_semantics/extract_sink_representations.py \\
      --config crosslingual_semantics/configs/e7_qwen05_base.yaml \\
      --manifest crosslingual_semantics/results/manifests/flores_devtest.json \\
      --langs all --out crosslingual_semantics/results/reps

Memory discipline is the constraint that shapes this module (``04`` §3.2)
-------------------------------------------------------------------------
300 sentences × 8 languages × 24 layers × 10 objects captured naively is tens of GB and
will not fit alongside the model on a 16 GB card. So:

* **Derived tensors only.** Full hidden states and attention maps are never persisted.
  ``Qmean``/``Rmean`` are reduced *inside* the trace by
  :func:`cross_example_patching.capture_span`, so a full ``[seq, hidden]`` span never
  leaves it.
* **Streamed to memmaps.** Each example is written into an ``np.lib.format.open_memmap``
  the moment it is captured and then dropped —
  ``capture_sources(..., on_item=..., keep=False)`` exists for exactly this.
* ``batch_size=1``, fp32 on CPU, and an explicit ``torch.cuda.empty_cache()`` cadence.

``tests/test_extraction_shapes.py`` asserts the output directory stays under a size bound.
That assertion, not a comment, is what catches an accidental full-state capture.

Three refusals that are load-bearing
------------------------------------
1. **Tokenizer coupling** (``04`` §1). The manifest is tokenizer-specific. If the model's
   tokenizer does not match the one the manifest was built with, extraction *refuses*
   rather than producing token ids that mean something else.
2. **A resumed run must be the same instrument.** The ledger key carries
   ``manifest_sha256``, the model fingerprint, the object set and the layer count; any
   change invalidates the unit and recomputes it.
3. **A partially written language is not reusable.** Units are marked complete only after
   every row is flushed, so an interrupted run recomputes that language rather than
   silently leaving zero rows in the middle of an array.

``--smoke`` runs the whole path on a random tiny Qwen2 and a synthetic manifest, CPU/fp32,
with no downloads.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import cross_example_patching as cep  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import paired_manifests as pm  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import normalised_depth_band  # noqa: E402

EXTRACTOR_VERSION = "extract_sink_representations_v1"

#: Written next to the representations; the retrieval step reads it rather than guessing.
INDEX_FILENAME = "index.json"

#: Append-only resumability ledger, one JSON object per line (``evaluate_transformation``
#: uses the identical idiom for ``eval_units.jsonl``).
LEDGER_FILENAME = "extraction_units.jsonl"


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ExtractionConfig:
    """The instrument, read from an ``e7_qwen*.yaml``. No thresholds live here."""

    path: Optional[Path]
    experiment_id: str
    model: str
    model_revision: Optional[str]
    tag: str
    variant: str
    dtype: str
    engine: str
    attn_implementation: str
    band_frac: Tuple[float, float]
    objects: Tuple[str, ...]
    layers: Any
    batch_size: int
    seed: int
    raw: Dict[str, Any] = field(default_factory=dict)


def load_config(path) -> ExtractionConfig:
    """Load an ``e7_qwen*.yaml``. ``yaml`` is imported here, not at module import."""
    import yaml

    path = Path(path)
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not parse to a mapping")
    return config_from_dict(payload, path=path)


def config_from_dict(payload: Dict[str, Any], *,
                     path: Optional[Path] = None) -> ExtractionConfig:
    """Build a config from an already-parsed mapping (used by ``--smoke`` and the tests)."""
    extraction = payload.get("extraction") or {}
    band_frac = tuple(payload.get("band_frac", (0.25, 0.90)))
    objects = tuple(extraction.get("objects") or cep.CAPTURE_OBJECTS)
    unknown = [o for o in objects if o not in cep.CAPTURE_OBJECTS]
    if unknown:
        raise ValueError(
            f"config lists unknown extraction objects {unknown}; "
            f"choose from {cep.CAPTURE_OBJECTS}")
    return ExtractionConfig(
        path=Path(path) if path is not None else None,
        experiment_id=str(payload.get("experiment_id", "e7")),
        model=str(payload["model"]),
        model_revision=payload.get("model_revision"),
        tag=str(payload.get("tag", "model")),
        variant=str(payload.get("variant", "base")),
        dtype=str(payload.get("dtype", "float32")),
        engine=str(payload.get("engine", "nnsight")),
        attn_implementation=str(payload.get("attn_implementation", "eager")),
        band_frac=(float(band_frac[0]), float(band_frac[1])),
        objects=objects,
        layers=extraction.get("layers", "all"),
        batch_size=int(extraction.get("batch_size", 1)),
        seed=int(payload.get("seed", 42)),
        raw=payload,
    )


def resolve_layers(spec: Any, num_layers: int) -> List[int]:
    """``"all"`` -> every layer; a list -> exactly those, bounds-checked."""
    if spec in (None, "all"):
        return list(range(num_layers))
    layers = [int(x) for x in spec]
    bad = [x for x in layers if not 0 <= x < num_layers]
    if bad:
        raise ValueError(f"config layers {bad} out of range for {num_layers} layers")
    return layers


# ═══════════════════════════════════════════════════════════════════════════════
# Tokenizer coupling (refusal 1)
# ═══════════════════════════════════════════════════════════════════════════════


def check_tokenizer(manifest: pm.ParallelManifest, tokenizer) -> Dict[str, Any]:
    """Refuse a manifest built with a different tokenizer (``04`` §1).

    The manifest stores token ids, not text offsets. Feeding them to a model whose
    tokenizer differs produces a sequence that is *valid* and *wrong* — exactly the kind of
    failure that yields plausible numbers rather than a crash, so it raises.
    """
    declared = manifest.provenance.get("tokenizer")
    actual = str(getattr(tokenizer, "name_or_path", None) or "unknown")
    report = {"manifest_tokenizer": declared, "model_tokenizer": actual,
              "checked": declared not in (None, "unknown")}
    if report["checked"] and declared != actual:
        raise ValueError(
            f"manifest {manifest.manifest_id!r} was built with tokenizer {declared!r} but "
            f"the model's tokenizer is {actual!r}. The manifest stores token ids; joining "
            "across tokenizers silently changes what every sentence is. Build a separate "
            "manifest for this model (04 §1, 'Tokenizer coupling').")
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Object geometry
# ═══════════════════════════════════════════════════════════════════════════════


def object_dim(handle, obj: str) -> int:
    """Flat width one ``obj`` occupies per layer, from the model's own geometry."""
    geom = cep.geometry(handle)
    spec = handle.nn_engine.spec
    surface, _kind, _variant = cep._object_parts(
        obj, rope_inside=bool(spec.rope_applied_inside_attn))
    if surface == "r":
        return int(geom["hidden"])
    if surface == "q":
        return int(geom["num_heads"] * geom["head_dim"])
    return int(geom["num_kv_heads"] * geom["head_dim"])


def object_shape(handle, obj: str) -> Tuple[int, ...]:
    """Structured (pre-flattening) shape, recorded in ``index.json`` for the analysis."""
    geom = cep.geometry(handle)
    spec = handle.nn_engine.spec
    surface, _kind, _variant = cep._object_parts(
        obj, rope_inside=bool(spec.rope_applied_inside_attn))
    if surface == "r":
        return (int(geom["hidden"]),)
    if surface == "q":
        return (int(geom["num_heads"]), int(geom["head_dim"]))
    return (int(geom["num_kv_heads"]), int(geom["head_dim"]))


# ═══════════════════════════════════════════════════════════════════════════════
# Ledger (refusals 2 and 3)
# ═══════════════════════════════════════════════════════════════════════════════


def unit_key(tag: str, lang: str) -> str:
    return f"{tag}|{lang}"


def read_ledger(root: Path) -> Dict[str, Dict[str, Any]]:
    """Last entry per unit wins, matching ``evaluate_transformation.read_ledger``."""
    path = Path(root) / LEDGER_FILENAME
    ledger: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return ledger
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "unit" in row:
            ledger[str(row["unit"])] = row
    return ledger


def append_ledger(root: Path, row: Dict[str, Any]) -> None:
    path = Path(root) / LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def unit_is_complete(ledger: Dict[str, Dict[str, Any]], key: str,
                     expected: Dict[str, Any]) -> bool:
    """A unit counts as done only if it succeeded *and* every instrument field matches."""
    row = ledger.get(key)
    if row is None or row.get("status") != "ok":
        return False
    return all(row.get(field_name) == value for field_name, value in expected.items())


# ═══════════════════════════════════════════════════════════════════════════════
# Writing
# ═══════════════════════════════════════════════════════════════════════════════


class _MemmapWriter:
    """Streams one example's captures into per-object ``.npy`` memmaps.

    One file per ``(model_tag, lang, object)`` of shape ``[n_sentences, n_layers, dim]``,
    fp32. Rows are written as they arrive and the arrays are flushed and closed at the end;
    nothing accumulates in RAM beyond one example's captures.
    """

    def __init__(self, out_dir: Path, *, objects: Sequence[str], layers: Sequence[int],
                 n_items: int, dims: Dict[str, int]):
        self.out_dir = Path(out_dir)
        self.objects = list(objects)
        self.layers = list(layers)
        self.dims = dict(dims)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.arrays: Dict[str, np.memmap] = {}
        for obj in self.objects:
            self.arrays[obj] = np.lib.format.open_memmap(
                self.out_dir / f"{obj}.npy", mode="w+", dtype=np.float32,
                shape=(int(n_items), len(self.layers), int(self.dims[obj])))
        self.row_of: Dict[str, int] = {}
        self.n_written = 0

    def write(self, item_id: str, cache) -> None:
        row = self.row_of[item_id]
        for obj in self.objects:
            array = self.arrays[obj]
            for layer_index, layer in enumerate(self.layers):
                tensor = cache.tensors[cep.PatchSite(obj, layer).key()]
                array[row, layer_index, :] = tensor.reshape(-1).numpy()
        self.n_written += 1

    def close(self) -> None:
        for array in self.arrays.values():
            array.flush()
            del array
        self.arrays.clear()
        gc.collect()


# ═══════════════════════════════════════════════════════════════════════════════
# Extraction
# ═══════════════════════════════════════════════════════════════════════════════


def extract(config: ExtractionConfig, manifest: pm.ParallelManifest, out_root: Path, *,
            handle=None, langs: Optional[Sequence[str]] = None, resume: bool = True,
            device: Optional[str] = None, tokenizer=None,
            progress: bool = True) -> Dict[str, Any]:
    """Run extraction for every requested language. Returns the summary dict.

    ``handle`` may be supplied already-loaded (the smoke path and the tests do this); when
    it is ``None`` the model is loaded from ``config.model``.
    """
    started = time.time()
    out_root = Path(out_root)
    owns_handle = handle is None
    if owns_handle:
        handle = fr.load_handle(
            arch_of(config.model), config.model, engine=config.engine,
            dtype=config.dtype, revision=config.model_revision, device=device,
            tokenizer_name=config.raw.get("tokenizer"))
    tokenizer = tokenizer if tokenizer is not None else handle.tokenizer

    try:
        tokenizer_report = check_tokenizer(manifest, tokenizer)
        num_layers = handle.nn_engine.num_layers
        layers = resolve_layers(config.layers, num_layers)
        band_start, band_end, band_meta = normalised_depth_band(
            num_layers, frac=config.band_frac)
        fingerprint = cep.model_fingerprint(handle)
        dims = {obj: object_dim(handle, obj) for obj in config.objects}
        shapes = {obj: list(object_shape(handle, obj)) for obj in config.objects}

        requested = list(langs) if langs else list(manifest.languages)
        unknown = [lang for lang in requested if lang not in manifest.languages]
        if unknown:
            raise ValueError(f"languages {unknown} are not in the manifest "
                             f"({manifest.languages})")

        model_root = out_root / config.tag
        ledger = read_ledger(model_root) if resume else {}
        expected = {
            "manifest_sha256": manifest.sha256,
            "model_fingerprint": fingerprint,
            "objects": list(config.objects),
            "layers": list(layers),
            "dtype": config.dtype,
            "extractor_version": EXTRACTOR_VERSION,
        }

        summary_langs: List[Dict[str, Any]] = []
        n_skipped = 0
        n_failed = 0
        for lang in requested:
            key = unit_key(config.tag, lang)
            lang_dir = model_root / lang
            if resume and unit_is_complete(ledger, key, expected) and \
                    (lang_dir / INDEX_FILENAME).exists():
                n_skipped += 1
                if progress:
                    print(f"  [extract] {lang}: complete, skipped")
                continue

            try:
                info = _extract_language(
                    handle, config, manifest, lang, lang_dir,
                    layers=layers, dims=dims, shapes=shapes, progress=progress)
            except Exception as exc:  # failures are recorded, never silent
                n_failed += 1
                append_ledger(model_root, {
                    "unit": key, "status": "failed",
                    "warning": f"{type(exc).__name__}: {exc}",
                    "measured_utc": prov.utc_now()})
                summary_langs.append({"language": lang, "status": "failed",
                                      "warning": f"{type(exc).__name__}: {exc}"})
                if progress:
                    print(f"  [extract] {lang}: FAILED {type(exc).__name__}: {exc}")
                continue

            append_ledger(model_root, {"unit": key, "status": "ok",
                                       "measured_utc": prov.utc_now(), **expected})
            summary_langs.append({"language": lang, "status": "ok", **info})
            if progress:
                print(f"  [extract] {lang}: {info['n_items']} items x {len(layers)} layers")

        summary = {
            "experiment_id": config.experiment_id,
            "model_tag": config.tag,
            "model": config.model,
            "variant": config.variant,
            "dtype": config.dtype,
            "engine": config.engine,
            "device": str(cep._model_device(handle)),
            "manifest_id": manifest.manifest_id,
            "manifest_sha256": manifest.sha256,
            "model_fingerprint": fingerprint,
            "objects": list(config.objects),
            "object_shapes": shapes,
            "layers": list(layers),
            "num_layers": num_layers,
            "layer_band": [int(band_start), int(band_end)],
            "layer_band_depth_interval": list(band_meta["depth_interval"]),
            "layer_band_version": band_meta["depth_band_version"],
            "tokenizer_check": tokenizer_report,
            "languages": summary_langs,
            "n_languages": len(requested),
            "n_skipped": n_skipped,
            "n_failed": n_failed,
            "seed": config.seed,
            "config_path": str(config.path) if config.path else None,
            "extractor_version": EXTRACTOR_VERSION,
            "wallclock_s": round(time.time() - started, 3),
        }
        prov.write_json(model_root / "run_config.json",
                        {**summary, **prov.provenance_block()})
        return summary
    finally:
        if owns_handle:
            del handle
            gc.collect()


#: HF architecture class -> ``ARCH_SPECS`` key, matching ``evaluate_transformation``.
ARCH_BY_CLASS: Dict[str, str] = {
    "Qwen2ForCausalLM": "qwen",
    "GPTNeoForCausalLM": "neo",
    "GPT2LMHeadModel": "gpt2",
    "OPTForCausalLM": "opt",
}


def arch_of(model_name_or_path: str) -> str:
    """Resolve the ``ARCH_SPECS`` key from a local checkpoint's config, or the HF id.

    A local directory is authoritative — its ``config.json`` names the architecture. Only
    when there is no local config does this fall back to the id, and an unrecognised id
    raises rather than defaulting: E7's capture surface is architecture-specific (RoPE
    inside attention, grouped KV), and guessing it wrong produces wrong tensors, not a
    crash.
    """
    local = Path(model_name_or_path) / "config.json"
    if local.exists():
        payload = json.loads(local.read_text(encoding="utf-8"))
        for name in payload.get("architectures") or []:
            if name in ARCH_BY_CLASS:
                return ARCH_BY_CLASS[name]
        raise ValueError(f"{local} declares architectures "
                         f"{payload.get('architectures')}, none of which map to an "
                         f"ArchSpec ({sorted(ARCH_BY_CLASS)})")
    lowered = str(model_name_or_path).lower()
    for token, arch in (("qwen", "qwen"), ("gpt-neo", "neo"), ("opt-", "opt"),
                        ("gpt2", "gpt2")):
        if token in lowered:
            return arch
    raise ValueError(
        f"cannot infer an ArchSpec key from {model_name_or_path!r}. E7 is specified for "
        "Qwen2.5 (04 §7); point --config at a Qwen model or a local checkpoint whose "
        "config.json names its architecture.")


def _extract_language(handle, config: ExtractionConfig, manifest: pm.ParallelManifest,
                      lang: str, lang_dir: Path, *, layers: Sequence[int],
                      dims: Dict[str, int], shapes: Dict[str, List[int]],
                      progress: bool = True) -> Dict[str, Any]:
    """Capture one language into ``lang_dir``. Writes ``index.json`` only on full success."""
    corpus = cp.flores_corpus(handle.tokenizer, lang, manifest.semantic_ids,
                              split=manifest.split, manifest=manifest)
    sites = [cep.PatchSite(obj, layer) for obj in config.objects for layer in layers]

    # `flores_corpus` names its items "<semantic_id>:<lang>". The *semantic id* is the
    # cross-language join key — the whole experiment rests on row i being the same sentence
    # in every language — so the index is keyed by it, and the language-qualified item id
    # is recorded separately rather than being used as the key.
    semantic_of = {item.item_id: str(item.meta.get("semantic_id", item.item_id))
                   for item in corpus.items}

    writer = _MemmapWriter(lang_dir, objects=config.objects, layers=layers,
                           n_items=len(corpus.items), dims=dims)
    positions: Dict[str, Dict[str, int]] = {}
    token_counts: Dict[str, int] = {}
    try:
        for row, item in enumerate(corpus.items):
            writer.row_of[item.item_id] = row

        def _on_item(item_id: str, cache) -> None:
            writer.write(item_id, cache)
            semantic_id = semantic_of[item_id]
            positions[semantic_id] = dict(cache.resolved_positions)
            token_counts[semantic_id] = int(cache.seq_len)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        cep.capture_sources(handle, corpus, sites, batch_size=config.batch_size,
                            on_item=_on_item, keep=False)

        if writer.n_written != len(corpus.items):
            raise RuntimeError(
                f"{lang}: wrote {writer.n_written} of {len(corpus.items)} rows; refusing to "
                "index a partially filled array")
    finally:
        writer.close()

    index = {
        "language": lang,
        "corpus_id": corpus.corpus_id,
        "manifest_id": manifest.manifest_id,
        "manifest_sha256": manifest.sha256,
        "semantic_ids": [semantic_of[item.item_id] for item in corpus.items],
        "row_of": {semantic_of[item_id]: row for item_id, row in writer.row_of.items()},
        "item_id_of": {semantic_of[item_id]: item_id for item_id in writer.row_of},
        "layers": list(layers),
        "objects": list(config.objects),
        "object_shapes": shapes,
        "object_dims": {obj: int(dims[obj]) for obj in config.objects},
        "dtype": "float32",
        "storage_dtype": "float32",
        "model_dtype": config.dtype,
        "n_items": len(corpus.items),
        "token_counts": token_counts,
        "resolved_positions": positions,
        "extractor_version": EXTRACTOR_VERSION,
    }
    prov.write_json(lang_dir / INDEX_FILENAME, {**index, **prov.provenance_block()})
    return {"n_items": len(corpus.items), "n_layers": len(layers),
            "dir": str(lang_dir), "corpus_id": corpus.corpus_id}


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke
# ═══════════════════════════════════════════════════════════════════════════════


def smoke_config(model_path: str, *, tag: str = "smoke_qwen",
                 objects: Optional[Sequence[str]] = None) -> ExtractionConfig:
    """Config for a random tiny Qwen2 — CPU, fp32, every capture object, no downloads."""
    return config_from_dict({
        "experiment_id": "e7",
        "model": model_path,
        "model_revision": None,
        "tag": tag,
        "variant": "base",
        "dtype": "float32",
        "engine": "nnsight",
        "attn_implementation": "eager",
        "band_frac": [0.25, 0.90],
        "extraction": {"objects": list(objects or cep.CAPTURE_OBJECTS),
                       "layers": "all", "batch_size": 1},
        "seed": 42,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Extract per-language sink representations for E7 (04 §3).")
    parser.add_argument("--config", required=True,
                        help="crosslingual_semantics/configs/e7_*.yaml")
    parser.add_argument("--manifest", default=None,
                        help="FLORES manifest JSON; defaults to the config's flores.manifest")
    parser.add_argument("--langs", nargs="+", default=["all"],
                        help="languages to extract, or 'all'")
    parser.add_argument("--out", default=None,
                        help="output root (default crosslingual_semantics/results/reps)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="skip languages already complete (the default)")
    parser.add_argument("--force", dest="resume", action="store_false",
                        help="recompute every language")
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
    manifest = pm.ParallelManifest.load(manifest_path)

    out = Path(args.out) if args.out else (_REPO / "crosslingual_semantics" /
                                           "results" / "reps")
    langs = None if args.langs == ["all"] else args.langs

    summary = extract(config, manifest, out, langs=langs, resume=args.resume,
                      device=args.device, progress=args.progress)
    n_ok = sum(1 for row in summary["languages"] if row.get("status") == "ok")
    print(f"Extracted {config.tag}: {n_ok} written, "
          f"{summary['n_skipped']} skipped, {summary['n_failed']} failed")
    print(f"  -> {out / config.tag}")
    # Non-zero only when nothing at all came out: a few failed languages are recorded as
    # rows and the run is still resumable, which is the point of the ledger.
    return 1 if summary["n_failed"] and not (n_ok or summary["n_skipped"]) else 0


if __name__ == "__main__":
    raise SystemExit(main())
