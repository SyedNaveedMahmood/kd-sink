# -*- coding: utf-8 -*-
"""block_corpus_cache.py — persist a packed block corpus so it is built once, not per run.

Packing a training corpus is deterministic in ``(dataset, split, block_size, seed,
eos_between, document window, tokenizer)`` and expensive: the E6A-GPT2 training window is
400,000 OpenWebText documents, and reaching the *validation* window means streaming past
all 400,000 of them first. Measured on the TinyStories train split, the equivalent work is
17-30 minutes and ~3 GB of RSS **before the first optimiser step** — and it is paid again
on every re-run, every resume, and every ``evaluate_transformation.py`` invocation that
rebuilds the same corpora.

Nothing here changes what is packed. It stores the output of a loader call and returns it
again, and **every hit re-derives ``prov.sha256_int_rows`` over the blocks it just read and
refuses if it does not match what was stored**. A corpus cache that can silently serve the
wrong tokens is worse than no cache at all: `manifest_sha256` is the join key for every
cached artefact in the project (``05`` §7.1), so a mismatch would not surface as a crash
but as two incomparable sets of results (CLAUDE.md trap 19).

Two properties worth stating because they are the reason the design is shaped this way:

* **``n_blocks`` is not part of the key.** Packing is a prefix operation — the loop emits
  blocks in order and stops when the cap is reached — so one cached full pack serves every
  smaller request by slicing, exactly as the loaders' own ``blocks[:n_blocks]`` does. A
  cache keyed on ``n_blocks`` would re-stream 400,000 documents for every distinct request
  size, which is most of what this module exists to avoid.
* **A larger request than what is cached is a miss, not a truncation.** Serving 300 blocks
  when 2,300 were asked for would be a silently short corpus.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

try:                                    # the repo's dual-import idiom
    from . import provenance as prov
except ImportError:                     # pragma: no cover - script-style sys.path use
    import provenance as prov

#: Bumped when the on-disk layout changes, so an old entry misses instead of mis-loading.
CACHE_FORMAT_VERSION = "block_corpus_cache_v1"

#: Manifest rows written per parquet row group — mirrors ``train_distillation``'s own
#: chunking, for the same reason: a 3.5M-row frame is several GB before parquet sees it.
MANIFEST_CHUNK_ROWS = 50_000

#: Loader keyword arguments that are **performance-only** and are therefore excluded from
#: the key while still being passed to the loader.
#:
#: A cache key must cover everything that can change the content and *nothing that cannot*.
#: ``tokenizer_batch_size`` only decides how many documents go into one tokenizer call, and
#: ``tests/test_batched_tokenisation_equivalence.py`` proves that changes no token, no block
#: boundary and no digest. Including it would give the trainer (which batches, because it
#: packs ~3.1M blocks) and a corpus provider (which does not, because it packs ~2,300) two
#: separate entries for one corpus — each re-streaming the whole document window, which is
#: the cost this module exists to remove. Anything whose effect on the output is not proven
#: belongs in the key.
NON_KEY_LOADER_KWARGS = frozenset({"tokenizer_batch_size"})


class CorpusCacheError(RuntimeError):
    """A cache entry exists but cannot be trusted. Never swallowed into a silent repack."""


# ═══════════════════════════════════════════════════════════════════════════════
# Key
# ═══════════════════════════════════════════════════════════════════════════════


def tokenizer_identity(tokenizer) -> Dict[str, Any]:
    """The tokenizer facts that can change a packed corpus, read rather than assumed.

    ``name_or_path`` alone is not enough — two tokenizers can share an id and differ in
    added tokens — and hashing the vocabulary of a 50k-entry tokenizer on every call is not
    worth it. ``len()``, ``vocab_size``, the class name and the EOS id together catch every
    difference that could change which tokens are packed or where blocks break.
    """
    return {
        "name_or_path": str(getattr(tokenizer, "name_or_path", "unknown")),
        "class": type(tokenizer).__name__,
        "len": int(len(tokenizer)) if hasattr(tokenizer, "__len__") else None,
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0) or 0),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
    }


def key_loader_kwargs(loader_kwargs: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """The content-determining subset of ``loader_kwargs``, in a stable order."""
    kwargs = loader_kwargs or {}
    return {k: kwargs[k] for k in sorted(kwargs) if k not in NON_KEY_LOADER_KWARGS}


def cache_key(*, dataset: str, split: str, block_size: int, seed: int,
              eos_between: bool, loader_kwargs: Optional[Dict[str, Any]],
              tokenizer) -> str:
    """The sha256 that names one packed corpus.

    ``n_blocks`` is deliberately absent (packing is a prefix operation), and so is anything
    in :data:`NON_KEY_LOADER_KWARGS`. Sorting the remaining kwargs means
    ``{"a": 1, "b": 2}`` and ``{"b": 2, "a": 1}`` name one entry, not two.
    """
    payload = {
        "format": CACHE_FORMAT_VERSION,
        "dataset": str(dataset),
        "split": str(split),
        "block_size": int(block_size),
        "seed": int(seed),
        "eos_between": bool(eos_between),
        "loader_kwargs": key_loader_kwargs(loader_kwargs),
        "tokenizer": tokenizer_identity(tokenizer),
    }
    return prov.sha256_json(payload)


# ═══════════════════════════════════════════════════════════════════════════════
# Read / write
# ═══════════════════════════════════════════════════════════════════════════════


def _entry_dir(cache_root, key: str) -> Path:
    return Path(cache_root) / key


def _read_manifest(path: Path) -> List[dict]:
    import pandas as pd

    frame = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    rows: List[dict] = []
    for record in frame.to_dict("records"):
        indices = record["source_story_indices"]
        if isinstance(indices, str):
            indices = json.loads(indices)
        rows.append({
            "block_index": int(record["block_index"]),
            "source_story_indices": [int(x) for x in indices],
            "n_tokens": int(record["n_tokens"]),
            "n_eos": int(record["n_eos"]),
        })
    return rows


def _write_manifest(manifest: Sequence[dict], path: Path) -> Path:
    import pandas as pd

    def _chunk(start: int, stop: int) -> "pd.DataFrame":
        return pd.DataFrame([{
            "block_index": int(row["block_index"]),
            "source_story_indices": json.dumps(
                [int(x) for x in row["source_story_indices"]]),
            "n_tokens": int(row["n_tokens"]),
            "n_eos": int(row["n_eos"]),
        } for row in manifest[start:stop]])

    columns = ["block_index", "source_story_indices", "n_tokens", "n_eos"]
    total = len(manifest)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        try:
            for start in range(0, total, MANIFEST_CHUNK_ROWS):
                table = pa.Table.from_pandas(
                    _chunk(start, min(start + MANIFEST_CHUNK_ROWS, total)),
                    preserve_index=False)
                if writer is None:
                    writer = pq.ParquetWriter(path, table.schema)
                writer.write_table(table)
            if writer is None:
                pd.DataFrame(columns=columns).to_parquet(path, index=False)
        finally:
            if writer is not None:
                writer.close()
        return path
    except ImportError:  # pragma: no cover - only when pyarrow is unavailable
        fallback = path.with_suffix(".csv")
        first = True
        for start in range(0, total, MANIFEST_CHUNK_ROWS):
            _chunk(start, min(start + MANIFEST_CHUNK_ROWS, total)).to_csv(
                fallback, index=False, encoding="utf-8",
                mode="w" if first else "a", header=first)
            first = False
        if first:
            pd.DataFrame(columns=columns).to_csv(fallback, index=False, encoding="utf-8")
        return fallback


def read_entry(cache_root, key: str, n_blocks: Optional[int]
               ) -> Optional[Tuple[np.ndarray, List[dict], str]]:
    """Return ``(blocks, manifest, digest)`` for a usable entry, else ``None``.

    ``None`` means "not usable, repack" — absent, or packed to fewer blocks than asked for.
    A *corrupt* entry raises :class:`CorpusCacheError` instead, because silently repacking
    over a digest mismatch would hide exactly the failure this verification exists to catch.
    """
    entry = _entry_dir(cache_root, key)
    meta_path = entry / "meta.json"
    blocks_path = entry / "blocks.npy"
    if not (meta_path.exists() and blocks_path.exists()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise CorpusCacheError(f"{meta_path} is unreadable: {exc}") from exc
    if meta.get("format") != CACHE_FORMAT_VERSION or meta.get("key") != key:
        return None
    if n_blocks is not None and int(meta.get("n_blocks", 0)) < int(n_blocks):
        # A shorter pack is a MISS, never a truncated hit.
        return None

    manifest_path = entry / str(meta.get("manifest_file", "manifest.parquet"))
    if not manifest_path.exists():
        raise CorpusCacheError(f"{entry}: blocks.npy present but {manifest_path.name} is not")

    blocks = np.load(blocks_path, mmap_mode=None)
    manifest = _read_manifest(manifest_path)
    if len(manifest) != len(blocks):
        raise CorpusCacheError(
            f"{entry}: {len(blocks)} blocks but {len(manifest)} manifest rows")

    # The verification, on every hit. Cheap next to repacking (~1.3 min for 3.57M rows,
    # against 17-30 min) and it is the only thing standing between a truncated or
    # half-written cache file and a run that trains on it without noticing.
    digest = prov.sha256_int_rows(blocks)
    if digest != meta.get("manifest_sha256"):
        raise CorpusCacheError(
            f"{entry}: cached blocks hash to {digest} but the entry records "
            f"{meta.get('manifest_sha256')}. The entry is corrupt or was written by a "
            "different packer; delete it rather than training on it.")

    if n_blocks is not None:
        blocks, manifest = blocks[:n_blocks], manifest[:n_blocks]
        digest = prov.sha256_int_rows(blocks)
    return blocks, manifest, digest


def write_entry(cache_root, key: str, blocks: np.ndarray, manifest: Sequence[dict],
                digest: str, *, description: Dict[str, Any]) -> Path:
    """Write one entry. Built in a temporary directory and renamed, so a hit is complete."""
    entry = _entry_dir(cache_root, key)
    staging = entry.with_name(entry.name + ".partial")
    if staging.exists():
        shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True, exist_ok=True)

    np.save(staging / "blocks.npy", np.ascontiguousarray(blocks))
    manifest_path = _write_manifest(manifest, staging / "manifest.parquet")
    prov.write_json(staging / "meta.json", {
        "format": CACHE_FORMAT_VERSION,
        "key": key,
        "n_blocks": int(len(blocks)),
        "block_size": int(blocks.shape[1]) if blocks.ndim == 2 else None,
        "manifest_sha256": digest,
        "manifest_file": manifest_path.name,
        "description": description,
        **prov.provenance_block(),
    })

    if entry.exists():
        shutil.rmtree(entry, ignore_errors=True)
    staging.rename(entry)
    return entry


# ═══════════════════════════════════════════════════════════════════════════════
# The entry point
# ═══════════════════════════════════════════════════════════════════════════════


def packed_blocks(loader: Callable, tokenizer, split: str, *, dataset: str,
                  block_size: int, seed: int, eos_between: bool,
                  n_blocks: Optional[int] = None,
                  loader_kwargs: Optional[Dict[str, Any]] = None,
                  cache_root=None) -> Tuple[np.ndarray, List[dict], str, Dict[str, Any]]:
    """Pack via ``loader``, through the cache when ``cache_root`` is given.

    Returns ``(blocks, manifest, manifest_sha256, info)``. ``info`` records ``cache_hit``,
    the ``cache_key`` and the entry path, so a run directory can say which it used —
    a cache whose use is invisible in the artefacts is not auditable.

    ``cache_root=None`` bypasses the cache entirely and simply calls the loader, which is
    what every caller did before this module existed.
    """
    kwargs = dict(loader_kwargs or {})
    info: Dict[str, Any] = {"cache_hit": False, "cache_key": None, "cache_path": None,
                            "cache_root": str(cache_root) if cache_root else None}

    def _pack(cap):
        blocks, manifest = loader(
            tokenizer, split, block_size=block_size, n_blocks=cap, seed=seed,
            eos_between=eos_between, as_array=True, manifest_input_ids=False, **kwargs)
        if len(blocks) == 0:
            raise ValueError(f"{dataset} split {split!r} produced no full blocks")
        return blocks, manifest, prov.sha256_int_rows(blocks)

    if cache_root is None:
        blocks, manifest, digest = _pack(n_blocks)
        return blocks, manifest, digest, info

    key = cache_key(dataset=dataset, split=split, block_size=block_size, seed=seed,
                    eos_between=eos_between, loader_kwargs=kwargs, tokenizer=tokenizer)
    info["cache_key"] = key
    info["cache_path"] = str(_entry_dir(cache_root, key))

    hit = read_entry(cache_root, key, n_blocks)
    if hit is not None:
        blocks, manifest, digest = hit
        info["cache_hit"] = True
        return blocks, manifest, digest, info

    blocks, manifest, digest = _pack(n_blocks)
    Path(cache_root).mkdir(parents=True, exist_ok=True)
    write_entry(cache_root, key, blocks, manifest, digest,
                description={"dataset": dataset, "split": split,
                             "block_size": block_size, "seed": seed,
                             "eos_between": eos_between,
                             # Both recorded: the key subset is what names the entry, and
                             # the full set says how it was actually produced.
                             "loader_kwargs_keyed": key_loader_kwargs(kwargs),
                             "loader_kwargs_used": kwargs,
                             "n_blocks_requested": n_blocks,
                             "tokenizer": tokenizer_identity(tokenizer)})
    return blocks, manifest, digest, info
