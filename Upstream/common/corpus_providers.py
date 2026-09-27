# -*- coding: utf-8 -*-
"""corpus_providers.py — injectable corpora for E6/E7 (WP1, closes G1).

The frozen harnesses bind their corpus to ``sample_benchmark_datasets()``. E6/E7 must run
the *same* intervention battery on other text: TinyStories validation blocks, SST-2
prompts, synthetic controls, and — as a regression bridge — the frozen E1 cross-domain
manifest. A :class:`Corpus` is the injectable unit, and its ``manifest_sha256`` is the
mechanism by which two runs are proven to have seen byte-identical inputs (05 schemas).

Tokenisation matches E1–E5 exactly: ``add_special_tokens=False`` by default and the same
``_normalize_text`` / ``_truncate`` helpers from :mod:`datasets_loader`.

The two E7 parallel providers (``flores_corpus`` / ``xnli_prompt_corpus``) are *projections
of an already-joined* ``ParallelManifest``, which is owned by WP8 (``paired_manifests``).
They are declared here with the spec signatures but raise until WP8 lands, rather than
guess a cross-language join that the XNLI-alignment trap (CLAUDE.md) makes unsafe to
improvise.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:  # package import
    from . import block_corpus_cache as _bcc
    from . import datasets_loader as _dl
except ImportError:  # scripts add common/ directly to sys.path
    import block_corpus_cache as _bcc
    import datasets_loader as _dl

DEFAULT_SEED = _dl.DEFAULT_SEED
DEFAULT_CUT_LENGTH = _dl.DEFAULT_CUT_LENGTH

CORPUS_SCHEMA_VERSION = "corpus_v1"

# TinyStories: block indices [0, SINK_BLOCKS_RESERVED) are reserved for the sink corpus so
# the ppl corpus (which starts after) is guaranteed disjoint from it at the same seed.
SINK_BLOCKS_RESERVED = 300


# ═══════════════════════════════════════════════════════════════════════════════
# Types
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class CorpusItem:
    """One evaluable example. ``item_id`` is the stable join key used everywhere."""

    item_id: str
    text: str
    input_ids: List[int]
    n_tokens: int
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Corpus:
    """A frozen, hashable set of :class:`CorpusItem` plus its provenance."""

    corpus_id: str
    items: Tuple[CorpusItem, ...]
    tokenizer_name: str
    tokenizer_revision: Optional[str]
    add_special_tokens: bool
    cut_length: Optional[int]
    seed: int
    manifest_sha256: str
    provenance: Dict[str, Any]

    def __len__(self) -> int:
        return len(self.items)

    # --- serialisation (JSONL-friendly single file, hash-verified on load) ---

    def save(self, path) -> None:
        obj = {
            "schema": CORPUS_SCHEMA_VERSION,
            "corpus_id": self.corpus_id,
            "tokenizer_name": self.tokenizer_name,
            "tokenizer_revision": self.tokenizer_revision,
            "add_special_tokens": self.add_special_tokens,
            "cut_length": self.cut_length,
            "seed": self.seed,
            "manifest_sha256": self.manifest_sha256,
            "provenance": self.provenance,
            "items": [asdict(item) for item in self.items],
        }
        Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                              encoding="utf-8")

    @staticmethod
    def load(path) -> "Corpus":
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        items = tuple(
            CorpusItem(item_id=row["item_id"], text=row["text"],
                       input_ids=[int(t) for t in row["input_ids"]],
                       n_tokens=int(row["n_tokens"]), meta=dict(row["meta"]))
            for row in obj["items"])
        recomputed = compute_manifest_sha256(items)
        if recomputed != obj["manifest_sha256"]:
            raise ValueError(
                f"Corpus manifest hash mismatch on load ({path}): stored "
                f"{obj['manifest_sha256']} != recomputed {recomputed}. The file is "
                "corrupt or was written by an incompatible schema.")
        return Corpus(
            corpus_id=obj["corpus_id"], items=items,
            tokenizer_name=obj["tokenizer_name"],
            tokenizer_revision=obj.get("tokenizer_revision"),
            add_special_tokens=bool(obj["add_special_tokens"]),
            cut_length=obj["cut_length"], seed=int(obj["seed"]),
            manifest_sha256=obj["manifest_sha256"], provenance=dict(obj["provenance"]))


def compute_manifest_sha256(items: Tuple[CorpusItem, ...]) -> str:
    """SHA-256 over the canonical JSON of ``[(item_id, input_ids, sorted(meta.items()))]``.

    Uses ``sort_keys=True, ensure_ascii=False`` (05 schemas). Tuples and lists serialise
    identically to JSON arrays, so a tuple->list round trip through :meth:`Corpus.load`
    does not change the hash.
    """
    payload = [[item.item_id, [int(t) for t in item.input_ids],
                sorted(item.meta.items())] for item in items]
    blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _tokenizer_identity(tokenizer) -> Tuple[str, Optional[str]]:
    name = getattr(tokenizer, "name_or_path", None) or "unknown"
    revision = None
    init = getattr(tokenizer, "init_kwargs", None)
    if isinstance(init, dict):
        revision = init.get("revision")
    return str(name), revision


def _make_corpus(corpus_id: str, items: List[CorpusItem], tokenizer, *,
                 add_special_tokens: bool, cut_length: Optional[int], seed: int,
                 provenance: Dict[str, Any]) -> Corpus:
    items = tuple(items)
    name, revision = _tokenizer_identity(tokenizer)
    return Corpus(
        corpus_id=corpus_id, items=items, tokenizer_name=name,
        tokenizer_revision=revision, add_special_tokens=add_special_tokens,
        cut_length=cut_length, seed=seed,
        manifest_sha256=compute_manifest_sha256(items),
        provenance={"schema": CORPUS_SCHEMA_VERSION, **provenance})


# ═══════════════════════════════════════════════════════════════════════════════
# Providers
# ═══════════════════════════════════════════════════════════════════════════════


def frozen_e1_corpus(tokenizer, sample_size: int = 100, cut_length: int = 40,
                     seed: int = DEFAULT_SEED) -> Corpus:
    """The frozen E1 cross-domain manifest, wrapped verbatim (regression bridge).

    Calls the frozen :func:`datasets_loader.sample_benchmark_datasets` and preserves its
    row order (sst2, gsm8k, humaneval; shuffled within domain by ``seed``), so the corpus
    manifest reproduces E1's ``sample_manifest.csv`` row-for-row. This is the seam
    ``tests/test_corpus_e1_bridge.py`` asserts against — if it drifts, E6/E7 measure a
    different input distribution than E1–E5.
    """
    _sampled, manifest_rows = _dl.sample_benchmark_datasets(
        tokenizer, sample_size=sample_size, cut_length=cut_length, seed=seed)
    items: List[CorpusItem] = []
    for row in manifest_rows:
        ids = [int(t) for t in tokenizer(row["text"], add_special_tokens=False)["input_ids"]]
        item_id = f"{row['dataset']}:{int(row['source_index'])}"
        items.append(CorpusItem(
            item_id=item_id, text=row["text"], input_ids=ids, n_tokens=len(ids),
            meta={"dataset": row["dataset"], "kind": row["kind"],
                  "split": row["split"], "source_index": int(row["source_index"]),
                  "original_token_length": int(row["original_token_length"]),
                  "token_length": int(row["token_length"])}))
    return _make_corpus(
        f"e1_{sample_size}x{cut_length}", items, tokenizer,
        add_special_tokens=False, cut_length=cut_length, seed=seed,
        provenance={"provider": "frozen_e1_corpus",
                    "datasets": [s["name"] for s in _dl.DATASET_SPECS],
                    "sample_size": sample_size})


def tinystories_corpus(tokenizer, split, n_blocks, *, block_size: int = 128,
                       seed: int = 0, purpose: str = "sink") -> Corpus:
    """Packed TinyStories blocks for E6 sink / perplexity measurement.

    ``purpose="sink"`` takes blocks ``[0, n_blocks)`` and ``purpose="ppl"`` takes blocks
    ``[SINK_BLOCKS_RESERVED, SINK_BLOCKS_RESERVED + n_blocks)`` from the *same* seeded
    block stream, so the two are disjoint by construction (02 §2.3).
    """
    if purpose not in ("sink", "ppl"):
        raise ValueError(f"purpose must be 'sink' or 'ppl', got {purpose!r}")
    if purpose == "sink" and n_blocks > SINK_BLOCKS_RESERVED:
        raise ValueError(
            f"sink corpus requests {n_blocks} blocks but only {SINK_BLOCKS_RESERVED} are "
            "reserved before the ppl range; disjointness would break")
    offset = 0 if purpose == "sink" else SINK_BLOCKS_RESERVED
    needed = offset + n_blocks
    blocks, manifest = _dl.load_tinystories_blocks(
        tokenizer, split, block_size=block_size, n_blocks=needed, seed=seed,
        eos_between=True)
    window = list(zip(blocks, manifest))[offset:offset + n_blocks]

    items: List[CorpusItem] = []
    for local_i, (ids, mrow) in enumerate(window):
        global_index = offset + local_i
        items.append(CorpusItem(
            item_id=f"tinystories:{split}:{seed}:blk{global_index}",
            text=tokenizer.decode(ids), input_ids=[int(t) for t in ids],
            n_tokens=len(ids),
            meta={"purpose": purpose, "block_index": global_index, "split": split,
                  "source_story_indices": mrow["source_story_indices"],
                  "n_eos": int(mrow["n_eos"])}))
    return _make_corpus(
        f"tinystories_{split}_{purpose}_{n_blocks}", items, tokenizer,
        add_special_tokens=False, cut_length=block_size, seed=seed,
        provenance={"provider": "tinystories_corpus", "split": split,
                    "block_size": block_size, "purpose": purpose,
                    "sink_blocks_reserved": SINK_BLOCKS_RESERVED})


def openwebtext_corpus(tokenizer, split, n_blocks, *, block_size: int = 128,
                       seed: int = 0, purpose: str = "sink",
                       train_documents: Optional[int] = None,
                       validation_documents: Optional[int] = None,
                       cache_root=None) -> Corpus:
    """Packed OpenWebText blocks — the E6A-GPT2 arm's in-domain corpus.

    The TinyStories twin of this function, with one extra layer of disjointness. Blocks are
    sliced ``sink`` vs ``ppl`` exactly as :func:`tinystories_corpus` does (02 §2.3), and the
    ``split`` itself is a **disjoint document window** of OpenWebText's single upstream
    train split (`datasets_loader.OPENWEBTEXT_SPLIT_WINDOWS`) — so a validation block can
    contain no training token, by construction rather than by convention.

    The corpus id is ``openwebtext_{split}_{purpose}_{n_blocks}``, i.e.
    ``openwebtext_validation_sink_300`` for the scored corpus, which is what
    ``e6a_gpt2_preregistration.yaml`` names.

    ``cache_root`` (``None`` = pack every time, the original behaviour) routes the pack
    through :mod:`block_corpus_cache`. That matters here more than the block packing does:
    OpenWebText has one upstream split, so reaching the *validation* window means streaming
    past all 400,000 training documents first — a cost every ``evaluate_transformation.py``
    invocation pays afresh. The cache re-verifies the block digest on every hit, and the
    ``Corpus``'s own ``manifest_sha256`` is derived from the items either way, so a cached
    and an uncached corpus are the same corpus or the cache refuses.
    """
    if purpose not in ("sink", "ppl"):
        raise ValueError(f"purpose must be 'sink' or 'ppl', got {purpose!r}")
    if purpose == "sink" and n_blocks > SINK_BLOCKS_RESERVED:
        raise ValueError(
            f"sink corpus requests {n_blocks} blocks but only {SINK_BLOCKS_RESERVED} are "
            "reserved before the ppl range; disjointness would break")
    offset = 0 if purpose == "sink" else SINK_BLOCKS_RESERVED
    needed = offset + n_blocks
    # No `tokenizer_batch_size` here, deliberately. A provider packs at most
    # SINK_BLOCKS_RESERVED + n_blocks (~2,300) blocks, so batching would save seconds —
    # the cost worth removing at this seam is `_load_openwebtext`'s 400,000-document
    # stream, and that is what the cache removes. Forcing it would also require every
    # caller's tokenizer to accept a list, tightening a seam that takes any tokenizer-like
    # object. The trainer, which packs ~3.1M blocks, does batch; the two still share one
    # cache entry because `tokenizer_batch_size` is excluded from the key.
    loader_kwargs = {"train_documents": train_documents,
                     "validation_documents": validation_documents}
    blocks, manifest, _digest, _info = _bcc.packed_blocks(
        _dl.load_openwebtext_blocks, tokenizer, split,
        dataset=_dl.OPENWEBTEXT_HF_PATH, block_size=block_size, seed=seed,
        eos_between=True, n_blocks=needed, loader_kwargs=loader_kwargs,
        cache_root=cache_root)
    window = list(zip(blocks, manifest))[offset:offset + n_blocks]

    items: List[CorpusItem] = []
    for local_i, (ids, mrow) in enumerate(window):
        global_index = offset + local_i
        # Normalised to Python ints BEFORE both uses. A cache hit returns int32 numpy rows
        # where a cold pack returns lists; converting once here means `decode` and
        # `input_ids` see literally the same object either way, so the corpus digest cannot
        # depend on which path produced the blocks.
        ids_list = [int(t) for t in ids]
        items.append(CorpusItem(
            item_id=f"openwebtext:{split}:{seed}:blk{global_index}",
            text=tokenizer.decode(ids_list), input_ids=ids_list,
            n_tokens=len(ids_list),
            meta={"purpose": purpose, "block_index": global_index, "split": split,
                  "source_story_indices": mrow["source_story_indices"],
                  "n_eos": int(mrow["n_eos"])}))
    document_window = _dl.OPENWEBTEXT_SPLIT_WINDOWS.get(split)
    return _make_corpus(
        f"openwebtext_{split}_{purpose}_{n_blocks}", items, tokenizer,
        add_special_tokens=False, cut_length=block_size, seed=seed,
        provenance={"provider": "openwebtext_corpus", "split": split,
                    "block_size": block_size, "purpose": purpose,
                    "sink_blocks_reserved": SINK_BLOCKS_RESERVED,
                    "document_window": list(document_window) if document_window else None,
                    "train_documents": train_documents,
                    "validation_documents": validation_documents})


def sst2_prompt_corpus(tokenizer, split, n=None, *, max_input_tokens: int = 64,
                       seed: int = 0, corrupted_manifest: Optional[Dict[int, int]] = None
                       ) -> Corpus:
    """SST-2 in the E6B prompt format; optional label corruption (train only).

    ``corrupted_manifest`` maps ``source_index -> flipped_label`` and, when present,
    rewrites the appended label token(s) for those items, recording ``corrupted=True`` and
    the ``original_label``. The caller is responsible for only ever applying it to train
    (02 §2.3); this provider does not enforce the split, it records it in provenance.
    """
    rows, _manifest, info = _dl.load_sst2_prompted(
        tokenizer, split, max_input_tokens=max_input_tokens, seed=seed, n=n)
    labels = tuple(info["label_token_ids"].keys())
    label_ids = info["label_token_ids"]
    corrupted_manifest = corrupted_manifest or {}

    items: List[CorpusItem] = []
    for row in rows:
        source_index = int(row["source_index"])
        label_value = int(row["label"])
        label_str = row["label_str"]
        lab_ids = row["label_token_ids"]
        corrupted = source_index in corrupted_manifest
        original_label = label_value
        if corrupted:
            label_value = int(corrupted_manifest[source_index])
            label_str = labels[0] if label_value == 1 else labels[1]
            lab_ids = label_ids[label_str]
        input_ids = list(row["prompt_ids"]) + list(lab_ids)
        label_span = [len(row["prompt_ids"]), len(row["prompt_ids"]) + len(lab_ids)]
        items.append(CorpusItem(
            item_id=f"sst2:{split}:{source_index}",
            text=row["prompt_text"] + label_str,
            input_ids=[int(t) for t in input_ids], n_tokens=len(input_ids),
            meta={"dataset": "sst2", "split": split, "source_index": source_index,
                  "sentence": row["sentence"], "label": label_value,
                  "label_str": label_str, "label_token_ids": [int(t) for t in lab_ids],
                  "label_span": label_span, "corrupted": corrupted,
                  "original_label": original_label}))
    return _make_corpus(
        f"sst2_{split}_prompted", items, tokenizer, add_special_tokens=False,
        cut_length=None, seed=seed,
        provenance={"provider": "sst2_prompt_corpus", "split": split,
                    "template": info["template"], "max_input_tokens": max_input_tokens,
                    "label_token_counts": info["label_token_counts"],
                    "n_corrupted": len(corrupted_manifest)})


def synthetic_corpus(tokenizer, kind, n, seed, *, cut_length: int = DEFAULT_CUT_LENGTH
                     ) -> Corpus:
    """Dataset-free control corpus, wrapping :func:`datasets_loader.build_degenerate_domains`.

    A minimal set of ``n`` deterministic pseudo-natural token sequences of length
    ``cut_length`` is synthesised (no download), then the requested degenerate ``kind`` is
    derived from them by the frozen control builder. This makes the determinism gate
    (``tests/test_corpus_determinism.py``) fully offline.
    """
    import numpy as np

    valid = {"random_uniform", "random_zipf", "shuffled_natural", "repeat_token"}
    if kind not in valid:
        raise ValueError(f"Unknown synthetic kind {kind!r}; choose from {sorted(valid)}")

    eos = tokenizer.eos_token_id
    vocab = np.arange(len(tokenizer), dtype=np.int64)
    if eos is not None:
        vocab = vocab[vocab != int(eos)]
    rng = np.random.default_rng(seed * 100003 + 17)
    natural_records = {"synthetic": [
        {"dataset": "synthetic", "example_id": i,
         "input_ids": [int(t) for t in rng.choice(vocab, size=cut_length, replace=True)]}
        for i in range(n)]}

    built, _manifest = _dl.build_degenerate_domains(
        tokenizer, natural_records, cut_length=cut_length, seed=seed, domains=[kind])
    rows = built[kind]

    items: List[CorpusItem] = []
    for row in rows:
        ids = [int(t) for t in row["input_ids"]]
        meta = {k: v for k, v in row.items()
                if k not in ("input_ids", "text")}
        meta = {k: (int(v) if isinstance(v, (np.integer,)) else v) for k, v in meta.items()}
        items.append(CorpusItem(
            item_id=f"synthetic:{kind}:{int(row['example_id'])}", text=row["text"],
            input_ids=ids, n_tokens=len(ids), meta=meta))
    return _make_corpus(
        f"synthetic_{kind}_{n}", items, tokenizer, add_special_tokens=False,
        cut_length=cut_length, seed=seed,
        provenance={"provider": "synthetic_corpus", "kind": kind, "n": n})


# ═══════════════════════════════════════════════════════════════════════════════
# E7 parallel providers (WP8) — projections of an already-joined ParallelManifest.
#
# Neither provider ever touches a dataset. The cross-language join is owned entirely by
# common/paired_manifests.py; these two only *project* it onto one language, in the exact
# order of `semantic_ids`, so row i is the same sentence in every language. Re-tokenising
# here would be a second source of truth, so the manifest's stored `input_ids` are used
# verbatim and the tokenizer is asserted against the one that built the manifest.
# ═══════════════════════════════════════════════════════════════════════════════


def _manifest_projection(manifest, tokenizer, lang, semantic_ids, *, provider: str,
                         corpus_id: str, meta_fn) -> Corpus:
    """Shared body of the two E7 providers: order-preserving projection onto one language."""
    if lang not in manifest.languages:
        raise ValueError(f"language {lang!r} is not in the manifest ({manifest.languages})")
    ids = list(manifest.semantic_ids) if semantic_ids is None else list(semantic_ids)
    unknown = [sid for sid in ids if sid not in manifest.rows]
    if unknown:
        raise KeyError(f"semantic ids not in manifest: {unknown[:5]}")

    manifest_tokenizer = manifest.provenance.get("tokenizer")
    this_tokenizer, _revision = _tokenizer_identity(tokenizer)
    if manifest_tokenizer not in (None, "unknown") and manifest_tokenizer != this_tokenizer:
        raise ValueError(
            f"manifest was built with tokenizer {manifest_tokenizer!r} but this corpus is "
            f"being projected with {this_tokenizer!r}. The manifest is tokenizer-specific "
            "(04 §1); build a separate manifest rather than joining across them.")

    items: List[CorpusItem] = []
    for sid in ids:
        row = manifest.rows[sid][lang]
        items.append(CorpusItem(
            item_id=f"{sid}:{lang}", text=row["text"],
            input_ids=[int(t) for t in row["input_ids"]],
            n_tokens=int(row["n_tokens"]), meta=meta_fn(sid, row)))

    return _make_corpus(
        corpus_id, items, tokenizer, add_special_tokens=False, cut_length=None,
        seed=manifest.seed,
        provenance={"provider": provider, "language": lang,
                    "manifest_id": manifest.manifest_id,
                    "manifest_sha256": manifest.sha256,
                    "dataset": manifest.dataset, "split": manifest.split,
                    "n_semantic_ids": len(ids)})


def flores_corpus(tokenizer, lang, semantic_ids=None, split: str = "devtest",
                  *, manifest=None) -> Corpus:
    """Projection of a joined FLORES :class:`ParallelManifest` onto one language (E7).

    ``manifest`` is required — a corpus cannot be built from a language alone, because the
    whole point is that row *i* is the same sentence in every language, and only the join
    knows that. ``semantic_ids`` selects and orders the rows; ``None`` uses the manifest's
    own order.
    """
    if manifest is None:
        raise ValueError(
            "flores_corpus projects an already-joined ParallelManifest; pass "
            "manifest=build_flores_manifest(...) (common/paired_manifests.py, WP8).")
    if manifest.dataset != "flores":
        raise ValueError(f"expected a FLORES manifest, got {manifest.dataset!r}")

    def _meta(sid, row):
        return {"semantic_id": sid, "language": lang, "split": manifest.split,
                "source_index": int(row["source_index"]),
                "first_token_id": int(row["first_token_id"]),
                "first_token_frequency": int(row["first_token_frequency"]),
                "punctuation_at_position_0": bool(row["punctuation_at_position_0"]),
                "script": row["script"], "language_family": row["language_family"],
                "token_length": int(row["n_tokens"])}

    ids = list(manifest.semantic_ids) if semantic_ids is None else list(semantic_ids)
    return _manifest_projection(
        manifest, tokenizer, lang, semantic_ids, provider="flores_corpus",
        corpus_id=f"flores_{manifest.split}_{lang}_{len(ids)}", meta_fn=_meta)


def xnli_prompt_corpus(tokenizer, lang, semantic_ids=None, split: str = "test",
                       *, manifest=None) -> Corpus:
    """Projection of a joined XNLI :class:`ParallelManifest` onto one language (E7).

    The join is on an explicit id, never a row index — see
    ``datasets_loader.load_xnli_aligned`` and ``paired_manifests`` §5.2 (CLAUDE.md trap 2).
    ``meta`` carries premise, hypothesis, gold label and the candidate token ids needed for
    length-normalised label scoring.
    """
    if manifest is None:
        raise ValueError(
            "xnli_prompt_corpus projects an already-joined ParallelManifest; pass "
            "manifest=build_xnli_manifest(...) (common/paired_manifests.py, WP8). A "
            "positional join would silently destroy every cross-language claim.")
    if manifest.dataset != "xnli":
        raise ValueError(f"expected an XNLI manifest, got {manifest.dataset!r}")

    def _meta(sid, row):
        return {"semantic_id": sid, "language": lang, "split": manifest.split,
                "premise": row["premise"], "hypothesis": row["hypothesis"],
                "gold_label": int(row["gold_label"]),
                "gold_label_name": row["gold_label_name"],
                "candidates": list(row["candidates"]),
                "label_candidate_ids": [list(map(int, c))
                                        for c in row["candidate_token_ids"]],
                "translated_instruction": bool(row["translated_instruction"]),
                "first_token_id": int(row["first_token_id"]),
                "first_token_frequency": int(row["first_token_frequency"]),
                "punctuation_at_position_0": bool(row["punctuation_at_position_0"]),
                "script": row["script"], "language_family": row["language_family"],
                "source_index": int(row["source_index"]),
                "token_length": int(row["n_tokens"])}

    ids = list(manifest.semantic_ids) if semantic_ids is None else list(semantic_ids)
    return _manifest_projection(
        manifest, tokenizer, lang, semantic_ids, provider="xnli_prompt_corpus",
        corpus_id=f"xnli_{manifest.split}_{lang}_{len(ids)}", meta_fn=_meta)
