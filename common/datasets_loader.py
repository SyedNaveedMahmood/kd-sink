# -*- coding: utf-8 -*-
"""datasets_loader.py — Dataset loading and sampling for intervention_analysis.py.

Three standard benchmarks are used:
  - SST-2       Natural language sentences (Stanford Sentiment Treebank).
  - GSM8K       Grade-school math word problems.
  - HumanEval   Python programming prompts.

Examples with fewer than `cut_length` tokens are discarded.
All kept examples are truncated to exactly `cut_length` tokens.
`sample_size` examples are then drawn with a fixed random seed.
"""

import hashlib   # WP8: content-hash join key for the structurally aligned XNLI config
import random
import numpy as np
from datasets import load_dataset

# ─── Dataset registry ────────────────────────────────────────────────────────

DATASET_SPECS = [
    {
        "name": "sst2",
        "hf_path": "stanfordnlp/sst2",
        "config": None,
        "split": "train",
        "text_field": "sentence",
        "kind": "natural_language",
    },
    {
        "name": "gsm8k",
        "hf_path": "openai/gsm8k",
        "config": "main",
        "split": "train",
        "text_field": "question",
        "kind": "math",
    },
    {
        "name": "humaneval",
        "hf_path": "openai/openai_humaneval",
        "config": None,
        "split": "test",
        "text_field": "prompt",
        "kind": "code",
    },
]

DEFAULT_SAMPLE_SIZE = 100
DEFAULT_CUT_LENGTH  = 40
DEFAULT_SEED        = 42

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _load_one_dataset(spec):
    kwargs = {}
    if spec.get("config"):
        kwargs["name"] = spec["config"]
    return load_dataset(spec["hf_path"], **kwargs, split=spec["split"])


def _normalize_text(text):
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def _truncate(tokenizer, text, cut_length):
    """Return *text* truncated to exactly *cut_length* GPT-2 tokens."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    return tokenizer.decode(ids[:cut_length])


# ─── Public API ──────────────────────────────────────────────────────────────

def verify_datasets(tokenizer,
                    sample_size=DEFAULT_SAMPLE_SIZE,
                    cut_length=DEFAULT_CUT_LENGTH):
    """Check that every dataset has at least `sample_size` examples with
    >= `cut_length` tokens (i.e. that can be truncated to the target length).

    Prints a summary table and raises ``ValueError`` for any shortfall.
    """
    print("Verifying datasets...")
    print(f"  Cut length: {cut_length} tokens (examples shorter than this are discarded)")
    print(f"  Required per dataset: {sample_size}\n")

    ok = True
    for spec in DATASET_SPECS:
        print(f"  Loading {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)
        count = 0
        for ex in ds:
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            tlen = len(tokenizer(text, add_special_tokens=False)["input_ids"])
            if tlen >= cut_length:
                count += 1

        status = "OK" if count >= sample_size else "FAIL"
        print(f"    [{status}] {count} usable examples  (need {sample_size})")

        if count < sample_size:
            ok = False

    if not ok:
        raise ValueError(
            "One or more datasets do not have enough usable examples. "
            "Reduce --sample-size or --cut-length."
        )
    print("\nAll datasets verified OK.\n")


def sample_benchmark_datasets(tokenizer,
                               sample_size=DEFAULT_SAMPLE_SIZE,
                               cut_length=DEFAULT_CUT_LENGTH,
                               seed=DEFAULT_SEED):
    """Load, filter, truncate, and sample from all three benchmark datasets.

    Only examples with at least `cut_length` GPT-2 tokens are kept.
    All selected examples are then decoded back to text after truncation to
    exactly `cut_length` tokens, so every example in the returned corpus has
    the same sequence length.

    Parameters
    ----------
    tokenizer   : GPT2Tokenizer
    sample_size : int  — examples to draw from each dataset.
    cut_length  : int  — token count threshold and truncation target.
    seed        : int  — random seed for reproducibility.

    Returns
    -------
    sampled       : dict[str, list[str]]
        dataset_name → list of truncated text strings (each == cut_length tokens).
    manifest_rows : list[dict]
        One row per sampled example (for audit / reproducibility CSV).
    """
    rng = random.Random(seed)
    sampled = {}
    manifest_rows = []

    for spec in DATASET_SPECS:
        print(f"Sampling {spec['name']} ({spec['hf_path']}) ...")
        ds = _load_one_dataset(spec)

        candidates = []
        for idx, ex in enumerate(ds):
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if len(ids) >= cut_length:
                truncated = tokenizer.decode(ids[:cut_length])
                candidates.append({
                    "dataset":               spec["name"],
                    "kind":                  spec["kind"],
                    "hf_path":               spec["hf_path"],
                    "split":                 spec["split"],
                    "source_index":          idx,
                    "text":                  truncated,
                    "original_token_length": len(ids),
                    "token_length":          cut_length,
                })

        if len(candidates) < sample_size:
            raise ValueError(
                f"Dataset '{spec['name']}' only has {len(candidates)} usable examples "
                f"(need {sample_size}). Run verify_datasets() first."
            )

        rng.shuffle(candidates)
        chosen = candidates[:sample_size]

        print(f"  Sampled {len(chosen)} examples  "
              f"(all truncated to {cut_length} tokens; "
              f"mean original length={np.mean([r['original_token_length'] for r in chosen]):.1f})")

        sampled[spec["name"]] = [r["text"] for r in chosen]
        manifest_rows.extend(chosen)

    return sampled, manifest_rows


def sample_long_benchmark_datasets(tokenizer, sample_size=DEFAULT_SAMPLE_SIZE,
                                   cut_length=1024, seed=DEFAULT_SEED,
                                   domains=None):
    """Build exact-length token sequences by deterministic within-domain concatenation.

    Unlike :func:`sample_benchmark_datasets`, this function returns token IDs so a
    decode/re-tokenize round trip cannot alter length.  Source examples are shuffled
    independently per domain, tokenized without special tokens, and concatenated only
    with examples from that domain.  Each returned record includes its source indices
    for a reproducible manifest.
    """
    if sample_size <= 0 or cut_length < 2:
        raise ValueError("sample_size must be positive and cut_length must be at least 2")
    wanted = set(domains or [s["name"] for s in DATASET_SPECS])
    unknown = wanted.difference(s["name"] for s in DATASET_SPECS)
    if unknown:
        raise ValueError(f"Unknown benchmark domain(s): {sorted(unknown)}")
    sampled, manifest = {}, []
    for domain_index, spec in enumerate(DATASET_SPECS):
        if spec["name"] not in wanted:
            continue
        ds = _load_one_dataset(spec)
        pieces = []
        for idx, ex in enumerate(ds):
            text = _normalize_text(ex.get(spec["text_field"], ""))
            if not text:
                continue
            ids = tokenizer(text, add_special_tokens=False)["input_ids"]
            if ids:
                pieces.append((idx, text, list(map(int, ids))))
        if not pieces:
            raise ValueError(f"Dataset '{spec['name']}' contains no tokenizable examples")
        # Use a cut-length-independent anchor order so example i has the same
        # natural 40-token prefix whether this function is asked for 40 or 1024
        # tokens.  This makes separately scheduled E5 modes pairable.
        anchors = [piece for piece in pieces if len(piece[2]) >= min(40, cut_length)]
        anchor_rng = random.Random(seed * 1009 + domain_index * 9176)
        anchor_rng.shuffle(anchors)
        if len(anchors) < sample_size:
            raise ValueError(
                f"Dataset '{spec['name']}' has only {len(anchors)} natural anchor examples "
                f"with at least {min(40, cut_length)} tokens (need {sample_size})"
            )
        records = []
        for example_id in range(sample_size):
            anchor = anchors[example_id]
            ids = list(anchor[2])
            source_indices, source_text = [anchor[0]], [anchor[1]]
            extension = list(pieces)
            extension_rng = random.Random(
                seed * 1000003 + domain_index * 9176 + example_id * 7919
            )
            extension_rng.shuffle(extension)
            cursor = 0
            while len(ids) < cut_length:
                if cursor and cursor % len(extension) == 0:
                    extension_rng.shuffle(extension)
                idx, text, part = extension[cursor % len(extension)]
                cursor += 1
                ids.extend(part)
                source_indices.append(idx)
                source_text.append(text)
            ids = ids[:cut_length]
            if len(ids) != cut_length:
                raise AssertionError("long-context sampler produced an inexact sequence")
            record = {
                "dataset": spec["name"], "example_id": example_id,
                "input_ids": ids, "text": tokenizer.decode(ids),
                "source_indices": source_indices,
                "source_component_count": len(source_indices),
            }
            records.append(record)
            manifest.append({
                "dataset": spec["name"], "kind": spec["kind"],
                "hf_path": spec["hf_path"], "split": spec["split"],
                "example_id": example_id, "source_indices": ";".join(map(str, source_indices)),
                "source_component_count": len(source_indices), "token_length": cut_length,
                "text": record["text"], "sampling": "within_domain_concatenation",
            })
        sampled[spec["name"]] = records
    if not sampled:
        raise ValueError("No benchmark domains were selected")
    return sampled, manifest


def build_degenerate_domains(tokenizer, natural_records, cut_length=DEFAULT_CUT_LENGTH,
                             seed=DEFAULT_SEED, domains=None):
    """Create exact-length dataset-free controls from token-ID natural examples.

    ``natural_records`` maps domains to records containing ``input_ids``.  Generated
    tokens exclude ``eos_token_id`` and use local NumPy generators only.
    """
    requested = domains or ["random_uniform", "random_zipf", "shuffled_natural", "repeat_token"]
    valid = {"random_uniform", "random_zipf", "shuffled_natural", "repeat_token"}
    unknown = set(requested).difference(valid)
    if unknown:
        raise ValueError(f"Unknown synthetic domain(s): {sorted(unknown)}")
    base = [r for records in natural_records.values() for r in records]
    if not base:
        raise ValueError("Cannot build degenerate domains from an empty natural sample")
    eos = tokenizer.eos_token_id
    vocab = np.arange(len(tokenizer), dtype=np.int64)
    if eos is not None:
        vocab = vocab[vocab != int(eos)]
    natural_tokens = np.asarray([
        t for r in base for t in r["input_ids"][:cut_length] if eos is None or t != eos
    ], dtype=np.int64)
    if natural_tokens.size == 0:
        raise ValueError("Natural sample has no allowed tokens")
    values, counts = np.unique(natural_tokens, return_counts=True)
    probs = counts.astype(np.float64) / counts.sum()
    result, manifest = {}, []
    for domain_no, name in enumerate(requested):
        rng = np.random.default_rng(seed * 10007 + domain_no * 7919)
        rows = []
        for i, source in enumerate(base):
            src = np.asarray(source["input_ids"][:cut_length], dtype=np.int64)
            if len(src) != cut_length:
                raise ValueError("Natural control inputs must already have exact cut_length")
            if name == "random_uniform":
                ids = rng.choice(vocab, size=cut_length, replace=True)
                meta = {"distribution": "uniform_allowed_vocabulary"}
            elif name == "random_zipf":
                ids = rng.choice(values, size=cut_length, replace=True, p=probs)
                meta = {"distribution": "empirical_natural_unigram"}
            elif name == "shuffled_natural":
                ids = src[rng.permutation(cut_length)]
                meta = {"distribution": "within_example_permutation"}
            else:
                allowed = src[src != eos] if eos is not None else src
                if len(allowed) == 0:
                    allowed = natural_tokens
                token = int(allowed[(seed + i) % len(allowed)])
                ids = np.full(cut_length, token, dtype=np.int64)
                meta = {"distribution": "repeat", "repeat_token_id": token}
            ids_list = list(map(int, ids))
            row = {"dataset": name, "example_id": i, "input_ids": ids_list,
                   "text": tokenizer.decode(ids_list), "source_domain": source["dataset"],
                   "source_example_id": source["example_id"], **meta}
            rows.append(row)
            manifest.append({k: v for k, v in row.items() if k != "input_ids"})
        result[name] = rows
    return result, manifest


def load_optional_flores(tokenizer, sample_size=DEFAULT_SAMPLE_SIZE,
                         cut_length=DEFAULT_CUT_LENGTH, seed=DEFAULT_SEED):
    """Load Bangla and Chinese FLORES-200 records, raising a readable RuntimeError.

    E5 catches this exception and records a skip reason, making multilingual data
    genuinely optional without concealing configuration or network failures.
    """
    configs = {"flores_bengali": "ben_Beng", "flores_chinese": "zho_Hans"}
    out, manifest = {}, []
    try:
        for domain_no, (name, lang) in enumerate(configs.items()):
            ds = load_dataset("facebook/flores", lang, split="devtest")
            candidates = []
            for idx, ex in enumerate(ds):
                text = _normalize_text(ex.get("sentence", ""))
                ids = tokenizer(text, add_special_tokens=False)["input_ids"] if text else []
                if len(ids) >= cut_length:
                    candidates.append((idx, ids[:cut_length]))
            rng = random.Random(seed * 1013 + domain_no)
            rng.shuffle(candidates)
            if len(candidates) < sample_size:
                raise ValueError(f"{lang} has {len(candidates)} usable examples; need {sample_size}")
            records = []
            for example_id, (source_index, ids) in enumerate(candidates[:sample_size]):
                ids = list(map(int, ids))
                record = {"dataset": name, "example_id": example_id, "input_ids": ids,
                          "text": tokenizer.decode(ids), "language_config": lang,
                          "source_index": source_index}
                records.append(record)
                manifest.append({**record, "input_ids": " ".join(map(str, ids)),
                                 "token_length": len(ids)})
            out[name] = records
    except Exception as exc:
        raise RuntimeError(f"Optional FLORES-200 loading failed: {type(exc).__name__}: {exc}") from exc
    return out, manifest


# ═══════════════════════════════════════════════════════════════════════════════
# Additive E6/E7 corpus loaders (01 §6)
#
# New functions only; nothing above is modified. They reuse DEFAULT_SEED and the same
# _normalize_text / add_special_tokens=False discipline as the E1–E5 samplers so E6/E7
# tokenisation is identical to the frozen path. load_flores_parallel / load_xnli_aligned
# are the E7 parallel-join loaders added by WP8; they return *raw rows keyed by a join
# key*, and the alignment contract itself lives in common/paired_manifests.py.
# ═══════════════════════════════════════════════════════════════════════════════

TINYSTORIES_HF_PATH = "roneneldan/TinyStories"
SST2_LABELS_DEFAULT = (" positive", " negative")   # index 0 -> label==1 (positive)


def _load_tinystories(split):
    """Load a TinyStories split (``text`` field). Split isolation is the caller's job."""
    return load_dataset(TINYSTORIES_HF_PATH, split=split)


def load_tinystories_blocks(tokenizer, split, block_size=128, n_blocks=None,
                            seed=0, eos_between=True, *,
                            as_array=False, manifest_input_ids=True):
    """Pack a TinyStories split into exact ``block_size``-token blocks.

    Stories are tokenised without special tokens, optionally joined with a single EOS
    between consecutive stories, and greedily packed; only *full* blocks are emitted so
    every returned block has exactly ``block_size`` tokens. Story order is a deterministic
    ``random.Random(seed)`` shuffle, so ``(split, seed)`` fixes the block sequence and a
    downstream provider can take disjoint contiguous slices (sink vs ppl) from one call.

    Returns ``(blocks, block_manifest)`` where each manifest row records the block's
    exact ``input_ids``, the sorted set of contributing source story indices, and the EOS
    count inside the block (the story-boundary count). ``n_blocks`` caps the output.

    **Two additive memory options, both defaulting to the original behaviour.** The full
    TinyStories train split packs to ~3.57M blocks, and this function used to hold each one
    *twice* — once in ``blocks`` and once again as ``manifest[i]["input_ids"]`` — as Python
    lists of Python ints. Measured at 5,630 bytes per block, i.e. ~20 GB for the train
    split, which is why E6A training could not start on a 16 GB host.

    * ``as_array=True`` returns ``blocks`` as a ``numpy`` ``int32`` array of shape
      ``(n_blocks, block_size)``: ~1.8 GB instead of ~12.8 GB, with identical values in
      identical order. ``prov.sha256_int_rows`` hashes it to the same digest as the list
      form, so ``manifest_sha256`` does not move.
    * ``manifest_input_ids=False`` drops the duplicated ``input_ids`` field from the
      manifest rows. The caller still has every token in ``blocks``, so nothing is lost —
      ``write_block_manifest`` reads the ids from there and the ``05`` §4 parquet keeps its
      column.

    Neither flag changes which tokens are packed, their order, or the block boundaries.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    eos = tokenizer.eos_token_id
    if eos_between and eos is None:
        raise ValueError("eos_between=True requires the tokenizer to define eos_token_id")

    ds = _load_tinystories(split)
    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    stream: list = []      # pending token ids not yet emitted in a block
    owners: list = []      # source story index per pending token (parallel to `stream`)
    blocks: list = []
    manifest: list = []

    # Under ``as_array`` the packed blocks are folded into int32 arrays as they are
    # produced, rather than accumulated as Python lists and converted at the end.
    # Converting at the end would be pointless: the peak is the list-of-lists, which for
    # the TinyStories train split is ~12.8 GB and is exactly what has to be avoided.
    # ``_ARRAY_FOLD_ROWS`` bounds the pending list; the finished arrays are concatenated
    # once at the end (a transient double of ~1.8 GB, not of ~12.8 GB).
    folded: list = []      # completed int32 arrays, only used when as_array

    def _fold_pending(force=False):
        if not as_array or (not force and len(blocks) < _ARRAY_FOLD_ROWS):
            return
        if blocks:
            import numpy as np

            folded.append(np.asarray(blocks, dtype=np.int32))
            del blocks[:]

    def _emit_full_blocks():
        while len(stream) >= block_size:
            chunk = stream[:block_size]
            chunk_owner = owners[:block_size]
            del stream[:block_size]
            del owners[:block_size]
            ids = [int(t) for t in chunk]
            index = _folded_count() + len(blocks)
            blocks.append(ids)
            row = {
                "block_index": index,
                "source_story_indices": sorted(set(int(o) for o in chunk_owner)),
                "n_tokens": block_size,
                "n_eos": int(sum(1 for t in chunk if eos is not None and t == eos)),
            }
            if manifest_input_ids:
                # A distinct copy, as before: a caller that mutates `blocks` must not
                # silently rewrite the manifest that is supposed to prove what was packed.
                row["input_ids"] = list(ids)
            manifest.append(row)
            _fold_pending()

    def _folded_count():
        return sum(len(part) for part in folded)

    def _total_blocks():
        return _folded_count() + len(blocks)

    for story_idx in order:
        text = _normalize_text(ds[story_idx].get("text", ""))
        if not text:
            continue
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if not ids:
            continue
        stream.extend(int(t) for t in ids)
        owners.extend(story_idx for _ in ids)
        if eos_between:
            stream.append(int(eos))
            owners.append(story_idx)   # the boundary EOS is attributed to the story it closes
        _emit_full_blocks()
        if n_blocks is not None and _total_blocks() >= n_blocks:
            break

    if as_array:
        import numpy as np

        _fold_pending(force=True)
        # int32 is exact for every vocabulary in this project (the largest is Qwen's
        # ~152k) and halves what int64 would cost. Shape is (n_blocks, block_size);
        # every block is full by construction, so the array is never ragged.
        blocks = (np.concatenate(folded, axis=0) if folded
                  else np.zeros((0, block_size), dtype=np.int32))
        if n_blocks is not None:
            blocks = blocks[:n_blocks]
    elif n_blocks is not None:
        blocks = blocks[:n_blocks]
    if n_blocks is not None:
        manifest = manifest[:n_blocks]
    return blocks, manifest


# ═══════════════════════════════════════════════════════════════════════════════
# OpenWebText blocks — the E6A-GPT2 arm's training and sink corpora
# ═══════════════════════════════════════════════════════════════════════════════
#
# The TinyStories arm's teacher has no sink on the corpus its criteria are scored on
# (baseline_sink 0.009309, i.e. 0.86x the uniform-attention floor at 128 tokens), so a
# second arm distils `gpt2` instead. GPT-2's own training corpus, WebText, was never
# released; `Skylion007/openwebtext` is its open reproduction and is therefore the closest
# available in-distribution corpus for that teacher. WikiText was rejected: the GPT-2 paper
# states WebText deliberately *excluded* Wikipedia, so scoring the teacher there would
# repeat the out-of-distribution confound this arm exists to avoid.

OPENWEBTEXT_HF_PATH = "Skylion007/openwebtext"

#: OpenWebText ships a single ``train`` split of ~8M documents and ~38 GB, so the loader
#: takes a deterministic *document prefix* rather than the whole set and carves train /
#: validation out of it as disjoint windows. These are the defaults; a config may override.
#:
#: The cap is sized for the **registered 10,000-step horizon**, not for the 2,000-step
#: pilot, and that is not conservatism — it is required for the two to be the same
#: experiment. ``_pack_documents`` shuffles ``range(len(documents))``, so the cap is an
#: input to the shuffle: a pilot packed from 60k documents and a Phase 2 run packed from
#: 400k would see *different block sequences*, and `--max-steps 2000` would no longer be a
#: truncation of the full run. One cap, both horizons.
#:
#: 10,000 steps x 16 x 4 x 128 = 81.9M tokens. At a deliberately pessimistic 250 tokens per
#: document that is 327,680 documents, so 400,000 leaves ~22% headroom on the pessimistic
#: figure and several-fold on the realistic one. Cost, measured on nothing yet and stated
#: as an estimate: OpenWebText averages ~4.75 KB/document, so the prefix is ~1.9 GB of text
#: streamed per run (streaming does not populate the HF cache, so it is re-fetched each
#: time) and packs in well under the TinyStories split's measured 17-30 minutes.
OPENWEBTEXT_TRAIN_DOCUMENTS = 400_000
#: 300 sink blocks + 2,000 ΔCE blocks = 2,300 x 128 = 294k tokens; ~1,180 documents at the
#: same pessimistic rate. 8,000 is ~6.8x that.
OPENWEBTEXT_VALIDATION_DOCUMENTS = 8_000

#: The document windows, keyed by the split name callers use. Disjoint by construction —
#: the same guarantee `corpus_providers.SINK_BLOCKS_RESERVED` gives for sink vs ppl blocks,
#: one level up. A validation block can therefore never contain a training token.
OPENWEBTEXT_SPLIT_WINDOWS = {
    "train": (0, OPENWEBTEXT_TRAIN_DOCUMENTS),
    "validation": (OPENWEBTEXT_TRAIN_DOCUMENTS,
                   OPENWEBTEXT_TRAIN_DOCUMENTS + OPENWEBTEXT_VALIDATION_DOCUMENTS),
}


#: Documents handed to the tokenizer in one call when ``tokenizer_batch_size`` is left to
#: the loader. Large enough that the Rust tokenizer's own thread pool has work to spread
#: across cores, small enough that one batch's decoded ids stay a bounded fraction of the
#: packing loop's footprint (~1,000 OpenWebText documents is ~1M token ids, tens of MB).
DEFAULT_TOKENIZER_BATCH = 1_000


def _iter_document_ids(documents, order, tokenizer, tokenizer_batch_size):
    """Yield ``(doc_idx, normalised_text, input_ids)`` in ``order``.

    ``tokenizer_batch_size=None`` calls the tokenizer once per document — the original
    behaviour, preserved exactly. Any positive value groups documents and makes **one**
    tokenizer call per group, which is what lets a HuggingFace *fast* tokenizer use its Rust
    thread pool instead of running one document at a time on one core. Packing the E6A-GPT2
    training window (400,000 OpenWebText documents) is otherwise ~17-30 minutes of
    single-threaded work before the first optimiser step, repeated on every run and every
    resume.

    The grouping changes **nothing about what is packed**: the same documents are tokenised,
    in the same order, and a fast tokenizer's per-sequence output does not depend on what
    else is in its batch (no padding or truncation is requested here).
    ``tests/test_batched_tokenisation_equivalence.py`` asserts that on both a stub tokenizer
    and the real gpt2 one, down to the ``manifest_sha256`` — the digest is the join key for
    every cached artefact in the project (``05`` §7.1) and must not move (CLAUDE.md trap 19).

    A batch may tokenise up to ``tokenizer_batch_size - 1`` documents past the point where
    the caller's ``n_blocks`` cap is reached. That costs a little work and changes no output:
    the caller still stops at exactly the same document.
    """
    if tokenizer_batch_size is None:
        for doc_idx in order:
            text = _normalize_text(documents[doc_idx].get("text", ""))
            if not text:
                continue
            yield doc_idx, text, tokenizer(text, add_special_tokens=False)["input_ids"]
        return

    size = int(tokenizer_batch_size)
    if size < 1:
        raise ValueError(f"tokenizer_batch_size must be >= 1, got {tokenizer_batch_size!r}")
    for start in range(0, len(order), size):
        group = []
        for doc_idx in order[start:start + size]:
            text = _normalize_text(documents[doc_idx].get("text", ""))
            if not text:
                continue
            group.append((doc_idx, text))
        if not group:
            continue
        encoded = tokenizer([text for _, text in group], add_special_tokens=False)["input_ids"]
        if len(encoded) != len(group):
            raise ValueError(
                f"batched tokenisation returned {len(encoded)} sequences for {len(group)} "
                "documents; refusing to pack a misaligned batch")
        for (doc_idx, text), ids in zip(group, encoded):
            yield doc_idx, text, ids


def _pack_documents(documents, tokenizer, block_size, n_blocks, seed, eos_between,
                    as_array, manifest_input_ids, tokenizer_batch_size=None):
    """Greedy fixed-size packing of ``documents`` — the shared block-packing loop.

    ``documents`` is any sequence supporting ``len()`` and integer indexing whose items
    expose a ``text`` field (an HF ``Dataset`` or a list of dicts). Order is a deterministic
    ``random.Random(seed)`` shuffle of the document indices, taken **before** packing, so
    ``(documents, seed)`` fixes the block sequence and a downstream provider can slice
    disjoint contiguous windows out of one call.

    This is a faithful copy of :func:`load_tinystories_blocks`'s loop rather than a
    refactor of it. `manifest_sha256` is the join key for every cached artefact in the
    project (`05` §7.1), and the TinyStories arm's records are already on disk — a body
    edit there that moved the digest by one byte would silently partition them into two
    incomparable sets (CLAUDE.md trap 19). `tests/test_block_packing_equivalence.py` drives
    both functions over the same synthetic documents and asserts byte-identical blocks,
    manifests and digests, so the copy is proved faithful rather than assumed to be.
    """
    if block_size < 2:
        raise ValueError("block_size must be at least 2")
    eos = tokenizer.eos_token_id
    if eos_between and eos is None:
        raise ValueError("eos_between=True requires the tokenizer to define eos_token_id")

    order = list(range(len(documents)))
    random.Random(seed).shuffle(order)

    stream: list = []      # pending token ids not yet emitted in a block
    owners: list = []      # source document index per pending token (parallel to `stream`)
    blocks: list = []
    manifest: list = []
    folded: list = []      # completed int32 arrays, only used when as_array

    def _fold_pending(force=False):
        if not as_array or (not force and len(blocks) < _ARRAY_FOLD_ROWS):
            return
        if blocks:
            import numpy as np

            folded.append(np.asarray(blocks, dtype=np.int32))
            del blocks[:]

    def _folded_count():
        return sum(len(part) for part in folded)

    def _total_blocks():
        return _folded_count() + len(blocks)

    def _emit_full_blocks():
        while len(stream) >= block_size:
            chunk = stream[:block_size]
            chunk_owner = owners[:block_size]
            del stream[:block_size]
            del owners[:block_size]
            ids = [int(t) for t in chunk]
            index = _folded_count() + len(blocks)
            blocks.append(ids)
            row = {
                "block_index": index,
                "source_story_indices": sorted(set(int(o) for o in chunk_owner)),
                "n_tokens": block_size,
                "n_eos": int(sum(1 for t in chunk if eos is not None and t == eos)),
            }
            if manifest_input_ids:
                row["input_ids"] = list(ids)
            manifest.append(row)
            _fold_pending()

    for doc_idx, _text, ids in _iter_document_ids(documents, order, tokenizer,
                                                  tokenizer_batch_size):
        if not ids:
            continue
        stream.extend(int(t) for t in ids)
        owners.extend(doc_idx for _ in ids)
        if eos_between:
            stream.append(int(eos))
            owners.append(doc_idx)     # the boundary EOS belongs to the document it closes
        _emit_full_blocks()
        if n_blocks is not None and _total_blocks() >= n_blocks:
            break

    if as_array:
        import numpy as np

        _fold_pending(force=True)
        blocks = (np.concatenate(folded, axis=0) if folded
                  else np.zeros((0, block_size), dtype=np.int32))
        if n_blocks is not None:
            blocks = blocks[:n_blocks]
    elif n_blocks is not None:
        blocks = blocks[:n_blocks]
    if n_blocks is not None:
        manifest = manifest[:n_blocks]
    return blocks, manifest


def _load_openwebtext(split, *, train_documents=None, validation_documents=None):
    """A deterministic document window of OpenWebText as a list of ``{"text": ...}`` dicts.

    OpenWebText has one ``train`` split, so ``"train"`` and ``"validation"`` here name
    **disjoint document windows** of it (:data:`OPENWEBTEXT_SPLIT_WINDOWS`), not upstream
    splits. The window is a *prefix*, taken with ``streaming=True`` so only the shards it
    touches are fetched — the full set is ~38 GB and a 2,000-step pilot needs 16.4M tokens.

    The prefix is deterministic; the seeded shuffle happens later, in :func:`_pack_documents`,
    which is where design §1.4 puts it. Taking a shuffled *sample* of the whole corpus
    instead would require reading all 38 GB to be reproducible.
    """
    windows = dict(OPENWEBTEXT_SPLIT_WINDOWS)
    if train_documents is not None or validation_documents is not None:
        n_train = int(train_documents if train_documents is not None
                      else OPENWEBTEXT_TRAIN_DOCUMENTS)
        n_val = int(validation_documents if validation_documents is not None
                    else OPENWEBTEXT_VALIDATION_DOCUMENTS)
        windows = {"train": (0, n_train), "validation": (n_train, n_train + n_val)}
    if split not in windows:
        raise ValueError(
            f"OpenWebText window {split!r} is not defined; this loader carves "
            f"{sorted(windows)} out of the single upstream train split")
    start, end = windows[split]

    stream = load_dataset(OPENWEBTEXT_HF_PATH, split="train", streaming=True)
    rows = []
    for index, row in enumerate(stream):
        if index >= end:
            break
        if index >= start:
            rows.append({"text": row.get("text", "")})
    if not rows:
        raise ValueError(
            f"OpenWebText window {split!r} = documents [{start}, {end}) yielded nothing; "
            "the dataset returned fewer documents than the window requires")
    return rows


def load_openwebtext_blocks(tokenizer, split, block_size=128, n_blocks=None,
                            seed=0, eos_between=True, *,
                            as_array=False, manifest_input_ids=True,
                            train_documents=None, validation_documents=None,
                            tokenizer_batch_size=None):
    """Pack an OpenWebText document window into exact ``block_size``-token blocks.

    Signature, return shape and flags mirror :func:`load_tinystories_blocks` exactly, so
    every caller (the trainer's block dataset, the corpus providers, the ΔCE corpora) is
    interchangeable between the two arms. ``train_documents`` / ``validation_documents``
    resize the disjoint document windows; the defaults are
    :data:`OPENWEBTEXT_TRAIN_DOCUMENTS` and :data:`OPENWEBTEXT_VALIDATION_DOCUMENTS`.

    ``tokenizer_batch_size`` (additive; ``None`` = one call per document, the original
    behaviour) groups documents into a single tokenizer call so a fast tokenizer can use
    more than one core — see :func:`_iter_document_ids`. It is offered here and **not** on
    :func:`load_tinystories_blocks`: that arm's packed records are already on disk, and the
    only safe number of ways to produce a `manifest_sha256` is one (CLAUDE.md trap 19).
    """
    documents = _load_openwebtext(split, train_documents=train_documents,
                                  validation_documents=validation_documents)
    return _pack_documents(documents, tokenizer, block_size, n_blocks, seed, eos_between,
                           as_array, manifest_input_ids,
                           tokenizer_batch_size=tokenizer_batch_size)


def load_sst2_prompted(tokenizer, split, max_input_tokens=64,
                       template="{sentence}\nSentiment:",
                       labels=SST2_LABELS_DEFAULT, seed=0, n=None):
    """Load SST-2 in the E6B prompt format.

    Each row is ``template.format(sentence=...)`` followed by the class label string, so
    the sentence begins at position 0 with no special-token prefix (``add_special_tokens=
    False``). The sentence is truncated to ``max_input_tokens`` tokens before formatting.
    Label mapping follows SST-2's convention (1 -> positive): ``labels[0]`` for a positive
    example, ``labels[1]`` for a negative one; multi-token labels are supported and the
    full label span is recorded for teacher-forced scoring / loss masking.

    Returns ``(rows, manifest)``. Each row carries ``sentence``, ``prompt_ids``,
    ``label_str``, ``label_token_ids``, ``input_ids`` (prompt+label), and ``label_span``
    ``(start, end)`` over ``input_ids``. ``n`` optionally caps the number of rows drawn
    after a deterministic ``random.Random(seed)`` shuffle of the candidate order.
    """
    if len(labels) != 2:
        raise ValueError("SST-2 has two classes; provide exactly two label strings")
    # Label tokenisations are context-free constants; compute and assert once.
    label_token_ids = {lab: [int(t) for t in tokenizer(lab, add_special_tokens=False)["input_ids"]]
                       for lab in labels}
    for lab, ids in label_token_ids.items():
        if not ids:
            raise ValueError(f"Label {lab!r} tokenised to zero tokens")
    label_token_counts = {lab: len(ids) for lab, ids in label_token_ids.items()}

    ds = load_dataset("stanfordnlp/sst2", split=split)

    order = list(range(len(ds)))
    random.Random(seed).shuffle(order)

    rows, manifest = [], []
    for source_index in order:
        ex = ds[source_index]
        sentence = _normalize_text(ex.get("sentence", ""))
        if not sentence:
            continue
        sentence = _truncate(tokenizer, sentence, max_input_tokens)
        label_value = int(ex.get("label", -1))
        if label_value not in (0, 1):
            continue
        label_str = labels[0] if label_value == 1 else labels[1]
        prompt_text = template.format(sentence=sentence)
        prompt_ids = [int(t) for t in tokenizer(prompt_text, add_special_tokens=False)["input_ids"]]
        lab_ids = label_token_ids[label_str]
        input_ids = prompt_ids + lab_ids
        label_span = (len(prompt_ids), len(prompt_ids) + len(lab_ids))
        row = {
            "dataset": "sst2",
            "source_index": source_index,
            "sentence": sentence,
            "label": label_value,
            "label_str": label_str,
            "prompt_text": prompt_text,
            "prompt_ids": prompt_ids,
            "label_token_ids": lab_ids,
            "label_token_count": len(lab_ids),
            "input_ids": input_ids,
            "label_span": label_span,
        }
        rows.append(row)
        manifest.append({k: v for k, v in row.items()
                         if k not in ("input_ids", "prompt_ids", "label_token_ids")})
        if n is not None and len(rows) >= n:
            break
    if not rows:
        raise ValueError("SST-2 prompt loader produced no rows for the given split")
    info = {"label_token_ids": label_token_ids, "label_token_counts": label_token_counts,
            "template": template, "max_input_tokens": max_input_tokens,
            "sentence_starts_at": 0}
    return rows, manifest, info


# ═══════════════════════════════════════════════════════════════════════════════
# E7 parallel loaders (WP8)
#
# These return raw, join-keyed rows. They never filter by length, never sample, and
# never build a manifest — common/paired_manifests.py owns all of that, so the join
# key and the alignment policy live in exactly one place.
# ═══════════════════════════════════════════════════════════════════════════════

#: Pending Python-list blocks folded into an int32 array once this many accumulate, under
#: ``load_tinystories_blocks(as_array=True)``. Bounds the list-of-lists peak to roughly
#: this many blocks (~0.3 GB at 50k x 128) instead of the whole corpus (~12.8 GB).
_ARRAY_FOLD_ROWS = 50_000

FLORES_HF_PATH = "facebook/flores"
XNLI_HF_PATH = "facebook/xnli"

#: XNLI's integer labels, in the dataset's own order.
XNLI_LABEL_NAMES = ("entailment", "neutral", "contradiction")

#: Candidate id columns, most specific first. XNLI's original TSV distribution carries
#: ``promptID``/``pairID``; some mirrors preserve them and some do not.
_XNLI_ID_COLUMNS = ("promptID", "promptid", "prompt_id", "pairID", "pairid", "pair_id")


def _flores_access_hint(lang, exc):
    """Turn an opaque FLORES download failure into an actionable one.

    Added because both ways this fails are environmental rather than programmatic, and
    both surface deep inside a manifest build where the raw exception reads like a bug in
    the join:

    * ``facebook/flores`` is a **gated** dataset — it needs a Hugging Face account, the
      terms accepted on the dataset page, and a token in the environment. Measured
      2026-07-30 on ``datasets==4.8.4``: ``DatasetNotFoundError: ... is a gated dataset``.
    * the ungated community mirrors (``Muennighoff/flores200``, ``gsarti/flores_101``) are
      **loading-script** datasets, and script datasets were removed in ``datasets`` 3.0:
      ``RuntimeError: Dataset scripts are no longer supported``. Switching the id does not
      help, which is worth saying explicitly so nobody spends an afternoon on it.

    Purely additive and only on the error path — the success path is byte-identical.
    """
    detail = f"{type(exc).__name__}: {exc}"
    return (
        f"FLORES-200 could not be loaded for language {lang!r} from {FLORES_HF_PATH!r}.\n"
        f"  underlying error: {detail}\n"
        "  FLORES is gated on the Hub. To fix, once per machine:\n"
        "    1. accept the terms at https://huggingface.co/datasets/facebook/flores\n"
        "    2. hf auth login          (or set HF_TOKEN in the environment)\n"
        "  Note that swapping in an ungated mirror does NOT work: Muennighoff/flores200 "
        "and gsarti/flores_101 are loading-script datasets, and datasets>=3.0 refuses "
        "those outright. E7's FLORES half needs the authenticated download.\n"
        "  XNLI (facebook/xnli) is NOT gated and loads without a token, so "
        "prepare_xnli_manifest.py and the whole patching pipeline are unaffected."
    )


def load_flores_parallel(langs, split="devtest"):
    """Load FLORES-200 for several languages, keyed by source index.

    FLORES ``devtest`` is line-aligned across language configs by construction: row *i* of
    every config is the same sentence. That makes the source index the legitimate join key
    here — unlike XNLI, where a positional join is a silent correctness bug (see
    :func:`load_xnli_aligned`).

    Returns ``(rows, info)`` where ``rows`` is ``{lang: {source_index: {"text": ...}}}``
    and ``info`` records the row count seen per language, so the caller can assert the
    configs really are the same length before joining.
    """
    langs = list(langs)
    if not langs:
        raise ValueError("load_flores_parallel needs at least one language")
    rows, counts = {}, {}
    for lang in langs:
        try:
            ds = load_dataset(FLORES_HF_PATH, lang, split=split)
        except Exception as exc:                       # noqa: BLE001 - re-raised below
            raise RuntimeError(_flores_access_hint(lang, exc)) from exc
        per_lang = {}
        for source_index, ex in enumerate(ds):
            text = _normalize_text(ex.get("sentence", ""))
            if not text:
                continue
            per_lang[source_index] = {"text": text, "source_index": source_index}
        rows[lang] = per_lang
        counts[lang] = len(ds)
    info = {"dataset": FLORES_HF_PATH, "split": split, "languages": langs,
            "rows_per_language": counts, "join_strategy": "flores_source_index"}
    return rows, info


def _xnli_id_column(example):
    """The first available explicit id column on an XNLI row, or ``None``."""
    for name in _XNLI_ID_COLUMNS:
        if name in example:
            return name
    return None


def load_xnli_aligned(langs, split="test"):
    """Load XNLI for several languages, joined on an explicit id — never on row index.

    **The alignment trap.** XNLI is a translation of one English premise/hypothesis set,
    but its per-language shards are *not* guaranteed to be in a common order, and a
    positional join produces a manifest that looks perfectly healthy while destroying every
    cross-language claim (CLAUDE.md trap 2). Two strategies are supported, resolved at
    runtime and recorded in ``info["join_strategy"]``:

    ``"promptID"``
        The per-language configs expose an explicit id column (``promptID``/``pairID``).
        Rows are keyed by it and joined across languages on that key.

    ``"all_languages_structural"``
        They do not — which is the case for the current ``facebook/xnli`` per-language
        configs, whose features are only ``premise``/``hypothesis``/``label``. The
        ``all_languages`` config is loaded instead: there, one row *is* the translation
        set (``premise`` is a language→text mapping and ``hypothesis`` carries parallel
        ``translation``/``language`` lists), so alignment is structural rather than
        positional. The semantic key is a content hash of the English pair, which is stable
        across shuffles and across machines.

    Neither available ⇒ raise. There is no positional fallback, by design.

    Returns ``(rows, info)`` with ``rows`` as ``{lang: {key: row}}`` and each row carrying
    ``premise``, ``hypothesis``, ``label`` (int) and ``label_name``.
    """
    langs = list(langs)
    if not langs:
        raise ValueError("load_xnli_aligned needs at least one language")

    # --- strategy 1: an explicit id column on the per-language configs ---
    try:
        probe = load_dataset(XNLI_HF_PATH, langs[0], split=split)
        id_column = _xnli_id_column(probe[0]) if len(probe) else None
    except Exception as exc:
        probe, id_column = None, None
        probe_error = f"{type(exc).__name__}: {exc}"
    else:
        probe_error = None

    if probe is not None and id_column is not None:
        rows = {}
        for lang in langs:
            ds = probe if lang == langs[0] else load_dataset(XNLI_HF_PATH, lang, split=split)
            per_lang = {}
            for source_index, ex in enumerate(ds):
                key = str(ex[id_column])
                label = int(ex.get("label", -1))
                per_lang[key] = {
                    "premise": _normalize_text(ex.get("premise", "")),
                    "hypothesis": _normalize_text(ex.get("hypothesis", "")),
                    "label": label,
                    "label_name": (XNLI_LABEL_NAMES[label]
                                   if 0 <= label < len(XNLI_LABEL_NAMES) else "unknown"),
                    "source_index": source_index,
                }
            rows[lang] = per_lang
        info = {"dataset": XNLI_HF_PATH, "split": split, "languages": langs,
                "join_strategy": "promptID", "join_key_column": id_column,
                "rows_per_language": {k: len(v) for k, v in rows.items()}}
        return rows, info

    # --- strategy 2: the structurally aligned all_languages config ---
    try:
        ds = load_dataset(XNLI_HF_PATH, "all_languages", split=split)
    except Exception as exc:
        raise RuntimeError(
            "XNLI could not be joined on an explicit id and the structurally aligned "
            "'all_languages' config is unavailable. Refusing to fall back to a positional "
            "join, which would silently destroy every cross-language claim. "
            f"per-language probe: {probe_error}; all_languages: "
            f"{type(exc).__name__}: {exc}") from exc

    rows = {lang: {} for lang in langs}
    for source_index, ex in enumerate(ds):
        premises = ex.get("premise", {})
        hypothesis = ex.get("hypothesis", {})
        translations = list(hypothesis.get("translation", []))
        hyp_langs = list(hypothesis.get("language", []))
        hyp_by_lang = dict(zip(hyp_langs, translations))
        label = int(ex.get("label", -1))

        english = f"{premises.get('en', '')}␟{hyp_by_lang.get('en', '')}"
        key = hashlib.sha256(english.encode("utf-8")).hexdigest()[:24]

        for lang in langs:
            premise = _normalize_text(premises.get(lang, ""))
            hyp = _normalize_text(hyp_by_lang.get(lang, ""))
            if not premise or not hyp:
                continue
            rows[lang][key] = {
                "premise": premise,
                "hypothesis": hyp,
                "label": label,
                "label_name": (XNLI_LABEL_NAMES[label]
                               if 0 <= label < len(XNLI_LABEL_NAMES) else "unknown"),
                "source_index": source_index,
            }
    info = {"dataset": XNLI_HF_PATH, "split": split, "languages": langs,
            "join_strategy": "all_languages_structural",
            "join_key_column": "sha256(en premise␟en hypothesis)[:24]",
            "rows_per_language": {k: len(v) for k, v in rows.items()}}
    return rows, info
