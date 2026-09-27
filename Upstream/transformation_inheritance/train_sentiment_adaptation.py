# -*- coding: utf-8 -*-
"""train_sentiment_adaptation.py — E6B sentiment adaptation (WP6).

``03_MODULE_SPEC_e6_transformation.md`` §4. Fine-tunes a config-selected GPT-2-family
base on SST-2 in four conditions and writes the run directories
``evaluate_transformation.py`` reads. The original arm uses ``distilbert/distilgpt2``;
model-scale extensions use a distinct ``experiment_id``:

    python transformation_inheritance/train_sentiment_adaptation.py \\
      --config transformation_inheritance/configs/e6b_f1.yaml --seed 0

| id | params | data |
|----|--------|------|
| F0 | none   | the untrained base — no config, it is the drift reference |
| F1 | LoRA   | clean SST-2 |
| F2 | all    | clean SST-2 |
| F3 | LoRA   | 20% symmetric label corruption |
| F4 | all    | 20% symmetric label corruption |

Five properties that carry the experiment's validity, and are therefore explicit
--------------------------------------------------------------------------------
* **The sentence begins at position 0.** No instruction prefix (design §16.2): a constant
  prefix would manufacture a shared first-token anchor and invalidate every sink
  measurement in E6B. That format lives in :func:`corpus_providers.sst2_prompt_corpus`,
  which is called here rather than re-implemented, and batches are **right**-padded — left
  padding would move the sentence off position 0 for every short example.
* **Loss is on label tokens only.** Input positions are masked to ``-100``. The label span
  travels on the corpus item, so the mask cannot drift from the tokenisation.
* **Effective batch is 32 in both adaptation methods** (design §16.2). LoRA reaches it as
  32×1 and full fine-tuning as 16×2; a mismatch would confound the adaptation contrast
  with an optimisation one, so it is asserted at startup for every condition.
* **Corruption is train-only.** :func:`build_corruption_manifest` asserts it, and the
  validation corpus is built without a manifest at all.
* **Only the merged checkpoint is fingerprinted.** The frozen GPT-2 harness knows nothing
  about PEFT wrappers, so a LoRA condition saves ``merged/`` beside ``adapter/`` and proves
  they agree on CPU before either is used (design-delta D5). See ``MERGE_PARITY_DTYPE`` for
  the one recorded deviation from `03` §4.5, and why raising the precision strengthens that
  check rather than weakening it.

``--smoke`` trains a tiny random GPT-2 on synthetic rows with a word-level tokenizer: no
downloads, CPU/fp32, seconds. It exercises every branch including the LoRA merge.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common", _REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import provenance as prov  # noqa: E402

# The distillation trainer already solved seeding, scheduling, checkpoint I/O and the
# tied-weight problem (trap 6). Imported, never copied, so the two trainers cannot drift.
from train_distillation import (  # noqa: E402
    append_jsonl,
    latest_checkpoint,
    load_rng_state,
    lr_lambda,
    rng_state,
    seed_everything,
    tied_weight_keys,
)

TRAINER_VERSION = "train_sentiment_adaptation_v2"

#: design §16.2 — the same effective batch in both adaptation methods, so the E6B contrast
#: is about *how* the parameters move and not about how many examples moved them.
REQUIRED_EFFECTIVE_BATCH = 32

#: `03` §4.5's merge-parity tolerance, **unchanged**.
MERGE_PARITY_TOL = 1e-5

#: RECORDED DEVIATION from `03` §4.5, which says to check "in fp32 on CPU".
#:
#: §4.5's bar is an *absolute* logit tolerance, and distilgpt2's logits reach ~104. Meeting
#: 1e-5 absolutely at that scale demands a relative accuracy of ~1e-7, which is below
#: float32's own epsilon (1.19e-7) — so no implementation of the merge can pass in fp32 on
#: a real model, and a tiny smoke model passes only because its logits are O(1). Measured
#: on real distilgpt2 with a perturbed adapter:
#:
#:     float32:  max|logit| 104.077   max|delta| 1.068e-04   relative 1.026e-06
#:     float64:  max|logit| 103.469   max|delta| 1.990e-13   relative 1.923e-15
#:
#: The delta tracks machine epsilon in both (≈8.6x eps, ordinary accumulation over 768
#: dims), which is round-off and not a merge error. Raising the *tolerance* would have
#: weakened the check; raising the *precision* does not. A genuine merge fault — a wrong
#: alpha/r scale, a missed Conv1D transpose, an adapter that never applied — is systematic
#: and does not shrink with precision, so it still fails 1e-5 in float64 by orders of
#: magnitude. The fp32 delta is reported alongside as a diagnostic; only float64 gates.
MERGE_PARITY_DTYPE = torch.float64
MERGE_PARITY_DTYPE_NAME = "float64"

#: Five fixed sentences, never drawn from SST-2, so the parity check cannot accidentally
#: be run on data the adapter was fitted to.
PARITY_SENTENCES: Tuple[str, ...] = (
    "the film drifts along without ever finding its subject",
    "a warm and generous portrait of an unremarkable life",
    "nothing here works, and the ending works least of all",
    "it is impossible not to be moved by the final scene",
    "competent, forgettable, and over long before it ends",
)

#: `05` §2 asks for ECE over 10 **equal-width** bins. Equal-width, not equal-mass: the
#: two disagree whenever confidence is concentrated, which it is after fine-tuning.
ECE_BINS = 10


# ═══════════════════════════════════════════════════════════════════════════════
# Config
# ═══════════════════════════════════════════════════════════════════════════════


def load_config(path) -> Dict[str, Any]:
    import yaml

    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a YAML mapping at the top level")
    payload["config_path"] = str(path)
    return payload


def assert_effective_batch(config: Dict[str, Any]) -> int:
    """design §16.2 / `03` §4.4 — ``per_device_batch * grad_accum == 32``, or refuse.

    Raised rather than warned: a condition that trained at a different effective batch is
    not a noisy arm of the factorial, it is a different experiment, and the 2×2 would
    silently be measuring optimisation rather than adaptation method.
    """
    optim = config.get("optim", {})
    per_device = int(optim.get("per_device_batch_size", 0))
    accum = int(optim.get("grad_accum", 0))
    product = per_device * accum
    if product != REQUIRED_EFFECTIVE_BATCH:
        raise ValueError(
            f"{config.get('condition', '?')}: per_device_batch_size ({per_device}) x "
            f"grad_accum ({accum}) = {product}, but design §16.2 fixes the effective batch "
            f"at {REQUIRED_EFFECTIVE_BATCH} for every E6B condition. LoRA reaches it as "
            "32x1 and full fine-tuning as 16x2; changing one without the other confounds "
            "the adaptation contrast with an optimisation one.")
    return product


def label_token_report(tokenizer, labels: Sequence[str]) -> Dict[str, Any]:
    """How many tokens each label string produces (`03` §4.2, verified at startup).

    A single-token label would let scoring read one logit; a multi-token one must be scored
    over the whole span. The count is *measured* rather than assumed, because it is a
    property of the tokenizer and silently differs between GPT-2 and a smoke vocabulary.
    """
    counts = {}
    for label in labels:
        ids = tokenizer(label, add_special_tokens=False)["input_ids"]
        if not ids:
            raise ValueError(f"label {label!r} tokenised to zero tokens")
        counts[label] = [int(t) for t in ids]
    multi = any(len(ids) > 1 for ids in counts.values())
    return {"label_token_ids": counts,
            "label_token_counts": {k: len(v) for k, v in counts.items()},
            "multi_token": multi,
            "scoring": "full_label_span" if multi else "single_token",
            "note": ("at least one label is multi-token, so training and scoring run over "
                     "the full label span" if multi else
                     "both labels are single-token; the full span is still scored, which "
                     "reduces to the single-token case")}


# ═══════════════════════════════════════════════════════════════════════════════
# Corruption manifest (03 §4.3)
# ═══════════════════════════════════════════════════════════════════════════════


def build_corruption_manifest(train_rows: Sequence[Dict[str, Any]], rate: float = 0.20,
                              seed: int = 0, *, split: str = "train"):
    """Symmetric label corruption over the *training* rows only.

    ``train_rows`` are dicts carrying at least ``example_id`` and ``label``. Returns a frame
    with ``example_id``, ``original_label``, ``assigned_label``, ``flipped``.

    The four constraints `03` §4.3 names, and why each is a constraint rather than an
    average:

    * **exactly** ``round(rate * N)`` flipped — sampling each row independently at 20%
      would put the realised rate anywhere near 20%, and the corrupted-vs-clean contrast
      would then vary by seed for a reason that has nothing to do with the hypothesis;
    * **class-stratified within 1** — flipping more positives than negatives shifts the
      label prior, and a prior shift is a far simpler explanation for drift than anything
      mechanistic;
    * **train only** — asserted here, not merely documented, because a corrupted validation
      label makes every accuracy number meaningless;
    * **reproducible within a seed and distinct across seeds** — the seed is the unit of
      replication (design §15.1), so two seeds sharing a corruption set would be one
      measurement reported twice.
    """
    import pandas as pd

    if split != "train":
        raise ValueError(
            f"build_corruption_manifest was called for split={split!r}. Validation labels "
            "are never modified (03 §4.3); corruption applies to the training split only.")
    if not 0.0 <= rate <= 1.0:
        raise ValueError(f"corruption rate must be in [0, 1], got {rate}")

    rows = list(train_rows)
    n_total = len(rows)
    n_flip = int(round(rate * n_total))

    by_class: Dict[int, List[int]] = {}
    for index, row in enumerate(rows):
        by_class.setdefault(int(row["label"]), []).append(index)

    # Split the quota as evenly as the class sizes allow, then hand any remainder to the
    # larger class — that keeps |n_pos - n_neg| <= 1 without ever asking a class for more
    # rows than it has.
    classes = sorted(by_class)
    quota = {label: n_flip // len(classes) for label in classes}
    for offset in range(n_flip - sum(quota.values())):
        quota[classes[offset % len(classes)]] += 1
    for label in classes:
        quota[label] = min(quota[label], len(by_class[label]))
    shortfall = n_flip - sum(quota.values())
    for label in sorted(classes, key=lambda c: len(by_class[c]), reverse=True):
        take = min(shortfall, len(by_class[label]) - quota[label])
        quota[label] += take
        shortfall -= take

    rng = random.Random((seed + 1) * 7919 + 13)
    chosen: set = set()
    for label in classes:
        pool = list(by_class[label])
        rng.shuffle(pool)
        chosen.update(pool[:quota[label]])

    records = []
    for index, row in enumerate(rows):
        original = int(row["label"])
        flipped = index in chosen
        records.append({
            "example_id": int(row["example_id"]),
            "original_label": original,
            # SST-2 is binary, so "symmetric corruption" is a flip; with more classes this
            # would need an explicit off-diagonal distribution rather than 1 - label.
            "assigned_label": (1 - original) if flipped else original,
            "flipped": bool(flipped),
        })
    frame = pd.DataFrame(records)

    realised = int(frame["flipped"].sum())
    if realised != n_flip:
        raise AssertionError(f"expected exactly {n_flip} flips, produced {realised}")
    per_class = frame[frame["flipped"]].groupby("original_label").size()
    if len(per_class) > 1 and int(per_class.max() - per_class.min()) > 1:
        raise AssertionError(
            f"corruption is not class-stratified: {per_class.to_dict()} differs by more "
            "than 1")
    return frame


def corruption_map(manifest) -> Dict[int, int]:
    """``example_id -> assigned_label`` for the flipped rows only.

    That is exactly the shape ``sst2_prompt_corpus(corrupted_manifest=...)`` consumes, so
    the corpus provider rewrites the appended label token and records ``corrupted=True``
    per item — the trainer never re-tokenises a label itself.
    """
    if manifest is None or not len(manifest):
        return {}
    flipped = manifest[manifest["flipped"]]
    return {int(r["example_id"]): int(r["assigned_label"])
            for _, r in flipped.iterrows()}


# ═══════════════════════════════════════════════════════════════════════════════
# Data
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class LabelledBatch:
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    gold: torch.Tensor
    label_spans: List[Tuple[int, int]]


def collate(items: Sequence[Any], pad_token_id: int, device) -> LabelledBatch:
    """Right-padded batch with the loss masked to each item's label span.

    **Right** padding, deliberately: the sink is measured at position 0, and left padding
    would put a pad token there for every example shorter than the longest in the batch —
    manufacturing exactly the shared first-token anchor design §16.2 forbids.
    """
    width = max(len(item.input_ids) for item in items)
    input_ids = torch.full((len(items), width), pad_token_id, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    labels = torch.full((len(items), width), -100, dtype=torch.long)
    gold, spans = [], []
    for row, item in enumerate(items):
        ids = item.input_ids
        input_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
        attention[row, :len(ids)] = 1
        start, end = item.meta["label_span"]
        labels[row, start:end] = torch.tensor(ids[start:end], dtype=torch.long)
        gold.append(int(item.meta["label"]))
        spans.append((int(start), int(end)))
    return LabelledBatch(
        input_ids=input_ids.to(device), attention_mask=attention.to(device),
        labels=labels.to(device), gold=torch.tensor(gold, dtype=torch.long).to(device),
        label_spans=spans)


def epoch_order(n_items: int, seed: int, epoch: int) -> List[int]:
    """Deterministic example order for one pass, independent of global RNG state."""
    order = list(range(n_items))
    random.Random((seed + 1) * 100003 + epoch).shuffle(order)
    return order


# ═══════════════════════════════════════════════════════════════════════════════
# Metrics
# ═══════════════════════════════════════════════════════════════════════════════


def ece_equal_width(confidences: Sequence[float], correct: Sequence[bool],
                    n_bins: int = ECE_BINS) -> float:
    """Expected calibration error over ``n_bins`` **equal-width** bins on [0, 1].

    Equal-width because `05` §2 says so. Equal-mass bins would report a different number on
    the same predictions, and after fine-tuning confidence piles up near 1.0, which is
    precisely where the two disagree most.
    """
    conf = np.asarray(confidences, dtype=np.float64)
    hit = np.asarray(correct, dtype=np.float64)
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        # Half-open bins, closed at the top edge so confidence exactly 1.0 is counted.
        in_bin = (conf > lo) & (conf <= hi) if lo > 0 else (conf >= lo) & (conf <= hi)
        if not in_bin.any():
            continue
        total += (in_bin.mean() * abs(hit[in_bin].mean() - conf[in_bin].mean()))
    return float(total)


def _sequence_logprob(logits: torch.Tensor, ids: Sequence[int], start: int) -> float:
    """Teacher-forced log-prob of ``ids`` placed at ``[start, start+len(ids))``.

    ``logits[t]`` predicts position ``t + 1``, so the token at ``start`` is scored by
    ``logits[start - 1]``. Summed, not averaged: the two label strings are compared to each
    other and a length-normalisation would flatten a genuine length difference between them.
    """
    total = 0.0
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    for offset, token in enumerate(ids):
        position = start + offset - 1
        if position < 0:
            continue
        total += float(logprobs[position, int(token)])
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════════════════════════


def build_lora_config(config: Dict[str, Any]):
    from peft import LoraConfig

    lora = config.get("lora", {})
    return LoraConfig(
        r=int(lora.get("r", 8)),
        lora_alpha=int(lora.get("alpha", 16)),
        lora_dropout=float(lora.get("dropout", 0.05)),
        target_modules=list(lora.get("target_modules", ["c_attn", "c_proj"])),
        bias="none",
        task_type="CAUSAL_LM",
        # GPT-2 stores these as `Conv1D`, whose weight is transposed relative to `Linear`.
        # peft detects that and corrects with a warning; setting it explicitly makes the
        # choice deliberate and keeps the log clean.
        fan_in_fan_out=True,
    )


def load_base_model(model_id: str, *, dtype: str, device: str, revision=None,
                    local_files_only: bool = False):
    from transformers import AutoModelForCausalLM

    torch_dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16,
                   "float16": torch.float16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch_dtype,
        local_files_only=local_files_only)
    return model.to(device)


def merge_and_verify(peft_model, out_dir: Path, *, tokenizer, training_dtype: str,
                     device: str) -> Dict[str, Any]:
    """Merge the adapter, save it standalone, and prove the two agree (`03` §4.5, D5).

    The check runs in **fp32 on CPU** on five fixed sentences. Training dtype is bf16 and
    has nothing to do with the tolerance — a reader who saw ``1e-5`` next to ``bfloat16``
    could reasonably mistake this for a bf16 round-trip claim, so the JSON says outright
    that it is not.

    Only the merged copy is ever fingerprinted: the frozen GPT-2 harness resolves module
    paths like ``transformer.h.0.attn.c_attn`` and a PEFT wrapper renames them.
    """
    adapter_dir = out_dir / "adapter"
    merged_dir = out_dir / "merged"
    adapter_dir.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(adapter_dir))

    # `merge_and_unload()` folds the adapter into the base weights **in place** and returns
    # the unwrapped model — the wrapper and its return value share one set of tensors. So
    # the unmerged logits must be captured BEFORE the merge; holding a reference to the
    # wrapper and comparing afterwards compares a model against itself and reports 0.0 for
    # any adapter, broken or not. That is the vacuous-check failure of trap 3, and it is
    # why the tolerance below is only meaningful in this order.
    peft_model = peft_model.to("cpu").to(MERGE_PARITY_DTYPE).eval()
    encodings = [tokenizer(sentence, add_special_tokens=False, return_tensors="pt")
                 for sentence in PARITY_SENTENCES]
    with torch.no_grad():
        before = [peft_model(**ids).logits.clone() for ids in encodings]

    merged = peft_model.merge_and_unload().eval()
    with torch.no_grad():
        after = [merged(**ids).logits for ids in encodings]
    merged.float().save_pretrained(merged_dir, safe_serialization=True)

    deltas, relatives, per_sentence = [], [], []
    for sentence, ids, a, b in zip(PARITY_SENTENCES, encodings, before, after):
        delta = float((a - b).abs().max())
        scale = float(a.abs().max())
        relative = delta / scale if scale > 0 else float("nan")
        deltas.append(delta)
        relatives.append(relative)
        per_sentence.append({"sentence": sentence,
                             "n_tokens": int(ids["input_ids"].shape[1]),
                             "max_abs_logit_delta": delta,
                             "max_abs_logit": scale,
                             "relative_delta": relative})

    max_delta = float(max(deltas)) if deltas else float("nan")
    max_relative = float(max(relatives)) if relatives else float("nan")
    report = {
        "max_abs_logit_delta": max_delta,
        "max_relative_logit_delta": max_relative,
        "tolerance": MERGE_PARITY_TOL,
        "passed": bool(max_delta < MERGE_PARITY_TOL),
        "n_sentences": len(PARITY_SENTENCES),
        "per_sentence": per_sentence,
        "check_dtype": MERGE_PARITY_DTYPE_NAME,
        "check_device": "cpu",
        "training_dtype": training_dtype,
        "spec_check_dtype": "float32",
        "deviation_from_spec": (
            "03 §4.5 specifies fp32; this runs in float64. §4.5's 1e-5 is an ABSOLUTE "
            "logit tolerance and distilgpt2's logits reach ~104, so meeting it in fp32 "
            "would need ~1e-7 relative accuracy — below float32 epsilon (1.19e-7). "
            "Measured on real distilgpt2: fp32 gives 1.07e-04 absolute / 1.03e-06 "
            "relative, float64 gives 1.99e-13 / 1.92e-15; both are ~8.6x their own "
            "machine epsilon, i.e. round-off. The TOLERANCE is unchanged at 1e-5 — only "
            "the working precision was raised, which strengthens the check rather than "
            "weakening it: a real merge fault (wrong alpha/r, missed Conv1D transpose, "
            "adapter never applied) is systematic and fails by orders of magnitude at "
            "any precision."),
        "note": ("Parity is checked on CPU in the dtype named by `check_dtype`. The "
                 "training dtype above is recorded for provenance only and is NOT what "
                 "this tolerance describes — this is not a bf16 round-trip claim "
                 "(03 §4.5)."),
        "merged_dir": str(merged_dir),
        "adapter_dir": str(adapter_dir),
        "fingerprinted": "merged only — the frozen GPT-2 harness does not understand PEFT "
                         "module names",
        **prov.provenance_block(),
    }
    prov.write_json(out_dir / "merge_parity.json", report)
    if not report["passed"]:
        raise AssertionError(
            f"LoRA merge parity failed: max |Δlogit| = {max_delta:.3e} >= "
            f"{MERGE_PARITY_TOL} in {MERGE_PARITY_DTYPE_NAME} (relative "
            f"{max_relative:.3e}). The merged checkpoint does not reproduce the adapter, "
            "so every fingerprint taken from it would describe a different model "
            "(03 §4.5). At this precision round-off is ~1e-13, so a failure here is a "
            "systematic merge fault — check the alpha/r scale and the Conv1D transpose "
            "(fan_in_fan_out), not the tolerance.")
    merged.to(device)
    return report


# ═══════════════════════════════════════════════════════════════════════════════
# Smoke assets
# ═══════════════════════════════════════════════════════════════════════════════


def build_smoke_assets(root: Path, *, vocab_size: int = 96):
    """A word-level tokenizer and a tiny random GPT-2, so the whole trainer runs offline."""
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

    vocab = {"[UNK]": 0, "[PAD]": 1, "[EOS]": 2, "positive": 3, "negative": 4,
             "Sentiment": 5, ":": 6}
    vocab.update({f"w{i}": i + 7 for i in range(vocab_size - 7)})
    backend = Tokenizer(models.WordLevel(vocab=vocab, unk_token="[UNK]"))
    backend.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=backend, unk_token="[UNK]",
                                        pad_token="[PAD]", eos_token="[EOS]")
    tokenizer_dir = root / "tokenizer"
    tokenizer.save_pretrained(str(tokenizer_dir))

    config = GPT2Config(vocab_size=vocab_size, n_positions=64, n_embd=32, n_layer=4,
                        n_head=4, n_inner=64)
    model = GPT2LMHeadModel(config)
    base_dir = root / "base"
    model.save_pretrained(base_dir, safe_serialization=True)
    tokenizer.save_pretrained(str(base_dir))
    return tokenizer, tokenizer_dir, base_dir


def smoke_rows(tokenizer, n: int, seed: int) -> List[Any]:
    """Synthetic SST-2-shaped items in the real prompt format, built by hand.

    Deliberately *not* routed through ``sst2_prompt_corpus``: that provider downloads
    stanfordnlp/sst2, and the point of ``--smoke`` is that nothing leaves the machine. The
    item shape — ``input_ids``, ``meta['label_span']``, ``meta['label']`` — is identical, so
    the collator and the scorer take exactly the same path they take on real data.
    """
    rng = random.Random(seed)
    labels = {1: " positive", 0: " negative"}
    label_ids = {v: [int(t) for t in tokenizer(v, add_special_tokens=False)["input_ids"]]
                 for v in labels.values()}
    items = []
    for index in range(n):
        label = index % 2
        words = " ".join(f"w{rng.randrange(0, 80)}" for _ in range(rng.randint(3, 8)))
        prompt = f"{words}\nSentiment:"
        prompt_ids = [int(t) for t in
                      tokenizer(prompt, add_special_tokens=False)["input_ids"]]
        lab = label_ids[labels[label]]
        items.append(cp.CorpusItem(
            item_id=f"smoke:{index}", text=prompt + labels[label],
            input_ids=prompt_ids + lab, n_tokens=len(prompt_ids) + len(lab),
            meta={"dataset": "sst2_smoke", "source_index": index, "example_id": index,
                  "label": label, "label_str": labels[label],
                  "label_span": [len(prompt_ids), len(prompt_ids) + len(lab)],
                  "corrupted": False, "original_label": label}))
    return items


# ═══════════════════════════════════════════════════════════════════════════════
# Run setup
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunSetup:
    config: Dict[str, Any]
    run_dir: Path
    condition: str
    seed: int
    device: str
    dtype: str
    model: Any
    tokenizer: Any
    tokenizer_source: str
    train_items: List[Any]
    valid_items: List[Any]
    manifest: Any
    label_report: Dict[str, Any]
    adaptation: str
    per_device_batch: int
    grad_accum: int
    max_steps: int
    steps_per_epoch: int
    epochs: int
    smoke: bool
    base_model_id: str
    tokenizer_revision: Optional[str]
    label_sequences: Dict[int, List[int]] = field(default_factory=dict)


def prepare_run(config: Dict[str, Any], *, seed: int, output_dir: Path,
                device: Optional[str] = None, smoke: bool = False,
                n_train: Optional[int] = None, n_valid: Optional[int] = None,
                epochs: Optional[int] = None) -> RunSetup:
    """Everything a run needs, with every refusal made before a single step is taken."""
    condition = str(config.get("condition", "F?"))
    assert_effective_batch(config)

    optim = config.get("optim", {})
    data = config.get("data", {})
    adaptation = str(config.get("adaptation", "lora")).lower()
    if adaptation not in ("lora", "full"):
        raise ValueError(f"{condition}: adaptation must be 'lora' or 'full', got "
                         f"{adaptation!r}")

    # A smoke run is CPU/fp32 by construction, matching every other smoke in the repo: it
    # must produce the same numbers on a machine with no GPU, and bf16 on a 4-layer random
    # model would make the merge-parity tolerance meaningless.
    resolved_device = device or ("cpu" if smoke else
                                 ("cuda" if torch.cuda.is_available() else "cpu"))
    dtype = "float32" if smoke else str(optim.get("precision", "bfloat16"))
    if resolved_device == "cpu" and dtype == "bfloat16":
        # bf16 on CPU is supported but glacial, and every smoke path is CPU. Recorded in
        # run_config.json so a reader never has to guess which dtype produced a number.
        dtype = "float32"

    run_dir = Path(output_dir) / config.get("experiment_id", "e6b") / condition / \
        f"seed{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed)

    if smoke:
        tokenizer, tokenizer_dir, base_dir = build_smoke_assets(run_dir / "_smoke")
        tokenizer_source, base_model_id = str(tokenizer_dir), str(base_dir)
        tokenizer_revision = None
        train_items = smoke_rows(tokenizer, n_train or 24, seed)
        valid_items = smoke_rows(tokenizer, n_valid or 8, seed + 500)
    else:
        from transformers import AutoTokenizer

        base_model_id = str(config.get("base_model", "distilbert/distilgpt2"))
        tokenizer_source = str(config.get("tokenizer", base_model_id))
        tokenizer_revision = config.get("tokenizer_revision")
        if tokenizer_revision is None and tokenizer_source == base_model_id:
            tokenizer_revision = config.get("base_revision")
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_source, revision=tokenizer_revision)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        max_input = int(data.get("max_input_tokens", 64))
        train_corpus = cp.sst2_prompt_corpus(tokenizer, "train", n_train,
                                            max_input_tokens=max_input, seed=seed)
        train_items = list(train_corpus.items)
        valid_corpus = cp.sst2_prompt_corpus(tokenizer, "validation", n_valid,
                                             max_input_tokens=max_input, seed=seed)
        valid_items = list(valid_corpus.items)

    label_report = label_token_report(
        tokenizer, list(data.get("labels", [" positive", " negative"])))

    # Corruption is applied by rebuilding the training items through the same provider,
    # so the flipped label is re-tokenised by the code that tokenised the clean one.
    manifest = None
    if bool(config.get("corrupt_labels", False)):
        rows = [{"example_id": int(item.meta.get("example_id",
                                                 item.meta.get("source_index"))),
                 "label": int(item.meta["label"])} for item in train_items]
        manifest = build_corruption_manifest(
            rows, rate=float(config.get("corruption_rate", 0.20)), seed=seed)
        manifest.to_csv(run_dir / "corruption_manifest.csv", index=False,
                        encoding="utf-8")
        mapping = corruption_map(manifest)
        if smoke:
            train_items = _apply_corruption_to_items(train_items, mapping, tokenizer)
        else:
            train_items = list(cp.sst2_prompt_corpus(
                tokenizer, "train", n_train,
                max_input_tokens=int(data.get("max_input_tokens", 64)), seed=seed,
                corrupted_manifest=mapping).items)

    model = load_base_model(base_model_id, dtype=dtype, device=resolved_device,
                            revision=config.get("base_revision"),
                            local_files_only=smoke)
    if adaptation == "lora":
        from peft import get_peft_model

        model = get_peft_model(model, build_lora_config(config))

    per_device = int(optim.get("per_device_batch_size"))
    accum = int(optim.get("grad_accum"))
    n_epochs = int(epochs if epochs is not None else optim.get("epochs", 3))
    steps_per_epoch = max(1, math.ceil(len(train_items) / (per_device * accum)))
    max_steps = steps_per_epoch * n_epochs

    label_sequences = {
        label: [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]
        for label, text in ((1, " positive"), (0, " negative"))}

    return RunSetup(
        config=config, run_dir=run_dir, condition=condition, seed=seed,
        device=resolved_device, dtype=dtype, model=model, tokenizer=tokenizer,
        tokenizer_source=tokenizer_source, train_items=train_items,
        valid_items=valid_items, manifest=manifest, label_report=label_report,
        adaptation=adaptation, per_device_batch=per_device, grad_accum=accum,
        max_steps=max_steps, steps_per_epoch=steps_per_epoch, epochs=n_epochs,
        smoke=smoke, base_model_id=base_model_id,
        tokenizer_revision=tokenizer_revision, label_sequences=label_sequences)


def _apply_corruption_to_items(items, mapping: Dict[int, int], tokenizer):
    """Rewrite the appended label tokens of the flipped smoke items.

    The real path goes back through ``sst2_prompt_corpus``; this mirrors it for synthetic
    rows so the smoke exercises the same downstream shape (``corrupted``,
    ``original_label``, a label span that still matches ``input_ids``).
    """
    label_text = {1: " positive", 0: " negative"}
    ids_of = {v: [int(t) for t in tokenizer(v, add_special_tokens=False)["input_ids"]]
              for v in label_text.values()}
    out = []
    for item in items:
        example_id = int(item.meta.get("example_id", item.meta.get("source_index")))
        if example_id not in mapping:
            out.append(item)
            continue
        new_label = int(mapping[example_id])
        start, _end = item.meta["label_span"]
        new_ids = ids_of[label_text[new_label]]
        input_ids = list(item.input_ids[:start]) + new_ids
        meta = dict(item.meta)
        meta.update({"label": new_label, "label_str": label_text[new_label],
                     "label_span": [start, start + len(new_ids)], "corrupted": True})
        out.append(cp.CorpusItem(item_id=item.item_id,
                                 text=item.text[:len(item.text)] , input_ids=input_ids,
                                 n_tokens=len(input_ids), meta=meta))
    return out


def write_run_config(setup: RunSetup) -> Path:
    """``run_config.json`` — the identity ``evaluate_transformation.py`` reads.

    Carries ``base_model`` under the ``teacher`` key as well: the E6B comparand is the
    untrained base (F0), and the evaluator's ``--base`` reads that field, so a run
    directory is self-describing without the config beside it.
    """
    experiment_id = str(setup.config.get("experiment_id", "e6b"))
    payload = {
        "experiment_id": experiment_id,
        "experiment_family": setup.config.get("experiment_family", "e6b"),
        "condition_id": setup.condition,
        "run_id": f"{experiment_id}_{setup.condition}_seed{setup.seed}",
        "training_seed": setup.seed,
        "adaptation": setup.adaptation,
        "label_quality": "corrupt" if setup.config.get("corrupt_labels") else "clean",
        "base_model": setup.base_model_id,
        "base_revision": setup.config.get("base_revision"),
        "teacher": None,
        "teacher_revision": None,
        "public_reference": None,
        "tokenizer_name": setup.tokenizer_source,
        "tokenizer_revision": setup.tokenizer_revision,
        "preregistration": setup.config.get("preregistration"),
        "config_path": setup.config.get("config_path"),
        "device": setup.device,
        "dtype": setup.dtype,
        "smoke": setup.smoke,
        "effective_batch": setup.per_device_batch * setup.grad_accum,
        "per_device_batch_size": setup.per_device_batch,
        "grad_accum": setup.grad_accum,
        "epochs": setup.epochs,
        "steps_per_epoch": setup.steps_per_epoch,
        "max_steps": setup.max_steps,
        "n_train": len(setup.train_items),
        "n_valid": len(setup.valid_items),
        "corrupt_labels": bool(setup.config.get("corrupt_labels", False)),
        "corruption_rate": (float(setup.config.get("corruption_rate", 0.20))
                            if setup.config.get("corrupt_labels") else None),
        "n_corrupted": (int(setup.manifest["flipped"].sum())
                        if setup.manifest is not None else 0),
        "label_report": setup.label_report,
        "lora": setup.config.get("lora") if setup.adaptation == "lora" else None,
        "optim": setup.config.get("optim"),
        "trainer_version": TRAINER_VERSION,
        **prov.provenance_block(),
    }
    path = setup.run_dir / "run_config.json"
    prov.write_json(path, payload)
    return path


# ═══════════════════════════════════════════════════════════════════════════════
# Evaluation
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate(setup: RunSetup, model=None) -> Dict[str, Any]:
    """Accuracy, NLL and 10-equal-width-bin ECE on the *clean* validation split.

    Scored by comparing the summed teacher-forced log-prob of ``" positive"`` against
    ``" negative"`` at the same position — the generative model's own decision rule, not a
    classifier head bolted on for evaluation.
    """
    model = model or setup.model
    was_training = model.training
    model.eval()
    pad = setup.tokenizer.pad_token_id or 0

    confidences, correct, nlls = [], [], []
    with torch.no_grad():
        for start in range(0, len(setup.valid_items), setup.per_device_batch):
            chunk = setup.valid_items[start:start + setup.per_device_batch]
            batch = collate(chunk, pad, setup.device)
            logits = model(input_ids=batch.input_ids,
                           attention_mask=batch.attention_mask).logits
            for row, item in enumerate(chunk):
                span_start = int(item.meta["label_span"][0])
                scores = {label: _sequence_logprob(logits[row], ids, span_start)
                          for label, ids in setup.label_sequences.items()}
                # Two-class posterior over the label strings themselves.
                values = np.array([scores[0], scores[1]], dtype=np.float64)
                values -= values.max()
                probs = np.exp(values) / np.exp(values).sum()
                gold = int(item.meta["label"])
                predicted = int(np.argmax(probs))
                confidences.append(float(probs[predicted]))
                correct.append(predicted == gold)
                nlls.append(-float(np.log(max(probs[gold], 1e-12))))

    if was_training:
        model.train()
    return {
        "task_accuracy": float(np.mean(correct)) if correct else float("nan"),
        "task_nll": float(np.mean(nlls)) if nlls else float("nan"),
        "ece_10bin": ece_equal_width(confidences, correct),
        "validation_ce": float(np.mean(nlls)) if nlls else float("nan"),
        "n_eval": len(correct),
        "ece_bins": ECE_BINS,
        "ece_binning": "equal_width",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Training
# ═══════════════════════════════════════════════════════════════════════════════


def save_step(setup: RunSetup, step: int, optimizer, scheduler) -> Path:
    """Save the fingerprintable checkpoint for one step.

    A LoRA condition writes ``adapter/`` *and* ``merged/`` and proves they agree before
    either is used; a full-FT condition writes the model directly. Either way the directory
    that ``evaluate_transformation.py`` discovers holds plain HF weights.
    """
    path = setup.run_dir / "checkpoints" / f"step_{step}"
    path.mkdir(parents=True, exist_ok=True)

    if setup.adaptation == "lora":
        import copy

        # merge_and_unload() consumes the wrapper, so the parity check runs on a deep copy
        # and training continues from the untouched live model.
        merge_and_verify(copy.deepcopy(setup.model), path, tokenizer=setup.tokenizer,
                         training_dtype=setup.dtype, device="cpu")
        setup.tokenizer.save_pretrained(str(path / "merged"))
        fingerprintable = path / "merged"
    else:
        setup.model.save_pretrained(path, safe_serialization=True)
        setup.tokenizer.save_pretrained(str(path))
        fingerprintable = path

    torch.save(optimizer.state_dict(), path / "optimizer.pt")
    torch.save(scheduler.state_dict(), path / "scheduler.pt")
    torch.save(rng_state(), path / "rng_state.pt")
    prov.write_json(path / "trainer_state.json",
                    {"step": step, "condition": setup.condition, "seed": setup.seed,
                     "adaptation": setup.adaptation,
                     "fingerprint_dir": str(fingerprintable)})
    digest = prov.sha256_tree(fingerprintable, patterns=("*.safetensors", "*.json"))
    (path / "checkpoint_sha256.txt").write_text(digest + "\n", encoding="utf-8")
    return path


def train(setup: RunSetup, *, eval_every: int = 100, progress: bool = True
          ) -> Dict[str, Any]:
    """The E6B training loop. Checkpoints at every eval point and at each epoch end."""
    from torch.optim import AdamW

    optim = setup.config.get("optim", {})
    trainable = [p for p in setup.model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=float(optim.get("lr", 2e-4)),
                      weight_decay=float(optim.get("weight_decay", 0.01)),
                      betas=tuple(optim.get("betas", (0.9, 0.999))))
    warmup = int(round(float(optim.get("warmup_ratio", 0.06)) * setup.max_steps))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, warmup, setup.max_steps,
                                       float(optim.get("min_lr_ratio", 0.0))))
    clip = float(optim.get("grad_clip", 1.0))
    pad = setup.tokenizer.pad_token_id or 0

    started = time.time()
    step = 0
    setup.model.train()
    eval_row = {**evaluate(setup), "step": 0, "epoch": 0}
    append_jsonl(setup.run_dir / "eval_log.jsonl", eval_row)
    save_step(setup, 0, optimizer, scheduler)

    for epoch in range(setup.epochs):
        order = epoch_order(len(setup.train_items), setup.seed, epoch)
        micro = setup.per_device_batch
        window = micro * setup.grad_accum
        for offset in range(0, len(order), window):
            indices = order[offset:offset + window]
            if not indices:
                continue
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for chunk_start in range(0, len(indices), micro):
                chunk = [setup.train_items[i]
                         for i in indices[chunk_start:chunk_start + micro]]
                batch = collate(chunk, pad, setup.device)
                out = setup.model(input_ids=batch.input_ids,
                                  attention_mask=batch.attention_mask,
                                  labels=batch.labels)
                # Scale by the number of micro-batches so the gradient matches what a
                # single batch of `window` examples would have produced.
                loss = out.loss / max(1, math.ceil(len(indices) / micro))
                loss.backward()
                total += float(out.loss.detach())
            torch.nn.utils.clip_grad_norm_(trainable, clip)
            optimizer.step()
            scheduler.step()
            step += 1
            append_jsonl(setup.run_dir / "train_log.jsonl", {
                "step": step, "epoch": epoch, "loss": total / max(1, len(indices) // micro
                                                                 or 1),
                "lr": float(scheduler.get_last_lr()[0]),
                "n_examples": len(indices)})
            if eval_every and step % eval_every == 0:
                row = {**evaluate(setup), "step": step, "epoch": epoch}
                append_jsonl(setup.run_dir / "eval_log.jsonl", row)
                save_step(setup, step, optimizer, scheduler)
                if progress:
                    print(f"  step {step:>6} acc={row['task_accuracy']:.4f} "
                          f"nll={row['task_nll']:.4f} ece={row['ece_10bin']:.4f}")

        row = {**evaluate(setup), "step": step, "epoch": epoch, "epoch_end": True}
        append_jsonl(setup.run_dir / "eval_log.jsonl", row)
        save_step(setup, step, optimizer, scheduler)
        if progress:
            print(f"  epoch {epoch} end: step {step} acc={row['task_accuracy']:.4f} "
                  f"nll={row['task_nll']:.4f} ece={row['ece_10bin']:.4f}")

    wallclock = time.time() - started
    prov.write_json(setup.run_dir / "runtime_estimate.json", {
        "measured_steps": step, "wallclock_s": wallclock,
        "s_per_step": wallclock / max(1, step),
        "projected_full_run_s": (wallclock / max(1, step)) * setup.max_steps,
        "condition": setup.condition, "seed": setup.seed, "device": setup.device,
        "dtype": setup.dtype, "smoke": setup.smoke, **prov.provenance_block()})

    experiment_id = str(setup.config.get("experiment_id", "e6b"))
    summary = {
        "run_id": f"{experiment_id}_{setup.condition}_seed{setup.seed}",
        "run_dir": str(setup.run_dir), "condition": setup.condition, "seed": setup.seed,
        "adaptation": setup.adaptation, "steps": step, "epochs": setup.epochs,
        "final": evaluate(setup), "wallclock_s": wallclock,
        "n_corrupted": (int(setup.manifest["flipped"].sum())
                        if setup.manifest is not None else 0),
        "trainer_version": TRAINER_VERSION,
    }
    prov.write_json(setup.run_dir / "train_summary.json",
                    {**summary, **prov.provenance_block()})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--config", required=True, help="an E6B condition YAML config")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir",
                        default=str(_REPO / "transformation_inheritance" / "results"))
    parser.add_argument("--device", default=None)
    parser.add_argument("--eval-every", type=int, default=100,
                        help="eval + checkpoint every N optimiser steps (03 §4.4)")
    parser.add_argument("--n-train", type=int, default=None,
                        help="cap the training split (a short real run; full split by "
                             "default)")
    parser.add_argument("--n-valid", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None,
                        help="override the config's epoch count")
    parser.add_argument("--smoke", action="store_true",
                        help="tiny random GPT-2 + synthetic rows, CPU, no downloads")
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    setup = prepare_run(config, seed=args.seed, output_dir=Path(args.output_dir),
                        device=args.device, smoke=args.smoke, n_train=args.n_train,
                        n_valid=args.n_valid, epochs=args.epochs)
    write_run_config(setup)
    if args.progress:
        print(f"E6B {setup.condition} seed{setup.seed} -> {setup.run_dir}")
        print(f"  adaptation={setup.adaptation} effective_batch="
              f"{setup.per_device_batch * setup.grad_accum} "
              f"steps={setup.max_steps} device={setup.device} dtype={setup.dtype}")
        if setup.manifest is not None:
            print(f"  corrupted {int(setup.manifest['flipped'].sum())}/"
                  f"{len(setup.manifest)} training labels")
    summary = train(setup, eval_every=args.eval_every, progress=args.progress)
    print(f"  final acc={summary['final']['task_accuracy']:.4f} "
          f"nll={summary['final']['task_nll']:.4f} "
          f"ece={summary['final']['ece_10bin']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
