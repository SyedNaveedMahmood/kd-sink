# Data, pairing and provenance

## Preparation is separate from training
Prepare versioned local token blocks before training; no live streaming shuffle inside the training loop. Pin dataset revision, split/selection policy, document IDs/content hashes, tokenizer ID/revision/files, EOS rule and preprocessing version. Save exact IDs/masks, block membership and aggregate SHA-256. Validate on load. Downloads are explicit commands, never import side effects.

S1 uses pinned `Skylion007/openwebtext` revision `79d93d786212f7344586290adb811d4ae6a1762c`, one `train` source stream. The historical `validation` name is an internally constructed window, not a Hub split. Source row indices are **training [0,400000)** and **validation [400000,408000)**. The new **training-only calibration [408000,416000)** window is disjoint from both and is not a historical evaluation region. Windows are fixed before tokenization; no document deduplication or hash partition changes historical membership. Record duplicate normalized hashes if present; do not claim deduplication or near-duplicate removal. S3 remains optional under its separately approved data recipe.

Normalize each source `text` with `" ".join(str(text).strip().split())`; `None` becomes empty. This collapses all whitespace and performs no NFC conversion or case conversion. Within each window, shuffle source document indices using `random.Random(seed).shuffle`. Skip empty normalized/tokenized documents. Call pinned GPT-2 tokenizer with `add_special_tokens=False`; append one EOS **after every nonempty document, including the last**, then concatenate and greedily cut nonoverlapping 128-token blocks. Drop only the incomplete tail. Preserve source row indices, normalized text hashes, block token IDs, document ownership, tails, and sealed aggregate hashes. No block-diagonal mask; absolute positions reset per block, and slot0 need not be BOS. This is the audited historical recipe, implemented independently in `sinklab.owt_compat` with no Upstream runtime dependency.

## Paired initialization and stream
Create the random student once per seed on CPU in FP32 from the verified architecture config. Save canonical tensor-content hash and exact weights; reuse across all conditions and GPU replicas. Never load pretrained medium or DistilGPT-2 weights. Preparation/calibration must not consume training RNG state.

Generate ordered block IDs independently of microbatching. The historical block-epoch shuffle uses `random.Random((seed + 1) * 100003 + epoch).shuffle` on each epoch's block indices. Update u receives exactly64 successive blocks, identically across paired conditions, split by the one common batch plan. Persist cursor, epoch, permutation, and reproducible prefix/rolling hash. Extension appends deterministic later epochs without rebuilding the consumed prefix.

At10k:640,000 sequence presentations,81,920,000 input tokens and normally81,280,000 shifted targets. Presentation counts are not necessarily unique-token counts. Log repeated epochs/data exposure and do not silently recycle/reorder when data ends.

## Frozen panels
Use source/packing order, not outcome-dependent ranking. Dense64 is the first64 of the first300 validation blocks. The next2000 validation blocks are a disjoint PPL/CE region; they are **not nested** in full300. These are fixed observations, not independent replicates.
- `owt_dense64`: first64 validation blocks, same IDs every100 updates; retain per-item outcomes.
- `owt_full300`: first300 validation blocks, all retained checkpoints and causal/scope analysis.
- `owt_lm2000`: validation blocks300..2299, clean CE/PPL/accuracy at0 and10k; no mandatory expensive interventions on all2000.
- `calibration16x64`: first1024 calibration-window blocks as16 effective batches, never production evaluation or training data.
- `xdomain300`:100 SST-2 validation sentences,100 GSM8K test questions,100 HumanEval prompts; deterministic content-hash ranking, max128 real tokens, minimum2, right padding with independent mask. Save exact source fields. No answer/completion/test text in inputs. A max40 rendering uses the same documents.
- `owt_long`:300 frozen natural-text windows for approved512/1024 context checks; use common underlying documents across lengths where feasible and report context shift.

Insufficient rows stop panel creation; no replacement domains or resampling after outcomes. No HumanEval execution. These are text language-modeling probes, not SST2 accuracy, GSM8K solution accuracy or HumanEval pass@k.

## Pythia
Retokenize the same raw documents with the pinned GPT-NeoX/Pythia tokenizer into separate manifests. Never feed GPT-2 IDs to Pythia. Preserve tokenizer-specific context, masks and valid-target counts. Within-family trajectories are primary; raw cross-tokenizer CE/PPL is not directly comparable. Document external training tokens from model metadata, not S1's batch.

## Immutable artifact/run records
Record study/condition/method/seed/hardware-block/replica IDs; protocol/code commit and clean-tree state; resolved config; model/tokenizer/data revisions and hashes; initialization/order/panel hashes; dependency lock; OS/Python/Torch/Transformers/CUDA/driver; GPU name/UUID/VRAM; numerical policy; common-batch hash; calibration hash; checkpoint parent/extension lineage; metric/intervention versions and timings.

Verified repository revisions for this amendment: large `32b71b12589c2f8d625668d2335a01cac3249519`, medium config `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`, GPT-2 tokenizer `607a30d783dfa663caf39e06633721c8d4cfcd7e`, OpenWebText `79d93d786212f7344586290adb811d4ae6a1762c`. The cached large weight and downloaded small tokenizer/config files have local hashes in `protocols/s1_artifact_partial_v1.json`. The production corpus and panels have **not** been prepared; their hashes remain null. An inaccessible revision must not silently become `main`.

Use an operator-specified external artifact root. Commit small manifests/schemas when licensing permits, not raw text, weights or credentials. Verify source licenses/terms. After preparation, training must work offline and without Upstream.
