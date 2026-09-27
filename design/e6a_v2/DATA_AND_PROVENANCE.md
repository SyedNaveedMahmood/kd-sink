# Data, pairing and provenance

## Preparation is separate from training
Prepare versioned local token blocks before training; no live streaming shuffle inside the training loop. Pin dataset revision, split/selection policy, document IDs/content hashes, tokenizer ID/revision/files, EOS rule and preprocessing version. Save exact IDs/masks, block membership and aggregate SHA-256. Validate on load. Downloads are explicit commands, never import side effects.

S1/S3 use OpenWebText. The legacy `validation` name is an internally constructed holdout, not an assumed Hub split. Split at document level before packing, with zero document overlap among training/calibration/evaluation. Deduplicate exact normalized document hashes first. Near-duplicate removal is not claimed without an implemented, locked method.

Proposed D18 constants: UTF-8 NFC text, CRLF-to-LF conversion, strip leading/trailing whitespace; do not lowercase or collapse internal whitespace. Reject empty texts. Interpret the first64 SHA-256 bits as an unsigned big-endian integer modulo1000:0-9 evaluation,10-19 calibration,20-999 training. Preserve pinned canonical source order with content-hash tie-breaking, then deterministic per-seed block permutations. These proportions are proposals, not inherited manuscript facts. Tiny tests use separate synthetic manifests, not modified production rules.

Tokenize with the pinned GPT-2 tokenizer, add_special_tokens=False, one EOS between documents, no forced BOS per block. Concatenate within a split and form128-token blocks; drop only the final incomplete block. Record document boundaries, but do not silently add a block-diagonal attention mask. Absolute position resets per block; slot0 is not necessarily a BOS token.

## Paired initialization and stream
Create the random student once per seed on CPU in FP32 from the verified architecture config. Save canonical tensor-content hash and exact weights; reuse across all conditions and GPU replicas. Never load pretrained medium or DistilGPT-2 weights. Preparation/calibration must not consume training RNG state.

Generate ordered block IDs independently of microbatching. Update u receives exactly64 successive blocks, identically across paired conditions, split by the one common batch plan. Persist cursor, epoch, sampler generator state and a reproducible prefix/rolling hash. Future epochs use a documented seeded permutation; extension appends a reproducible suffix without rebuilding the consumed prefix.

At10k:640,000 sequence presentations,81,920,000 input tokens and normally81,280,000 shifted targets. Presentation counts are not necessarily unique-token counts. Log repeated epochs/data exposure and do not silently recycle/reorder when data ends.

## Frozen panels
Rank held-out blocks by a second fixed hash with salt `e6a-v2-panels`. Dense64 is nested in full300, nested in NLL2000. They are not independent replicates.
- `owt_dense64`:64 blocks, same IDs every100 updates; retain per-item outcomes.
- `owt_full300`:300 blocks, all retained checkpoints and causal/scope analysis.
- `owt_lm2000`:2,000 blocks, clean CE/PPL/accuracy at0 and10k; do not mandate expensive interventions on all2000.
- `calibration16x64`: separate calibration-partition blocks,16 effective batches, never production evaluation or training data.
- `xdomain300`:100 SST-2 validation sentences,100 GSM8K test questions,100 HumanEval prompts; deterministic content-hash ranking, max128 real tokens, minimum2, right padding with independent mask. Save exact source fields. No answer/completion/test text in inputs. A max40 rendering uses the same documents.
- `owt_long`:300 frozen natural-text windows for approved512/1024 context checks; use common underlying documents across lengths where feasible and report context shift.

Insufficient rows stop panel creation; no replacement domains or resampling after outcomes. No HumanEval execution. These are text language-modeling probes, not SST2 accuracy, GSM8K solution accuracy or HumanEval pass@k.

## Pythia
Retokenize the same raw documents with the pinned GPT-NeoX/Pythia tokenizer into separate manifests. Never feed GPT-2 IDs to Pythia. Preserve tokenizer-specific context, masks and valid-target counts. Within-family trajectories are primary; raw cross-tokenizer CE/PPL is not directly comparable. Document external training tokens from model metadata, not S1's batch.

## Immutable artifact/run records
Record study/condition/method/seed/hardware-block/replica IDs; protocol/code commit and clean-tree state; resolved config; model/tokenizer/data revisions and hashes; initialization/order/panel hashes; dependency lock; OS/Python/Torch/Transformers/CUDA/driver; GPU name/UUID/VRAM; numerical policy; common-batch hash; calibration hash; checkpoint parent/extension lineage; metric/intervention versions and timings.

Inherited candidate revisions to VERIFY, not newly validated downloads: large `32b71b12589c2f8d625668d2335a01cac3249519`, medium `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`, GPT-2 config/tokenizer `607a30d783dfa663caf39e06633721c8d4cfcd7e`. Assert actual layer/head/width counts and canonical IDs. An inaccessible revision must not silently become `main`.

Use an operator-specified external artifact root. Commit small manifests/schemas when licensing permits, not raw text, weights or credentials. Verify source licenses/terms. After preparation, training must work offline and without Upstream.
