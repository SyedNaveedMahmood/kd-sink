# 03 — Module Spec: `transformation_inheritance/` (E6)

Closes G7. Everything here is new code; nothing in this package modifies a frozen file.

---

## 1. `train_distillation.py` (WP5) — E6A

### 1.1 CLI

```bash
python transformation_inheritance/train_distillation.py \
  --config transformation_inheritance/configs/e6a_logit_attention_kd.yaml \
  --seed 0 --output-dir transformation_inheritance/results \
  [--max-steps 2000] [--resume auto] [--smoke]
```

`--max-steps` overrides the config for the pilot gate. `--smoke` runs 5 steps on random
tiny configs, CPU, fp32, no HF download.

### 1.2 Startup assertions (design-delta D6)

Before any training step, load teacher, public-8M reference config, and the student
config, then assert and record:

```python
assert tok.vocab_size == teacher.config.vocab_size
assert len(tok) == public8m_cfg.vocab_size == student_cfg.vocab_size
assert tok.bos_token_id == teacher.config.bos_token_id
assert tok.eos_token_id == teacher.config.eos_token_id
assert student_cfg.num_heads == teacher.config.num_heads == 16
```

Also read and record `teacher.config.attention_layers` and
`student_cfg.attention_layers` — the global/local pattern is needed for `L_ATTN`
masking and must not be assumed. Abort with a clear message on any mismatch; write the
results into `run_config.json["vocab_assertions"]`.

### 1.3 Student construction

```python
cfg = AutoConfig.from_pretrained("roneneldan/TinyStories-8M")
student = AutoModelForCausalLM.from_config(cfg)   # RANDOM INIT — never from_pretrained
```

Seed everything before construction: `torch.manual_seed`, `np.random.seed`,
`random.seed`, `torch.cuda.manual_seed_all`. Record `param_count` and a sha256 of the
initial state dict — three conditions at the same seed must have **byte-identical**
initial weights, and `tests/test_distillation_init.py` asserts this.

### 1.4 Data

`load_tinystories_blocks(tokenizer, "train", block_size=128, seed=seed)`:
tokenise without special tokens, one EOS between stories, shuffle story order with the
run seed **before** packing, pack into exact 128-token blocks, and save the contributing
story indices per block. The block manifest is written to
`<run_dir>/block_manifest.parquet` and its sha256 recorded. Conditions sharing a seed
must share this hash exactly — the design pairs conditions by seed and data order, and
that pairing must be provable, not assumed.

### 1.5 Losses

```python
def ce_loss(student_logits, labels) -> Tensor

def kd_loss(student_logits, teacher_logits, T=2.0, mask=None) -> Tensor
    # KL(teacher_T || student_T) per token, mean over valid tokens, times T**2.
    # Applied ONCE. Teacher logits carry no gradient (detach + torch.no_grad
    # around the teacher forward).

def attn_js_loss(student_attn, teacher_attn, layer_map, valid_mask) -> Tensor
    # Mean Jensen-Shannon divergence between post-softmax distributions over
    # mapped layers, all 16 heads, non-padding query positions.
    # JSD(p,q) = 0.5*KL(p||m) + 0.5*KL(q||m), m = 0.5*(p+q).
    # Masked entries are EXCLUDED, not renormalised over invalid positions.
```

Layer map (design §8.3): teacher 0→student 1, 1→3, 2→5, 3→7.

**Local-attention masking.** Teacher layer *t* and student layer *s* may have different
attention types (global vs local window). `valid_mask[t,s]` is the elementwise AND of
both layers' causal/window masks. Where a mapped pair has different attention types, the
loss is computed on the intersection and the pair is flagged; `attn_js_loss` returns a
per-pair breakdown so global-mapped and local-mapped pairs can be reported separately
(design §16.1). Read window size from `config.window_size` and pattern from
`config.attention_layers`.

Objectives:

| id | loss |
|---|---|
| D0 | `L_CE` |
| D1 | `0.50*L_CE + 0.50*L_KD`, T=2.0 |
| D2 | `0.45*L_CE + 0.45*L_KD + 0.10*L_ATTN`, T=2.0 |

Log every component separately every step, even when its weight is zero — `L_ATTN` is
an evaluation metric for D0 and D1 (design §8.6).

### 1.6 Training configuration

Exactly design §8.4. bf16 on CUDA / fp32 on CPU; seq 128; per-device batch 16; grad
accum 4; AdamW lr 5e-4, betas (0.9, 0.95), eps 1e-8, wd 0.1; 500 warmup; cosine to 10%
of peak; clip 1.0; max 10,000 steps; seeds 0,1,2.

Checkpoints at 0, 100, 250, 500, 1000, 2000, 5000, 10000. **Step 0 is saved before the
first optimiser update** — implement as an explicit save call before the training loop,
not as a step-0 branch inside it.

Each checkpoint saves model, optimiser, scheduler, and RNG state
(`torch`, `cuda`, `numpy`, `random`) so `--resume auto` is exact.
`tests/test_resume_exactness.py`: train 20 steps, checkpoint at 10, resume, and assert
the step-20 weights are bit-identical to the uninterrupted run.

Teacher is loaded once, `eval()`, `requires_grad_(False)`, and kept in bf16. Teacher
forward runs under `torch.no_grad()` inside the step for D1/D2 only.

### 1.7 VRAM

Teacher (33M, bf16) + student (8M) + activations at batch 16 × 128 fits far inside
16 GB. `L_ATTN` retains attention maps for four teacher and four student layers at
`[16, 16, 128, 128]` ≈ 8.4M floats each — still small, but request
`output_attentions=True` only for the mapped layers if the transformers version
supports per-layer selection; otherwise accept full capture and document it.

### 1.8 Outputs

```
results/e6a/<condition>/seed<k>/
  run_config.json
  block_manifest.parquet
  train_log.jsonl              # per-step: all loss components, lr, grad norm, tokens
  runtime_estimate.json        # written after step 100
  checkpoints/step_<n>/        # model + optim + sched + rng + checkpoint_sha256.txt
  eval_log.jsonl               # per-checkpoint val CE, KL, top-1, top-5, L_ATTN
```

---

## 2. Pilot gate (design §8.5)

`train_distillation.py --max-steps 2000 --seed 0` for D0/D1/D2, then
`evaluate_transformation.py` at steps 0, 250, 500, 1000, 2000. Add a script
`transformation_inheritance/check_pilot_gate.py` that evaluates the four criteria and
writes `pilot_gate.json` with an explicit `proceed: true|false`:

1. all conditions reduce TinyStories val CE vs step 0;
2. at least one condition has baseline sink > 0.15 by step 2000;
3. at least one mechanistic metric differs between D0 and D2 by ≥ 0.10;
4. manual vs NNsight attention agree within the Neo tolerance on 5 examples.

On gate-2 failure the script prints the design's prescribed remedy (extend seed 0 to
5,000 steps) rather than concluding failure. Do not launch Phase 2 without
`proceed: true`.

---

## 3. `configs/e6a_*.yaml`

```yaml
experiment_id: e6a
condition: D2                     # D0 | D1 | D2
teacher: roneneldan/TinyStories-33M
teacher_revision: null
student_config_from: roneneldan/TinyStories-8M
student_init: random
public_reference: roneneldan/TinyStories-8M
tokenizer: roneneldan/TinyStories-33M

loss:
  ce_weight: 0.45
  kd_weight: 0.45
  attn_weight: 0.10
  temperature: 2.0
  layer_map: {0: 1, 1: 3, 2: 5, 3: 7}

data:
  dataset: roneneldan/TinyStories
  train_split: train
  eval_split: validation
  block_size: 128
  eos_between: true

optim:
  lr: 5.0e-4
  betas: [0.9, 0.95]
  eps: 1.0e-8
  weight_decay: 0.1
  warmup_steps: 500
  schedule: cosine
  min_lr_ratio: 0.10
  grad_clip: 1.0
  per_device_batch_size: 16
  grad_accum: 4
  max_steps: 10000
  precision: bfloat16

checkpoints: [0, 100, 250, 500, 1000, 2000, 5000, 10000]
seeds: [0, 1, 2]

eval:
  sink_corpus: {split: validation, n_blocks: 300, purpose: sink}
  ppl_corpus:  {split: validation, n_blocks: 2000, purpose: ppl}
  ce_corpus_trajectory: {n_blocks: 300}       # design-delta D3
  ce_corpus_endpoints:  {n_blocks: 2000}
  cross_domain: frozen_e1
  band_frac: [0.25, 0.90]
```

`e6a_ce.yaml` sets weights (1.0, 0, 0); `e6a_logit_kd.yaml` (0.5, 0.5, 0).
Everything else is identical across the three — diff them in review to confirm.

---

## 4. `train_sentiment_adaptation.py` (WP6) — E6B

### 4.1 Conditions

| ID | params | data |
|---|---|---|
| F0 | none | base `distilbert/distilgpt2` |
| F1 | LoRA | clean SST-2 |
| F2 | all | clean SST-2 |
| F3 | LoRA | 20% symmetric label corruption |
| F4 | all | 20% symmetric label corruption |

### 4.2 Prompt format

`{sentence}\nSentiment:` with targets `" positive"` / `" negative"`. The sentence
**begins at position 0** — no instruction prefix (design §16.2: a constant prefix would
manufacture a shared first-token anchor and invalidate the whole measurement). Loss is
applied to label tokens only; input tokens are masked with `-100`. Verify at startup how
many tokens each label string produces and score the full sequence if either is
multi-token.

### 4.3 Corruption manifest

```python
def build_corruption_manifest(train_rows, rate=0.20, seed=0) -> pd.DataFrame
    # Columns: example_id, original_label, assigned_label, flipped
    # Constraints:
    #   - exactly round(rate * N) flipped
    #   - class-stratified: |n_flipped_pos - n_flipped_neg| <= 1
    #   - validation labels NEVER modified (assert the manifest covers train only)
    #   - distinct per seed, reproducible per seed
```

`tests/test_corruption_manifest.py` checks all four properties plus cross-seed
distinctness and within-seed reproducibility.

### 4.4 Hyperparameters

Design §9.5. LoRA: `target_modules=["c_attn","c_proj"]`, r=8, α=16, dropout 0.05,
lr 2e-4, batch 32, accum 1, wd 0.01, warmup ratio 0.06, cosine. Full FT: lr 5e-5,
batch 16, accum 2, same wd/warmup/schedule. **Effective batch size is 32 in both**
(design §16.2) — assert `per_device_batch * grad_accum == 32` for every F-condition at
startup.

3 epochs, seeds 0/1/2, eval every 100 steps and at each epoch end, early stopping
disabled, bf16, clip 1.0, max 64 input tokens before the label.

### 4.5 LoRA merge parity (design-delta D5)

After training, merge the adapter (`merge_and_unload()`) and save a standalone
checkpoint. Keep both. Then verify: load merged and unmerged **in fp32 on CPU**, run
five fixed parity sentences, assert max abs logit difference < 1e-5. Write
`merge_parity.json`. Training dtype is bf16 and irrelevant to this check — state that
in the JSON so it is not mistaken for a bf16 tolerance claim.

Only the merged checkpoint is fingerprinted, because the frozen GPT-2 harness knows
nothing about PEFT wrappers.

### 4.6 Outputs

```
results/e6b/<condition>/seed<k>/
  run_config.json
  corruption_manifest.csv        # F3/F4 only
  train_log.jsonl
  eval_log.jsonl                 # acc, NLL, ECE(10 equal-width bins) per checkpoint
  checkpoints/step_<n>/          # merged/ and adapter/ for LoRA conditions
  merge_parity.json
  runtime_estimate.json
```

---

## 5. Public-checkpoint screening

`transformation_inheritance/screen_public_pairs.py`: a thin driver over
`fingerprint_runner` for `gpt2` vs `distilgpt2`, and Qwen2.5 0.5B/1.5B base vs instruct
(design §9.9). No training. Writes `screening_fingerprints.csv` in the
`FingerprintRecord` schema. This is the cheapest evidence in the whole project and
should run first in Phase 1.

The 3B pair is run only if 0.5B and 1.5B agree in direction.

---

## 6. `evaluate_transformation.py` (WP10)

### 6.1 CLI

```bash
python transformation_inheritance/evaluate_transformation.py \
  --run-dir results/e6a/D2/seed0 \
  --steps all --engine nnsight --with-delta-ce --resume
```

### 6.2 Behaviour

For each checkpoint:

1. `handle = load_handle(arch, ckpt_path, engine=..., dtype=...)`, arch inferred from
   config (`GPTNeoForCausalLM` → `neo`, `GPT2LMHeadModel` → `gpt2`).
2. `band = normalised_depth_band(num_layers)`.
3. `compute_fingerprint` on the in-domain sink corpus, the frozen E1 cross-domain
   corpus, and (E6B) the SST-2 validation corpus.
4. ΔCE per design-delta D3: 300-block subset for every checkpoint; 2,000-block full set
   at steps 0 and final only.
5. Teacher fingerprint computed once and cached; every comparison uses
   `mutual_interventions(teacher_handle, student_handle)`.
6. Write one row per `(run_id, step, corpus_id)` to `checkpoint_metrics.csv`
   (schema in `05` §2).

### 6.3 Resumability

The unit of work is `(run_id, step, corpus_id, intervention)`. Before computing, check
the fingerprint cache (`02` §3.6) and the existing CSV rows; skip completed units.
`--resume` is the default; `--force` recomputes. Non-negotiable given the
1.5–3 GPU-day budget for E6A alone.

### 6.4 Matched-loss selection

Design §8.7 requires matched-loss as well as equal-step comparison. Implement in
`aggregate_transformation.py`, not here: for each condition, select the **earliest**
checkpoint whose validation CE is closest to D0's final CE. Since checkpoints are
discrete, record the realised CE gap alongside the selection and refuse the comparison
if the gap exceeds 0.05 nats, flagging it rather than silently comparing mismatched
losses. The same logic gives E6B's matched-accuracy comparison.

---

## 7. `aggregate_transformation.py` (WP10)

Reads all `checkpoint_metrics.csv`, produces:

| Output | Contents |
|---|---|
| `e6a_contrasts.csv` | the five primary contrasts (design §8.7), per seed and pooled |
| `e6a_matched_loss.csv` | equal-step vs matched-loss, with realised CE gaps |
| `e6b_drift.csv` | fingerprint/topology/carrier drift trajectories |
| `e6b_early_warning.csv` | the three onset steps per run (design §9.8) |
| `e6b_factorial.csv` | 2×2 (LoRA vs full) × (clean vs corrupt), including interaction |
| `table2_inheritance_components.csv` | topological, mechanistic, functional as separate columns |
| `fig2_trajectories.pdf`, `fig3_matched_loss.pdf`, `fig4_drift.pdf` | design §19 |
| `go_no_go.json` | design §18 criteria evaluated mechanically |

Statistics via `inheritance_metrics`: paired seed contrasts with the
`min_attainable_p` note, bootstrap CIs labelled as sampling uncertainty (never as
model-training uncertainty), BH correction within each pre-registered family.

`go_no_go.json` must be produced automatically and consulted before Phase 3. Writing
the criteria as code before seeing results is the pre-registration.

Plot conventions follow `emergence_dynamics_analysis.py` — import its style helpers
rather than restyling.
