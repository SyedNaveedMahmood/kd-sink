# COMMANDS.md — running the paper-ready E6/E7 experiments

Every command needed to go from a fresh clone to the artefacts the paper quotes, in
dependency order. Copy-paste runnable.

**Conventions.** The interpreter is the repo venv's: `./.venv/Scripts/python.exe` on
Windows, `./.venv/bin/python` on Linux. All paths are relative to the repo root, and every
command is run from there.

**Shell.** The loops below are **bash** (Git Bash ships with Git for Windows and is the
easiest way to run them verbatim). PowerShell equivalents are given beside each loop,
because PowerShell 5.1 does not understand `for x in ...; do ... done`. Two PowerShell
gotchas that bite this repo specifically:

- `transformers` writes progress bars to **stderr**, and under
  `$ErrorActionPreference = "Stop"` piping them (`2>&1 | Tee-Object`) wraps each line in a
  terminating `NativeCommandError` even when python exits 0. Take the verdict from
  `$LASTEXITCODE`, not from whether an error was printed. `scripts/run_nnsight_rest_smoke_tests.ps1`
  already does this.
- `&&` and `||` do not exist in PowerShell 5.1. Use `; if ($?) { ... }`.

**Everything is resumable.** Training resumes bit-exactly from its last checkpoint,
evaluation skips completed `(run_id, step, corpus_id)` units, and patching skips completed
`(model, target language, object, layer)` units. Re-run any command freely after an
interruption; none of them redoes finished work.

> **One exception, and it is a trap.** Do **not** resume a pilot run into the longer Phase 2
> run in the same directory. `LambdaLR` does not serialise its lambda, so the restored
> scheduler would keep the pilot's step count while following the 10,000-step cosine — the
> learning rate jumps at the resume point and the run follows neither pre-registered
> schedule (§1.6). `assert_resumable_schedule` now refuses this with an explanation. Use a
> **fresh run directory** for Phase 2, or `--resume none`.

---

## 0. Environment

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
```

Two datasets need attention before anything downloads:

```bash
# XNLI is ungated and needs nothing. FLORES-200 is GATED on the Hub: E7's retrieval half
# will not download without an authenticated account that has accepted its terms.
#   1. accept at https://huggingface.co/datasets/facebook/flores
#   2. then, once per machine:
hf auth login            # or: export HF_TOKEN=hf_...
```

> Swapping in an ungated FLORES mirror does **not** work. `Muennighoff/flores200` and
> `gsarti/flores_101` are loading-script datasets and `datasets>=3.0` refuses those
> outright; `openlanguagedata/flores_plus` is gated too. Verified 2026-07-30 on
> `datasets==4.8.4`. `load_flores_parallel` raises with these instructions rather than an
> opaque HF error. **E7's XNLI/patching half is unaffected** — only FLORES retrieval needs
> the token.

### Hardware notes (measured, not assumed)

The design targets a 16 GB RTX 4080 SUPER. This repository was validated on a **12 GB RTX
2060** (compute capability 7.5) with **15.9 GB of host RAM**.

- **VRAM.** bf16 works on Turing and measures ~4.5× faster than fp32 here (3.25 vs 0.73
  TFLOPS on a 2048³ matmul), so the configs' `precision: bfloat16` is the right setting
  rather than a concession. The heaviest single step measured was `screen_public_pairs.py`
  fingerprinting **Qwen2.5-1.5B in fp32: 11.9 GB of 12 GB**. That is the ceiling on this
  card — E7's `e7_qwen15_*.yaml` are bf16 for the same reason, and switching them to fp32
  would not fit.
- **Host RAM is the binding constraint for E6A, not VRAM.** Packing the TinyStories train
  split (2,119,719 stories → 457M tokens → **3.57M blocks** of 128) costs ~17–30 min of
  tokenisation once per `(split, seed)` and about **3 GB** of RSS. Before the memory fix in
  `load_tinystories_blocks` / `build_block_dataset` it peaked near **29 GB** and froze a
  16 GB machine; if you are on a checkout without that fix, expect the same. Since
  2026-08-02 the packed corpus is **cached to disk** (`--corpus-cache`, default
  `<repo>/.corpus_cache`), so this is paid once per `(dataset, split, block_size, seed,
  document window, tokenizer)` rather than once per run, per resume and per evaluation. The
  block digest is re-derived on every cache hit, so a corrupt entry raises instead of being
  served. Pass `--corpus-cache none` to force a rebuild.

### Measured E6A training cost

**Read this before scheduling anything: the old figures were ~80% CPU stall.** Until
2026-08-02 the loop rebuilt a 3.1–3.6M element Python shuffle once per micro-batch
(NEXT_STEPS §2.13, CLAUDE.md trap 27), so the "6.67 s/step" and "4.85 s/step" recorded here
and in the pilot were dominated by a defect, not by the GPU. The fix is byte-identical —
`train_log.jsonl`, `eval_log.jsonl` and checkpoint sha256 are unchanged across a 20-step real
CUDA run — so **runs started before it remain valid and poolable**, and one still in flight
can simply be resumed with `--resume auto` at the same `--max-steps`.

Measured after the fix, real gpt2 → random distilgpt2 at 3.1M blocks, batch 16 × accum 4,
40 steps, on the RTX 2060:

| | before | after |
|---|---|---|
| s/step | 6.96 | **1.96** |
| GPU utilisation (mean / median) | 29% / 29% | **96% / 99%** |
| peak VRAM | 6.50 GB | 6.50 GB |

1.96 s/step is the isolated GPU-side cost of the step, so the loop is now GPU-bound and
there is no further CPU stall to remove at this batch size. Scale by your card's throughput:
a 4090 should land near **0.4 s/step → ~1.1 h per 10,000-step run**, a 5090 lower. Read the
run's own `runtime_estimate.json` (written after step 100) rather than trusting any table.

**D0 is no cheaper than D2.** `need_attention` is hard-wired true — `L_ATTN` is logged as a
metric for every condition — so even the CE-only control pays the teacher forward and the
attention-JSD computation every step. That is deliberate (the metric is wanted for all
conditions), so there is no cheap arm. It is, however, now computed under `no_grad` wherever
its weight is zero, which is worth ~11% on D0/D1/G0/G1.

**Do not raise `per_device_batch_size` to "use the GPU better".** It is in the
pre-registered §1.6 `optim` block, peak VRAM is only 6.5 GB because the loop is compute-
rather than memory-bound, and at 96% utilisation there is nothing left to fill. Changing it
would also break floating-point comparability with the runs already spent.

---

## 1. Phase 0 — the gates that must be green before any real model runs

```bash
# 1a. the frozen-baseline rule, enforced mechanically (stdlib only, no venv needed)
python tests/test_frozen_files.py

# 1b. the full suite — must be 0 failed and 0 skipped
./.venv/Scripts/python.exe -m pytest tests/ -q

# 1c. all eight offline smoke programs, end to end on random tiny models
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nnsight_rest_smoke_tests.ps1
```

`run_nnsight_rest_smoke_tests.ps1` defaults to `.venv` and prints the interpreter it chose;
`-Python` overrides. It relaxes `$ErrorActionPreference` around the native call because
`transformers` writes progress bars to stderr and PowerShell 5.1 would otherwise turn each
line into a terminating `NativeCommandError` on a process that exited 0.

### 1d. The two bridge gates (need `nn_results/`)

```bash
./.venv/Scripts/python.exe -m pytest tests/test_corpus_e1_bridge.py \
    tests/test_fingerprint_runner_bridge.py -q
```

These prove the new E6/E7 instrument measures what the frozen E1–E5 code measures.
`nn_results/` is **untracked and ~2.5 GB**, so a fresh clone will see these *skip* rather
than fail — copy it across, or the load-bearing "same instrument" claim is unverified on the
machine producing the numbers.

---

## 2. Phase 1 — screening and the E6A pilot

### 2a. The cheapest evidence in the project (no training)

```bash
./.venv/Scripts/python.exe transformation_inheritance/screen_public_pairs.py --device cuda
```

Fingerprints gpt2/distilgpt2 and Qwen2.5 base/instruct at 0.5B and 1.5B, and asks whether a
publicly distilled or instruction-tuned pair already shares a fingerprint. No training at
all. Writes `results/screening/screening_fingerprints.csv`.

### 2b. Train the pilot

```bash
# the three §8.5 pilot conditions at seed 0
for c in e6a_ce e6a_logit_kd e6a_logit_attention_kd; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
done

# D0 at seeds 1 and 2 — REQUIRED by the framing-(B) calibration, see the note below
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_ce.yaml --seed $s --max-steps 2000
done
```

<details><summary>PowerShell</summary>

```powershell
foreach ($c in 'e6a_ce','e6a_logit_kd','e6a_logit_attention_kd') {
  & ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py `
    --config "transformation_inheritance/configs/$c.yaml" --seed 0 --max-steps 2000
  if ($LASTEXITCODE -ne 0) { throw "$c failed with $LASTEXITCODE" }
}
foreach ($s in 1,2) {
  & ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py `
    --config transformation_inheritance/configs/e6a_ce.yaml --seed $s --max-steps 2000
  if ($LASTEXITCODE -ne 0) { throw "seed $s failed with $LASTEXITCODE" }
}
```
</details>

> **Why D0 needs three seeds.** Decisions D2/D3/D4 took framing (B): `e6a_3`, `e6a_4` and
> `e6b_2` pre-register a *rule* — a threshold calibrated against the metric's own
> across-seed null spread — rather than a number. `resolve_threshold_rule` **refuses** a
> spread built from fewer than three seeds, because a spread from two points is a gap and
> would make the threshold an artefact of which two seeds ran. With only seed 0 those
> criteria report `met: null` and `go_no_go.json` stays `incomplete`, correctly.

### 2c. Fingerprint every checkpoint

```bash
for c in D0 D1 D2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a/$c/seed0 \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a/D0/seed$s \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done
```

<details><summary>PowerShell</summary>

```powershell
$runs = @('e6a/D0/seed0','e6a/D1/seed0','e6a/D2/seed0','e6a/D0/seed1','e6a/D0/seed2')
foreach ($r in $runs) {
  & ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py `
    --run-dir "transformation_inheritance/results/$r" `
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
  if ($LASTEXITCODE -ne 0) { throw "$r failed with $LASTEXITCODE" }
}
```
</details>

> **`--with-delta-ce` is not optional.** `functional_cosine_to_teacher` exists only when ΔCE
> is computed, and `e6a_3`'s first limb *and* `e6a_4`'s first limb both calibrate on it.
> Without the flag both criteria are permanently unevaluable.
>
> **Do not pass `--sink-blocks`.** The default 300 produces
> `tinystories_validation_sink_300`, which is the corpus id
> `e6_preregistration.yaml` names. A smaller value silently produces a different corpus id
> that no criterion selects.

### 2d. The seedless public reference (`08` §4)

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_ce.yaml \
  --evaluate-public-reference roneneldan/TinyStories-8M \
  --with-delta-ce --dtype bfloat16
```

Run **once**, not once per condition. Design §8.7 contrast 4 and criterion `e6a_4` select
D2's matched checkpoint against this run's `validation_ce`; without `--with-delta-ce` that
column is honestly empty and `e6a_c4a`/`e6a_c4b` report `no_data` rather than selecting
anything.

### 2e. Aggregate

```bash
./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```

Writes `results/aggregate/`: `e6a_contrasts.csv`, `e6a_matched_loss.csv`,
`table2_inheritance_components.csv`, `go_no_go.json`, `e6b_go_no_go.json`,
`aggregate_summary.json` and `figures/fig2..fig4`.

### 2f. The pilot gate

```bash
# criterion 4 needs a parity report; nothing else produces one for a student checkpoint
./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir transformation_inheritance/results/e6a/D0/seed0 --step 2000

./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --seed 0 --pilot-step 2000 \
  --parity-report transformation_inheritance/results/e6a/D0/seed0/parity/step_2000/parity_report.json
```

`run_pilot_parity.py` stages a copy of the checkpoint with the run's tokenizer beside it —
`save_pretrained` writes no tokenizer, and the frozen Neo harness loads model and tokenizer
from the same path — then calls the frozen `run_parity_check` with the design's **five**
examples and every operand built by the frozen Neo module's own functions. It exits 0 whether
parity passes or fails; the verdict is in the report, and exiting non-zero on a legitimate
`false` would make a failing parity indistinguishable from a broken invocation.

The frozen Neo CLI's own `--verify-parity` path would run three sentences and has no flag for
more, and criterion 4 registers five — so a report from the CLI leaves the criterion
`met: null` with a remedy instead of passing on a short sample (CLAUDE.md trap 25). Check
`n_sentences: 5` in the report; `n_rows` counts interventions, not examples.

Criteria 2 and 3 are scored on `tinystories_validation_sink_300` and say so in the artefact.
`--corpus` overrides the row selection for a smoke run that built a synthetic corpus; it is a
row selector, never a threshold.

**Do not launch Phase 2 without `proceed: true`.**

### 2g. The gpt2 arm (`e6a_gpt2`) — a second teacher, one that has a sink

The TinyStories teacher measures `baseline_sink` 0.009309 on the corpus its criteria are
scored on — 0.86× the uniform floor, against a 0.15 bar — so nothing there can be inherited
(CLAUDE.md trap 26). This arm distils `gpt2` (0.563006 on `e1_100x40`, 16.5× floor) into a
randomly-initialised distilgpt2 on OpenWebText. It is isolated by `experiment_id`, writes to
`results/e6a_gpt2/`, and reads `configs/e6a_gpt2_preregistration.yaml`.

```bash
# G0. THE PRE-FLIGHT, before any training: does gpt2 have a sink on 128-token OWT blocks?
#     Its 0.563 is on 40-token E1 windows, and that transfer is where TinyStories collapsed.
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference gpt2 --with-delta-ce --dtype bfloat16
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference distilbert/distilgpt2 --with-delta-ce --dtype bfloat16
# uniform floor at 128 tokens = 0.010770. Near it => stop and report (e6a_gpt2_stop_6);
# do not move the threshold.

# G1. train: G0/G1/G2 at seed 0, plus G0 at seeds 1 and 2 (framing (B) needs three)
for c in e6a_gpt2_ce e6a_gpt2_logit_kd e6a_gpt2_logit_attention_kd; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
done
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_gpt2_ce.yaml --seed $s --max-steps 2000
done

# G2. evaluate
for c in G0 G1 G2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a_gpt2/$c/seed0 \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a_gpt2/G0/seed$s \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done

# G3. aggregate + parity + gate, all pointed at the arm
./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment e6a_gpt2 \
  --results transformation_inheritance/results/e6a_gpt2 \
  --preregistration transformation_inheritance/configs/e6a_gpt2_preregistration.yaml \
  --reference-condition G0
./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir transformation_inheritance/results/e6a_gpt2/G0/seed0 --step 2000
./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --experiment e6a_gpt2 --seed 0 --pilot-step 2000 \
  --parity-report transformation_inheritance/results/e6a_gpt2/G0/seed0/parity/step_2000/parity_report.json
```

This arm's gate has **five** criteria: the four §8.5 ones plus `e6a_pilot_2b`, which asks
whether a condition reached 50% of the *teacher's own* sink. Both must be met — the absolute
bar is kept at the design's 0.15 precisely because this arm exists after that number failed
elsewhere. `run_pilot_parity.py` picks the frozen GPT-2 harness automatically, from the
checkpoint's `config.json`.

Cost: **6.50 GB VRAM measured** at batch 16 × 128 × accum 4, and — after the trap-27 fix —
GPU-bound at 1.96 s/step on an RTX 2060, so a 3090-class card should be well under an hour
per 2,000-step run rather than the 7–11 h estimated here before the CPU stall was found.
Read `runtime_estimate.json` after step 100 of G0 seed 0 before launching the rest.

### 2h. Canonical GPT-2-medium -> GPT-2-small arm and equal-head method bridge

This is the paper-facing model-scale extension. It is isolated from `e6a` and `e6a_gpt2`:

| experiment | conditions | purpose |
|---|---|---|
| `e6a_gpt2_medium_small` | G0/G1/G2-aligned | primary canonical scale arm; AMAD-style 16-to-12 alignment followed by E6A JSD |
| `e6a_gpt2_alignment_bridge` | G0/G1/G2-legacy/G2-aligned | fresh equal-head control for the change in attention loss |

Run the offline proof first:

```bash
./.venv/Scripts/python.exe -m pytest \
  tests/test_distillation_loss.py \
  tests/test_e6a_gpt2_medium_small_configs.py \
  tests/test_pilot_gate.py -q
./.venv/Scripts/python.exe tests/nnsight_e6a_gpt2_medium_small_smoke.py
```

Then follow [`RUNBOOK/03_E6A_GPT2_MEDIUM_SMALL.md`](RUNBOOK/03_E6A_GPT2_MEDIUM_SMALL.md)
verbatim. It contains the complete primary and bridge preflights, seed-0-first pilots,
five-example parity gates, all-seed full runs, public-reference evaluations, and isolated
aggregation commands. The preflight command for the primary arm begins:

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_medium_small_ce.yaml \
  --evaluate-public-reference --reference-condition PG2S \
  --reference-out transformation_inheritance/results/e6a_gpt2_medium_small/PG2S/reference \
  --with-delta-ce --dtype bfloat16 --device cuda
```

Read the cached GPT-2-medium teacher record on
`openwebtext_validation_sink_300` before training. A value at or below 0.15 is registered
stop condition `e6a_medium_small_stop_6`; report it and do not move the threshold. Real GPT-2-medium
runtime, peak VRAM, and checkpoint storage are unmeasured here, so run G0 seed 0 first and
inspect `runtime_estimate.json` after step 100.

---

## 3. Phase 2 — E6A in full

```bash
for c in e6a_ce e6a_logit_kd e6a_logit_attention_kd; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/$c.yaml --seed $s
  done
done

for c in D0 D1 D2; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6a/$c/seed$s \
      --with-delta-ce --dtype bfloat16
  done
done

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```

<details><summary>PowerShell</summary>

```powershell
foreach ($c in 'e6a_ce','e6a_logit_kd','e6a_logit_attention_kd') {
  foreach ($s in 0,1,2) {
    & ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py `
      --config "transformation_inheritance/configs/$c.yaml" --seed $s
    if ($LASTEXITCODE -ne 0) { throw "$c seed $s failed with $LASTEXITCODE" }
  }
}
foreach ($c in 'D0','D1','D2') {
  foreach ($s in 0,1,2) {
    & ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py `
      --run-dir "transformation_inheritance/results/e6a/$c/seed$s" `
      --with-delta-ce --dtype bfloat16
    if ($LASTEXITCODE -ne 0) { throw "$c seed $s failed with $LASTEXITCODE" }
  }
}
& ./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```
</details>

10,000 steps per run × 9 runs. The design's estimate is 4–9 h per run on a 4080 SUPER, and
the pre-trap-27 measurements agreed with it for the wrong reason — most of that time was the
per-micro-batch shuffle. Expect substantially less now, and measure yours from
`runtime_estimate.json`, written from the first completed steps of every run, before
committing. The nine runs share one packed corpus per seed, so the first run of each seed
pays the pack and the rest hit `--corpus-cache`.

---

## 4. E6B — sentiment adaptation (scientifically conditional)

```bash
for c in f1 f2 f3 f4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_sentiment_adaptation.py \
      --config transformation_inheritance/configs/e6b_$c.yaml --seed $s
  done
done

# --base is what fills the three *_drift_from_base columns; design §9.8's early-warning
# analysis is nothing without them. Absent it they stay EMPTY (never 0.0), with a reason.
for c in F1 F2 F3 F4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6b/$c/seed$s \
      --base distilbert/distilgpt2 --dtype bfloat16
  done
done

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```

<details><summary>PowerShell</summary>

```powershell
foreach ($c in 'f1','f2','f3','f4') {
  foreach ($s in 0,1,2) {
    & ./.venv/Scripts/python.exe transformation_inheritance/train_sentiment_adaptation.py `
      --config "transformation_inheritance/configs/e6b_$c.yaml" --seed $s
    if ($LASTEXITCODE -ne 0) { throw "e6b_$c seed $s failed with $LASTEXITCODE" }
  }
}
foreach ($c in 'F1','F2','F3','F4') {
  foreach ($s in 0,1,2) {
    & ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py `
      --run-dir "transformation_inheritance/results/e6b/$c/seed$s" `
      --base distilbert/distilgpt2 --dtype bfloat16
    if ($LASTEXITCODE -ne 0) { throw "$c seed $s failed with $LASTEXITCODE" }
  }
}
& ./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```
</details>

- `merge_parity.json` must say `passed: true` at every checkpoint. At float64 round-off is
  ~1e-13, so a failure means a systematic merge fault — check the `alpha/r` scale and the
  `Conv1D` transpose, **not** the tolerance.
- Only `step_<n>/merged/` is fingerprinted; `fingerprint_dir()` resolves it from the weights
  on disk, so you pass nothing.
- `e6b_2` calibrates on F1's across-seed spread, so **F1 needs all three seeds** before the
  E6B conjunction can resolve.

---

## 5. E7 — cross-lingual semantics

```bash
# 0. manifests (downloads; FLORES needs the token from §0)
./.venv/Scripts/python.exe crosslingual_semantics/prepare_flores_manifest.py \
  --tokenizer Qwen/Qwen2.5-0.5B --n 300 --seed 42
./.venv/Scripts/python.exe crosslingual_semantics/prepare_xnli_manifest.py \
  --tokenizer Qwen/Qwen2.5-0.5B --n 600 --seed 42
```

**Check the printed `join_strategy` on the XNLI run.** If it says
`all_languages_structural`, the shards carried no `promptID` and alignment came from the
structurally aligned config — record that in the paper's data section rather than claiming a
promptID join. It never joins by position.

Then, per model config:

```bash
CFG=crosslingual_semantics/configs/e7_qwen05_base.yaml

./.venv/Scripts/python.exe crosslingual_semantics/extract_sink_representations.py \
  --config $CFG --langs all

./.venv/Scripts/python.exe crosslingual_semantics/evaluate_crosslingual_retrieval.py \
  --reps crosslingual_semantics/results/reps/qwen05_base \
  --manifest crosslingual_semantics/results/manifests/flores_devtest.json

for v in matched unmatched length_matched; do
  ./.venv/Scripts/python.exe crosslingual_semantics/run_multilingual_fingerprint.py \
    --config $CFG --variant $v
done

# the §5.5 smoke FIRST — it writes runtime_estimate.json; read it before the full screen
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage smoke
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage screen
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage window
```

<details><summary>PowerShell — all four configs</summary>

```powershell
$tags = @{ 'e7_qwen05_base'='qwen05_base'; 'e7_qwen05_instruct'='qwen05_instruct';
           'e7_qwen15_base'='qwen15_base'; 'e7_qwen15_instruct'='qwen15_instruct' }
foreach ($name in $tags.Keys) {
  $cfg = "crosslingual_semantics/configs/$name.yaml"
  & ./.venv/Scripts/python.exe crosslingual_semantics/extract_sink_representations.py `
    --config $cfg --langs all
  if ($LASTEXITCODE -ne 0) { throw "extract $name failed" }

  & ./.venv/Scripts/python.exe crosslingual_semantics/evaluate_crosslingual_retrieval.py `
    --reps "crosslingual_semantics/results/reps/$($tags[$name])" `
    --manifest crosslingual_semantics/results/manifests/flores_devtest.json
  if ($LASTEXITCODE -ne 0) { throw "retrieval $name failed" }

  foreach ($v in 'matched','unmatched','length_matched') {
    & ./.venv/Scripts/python.exe crosslingual_semantics/run_multilingual_fingerprint.py `
      --config $cfg --variant $v
    if ($LASTEXITCODE -ne 0) { throw "fingerprint $name/$v failed" }
  }
  foreach ($stage in 'smoke','screen','window') {
    & ./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py `
      --config $cfg --stage $stage
    # NON-ZERO HERE MEANS AN IDENTITY CONTROL FAILED. Stop and read invalid_units.json;
    # do not continue to the next stage and do not aggregate around it.
    if ($LASTEXITCODE -ne 0) { throw "patching $name/$stage exited $LASTEXITCODE" }
    if ($stage -eq 'smoke') { Write-Host "read runtime_estimate.json before continuing" }
  }
}
```
</details>

Repeat for `e7_qwen05_instruct.yaml`, `e7_qwen15_base.yaml`, `e7_qwen15_instruct.yaml` —
the §6.4 claim gate needs **two model sizes** and contrast `e7_c5` needs **both variants**,
or they report `no_data`. Then once, over everything:

```bash
./.venv/Scripts/python.exe crosslingual_semantics/aggregate_crosslingual.py \
  --retrieval crosslingual_semantics/results/retrieval
```

- **A non-zero exit from `run_cross_language_patching.py` means an identity control
  failed.** That is not a warning: `invalid_units.json` names the run units, the aggregator
  drops their rows, and the affected sites must be re-measured. Never aggregate around one.
- `--stage window` refuses to run without `layer_selection.json`, and asserts the dev/test
  partitions are disjoint at startup. That refusal is the audit trail for "selected on dev,
  reported on test" — do not work around it.
- **0.5B runs fp32 and 1.5B runs bf16** as shipped. `04` §7 warns these are not comparable
  in absolute intervention percentage — only the *direction* of the parallel-versus-control
  contrast replicates across them. Say so in the paper, or re-run 1.5B in fp32.

---

## 6. Reading the artefacts before quoting anything

| Check | Where | What a bad value means |
|---|---|---|
| `decision` | `go_no_go.json`, `e7_go_no_go.json` | `incomplete` ⇒ something is unresolved. Read `pending_decisions_blocking` (should be empty in v4) and `n_unknown` (a calibration that could not resolve). |
| realised thresholds | `go_no_go.json` → `criteria[].threshold_rule` | The rule is the pre-registration; the number beside it is a **result**. Quote both. |
| exclusions | `aggregate_summary.json` → `exclusions` | Any contrast above the 2% failure rate is written `status="excluded"` with its reason, not computed. |
| failed units | `evaluation_summary.json` → `n_failed` | `n_failed > 0` with `n_written == 0` means every unit failed and the exit code is non-zero. |
| identity controls | `invalid_units.json` | Non-empty ⇒ re-measure those sites. |
| merge parity | `merge_parity.json` → `passed` | Must be `true` at every E6B checkpoint. |
| interpretation matrix | `interpretation_matrix.json` | `matched_row: null` is legitimate — the six rows are disjoint but not exhaustive (`04` §6.3). |

**Do not put results back into the pre-registrations.** Thresholds and criteria live only in
`transformation_inheritance/configs/e6_preregistration.yaml` and
`crosslingual_semantics/configs/e7_preregistration.yaml`. Changing one after seeing results
requires an `amendments:` entry with a date and a reason, and stops being a pre-registration.
