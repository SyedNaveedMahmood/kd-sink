# RUNBOOK 02 — the paper-ready experiments

Everything after the pilot gate. **Do not start until `pilot_gate.json` says
`proceed: true`** (design §8.5), and until at least one of E6A/E7 clears its §18 criteria at
the Phase 1 decision point.

For the canonical GPT-2-medium -> GPT-2-small primary arm and its equal-head alignment
bridge, use [`03_E6A_GPT2_MEDIUM_SMALL.md`](03_E6A_GPT2_MEDIUM_SMALL.md), sections 3 and 5.
Those experiments have separate preregistrations and fresh full-run roots; do not substitute
their configs into the legacy Stage A loop below.

**Total on the validation box: ~9–11 GPU-days.** E6A dominates. Timings are tagged
**measured** / *derived* / **unmeasured** exactly as in `01_PILOT.md`; scale for your card,
where a 4080 SUPER should be ~2–4× faster.

---

## Before anything: two prerequisites

### P1 — FLORES needs a credential (E7 only, one-off)

`facebook/flores` is **gated**. Verified 2026-07-30 on `datasets==4.8.4`:
`load_dataset("facebook/flores", "eng_Latn")` raises
`DatasetNotFoundError: ... is a gated dataset on the Hub`.

```bash
# 1. accept the terms at https://huggingface.co/datasets/facebook/flores
# 2. then, once per machine:
hf auth login          # or set HF_TOKEN in the environment
```

> **Do not go looking for a mirror.** `Muennighoff/flores200` and `gsarti/flores_101` are
> loading-script datasets and `datasets>=3.0` refuses those outright;
> `openlanguagedata/flores_plus` is gated too. All three were tried.
> `load_flores_parallel` raises with these instructions rather than an opaque HF error.
> **XNLI is not gated**, so E7's entire patching half is unaffected.

### P2 — budget disk

| item | size | source |
|---|---|---|
| E6A run directory | **1.8 GB** each | 1.1 GB `block_manifest.parquet` + 8 × 79 MB checkpoints (**measured**) |
| E6A total (9 runs) | *~16 GB* | |
| HF cache | **4.4 GB** and up (**measured**) | grows with the Qwen and SST-2 downloads |
| E7 representations | **unmeasured** | 300 sentences × 8 languages × every layer × 10 objects, per model config, streamed to `.npy` memmaps |

---

## Stage A — E6A Phase 2 (~7 GPU-days)

3 conditions × 3 seeds × 10,000 steps.

> **Use a FRESH run directory, or `--resume none`.** Resuming the 2,000-step pilot into a
> 10,000-step run splices two different cosine schedules: `LambdaLR` does not serialise its
> lambda, so the restored scheduler keeps the pilot's step count while following the new
> cosine, and the learning rate jumps at the join. The result is neither pre-registered
> schedule (§1.6). `assert_resumable_schedule` refuses it and names the fix.

```bash
for c in e6a_ce e6a_logit_kd e6a_logit_attention_kd; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/$c.yaml --seed $s
  done
done
```

| unit | time | basis |
|---|---|---|
| corpus pack, per run | **20 min** | measured; rebuilt every run, ~3 GB RSS |
| 10,000 steps | *18.5 h* | **6.67 s/step measured** |
| one run | *~18.9 h* | |
| **all nine** | ***~7.1 GPU-days*** | |

Then evaluate every checkpoint:

```bash
for c in D0 D1 D2; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6a/$c/seed$s \
      --with-delta-ce --dtype bfloat16
  done
done
```

**Unmeasured at scale.** Anchor: a 300-item fingerprint of 6-layer distilgpt2 took **44 s**,
and the E6A student (8M) and teacher (33M) are smaller. Nine runs × 8 checkpoints × 3
corpora, plus ΔCE. Read `evaluation_summary.json` after the first run for the real rate.
`--with-delta-ce` is **required** — `functional_cosine_to_teacher` exists only with it, and
`e6a_3`/`e6a_4` both calibrate on it. Never pass `--sink-blocks`.

And the seedless public reference, **once**:

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_ce.yaml \
  --evaluate-public-reference roneneldan/TinyStories-8M \
  --with-delta-ce --dtype bfloat16
```

---

## Stage B — E6B, sentiment adaptation (**unmeasured**)

Scientifically conditional (`00` §6 puts it outside the minimum publishable version), but
code-complete. 4 conditions × 3 seeds × 3 epochs on SST-2 (~67k rows), `distilgpt2`.

```bash
for c in f1 f2 f3 f4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_sentiment_adaptation.py \
      --config transformation_inheritance/configs/e6b_$c.yaml --seed $s
  done
done

# --base fills the three *_drift_from_base columns. Design §9.8's early-warning analysis is
# nothing without them; absent it they stay EMPTY (never 0.0), with a recorded reason.
for c in F1 F2 F3 F4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6b/$c/seed$s \
      --base distilbert/distilgpt2 --dtype bfloat16
  done
done
```

**Unmeasured** — only a 16-step verification run has executed here (see
`transformation_inheritance/_exploratory/`). It is small next to E6A: distilgpt2 is 82M and
SST-2 is ~67k short rows, with no 3.57M-block corpus build. Expect hours, not days, for all
twelve runs — but measure the first one.

Three things to watch:

- **`merge_parity.json` must say `passed: true` at every checkpoint.** At float64 round-off
  is ~1e-13 (measured 2.700e-13), so a failure means a systematic merge fault — check the
  `alpha/r` scale and the `Conv1D` transpose, **not** the tolerance.
- Only `step_<n>/merged/` is fingerprinted; `fingerprint_dir()` resolves it from the weights
  on disk, so you pass nothing.
- **`e6b_2` calibrates on F1's across-seed spread**, so F1 needs all three seeds before the
  E6B conjunction can resolve.

---

## Stage C — E7, cross-lingual semantics (**unmeasured**)

### C1 — manifests (one-off)

```bash
./.venv/Scripts/python.exe crosslingual_semantics/prepare_flores_manifest.py \
  --tokenizer Qwen/Qwen2.5-0.5B --n 300 --seed 42
./.venv/Scripts/python.exe crosslingual_semantics/prepare_xnli_manifest.py \
  --tokenizer Qwen/Qwen2.5-0.5B --n 600 --seed 42
```

Minutes plus downloads. **Check the printed `join_strategy` on the XNLI run.** If it says
`all_languages_structural`, the shards carried no `promptID` and alignment came from the
structurally aligned config — record *that* in the paper's data section rather than claiming
a promptID join. It never joins by position.

### C2 — per model config

Run this block four times: `e7_qwen05_base`, `e7_qwen05_instruct`, `e7_qwen15_base`,
`e7_qwen15_instruct`. The §6.4 claim gate needs **two model sizes** and contrast `e7_c5`
needs **both variants**, or they report `no_data`.

```bash
CFG=crosslingual_semantics/configs/e7_qwen05_base.yaml
TAG=qwen05_base

./.venv/Scripts/python.exe crosslingual_semantics/extract_sink_representations.py \
  --config $CFG --langs all

./.venv/Scripts/python.exe crosslingual_semantics/evaluate_crosslingual_retrieval.py \
  --reps crosslingual_semantics/results/reps/$TAG \
  --manifest crosslingual_semantics/results/manifests/flores_devtest.json

for v in matched unmatched length_matched; do
  ./.venv/Scripts/python.exe crosslingual_semantics/run_multilingual_fingerprint.py \
    --config $CFG --variant $v
done

# the §5.5 smoke FIRST — it writes runtime_estimate.json. READ IT before the full screen.
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage smoke
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage screen
./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
  --config $CFG --stage window
```

| stage | time | basis |
|---|---|---|
| extraction | **unmeasured** | 300 sentences × 8 languages × all layers × 10 objects |
| retrieval | **unmeasured** | the language probe is the expensive part; `--probe-layers` restricts it without touching the retrieval tables |
| fingerprint × 3 variants | *~3 × the per-model anchor* | Qwen2.5-0.5B fingerprinted 300 items in **139 s**; 1.5B in **276 s** |
| patch smoke (50 examples, 4 layers) | **unmeasured** | **writes `runtime_estimate.json` — this is the whole point of the stage** |
| screen (200 dev) → window (400 test) | **unmeasured** | project from the smoke's estimate |

> **The patching stages are the one genuinely unknown cost in the project.** That is exactly
> why `--stage smoke` exists and writes `runtime_estimate.json`. Read it and multiply before
> committing to `screen`, and again before `window`.

### C3 — aggregate, once, over everything

```bash
./.venv/Scripts/python.exe crosslingual_semantics/aggregate_crosslingual.py \
  --retrieval crosslingual_semantics/results/retrieval
```

**~10 min** — the hierarchical bootstrap is 10,000 replicates and took ~6 min on the smoke
fixture (**measured**).

### E7 rules that are not advice

- **A non-zero exit from `run_cross_language_patching.py` means an identity control
  failed.** `invalid_units.json` names the run units, the aggregator drops their rows, and
  those (language, object, layer) sites must be re-measured. **Never aggregate around one.**
- **`--stage window` refuses to run without `layer_selection.json`** and asserts the
  dev/test partitions are disjoint at startup. That refusal is the audit trail for "selected
  on dev, reported on test" — do not work around it.
- **0.5B runs fp32, 1.5B runs bf16** as shipped. `04` §7: only the *direction* of the
  parallel-versus-control contrast replicates across them, not absolute intervention
  percentages. Say so in the paper, or re-run 0.5B in bf16 for symmetry. Do **not** switch
  1.5B to fp32 on a 12 GB card — measured, that fills 11.9 GB and thrashes (982 s vs 276 s
  for identical work).

---

## Stage D — read the artefacts before quoting anything

| check | where | a bad value means |
|---|---|---|
| `decision` | `go_no_go.json`, `e7_go_no_go.json` | `incomplete` ⇒ something is unresolved. `pending_decisions_blocking` should be `[]`; if so, look at `n_unknown` and each `threshold_rule.reason` |
| realised thresholds | `go_no_go.json` → `criteria[].threshold_rule` | the rule is the pre-registration, the number beside it is a **result**. Quote both |
| exclusions | `aggregate_summary.json` → `exclusions` | any contrast above the 2 % failure rate is `status="excluded"` with a reason, not computed |
| failure rate | `aggregate_summary.json` → `max_failure_rate_observed` | `--allow-high-failure` is a deliberate override, never a default |
| identity controls | `invalid_units.json` | non-empty ⇒ re-measure those sites |
| merge parity | `merge_parity.json` → `passed` | must be `true` at every E6B checkpoint |
| interpretation matrix | `interpretation_matrix.json` | `matched_row: null` is legitimate — the six rows are disjoint but **not** exhaustive (`04` §6.3) |
| provenance | every artefact's tail | git sha, `manifest_sha256`, seed, dtype, device, registry version |

**Do not write results back into the pre-registrations.** Thresholds and criteria live only
in `transformation_inheritance/configs/e6_preregistration.yaml` and
`crosslingual_semantics/configs/e7_preregistration.yaml`. Changing one after seeing results
requires an `amendments:` entry with a date and reason, and stops being a pre-registration.

---

## Things the paper must state, from these runs

Collected here because each is a decision or a deviation that only exists in an artefact:

1. **`unrelated_transfer = 0.10` is judgement with no design anchor** (decision D7). Say so.
2. **§10.11 contrast 6 could not be pre-registered** for want of a defensible language
   tiering (decision D5, route C) — the Qwen2.5 report gives no per-language token share.
3. **`e6a_c3` is a distance**, so its sign runs opposite to `e6a_c1`/`e6a_c2` (decision D1).
4. **`e6a_3`/`e6a_4`/`e6b_2` thresholds are calibrated, not fixed** (D2/D3/D4) — report the
   rule and the realised number together.
5. **The XNLI join strategy actually used**, from the manifest's provenance.
6. **The merge-parity check runs in float64, not `03` §4.5's fp32**, because that spec's
   1e-5 bar is absolute and distilgpt2's logits reach ~104. The tolerance did not move; the
   precision did.
7. **0.5B fp32 versus 1.5B bf16** are not comparable in absolute terms (`04` §7).
