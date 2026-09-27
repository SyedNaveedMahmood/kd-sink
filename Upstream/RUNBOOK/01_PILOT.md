# RUNBOOK 01 — the pilot

Everything up to and including design §8.5's pilot gate. Finish this before starting
`02_FULL_EXPERIMENTS.md`; §8.5 forbids launching Phase 2 without `proceed: true`.

**Total: ~21 hours**, most of it the five training runs. Every step is resumable.

---

## How to read the timings

| tag | means |
|---|---|
| **measured** | timed on the validation box — 12 GB RTX 2060, 15.9 GB RAM, torch 2.10+cu128 |
| *derived* | arithmetic from a measured rate, shown inline so you can re-scale it |
| **unmeasured** | never run at full scale here. Read the artefact named in the row; do not trust a guess |

Scale for your own card. A 4080 SUPER should be roughly 2–4× faster than these figures.

**Shell.** Loops are bash (Git Bash). PowerShell equivalents are in `COMMANDS.md`. In
PowerShell take the verdict from `$LASTEXITCODE`, never from whether text appeared on
stderr — `transformers` writes progress bars there and PowerShell 5.1 reports them as
errors on a process that exited 0.

> **Running this on bwUniCluster 3.0?** [`cluster/bwunicluster3/`](../cluster/bwunicluster3/)
> has every phase below as a Slurm job, with the five training runs as a job array
> chained by `afterok`. Same commands and flags; the cluster adds workspace storage, an
> offline asset cache and the scheduler. One GPU per run means the ~20 h below becomes
> **1.3–2.2 h of wall-clock**. Start at
> [`cluster/bwunicluster3/README.md`](../cluster/bwunicluster3/README.md).

---

## Phase 0 — gates (~10 min, no GPU)

Nothing below this line should run until all four are green.

| # | command | time |
|---|---|---|
| 0.1 | `python -m venv .venv` then `./.venv/Scripts/python.exe -m pip install -r requirements.txt` | ~3 min (first time only) |
| 0.2 | `python tests/test_frozen_files.py` | **<1 s** |
| 0.3 | `./.venv/Scripts/python.exe -m pytest tests/ -q` | **9 min** (588 passed, 0 skipped) |
| 0.4 | `powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nnsight_rest_smoke_tests.ps1` | **~11 min** (9 programs; E7-PIPE alone is ~6 min of hierarchical bootstrap) |

```bash
python tests/test_frozen_files.py
./.venv/Scripts/python.exe -m pytest tests/ -q
powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_nnsight_rest_smoke_tests.ps1
```

**0.2 must print `OK: all 12 frozen files match`.** It hashes EOL-canonical bytes, so it
holds on a Linux checkout too. 0.3 must be **0 failed and 0 skipped** — a skip means
`nn_results/` is missing and the two E1 bridge gates silently did not run.

---

## Phase 1 — screening (~30–60 min, GPU)

No training. `07` P6 calls this the cheapest evidence in the project.

```bash
./.venv/Scripts/python.exe transformation_inheritance/screen_public_pairs.py --device cuda
```

**28 min of compute (measured)** plus first-run model downloads — gpt2, distilgpt2 and four
Qwen2.5 checkpoints, **~4.4 GB of HF cache**, so budget 30–60 min the first time.

Per-model fingerprint cost on 300 items, measured, useful as an anchor elsewhere:

| model | layers | time |
|---|---|---|
| distilgpt2 | 6 | 44 s |
| gpt2 | 12 | 76 s |
| Qwen2.5-0.5B (base / instruct) | 24 | 139 s / 138 s |
| Qwen2.5-1.5B base | 28 | 276 s |
| Qwen2.5-1.5B instruct | 28 | **982 s** |

> That last row is not a typo and not a property of the model. Qwen2.5-1.5B in **fp32 fills
> 11.9 GB of this card's 12 GB**, so the second 1.5B model thrashes — 3.5× the base model's
> time for identical work. On a 16 GB card expect it to look like the 276 s row. It is why
> E7's `e7_qwen15_*.yaml` are bf16.

Check `results/screening/screening_summary.json` → `n_failed` should be `0`.
`qwen3_base_instruct` is recorded as *not run* — it is conditional in design §9.9, not a
failure.

---

## Phase 2 — train the pilot (~20 h, GPU)

Five runs: the three §8.5 conditions at seed 0, **plus D0 at seeds 1 and 2**.

> **Why five and not three.** Decisions D2/D3 took framing (B): `e6a_3` and `e6a_4`
> pre-register a *rule* — a threshold calibrated against D0's own across-seed spread — not a
> number. `resolve_threshold_rule` **refuses** a spread from fewer than three seeds, because
> a spread from two points is a gap. With seed 0 alone both criteria report `met: null` and
> `go_no_go.json` reads `incomplete`, correctly. Drop the two extra runs only if you accept
> that outcome.

```bash
# the three pilot conditions at seed 0
for c in e6a_ce e6a_logit_kd e6a_logit_attention_kd; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
done

# D0 at seeds 1 and 2 — required by the framing-(B) calibration
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_ce.yaml --seed $s --max-steps 2000
done
```

Per run, and it is the same for every condition:

| stage | time | note |
|---|---|---|
| pack the TinyStories corpus | **20 min (measured)** | 2,119,719 stories → 3.57M blocks; ~3 GB RSS; rebuilt every run |
| write `block_manifest.parquet` | included above | **1.1 GB on disk, per run** |
| step-0 checkpoint + evaluation | **~10 s (measured)** | saved *before* the first optimiser update |
| 2,000 training steps | *3.7 h* | at **6.67 s/step measured** (27 steps in 180 s) |
| **per run** | **≈ 4 h** | |
| **five runs** | **≈ 20 h** | |

> **There is no cheap arm.** `need_attention` is hard-wired true because `L_ATTN` is logged
> as a metric for *every* condition, so the CE-only D0 control pays the teacher forward and
> the attention-JSD computation exactly like D2. Budget all five equally.

**Disk:** ~1.2 GB manifest + 8 checkpoints × 79 MB ≈ **1.8 GB per run**, so ~9 GB for the
pilot.

**Interrupted?** Just re-run the same command — resume is bit-exact. But **do not** change
`--max-steps` on a resume; `assert_resumable_schedule` will refuse and tell you why.

---

## Phase 3 — evaluate (**unmeasured**, GPU)

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

**Never run at this scale here.** The anchor: a 300-item fingerprint of a 6-layer
distilgpt2 took 44 s, and the E6A student (8M, 8 layers) and teacher (33M, 4 layers) are
both smaller. Each run evaluates 5 checkpoints × 3 corpora against the teacher, plus ΔCE
over 300 blocks × 7 interventions — so **expect tens of minutes per run, not hours**, but
read `evaluation_summary.json` after the first checkpoint for the real rate rather than
trusting that.

Two flags that are not optional:

- **`--with-delta-ce`.** `functional_cosine_to_teacher` exists only when ΔCE is computed,
  and *both* `e6a_3`'s first limb and `e6a_4`'s first limb calibrate on it. Without the flag
  those criteria are permanently unevaluable.
- **Do not pass `--sink-blocks`.** The default 300 produces
  `tinystories_validation_sink_300`, which is the corpus id the pre-registration names. A
  different value produces a different corpus id that no criterion selects, and everything
  silently reports `no_data`.

Resumable at `(run_id, step, corpus_id)` — re-running recomputes nothing.

---

## Phase 4 — the public reference (**unmeasured**, GPU)

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_ce.yaml \
  --evaluate-public-reference roneneldan/TinyStories-8M \
  --with-delta-ce --dtype bfloat16
```

Run **once**, not once per condition. Design §8.7 contrast 4 and criterion `e6a_4` select
D2's matched checkpoint against this run's `validation_ce`; without `--with-delta-ce` that
column is honestly empty and `e6a_c4a`/`e6a_c4b` report `no_data`.

Same order as one checkpoint's evaluation — minutes.

---

## Phase 5 — aggregate, parity, gate (~10–20 min)

```bash
# 5a. contrasts, matched loss, figures, go_no_go.json  (also writes e6b_go_no_go.json)
./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py

# 5b. the parity report criterion 4 reads — nothing else produces one for a checkpoint
./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir transformation_inheritance/results/e6a/D0/seed0 --step 2000

# 5c. the gate
./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --seed 0 --pilot-step 2000 \
  --parity-report transformation_inheritance/results/e6a/D0/seed0/parity/step_2000/parity_report.json
```

| step | time | note |
|---|---|---|
| 5a aggregate | **unmeasured**; the E7 aggregator's 10,000-replicate bootstrap takes ~6 min in smoke, so allow ~10 min | pure CPU |
| 5b parity | **unmeasured**; 5 sentences × 10 interventions on an 8M model — expect a few minutes | stages a copy of the checkpoint with its tokenizer, then calls the frozen `run_parity_check` in-process |
| 5c gate | **<5 s** | reads artefacts only |

`run_pilot_parity.py` exits 0 whether parity passes or fails — the verdict is in the report.
Exiting non-zero on a legitimate `false` would make a failing parity indistinguishable from
a broken invocation.

**Check `n_sentences: 5` in the report.** Criterion 4 registers five examples; the frozen Neo
CLI's own `--verify-parity` path runs the three-sentence `PARITY_SENTENCES` default and has no
flag for more, so a report produced that way leaves criterion 4 `met: null` with a remedy
rather than passing on a short sample (CLAUDE.md trap 25). That is why this step runs the
frozen `run_parity_check` in-process instead of shelling out to the CLI.

---

## What "done" looks like

Read these before quoting anything:

| artefact | expect |
|---|---|
| `pilot_gate.json` | four criteria, none `met: null`, and `proceed: true` |
| `pilot_gate.json` → `criteria[e6a_pilot_2].observed` | three readings per condition (`at_pilot_step`, `max_through_pilot_step`, `max_including_step_0`) on `corpus_id: tinystories_validation_sink_300`. **Identical values across D0/D1/D2 mean the step-0 row is being read** — the three share an initialisation (trap 23) |
| `pilot_gate.json` → `criteria[e6a_pilot_4].observed.n_sentences` | `5`. `n_rows` is interventions, not examples |
| `run_config.json` → `attn_reduction` | `sum_over_keys_mean_over_queries`. Absent ⇒ the run predates the trap-22 fix and its D2 arm trained on a 64.5× under-scaled `L_ATTN`; do not pool it with a corrected run |
| `go_no_go.json` → `pending_decisions_blocking` | `[]` — all seven decisions are made |
| `go_no_go.json` → `criteria[].threshold_rule` | a realised `spread`, `null_seed_values` and derived `threshold` for `e6a_3`/`e6a_4`. **Quote the rule and the number together**: the rule is the pre-registration, the number is a result |
| `aggregate_summary.json` → `exclusions` | anything above the 2 % failure rate is `status="excluded"` with a reason, not computed |
| `evaluation_summary.json` → `n_failed` | `n_failed > 0` with `n_written == 0` means every unit failed |

**`incomplete` is a legitimate verdict, not a bug.** If D0 has fewer than three usable
seeds, `e6a_3`/`e6a_4` report `met: null` with the calibration's own reason and the decision
is `incomplete`. That is the registered rule working. Check `n_unknown` and each criterion's
`threshold_rule.reason` — not `pending_decisions_blocking`, which should be empty.

**Do not start `02_FULL_EXPERIMENTS.md` without `proceed: true`.**

---

# The gpt2 arm (`e6a_gpt2`)

A second distillation arm with a teacher that has a sink. Everything above stays as it is;
this runs beside it, writes to `results/e6a_gpt2/`, and reads its own pre-registration.

**Why:** the TinyStories teacher measures `baseline_sink` **0.009309** on the corpus its
criteria are scored on — 0.86× the uniform-attention floor, zero cells above 0.2, against a
0.15 bar. Nothing there could be inherited. gpt2 measures **0.563006** on `e1_100x40`
(16.5× floor, 73% of cells above 0.2) and distilgpt2 0.463267, so this pair has a real sink
at both ends (`results/screening/screening_fingerprints.csv`, CLAUDE.md trap 26).

Phase 0 is shared — the same gates, now **588 passed** and **9** smoke programs.

## Phase G0 — the pre-flight (~40 min, GPU, **no training**)

**Do this before anything else.** It is the step the TinyStories arm did not have, and it
decides whether the next 30+ hours are worth spending.

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference gpt2 --with-delta-ce --dtype bfloat16
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference distilbert/distilgpt2 --with-delta-ce --dtype bfloat16
```

Read `baseline_sink` and `frac_cells_above_0_2` for **gpt2 on
`openwebtext_validation_sink_300`** in the written `checkpoint_metrics.csv`. The uniform
floor at 128 tokens is **0.010770**, so ×floor = sink ÷ 0.010770.

| reading | do |
|---|---|
| **≥ 0.15, cells above 0.2** | proceed — criterion 2 is reachable and 2b is meaningful |
| **near the floor** | **stop and report.** Registered as `e6a_gpt2_stop_6`. Same failure as TinyStories; more steps cannot fix it and the threshold must not move |

This also downloads and packs the OpenWebText prefix for the first time (~1.9 GB streamed).

## Phases G1–G5 — the pilot

Identical in shape to Phases 2–5 above. Five runs: G0/G1/G2 at seed 0, plus **G0 at seeds 1
and 2** (framing (B) refuses a null spread from fewer than three).

```bash
for c in e6a_gpt2_ce e6a_gpt2_logit_kd e6a_gpt2_logit_attention_kd; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
done
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_gpt2_ce.yaml --seed $s --max-steps 2000
done

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

| stage | cost | note |
|---|---|---|
| pack OpenWebText | **unmeasured**; ~1.9 GB streamed, expect well under TinyStories' 20 min | streaming does not populate the HF cache, so it is re-fetched per run |
| 2,000 steps | **unmeasured**; *7–11 h* estimated on a 3090 at 12–20 s/step | scaled from the 4080 SUPER's measured 4.85 s/step by ~4.2× student FLOPs |
| VRAM | *~7–9 GB* estimated | the 50257-vocab KD softmax dominates and is unchanged from the measured 4.68 GB |
| five runs | *33–55 h* | |

**Read `runtime_estimate.json` after step 100 of G0 seed 0 before launching the rest** — it
carries the real `units_per_second` and `peak_vram_bytes`. If the budget is tight, run seed 0
only and accept `incomplete`; G0 seeds 1–2 can follow later.

**Do not shrink `train_documents` for the pilot.** The packing shuffle runs over
`range(len(documents))`, so a smaller cap is a *different block sequence* and `--max-steps
2000` would stop being a truncation of the 10,000-step run.

### What "done" looks like, for this arm

| artefact | expect |
|---|---|
| `results/e6a_gpt2/pilot_gate.json` | **five** criteria — this arm registers `e6a_pilot_2b` |
| → `criteria[e6a_pilot_2b].observed.teacher_sink` | gpt2's own sink, read from its per-corpus fingerprint record. `met: null` here means the teacher was never fingerprinted, or its record predates the trap-24 per-corpus layout |
| → `corpus_id` | `openwebtext_validation_sink_300`, never the TinyStories one |
| `aggregate_summary.json` → `experiment` | `e6a_gpt2`, and `preregistration_version: e6a_gpt2_prereg_v1` |
| `run_config.json` → `arch_key` | `gpt2`. `neo` would mean the student is being fingerprinted through the wrong harness |
| the TinyStories arm | unchanged. `aggregate_transformation.py` with no `--experiment` must still see zero gpt2 rows |

---

# Canonical model-scale extension (`e6a_gpt2_medium_small`)

The paper-facing scale arm is now the standard **GPT-2-medium teacher -> GPT-2-small random
student** pair. G0/G1 keep their original optimization objectives; the separately named
`G2-aligned` uses a preregistered **AMAD-style JSD extension** for the unavoidable 16-to-12
head mismatch. It is not described as exact AMAD.

A fresh equal-head bridge, `e6a_gpt2_alignment_bridge`, reruns GPT-2 -> DistilGPT-2 with
G0/G1, `G2-legacy`, and `G2-aligned`. The paired aligned-minus-legacy contrast separates an
attention-loss-method effect from the model-scale comparison. Existing `e6a_gpt2` artifacts
are never reused or relabeled.

The exact preflight, pilot, full-run, evaluation, parity, gate, and aggregation commands are
in [`03_E6A_GPT2_MEDIUM_SMALL.md`](03_E6A_GPT2_MEDIUM_SMALL.md). Its two critical rules are:

1. fingerprint each pinned teacher on `openwebtext_validation_sink_300` before student
   training and stop if it misses the unchanged 0.15 bar; and
2. use a fresh output root for every 10,000-step run, never a 2,000-step pilot directory.

The primary unequal-head arm must report `carrier_jaccard_to_teacher` as null with a warning.
Truncating or inventing a 16-to-12 carrier correspondence is not an allowed workaround.
