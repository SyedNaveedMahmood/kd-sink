# RUNBOOK 03 - GPT-2-medium -> GPT-2-small E6A extension

This is the operator runbook for the prospectively registered model-scale extension and
its equal-head method bridge. The implementation is smoke-verified; the real GPU pilot and
full runs have not been executed in this checkout.

The primary scientific arm is canonical GPT-2-medium -> canonical GPT-2-small. The student
is initialized randomly from the pinned GPT-2-small config; the public GPT-2-small weights
are a seedless reference and never initialize the student.

| arm | experiment id | conditions | teacher -> random student | public reference |
|---|---|---|---|---|
| primary | `e6a_gpt2_medium_small` | `G0`, `G1`, `G2-aligned` | GPT-2-medium (24L/16H/1024) -> GPT-2-small (12L/12H/768) | `PG2S`, GPT-2-small |
| method bridge | `e6a_gpt2_alignment_bridge` | `G0`, `G1`, `G2-legacy`, `G2-aligned` | GPT-2 (12L/12H/768) -> DistilGPT-2 (6L/12H/768) | `PDG2`, DistilGPT-2 |

The primary registration is
`transformation_inheritance/configs/e6a_gpt2_medium_small_preregistration.yaml`; the bridge
registration is
`transformation_inheritance/configs/e6a_gpt2_alignment_bridge_preregistration.yaml`.

## Frozen method definition

`G2-aligned` is an **AMAD-style JSD extension**, not exact AMAD:

1. At every registered layer pair, flatten each teacher and student attention map over the
   common valid cells and L2-normalize it for cosine similarity.
2. For each teacher head, softmax its cosine similarities over all student heads. Direction
   is teacher-to-student, temperature is exactly `1.0`, and the weights remain attached to
   autograd.
3. Form the convex weighted mixture of the raw student attention tensors. Do not add a
   renormalization step; this preserves E6A's existing dropout and masking semantics.
4. Apply E6A's original JSD-form divergence and
   `sum_over_keys_mean_over_queries` reduction. The loss weights remain
   `0.45 CE + 0.45 KD + 0.10 attention` at KD temperature 2.

The primary map is `{1:0, 3:1, ..., 23:11}` (zero-indexed): the last teacher layer in each
two-layer block maps to the corresponding student layer, including final layer 23 -> 11.

The bridge is the loss-method control. `G2-legacy` and `G2-aligned` differ only in condition
name and head-alignment method. Its G0/G1 conditions retain the legacy index-JSD diagnostic,
so they are fully faithful fresh controls. Do not reuse or relabel completed `e6a_gpt2`
artifacts as bridge rows.

## Non-negotiable execution rules

- Do not edit either preregistration after inspecting real results. Any necessary change is
  a dated amendment and a fresh experiment id.
- Do not change model ids or revision hashes. Startup checks enforce 24/16/1024 ->
  12/12/768 for the primary arm and 12/12/768 -> 6/12/768 for the bridge.
- Do not silently change batch size, accumulation, sequence length, optimizer, data window,
  layer map, temperature, or alignment temperature after an OOM. Stop and register the
  deviation first.
- A 2,000-step pilot and a 10,000-step run have different cosine schedules. Never resume a
  pilot into a full run. The commands below use separate output roots.
- The primary teacher has 16 heads and the student has 12. The evaluator must therefore
  write `carrier_jaccard_to_teacher = null` with an unequal-head warning. Do not truncate,
  duplicate, or align heads merely to populate that metric.
- Run seed 0 first. Inspect `runtime_estimate.json` after step 100 and disk usage after the
  first full optimizer checkpoint before scheduling the other runs. Real runtime, VRAM,
  and checkpoint storage are unmeasured for GPT-2-medium in this repository.

All loops below are bash/Git Bash and run from the repository root.

## 0. Offline verification

Run this after checkout and before using a GPU:

```bash
./.venv/Scripts/python.exe -m pytest \
  tests/test_distillation_loss.py \
  tests/test_distillation_init.py \
  tests/test_e6a_gpt2_configs.py \
  tests/test_e6a_gpt2_medium_small_configs.py \
  tests/test_public_reference_rows.py \
  tests/test_pilot_gate.py -q

./.venv/Scripts/python.exe tests/nnsight_e6a_gpt2_medium_small_smoke.py
```

The end-to-end smoke must report:

- identical seed-0 initial state hashes across primary G0/G1/G2-aligned;
- exact smoke depth/head geometry `24/16 -> 12/12`;
- identical initialization for bridge G2-legacy/G2-aligned;
- successful training, evaluation, aggregation, and gate dispatch;
- explicit null carrier overlap for the unequal-head primary pair; and
- zero leakage into the original `e6a` aggregation.

## 1. Primary teacher/reference preflight - no student training

This evaluates the pinned public GPT-2-small reference and, in the same instrument, caches
the pinned GPT-2-medium teacher fingerprint on OpenWebText.

```bash
MS_PILOT=transformation_inheritance/results
MS_EXP=e6a_gpt2_medium_small

./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_medium_small_ce.yaml \
  --evaluate-public-reference \
  --reference-condition PG2S \
  --reference-out "$MS_PILOT/$MS_EXP/PG2S/reference" \
  --with-delta-ce --dtype bfloat16 --device cuda

./.venv/Scripts/python.exe -c "import json,pathlib; root=pathlib.Path('transformation_inheritance/results/e6a_gpt2_medium_small/PG2S/reference/fingerprints'); p=next(root.glob('teacher_*/step_na/openwebtext_validation_sink_300/fingerprint.json')); r=json.loads(p.read_text(encoding='utf-8'))['record']; print({'path':str(p),'baseline_sink':r['baseline_sink'],'frac_cells_above_0_2':r['frac_cells_above_0_2']})"
```

Apply the registered stop rule before training:

- If GPT-2-medium clears `baseline_sink > 0.15` on
  `openwebtext_validation_sink_300`, continue. `frac_cells_above_0_2` is a diagnostic.
- If it is at or below 0.15, stop and report `e6a_medium_small_stop_6`. Do not lower the
  threshold or substitute the already-known 40-token E1 measurement.

Confirm that the public-reference CSV is at
`$MS_PILOT/$MS_EXP/PG2S/reference/checkpoint_metrics.csv`, with condition `PG2S`, seed `-1`,
checkpoint step `-1`, and all teacher/reference/tokenizer revision pins in
`reference_summary.json`.

## 2. Primary 2,000-step pilot

Run G0 seed 0 first. The output root is the pilot root, not the later full-run root.

```bash
./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
  --config transformation_inheritance/configs/e6a_gpt2_medium_small_ce.yaml \
  --seed 0 --max-steps 2000 --output-dir "$MS_PILOT"
```

Inspect:

```bash
./.venv/Scripts/python.exe -c "import json,pathlib; p=pathlib.Path('transformation_inheritance/results/e6a_gpt2_medium_small/G0/seed0/runtime_estimate.json'); print(json.dumps(json.loads(p.read_text()),indent=2))"
```

Only after checking measured throughput, peak VRAM, and disk capacity, run the remaining
seed-0 conditions and the two extra G0 seeds required for the null-spread rule:

```bash
for cfg in \
  e6a_gpt2_medium_small_logit_kd \
  e6a_gpt2_medium_small_logit_attention_kd_aligned; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$cfg.yaml \
    --seed 0 --max-steps 2000 --output-dir "$MS_PILOT"
done

for seed in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_gpt2_medium_small_ce.yaml \
    --seed $seed --max-steps 2000 --output-dir "$MS_PILOT"
done
```

Evaluate the registered checkpoints:

```bash
for condition in G0 G1 G2-aligned; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir "$MS_PILOT/$MS_EXP/$condition/seed0" \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16 --device cuda
done

for seed in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir "$MS_PILOT/$MS_EXP/G0/seed$seed" \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16 --device cuda
done
```

Aggregate, run five-example parity, and evaluate the arm-specific gate:

```bash
./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment "$MS_EXP" \
  --results "$MS_PILOT/$MS_EXP" \
  --preregistration transformation_inheritance/configs/e6a_gpt2_medium_small_preregistration.yaml \
  --reference-condition G0

./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir "$MS_PILOT/$MS_EXP/G0/seed0" --step 2000

./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --results "$MS_PILOT" --experiment "$MS_EXP" --seed 0 --pilot-step 2000 \
  --parity-report "$MS_PILOT/$MS_EXP/G0/seed0/parity/step_2000/parity_report.json"
```

Do not launch the full arm unless `pilot_gate.json` has `proceed: true`. A nonzero gate exit
with `decision: incomplete` means inputs are missing; it is different from a measured hold.

## 3. Primary full experiment - fresh root

The full arm is 3 conditions x 3 seeds x 10,000 steps. It must not share run directories
with the pilot.

```bash
MS_FULL=transformation_inheritance/results_full_medium_small
MS_EXP=e6a_gpt2_medium_small

for cfg in \
  e6a_gpt2_medium_small_ce \
  e6a_gpt2_medium_small_logit_kd \
  e6a_gpt2_medium_small_logit_attention_kd_aligned; do
  for seed in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/$cfg.yaml \
      --seed $seed --output-dir "$MS_FULL"
  done
done

for condition in G0 G1 G2-aligned; do
  for seed in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir "$MS_FULL/$MS_EXP/$condition/seed$seed" \
      --with-delta-ce --dtype bfloat16 --device cuda
  done
done

./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_medium_small_ce.yaml \
  --evaluate-public-reference --reference-condition PG2S \
  --reference-out "$MS_FULL/$MS_EXP/PG2S/reference" \
  --with-delta-ce --dtype bfloat16 --device cuda

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment "$MS_EXP" --results "$MS_FULL/$MS_EXP" \
  --preregistration transformation_inheritance/configs/e6a_gpt2_medium_small_preregistration.yaml \
  --reference-condition G0
```

## 4. Equal-head bridge preflight and pilot

The bridge is deliberately a fresh four-condition experiment. Its direct estimand is the
paired `G2-aligned - G2-legacy` difference under equal head counts. No post-hoc success
threshold is assigned to that method difference; report its uncertainty and all seeds.

```bash
BR_PILOT=transformation_inheritance/results
BR_EXP=e6a_gpt2_alignment_bridge

./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_alignment_bridge_ce.yaml \
  --evaluate-public-reference --reference-condition PDG2 \
  --reference-out "$BR_PILOT/$BR_EXP/PDG2/reference" \
  --with-delta-ce --dtype bfloat16 --device cuda
```

Inspect the bridge teacher record exactly as in section 1, changing the fingerprint root to
`$BR_PILOT/$BR_EXP/PDG2/reference/fingerprints`. Apply `e6a_bridge_stop_6` before training.

Run all four seed-0 conditions, then G0 seeds 1 and 2:

```bash
for cfg in \
  e6a_gpt2_alignment_bridge_ce \
  e6a_gpt2_alignment_bridge_logit_kd \
  e6a_gpt2_alignment_bridge_logit_attention_kd_legacy \
  e6a_gpt2_alignment_bridge_logit_attention_kd_aligned; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$cfg.yaml \
    --seed 0 --max-steps 2000 --output-dir "$BR_PILOT"
done

for seed in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_gpt2_alignment_bridge_ce.yaml \
    --seed $seed --max-steps 2000 --output-dir "$BR_PILOT"
done

for condition in G0 G1 G2-legacy G2-aligned; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir "$BR_PILOT/$BR_EXP/$condition/seed0" \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16 --device cuda
done

for seed in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir "$BR_PILOT/$BR_EXP/G0/seed$seed" \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16 --device cuda
done

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment "$BR_EXP" --results "$BR_PILOT/$BR_EXP" \
  --preregistration transformation_inheritance/configs/e6a_gpt2_alignment_bridge_preregistration.yaml \
  --reference-condition G0

./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir "$BR_PILOT/$BR_EXP/G0/seed0" --step 2000

./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --results "$BR_PILOT" --experiment "$BR_EXP" --seed 0 --pilot-step 2000 \
  --parity-report "$BR_PILOT/$BR_EXP/G0/seed0/parity/step_2000/parity_report.json"
```

The bridge gate compares G0 with the **last registered condition, G2-aligned**. It does not
mistake G2-legacy for the attention arm. The aligned-minus-legacy contrast is separately
written by the aggregator.

## 5. Equal-head bridge full experiment - fresh root

For the paper-worthy bridge estimate, run all four conditions at all three seeds. One seed
is a smoke/pilot reading, not a stable method comparison.

```bash
BR_FULL=transformation_inheritance/results_full_alignment_bridge
BR_EXP=e6a_gpt2_alignment_bridge

for cfg in \
  e6a_gpt2_alignment_bridge_ce \
  e6a_gpt2_alignment_bridge_logit_kd \
  e6a_gpt2_alignment_bridge_logit_attention_kd_legacy \
  e6a_gpt2_alignment_bridge_logit_attention_kd_aligned; do
  for seed in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/$cfg.yaml \
      --seed $seed --output-dir "$BR_FULL"
  done
done

for condition in G0 G1 G2-legacy G2-aligned; do
  for seed in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir "$BR_FULL/$BR_EXP/$condition/seed$seed" \
      --with-delta-ce --dtype bfloat16 --device cuda
  done
done

./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_alignment_bridge_ce.yaml \
  --evaluate-public-reference --reference-condition PDG2 \
  --reference-out "$BR_FULL/$BR_EXP/PDG2/reference" \
  --with-delta-ce --dtype bfloat16 --device cuda

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment "$BR_EXP" --results "$BR_FULL/$BR_EXP" \
  --preregistration transformation_inheritance/configs/e6a_gpt2_alignment_bridge_preregistration.yaml \
  --reference-condition G0
```

## 6. Paper interpretation checks

Before quoting results, verify:

| artifact | required reading |
|---|---|
| each `run_config.json` | exact model revisions, geometry contract passed, expected `head_alignment.method`, identical initial-state hashes within each seed |
| primary `checkpoint_metrics.csv` | carrier overlap is null with an unequal-head warning; fingerprint/topology metrics remain populated |
| bridge contrast rows | `G2-aligned - G2-legacy`, paired by seed; no invented threshold |
| `aggregate_summary.json` | correct experiment id and preregistration version; failure exclusions at or below the registered 2% rule |
| `go_no_go.json` | no unresolved null-spread rule once three G0 seeds exist |
| `pilot_gate.json` | correct OpenWebText corpus, five criteria including teacher-relative 2b, five-example parity, `proceed: true` before full execution |

The primary arm answers whether the E6A result survives a canonical model-scale change.
The bridge estimates how much of any difference comes from replacing index-JSD with the
AMAD-style JSD extension when head counts are already equal. It does not by itself prove
that every cross-arm difference is caused by model scale; report the two estimates together.
