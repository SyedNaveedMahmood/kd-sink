# E6A: GPT-2 large teacher -> random GPT-2 medium student

This is an isolated arm with experiment id `e6a_gpt2_large_medium`. It does not read from
or write to `e6a` or `e6a_gpt2` result directories.

## Frozen design

- Teacher: `gpt2-large` at revision `32b71b12589c2f8d625668d2335a01cac3249519`.
- Student: randomly initialised `gpt2-medium` config at revision
  `6dcaa7a952f72f9298047fd5137cd6e4f05f41da`.
- Seedless public reference: pretrained `gpt2-medium` at that same revision.
- Data, loss weights, optimiser, effective batch (16 x 4), seeds, steps, checkpoints and
  evaluation corpora are unchanged from `e6a_gpt2`.
- Layer alignment is the registered 36->24 endpoint-of-bin map.
- Attention-head alignment is the mean post-softmax attention distribution per mapped
  layer. This is required because teacher/student have 20/16 heads. Existing arms remain
  strict. Teacher-student carrier Jaccard is intentionally unavailable rather than
  truncating heads.

The preregistration is `configs/e6a_gpt2_large_medium_preregistration.yaml`. Do not alter it
after looking at the teacher preflight or pilot results; record any change as an amendment.

## Capacity and timing

The real-model RTX 3090 preflight used 13.8 GiB reserved at batch 16, including AdamW state
allocation, so the original batch fits a 24 GiB card. Expect roughly 8-11 hours per
10,000-step training run from the preflight; use each run's `runtime_estimate.json` after
step 100 as the authoritative estimate.

A medium-student run with all eight resumable checkpoints is expected to occupy 30-33 GiB.
Nine runs need approximately 290-310 GiB before fingerprints and aggregates. The current C:
free space is insufficient; E: had about 425 GiB free when this arm was prepared. Keep the
full output on E: or another volume with at least 350 GiB free.

## PowerShell setup

Run from the repository root:

```powershell
$Results = 'E:\E6A_GPT2_LARGE_MEDIUM'
$Preflight = 'E:\E6A_GPT2_LARGE_MEDIUM_PREFLIGHT'
$Experiment = 'e6a_gpt2_large_medium'
$ConfigRoot = '.\transformation_inheritance\configs'
$Configs = @(
  "$ConfigRoot\e6a_gpt2_large_medium_ce.yaml",
  "$ConfigRoot\e6a_gpt2_large_medium_logit_kd.yaml",
  "$ConfigRoot\e6a_gpt2_large_medium_logit_attention_kd.yaml"
)
$Conditions = @('G0','G1','G2')
New-Item -ItemType Directory -Force -Path $Results,$Preflight | Out-Null
```

## Phase 0: teacher stop-condition preflight

This output is deliberately outside `$Results`, so the teacher row cannot enter the student
aggregation.

```powershell
& .\.venv\Scripts\python.exe .\transformation_inheritance\evaluate_transformation.py `
  --config $Configs[0] `
  --evaluate-public-reference gpt2-large `
  --revision 32b71b12589c2f8d625668d2335a01cac3249519 `
  --reference-condition TGL `
  --reference-out "$Preflight\teacher" `
  --with-delta-ce --dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw "teacher preflight failed with $LASTEXITCODE" }
```

Read the `openwebtext_validation_sink_300` row in
`$Preflight\teacher\checkpoint_metrics.csv`. Stop and report if `baseline_sink <= 0.15`;
do not move the threshold.

The seedless public medium reference can also be measured now. It is written inside the
experiment tree because the final aggregator needs it:

```powershell
& .\.venv\Scripts\python.exe .\transformation_inheritance\evaluate_transformation.py `
  --config $Configs[0] `
  --evaluate-public-reference `
  --reference-condition PGM `
  --reference-out "$Results\$Experiment\PGM\reference" `
  --with-delta-ce --dtype bfloat16
if ($LASTEXITCODE -ne 0) { throw "public-reference evaluation failed with $LASTEXITCODE" }
```

## Phase 1: resumable seed-0 pilot

`--stop-after 2000` is intentional. It keeps the registered 10,000-step cosine schedule and
pauses at the gate checkpoint. Do **not** substitute `--max-steps 2000`: that creates a
different learning-rate schedule and cannot be resumed into the full run.

```powershell
foreach ($config in $Configs) {
  & .\.venv\Scripts\python.exe .\transformation_inheritance\train_distillation.py `
    --config $config --seed 0 --output-dir $Results --stop-after 2000 --resume auto
  if ($LASTEXITCODE -ne 0) { throw "$config seed 0 pilot failed with $LASTEXITCODE" }
}
```

Evaluate the five pilot checkpoints:

```powershell
foreach ($condition in $Conditions) {
  & .\.venv\Scripts\python.exe .\transformation_inheritance\evaluate_transformation.py `
    --run-dir "$Results\$Experiment\$condition\seed0" `
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
  if ($LASTEXITCODE -ne 0) { throw "$condition pilot evaluation failed with $LASTEXITCODE" }
}
```

Run parity and the gate:

```powershell
& .\.venv\Scripts\python.exe .\transformation_inheritance\run_pilot_parity.py `
  --run-dir "$Results\$Experiment\G0\seed0" --step 2000
if ($LASTEXITCODE -ne 0) { throw "parity harness failed with $LASTEXITCODE" }

& .\.venv\Scripts\python.exe .\transformation_inheritance\check_pilot_gate.py `
  --results $Results --experiment $Experiment --seed 0 --pilot-step 2000 `
  --parity-report "$Results\$Experiment\G0\seed0\parity\step_2000\parity_report.json"
if ($LASTEXITCODE -ne 0) { throw "pilot gate command failed with $LASTEXITCODE" }
```

Open `$Results\$Experiment\pilot_gate.json`. Continue only when `proceed` is `true`.

## Phase 2: full three-condition, three-seed run

First resume the three seed-0 pilots from step 2,000 to step 10,000:

```powershell
foreach ($config in $Configs) {
  & .\.venv\Scripts\python.exe .\transformation_inheritance\train_distillation.py `
    --config $config --seed 0 --output-dir $Results --resume auto
  if ($LASTEXITCODE -ne 0) { throw "$config seed 0 continuation failed with $LASTEXITCODE" }
}
```

Then run seeds 1 and 2:

```powershell
foreach ($config in $Configs) {
  foreach ($seed in 1,2) {
    & .\.venv\Scripts\python.exe .\transformation_inheritance\train_distillation.py `
      --config $config --seed $seed --output-dir $Results --resume auto
    if ($LASTEXITCODE -ne 0) { throw "$config seed $seed failed with $LASTEXITCODE" }
  }
}
```

## Phase 3: full evaluation and aggregation

Evaluation is resumable; pilot units already completed under identical settings are skipped.

```powershell
foreach ($condition in $Conditions) {
  foreach ($seed in 0,1,2) {
    & .\.venv\Scripts\python.exe .\transformation_inheritance\evaluate_transformation.py `
      --run-dir "$Results\$Experiment\$condition\seed$seed" `
      --steps all --with-delta-ce --dtype bfloat16
    if ($LASTEXITCODE -ne 0) { throw "$condition seed $seed evaluation failed with $LASTEXITCODE" }
  }
}
```

Aggregate only this experiment:

```powershell
& .\.venv\Scripts\python.exe .\transformation_inheritance\aggregate_transformation.py `
  --results $Results --experiment $Experiment `
  --preregistration "$ConfigRoot\e6a_gpt2_large_medium_preregistration.yaml" `
  --reference-condition G0
if ($LASTEXITCODE -ne 0) { throw "aggregation failed with $LASTEXITCODE" }
```

The final aggregate is `$Results\aggregate\aggregate_summary.json`. Before interpreting it,
confirm all nine runs have step 10,000 logs/checkpoints, evaluation summaries report zero
failed units, all same-seed initial-state and corpus hashes match across G0/G1/G2, and the
aggregate excludes every other experiment id.
