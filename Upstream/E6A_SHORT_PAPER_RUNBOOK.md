# E6A Short-Paper Runbook — Deadline: Tuesday, 2026-08-04 at 01:00 PDT

## 0. Decision, scope, and non-negotiable rules

This runbook targets a focused E6A-only short paper:

> Does sink organization inherit differently under ordinary language-model training,
> logit distillation, and attention-aware distillation?

The production study keeps the scientifically important E6A design intact:

- conditions D0, D1, and D2;
- seeds 0, 1, and 2;
- 10,000 optimizer steps;
- all eight registered checkpoints;
- the 300-block in-domain sink corpus and frozen-E1 cross-domain corpus;
- functional ΔCE (`--with-delta-ce`);
- the independently trained TinyStories-8M public reference;
- three-seed contrasts and calibrated null spreads.

The paper drops E6B, E7, and the conditional 3B confirmation. Do not make E7 or sentiment
claims from this run.

The code revision for every production and evaluation PC must be:

```text
afc150d1a81707320efe78b419075a61fc3f169d
```

The deadline and fixed auxiliary windows are in Pacific daylight time:

- hard final deadline: **Tuesday 2026-08-04 01:00 PDT**;
- 4090/5090 window 1: **Saturday 2026-08-01 13:00–Sunday 01:00 PDT**;
- 4090/5090 window 2: **Monday 2026-08-03 13:00–Tuesday 01:00 PDT**.

Do not silently change a seed count, loss weight, step horizon, corpus size, intervention
set, threshold, or failure rule to meet the deadline. If time becomes tight, use the
checkpoint-priority order in §10 and narrow the reported analysis transparently.

---

## 1. Is the current pilot necessary?

### Determination

The current D0/seed-0 2,000-step pilot is **not part of the short paper's inferential
sample** and cannot be resumed into a 10,000-step production run. Its cosine scheduler was
constructed with a 2,000-step horizon; resuming it under 10,000 steps would splice two
different schedules.

However, the run was already healthy at step 503/2,000 on Thursday morning, with validation
CE falling from 10.872 at step 0 to 3.533 at step 500. Finish this one D0 pilot because:

1. only about two hours remained when this runbook was written;
2. it verifies the complete real-data training path before nine expensive production runs;
3. its step-2,000 checkpoint can run the frozen manual-vs-NNsight parity check.

Do **not** spend time on D1 or D2 pilot runs. If the existing PowerShell loop automatically
starts `e6a_logit_kd --max-steps 2000` after D0 finishes, press **Ctrl+C immediately**.

Do not run `check_pilot_gate.py`: the formal gate needs all three pilot conditions plus
three D0 seeds. For this compute-constrained short-paper amendment, use the completed D0
pilot only as a training/parity sanity check, then exclude it from production.

### After D0 reaches step 2,000

From the 4080 SUPER repository:

```powershell
Set-Location 'C:\Users\user3\Sink-Small Paper'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path

# Confirm that the run really completed. E6A records completion in the checkpoint and
# per-step log; unlike E6B, it does not write train_summary.json.
$pilotRun = '.\transformation_inheritance\results\e6a\D0\seed0'
if (-not (Test-Path "$pilotRun\checkpoints\step_2000\checkpoint_sha256.txt")) {
  throw "Pilot step-2000 checkpoint is missing"
}
$lastTrain = Get-Content "$pilotRun\train_log.jsonl" -Tail 1 | ConvertFrom-Json
if ($lastTrain.step -ne 2000) {
  throw "Pilot training log stops at step $($lastTrain.step), not 2000"
}

# Inspect validation CE; the final value must be finite and lower than step 0.
$evalRows = Get-Content `
  '.\transformation_inheritance\results\e6a\D0\seed0\eval_log.jsonl' |
  ForEach-Object { $_ | ConvertFrom-Json }
$evalRows | Select-Object step,validation_ce,teacher_kl,l_attn
$ce0 = ($evalRows | Where-Object step -eq 0 | Select-Object -First 1).validation_ce
$ceFinal = ($evalRows | Sort-Object step | Select-Object -Last 1).validation_ce
if (-not [double]::IsFinite([double]$ceFinal) -or $ceFinal -ge $ce0) {
  throw "Pilot validation CE did not improve: step0=$ce0 final=$ceFinal"
}

# Mandatory instrument sanity check.
& $Py transformation_inheritance/run_pilot_parity.py `
  --run-dir transformation_inheritance/results/e6a/D0/seed0 `
  --step 2000
if ($LASTEXITCODE -ne 0) { throw "Pilot parity invocation failed" }

$parityPath = `
  '.\transformation_inheritance\results\e6a\D0\seed0\parity\step_2000\parity_report.json'
$parity = Get-Content $parityPath -Raw | ConvertFrom-Json
if ($parity.all_rows_pass -ne $true) {
  throw "Parity did not pass. Do not launch production."
}
```

Archive the pilot outside `transformation_inheritance/results`, so the production run gets
a genuinely fresh directory and the aggregator can never discover pilot metrics:

```powershell
$pilotSource = (Resolve-Path `
  '.\transformation_inheritance\results\e6a\D0\seed0').Path
$archiveRoot = Join-Path (Resolve-Path '.\results').Path '_exploratory'
$pilotArchive = Join-Path $archiveRoot 'e6a_D0_seed0_pilot_2000_20260730'
New-Item -ItemType Directory -Path $archiveRoot -Force | Out-Null
if (Test-Path -LiteralPath $pilotArchive) {
  throw "Pilot archive already exists: $pilotArchive"
}
Move-Item -LiteralPath $pilotSource -Destination $pilotArchive
```

If D1 or D2 pilot directories were accidentally started, stop their processes and archive
those directories under `results/_exploratory/` as explicitly labelled partial pilots.

Record this protocol amendment before launching production:

> On 2026-07-30, after the D0/seed-0 pilot had begun but before any 10,000-step E6A
> production run, the study was narrowed to an E6A-only short paper because of a fixed
> 2026-08-04 compute deadline. The completed D0 pilot was used only for training-path and
> parity validation and was excluded from all production aggregation. The D0/D1/D2
> production design, three seeds, 10,000-step horizon, corpora, checkpoints, metrics, and
> registered analysis rules were unchanged. E6B and E7 were not run and no claims about
> them are made.

---

## 2. Compute allocation

Training remains on the continuously available PCs. The short-window PCs perform
checkpoint-granular evaluation, which is resumable and can stop between checkpoints.

| PC | Production training | Evaluation ownership |
|---|---|---|
| 4080 SUPER, continuous | seed 0: D0/D1/D2; then seed 2: D0/D1/D2 | seed 2/D2; final collection and aggregation |
| 3090, continuous | seed 1: D0/D1/D2 | seed 0/D2 and all seed-1 runs |
| 4090, Saturday + Monday | none | Saturday seed 0/D0; Monday seed 2/D0 |
| 5090, Saturday + Monday | none | public reference + Saturday seed 0/D1; Monday seed 2/D1 |

Why the 4090/5090 are not assigned production training:

- a measured 4080S 10,000-step run projects to about 14 hours including corpus setup;
- the 4090/5090 rates are not measured yet;
- a hard 12-hour cutoff could interrupt a run after step 5,000 and lose several hours
  before the next registered checkpoint;
- evaluation naturally commits one `(run, checkpoint, corpus)` unit at a time.

Conditions within each seed stay on one training GPU. This preserves the paired-seed design
and avoids confounding D0/D1/D2 with training hardware.

### Conservative expected timeline

| Time (PDT) | 4080 SUPER | 3090 | 4090 | 5090 |
|---|---|---|---|---|
| Thu morning | finish D0 pilot, parity, archive | environment/preflight | unavailable | unavailable |
| Thu ~11:30 onward | train seed 0, then seed 2 | train seed 1 | unavailable | unavailable |
| Sat 13:00–Sun 01:00 | continue seed 2 | finish training/evaluate | evaluate seed 0/D0 | public reference, evaluate seed 0/D1 |
| Sunday | finish seed 2, begin evaluation | evaluate seed 0/D2 + seed 1 | unavailable | unavailable |
| Mon 13:00–22:30 | evaluate/collect | evaluate/transfer | evaluate seed 2/D0 | evaluate seed 2/D1 |
| Mon 22:30–Tue 00:30 | final audit + aggregate | transfer/check | transfer/check | transfer/check |
| Tue 00:30–01:00 | contingency only | contingency only | no new work | no new work |

---

## 3. One-time preflight on every PC

Use one local checkout per PC. Replace `<REPO>` with that PC's repository path.

```powershell
Set-Location '<REPO>'
$ExpectedSha = 'afc150d1a81707320efe78b419075a61fc3f169d'

# On a fresh or otherwise clean checkout, pin the exact code revision. Do this before any
# result is produced on that PC.
if (git status --porcelain --untracked-files=no) {
  throw "Tracked working-tree changes exist; do not switch revisions over them"
}
git fetch origin
git switch --detach $ExpectedSha

if (-not (Test-Path '.\.venv\Scripts\python.exe')) {
  py -3.12 -m venv .venv
  & .\.venv\Scripts\python.exe -m pip install -r requirements.txt
  if ($LASTEXITCODE -ne 0) { throw "Environment installation failed" }
}
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path

if ((git rev-parse HEAD).Trim() -ne $ExpectedSha) {
  throw "Wrong git revision on this PC"
}
if (git status --porcelain) {
  throw "Working tree is dirty. Resolve it before producing paper artifacts."
}

$env:CUDA_VISIBLE_DEVICES = '0'
& $Py -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); print(torch.cuda.get_device_properties(0).total_memory)"
if ($LASTEXITCODE -ne 0) { throw "CUDA/Python preflight failed" }

& $Py -m pip check
if ($LASTEXITCODE -ne 0) { throw "Broken Python requirements" }

& $Py -m pytest tests/ -q
if ($LASTEXITCODE -ne 0) { throw "Full Phase-0 test suite failed" }
```

Also ensure:

- Windows sleep/hibernation is disabled while jobs run;
- at least 75 GB is free on the two training PCs;
- at least 20 GB is free on each auxiliary evaluation PC;
- system clocks and timezone are correct;
- no second Python training process is using the GPU;
- the TinyStories models/dataset can download or are already cached.

Do not pull new code after the first production run starts.

---

## 4. 4080 SUPER commands — seed 0 and seed 2 production

Run only after §1 parity passes and the pilot directory has been archived.

```powershell
Set-Location 'C:\Users\user3\Sink-Small Paper'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'
$ExpectedSha = 'afc150d1a81707320efe78b419075a61fc3f169d'

$configs = @(
  'e6a_ce',
  'e6a_logit_kd',
  'e6a_logit_attention_kd'
)

foreach ($seed in 0,2) {
  foreach ($config in $configs) {
    Write-Host "START production $config seed $seed at $(Get-Date -Format o)"
    & $Py transformation_inheritance/train_distillation.py `
      --config "transformation_inheritance/configs/$config.yaml" `
      --seed $seed --resume auto --device cuda
    if ($LASTEXITCODE -ne 0) {
      throw "Production training failed: $config seed $seed"
    }
    Write-Host "DONE production $config seed $seed at $(Get-Date -Format o)"
  }
}
```

There is deliberately no `--max-steps`: the YAML's registered 10,000-step horizon must be
used. `--resume auto` is safe only within one 10,000-step production run.

After each run, verify:

```powershell
$conditions = 'D0','D1','D2'
foreach ($seed in 0,2) {
  foreach ($condition in $conditions) {
    $run = ".\transformation_inheritance\results\e6a\$condition\seed$seed"
    if (-not (Test-Path "$run\checkpoints\step_10000\checkpoint_sha256.txt")) {
      Write-Warning "$condition seed $seed is not complete yet"
      continue
    }
    $cfg = Get-Content "$run\run_config.json" -Raw | ConvertFrom-Json
    if ($cfg.max_steps -ne 10000 -or $cfg.git_sha -ne $ExpectedSha) {
      throw "Invalid run configuration for $condition seed $seed"
    }
    Write-Host "$condition seed $seed complete"
  }
}
```

When seed 0 is complete (expected before Saturday's auxiliary window), export its three
evaluation bundles as described in §8.

After seed 2/D2 training completes, this PC owns that run's evaluation:

```powershell
& $Py transformation_inheritance/evaluate_transformation.py `
  --run-dir transformation_inheritance/results/e6a/D2/seed2 `
  --steps all --with-delta-ce --dtype bfloat16 --device cuda
if ($LASTEXITCODE -ne 0) { throw "D2 seed2 evaluation failed" }
```

If `--steps all` cannot finish by Monday 22:00, use the priority groups in §7.

---

## 5. 3090 commands — seed 1 production

Start after the current D0 pilot's parity passes. The 3090 can prepare its environment and
download assets while the pilot finishes.

```powershell
Set-Location '<3090_REPO>'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'

foreach ($config in 'e6a_ce','e6a_logit_kd','e6a_logit_attention_kd') {
  Write-Host "START production $config seed 1 at $(Get-Date -Format o)"
  & $Py transformation_inheritance/train_distillation.py `
    --config "transformation_inheritance/configs/$config.yaml" `
    --seed 1 --resume auto --device cuda
  if ($LASTEXITCODE -ne 0) {
    throw "Production training failed: $config seed 1"
  }
  Write-Host "DONE production $config seed 1 at $(Get-Date -Format o)"
}
```

Verify all three `step_10000/checkpoint_sha256.txt` files exist. Then evaluate the three
local seed-1 runs, one at a time:

```powershell
foreach ($condition in 'D0','D1','D2') {
  & $Py transformation_inheritance/evaluate_transformation.py `
    --run-dir "transformation_inheritance/results/e6a/$condition/seed1" `
    --steps all --with-delta-ce --dtype bfloat16 --device cuda
  if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed: $condition seed1"
  }
}
```

The 3090 also owns seed 0/D2 evaluation. Import that evaluation bundle from the 4080S and
run:

```powershell
& $Py transformation_inheritance/evaluate_transformation.py `
  --run-dir transformation_inheritance/results/e6a/D2/seed0 `
  --steps all --with-delta-ce --dtype bfloat16 --device cuda
if ($LASTEXITCODE -ne 0) { throw "D2 seed0 evaluation failed" }
```

If time is tight, use §7's priority groups rather than `--steps all`.

---

## 6. Saturday and Monday 4090/5090 evaluation assignments

### Saturday 4090 — seed 0/D0

Before 13:00, import the complete seed 0/D0 evaluation bundle from the 4080S.

```powershell
Set-Location '<4090_REPO>'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'

$RunDir = 'transformation_inheritance/results/e6a/D0/seed0'
$NoNewWorkAfter = Get-Date '2026-08-01 23:00'
```

Run the checkpoint groups using §7. Stop starting new groups at 23:00, preserve completed
units, and export the updated run directory before the 01:00 access cutoff.

### Saturday 5090 — public reference first, then seed 0/D1

```powershell
Set-Location '<5090_REPO>'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'

# Run exactly once in the whole study.
& $Py transformation_inheritance/evaluate_transformation.py `
  --config transformation_inheritance/configs/e6a_ce.yaml `
  --evaluate-public-reference roneneldan/TinyStories-8M `
  --with-delta-ce --dtype bfloat16 --device cuda
if ($LASTEXITCODE -ne 0) { throw "Public-reference evaluation failed" }

$RunDir = 'transformation_inheritance/results/e6a/D1/seed0'
$NoNewWorkAfter = Get-Date '2026-08-01 23:00'
```

Run §7's checkpoint groups, then export:

- `transformation_inheritance/results/e6a/P8M/reference/`;
- the updated seed 0/D1 run.

### Monday 4090 — seed 2/D0

Import the seed 2/D0 bundle before 13:00.

```powershell
Set-Location '<4090_REPO>'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'
$RunDir = 'transformation_inheritance/results/e6a/D0/seed2'
$NoNewWorkAfter = Get-Date '2026-08-03 22:00'
```

Run §7's groups. Export completed metrics no later than 22:30.

### Monday 5090 — seed 2/D1

Import the seed 2/D1 bundle before 13:00.

```powershell
Set-Location '<5090_REPO>'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$env:CUDA_VISIBLE_DEVICES = '0'
$RunDir = 'transformation_inheritance/results/e6a/D1/seed2'
$NoNewWorkAfter = Get-Date '2026-08-03 22:00'
```

Run §7's groups. Export completed metrics no later than 22:30.

Do not start a new evaluation group after `$NoNewWorkAfter`. If a group remains active at
00:30, stop it with Ctrl+C; already committed ledger units remain resumable. Copy the
updated artifacts before access ends.

---

## 7. Resumable checkpoint-priority evaluator

Use this block after setting `$Py`, `$RunDir`, and `$NoNewWorkAfter`.

The priority order is:

1. endpoints 0 and 10,000 — essential primary result and full endpoint ΔCE;
2. 1,000, 2,000, and 5,000 — matched-loss and trajectory coverage;
3. 100, 250, and 500 — early trajectory resolution.

```powershell
$StepGroups = @(
  '0,10000',
  '1000,2000,5000',
  '100,250,500'
)

foreach ($group in $StepGroups) {
  if ((Get-Date) -ge $NoNewWorkAfter) {
    Write-Warning "Stopping before group $group because the slot safety cutoff was reached"
    break
  }

  Write-Host "START eval $RunDir steps=$group at $(Get-Date -Format o)"
  & $Py transformation_inheritance/evaluate_transformation.py `
    --run-dir $RunDir --steps $group `
    --with-delta-ce --dtype bfloat16 --device cuda
  if ($LASTEXITCODE -ne 0) {
    throw "Evaluation failed: $RunDir steps=$group"
  }
  Write-Host "DONE eval $RunDir steps=$group at $(Get-Date -Format o)"
}
```

Re-running the block is safe. The evaluator skips completed
`(run_id, checkpoint, corpus)` units.

Afterward, inspect:

```powershell
$summaryPath = Join-Path $RunDir 'evaluation_summary.json'
$metricsPath = Join-Path $RunDir 'checkpoint_metrics.csv'
if (Test-Path $summaryPath) {
  Get-Content $summaryPath -Raw | ConvertFrom-Json |
    Select-Object n_checkpoints,n_written,n_skipped,n_failed
}
if (Test-Path $metricsPath) {
  Import-Csv $metricsPath |
    Group-Object checkpoint_step |
    Sort-Object { [int]$_.Name } |
    Select-Object Name,Count
}
```

---

## 8. Moving evaluation bundles between PCs

Use a shared drive, fast external SSD, or LAN share. Set the same transfer root convention
on every PC:

```powershell
$TransferRoot = '<SHARED_OR_EXTERNAL_PATH>\E6A_20260804'
```

An evaluation bundle needs:

- `run_config.json`;
- `eval_log.jsonl`;
- the complete `checkpoints/` directory.

It does not need the multi-million-row `block_manifest.parquet`.

### Export a training run for evaluation

Example for D0/seed0:

```powershell
$SourceRun = `
  '.\transformation_inheritance\results\e6a\D0\seed0'
$Bundle = Join-Path $TransferRoot 'e6a\D0\seed0'
New-Item -ItemType Directory -Path $Bundle -Force | Out-Null

Copy-Item "$SourceRun\run_config.json","$SourceRun\eval_log.jsonl" `
  -Destination $Bundle -Force
robocopy "$SourceRun\checkpoints" "$Bundle\checkpoints" /E /Z /R:2 /W:5
if ($LASTEXITCODE -ge 8) { throw "Checkpoint bundle copy failed" }
```

### Import on an evaluator PC

```powershell
$Bundle = Join-Path $TransferRoot 'e6a\D0\seed0'
$LocalRun = `
  '.\transformation_inheritance\results\e6a\D0\seed0'
New-Item -ItemType Directory -Path $LocalRun -Force | Out-Null

Copy-Item "$Bundle\run_config.json","$Bundle\eval_log.jsonl" `
  -Destination $LocalRun -Force
robocopy "$Bundle\checkpoints" "$LocalRun\checkpoints" /E /Z /R:2 /W:5
if ($LASTEXITCODE -ge 8) { throw "Checkpoint bundle import failed" }
```

### Export evaluation outputs

Each run has exactly one evaluation owner at a time. Never evaluate the same copied run
concurrently on two PCs.

```powershell
$ResultFiles = @(
  'checkpoint_metrics.csv',
  'eval_units.jsonl',
  'evaluation_summary.json'
)
foreach ($name in $ResultFiles) {
  $source = Join-Path $LocalRun $name
  if (Test-Path $source) {
    Copy-Item $source -Destination (Join-Path $Bundle $name) -Force
  }
}
```

On the final collector (4080S), import those three result files into the matching
production run directory. Preserve the exact D0/D1/D2 and seed0/seed1/seed2 hierarchy.

Transfer the public-reference directory as:

```text
transformation_inheritance/results/e6a/P8M/reference/
```

Do not copy a pilot `checkpoint_metrics.csv` into the production tree.

---

## 9. Final completeness audit and aggregation — 4080 SUPER collector

Target time: Monday 22:30 PDT. Do not wait until 00:50 to discover a missing seed.

### Verify all nine production runs

```powershell
Set-Location 'C:\Users\user3\Sink-Small Paper'
$Py = (Resolve-Path '.\.venv\Scripts\python.exe').Path
$ExpectedSteps = @(0,100,250,500,1000,2000,5000,10000)
$problems = @()

foreach ($condition in 'D0','D1','D2') {
  foreach ($seed in 0,1,2) {
    $run = ".\transformation_inheritance\results\e6a\$condition\seed$seed"
    $metrics = Join-Path $run 'checkpoint_metrics.csv'
    if (-not (Test-Path $metrics)) {
      $problems += "missing metrics: $condition seed$seed"
      continue
    }

    $rows = Import-Csv $metrics
    $observedSteps = @($rows | ForEach-Object { [int]$_.checkpoint_step } |
      Sort-Object -Unique)
    $missingSteps = @($ExpectedSteps | Where-Object { $_ -notin $observedSteps })
    $badRows = @($rows | Where-Object { $_.status -ne 'ok' })

    Write-Host "$condition seed$seed rows=$($rows.Count) steps=$($observedSteps -join ',') bad=$($badRows.Count)"
    if ($missingSteps.Count) {
      $problems += "$condition seed$seed missing steps $($missingSteps -join ',')"
    }
    if ($badRows.Count) {
      $problems += "$condition seed$seed has $($badRows.Count) non-ok rows"
    }
  }
}

# The three conditions at one seed must have identical random initialization and
# block-manifest hashes; this is what makes the contrasts paired.
foreach ($seed in 0,1,2) {
  $configs = @(
    Get-Content ".\transformation_inheritance\results\e6a\D0\seed$seed\run_config.json" -Raw | ConvertFrom-Json
    Get-Content ".\transformation_inheritance\results\e6a\D1\seed$seed\run_config.json" -Raw | ConvertFrom-Json
    Get-Content ".\transformation_inheritance\results\e6a\D2\seed$seed\run_config.json" -Raw | ConvertFrom-Json
  )
  $initHashes = @($configs.initial_state_sha256 | Sort-Object -Unique)
  $manifestHashes = @($configs.block_manifest_sha256 | Sort-Object -Unique)
  if ($initHashes.Count -ne 1 -or $manifestHashes.Count -ne 1) {
    $problems += "seed$seed is not paired across D0/D1/D2 initialization/manifests"
  }
}

$reference = `
  '.\transformation_inheritance\results\e6a\P8M\reference\checkpoint_metrics.csv'
if (-not (Test-Path $reference)) {
  $problems += 'missing P8M public-reference metrics'
}

if ($problems.Count) {
  $problems | ForEach-Object { Write-Warning $_ }
  throw "Completeness audit failed; repair or explicitly narrow the analysis before aggregation"
}
```

### Aggregate

```powershell
& $Py transformation_inheritance/aggregate_transformation.py `
  --results transformation_inheritance/results
if ($LASTEXITCODE -ne 0) { throw "E6A aggregation failed" }
```

Inspect the decision and exclusion artifacts:

```powershell
$aggregate = '.\transformation_inheritance\results\aggregate'
Get-Content "$aggregate\aggregate_summary.json" -Raw | ConvertFrom-Json |
  Select-Object n_rows_read,n_rows_used,exclusions,pending_decisions
Get-Content "$aggregate\go_no_go.json" -Raw | ConvertFrom-Json |
  Select-Object decision,n_met,n_not_met,n_unknown,pending_decisions_blocking
```

Required integrity conditions:

- all nine production runs represented;
- all three seeds represented for every condition;
- public reference represented exactly once;
- no pilot rows;
- `pending_decisions_blocking` empty;
- `n_unknown == 0` for the full registered E6A analysis;
- no unexamined failed rows or >2% failure exclusion;
- all artifacts retain their original git SHA, device, dtype, and manifest provenance.

A scientifically negative `decision` is still a reportable result. Missing data are not.

---

## 10. Deadline contingency hierarchy

If the schedule slips, make reductions in this order. Record every reduction.

1. **Drop E6B/E7/3B** — already done by this scope.
2. **Stop running any pilot beyond the current D0 pilot.**
3. **Preserve all nine 10,000-step production training runs.**
4. **Preserve endpoint evaluation (0 and 10,000) for all nine runs and P8M.**
5. Preserve 1,000/2,000/5,000 for matched-loss and trajectory claims.
6. Only then omit 100/250/500 evaluation.

Never reduce:

- three seeds;
- D0/D1/D2;
- 10,000 production steps;
- `--with-delta-ce`;
- the 300-block sink corpus;
- the public reference;
- provenance, parity, or failure reporting.

If only endpoint evaluation is complete by Monday 22:00, produce an explicitly endpoint-only
short paper and do **not** claim the full preregistered matched-loss/trajectory analysis.
Do not change the YAML to make an incomplete result look complete.

---

## 11. Final paper framing

The short paper may claim only what the completed E6A data support:

- whether D1/D2 alter teacher-student sink fingerprint similarity relative to D0;
- whether mechanistic, topological, and functional inheritance agree or dissociate;
- whether conclusions replicate across three seeds;
- whether they survive the frozen-E1 cross-domain corpus;
- how distilled students compare with an independently trained public 8M model.

State explicitly:

- the pilot-gate deviation and timing;
- that the D0 pilot was excluded from inference;
- that production remained D0/D1/D2 × three seeds × 10,000 steps;
- which GPUs trained which seeds;
- that all conditions within a seed used the same training GPU;
- total GPU-hours and per-run runtimes;
- any checkpoint evaluation omitted because of the deadline;
- all failed or excluded units;
- that E6B/E7 were outside this short paper and therefore provide no evidence here.

This is a focused E6A paper, not a reduced claim that the entire sink-inheritance program
was completed.
