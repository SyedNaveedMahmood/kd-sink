# Minimal software and artifact interfaces

## Target structure
```text
AGENTS.md
NEXT_STEPS.md
implementation_notes/
  IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md
  IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md ... _S6.md
pyproject.toml
src/sinklab/
  cli.py config.py provenance.py data.py
  models.py objectives.py interventions.py metrics.py
  hardware.py train.py checkpoint.py evaluate.py
  pythia.py analysis.py plotting.py
configs/
protocols/
reports/
tests/unit/ tests/integration/ tests/gpu/
design/e6a_v2/
Upstream/  # reference only, never imported
```

Start with a few explicit modules. Split only for a demonstrated responsibility. No plugin hierarchy, experiment class per condition, mandatory cloud tracker, database, default network telemetry or general distributed-training framework. Use typed dataclasses or another small strict schema; reject unknown keys, wrong variants and missing seeds.

Candidate dependencies: Torch, Transformers, datasets/huggingface_hub for explicit preparation, NumPy,SciPy,PyYAML,tqdm,matplotlib,pytest; safetensors/pyarrow only where useful. Stage00 resolves exact compatible versions from official sources and actual tests, then locks them. Do not blindly install latest or import upstream's full environment. No NNsight/PEFT requirement. Prepared training works offline.

## Boundaries
- resolve_config -> immutable RunSpec with one study/condition/seed, exact variant and approval digest; no fallback.
- build_batch_plan(profile_reports,protocol) -> common microbatch, accumulation, eligibility and hashes.
- make_student(initial_state,contract) -> exact shared random state, never teacher/pretrained initialization.
- compute_auxiliary(features,spec) -> differentiable loss tensor plus detached diagnostics; pure, no optimizer/hidden state.
- apply_attention_intervention(scores,probabilities,mask,spec) -> probabilities before value aggregation; pure and no-op tested.
- evaluate(model,manifest,spec) -> item records, restoring RNG/mode and exact denominators.
- save_resume/load_resume -> atomic verified full-state persistence and compatibility validation.

## Locks
protocol.lock.json contains approval/version/source commit; study/condition variants; model/tokenizer/data/calibration hashes; layer maps; losses/weights/masks; optimizer/schedule/dropout/precision/backend; evaluation scope/cadence/panels; retention/analysis/budget; amendments. Hash canonical sorted JSON payload, then wrap with its digest; never hash a self-referential digest field.

hardware.lock.json includes every required measured profile and the global schedule. Both locks are required. Paths may differ across PCs but content hashes must match. A mutable filename or matching seed alone does not identify scientific equivalence.

## Records
run.json: run ID,S/C/variant/seed,device block, all locks and initialization/input/model hashes, code commit, dependencies, UTC start, parameter counts, precision,horizon,parent/extension,status. Complete means final checkpoint AND expected evaluation coverage verified.

train.jsonl: optimizer step, elapsed,input/target counts, raw/weighted losses,LR,grad norm,throughput,memory,events/checkpoint refs. Inactive losses are null/not-computed, not fake zero measurements.

Item records in JSONL or partitionedParquet: schema version,run/study/condition/seed/model revision,step,tokens,corpus/manifest/item IDs,scope integer list,intervention/strength/version,precision,valid-target count,clean/intervened NLL sums,KL sum,absolute-log-change sum,flip count,structure,numerical diagnostics,status. Fields have units; unavailable/not_applicable/failed differ from zero. Aggregates retain contributing IDs and denominators.

Checkpoint index: path,step,class(weights/full),checksum manifest,complete,protected,parent,metric refs. One writer lock per run directory. Do not overwrite completed scientific jobs with a convenience force flag.

## Run plans and analyses
Plans enumerate independent seed/condition/device jobs, including hardware replicas and optionalS3. No implicit multiseed default. Aggregation checks protocol/init/data/schedule/metric versions and duplicate IDs. Missing jobs stay missing. Never pool bugged/corrected objectives, oldE6A rows or different variants without an explicitly separate analysis.

The S1 amendment's mandatory plan is seven C0-C6 seed0 jobs on RTX3090. Optional seed1 and seed2 plans are separate explicit seven-job manifests on the same role. The 4080 S1 configs are engineering-only; production entry and Trainer reject that role for every S1 condition. `protocols/s1_researcher_amendment_20260928.json` records partial researcher authority; `s1_artifact_partial_v1.json` and `s1_environment_verified_v1.json` record verified components. None is a production protocol lock. Corpus/panel, full environment install, common hardware and calibration locks remain pending.

## Coding-session records and commits
Every session appends to CORE and affected S1-S6 required journals: actual agent identity, UTC,start commit,tasks,sources,changes,decisions/reasons,discoveries,user-visible summary,exact tests/exit codes/pass-fail-skip counts,artifacts,next action. Preserve history and add dated corrections; do not expose secrets/private reasoning. Use exact CLAUDE filenames even forCodex.

Significant milestone commits include code/tests/journal/NEXT_STEPS together. No weights,datasets,credentials or huge logs in Git. A task is checked only when implemented AND its required tests ran and passed; missingGPU is BLOCKED.

Borrowed code requires source path/commit/blob/license/function, destination, literal-or-conceptual reuse, changes and independent tests in the ledger. Build/install a wheel into a clean directory with Upstream absent and networking disabled. All imports, fixtures,configs and prepared workflows must still work. Historical documentary links are allowed; runtime paths/symlinks/sys.path tricks are not. Do not delete Upstream until separately authorized.
