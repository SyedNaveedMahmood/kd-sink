# Append-only implementation journal - S5

Required CLAUDE filename applies to every agent; identify actual agent. Follow design/e6a_v2/templates/IMPLEMENTATION_LOG_TEMPLATE.md.

## 2026-09-27 - ChatGPT design handoff
Design only; no new S5 training is planned. Reuse S1 full/NoSink/SinkOnly/LogitKD records. NoSink removes key0 BEFORE alignment; SinkOnly matches binary mass,not a normalized single column. Necessity/sufficiency claims are restricted to supervision under this recipe. Read studies/S5_SINK_SUPERVISION.md; stage08. Append actual session tests,communications,decisions and milestones.
## 2026-09-27T07:52:26Z - Codex - Stage00 shared foundation

Starting commit `05004b1`; user requested Stage00 only. The installable package, exact Windows Python 3.12 lock, and protocol integrity boundary are shared foundations; S5 reanalysis is not implemented. No Upstream code reused or GPU/scientific run occurred. Exact CPU suite: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'`, exit 0, 17 passed/0 failed/0 skipped; report pending. User was told the foundation and environment results. NEXT_STEPS 00.1-00.4 checked; next is final T00 report, then Stage01. Milestone SHA to be appended after commit.

## 2026-09-27T07:57:40Z - Codex (GPT-6) - Stage00 closeout
First tested milestone 3a84d0ae50750a6f71bc145f228a57b69f4bb514. T00 report: reports/stage00.json. Exact CPU suite reran with exit 0, 17 passed/0 failed/0 skipped. No GPU or S5 run occurred. NEXT_STEPS 00.1-00.5 and Stage00 checked; next Stage01 task 01.1.

## 2026-09-27T07:59:27Z - Codex (GPT-6) - milestone SHA correction
Stage00 completion report and checklist were committed as bd14d04 (docs(stage00): record CPU gate and handoff). The first tested foundation milestone was 3a84d0a. No study or GPU run was performed.

## 2026-09-27T09:30:40Z - Codex (GPT-6) - Stage03 S5 supervision primitives
Starting commit `e008ee5`; read Stage03, OBJECTIVES/DECISIONS, model contract and latest CORE/S1/S3/S5 journals. Implemented shared S1 C2/C5/C6 objectives for later reuse-only S5 analysis: C5 removes sink information via fresh non-sink score softmax before cosine alignment and C6 aligns a two-outcome sink/rest distribution. Independent CPU forward/gradient references verify C5 direct original sink-score derivative exactly zero, +1000 sink-score perturbation invariance, C6 nonzero loss and invariance to non-sink redistribution at fixed sink mass. No `Upstream/` code copied or changed. Initial JSD backward NaNs (8 passed/5 failed) were fixed without relaxing checks. Final exact targeted command `.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_calibration.py tests/unit/test_objectives.py` exit 0, 18 passed; full CPU regression `.\.venv\Scripts\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'` exit 0, 86 passed, 0 failed/skipped, one external warning. `reports/stage03.json` records parity and limits. User was updated on implementation and tests. No S5 reanalysis, S1 training, GPU test, or scientific run. NEXT_STEPS Stage03 checked. Implementation commits `a32deba`/`5f7daae`; evidence commit pending. Next Stage04 under a new scope; S5 analysis remains Stage08.

## 2026-09-27T09:34:05Z - Codex (GPT-6) - Stage03 S5 milestone SHA
Stage03 CPU evidence/checklist commit: `362baf170892f853a7c77a3dd5996f987c11d5ef`. No S5 reanalysis was run.
## 2026-09-27T17:08:43Z - Codex (GPT-6) - Stage08 S5 reuse-only analysis

- Starting commit: `94cad936157c700a01d17296cb696853e5c3aac3`; task 08.2 and CPU portion of 08.4. Read S5 study, objectives, Stage08, evaluator record contract and latest CORE/S5 journal. User required no new S5 training and preservation of Stage06/07 blocks.
- Files: `src/sinklab/s5_analysis.py`, `tests/unit/test_stage08_s5.py`, `reports/stage08.json`, NEXT_STEPS and CORE/S5 journals. Implementation milestone `f17150d0499fd148ab41389d37936493ebb942ea`.
- Decision: verify sealed existing S1 aggregate and each per-item clean/delete/relocate record before extracting C1/C2/C5/C6 measures. Join by original seed/device/panel/protocol/initialization/data/precision/scope/items; report missing quartets without interpolation or hardware-replica seed inflation. Contrasts include S, full and sink-excluded attention, clean CE and causal effects. Source run IDs and record digests remain attached. No training entry point or outcome-dependent pairing was added.
- Tests: `.venv/Scripts/python.exe -m pytest -q tests/unit/test_stage08_probes.py tests/unit/test_stage08_s5.py tests/unit/test_stage08_s6.py` exit 0, 10 passed/0 failed/0 skipped; `.venv/Scripts/python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'` exit 0, 130 passed/0 failed/0 skipped. One external astor warning. `reports/stage08.json` records evidence. No production S1 evaluation records were present; no S5 scientific contrast was computed. User was updated on read-only joins and the absence of production records; no messages sent to others.
- NEXT_STEPS 08.2 checked for CPU capability; 08.4/Stage08 gate open. Next: reuse actual approved S1 records after those runs exist. Evidence commit pending; append SHA after commit.

## 2026-09-27T17:36:48Z - Codex (GPT-6) - Stage08 S5 evidence SHA

Evidence report/checklist/journals were committed locally as `6f4a6812e2a29b50a7199d75a0d765d5dda489dc`; implementation milestone remains `f17150d0499fd148ab41389d37936493ebb942ea`. The final full CPU suite passed 130/130. S5 still has zero production records joined and zero new training jobs; 08.4 and the Stage08 exit gate remain open. No push. Final polish commit pending.

## 2026-09-27T17:39:52Z - Codex (GPT-6) - Stage08 final polish SHA

Final S6 audit/report polish was committed as `d8c977184f59465ad65853b65e433f189c00c78c`. S5 reuse-only implementation remains at `f17150d`; no production S1 records were joined and no new S5 training was launched. Stage08 08.4 remains open.

## 2026-09-27T18:09:38Z - Codex (GPT-6) - Stage08 08.4 S5 external qualification boundary

- Starting commit `de829af`; user authorized Stage08 08.4 only. Read S5/Stage08 contracts and latest CORE/S5 journal. `src/sinklab/s5_analysis.py` was unchanged: S5 remains pure read-only joins over verified S1 C1/C2/C5/C6 records with no GPU kernel and no training path.
- On the RTX 4080 SUPER host, the missing-record path returned all four absent condition rows for one engineering seed/step. No S1-labeled fixture records were created to fill them, no production record was joined, and no S5 training occurred. GPU compute is not applicable to this analysis-only function. The CPU fixture join tests remain the applicable capability check.
- Evidence: `reports/stage08_4080_gpu_evidence.json` and `reports/stage08.json`. Final GPU command exit0,2 passed (S4/S6 tests plus S5 missing-record assertion); relevant T07-T10 CPU tests exit0,41 passed; full CPU regression exit0,130 passed/0 failed/0 skipped, one external astor warning. Initial missing temp parent and native-hook harness failures are recorded in CORE; neither involved S5. No network or S5 scientific contrast. User was updated on the reuse-only limit; no messages sent to others.
- Files changed for Stage08 08.4: GPU harness/frozen panel, reports, NEXT_STEPS and CORE/S4/S5/S6 journals. 08.4 capability gate checked; real S5 contrasts await genuine approved S1 records. Final evidence milestone commit pending; append SHA after commit.

## 2026-09-27T18:12:03Z - Codex (GPT-6) - S5 external-boundary evidence SHA

Stage08 08.4 evidence was committed as `51e7903a0a400e2daac80e2aa0b295d8603dba42`. S5 remains reuse-only with four missing real condition records reported and no fabricated replacements, new training or scientific contrast. No push.

## 2026-10-04T15:49:54Z - Codex (GPT-6) - D24 prospective S1 seed0 checkpoint follow-ups

- Starting branch/commit: main / `6c772b8845d5dc7446aa45a32bfd45c0879399c3`. Researcher explicitly approved all future checkpoint-dependent S1 analyses as seed0-only, with S4 C0-C6 at0,100,500,2000,10000 and S6 at0,500,2000,10000; preserve seed1/2 results/provenance and S2, allow S5 recorded-number reuse, and commit/push after validation. No scientific execution authorized.
- Read AGENTS, NEXT_STEPS, design README/DECISIONS, Stage08/09, model/intervention, software/artifact, training/checkpoint and metrics contracts; S1/S4/S5/S6 studies; D23 and historical approval locks; Stage08 report; relevant tests, runtime binding code and latest affected journals. No borrowed code.
- Changes: separate sealed D24 amendment and seed0 requirements config; small shared follow-up admission/inventory module; S4/S6/new shared-evaluator admission and separate policy provenance; explicit original in-training context; S5 extraction accepts both original and new keys; read-only readiness script; regression tests; active design/study/protocol guidance, NEXT_STEPS and `reports/d24_seed0_followups_validation.json`. Exact file inventory is in that report. Historical locks/configs/reports and source runs remain unchanged.
- Policy SHA `46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6` is a follow-up root, not a replacement training root. S4/S6 are designated-seed0 follow-ups, without across-training-seed reproducibility claims. Readiness needs 35/28 logical seed0 states and never seed1/2 weights. Checkpoint `protocol_hash` and evaluator `protocol_sha256` are accepted without mutation; conflicting roots fail. S2 and separately approved optional S3 scope remain unchanged; no optional execution enabled.
- Important runtime boundary: package edits change the execution-critical tree. Existing D23 training runtime bindings remain unchanged and reject this tree; no old lock was resealed to authorize training/resume. No active training process or run directory was touched.
- Validation environment: CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1. Initial focused command `.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_followup_policy.py tests/unit/test_stage08_probes.py tests/unit/test_stage08_s5.py tests/unit/test_stage08_s6.py` exited0: 53 passed/0 failed/0 skipped in54.25s. Final command `.\.venv\Scripts\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'` exited0: 310 passed/0 failed/0 skipped in42.51s, one external astor deprecation warning. Includes offline clean wheel and unchanged S2 regression; all model operations were tiny synthetic CPU tests.
- Failure preserved: first full command exited1, 309 passed/1 failed in52.90s because the original approval hashes all of DECISIONS.md. Restored that register exactly under its original LF-normalized SHA; did not change the old approval or loosen the test. D24 remains separate. Two draft documentation patch applications failed context matching before writing anything and were reapplied with verified exact text.
- Additional checks: 28 sealed protocol envelopes verified; 180 pre-existing protocol/config/report/decision-register/S2 files unchanged against the starting commit; git diff --check exit0. Origin/main was read-only checked at the starting commit before push. No GPU/network tests, real training/evaluation, profiling, calibration, downloads or artifact edits. Actual checkpoint readiness/coverage is not claimed.
- User-visible updates explained scope tracing, separate policy/original roots, successful focused tests, the schema distinction, the preserved historical-hash test failure and final 310-test pass. No messages sent to others.
- NEXT_STEPS: D24 policy/implementation/CPU validation checked; original launch-health statements labeled historical. Next: explicit seed0 source/device selection and read-only artifact/panel/runtime verification before separately authorized work. Commit/push this milestone; append exact milestone SHA after creation.

## 2026-10-04T15:51:34Z - Codex (GPT-6) - D24 milestone SHA and final validation

- Amendment, implementation, tests and evidence milestone: `320d1d963936856a937d094a1fc1d64c490dfa64`. D24 policy SHA remains `46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6`; final full CPU regression remains 310 passed, zero failures/skips, one external warning.
- Post-commit read-only `validate_runtime_source` against the original source milestone `9cb1815d44c3b2f42ddd508e5f9c71cc2ea274ec` returned the required refusal: `execution-critical committed tree differs from source milestone`. This confirms the old production binding was not weakened/resealed. No model/GPU access or production preflight was invoked. Changed JSON/Python parse, D24 seal, historical journal-prefix and Git whitespace checks passed.
- This evidence-only update records the actual milestone SHA in the report and affected journals. No runtime or scientific file changes since the passing suite. Next: commit this evidence and push both commits to origin/main; no scientific execution is authorized here.
