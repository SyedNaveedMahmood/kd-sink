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
