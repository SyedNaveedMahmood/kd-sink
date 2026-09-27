# Append-only implementation journal - S4

Required CLAUDE filename applies to every agent; identify actual agent. Follow design/e6a_v2/templates/IMPLEMENTATION_LOG_TEMPLATE.md.

## 2026-09-27 - ChatGPT design handoff
Design only; mechanistic probes not implemented or validated on new teacher. Read studies/S4_MECHANISTIC_INHERITANCE.md and model/intervention contract. Coordinates are model-local; EPE transport conserves vector sum,not arbitrary norms. Teacher responsiveness must be measured; oldmedium circuit cannot be assumed. Stage08 after shared dependencies. Append actual tests,communications,decisions and milestones.
## 2026-09-27T07:52:26Z - Codex - Stage00 shared foundation

Starting commit `05004b1`; user requested Stage00 only. The installable package, exact Windows Python 3.12 lock, and protocol integrity boundary are shared foundations; S4 probes are not implemented. No Upstream code reused or GPU/scientific run occurred. Exact CPU suite: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'`, exit 0, 17 passed/0 failed/0 skipped; report pending. User was told the foundation and environment results. NEXT_STEPS 00.1-00.4 checked; next is final T00 report, then Stage01. Milestone SHA to be appended after commit.

## 2026-09-27T07:57:40Z - Codex (GPT-6) - Stage00 closeout
First tested milestone 3a84d0ae50750a6f71bc145f228a57b69f4bb514. T00 report: reports/stage00.json. Exact CPU suite reran with exit 0, 17 passed/0 failed/0 skipped. No GPU or S4 run occurred. NEXT_STEPS 00.1-00.5 and Stage00 checked; next Stage01 task 01.1.

## 2026-09-27T07:59:27Z - Codex (GPT-6) - milestone SHA correction
Stage00 completion report and checklist were committed as bd14d04 (docs(stage00): record CPU gate and handoff). The first tested foundation milestone was 3a84d0a. No study or GPU run was performed.
## 2026-09-27T17:08:43Z - Codex (GPT-6) - Stage08 S4 probe capability

- Starting commit: `94cad936157c700a01d17296cb696853e5c3aac3`; task 08.1 and CPU portion of 08.4. Read S4 study, model/intervention contract, Stage08 and latest CORE/S4 journal. User requested Stage08 only and no production S1/S2 or optional long run.
- Files: `src/sinklab/probes.py`, `tests/unit/test_stage08_probes.py`, `reports/stage08.json`, NEXT_STEPS and CORE/S4 journals. Implementation milestone `f17150d0499fd148ab41389d37936493ebb942ea`.
- Decision: position0->1, absolute-position removal, all-layer Q-bias, layer0 EPE transport, model-local top3 K-input and five distinct random controls have explicit loci/layers/coordinates. Batched EPE matches per-item reference and conserves vector sum only. Sealed per-item records include panel/checkpoint/run provenance, masks, behavior, guarded sink ratios and responsiveness. Edits and module modes/RNG restore after exceptions. GPT-NeoX route battery is explicitly inapplicable.
- Tests: `python -m pytest -q tests/unit/test_stage08_probes.py tests/unit/test_stage08_s5.py tests/unit/test_stage08_s6.py` under `.venv` exit 0, 10 passed/0 failed/0 skipped; full CPU regression `python -m pytest -q tests/unit tests/integration -m 'not gpu and not network'` exit 0, 130 passed/0 failed/0 skipped. Exact Windows launcher was `.venv/Scripts/python.exe`. An intermediate stale S6 test import caused collection exit 1, corrected before final pass. Report: `reports/stage08.json`.
- No real GPT-2-large teacher or frozen calibration panel was available. Real teacher responsiveness/nonresponsiveness is unmeasured; synthetic fixture labels cannot substitute. RTX 3090 absent and no Stage08 GPU smoke authorized. No S4 scientific state evaluated. User was updated on implementation, parity/restoration and these limits; no messages sent to others.
- NEXT_STEPS 08.1 checked for CPU capability; 08.4/Stage08 gate open. Next: separately authorized real-teacher qualification and GPU smoke. Evidence commit pending; append SHA after commit.

## 2026-09-27T17:36:48Z - Codex (GPT-6) - Stage08 S4 evidence SHA

Evidence report/checklist/journals were committed locally as `6f4a6812e2a29b50a7199d75a0d765d5dda489dc`; implementation milestone remains `f17150d0499fd148ab41389d37936493ebb942ea`. Final CPU suite in `reports/stage08.json` passed 130/130; no real teacher responsiveness or GPU smoke was inferred. No push or scientific run. Final polish commit pending.

## 2026-09-27T17:39:52Z - Codex (GPT-6) - Stage08 final polish SHA

Final S6 audit/report polish was committed as `d8c977184f59465ad65853b65e433f189c00c78c`. S4 probe implementation remains at `f17150d`; no new S4 experiment or teacher calibration was run. Stage08 08.4 remains open.

## 2026-09-27T18:09:38Z - Codex (GPT-6) - Stage08 08.4 real teacher S4 qualification

- Starting commit `de829af`; user authorized only Stage08 08.4. The existing fixed probe definitions were untouched. Read Stage08/S4/model contracts and latest CORE/S4 journal. The engineering calibration text and predeclared control seed/floors were committed as `25e8a75` before any probe result; source SHA-256 `29974142a3f3c9aead3d0e5f314da64b107aaf136a46df844a449a2535ad4745`.
- Pinned real GPT-2-large revision `32b71b12589c2f8d625668d2335a01cac3249519`, weight SHA-256 `5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce`, FP32 on RTX 4080 SUPER; four agent-authored engineering items at max128, 71/68/72/67 real tokens. All ten probes, including five random controls, completed and were labeled responsive at the fixed1e-6 numerical floor. No real-teacher nonresponsive/inapplicable probe. This label is not route specificity or scientific transfer; random controls also respond. Baseline S0.4235755881; EPE DeltaS-0.416383/self-KL0.268398, Q-bias DeltaS-0.111599, top3 K-input DeltaS-0.027863. Full effects are in `reports/stage08_4080_gpu_evidence.json`.
- Teacher top3[792,870,440] and random student top3[657,366,563] were independently computed from their own E0. Batched EPE/reference max error5.96e-8 and vector-sum max error2.38e-7; clean/no-op teacher logits exact. 40 sealed records retain masks, loci, coordinates and provenance under the external v3 scratch path in CORE. Edited parameter hash, modes, probe hooks and CPU/CUDA RNG restored on normal and injected-exception paths. Native Transformers lazy output-capture hooks were distinguished from probe hooks.
- First GPU command failed setup (missing temp parent; no probe). Second failed a hook-baseline harness assertion after battery; no scientific probe change. `660c716` fixed native-hook accounting; `5f0162c` added no-op/record assertions. Final exact GPU command and test counts are in CORE and `reports/stage08.json`: RTX4080 v3 exit0,2 passed/0 failed/0 skipped; T07-T10 targeted CPU exit0,41 passed; full CPU exit0,130 passed, one external warning. No network, 3090, production state or scientific S4 evaluation. User was updated on these findings; no messages sent to others.
- Files: frozen panel/GPU test, compact GPU evidence, Stage08 report, NEXT_STEPS and CORE/S4/S5/S6 journals. 08.4 capability gate checked; Stage06/07 and scientific coverage remain separate. Final evidence milestone commit pending; append SHA after commit.

## 2026-09-27T18:12:03Z - Codex (GPT-6) - S4 real-teacher evidence SHA

Real GPT-2-large S4 engineering qualification and compact probe evidence were committed as `51e7903a0a400e2daac80e2aa0b295d8603dba42`. The full fixed battery passed on the RTX 4080 SUPER; no scientific S4 retained-state coverage or production panel was inferred. No push.
