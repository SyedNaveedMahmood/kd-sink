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
