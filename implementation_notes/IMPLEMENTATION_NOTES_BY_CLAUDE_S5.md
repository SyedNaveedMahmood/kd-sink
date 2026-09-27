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
