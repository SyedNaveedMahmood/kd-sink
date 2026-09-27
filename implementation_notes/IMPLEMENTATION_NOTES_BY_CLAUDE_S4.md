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
