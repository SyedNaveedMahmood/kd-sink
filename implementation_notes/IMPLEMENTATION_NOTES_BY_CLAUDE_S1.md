# Append-only implementation journal - S1

Use the required CLAUDE filename even for Codex; record actual agent. Session template: design/e6a_v2/templates/IMPLEMENTATION_LOG_TEMPLATE.md.

## 2026-09-27 - ChatGPT design handoff
Design only; no S1 implementation,GPU test or training run completed. Read studies/S1_LONGITUDINAL_KD.md and shared contracts. Planned21 unique jobs; recommended27 physical jobs is an unapproved hardware-overlap proposal,not extra seeds. Dense every100 updates; full final resume mandatory. User-visible context and source discrepancies are in CORE journal. Next:stage00 and sequential dependencies. Record every future session,tests,decisions,communications and milestone here.

## 2026-09-27T07:52:26Z - Codex - Stage00 shared foundation
Starting commit `05004b1`; user requested Stage00 only. Added and CPU-tested strict S1 C0-C6 run selection, dimensions, random-student requirement, explicit seed, C4 device restriction, approval digest boundary, and a validation-only CLI. No Upstream code reused; no model weights, data, GPU, or S1 training run. Exact suite: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'`, CPython 3.12.3 CPU, exit 0, 17 passed/0 failed/0 skipped; final stage report pending. User was told the foundation and environment results. NEXT_STEPS 00.1-00.4 checked; Stage00 remains open until T00 report. Next: final Stage00 evidence, then Stage01 only after Stage00 completion. Milestone SHA to be appended after commit.

## 2026-09-27T07:57:40Z - Codex (GPT-6) - Stage00 closeout
First tested milestone 3a84d0ae50750a6f71bc145f228a57b69f4bb514. T00 report: reports/stage00.json. Exact CPU suite reran with exit 0, 17 passed/0 failed/0 skipped; offline wheel and tiny GPT-2 smoke included. No GPU or S1 run occurred. NEXT_STEPS 00.1-00.5 and Stage00 checked; next Stage01 task 01.1.
