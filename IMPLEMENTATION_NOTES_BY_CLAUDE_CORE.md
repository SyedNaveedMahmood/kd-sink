# Append-only implementation journal - CORE

The owner requires this filename for Claude,Codex and other coding agents. Identify the actual agent in entries. Use design/e6a_v2/templates/IMPLEMENTATION_LOG_TEMPLATE.md.

## 2026-09-27 - ChatGPT design handoff
Scope: researched E6A v2 design,not implementation/training. Reference commit96a80e1e7ad581728c8befd697547f9b3f3a2e85; uppercase Upstream preserved. User requirements include independent single-seed objectives,10k continuation,dense100-update causal measurements,common paired batching,REL-only3090,extensive tests,sequential stages and milestone commits.

Important findings: old large/medium used mean-head JSD; paper-facing amad_jsd is differentiable cosine-soft alignment without a separate adapter. New probability-MSE/causal REL are explicit adaptations. Binary SinkOnly and masking BEFORE alignment are specified to prevent vacuous/leaky controls. Main architecture is20-to16 heads. Decisions/proposals/unknown measurements are separated in DECISIONS.md; no new research approval is inferred.

User-visible communication: the design pack includes S1-S6,C0-C6,ten stages,checkpoint contracts,test gates,journals; source naming discrepancy reported; GitHub additions preserve Upstream. Experiment CPU/GPU tests and scientific runs are NOT performed by this design handoff. Subsequent agents must replace no claims with invented evidence.

Next action: read AGENTS.md,NEXT_STEPS.md,DECISIONS.md and implement stage00. Do not launch a campaign until approvals/locks/actualGPU gates are complete.
