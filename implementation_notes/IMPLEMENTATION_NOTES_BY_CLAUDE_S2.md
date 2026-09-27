# Append-only implementation journal - S2

Use the required CLAUDE filename even for Codex; record actual agent. Follow design/e6a_v2/templates/IMPLEMENTATION_LOG_TEMPLATE.md.

## 2026-09-27 - ChatGPT design handoff
Design only; no public trajectory evaluation or GPU tests completed. S2 is an external observational baseline,not S1's randomized CE-only control. Public training seeds differ from evaluator RNG. Native checkpoint inventory,immutable revisions and download budget remain gates. No automatic924-checkpoint fetch. Read studies/S2_PRETRAINING_DEVELOPMENT.md; sequential implementation reaches stage07. Preserve actual session communications/tests/decisions/milestones here.
## 2026-09-27T07:52:26Z - Codex - Stage00 shared foundation

Starting commit `05004b1`; user requested Stage00 only. The new package, Windows Python 3.12/CUDA 12.8 `uv.lock`, and approval boundary provide shared infrastructure; S2 evaluator/configuration is not implemented or enabled. No Upstream code reused and no dataset/model/network/GPU work occurred. Exact CPU suite: `.\\.venv\\Scripts\\python.exe -m pytest -q tests/unit tests/integration -m 'not gpu and not network'`, exit 0, 17 passed/0 failed/0 skipped; stage report pending. User was told the foundation and environment results. NEXT_STEPS 00.1-00.4 checked; next is final T00 report, then Stage01. Milestone SHA to be appended after commit.

## 2026-09-27T07:57:40Z - Codex (GPT-6) - Stage00 closeout
First tested milestone 3a84d0ae50750a6f71bc145f228a57b69f4bb514. T00 report: reports/stage00.json. Exact CPU suite reran with exit 0, 17 passed/0 failed/0 skipped. No GPU or S2 run occurred. NEXT_STEPS 00.1-00.5 and Stage00 checked; next Stage01 task 01.1.

## 2026-09-27T07:59:27Z - Codex (GPT-6) - milestone SHA correction
Stage00 completion report and checklist were committed as bd14d04 (docs(stage00): record CPU gate and handoff). The first tested foundation milestone was 3a84d0a. No study or GPU run was performed.

## 2026-09-27T09:07:48Z - Codex (GPT-6) - Stage02 S2 GPT-NeoX adapter primitives
Starting commit `5108da8`. Implemented tasks 02.1-02.4 for local eager `GPTNeoXForCausalLM`: native clean forward, exact fused-QKV layout with rotary Q/K, pure differentiable pre-dropout FP32 probabilities/projected Q/K/V, and transactional no-op/delete/relocate before value aggregation across exact native-layer scopes. Tiny/manual tests cover masks/right padding, rotary behavior, forward/backward equality, checkpoint recomputation, q0/length-1/saturated sinks, multiple doses, analytic downstream changes and exception restoration. Targeted CPU command `.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_interventions.py tests/unit/test_models.py` exited 0 with 40 passed/0 failed/0 skipped. Measured tiny GPT-NeoX no-op logit/gradient and feature-probability max errors were all 0; delete changed logits by max 0.1185167283. No public Pythia checkpoint, network, GPU, trajectory evaluation or scientific output was used; Stage07 inventory and Stage06 full-size/mixed-precision gates remain separate. `Upstream/` was neither used nor changed. Task 02.5/report/full regression remain pending; milestone commit pending.

## 2026-09-27T09:11:11Z - Codex (GPT-6) - Stage02 S2 CPU handoff
Implementation milestone `c7bd0202024a8fcea962f3aa4c027b19f74bf879`; clean-wheel regression fix `464fdd24ae46a74b9b88c8f4dcf6d4c5883f321e`. After the initial dependency-free wheel failure from eager Torch exports (67 passed/1 failed), lazy package exports restored the boundary; repair scope passed 41/41 and the full CPU regression passed 68/68 with one external warning. Pip and environment-lock checks passed. `reports/stage02.json` records exact commands/numerics. Stage02 is complete for tiny CPU GPT-NeoX only. No public checkpoint inventory/download, S2 trajectory evaluation, GPU/network/scientific run or `Upstream/` change occurred; full-size checks remain Stage06 and S2 inventory remains Stage07. Stop before Stage03; final evidence commit pending.
