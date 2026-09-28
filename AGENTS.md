# Agent instructions: E6A v2 / sink inheritance

## Start every session here
Read `NEXT_STEPS.md`, `design/e6a_v2/README.md`, `design/e6a_v2/DECISIONS.md`, and the current stage file. Then read its required contracts and the latest relevant implementation journals. Do not rely on a previous chat. Work on the first incomplete stage whose dependencies are satisfied. Implement one stage at a time; do not launch an entire research campaign.

This is a new implementation of the revised attention-sink study, not a patch to the old trainer. The research question is whether sink **pattern, mechanism, and causal dependence** transfer together, and how this changes with training. A null result or increased student dependence is a valid result. Never encode the old paper's conclusion as an acceptance test.

## Immutable user requirements
- Main architecture: frozen GPT-2-large teacher, GPT-2-medium student initialized from configuration, never pretrained student weights.
- S1 conditions C0-C6 are independent, single-seed commands. A command must require one explicit `--seed`; no default seed list, hidden sweeps, or automatic next condition.
- Main horizon is 10,000 optimizer updates, not microsteps. Evaluate structure AND causal effects at step 0 and every 100 updates. Preserve a final, fully resumable checkpoint and allow later continuation without resetting optimizer, RNG, data order, or schedule history.
- Effective batch is 64 sequences of 128 input tokens: 8,192 input tokens/update. Standard next-token shifting scores 8,128 targets/update; report both counts accurately.
- Autotune microbatch only before training. The paired campaign uses ONE common microbatch and accumulation schedule, selected against its worst eligible GPU/condition combination. Never adapt it mid-run. Different safe maxima on two GPUs do not authorize different paired schedules.
- RTX 3090 is eligible for S1 C0-C6. RTX 4080 SUPER may train S1 conditions that pass the real production-shape profile/headroom gate; C0/C1/C2/C5/C6 have measured 4080 engineering profiles, C3 remains pending_4080_profile, and C4/REL is RTX-3090-only. Each approved production job has a fixed device role; no mid-run migration or automatic retuning. The final hardware lock records the condition/device eligibility matrix and one common paired batch schedule. CPU synthetic REL unit tests are allowed.
- The checked repository uses **`Upstream/`**, with an uppercase U. Treat both `Upstream/` and `upstream/` as reference-only. No runtime/test imports, paths, symlinks, configs, data, or checkpoint dependencies on either directory. Do not edit/delete it now. Eventual deletion needs researcher authorization and the independence gate.
- Borrow small reviewed functions/ideas with source SHA, license, modifications, and tests recorded. Do not copy the entire legacy framework or its conclusion-conditioned pilot gates.
- Keep code small and explicit: one package, one trainer, pure objective functions, explicit model adapters, one evaluator, one analysis layer. No distributed framework, experiment platform, plugin registry, or elaborate orchestration unless separately approved.

## Scientific authority
`DECISIONS.md` separates user requirements, verified source facts, proposed choices, and measured/pending choices. Templates are deliberately `draft`, not preregistrations. Implement declared proposals; do not silently choose unresolved scientific values. Production requires an approved, hashed protocol plus artifact/environment/hardware locks. Record amendments before affected runs. Never claim a plan is preregistered merely because it is committed.

Important corrections: this architecture is 20 -> 16 attention heads; the older medium/small case was 16 -> 12. Legacy `amad_js_attn_loss` computes differentiable cosine-soft weights without a separately parameterized alignment module. It is not the published Jin A2D module. C3 is post-softmax, head-mean probability MSE, inspired by but not an exact reproduction of TinyBERT. C4 is a causal, multilayer adaptation of MiniLMv2. Details are in `OBJECTIVES.md`.

## Session records and milestone commits
Maintain `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md` and, for every affected study, `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md` through `..._S6.md`. Keep these exact names even when the agent is Codex; identify the actual agent in each entry. Append chronological entries; never rewrite previous findings. Use the template under `design/e6a_v2/templates/`.

Record UTC time, starting commit, task IDs, files changed, concise decision rationale, discoveries, user-visible communications, exact test commands and exit status, failures/skips, artifact locations, next action, and milestone commit. Do not record secrets or hidden reasoning. A summary of technical reasons is sufficient.

Commit at significant milestones: contracts/tests, working implementation, completed stage. Use descriptive messages such as `feat(stage04): add exact-resume trainer`. Do not commit datasets, weights, credentials, or bulk logs. Include code, tests, notes, small sanitized evidence summaries, and NEXT_STEPS updates together. Never force-push or reset user work.

## Definition of done
Only mark a NEXT_STEPS task `[x]` after its implementation and required tests have actually passed. A missing GPU is **blocked**, not passed. Distinguish CPU tests, network tests, RTX 4080 SUPER tests, RTX 3090 tests, and scientific production runs. No test suite guarantees absence of all bugs. Preserve failing evidence and fix the cause rather than loosening scientific checks.

Every training command shows study/condition/seed/run ID, GPU, microbatch/accumulation/effective batch, optimizer-step tqdm progress, elapsed time, ETA, tokens/sec, current loss, and memory. Evaluation has separate progress. Resume restores cumulative counters. Structured logs must remain usable without an interactive terminal.

Never start long training, fetch hundreds of checkpoints, delete checkpoints, or enable optional studies without explicit operator scope. Never treat GPU-specific results as independently replicated seeds. Read all of the current stage's exit gates before coding.
