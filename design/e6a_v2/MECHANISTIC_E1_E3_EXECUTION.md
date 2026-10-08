# Mechanistic E1-E3 execution contract

These commands extend the separate mechanism worktree. Original S1 training,
S4 probes, historical locks and results retain their original definitions.
E1, E2 and E3 are inference capabilities; their scientific results require
separate approved locks. The templates are **drafts, not preregistrations**.

## Scientific scope and explicit choices

E1 uses C2/C3/C6 at500/2000/10000 and C1/C5 at500/10000 (13 students),
plus one static teacher reference for each panel. Its declared five-alpha grid
is0/.25/.5/.75/1. Q-bias and model-local K-input top3/five random sets have
native scopes; teacher native36 and explicit mapped24 are separate. EPE
transport is the existing layer0 diagnostic, and position0-to1 is a broad
anchor control. No-op and reapplication controls are retained. Coordinates
across models are not homologous. Original S4 records are never overwritten.

The researcher chose E2's15 states: C1/C2/C3/C5/C6 at500/2000/10000.
Production E2 covers every native layer/head, plus a static teacher. Live
FP32/eager/cache-free Q/K/V/probability/output captures are checked against
actual single-layer deletion and clean-delta injection. Conditional softmax
handles sink mass rounding to one. Layer-local factors are not an additive
decomposition of simultaneous all-layer deletion.

E3 requires explicitly frozen states/layers, eta grid including zero,
normalization floor, random-control seed, non-sink key positions and causal
query support. Its reference is the clean residual **before ln_1**, entering
the selected attention sublayer. Sink/random/orthogonal/non-sink directions
edit a common support with equal requested norms; degeneracy and zero
reference norms remain unavailable. Their directions/bases differ across
checkpoints. Each item passes E2 parity before E3 injection. Clean-output
single-layer rescue must restore clean logits. All-layer deletion plus one
clean-output rescue is conditional and nonlinear. Explicit layer orders
produce exact telescopes; discovery requires an alternate order. Optional
Jacobian-vector analysis is not part of this implementation.

Teacher KL uses clean frozen teacher logits at temperature1. Behavior is
token-weighted over valid shifted targets. Sink/dose summaries identify their
item-weighted reductions; factor summaries use declared query support and
query-weighted pooling. Full-vocabulary, chunked, double-logsumexp geometry
saves per-target scalar accounts, signed positive/negative loss changes and
raw versus gauge-invariant centered logit displacement. No attention matrices,
activation tensors or full logits are written into records.

## Panel preparation and admission

Provide confirmation block IDs explicitly in a JSON list; no selection sweep
or selection based on confirmation measurements exists. Preparation verifies
the original artifact lock, full corpus envelope/packing/tokenizer contracts,
exact registered Full300 discovery, token IDs and underlying document hashes.
Blocks sharing any source document ID or normalized text hash are rejected.
Admission reconstructs this prepared panel read-only from the original pinned
inputs; this can take time for the complete corpus. Confirmation currently
uses explicitly selected blocks from the frozen OWT evaluation partition.
Other corpora require a prospective implementation/protocol amendment.

```powershell
$env:PYTHONPATH='E:/kd-sink-mechanistic-e0/src'
& E:/kd-sink/.venv/Scripts/python.exe scripts/run_mechanistic.py prepare-panel `
  --corpus <original-corpus-json> --registered-panels <original-panels-json> `
  --confirmation-ids <researcher-selected-id-list-json> --output <external-panel-json>
```

Fill the phase-specific draft template under `templates/`. Seal its payload
with `sinklab.provenance.seal_payload` after researcher approval; the operator
must pass the externally approved envelope digest. Unresolved values stay null
in drafts. Approved locks bind the exact grid, every selected source/config,
original S1 identity and separate D24 policy, panel/counts, settings and runtime.
The runtime binds all package modules, both new scripts, pyproject/uv pins,
actual Python/packages/CUDA/TF32 and fixed device name/UUID/memory. Qualification
must match this exact runtime/settings/phase and contain real GPT-2-large and
medium128-token FP32 parity and headroom evidence. Synthetic evidence cannot
authorize production. Changing execution-critical code requires a new lock.

Student sources require registered seed0 S1 checkpoint identity, original
protocol root, step, manifest/COMPLETE and weight/config SHA. They must declare
configuration-based random initialization. Teacher sources require the pinned
GPT-2-large revision. Only local safetensors/configs are loaded, with strict
tensor coverage and tied-embedding checks; no optimizer or network access.

```powershell
& E:/kd-sink/.venv/Scripts/python.exe scripts/run_mechanistic.py scientific `
  --phase E2 --seed 0 --state C2/step500 --panel confirmation --device cuda:0 `
  --lock <approved-phase-lock-json> --approved-sha256 <approved-envelope-digest> `
  --output <fresh-external-attempt-directory>
```

One command executes one state/panel, and never launches another condition,
seed or phase. Outputs must be external and fresh. No automatic cache reuse
or mid-run device migration occurs. Admission errors happen before model load
and output creation. Forward errors preserve partial records and FAILED.json;
a failed bundle is rejected even if a COMPLETE marker is present.

## Artifacts, audits and engineering commands

Every bundle contains invocation.json, ordinal per-item JSON, flushed structured
progress events, a recomputed summary, file-SHA manifest and COMPLETE binding.
Verification rehashes every file, rejects extra/missing files and paths, checks
identity and exact item/operation coverage, and reaggregates the summary.
Per-item progress reports phase/state/seed/run/device, elapsed time/ETA, input
tokens and allocated/reserved CUDA memory. Training counters are untouched.

```powershell
& E:/kd-sink/.venv/Scripts/python.exe scripts/run_mechanistic.py engineering `
  --phase E3 --seed 0 --device cpu --output <fresh-external-fixture-directory>
& E:/kd-sink/.venv/Scripts/python.exe scripts/run_mechanistic.py verify `
  --output <completed-bundle-directory>
& E:/kd-sink/.venv/Scripts/python.exe scripts/report_mechanistic.py `
  --bundle <completed-bundle-directory> --output <external-report-json> `
  --figure-prefix <external-figure-prefix>
```

Engineering runs use two synthetic128-token items and a tiny random GPT-2,
explicit fixture tolerances/doses, no scientific protocol or source checkpoint.
They are always labeled engineering_only. PNG/SVG figures and JSON reports
are separate artifacts and never modify a sealed bundle.

Complete scientific joins require one bundle per locked state (including the
teacher) and identical panel/item pairing. Supply repeated `--bundle`, `--lock`,
`--approved-sha256` and `--panel` to the report command. It rejects incomplete
grids and mixed protocols/runtime/settings/sources. Optional E1 `--dose-metric`
and explicit repeated `--dose-target` produce observed-bracket comparisons
between mapped24 teacher and native24 student Q/K curves. No extrapolation or
matching on loss is allowed; missing and ambiguous doses remain explicit.
EPE/anchor retain their separate raw curves and locus/scope caveats. No
automatic inheritance, equivalence, onset or mediation verdict is generated.

CPU tests, synthetic GPU tests, real production qualification and scientific
inference are separate evidence categories. Passing tests cannot guarantee
absence of bugs or establish a scientific conclusion.
