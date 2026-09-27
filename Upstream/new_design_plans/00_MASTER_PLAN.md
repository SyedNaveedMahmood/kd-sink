# 00 — Master Plan: Porting `MechanisticAccountofSinks` to Sink Inheritance (E6/E7)

**Scope:** how the existing repository must change, what new code is required, and in what order.
**Audience:** an implementing agent (Claude) with the repo checked out and this design pack in context.
**Hardware target:** 1× RTX 4080 SUPER, 16 GB.

---

## 1. Verdict on the existing codebase

The repo is a **single-shot analysis harness**, not a library. Three CLI scripts
(`intervention_analysis*.py` for gpt2 / neo / qwen) each own the full pipeline:
load model by HF id → sample the fixed SST-2/GSM8K/HumanEval benchmark → run
interventions a–j → write CSV/TXT → return `None`.

E6/E7 need the opposite shape: **many checkpoints × many corpora × programmatic
fingerprint records**, plus training code and cross-example patching that do not
exist at all.

The correct strategy is **not** to rewrite these harnesses. It is to extract a thin
programmatic core from them, leave the CLI entry points byte-identical in behaviour
(so the frozen E1–E5 numbers stay reproducible), and build E6/E7 on the extracted core.

### 1.1 What is directly reusable, unchanged

| Asset | Used by |
|---|---|
| `compute_bos_attention_metric` | every E6/E7 sink measurement |
| `INTERVENTIONS` registry + `intervention_a..j` (per arch) | E6 fingerprints |
| `NNsightEngine.run_all` / `run_intervention` / `EditPlan` | E6/E7 fingerprints |
| `run_parity_check`, `verify_parity`, `METRIC_ATOL/RTOL` | all parity gates |
| `identify_massive_coords*`, `select_massive_coords` | interventions (i)/(j) |
| `build_run_config` | provenance |
| `emergence_dynamics` onset/Jaccard/stability helpers | E6 drift trajectories |
| `evaluation_robustness.token_cross_entropy`, bootstrap utils | E6 functional cost |
| `tests/nnsight_smoke_utils.py` random-model pattern | E6/E7 offline tests |

### 1.2 Blocking gaps — nine of them

These are the reasons the code cannot run E6/E7 today. Each maps to a work package in §3.

| # | Gap | Evidence in repo | Severity |
|---|---|---|---|
| G1 | **Corpus is hard-wired.** `dataset_analysis()` calls `sample_benchmark_datasets()` internally in all three harnesses. No way to inject TinyStories blocks, SST-2 prompt-formatted text, FLORES rows, or XNLI prompts. | `intervention_analysis_neo.py:742-790`; same in gpt2/qwen | **Blocking** |
| G2 | **No programmatic return.** `dataset_analysis` writes files and returns `None`. E6 needs a fingerprint vector per checkpoint × 8 checkpoints × 3 conditions × 3 seeds = 72 fingerprint records to be joined in memory. | same functions | **Blocking** |
| G3 | **No functional CE for Neo/Qwen.** `capture_token_ce` exists only on `NNsightEngine.run_gpt2_trace` (GPT-2-only, `GPT2TracePlan`). Design §8.6 requires ΔCE under interventions c,d,f,g,h,i,j on a **GPT-Neo** student. The manual Neo/Qwen paths return attention only — they never reach `lm_head`. | `nnsight_engine.py:636-852`; `manual_self_attention_neo` returns no logits | **Blocking** |
| G4 | **No cross-example patching.** `NNsightEngine` applies edits computed from static weights inside one trace. There is no source-capture-trace → target-inject-trace mechanism, no activation cache keyed by example, no shape/dtype/device guard. E7 §10.9 is entirely new machinery. | absent | **Blocking** |
| G5 | **`ArchSpec` has no `v_path`.** Only `q_path`/`k_path`. E7 must extract and patch `V0` before KV-head repetition. Also no declared capture surface for post-RoPE K (Qwen applies RoPE *inside* `self_attn`, after `k_proj`). | `nnsight_engine.py:88-95` | **Blocking** |
| G6 | **Band policy breaks for a 4-layer teacher.** `compute_band(4,"scaled")` → `(3,4)`: **one layer**. `compute_band(8,"scaled")` → `(3,7)`: four layers. Teacher/student sink strengths would be computed over non-comparable depth regions, silently. | `intervention_analysis_legacy.py:620-631` | **Blocking, silent** |
| G7 | **No training code whatsoever.** No optimiser, no dataloader, no checkpointing, no LoRA. E6A and E6B are both training experiments. | absent | **Blocking** |
| G8 | **No local-checkpoint loading path.** `load_neo`/`load_qwen` are HF-id shaped; there is no way to hand in an already-instantiated `nn.Module` (needed for step-0 pre-update checkpoints and for merged-LoRA models held in memory). | `intervention_analysis_neo.py:125`, `_qwen.py:182` | High |
| G9 | **No cross-architecture comparison layer.** Nothing normalises depth, matches head counts, or computes fingerprint/topology/carrier distances. Teacher is 4L×768, student is 8L×256. | absent | **Blocking** |

### 1.3 Two silent-correctness traps to name explicitly

- **G6 (band).** Do not paper over it with `--layer-mode fixed`, which would give `(3,4)`
  for the teacher too. E6 must adopt a **normalised-depth band** and record it in every
  `run_config.json`. See `01_REFACTOR_SPEC` §3.
- **Neo local attention.** GPT-Neo alternates global/local. `L_ATTN` (D2) and any
  teacher/student attention comparison must mask invalid local positions rather than
  renormalise, and must report global and local layers separately. The teacher's layer
  0/1/2/3 attention types must be read from `config.attention_layers`, not assumed.

---

## 2. Target architecture after the refactor

```
common/
  intervention_analysis.py            (UNCHANGED — frozen)
  intervention_analysis_legacy.py     (UNCHANGED — frozen)
  nnsight_engine.py                   (ADDITIVE ONLY: v_path, run_arch_trace, capture hooks)
  datasets_loader.py                  (ADDITIVE ONLY: new providers)
  corpus_providers.py                 NEW  — G1
  fingerprint_runner.py               NEW  — G2, G3, G8
  inheritance_metrics.py              NEW  — G9
  paired_manifests.py                 NEW  — E7 manifests
  cross_example_patching.py           NEW  — G4, G5
  depth_band.py                       NEW  — G6

transformation_inheritance/           NEW  — G7 (E6)
  train_distillation.py
  train_sentiment_adaptation.py
  evaluate_transformation.py
  aggregate_transformation.py
  configs/*.yaml
  results/

crosslingual_semantics/               NEW  (E7)
  prepare_flores_manifest.py
  prepare_xnli_manifest.py
  extract_sink_representations.py
  evaluate_crosslingual_retrieval.py
  run_cross_language_patching.py
  aggregate_crosslingual.py
  configs/*.yaml
  results/

tests/                                NEW files added alongside existing
```

The design doc's file list is kept, plus three files it omits but which the gap
analysis shows are mandatory: `corpus_providers.py`, `fingerprint_runner.py`,
`depth_band.py`. Without them every E6 script would re-implement corpus injection
and checkpoint fingerprinting independently.

### 2.1 The one invariant

> **No existing file's observable behaviour may change.**
> Additive-only edits to `nnsight_engine.py`, `datasets_loader.py`.
> Zero edits to `intervention_analysis.py`, `intervention_analysis_legacy.py`,
> `intervention_analysis_neo.py`, `intervention_analysis_qwen.py`,
> `emergence_dynamics_analysis.py`, `evaluation_robustness_analysis.py`.

New code *calls into* those modules; it never modifies them. `evaluate_transformation.py`
imports `run_all_interventions` from the Neo harness rather than reimplementing a–j.
This is what makes the frozen-baseline rule (design §5) mechanically enforceable:
if the diff touches none of those files, the E1–E5 numbers cannot have moved.

---

## 3. Work packages

Twelve packages, WP0–WP11. Dependencies are strict — do not parallelise across an arrow.

```
WP0  freeze & baseline hashes
  │
  ├── WP1  depth_band.py + corpus_providers.py            (G1, G6)
  │     │
  │     └── WP2  fingerprint_runner.py                    (G2, G8)
  │           │
  │           ├── WP3  nnsight_engine additive: CE + v_path (G3, G5)
  │           │     │
  │           │     └── WP7  cross_example_patching.py     (G4)
  │           │
  │           ├── WP4  inheritance_metrics.py              (G9)
  │           │     │
  │           │     └── WP5  train_distillation.py         (G7, E6A)
  │           │           │
  │           │           └── WP6  train_sentiment_adaptation.py (E6B)
  │           │
  │           └── WP8  paired_manifests.py + prepare_*     (E7 data)
  │                 │
  │                 └── WP9  extract_sink_representations.py + retrieval
  │
  └── WP10 evaluate_transformation.py + aggregate_*  (needs WP2,WP4,WP5,WP6)
        │
        └── WP11 run_cross_language_patching.py + aggregate (needs WP7,WP8,WP9)
```

| WP | Deliverable | Spec file | Prompt |
|---|---|---|---|
| WP0 | Frozen baseline archive, `BASELINE_HASHES.json` | `01` §1 | P0 |
| WP1 | `depth_band.py`, `corpus_providers.py` | `02` §1–2 | P1 |
| WP2 | `fingerprint_runner.py` | `02` §3 | P2 |
| WP3 | `nnsight_engine.py` additive: `run_arch_trace`, `v_path`, `k_post_rope_path` | `01` §4–5 | P3 |
| WP4 | `inheritance_metrics.py` | `02` §4 | P4 |
| WP5 | `train_distillation.py` + 3 configs | `03` §1–3 | P5 |
| WP6 | `train_sentiment_adaptation.py` + 4 configs | `03` §4–5 | P6 |
| WP7 | `cross_example_patching.py` | `02` §6 | P7 |
| WP8 | `paired_manifests.py`, `prepare_flores_manifest.py`, `prepare_xnli_manifest.py` | `02` §5, `04` §1–2 | P8 |
| WP9 | `extract_sink_representations.py`, `evaluate_crosslingual_retrieval.py` | `04` §3–4 | P9 |
| WP10 | `evaluate_transformation.py`, `aggregate_transformation.py` | `03` §6–7 | P10 |
| WP11 | `run_cross_language_patching.py`, `aggregate_crosslingual.py` | `04` §5–6 | P11 |

Tests are **not** a separate WP. Each WP ships its own tests; the WP is not done
until they pass. `06_TEST_PLAN.md` assigns every test to its WP.

---

## 4. Execution order against compute

Map the WPs onto the design's Phase 0–3 compute plan:

**Phase 0 — no GPU, ~2 days of implementation**
WP0, WP1, WP2, WP3, WP4, WP7, WP8. All testable on random tiny models offline
(`tests/nnsight_smoke_utils.py` pattern). Do not touch a real checkpoint yet.

**Phase 1 — screening, ~1 GPU-day**
1. `gpt2` vs `distilgpt2` fingerprint via `fingerprint_runner` (WP2). Sanity: reproduces
   the frozen GPT-2 numbers exactly when given the E1 corpus.
2. Qwen 0.5B/1.5B base vs instruct fingerprints.
3. WP5 → E6A seed-0 pilot to 2,000 steps → **pilot gate (design §8.5)**.
4. WP9 → FLORES retrieval on Qwen2.5-0.5B only.
5. WP11 → 50-example XNLI patch smoke, four screening layers.

**Decision point.** Design §18 go/no-go. Do not start Phase 2 until at least one of
E6A/E7 clears its criteria.

**Phase 2 — core confirmation, ~1–2 GPU-weeks**
WP5 full (3 conditions × 3 seeds × 10k steps), WP6 (4 conditions × 3 seeds),
WP9/WP11 on 0.5B and 1.5B base+instruct.

**Phase 3 — conditional**
3B-Instruct, five-layer patch window, extra distillation objective.

### 4.1 Compute reality check

The design's compute plan is optimistic in one place. E6A is 3 conditions × 3 seeds
× 10,000 steps at 8,192 tokens/step ≈ **74M tokens per run, 663M total**, plus a
frozen 33M teacher forward every step for D1/D2. On a single 4080 SUPER, expect
roughly 4–9 hours per run, so **1.5–3 GPU-days for E6A alone** before any analysis.
Then evaluation is 8 checkpoints × 9 interventions × 300 sink blocks × 9 runs, plus
2,000-block ΔCE for 7 interventions.

Consequence for implementation: `evaluate_transformation.py` must be **resumable and
idempotent at (run_id, step, intervention) granularity**, writing one row per unit and
skipping completed units on restart. Treat this as a hard requirement, not a nicety.
Write `runtime_estimate.json` from the first 100 steps / 50 patch examples as the
design requires, and check it before committing to Phase 2.

---

## 5. Design-document deltas

Six places where implementation forces a decision the design leaves open. Each is
resolved in the spec files; listing them here so the choices are visible and auditable.

| # | Issue | Resolution | Where |
|---|---|---|---|
| D1 | Band for 4-layer teacher vs 8-layer student is incomparable (G6). | Normalised-depth band `[0.25, 0.90)` of layers, with a hard floor of ≥1 layer, recorded per model. Legacy `compute_band` retained for E1–E5 reproduction only. | `01` §3 |
| D2 | Design says "GPT-Neo intervention `b` is a structural no-op" and drops it. But `r_b` is also structurally absent for the **teacher**, so `F_Neo` is well-defined — however `r_e` (swap raw PE) and `r_d` differ in meaning across 4L vs 8L. | Fingerprint comparison uses the **mutually-defined intervention set** computed at runtime from each model's `INTERVENTIONS` registry and arch capabilities, not a hard-coded list. `available_interventions(model)` is a required helper. | `02` §4.2 |
| D3 | ΔCE on a 2,000-block validation set × 7 interventions × 8 checkpoints × 9 runs is ~1M forward passes. | ΔCE uses a **fixed 300-block subset** for per-checkpoint trajectories, and the full 2,000 blocks only at steps 0 and 10,000. Both are reported; the subset's sampling error is bootstrapped. | `03` §6.3 |
| D4 | Qwen K0 must be captured *post-RoPE*, but RoPE is applied inside `Qwen2Attention.forward`, after `k_proj`. There is no envoy on the post-RoPE tensor. | Capture `k_proj.output` and apply the repo's existing `_apply_rope`/`build_rope_cos_sin` (already implemented in the Qwen harness) outside the trace. Patching injects a pre-RoPE key derived by inverse rotation, **or** — preferred — patches at `k_proj.output` with the source's pre-RoPE key and documents that RoPE is then re-applied at the target position. Both variants must be implemented and cross-checked; they answer different questions. | `02` §6.4 |
| D5 | LoRA merged/unmerged parity at `1e-5` in float32 — but training is bf16. | Parity is checked by loading both checkpoints in fp32 on CPU. Training dtype is irrelevant to the check. State this in `run_config.json`. | `03` §5.4 |
| D6 | "Public TinyStories-8M as independent-convergence reference" — its tokenizer and vocab must be asserted identical to the 33M teacher's. | `train_distillation.py` asserts vocab size, BOS/EOS ids, and `len(tokenizer)` across teacher / public-8M / student config at startup and writes the assertion result to `run_config.json`. Abort on mismatch. | `03` §1.2 |

---

## 6. Minimum viable path

If motivation or compute runs short, the ordering above already front-loads the
cheapest decisive evidence. The smallest set that still supports a paper:

WP0–WP4 → WP5 (E6A, 3 seeds) → WP10, and WP7–WP9 → WP11 (E7 on 0.5B + 1.5B).
That is design §21's minimal publishable version. WP6 (E6B corrupted conditions)
and the 3B confirmation stay conditional.

---

## 7. How to use this pack

| File | Contents |
|---|---|
| `00_MASTER_PLAN.md` | this file — gaps, architecture, WP DAG, ordering |
| `01_REFACTOR_SPEC_existing_code.md` | exact edits to existing files, with the frozen-file list |
| `02_MODULE_SPEC_common.md` | full signatures for the six new `common/` modules |
| `03_MODULE_SPEC_e6_transformation.md` | E6A/E6B training, evaluation, aggregation |
| `04_MODULE_SPEC_e7_crosslingual.md` | E7 manifests, extraction, retrieval, patching |
| `05_SCHEMAS_AND_CONTRACTS.md` | every JSON/CSV schema, registry versions, run-dir layout |
| `06_TEST_PLAN.md` | every test, its WP, and the parity gates |
| `07_IMPLEMENTATION_PROMPTS.md` | P0–P11, one prompt per work package |

Give the implementing agent: the repo, this file, the relevant `0X_` spec, and the
matching prompt from `07`. Do not give it all eight files at once — the specs are
written to be loaded one WP at a time.
