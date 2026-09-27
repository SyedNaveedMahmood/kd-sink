# 07 — Implementation Prompts (P0–P11)

One prompt per work package. Copy verbatim into a fresh session.

**How to use these.** Each prompt names the files to attach. Attach *only* those — the
specs are written to be loaded one WP at a time, and loading all eight design files
plus the repo degrades output quality. Every prompt should be run against a checked-out
repo at the `sink-inheritance` branch.

**Standing preamble** — prepend to every prompt below:

> You are extending an existing research codebase (`MechanisticAccountofSinks`) for an
> ACL-level mechanistic interpretability paper on attention sinks. Absolute rules:
> (1) Twelve files listed in `BASELINE_HASHES.json` are frozen and must not be edited —
> new code calls into them. (2) Edits to `common/nnsight_engine.py` and
> `common/datasets_loader.py` are additive only: new functions, or new keyword arguments
> whose defaults reproduce current behaviour exactly. (3) Write complete, runnable,
> modular code with type hints and docstrings — no placeholders, no `TODO`, no
> `pass  # implement later`. (4) State assumptions, edge cases, and limitations
> explicitly in comments where the code makes a non-obvious choice. (5) Ship the tests
> named in the spec in the same response as the module. (6) Every output artefact
> carries provenance: git sha, manifest hash, seed, dtype, device. (7) If a spec
> requirement is ambiguous or appears wrong, say so before implementing rather than
> guessing.

---

## P0 — Freeze the baseline

**Attach:** repo, `01_REFACTOR_SPEC_existing_code.md`

> Execute WP0 from section 1 of the refactor spec.
>
> 1. Create the `sink-inheritance` branch and the `frozen-e1-e5` tag.
> 2. Write `BASELINE_HASHES.json` at the repo root with the exact schema in §1,
>    computing real sha256 values for all twelve frozen files and capturing the actual
>    installed versions of torch, transformers, nnsight, numpy, scipy, and python.
> 3. Write `tests/test_frozen_files.py`, which recomputes every hash and asserts
>    equality, with a failure message naming which file drifted.
> 4. Tell me the exact command to run the frozen GPT-2 dataset analysis so I can
>    populate `reference_results.gpt2_small_table1` from the real
>    `bos_attention_stats_overall.csv`. Do not invent those numbers — leave the field
>    with a clear sentinel and instruct me to fill it.
> 5. Append `peft`, `pyyaml`, and `scipy` to `requirements.txt` without reordering or
>    re-pinning any existing line. Report exactly which lines you added.
>
> Then confirm the existing test suite still passes.

---

## P1 — `depth_band.py` and `corpus_providers.py`

**Attach:** repo, `02_MODULE_SPEC_common.md`, `01_REFACTOR_SPEC_existing_code.md` (§3, §6)

> Implement WP1: `common/depth_band.py` and `common/corpus_providers.py`, plus the
> additive loaders in `common/datasets_loader.py` described in refactor spec §6.
>
> Context you must internalise first: `compute_band(4, "scaled")` returns `(3, 4)` — a
> single layer — for the 4-layer TinyStories-33M teacher, while the 8-layer student gets
> four layers. Comparing sink strength across those bands is meaningless and produces no
> error. `depth_band.py` fixes this for new experiments without touching `compute_band`,
> which E1–E5 reproduction depends on.
>
> Requirements:
> - Follow module spec §1 and §2 exactly, including the `Corpus` / `CorpusItem`
>   dataclasses and the `manifest_sha256` construction (canonical JSON, `sort_keys=True`,
>   `ensure_ascii=False`).
> - `frozen_e1_corpus` must wrap the existing `sample_benchmark_datasets` verbatim, not
>   reimplement its sampling or truncation.
> - The new `datasets_loader` functions are additions; `load_optional_flores` stays and
>   is not replaced.
> - Reuse the existing dual-import idiom used by the cross-arch harnesses.
>
> Ship these tests: `test_depth_band.py`, `test_corpus_determinism.py`,
> `test_corpus_e1_bridge.py`, `test_tinystories_packing.py`,
> `test_sst2_prompt_format.py` — contents per `06_TEST_PLAN.md` §1. The band test must
> assert that `normalised_depth_band(12)` equals `compute_band(12, "scaled")`, which is
> the reason `(0.25, 0.90)` was chosen.
>
> Print the realised band and depth interval for L ∈ {4, 6, 8, 12, 24, 28, 36} so I can
> check the choice by eye.

---

## P2 — `fingerprint_runner.py`

**Attach:** repo, `02_MODULE_SPEC_common.md`, output of P1

> Implement WP2: `common/fingerprint_runner.py` per module spec §3.
>
> This is the seam between the frozen CLI harnesses and every new experiment. The frozen
> `dataset_analysis()` functions hard-code their corpus and return `None`; this module
> gives the same measurement, on any corpus, as a returnable `FingerprintRecord`.
>
> Requirements:
> - `ModelHandle`, `load_handle`, `handle_from_module`, `compute_fingerprint`,
>   `available_interventions`, `mutual_interventions`, and the fingerprint cache, exactly
>   as specified.
> - **Dispatch, never reimplement.** For `engine="manual"` call the frozen harness's
>   `run_all_interventions`; for `engine="nnsight"` call `NNsightEngine.run_all`. Reduce
>   with the frozen `compute_bos_attention_metric`, passing `layer_start`/`layer_end`
>   from the band. If you find yourself writing an attention computation, stop.
> - `handle_from_module` must wrap an in-memory `nn.Module` for NNsight without a disk
>   round-trip — needed for step-0 pre-update checkpoints and merged LoRA models.
> - `available_interventions` computes the applicable set from the arch (e.g. `int_b` is
>   structurally a no-op for GPT-Neo because `q_proj.bias is None`; `int_e` is undefined
>   for RoPE models). Do not hard-code a list.
> - Guard `raw["int_a"] < 1e-8`: record a failure and emit NaN rather than dividing.
> - Failures are recorded with item id, intervention, exception class, and message —
>   never dropped.
>
> Ship `test_fingerprint_runner_bridge.py`, `test_available_interventions.py`,
> `test_fingerprint_cache.py`.
>
> The bridge test is the most important thing in this WP: `compute_fingerprint` on GPT-2
> with `frozen_e1_corpus` must reproduce the frozen `bos_attention_stats_overall.csv`
> within `METRIC_ATOL`. If it does not, the seam is wrong and everything downstream is
> untrustworthy. Report the max abs deviation per intervention.

---

## P3 — `nnsight_engine.py` additive edits

**Attach:** repo, `01_REFACTOR_SPEC_existing_code.md` (§4, §5), `02_MODULE_SPEC_common.md` (§6.4)

> Implement WP3: additive edits to `common/nnsight_engine.py` per refactor spec §4–5.
>
> Two gaps to close.
>
> **G5 — `ArchSpec` capture surface.** Add `v_path`, `o_path`,
> `rope_applied_inside_attn`, `kv_grouped`, all defaulting so existing `ARCH_SPECS`
> entries are unchanged in behaviour. Populate for gpt2/opt/neo/qwen per §4.
>
> **G3 — CE for non-GPT-2 architectures.** `capture_token_ce` exists only on
> `run_gpt2_trace`, which is GPT-2-specific. E6A needs ΔCE under interventions
> c,d,f,g,h,i,j on a GPT-Neo student. Add `run_arch_trace(plan: EditPlan, ...)` per §5.2.
>
> Critical constraint: **factor, do not fork.** Extract the edit body of
> `run_intervention` (roughly lines 534–588) into a private `_apply_edits`, and call it
> from both `run_intervention` and `run_arch_trace`. Preserve the existing envoy access
> order (out-of-order access raises `MissedProviderError`) and do not add an outer
> `torch.no_grad()` — the existing comment explains why.
>
> Reuse `evaluation_robustness.token_cross_entropy`; do not write a second CE
> implementation unless you also ship a test proving the two agree to 1e-7.
>
> `capture_attention="targets"` must reduce to `[num_layers, num_heads, n_targets]`
> inside the trace so full `[H,S,S]` maps never leave it — this is what makes 300-block ×
> 9-intervention × 72-checkpoint evaluation fit in memory.
>
> Ship `test_run_intervention_parity.py`, `test_arch_trace_ce.py`,
> `test_archspec_vpaths.py`, `tests/nnsight_e6_smoke.py`.
>
> The parity test is the gate: GPT-2 and Neo fingerprints through `run_intervention`
> after the refactor must match the pre-refactor values within `METRIC_ATOL`/`METRIC_RTOL`.
> Capture those values before you refactor.
>
> Also answer explicitly: for Qwen2, is `k_proj.output` at position 0 identical to the
> post-RoPE key at position 0? Verify numerically on a random Qwen2 config and report
> the max abs difference. This determines how `02` §6.4 is implemented.

---

## P4 — `inheritance_metrics.py`

**Attach:** `02_MODULE_SPEC_common.md` (§4), `06_TEST_PLAN.md`

> Implement WP4: `common/inheritance_metrics.py` per module spec §4. Pure analysis —
> numpy/scipy/pandas only, no model loading, no HF imports.
>
> Implement every function in §4.1–4.6.
>
> Non-obvious requirements, all of which are there for a reason:
> - `keys` is always an explicit argument. No function may take a union or intersection
>   of dict keys implicitly — teacher and student have different applicable intervention
>   sets, and an implicit key set is how that silently goes wrong.
> - NaN policy: any key NaN in either fingerprint is dropped from all metrics; the report
>   records `n_keys_used` and `dropped_keys`; comparisons with different `n_keys_used`
>   must be refusable downstream.
> - `weighted_jaccard` **raises** on unequal head counts rather than truncating.
> - `interp_depth_profile` handles `L == 1` by returning a constant profile with a
>   warning, not by dividing by zero.
> - `topology_wasserstein` L1-normalises both profiles to sum 1 and returns the
>   normalisation constants, so an all-zero profile is reported rather than yielding a
>   spurious distance of 0.
> - `paired_seed_contrast` must return `min_attainable_p` and a note that the test is
>   descriptive. With three seeds the minimum two-sided p from a paired permutation test
>   is 0.25; the code must make it impossible to present that as significance.
> - **Do not implement a composite `inheritance_score`.** The design forbids a single
>   aggregate score; the four components are always reported separately.
>
> Ship `test_inheritance_metrics.py` covering every identity case, the NaN policy, the
> head-count raise, `L=1`, and the `min_attainable_p=0.25` assertion at n=3.

---

## P5 — `train_distillation.py` (E6A)

**Attach:** repo, `03_MODULE_SPEC_e6_transformation.md` (§1–3), outputs of P1/P2/P4

> Implement WP5: `transformation_inheritance/train_distillation.py`, the three configs
> `e6a_ce.yaml` / `e6a_logit_kd.yaml` / `e6a_logit_attention_kd.yaml`, and
> `check_pilot_gate.py`, per module spec §1–3.
>
> The repo has no training code at all — this is the first. Follow §1.6's hyperparameters
> exactly; they are pre-registered and must not be "improved."
>
> Points that are easy to get wrong and that the tests will catch:
> - Student is **randomly initialised from the TinyStories-8M config**, never
>   `from_pretrained`. The public 8M checkpoint is an independent-convergence reference
>   only.
> - Startup assertions on vocab size, BOS/EOS ids, and head count across teacher,
>   public-8M config, and student config; abort on mismatch; record the results in
>   `run_config.json`.
> - Read `config.attention_layers` and `config.window_size` from both models. GPT-Neo
>   alternates global and local attention, and `L_ATTN` must mask invalid local positions
>   by **exclusion, not renormalisation**, reporting global-mapped and local-mapped layer
>   pairs separately.
> - Step-0 checkpoint saved **before** the first optimiser update — an explicit save call
>   before the loop, not a branch inside it.
> - Three conditions at the same seed must have byte-identical initial weights and an
>   identical block-manifest hash. The design pairs conditions by seed and data order and
>   that pairing has to be provable.
> - Teacher: `eval()`, `requires_grad_(False)`, forward under `no_grad`, bf16.
> - `T**2` rescaling applied exactly once.
> - Save model + optimiser + scheduler + RNG state so `--resume auto` is bit-exact.
> - Write `runtime_estimate.json` after step 100 per `05` §5.
>
> Ship `test_distillation_loss.py`, `test_distillation_init.py`,
> `test_resume_exactness.py`. Add a `--smoke` mode running 5 steps on tiny random configs
> on CPU in fp32 with no downloads.
>
> Then tell me the exact command sequence for the seed-0 pilot to 2,000 steps and what
> `check_pilot_gate.py` will report.

---

## P6 — `train_sentiment_adaptation.py` (E6B)

**Attach:** repo, `03_MODULE_SPEC_e6_transformation.md` (§4–5), output of P5

> Implement WP6: `transformation_inheritance/train_sentiment_adaptation.py`, the four
> `e6b_*.yaml` configs, and `screen_public_pairs.py`, per module spec §4–5.
>
> Five conditions F0–F4 on `distilbert/distilgpt2` with SST-2, three seeds.
>
> Requirements that carry the experiment's validity:
> - Prompt is `{sentence}\nSentiment:` with the **sentence at position 0**. No
>   instruction prefix — a constant prefix would manufacture a shared first-token anchor
>   and invalidate every sink measurement in E6B. Loss on label tokens only, inputs
>   masked to `-100`.
> - Verify at startup how many tokens `" positive"` and `" negative"` produce; if either
>   is multi-token, train and score on the full sequence.
> - `build_corruption_manifest`: exactly 20% flipped, class-stratified within 1,
>   validation never touched, distinct per seed, reproducible within seed.
> - Effective batch size 32 in **both** LoRA and full FT — assert
>   `per_device_batch * grad_accum == 32` at startup for every F-condition.
> - Merge the LoRA adapter to a standalone checkpoint, keep both, and verify merged vs
>   unmerged logits agree within 1e-5 **in fp32 on CPU**. Training is bf16 and irrelevant
>   to that check; say so in `merge_parity.json` so it is not mistaken for a bf16
>   tolerance claim.
> - Only the merged checkpoint is fingerprinted — the frozen GPT-2 harness knows nothing
>   about PEFT wrappers.
> - Record accuracy, NLL, and ECE with 10 equal-width bins at every eval checkpoint.
>
> Ship `test_corruption_manifest.py`, `test_lora_merge_parity.py`,
> `test_effective_batch.py`.
>
> `screen_public_pairs.py` is a thin driver over `compute_fingerprint` for gpt2 vs
> distilgpt2 and Qwen2.5 0.5B/1.5B base vs instruct — no training. It is the cheapest
> evidence in the project; make it runnable standalone in one command.

---

## P7 — `cross_example_patching.py`

**Attach:** repo, `02_MODULE_SPEC_common.md` (§6), output of P3, `06_TEST_PLAN.md`

> Implement WP7: `common/cross_example_patching.py` per module spec §6.
>
> This is the most delicate module in the project. Every causal claim in E7 rests on it.
> Nothing like it exists in the repo: `NNsightEngine` applies edits computed from static
> weights within a single trace, with no source-capture → target-inject path.
>
> Implement `PatchSite`, `PatchSpec`, `SourceCache`, `capture_sources`, `run_patched`,
> and the label scorer.
>
> The nine invariants in §6.2 are non-negotiable and each is a test:
> identity patch is an exact no-op; self-source is an exact no-op; locality (a `K0@L5`
> patch changes nothing below layer 5); Qwen V0 patched **before** `_repeat_kv`; norm
> matching to 1e-6; source and target may differ in length with `*mid` positions resolved
> per example and recorded; no batch leakage; model-fingerprint mismatch raises; shapes
> and dtypes asserted at write time.
>
> Invariants 1 and 2 deserve particular care: a patching implementation that fails them
> can still produce smooth, plausible, entirely fictitious effect sizes. Build them first
> and make them runnable as a continuous check during production runs, not only in the
> test suite.
>
> Scoring: length-normalised sequence log-probability over the candidate's full token
> sequence, implemented **once** and used for both baseline and patched forwards, so a
> scoring bug cannot differentially affect them. Record normalised and unnormalised
> margins.
>
> RoPE: implement both `K0_prerope` and `K0_postrope` as distinct objects per §6.4,
> reusing the Qwen harness's existing `build_rope_cos_sin` / `_apply_rope` / `_rotate_half`.
> Use your numerical finding from P3 about position-0 equivalence, and do not assume it
> holds at `Kmid`.
>
> Failures write a row with `status` and `warning` — never dropped.
>
> Ship `test_cross_example_patching.py`, `test_patch_scoring.py`,
> `tests/nnsight_e7_smoke.py` (random Qwen2 config, `num_key_value_heads=2`,
> `num_attention_heads=8`, so the pre-repeat test is meaningful).

---

## P8 — manifests

**Attach:** `02_MODULE_SPEC_common.md` (§5), `04_MODULE_SPEC_e7_crosslingual.md` (§1–2)

> Implement WP8: `common/paired_manifests.py`,
> `crosslingual_semantics/prepare_flores_manifest.py`, and
> `crosslingual_semantics/prepare_xnli_manifest.py`.
>
> One trap to internalise before writing anything: FLORES `devtest` rows are aligned by
> position within each language config, but **XNLI is not** — it must be joined on
> `promptID`, never on row index. A positional-alignment bug there produces a manifest
> that looks fine and silently destroys every cross-language claim.
> `test_parallel_manifest_alignment.py` catches it by asserting gold labels agree across
> all seven languages for every semantic id; write that test first.
>
> Implement everything in module spec §5.1: `ParallelManifest`, `build_flores_manifest`,
> `build_length_matched_subset` (bipartite matching on token count via
> `scipy.optimize.linear_sum_assignment`, max 20% relative difference),
> `build_xnli_manifest`, `assign_patch_controls`, `grouped_partition`, `verify_manifest`.
>
> `assign_patch_controls` constraints: no id is its own control; `same_label_en` shares
> the gold label; `different_label_en` does not; `random_en` is drawn independently of
> label; fully deterministic from seed 42.
>
> `grouped_partition` splits by **semantic id**, so no sentence appears in two partitions
> in any language. It serves both the 200/400 dev/test split and the 100/200 Procrustes
> split.
>
> FLORES keeps a row only if every one of the eight languages passes the 12–160 token
> filter. Fewer than 200 surviving rows ⇒ `provenance["valid"] = False` and a non-zero
> exit.
>
> Also emit the per-language covariates the analysis needs later: first-token frequency,
> punctuation-at-position-0 indicator, Unicode script, language family.
>
> Ship `test_parallel_manifest_alignment.py`, `test_patch_controls.py`,
> `test_grouped_partition.py`.

---

## P9 — extraction and retrieval

**Attach:** repo, `04_MODULE_SPEC_e7_crosslingual.md` (§3–4, §7), outputs of P3/P8

> Implement WP9: `crosslingual_semantics/extract_sink_representations.py`,
> `evaluate_crosslingual_retrieval.py`, and the four `e7_qwen*.yaml` configs.
>
> Memory discipline is the design constraint here. 300 sentences × 8 languages × 24
> layers × 9 objects, captured naively, is tens of GB and will not fit alongside the
> model on a 16 GB card. Requirements: derived tensors only — **never** persist full
> hidden states or attention maps; fp32 on CPU written incrementally to per-(model,
> language, object) `.npy` memmaps with a `semantic_id → row` index; free per-example
> activations immediately; `batch_size=1`. Ship
> `test_extraction_shapes.py` including an assertion that the output directory stays
> under a size bound, which is what catches accidental full-state capture.
>
> Extract `R0`, `K0_prerope`, `K0_postrope`, `V0`, `Qmean` at every layer, plus the
> controls `Rlast`, `Rmean`, `Rmid`, `Kmid`, `Vmid`.
>
> For retrieval per §4: centre each language separately **before** computing cosine
> similarity — without this, similarity is dominated by the language mean and retrieval
> measures script identity rather than semantics. Report top-1, top-5, MRR, median rank,
> and the ratio to chance (1/300), since the go/no-go bar is "at least twice chance."
> Report raw cosine and orthogonal Procrustes (fitted on 100 held-out ids, evaluated on
> the disjoint 200 — assert non-overlap).
>
> The language-identity probe is grouped 5-fold CV split by semantic id, never by row,
> standardised on training folds only, reporting macro-F1 and balanced accuracy.
>
> Every retrieval table must carry the matched middle-token controls in the same rows.
> A position-0 number without `Kmid`/`Vmid`/`Rmid` alongside it cannot support the claim.
>
> Ship `test_retrieval_chance.py`: random representations give top-1 ≈ 1/300, identical
> representations give exactly 1.0.

---

## P10 — E6 evaluation and aggregation

**Attach:** repo, `03_MODULE_SPEC_e6_transformation.md` (§6–7), `05_SCHEMAS_AND_CONTRACTS.md`, outputs of P2/P4/P5/P6

> Implement WP10: `transformation_inheritance/evaluate_transformation.py` and
> `aggregate_transformation.py` per module spec §6–7, writing the schemas in
> `05_SCHEMAS_AND_CONTRACTS.md` §2 exactly.
>
> **Resumability is a hard requirement, not a nicety.** E6A alone is roughly 1.5–3
> GPU-days on a single 4080 SUPER before analysis, and evaluation is 8 checkpoints × 9
> interventions × 300 blocks × 9 runs plus 2,000-block ΔCE. The unit of work is
> `(run_id, step, corpus_id, intervention)`; check the fingerprint cache and existing CSV
> rows before computing, skip completed units, and make `--resume` the default.
>
> ΔCE follows design-delta D3: a fixed 300-block subset at every checkpoint for
> trajectories, the full 2,000 blocks only at step 0 and final. Report both and bootstrap
> the subset's sampling error.
>
> Matched-loss selection lives in the aggregator, not the evaluator: for each condition
> pick the **earliest** checkpoint whose validation CE is closest to D0's final CE, record
> the realised CE gap, and refuse the comparison if the gap exceeds 0.05 nats — flag it
> rather than silently comparing mismatched losses. The same logic gives E6B's
> matched-accuracy comparison.
>
> All cross-model comparisons use `mutual_interventions(teacher, student)` and assert
> matching `manifest_sha256`, `intervention_registry_version`, and `n_keys_used`.
>
> Produce every output in §7's table, including `go_no_go.json` with design §18's
> criteria transcribed as literal code. Write those criteria **before** any results exist;
> that is the pre-registration.
>
> Statistics via `inheritance_metrics`: paired seed contrasts carrying the
> `min_attainable_p` note, bootstrap CIs explicitly labelled as sampling uncertainty and
> never as model-training uncertainty, BH correction within each pre-registered family.
>
> Import plot style helpers from `emergence_dynamics_analysis.py` rather than restyling.

---

## P11 — E7 patching and aggregation

**Attach:** repo, `04_MODULE_SPEC_e7_crosslingual.md` (§5–6), `05_SCHEMAS_AND_CONTRACTS.md`, outputs of P7/P8/P9

> Implement WP11: `crosslingual_semantics/run_cross_language_patching.py` and
> `aggregate_crosslingual.py`, writing `patching_per_example.csv` exactly per
> `05_SCHEMAS_AND_CONTRACTS.md` §3.
>
> Run unit = (model, target language, patch object, layer, norm condition). One CSV row
> per (target example, source condition, object, layer, norm condition). Capture English
> source activations once and cache them keyed by model fingerprint.
>
> Layer protocol, which is the entire defence against sweep overfitting: screen layers
> 5/11/17/22 (24-layer) or 6/13/20/26 (28-layer) on the **200 dev** examples, select the
> layer with the largest parallel-vs-unrelated effect, then evaluate the contiguous
> five-layer window centred on it on the **disjoint 400 test** examples. Write
> `layer_selection.json` recording the dev effect for every screened layer, the selection,
> and the resulting window. Make the dev/test separation auditable from the artefacts
> alone.
>
> All four norm conditions in every patch family. `identity` doubles as a continuous
> correctness monitor: if any identity row's margin change exceeds 1e-5, **abort the run
> unit and mark it invalid** — do not merely warn.
>
> Non-finite logits, OOM, and tokenisation failures write rows with `status` set, never
> dropped. Aggregation reports failure counts per condition and refuses any contrast
> whose failure rate exceeds 2% without an explicit override flag.
>
> Aggregation produces the six contrasts of §6.1, paired **within target example** so
> example difficulty cancels; hierarchical bootstrap (resample semantic ids, then
> languages, preserving all patch conditions within a selected id); BH correction within
> each pre-registered family; pre-selected object contrasts in a **separate file** from
> exploratory layer sweeps.
>
> `interpretation_matrix.json` maps the observed pattern onto exactly one row of design
> §10.12, or states that it matches none. `e7_go_no_go.json` evaluates design §18's four
> criteria plus the §10.11 claim gate (same direction in ≥2 model sizes and ≥5 of 7
> languages) and emits a single `supported: true|false`.
>
> Also implement the Phase 1 smoke study driver: 50 XNLI examples × 4 screening layers ×
> {K0, V0} × 4 source conditions × 4 norm conditions on Qwen2.5-0.5B, writing
> `runtime_estimate.json`. That runs before anything larger.

---

## Two prompts to run between packages

**Gate check** (after P0–P4, P7, P8 — i.e. all of Phase 0):

> Run the Phase 0 gate from `06_TEST_PLAN.md` §2: every listed test, plus the design's
> §14.5 real-model fidelity checks on five examples per model. Write `phase0_gate.json`
> with a single `passed` boolean and a per-check breakdown. If anything fails, diagnose
> the root cause before proposing a fix — do not loosen a tolerance to make a test pass,
> and if a tolerance genuinely needs revisiting, say so explicitly and justify it.

**Reviewer pass** (after each of P5, P7, P10, P11):

> Review the module you just wrote as an ACL/ICML reviewer would review the
> corresponding experimental claim. Specifically: which results produced by this code
> could a reviewer attribute to an implementation artefact rather than the phenomenon?
> For each, state whether the current code rules it out, and if not, what the minimal
> additional control would be. Be concrete about failure modes that produce
> plausible-looking numbers rather than crashes — those are the ones that survive to
> publication.
