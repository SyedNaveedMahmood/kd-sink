# 06 — Test Plan and Parity Gates

Every test is owned by a work package. A WP is not complete until its tests pass.
No full experiment starts until the Phase 0 gate is green (design §14).

---

## 1. Test inventory

| Test file | WP | What it proves |
|---|---|---|
| `test_frozen_files.py` | WP0 | The twelve frozen files are byte-identical to `frozen-e1-e5`. |
| `test_depth_band.py` | WP1 | Band never empty; `num_layers=12` coincides with `compute_band(12,"scaled")`; monotone in `num_layers`; `band_agreement_report` flags the 4L/8L pair correctly. |
| `test_corpus_determinism.py` | WP1 | Each provider yields an identical `manifest_sha256` twice in-process and once in a subprocess. |
| `test_corpus_e1_bridge.py` | WP1 | `frozen_e1_corpus` reproduces the checked-in E1 `sample_manifest.csv` row-for-row. |
| `test_tinystories_packing.py` | WP1 | Blocks are exactly `block_size`; one EOS between stories; block manifest maps to real story indices; sink and ppl corpora are disjoint. |
| `test_sst2_prompt_format.py` | WP1 | Sentence starts at position 0; no prefix; label token counts recorded; loss mask covers label tokens only. |
| `test_fingerprint_runner_bridge.py` | WP2 | `compute_fingerprint(gpt2, frozen_e1_corpus)` reproduces frozen `bos_attention_stats_overall.csv` within `METRIC_ATOL`. |
| `test_available_interventions.py` | WP2 | `int_b` excluded for Neo (`q_proj.bias is None`); `int_e` excluded for Qwen (`wpe_path is None`); `mutual_interventions` preserves `INTERVENTION_ORDER`. |
| `test_fingerprint_cache.py` | WP2 | Cache hit on identical key; recompute on any key field change; cached and fresh records equal. |
| `test_run_intervention_parity.py` | WP3 | Post-`_apply_edits`-refactor GPT-2 and Neo fingerprints match pre-refactor values within `METRIC_ATOL`/`METRIC_RTOL`. |
| `test_arch_trace_ce.py` | WP3 | `run_arch_trace(capture_token_ce=True)` agrees with a direct HF forward + `token_cross_entropy` to `1e-6` fp32, on gpt2/neo/qwen/opt random models. |
| `test_archspec_vpaths.py` | WP3 | Every `v_path`/`o_path` resolves on a random model of that arch; K/V trailing dims equal `num_kv_heads * head_dim`. |
| `test_inheritance_metrics.py` | WP4 | Identity ⇒ cosine 1, L1 0, Wasserstein 0, Jaccard 1. NaN keys dropped and reported. `weighted_jaccard` raises on unequal head counts. `interp_depth_profile` handles `L=1`. Permutation test returns `min_attainable_p=0.25` at n=3. |
| `test_distillation_loss.py` | WP5 | KD zero when logits identical; attention loss zero when mapped maps identical; masked local positions excluded not renormalised; `T**2` applied exactly once; teacher tensors have `requires_grad=False` and no grad reaches teacher params. |
| `test_distillation_init.py` | WP5 | D0/D1/D2 at the same seed have byte-identical initial state dicts and identical `block_manifest_sha256`. |
| `test_resume_exactness.py` | WP5 | Resume from step 10 reproduces the uninterrupted step-20 weights bit-for-bit. |
| `test_corruption_manifest.py` | WP6 | Exactly 20% flipped; class-stratified within 1; validation untouched; distinct across seeds; reproducible within seed. |
| `test_lora_merge_parity.py` | WP6 | Merged vs unmerged logits agree < 1e-5 in fp32 CPU on five parity sentences. |
| `test_effective_batch.py` | WP6 | `per_device_batch * grad_accum == 32` for every F-condition config. |
| `test_cross_example_patching.py` | WP7 | The nine invariants of `02` §6.2. |
| `test_patch_scoring.py` | WP7 | Length-normalised sequence log-prob matches a hand-computed value on a toy vocab; margin sign convention correct; baseline and patched use the same scorer. |
| `test_parallel_manifest_alignment.py` | WP8 | XNLI joined on `promptID` not row index; gold labels agree across all seven languages for every semantic id; FLORES rows share a source index. |
| `test_patch_controls.py` | WP8 | No id is its own control; `same_label_en` matches label; `different_label_en` differs; deterministic from seed; every (semantic_id, lang) has all four conditions. |
| `test_grouped_partition.py` | WP8 | Dev/test disjoint by semantic id in every language; Procrustes train/test disjoint; sizes 200/400 and 100/200 as specified. |
| `test_extraction_shapes.py` | WP9 | Object shapes per layer; fp32 CPU; no full hidden-state file written (assert output dir size below a bound); index maps semantic_id → row correctly. |
| `test_retrieval_chance.py` | WP9 | Random representations give top-1 ≈ 1/300; identical representations give top-1 = 1.0; per-language centring applied before similarity. |
| `nnsight_e6_smoke.py` | WP3/WP5 | Offline random gpt2/neo/opt models: fingerprint + `run_arch_trace` + CE, CPU fp32, no downloads. |
| `nnsight_e7_smoke.py` | WP7/WP9 | Offline random Qwen2 config with `num_key_value_heads=2, num_attention_heads=8`: extraction, all patch objects, all norm conditions. |

Follow the existing `tests/nnsight_smoke_utils.py` pattern for every random-model test —
same construction helper, same CPU/fp32 discipline, no network.

---

## 2. Phase 0 gate — must all pass before any real-model experiment

1. `test_frozen_files.py`
2. All existing tests: `test_crfm_attention_config_parity.py`,
   `test_e4_parity_tolerances.py`, `nnsight_e3_smoke.py`, `nnsight_e4_smoke.py`,
   `nnsight_e5_smoke.py`
3. `test_run_intervention_parity.py`
4. `test_fingerprint_runner_bridge.py`
5. `test_cross_example_patching.py` (all nine invariants)
6. `nnsight_e6_smoke.py`, `nnsight_e7_smoke.py`
7. Every other test above

Plus the design's §14.5 real-model fidelity checks, on five examples per model:

- HF eager-attention baseline vs NNsight capture;
- existing Qwen intervention output vs the patching runner with **no patch applied**;
- task-label log-probs from the patch scorer vs a direct HF forward.

Write all of it to `phase0_gate.json` with a single `passed: true|false`.

---

## 3. Three tests that carry unusual weight

**`test_fingerprint_runner_bridge.py`.** The entire refactor rests on the claim that the
new programmatic path computes the same thing as the frozen CLI path. If this drifts,
every E6/E7 number is measured with a subtly different instrument than E1–E5, and the
paper's "reused validated infrastructure" claim is false. Run it in CI on every commit.

**`test_cross_example_patching.py` invariants 1 and 2.** Identity patch and self-source
patch must be exact no-ops. A patching implementation that fails these can still produce
smooth, plausible, entirely fictitious effect sizes. Run the identity control
*continuously during production runs* too (`04` §5.4), not only in the test suite.

**`test_distillation_loss.py` masking.** If local-attention masking renormalises instead
of excluding, `L_ATTN` optimises a quantity that does not exist in the teacher, and H2
is tested against an artefact. Verify by constructing a teacher/student pair whose
mapped maps agree on valid positions and differ on invalid ones — the loss must be
exactly zero.

---

## 4. Continuous checks during production runs

| Check | Where | Action on failure |
|---|---|---|
| identity patch ≈ baseline (1e-5) | every E7 run unit | abort the unit, mark run invalid |
| `raw["int_a"] > 1e-8` | every fingerprint | record failure, set `r_j = nan` |
| non-finite logits | every forward | write row with `status="nonfinite"` |
| failure rate ≤ 2% per condition | aggregation | refuse contrast without override |
| `manifest_sha256` match | every comparison | raise |
| `intervention_registry_version` match | every comparison | raise |
| band depth mismatch ≤ 0.10 | every cross-model comparison | raise unless `allow_band_mismatch` |
| VRAM headroom | training loop | log peak; fail fast rather than thrash |

---

## 5. What is deliberately not tested

Numerical *values* of E6/E7 results have no expected values — that is the point of the
experiment. Tests assert **invariants, determinism, and agreement with the frozen
instrument**, never that a scientific quantity takes a particular value. Do not write a
test asserting, say, that D2's fingerprint cosine exceeds D0's; that is the hypothesis,
and encoding it as a test would make a null result look like a bug.
