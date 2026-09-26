# 04 — Module Spec: `crosslingual_semantics/` (E7)

Depends on `paired_manifests` (WP8), `cross_example_patching` (WP7), and the
`ArchSpec` additions (WP3). All models are Qwen2.5; all execution is NNsight with
eager attention.

---

## 1. `prepare_flores_manifest.py` (WP8)

```bash
python crosslingual_semantics/prepare_flores_manifest.py \
  --tokenizer Qwen/Qwen2.5-0.5B --n 300 --seed 42 \
  --out crosslingual_semantics/results/manifests/flores_devtest.json
```

Languages: `eng_Latn ben_Beng zho_Hans arb_Arab deu_Latn hin_Deva swh_Latn tur_Latn`.

Steps: join `facebook/flores` `devtest` by source index → tokenise without special
tokens → keep rows where **every** language is 12–160 tokens → sample 300 with seed 42
→ record raw text, token ids, token count, first token id, Unicode script, source index.

Also emits the length-matched secondary subset via `build_length_matched_subset`
(design §10.3) and per-language covariates used later: first-token frequency (from a
corpus count over the full devtest split), a punctuation-at-position-0 indicator
(`unicodedata.category(ch).startswith("P")` on the first character of the first token's
decoded string), script, and language family.

Validity: fewer than 200 surviving rows ⇒ `provenance["valid"] = False` and the script
exits non-zero. Do not proceed on an invalid manifest.

**Tokenizer coupling.** The manifest is tokenizer-specific. 0.5B and 1.5B share the
Qwen2.5 tokenizer — assert this rather than assuming it, and if a model in the sweep
differs, build a separate manifest and never join across them.

---

## 2. `prepare_xnli_manifest.py` (WP8)

Languages: `en zh ar de hi sw tr`. 600 aligned semantic ids, balanced across
entailment / neutral / contradiction, seed 42, after tokenisation and length filtering.

Prompt (design §10.4), premise at position 0:

```
{premise}
{hypothesis}
Question: Is the second statement entailed by, neutral with respect to, or contradictory to the first?
Answer:
```

Candidates: `" entailment"`, `" neutral"`, `" contradiction"`, scored by
length-normalised sequence log-probability. No free-form generation.

Emits: the manifest, `patch_controls.csv` from `assign_patch_controls`, and the grouped
200 dev / 400 test partition. Then runs `verify_manifest` and refuses to write on
failure.

**The alignment trap (repeat from `02` §5.2):** join on XNLI's `promptID`, never on row
index. The manifest test asserts gold labels agree across all seven languages for every
semantic id.

**Prompt-language confound.** The instruction is English in all conditions (design
§16.3). This is a deliberate constant, but it means position 0 is a target-language
premise while the task words are English. Implement `--translated-instruction` as a
robustness switch that swaps in translated task wording for a named subset of languages,
and run it on at least two languages as the design's prescribed check.

---

## 3. `extract_sink_representations.py` (WP9)

### 3.1 Objects

At **every** layer, position 0:

| key | definition |
|---|---|
| `R0` | residual stream after the decoder layer |
| `K0` | key at position 0, pre-KV-repeat — both `K0_prerope` and `K0_postrope` (`02` §6.4) |
| `V0` | value at position 0, pre-KV-repeat |
| `Qmean` | mean query over second-half source positions |

Controls: `Rlast` (final content token), `Rmean` (mean over content tokens), `Rmid`,
`Kmid`, `Vmid`.

### 3.2 Memory discipline

This is the constraint that shapes the module. 300 sentences × 8 languages × 24 layers ×
9 objects, naive, is tens of GB. Required behaviour:

- Derived tensors only. **Never** persist full hidden states or attention maps for the
  dataset. The design says so explicitly and the 16 GB budget enforces it.
- Store fp32 on CPU, written incrementally to per-(model, language, object) `.npy`
  memmaps with an accompanying index of `semantic_id → row`.
- Free per-example activations immediately; run with `batch_size=1` and an explicit
  `torch.cuda.empty_cache()` cadence.
- Rough size: K0/V0 for 0.5B are `[n_kv_heads=2, head_dim=64]` = 128 floats/layer;
  R0 is 896 floats/layer. Full set per model per language ≈ 300 × 24 × ~2.5k floats
  ≈ 72 MB fp32. Tractable; the failure mode is only ever accidental full-state capture.

### 3.3 Interface

```bash
python crosslingual_semantics/extract_sink_representations.py \
  --config crosslingual_semantics/configs/e7_qwen05_base.yaml \
  --manifest .../flores_devtest.json --langs all --out .../reps/
```

Writes `reps/<model_tag>/<lang>/<object>.npy` `[n_sentences, n_layers, dim]` plus
`index.json` and `run_config.json` including dtype — design §10.2 warns that
intervention percentages are precision-sensitive, so dtype is a first-class field.

### 3.4 Baseline multilingual fingerprint

The same script (or a sibling `run_multilingual_fingerprint.py`) runs the full Qwen
intervention battery per language via `compute_fingerprint`, plus the controls in
design §10.5: same semantic ids across languages, unmatched random ids, the
length-controlled subset, first-token frequency covariate, punctuation indicator,
script/family labels, base vs instruct.

---

## 4. `evaluate_crosslingual_retrieval.py` (WP9)

For each (model, source lang, target lang, layer, object):

1. Centre each language's representations **separately** (this removes the language
   mean, which otherwise dominates cosine similarity and would produce retrieval driven
   entirely by script identity).
2. Cosine similarity, retrieve matching semantic id among all 300 candidates.
3. Report top-1, top-5, MRR, median rank.
4. Chance baseline is 1/300 for top-1; report the ratio to chance since design §18 sets
   the bar at "at least twice chance."

Alignment variants: raw cosine, and orthogonal Procrustes fitted on 100 held-out
semantic ids and evaluated on the disjoint 200 (`scipy.linalg.orthogonal_procrustes`).
Train/test ids come from `grouped_partition` and **must never overlap** — assert it.

Language-identity probe: multinomial logistic regression per object, grouped 5-fold CV
split by semantic id (never by row), standardised on training folds only, reporting
macro-F1 and balanced accuracy. Interpretation follows design §10.8.

Outputs: `retrieval_by_pair.csv`, `retrieval_by_object.csv`, `language_probe.csv`,
`fig5_retrieval_heatmap.pdf`.

**Report the middle-token control alongside every position-0 number.** Design §18
requires position-0 to beat the matched middle-token control by ≥5 points; a retrieval
table without `Kmid`/`Vmid`/`Rmid` in the same rows cannot support the claim.

---

## 5. `run_cross_language_patching.py` (WP11)

### 5.1 Unit of work

One run unit = (model, target language, patch object, layer, norm condition). Within a
unit, iterate target examples × four source conditions. One CSV row per
(target example, source condition, object, layer, norm condition) — schema in `05` §3.

### 5.2 Procedure

1. Load the model handle and the XNLI manifest + `patch_controls.csv`.
2. `capture_sources` for all needed English source examples at the required sites, once,
   cached to disk keyed by model fingerprint.
3. For each target example: baseline forward (scores all three labels), then each patch
   spec via `run_patched`.
4. Record baseline and patched: gold log-prob, margin, prediction, XNLI accuracy
   contribution, JSD, position-0 attention, attention to positions 1–4, carrier-head
   sink change, status, warning.

### 5.3 Layer screening then window

Screen layers 5, 11, 17, 22 (24-layer) or 6, 13, 20, 26 (28-layer) on the **200 dev
examples**. Select the layer with the largest parallel-vs-unrelated effect. Evaluate the
contiguous five-layer window centred on it, clamped at boundaries, on the **disjoint
400 test examples**.

The selection step must write `layer_selection.json` recording the dev-set effect for
every screened layer, the selected layer, and the resulting window. Selecting on dev and
reporting on test is the entire defence against layer-sweep overfitting; make the
separation auditable.

### 5.4 Norm conditions

Every patch family includes all four: `direct`, `rescaled` (source rescaled to target
norm), `random` (norm-matched Gaussian), `identity` (target's own vector restored).
`identity` doubles as a continuous correctness monitor — if any identity row shows a
non-zero margin change beyond 1e-5, the run is invalid and must abort, not merely warn.

### 5.5 Phase 1 smoke study

50 XNLI examples × 4 screening layers × {K0, V0} × 4 source conditions × 4 norm
conditions on Qwen2.5-0.5B, before anything larger. Confirms the pipeline end-to-end
and produces the first `runtime_estimate.json` for E7.

---

## 6. `aggregate_crosslingual.py` (WP11)

### 6.1 Contrasts (design §10.11)

1. parallel-EN K0 vs unrelated same-label K0
2. parallel-EN V0 vs unrelated same-label V0
3. K0 vs V0 sensitivity
4. position-0 vs middle-token
5. base vs instruct
6. high-resource vs lower-resource

Primary outcome is paired difference in `CorrectLabelMargin` between parallel and control
patches, paired **within target example** so example difficulty cancels.

### 6.2 Statistics

Hierarchical bootstrap (design §15.3): resample semantic ids, then languages, preserving
all patch conditions for a selected semantic id. BH correction within each pre-registered
family of layer × object comparisons. Pre-selected object-level contrasts are reported
separately from exploratory layer sweeps — separate tables, separate files, no mixing.

Every reported effect carries: raw values, absolute difference, relative difference where
meaningful, 95% CI, language consistency count, and corrected + uncorrected p.

### 6.3 Outputs

| File | Contents |
|---|---|
| `patching_contrasts.csv` | the six contrasts per language and model |
| `table4_patch_effects.csv` | design §19 Table 4 |
| `fig6_patch_decomposition.pdf` | parallel vs unrelated for K0/V0/R0 |
| `interpretation_matrix.json` | design §10.12 evaluated against observed results |
| `e7_go_no_go.json` | design §18 criteria, mechanical |
| `failures.csv` | every non-`ok` row, counted per condition |

`interpretation_matrix.json` should map the observed pattern onto exactly one of the six
rows of design §10.12, or state that the result matches none of them. Writing that
mapping as code, before the results exist, keeps the interpretation honest.

### 6.4 The claim gate

Design §10.11: the main claim requires the same direction in ≥2 model sizes and ≥5 of 7
XNLI languages. `e7_go_no_go.json` computes this and the four §18 criteria and emits a
single `supported: true|false`. Nothing goes in the paper's main claims if this is false.

---

## 7. `configs/e7_qwen*.yaml`

```yaml
experiment_id: e7
model: Qwen/Qwen2.5-0.5B
model_revision: null
tag: qwen05_base
dtype: float32            # bfloat16 for 1.5B and 3B
attn_implementation: eager
engine: nnsight
band_frac: [0.25, 0.90]

flores:
  manifest: results/manifests/flores_devtest.json
  langs: [eng_Latn, ben_Beng, zho_Hans, arb_Arab, deu_Latn, hin_Deva, swh_Latn, tur_Latn]

xnli:
  manifest: results/manifests/xnli_test.json
  langs: [en, zh, ar, de, hi, sw, tr]
  dev_n: 200
  test_n: 400

extraction:
  objects: [R0, K0_prerope, K0_postrope, V0, Qmean, Rlast, Rmean, Rmid, Kmid, Vmid]
  layers: all
  batch_size: 1

patching:
  objects: [K0_prerope, V0, R0, Kmid, Vmid, Rmid]
  screen_layers: [5, 11, 17, 22]
  window: 5
  norm_conditions: [direct, rescaled, random, identity]
  source_conditions: [parallel_en, same_label_en, different_label_en, random_en]
  batch_size: 1

seed: 42
```

Record `dtype` in every output row. A 0.5B fp32 result and a 1.5B bf16 result are not
directly comparable in absolute intervention percentage; only the **direction** of the
parallel-vs-control contrast replicates across them, and that is what design §10.11
actually asks for.
