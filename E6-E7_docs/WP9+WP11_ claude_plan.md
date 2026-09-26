# WP9 + WP11 — E7 cross-lingual extraction, retrieval, patching, aggregation

## Context

`MechanisticAccountofSinks` is adding E6 (sink inheritance under distillation) and E7
(cross-lingual semantic transport through the sink) to a frozen E1–E5 paper codebase.
Per [NEXT_STEPS.md](NEXT_STEPS.md), the E6 half is code-complete (WP0–WP5, WP10) and the
E7 half is half-built: WP7 (`common/cross_example_patching.py`) and WP8
(`common/paired_manifests.py` + the two `prepare_*_manifest.py` scripts) shipped, but
nothing yet **uses** them. WP9 and WP11 are the two remaining packages on the
minimum-publishable path.

Today `crosslingual_semantics/` holds only the two manifest builders and an empty
`configs/`. After this change it holds the full E7 pipeline: extract representations →
measure retrieval → run cross-language patching under a dev/test layer protocol →
aggregate into pre-registered contrasts and a go/no-go verdict. The whole thing must be
runnable offline on random tiny models (smoke) and resumable on the compute PC, because
E7 is a multi-GPU-day run on a 16 GB card.

Three decisions confirmed with the user up front:
1. Design §18 (four E7 criteria) and §10.12 (six interpretation-matrix rows) are **not in
   this repo**. Follow the WP10 precedent — a new `e7_preregistration.yaml` with `source:`
   on what the spec pack quotes and `PENDING_DESIGN_*` sentinels on the rest, forcing
   `decision: "incomplete"`. No threshold is invented (CLAUDE.md rule 4).
2. `run_multilingual_fingerprint.py` (spec `04` §3.4) ships as a thin sibling.
3. `Qmean`/`Rlast`/`Rmean` are added to `cross_example_patching` as **capture-only**
   objects, strictly additively, so extraction has one trace path.

---

## Constraints that shape the code

- **Frozen files.** The twelve in `BASELINE_HASHES.json` are untouched; new code calls in.
- **`nnsight_engine.py` / `datasets_loader.py` stay additive-only.** Neither needs editing
  here — `run_arch_trace(capture_position0=...)` still raises, and
  `tests/test_arch_trace_ce.py` asserts that. `cross_example_patching` supersedes it.
- **Memory (`04` §3.2).** 300 sentences × 8 languages × 24 layers × 10 objects captured
  naively is tens of GB. Derived tensors only, fp32 CPU, incremental `.npy` memmaps,
  `batch_size=1`, per-example tensors freed immediately.
- **Identity patch is a production monitor (`04` §5.4).** `run_patched` already sets
  `status="identity_violation"` past `IDENTITY_TOLERANCE`; WP11 must **abort the run unit**,
  not merely record it.
- **Failures are rows**, provenance on every artefact, thresholds only in YAML.
- Windows-safe teardown in every test that builds a model: `TemporaryDirectory(
  ignore_cleanup_errors=True)` + `try/finally: del handle; gc.collect()`.

---

## Reuse map (do not reimplement any of this)

| Need | Existing API |
|---|---|
| capture activations at a site | `cross_example_patching.capture_sources(handle, corpus, sites)` |
| patch + measure | `cross_example_patching.run_patched(...)` → rows already ≈ `05` §3 |
| patch objects / norms / tolerances | `PATCH_OBJECTS`, `NORM_CONDITIONS`, `IDENTITY_TOLERANCE` |
| RoPE at position 0 | `rope_position0_deviation`, `apply_rope_forward/_inverse` |
| model identity | `cross_example_patching.model_fingerprint(handle)`, `geometry(handle)` |
| load Qwen | `fingerprint_runner.load_handle("qwen", ..., engine="nnsight")` |
| per-language corpus | `corpus_providers.flores_corpus` / `xnli_prompt_corpus` (require `manifest=`) |
| manifest + splits | `paired_manifests.ParallelManifest.load`, `grouped_partition`, `verify_manifest` |
| band | `depth_band.normalised_depth_band(num_layers)` |
| stats | `inheritance_metrics.hierarchical_bootstrap` (`group_cols=("semantic_id","language")`), `bh_correct`, `bootstrap_ci` |
| provenance | `provenance.provenance_block()`, `write_json`, `git_sha`, `utc_now` |
| tiny models / fake datasets | `tests/nnsight_smoke_utils.create_tiny_qwen_checkpoint`, `tests/wp8_fake_data` |

Script conventions come from [evaluate_transformation.py](transformation_inheritance/evaluate_transformation.py)
and [aggregate_transformation.py](transformation_inheritance/aggregate_transformation.py):
`_REPO`/`common` sys.path loop with `# noqa: E402`; heavy imports (`pandas`, `yaml`,
`matplotlib`, `transformers`) inside the function that needs them; a
`<SCRIPT>_VERSION` constant stamped on every row; `build_parser()` / `main(argv=None)`;
figures via `_figure_setup()` with `Agg`, `dpi=200, bbox_inches="tight"`.

---

## Part 0 — additive extension to `common/cross_example_patching.py`

Two strictly-additive changes; every existing default is unchanged and all nine WP7
invariant tests must stay green.

1. **`CAPTURE_ONLY_OBJECTS = ("Qmean", "Rlast", "Rmean")`** plus multi-position support in
   the internal `_Capture` path: `_sites_to_captures` resolves a *tuple* of positions and a
   reduction (`mean` / `last`), `_shape_capture` reduces inside the trace. `Qmean` reads
   `spec.q_path`; `Rlast`/`Rmean` read the block output. `_make_write` raises a clear
   `ValueError` if one is used as a patch target — they are measurement objects only.
2. **`capture_sources(..., on_item=None, keep=True)`** — when `on_item` is supplied the
   `SourceCache` is handed to the callback and, with `keep=False`, dropped instead of
   accumulated. This is what lets extraction stream 300 sentences to a memmap while paying
   `model_fingerprint()`'s weight digest **once** per call rather than once per item.

---

## Part 1 — WP9

### `crosslingual_semantics/configs/e7_qwen{05,15}_{base,instruct}.yaml`
Four configs, shape verbatim from `04` §7. `dtype: float32` for 0.5B, `bfloat16` for 1.5B;
`screen_layers: [5,11,17,22]` for the 24-layer 0.5B and `[6,13,20,26]` for the 28-layer
1.5B (both figures are given in `04` §5.3). `attn_implementation: eager`, `engine: nnsight`,
`band_frac: [0.25, 0.90]`, `seed: 42`.

### `crosslingual_semantics/extract_sink_representations.py`
```
--config <e7_*.yaml> --manifest <flores_devtest.json> --langs all --out <reps/> [--resume] [--smoke]
```
- Load config + `ParallelManifest.load` (re-verifies its sha256). **Assert** the manifest's
  tokenizer identity matches the model's (`04` §1 tokenizer coupling) — refuse, don't warn.
- Per language: `flores_corpus(tokenizer, lang, manifest.semantic_ids, manifest=manifest)`,
  then one `capture_sources(..., on_item=writer, keep=False)` pass over all ten objects ×
  all layers.
- Writer opens `np.lib.format.open_memmap` per `(model_tag, lang, object)` →
  `[n_sentences, n_layers, dim]` fp32, fills the row, flushes, frees. K/V rows flatten
  `[n_kv_heads, head_dim]`; the shape is recorded in `index.json` so retrieval never guesses.
- Emits `reps/<model_tag>/<lang>/<object>.npy`, `index.json` (`semantic_id → row`, plus
  object shapes and layer count) and `run_config.json` with `dtype` first-class (`04` §3.3).
- Resumable at `(model_tag, lang, object)` via an `extraction_units.jsonl` ledger keyed on
  `manifest_sha256` + `model_fingerprint` + object set, mirroring `eval_units.jsonl`.
- `--smoke`: tiny random Qwen2 + a synthetic manifest, CPU/fp32, no downloads.

### `crosslingual_semantics/evaluate_crosslingual_retrieval.py`
Per `04` §4. For each (model, source lang, target lang, layer, object):
- **Centre each language separately before cosine** — without this, retrieval measures
  script identity. Asserted by a test, not just commented.
- top-1, top-5, MRR, median rank, and ratio-to-chance (chance = 1/n_ids).
- Two alignment variants: raw cosine, and orthogonal Procrustes
  (`scipy.linalg.orthogonal_procrustes`) fitted on the manifest's `procrustes_train`
  partition and evaluated on `procrustes_test` — **assert non-overlap** before fitting.
- Language-identity probe: sklearn `LogisticRegression` + `GroupKFold(5)` grouped by
  semantic id (never by row), `StandardScaler` fit on training folds only; macro-F1 and
  balanced accuracy.
- Outputs `retrieval_by_pair.csv`, `retrieval_by_object.csv`, `language_probe.csv`,
  `fig5_retrieval_heatmap.pdf`. `retrieval_by_object.csv` carries the matched middle-token
  control (`Kmid*`/`Vmid`/`Rmid`) in the same rows plus an explicit `position0_minus_mid`
  column, so §18's "≥5 points" bar is computable from one table (`04` §4, last paragraph).

### `crosslingual_semantics/run_multilingual_fingerprint.py`
Thin driver over `compute_fingerprint` — the full Qwen battery per language →
`results/fingerprints/<model_tag>/<lang>.json`, with the `04` §3.4 / design §10.5 control
variants selectable (`--variant matched|unmatched|length_matched`) and the per-language
covariates the manifest already carries (first-token frequency, punctuation-at-0, script,
family) copied into the record's provenance. No new measurement machinery.

### Tests
- `tests/test_extraction_shapes.py` — per-object shapes and layer count; fp32 CPU;
  `index.json` maps `semantic_id → row` correctly; **output directory stays under a size
  bound** (the assertion that catches accidental full-state capture, `06` §1).
- `tests/test_retrieval_chance.py` — random representations give top-1 ≈ 1/n; identical
  representations give exactly 1.0; per-language centring is applied before similarity;
  Procrustes train/test ids are disjoint.
- `tests/test_capture_only_objects.py` — the three new objects capture at the right shapes,
  reduce over the right positions, and **raise** when used as a patch target.

---

## Part 2 — WP11

### `crosslingual_semantics/configs/e7_preregistration.yaml`
Modelled line-for-line on [e6_preregistration.yaml](transformation_inheritance/configs/e6_preregistration.yaml),
including its header explaining the sourced-vs-PENDING rule.
- `contrasts:` — the six of `04` §6.1, each with `source: "04 §6.1"` (they are enumerated
  verbatim there): parallel-vs-unrelated K0, parallel-vs-unrelated V0, K0-vs-V0,
  position-0-vs-middle, base-vs-instruct, high-vs-lower-resource.
- `go_no_go:` — `e7_1` (retrieval ≥ 2× chance) and `e7_2` (position-0 beats matched
  middle-token by ≥ 5 points) sourced from `04` §4; `e7_claim_gate` (same direction in ≥ 2
  model sizes and ≥ 5 of 7 XNLI languages) sourced from `04` §6.4; `e7_3`/`e7_4` as
  `status: PENDING_DESIGN_18`.
- `interpretation_matrix:` — `status: PENDING_DESIGN_10_12`, `rows: []`.
- `analysis:` — `max_failure_rate: 0.02`, `bootstrap_n: 10000`, `bh_alpha: 0.05`,
  `identity_tolerance` mirrored from the module constant. `amendments: []`.

### `crosslingual_semantics/run_cross_language_patching.py`
```
--config <e7_*.yaml> --manifest <xnli_test.json> --controls <patch_controls.csv> \
--stage {smoke,screen,window} --out <patching/<model_tag>/> [--resume]
```
- **Run unit = (model, target language, patch object, layer, norm condition)** (`04` §5.1),
  tracked in a `patch_units.jsonl` ledger keyed on `manifest_sha256`, `model_fingerprint`,
  `patch_registry_version` and the source-condition set — same discipline as
  `eval_units.jsonl`.
- English source activations captured **once** per (model, site set) and cached to disk
  keyed by `model_fingerprint`; refused and recaptured on mismatch.
- Per target example: `run_patched(..., score_candidates=meta["label_candidate_ids"],
  gold_index=..., capture_sink=True, band=normalised_depth_band(L))`.
- One CSV row per (target example, source condition, object, layer, norm condition) →
  `patching_per_example.csv`, **exactly** the `05` §3 columns, plus an added tail
  (`run_unit_id`, `unit_status`, `partition`, `stage`, `runner_version`, `measured_utc`) —
  adding is permitted, renaming is not (`05` §5).
- **Identity abort.** Any row with `status="identity_violation"` marks the whole run unit
  `unit_status="invalid"`; every row of that unit is written with it, the unit is recorded
  in `invalid_units.json`, aggregation excludes it, and the process exits non-zero. The
  identity condition is scheduled **first** in each unit so an invalid unit aborts before
  paying for the rest.
- **Layer protocol (`04` §5.3), the whole defence against sweep overfitting.**
  `--stage screen` runs `config.patching.screen_layers` on the **dev** partition only and
  writes `layer_selection.json` with the dev effect for every screened layer, the selection
  rule, the selected layer and the resulting clamped five-layer window. `--stage window`
  refuses to start without that file, and refuses to score a test number from a dev id.
  The dev/test separation is auditable from the artefacts alone.
- `runtime_estimate.json` (`05` §5) after 50 patch examples.
- `--stage smoke` = the `04` §5.5 Phase 1 study: 50 XNLI examples × 4 screening layers ×
  {`K0_prerope`, `V0`} × 4 source conditions × 4 norm conditions. `04` §5.5 says "K0"; on
  Qwen a bare `K0` is refused by design (trap 4), so it resolves to `K0_prerope` and the
  run records `rope_position0_deviation` as the numerical justification.

### `crosslingual_semantics/aggregate_crosslingual.py`
- `failures.csv` — every non-`ok` row, counted per condition; contrasts above the 2%
  failure rate are refused without `--allow-high-failure` (`05` §7.2).
- `patching_contrasts.csv` — the six pre-registered contrasts, primary outcome the paired
  difference in `CorrectLabelMargin` **within target example** (`04` §6.1), CI by
  `hierarchical_bootstrap` (resample semantic ids, then languages), BH-corrected within
  family, carrying raw values, absolute and relative difference, language-consistency count,
  and corrected + uncorrected p.
- `exploratory_layer_sweep.csv` — a **separate file**; `04` §6.2 forbids mixing
  pre-selected object contrasts with exploratory sweeps.
- `table4_patch_effects.csv`, `fig6_patch_decomposition.pdf`.
- `interpretation_matrix.json` — computes and records the observed pattern (real numbers),
  then reports `matched_row: null` with `status: PENDING_DESIGN_10_12` until the six rows
  are transcribed; the mapping is pure YAML-driven code, so it evaluates mechanically the
  moment they are.
- `e7_go_no_go.json` — every criterion from the YAML, `met: null` + reason on PENDING,
  `supported: null` and `decision: "incomplete"` while any is unresolved.
- `aggregate_summary.json` with exclusions and provenance.

### Tests
- `tests/test_layer_selection.py` — dev/test id disjointness; selection picks the largest
  dev effect; the window is contiguous, size 5, clamped at boundaries; `layer_selection.json`
  records every screened layer; `--stage window` refuses without it.
- `tests/test_patching_identity_abort.py` — an injected identity violation invalidates the
  whole run unit, marks every row, lands in `invalid_units.json`, is excluded by the
  aggregator, and produces a non-zero exit.
- `tests/test_e7_aggregate_contracts.py` — mirrors `test_aggregate_contracts.py`: PENDING
  entries force `decision: "incomplete"` / `supported: null`; **a threshold change in the
  YAML changes the verdict** (proving nothing is hard-coded); the 2% refusal fires;
  pre-registered and exploratory results land in different files.
- `tests/nnsight_e7_pipeline_smoke.py` — the whole E7 pipeline offline on a random tiny
  Qwen2 and a `wp8_fake_data`-backed manifest: prepare → extract → retrieve → patch
  (screen + window) → aggregate. Phase prints `  [extract] …` etc., ends with
  `print("E7 pipeline offline smoke test passed")`. Appended to `$Tests` in
  [run_nnsight_rest_smoke_tests.ps1](scripts/run_nnsight_rest_smoke_tests.ps1) as
  `E7-PIPE` — that runner is the only thing that executes `nnsight_*_smoke.py`.

---

## Verification

```bash
# 1. full suite — must stay green, WP7's nine invariants included
./.venv/Scripts/python.exe -m pytest tests/ -q

# 2. all seven offline smoke programs, including the new E7-PIPE
powershell -File scripts/run_nnsight_rest_smoke_tests.ps1 -Python .\.venv\Scripts\python.exe

# 3. the frozen-file gate, explicitly
./.venv/Scripts/python.exe -m pytest tests/test_frozen_files.py -q
```

Expected: the current **191 passed, 2 skipped** grows by the new tests with the same 2
compute-PC skips; the smoke runner exits 0 reporting E3/E4/E5/E6/E7/E6-EVAL/E7-PIPE.

Acceptance beyond "green":
- `test_cross_example_patching.py` (all nine invariants) and `test_patch_scoring.py` pass
  **unchanged** — proof the Part 0 extension was additive.
- The pipeline smoke's `e7_go_no_go.json` reads `decision: "incomplete"` and names the
  PENDING criteria; `interpretation_matrix.json` reports `matched_row: null`.
- The smoke's identity rows show `|margin_delta| < IDENTITY_TOLERANCE` and the run unit
  stays valid; the dedicated abort test proves the failure path.
- Extraction's output directory stays under the size bound asserted in
  `test_extraction_shapes.py`.

## Documentation updates

- **`NEXT_STEPS.md`** — move WP9/WP11 from §3 "what will be implemented next" into §1's
  shipped table with their tests; add a §1.3 recording this slice's decisions (the three
  confirmed above, plus the `K0` → `K0_prerope` resolution and the run-unit granularity);
  extend §2 with a new §2.6 for transcribing design §18/§10.12 into
  `e7_preregistration.yaml` (blocking, no GPU, must precede results); add the E7 compute-PC
  command sequence (real FLORES/XNLI manifests → extract → retrieve → screen → window →
  aggregate) alongside the existing E6A one; update the suite counts and the
  minimum-publishable-path line to `WP7 ✓ → WP8 ✓ → WP9 ✓ → WP11 ✓`.
- **`CLAUDE.md`** — add a `crosslingual_semantics/` (E7 pipeline) table mirroring the E6
  one; note `e7_preregistration.yaml` as the second and only other place a threshold lives;
  extend rule 4's pre-registration paragraph to cover `PENDING_DESIGN_18` /
  `PENDING_DESIGN_10_12`; add a ninth trap — *an identity violation invalidates the run
  unit, not just the row* — since that is the E7 analogue of trap 3 and is the one thing a
  production run must act on rather than record.
