# NEXT STEPS — Sink Inheritance (E6/E7)

Status and hand-off for the `sink-inheritance-foundation` branch. Read
`new_design_plans/00_MASTER_PLAN.md` for the full WP DAG; this file tracks what is *done*,
what you must *run on the compute PC*, and what is *left to build*.

**Branch:** `sink-inheritance-foundation` · **Tag:** `frozen-e1-e5`
**Verification split:** this repo checkout is **smoke-only** (CPU venv at `.venv`, no
downloads, HF cache pinned to `.hf_cache`). Real datasets / GPU runs happen on a separate
**compute PC**.

Bootstrap the venv from scratch if it is missing (it is gitignored, so a fresh clone has
none, and the suite hard-fails rather than skipping without `nnsight`):

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests/ -q
```

Current suite: **685 passed, 0 skipped** (640 before the §2.14 canonical scale/alignment
slice; 538 before the §2.13 throughput slice; 514 before the §2.11 defect slice; 466 before
the §1.6 decisions slice). The two compute-PC gates in §2 **now run and pass** —
`nn_results/` carries real E1 artefacts, so they are no longer skipped. All ten offline
smoke scripts (`nnsight_e3/e4/e5/e6/e7_smoke.py` + `nnsight_e6_eval_smoke.py` +
`nnsight_e7_pipeline_smoke.py` + `nnsight_e6b_smoke.py` + `nnsight_e6a_gpt2_smoke.py` +
`nnsight_e6a_gpt2_medium_small_smoke.py`) exit 0 under
`scripts/run_nnsight_rest_smoke_tests.ps1`,
which returns 0 as a whole; the previously noted E4/E5 Windows temp-cleanup flake did not
reproduce on torch 2.10 / transformers 5.3. No frozen file has changed —
`tests/test_frozen_files.py` enforces this, now over EOL-canonical bytes so it holds on a
Linux checkout too (§2.3).

> **The smoke runner now defaults to `.venv`.** It used to default to `-Python "python"`,
> which on a fresh shell resolves to the *system* interpreter — no `nnsight` — and the run
> dies inside E3 with a `ModuleNotFoundError` that reads like a code fault. It prints the
> interpreter it chose; `-Python` still overrides.

> **The design sections are transcribed and the seven decisions are made.** `files (7).zip`
> supplied `08_AGGREGATOR_EXTENSION_SPEC.md`, `DECISIONS_REQUIRED.md` and the v3
> pre-registrations, closing §2.5/§2.6. D1–D7 were then resolved on 2026-07-30 in
> `e6_prereg_v4` / `e7_prereg_v4`, **before any E6/E7 run existed** — see §1.6 and §2.7.
> `BASELINE_HASHES.json` is complete (§2.3). Nothing in the pre-registrations is open.
>
> **The remaining blocker is compute, plus one credential**: FLORES-200 is gated on the Hub
> and E7's retrieval half will not download without an authenticated account that has
> accepted its terms (§1.6). XNLI is ungated, so E7's patching half is unaffected.
>
> `COMMANDS.md` at the repo root is the runnable form of everything below.

> **The runner needed a fix to work at all.** `transformers` writes its progress bars to
> stderr; `2>&1 | Tee-Object` under `$ErrorActionPreference = "Stop"` wraps each such line
> in a terminating `NativeCommandError`, so the script aborted during E3 despite python
> exiting 0. The preference is now relaxed around the native call only and the verdict
> comes from `$LASTEXITCODE`. If you ever saw this runner "pass" before, it was on a stack
> that printed no progress bars.

---

## 1. What is implemented (done + smoke-verified)

| WP | Deliverable | Commit | Tests |
|---|---|---|---|
| WP0 | `BASELINE_HASHES.json` (12 frozen hashes), `test_frozen_files.py` | `57788cc` | frozen files |
| WP1 | `common/depth_band.py` | `57788cc` | `test_depth_band` |
| WP1 | `common/corpus_providers.py` + additive `datasets_loader` loaders | `a6bbe5a` | determinism, packing, sst2 |
| WP2 | `common/fingerprint_runner.py` | `a6bbe5a` | available-interventions, cache |
| WP3 | `nnsight_engine`: `_apply_edits` refactor + `run_arch_trace` + ArchSpec v/o | `a1d0f29` | parity, arch-trace CE, vpaths |
| WP4 | `common/inheritance_metrics.py` | `57788cc` | `test_inheritance_metrics` |
| WP7 | `common/cross_example_patching.py` | *this slice* | `test_cross_example_patching` (9 invariants), `test_patch_scoring`, `nnsight_e7_smoke` |
| WP8 | `common/paired_manifests.py`, additive `load_flores_parallel`/`load_xnli_aligned`, real `flores_corpus`/`xnli_prompt_corpus`, `crosslingual_semantics/prepare_*_manifest.py` | *this slice* | `test_parallel_manifest_alignment`, `test_patch_controls`, `test_grouped_partition`, `test_prepare_manifests_cli` |
| WP5 | `transformation_inheritance/train_distillation.py` + 3 configs + `check_pilot_gate.py` | `89d2cb9` | `test_distillation_loss`, `test_distillation_init`, `test_resume_exactness`, `test_pilot_gate` |
| WP10 | `evaluate_transformation.py`, `aggregate_transformation.py`, `configs/e6_preregistration.yaml` | `89d2cb9`+ | `test_evaluate_transformation` (16), `test_matched_loss_selection` (9), `test_aggregate_contracts` (12), `nnsight_e6_eval_smoke` |
| WP9 | `extract_sink_representations.py`, `evaluate_crosslingual_retrieval.py`, `run_multilingual_fingerprint.py`, 4 × `configs/e7_qwen*.yaml` | *this slice* | `test_extraction_shapes` (12), `test_retrieval_chance` (10), `test_capture_only_objects` (17) |
| WP11 | `run_cross_language_patching.py`, `aggregate_crosslingual.py`, `configs/e7_preregistration.yaml` | `56d3c59` | `test_layer_selection` (20), `test_patching_identity_abort` (7), `test_e7_aggregate_contracts` (21), `nnsight_e7_pipeline_smoke` |
| WP12 | `08_AGGREGATOR_EXTENSION_SPEC.md` + `DECISIONS_REQUIRED.md` relocated; both aggregators extended; `e6_prereg_v3`/`e7_prereg_v3`; `evaluate_transformation.py --evaluate-public-reference` | `c65c2d0` | `test_seed_consistency` (8), `test_corpus_pair_contrast` (7), `test_compound_criterion` (12), `test_pair_specific_matched_loss` (8), `test_threshold_rule` (10), `test_public_reference_rows` (10), `test_source_level_contrast` (11), `test_e7_interpretation_exclusivity` (12), `test_claim_gate_no_double_count` (12) |
| — | **combine refusals + threshold-declaration guard** (see §1.5) | `03b84ae` | `test_combine_refusals` (26) |
| WP6 | `train_sentiment_adaptation.py` + 4 `e6b_f*.yaml` configs + `screen_public_pairs.py`; `evaluate_transformation.py --base` and the task-metric link | *this slice* | `test_corruption_manifest` (22), `test_effective_batch` (24), `test_lora_merge_parity` (8), `test_e6b_drift_columns` (12), `nnsight_e6b_smoke` |
| — | `common/provenance.py` (git sha, digests, versions) | `89d2cb9` | used by WP5/WP7/WP8 artefacts |

Key properties already proven offline:

- `run_intervention` is **byte-identical** after the `_apply_edits` factoring (golden fixture).
- `run_arch_trace` CE agrees with a direct HF forward + frozen `token_cross_entropy` to
  **1e-6** on random gpt2/neo/qwen/opt.
- Corpus `manifest_sha256` is deterministic in-process and across a subprocess.
- `compute_fingerprint` runs end-to-end on a random gpt2 with `with_delta_ce=True`
  (all 10 interventions + ΔCE), and its cache hits on an identical key.
- **Identity and self-source patches are exact no-ops** — max |Δlogits| is `0.0` for K, V
  and R writes on a random Qwen2, and identity rows show ΔJSD `0.0` and Δmargin `0.0`.
- **Qwen `V0` is patched pre-`_repeat_kv`**: patching one of two KV heads moves exactly
  query heads 0–3 (magnitude 0.32) and leaves 4–7 at the fp32 solve residual (~9e-7).
- **RoPE at position 0 is the identity**, measured not assumed: deviation `0.0`; the
  forward/inverse rotation pair round-trips at positions 0, 1 and 9.
- **D0/D1/D2 at one seed share a byte-identical initial state dict** and block-manifest
  hash; their losses equal their weight vectors exactly.
- **`--resume auto` is bit-exact**: interrupting a 20-step run after step 10 and resuming
  reproduces the uninterrupted step-20 weights tensor-for-tensor.
- **The whole E6 pipeline runs offline end to end** (`nnsight_e6_eval_smoke.py`): three
  conditions trained → evaluated → aggregated → pilot gate, on random tiny 4L teacher /
  8L student. Evaluation resume recomputes **nothing** (asserted by call counter, not by
  row count), and the gate's criteria 2 and 3 — unevaluable before WP10 — now read real
  values from `checkpoint_metrics.csv`.
- **The whole E7 pipeline runs offline end to end** (`nnsight_e7_pipeline_smoke.py`):
  manifests → extraction → retrieval → multilingual fingerprints → patching (screen on
  dev, window on the disjoint test ids) → aggregation, on a random tiny Qwen2.
- **`compute_fingerprint` reproduces the frozen GPT-2 `bos_attention_stats_overall.csv`**
  on real GPT-2 + real datasets, every intervention within `METRIC_ATOL`/`METRIC_RTOL`
  (e.g. `int_f` 0.5614430 vs 0.5614430). This is the §2.2 gate and it is **green**.
- **`frozen_e1_corpus` reproduces the E1 `sample_manifest.csv` row-for-row** — the §2.1
  corpus gate, also green.
- **Retrieval self-pairs are an exact identity monitor**: after per-language centring a
  language retrieves itself at top-1 = 1.0, and a degenerate all-constant representation
  is reported at chance rather than at 1.0 (pessimistic tie-breaking).
- **`Rmean`/`Rlast` equal per-position captures exactly** (max abs error `0.0`), and
  `Qmean` carries the *query*-head shape `[num_heads, head_dim]`, not the KV-head shape.
- **E7 identity patches are exact no-ops in production**, not only in tests: every
  `norm_condition="identity"` row in the smoke shows `|margin_delta| = 0.0`.
- **The v3 §10.12 interpretation matrix is provably mutually exclusive** — all 64 truth
  assignments over its six condition keys swept exhaustively, zero overlaps, and the sweep
  is *not* exhaustive over outcomes (some assignments match no row, which `04` §6.3 permits).
- **The P8M reference path runs on real weights.** `roneneldan/TinyStories-8M`
  fingerprinted against the real `TinyStories-33M` teacher on CUDA — see §2.7.1.
- **The whole E6B pipeline runs offline end to end** (`nnsight_e6b_smoke.py`): F1/F2/F3
  trained → LoRA adapters merged and parity-checked → evaluated against the untrained base
  → aggregated, with `e6b_drift.csv`, `e6b_early_warning.csv` and `e6b_factorial.csv`
  carrying real rows instead of the header-only placeholders they held since WP10.
- **Drift is measured, not asserted.** Step 0 is saved before the first optimiser update,
  so its `fingerprint_drift_from_base` is at machine epsilon (2.2e-16, the `1 − cosine`
  floor) and later steps are strictly above it — the two assertions that catch a base which
  is secretly the checkpoint, and a base which is secretly a different model.
- **The LoRA merge reproduces the adapter to one float64 ULP** on real distilgpt2
  (1.99e-13, relative 1.92e-15) — see §1.5 for why the check runs in float64 and why the
  tolerance did not move.

### Deliberate scope limits in this slice (recorded, not fabricated)

- `fingerprint_runner` implements **`engine="nnsight"` only**; the manual path raises
  (it is the parity reference, and nnsight is metric-equivalent within `METRIC_ATOL`).
- Massive coords: GPT-2 uses the frozen **EPE-0** set (so the E1 bridge is exact); other
  archs use the frozen **residual-stream** set. Qwen's per-sentence scope is *recorded*,
  not recomputed per item — `(i)/(j)` are marked failures if a set can't be resolved.
- `(d)/(e)` swap directions are dispatched for **gpt2/neo** (frozen `*_swap_directions`);
  Qwen needs none (position-id edits); other archs record a failure rather than no-op.
- `flores_corpus` / `xnli_prompt_corpus` and `load_flores_parallel` / `load_xnli_aligned`
  were deferred to WP8 and are now **implemented** (see §1.1 below for the join decision).

### 1.1 Decisions taken in the WP7/WP8/WP5 slice (recorded, not silent)

- **`Kmid` is realised as `Kmid_prerope` / `Kmid_postrope`.** `05` §3 lists a bare `Kmid`
  patch object, but `02` §6.4 requires both RoPE variants to be reported and never mixed.
  Bare `K0`/`Kmid` are accepted only where RoPE is *not* applied inside the attention
  module; on Qwen they raise and name the two variants. New *values*, no renamed column.
- **`K*_postrope` writes an inverse rotation** (`k = k_post*cos − rotate_half(k_post)*sin`)
  so the model's own in-attention RoPE reproduces the source's post-RoPE key at the target
  position. There is no envoy on the post-RoPE tensor, so this is the only faithful reading
  of "patch K".
- **K/V patching requires `qkv_layout == "separate"`** (neo/opt/qwen). GPT-2's fused
  `c_attn` raises a clear `NotImplementedError`; E7 is Qwen-only and a fused-slice write
  path would be untested risk. `R*` objects work on all four architectures.
- **`*mid` positions resolve from the prompt length, not prompt+candidate.** Label scoring
  appends candidate tokens; resolving `seq_len // 2` on the concatenated sequence would
  move the patch site for every candidate.
- **XNLI join strategy is resolved at runtime and recorded.** HF's `facebook/xnli`
  per-language configs expose only `premise`/`hypothesis`/`label` — no `promptID`. So
  `load_xnli_aligned` uses an explicit id column when present, else the structurally
  aligned `all_languages` config (one row *is* the translation set, keyed by a content hash
  of the English pair), and **raises** if neither is available. There is no positional
  fallback anywhere. `provenance["loader"]["join_strategy"]` records which was used.
- **`nnsight_engine.py` was not modified by WP7.** `run_arch_trace(capture_position0=...)`
  still raises `NotImplementedError` — a write path cannot be expressed through it, and
  `test_arch_trace_ce.py` asserts that error. `cross_example_patching` supersedes it.
- **A real bug was fixed in `load_checkpoint`**: `save_pretrained` omits tied weights
  (GPT-Neo ties `lm_head` to `wte`), so a strict `load_state_dict` failed on every resume.
  It now loads non-strictly and *verifies* the only absent keys are tied ones, rather than
  using a blanket `strict=False` that would let a truncated checkpoint resume silently.
- **`requirements.txt` gained** `pyarrow`, `pytest`, `pytest-timeout` (append-only). The
  suite always needed pytest; it was never declared.

### 1.2 Decisions taken in the WP10 slice (recorded, not silent)

- **The design document is not in this repository, and WP10 needs two of its sections.**
  `03` §7 requires `e6a_contrasts.csv` to carry "the five primary contrasts (design §8.7)"
  and `go_no_go.json` to carry design §18's criteria as literal code — the
  pre-registration. `new_design_plans/` *cites* both and reproduces neither; the only
  criterion whose text exists anywhere here is the worked example in `05` §6 (`e6a_1`,
  threshold 0.15). **This was a blocking input from you**; it arrived as
  `e6_prereg_v3` and §2.5 is closed. What remains open is §2.7's decisions.
  Resolution: every contrast and criterion lives in
  `transformation_inheritance/configs/e6_preregistration.yaml`, never in code. Four
  contrasts are mechanically derivable (the D0/D1/D2 pairings and `03` §6.4's matched-loss
  comparison) and are filled in with their sources; the fifth contrast and two go/no-go
  criteria are `PENDING_DESIGN_8_7` / `PENDING_DESIGN_18` sentinels that emit
  `met: null` + a reason and force `decision: "incomplete"`. No threshold is estimated.
- **Resumability is at `(run_id, step, corpus_id)`, not `(…, intervention)`.** `03` §6.3
  names the finer unit, but `NNsightEngine.run_all` computes the entire a–j battery in one
  forward sweep per corpus item — resuming per intervention would re-run that sweep nine
  times and *increase* total forwards. The intervention set is part of the ledger key
  (`eval_units.jsonl`), so adding or removing an intervention still invalidates and
  recomputes every affected unit, which is the property the finer granularity was for.
- **ΔCE gained a per-item path.** `07` P10 requires the 300-block subset's sampling error
  to be bootstrapped, and `_delta_ce` returned means only. `compute_fingerprint` now takes
  `delta_ce_per_item=False` (default byte-identical to before); when set, per-item lists
  land in `provenance` — the record schema is unchanged, and the flag joins the cache key
  only when true, so records cached before WP10 still hit. The CI is labelled
  `sampling_over_corpus_items` in the artefact, never as model-training uncertainty.
- **`run_config.json` gained `teacher` / `teacher_revision` / `public_reference` /
  `student_config_from` / `config_path`.** Nothing in a run directory recorded which model
  the student was distilled from, so the evaluator had no way to find its comparison
  target. Adding fields is permitted (`05` §5: a column may be added, never renamed) and
  no Phase 2 run has started. `--teacher` / `--config` remain as fallbacks for run
  directories written before this.
- **`train_distillation.py --smoke` now saves its synthetic teacher** to
  `<run_dir>/_smoke/teacher` and records that path as `teacher`. A smoke teacher was
  created in-process and unrecoverable, so the smoke evaluation fell back to the config's
  HF id — unreachable offline — and silently produced student-only rows. It now exercises
  the whole teacher-comparison branch: `mutual_interventions`, the band guard, and every
  `*_to_teacher` column. Real runs are untouched.
- **`03` §7 says to import plot style helpers from `emergence_dynamics_analysis.py`. There
  are none** — its plotting is experiment-specific `plot_*` functions with the style
  inline, and importing it would pull torch and the HF stack into a pure analysis step. Its
  conventions are matched literally instead: `Agg` backend, `dpi=200,
  bbox_inches="tight"`.
- **The smoke-test runner was broken and is fixed** (see the note at the top of this file).

### 1.3 Decisions taken in the WP9/WP11 slice (recorded, not silent)

- **Design §18, §10.11's language tiers and §10.12's interpretation rows are not in this
  repository either.** The WP10 resolution is applied verbatim:
  `crosslingual_semantics/configs/e7_preregistration.yaml` holds every contrast and
  criterion; what `04` quotes is transcribed with a `source:` line (the six §6.1
  contrasts, the "≥ 2× chance" and "≥ 5 points" §18 bars from `04` §4, and the §6.4 claim
  gate), and the rest are `PENDING_DESIGN_18` / `PENDING_DESIGN_10_11_TIERS` /
  `PENDING_DESIGN_10_12` sentinels that emit `met: null` + a reason and force
  `supported: null`, `decision: "incomplete"`. Superseded by `e7_prereg_v3`; §2.6 is
  closed and the open items moved to §2.7.
- **`Qmean`/`Rlast`/`Rmean` are capture-only objects on `cross_example_patching`.** `04`
  §3.1 asks extraction for three objects that are *reductions over a span*, which
  `PATCH_OBJECTS` has no room for. They were added additively as `CAPTURE_ONLY_OBJECTS`
  with span support in the existing trace path (so extraction opens no second path), and
  both `PatchSpec` and the writer **refuse** them: a mean over a span has no single site
  to write back to, and accepting one would produce a plausible effect size for a patch
  that never happened. All 30 WP7 tests pass unchanged.
- **`capture_sources` gained `on_item`/`keep`** (defaults byte-identical to before). WP9
  extraction is 300 sentences × 8 languages × every layer × ten objects; it streams each
  example into an `.npy` memmap instead of holding the set in RAM. The callback form pays
  `model_fingerprint`'s weight digest once per *call*, which a one-item-corpus loop would
  not.
- **The extraction index is keyed by `semantic_id`, not by corpus item id.**
  `flores_corpus` names items `"<semantic_id>:<lang>"`, so keying `row_of` by item id
  would make the per-language indexes non-joinable — the exact property the experiment
  rests on. `index.json` carries both (`row_of` and `item_id_of`).
- **Patching's unit of work is `(model, target language, object, layer)`**, not `05` §3's
  finer `(…, norm condition)`. All four norm conditions at one site share a single
  baseline forward inside `run_patched`; splitting them would recompute it four times.
  `norm_condition` is still a column and part of the ledger key, so adding or removing one
  still invalidates the site — the same trade WP10 made for `(run_id, step, corpus_id)`.
- **`source_condition="none"` for `random` and `identity`.** Neither reads a source, so
  crossing them with the four source conditions would write four byte-identical rows and
  pay four times for them. A new *value*, never a renamed column (`05` §5).
- **`04` §5.5 says "{K0, V0}"; on Qwen that resolves to `K0_prerope`.** A bare `K0` is
  refused by design (trap 4), and the runs record `rope_position0_deviation` as the
  numerical justification rather than asserting the position-0 identity.
- **An identity violation aborts the run unit, and identity is scheduled first.** `04`
  §5.4 requires the abort; scheduling identity first makes an invalid unit *cheaper* than
  a clean one instead of merely flagged after full cost. Every row of the unit is stamped
  `unit_status="identity_violation"`, the unit lands in `invalid_units.json`, the
  aggregator drops it, and the process exits non-zero.
- **The 2% failure-rate refusal is scoped to the rows a contrast consumes**, not global.
  One broken language-object-layer unit must not refuse a contrast that never touches it,
  and must not be diluted away by healthy rows in other conditions.
- **Three defects were found and fixed while wiring WP9/WP11 up.** All three were
  pre-existing and none is a tolerance change:
  1. **`compute_fingerprint` raised on every Qwen item.** It correctly recorded `int_i`/
     `int_j` as unavailable when massive coordinates cannot be resolved, then asked
     `NNsightEngine.run_all` to run the whole registry anyway — which raises on (i) and
     took every item down with it. `run_all` gained an additive `keys=None` argument
     (default runs all ten, exactly as before) and the runner now passes its own
     `runnable` set. Qwen fingerprints now complete with (i)/(j) recorded as failures,
     which is what `02` §3.4 always said should happen.
  2. **`test_fingerprint_runner_bridge` compared GPT-2 against a GPT-Neo reference.** Its
     `_reference_csv()` took the first `rglob` hit; once `nn_results/` existed that was
     `gpt-neo-1.3B`. Most interventions landed close enough to look fine and `int_f`
     diverged (0.561 vs 0.127), which reads as a seam regression rather than a mismatched
     comparand. It now reads the sibling `run_config.json` and only accepts a reference
     produced by `gpt2` (`05` §7.1, hash before compare). Against the right artefact the
     seam agrees to ~1e-7.
  3. **`hierarchical_bootstrap` re-ran a pandas `groupby` inside every replicate.** At
     `n_boot=10000` that made E7 aggregation take tens of minutes. The clustering is now
     derived once; the `rng.choice` calls are made in the same order on the same key
     lists, so every replicate draws exactly what it drew before — **verified
     bit-identical** to the previous implementation, 45× faster.
- **`hierarchical_bootstrap` gained `return_draws`** and `inheritance_metrics` gained
  `bootstrap_two_sided_p` (both additive). `04` §6.2 wants a corrected and uncorrected p
  beside a hierarchical-bootstrap CI; deriving the p from *that same resampling* avoids
  putting two inference frameworks in one row. The p is floored at `1/n_boot` and reports
  `min_attainable_p`, for the same reason `paired_seed_contrast` does.

### 1.4 Decisions taken in the WP12 slice (recorded, not silent)

- **The four delivered files were relocated verbatim, with exactly one edit.** Every
  `status: PENDING_DECISION_*` entry carried a `decision_required:` block but no `reason:`,
  which both aggregators and `test_aggregate_contracts.py` read — the artefact would have
  reported a pending criterion with an empty explanation. A one-line `reason:` pointing at
  `new_design_plans/DECISIONS_REQUIRED.md` §D*n* was added to each. **No threshold, tier,
  metric or number was touched**; `decision_required:` is verbatim.
- **`pending_decisions` is a separate key from `pending_preregistration`, and only some
  entries block.** `08` §12 wants `incomplete` while any `PENDING_DECISION_*` remains, but
  a blanket veto is wrong: the interpretation matrix is a separate `04` §6.3 artefact whose
  own `matched_row` is already `null`, so letting an undecided *reading* suppress a result
  would be the wrong failure — and it broke two existing E7 contract tests when tried.
  Each sentinel now carries `blocks_decision`. It is true for `criterion_3_combine` (it
  decides how §18 criterion 3 folds) and for a PENDING contrast some criterion or the claim
  gate actually reads; false for `primary_topology_metric` (it chooses which of two
  *already computed* contrasts the paper calls primary), for the E6B criteria (evaluated in
  their own `e6b_go_no_go.json`, where they already report `met: null`), and for the matrix.
  Non-blocking sentinels are still reported — an operator must know they are open.
- **`criterion_3_combine` has no handling in `08`, and could not be ignored.** The YAML
  makes it govern how `e7_3k`/`e7_3v` fold into the decision; ignoring it would count
  design §18 criterion 3 twice under `go_no_go_combine: all`, turning a disjunction the
  design may have intended into a conjunction. While it carries `PENDING_DECISION_COMBINE`
  the fold is refused and the two limbs are reported unresolved.
- **E6 contrasts now see the unfiltered frame** (CLAUDE.md trap 12). Matched loss, table 2
  and the figures still read the corpus-selected one, so no existing number moved.
- **The matched-loss cache is keyed on the resolved corpus id, never `id(frame)`.**
  `entry_corpus` returns a freshly filtered DataFrame every call, so an identity key both
  misses every time *and* can be reused by a later object after garbage collection —
  handing one corpus's matched-loss table to another corpus's contrast. Caught by
  `test_pair_specific_matched_loss.py`.
- **`08` §5.1's own worked example is arithmetically wrong, and is recorded rather than
  fudged.** It asks for per-seed diffs `[0.30, 0.05, 0.05]` against threshold `0.15` to be
  `met: true` without `seed_consistency` — but those average to `0.1333`, below the bar, so
  the pooled reading is `False` too and the example demonstrates nothing. The literal triple
  is asserted as it actually behaves, and the contrast it was reaching for is made with
  `[0.30, 0.10, 0.10]`. See `tests/test_seed_consistency.py`.
- **`08` §12's "test_e7_aggregate_contracts.py passes unchanged" could not hold**, and the
  exception is documented in the test itself. One of its 21 tests asserted v1 sentinel
  *identifiers* — `pending_preregistration >= {"e7_3","e7_4"}` and
  `interpretation_matrix_status == "PENDING_DESIGN_10_12"` — and v3 renames criterion 3 to
  `e7_3k`/`e7_3v`, makes criteria 4–5 claim-gate components, and restyles the matrix
  sentinel. Two assertions in that one test were updated; the other 20 are untouched, and
  the behavioural invariant §12 protects (shipped YAML ⇒ `incomplete`, `supported: null`)
  is preserved and still asserted. `test_aggregate_contracts.py`,
  `test_matched_loss_selection.py` and `test_evaluate_transformation.py` pass unchanged.
- **Both smoke scripts were updated for the same reason** (they asserted v1 ids), and
  `nnsight_e6_eval_smoke.py` now passes its discovered synthetic corpus id as `--corpus`:
  the shipped YAML names the real corpora, which a smoke run does not build.
- **`08` §6 (`topology_area_diff_to_teacher` as a CSV column) is deliberately not
  implemented.** §6 is explicitly conditional on decision D1 resolving to that metric, and
  D1 is open. `im.topology_report` already returns `area_diff_normalised`, so it is a
  two-line change if D1 lands there.
- **`--evaluate-public-reference` works with or without a run directory.** Requiring a
  finished run to measure the public 8M would make design §8.7 contrast 4 wait on 1.5–3
  GPU-days that have nothing to do with it, so `--config <e6a config>` is accepted as an
  alternative. Run alongside `evaluate_run` it shares the already-built corpora and teacher
  objects, which is `08` §4's requirement and also the only way the `manifest_sha256` guard
  is *guaranteed* rather than merely expected.
- **A public reference has no trainer `eval_log`, so its `validation_ce` comes from the
  record's own baseline CE** — and is honestly empty without `--with-delta-ce`. Design §8.7
  contrast 4 selects D1/D2's matched checkpoint against exactly that number, so the real
  runs must pass `--with-delta-ce`.
- **Both aggregator versions were bumped** (`aggregate_transformation_v2`,
  `aggregate_crosslingual_v2`) and `evaluate_transformation_v2`, because the artefact shape
  gained keys. No column was renamed (`05` §5).

### 1.5 Decisions taken in the WP6 slice (recorded, not silent)

- **Two defects were found in WP12's own output and fixed first** (`03b84ae`).
  `criterion_3_combine` was read as `str(spec.get("combine") or "any")`, so deleting
  `status: PENDING_DECISION_COMBINE` without adding `combine:` — the natural half-finished
  edit — produced `status: "ok"` and a verdict, resolving decision D6 *in code*. Worse,
  `combine: ALL_` fell through to `any` while the row recorded `"all_"`: an artefact that
  looks like an audit trail and is not one. Both are refused now, and `go_no_go_combine` /
  `e6b_go_no_go.combine` get the same validation (an *absent* key still defaults to `any`,
  which `08` §5.2 specifies). Separately, the interpretation matrix's `thresholds:` block
  is a *declaration* — `_condition_holds` reads the literals inside the six rows — so
  editing it alone changes nothing silently, which is exactly the shape decision D7's edit
  takes; `check_threshold_declaration` now compares the two and the exclusivity test
  asserts it. Both are CLAUDE.md traps 11 and 13.
- **The E6B evaluator link was in scope, and had to be.** `03` §6 says nothing about drift
  or a base model, but `05` §2 has carried `task_accuracy`, `task_nll`, `ece_10bin` and the
  three `*_drift_from_base` columns since WP10 with nothing populating them. Shipping the
  trainer alone would have left `e6b_drift.csv` header-only forever and criteria `e6b_2` /
  `e6b_3` permanently unevaluable — trap 12's failure in a new place. Task metrics are now
  *read* from `eval_log.jsonl` (the seam `validation_ce` already uses; the trainer measured
  them with the model's own scorer and a second implementation would be a second source of
  truth), and `--base` supplies the F0 comparand for the three drift columns, guarded by
  the same `guard_comparison` the teacher gets.
- **Absent a base, the drift columns stay empty — never `0.0`.** Zero drift is a
  *measurement*; a run that simply had no comparand must not be readable as one that did
  not move.
- **`merge_and_unload()` mutates the PEFT wrapper in place**, so the first implementation
  of the parity check compared a model against itself and reported `0.0` for every adapter.
  It passed. `test_lora_merge_parity.py` now perturbs the zero-initialised `lora_B` and
  asserts the adapter actually moves the logits *before* asserting the merge preserves
  them, so the check cannot be vacuous again (CLAUDE.md trap 14).
- **The merge-parity check runs in float64, and the tolerance did not move.** `03` §4.5
  says fp32 and 1e-5. Its bar is *absolute*, and real distilgpt2 logits reach ~104, so
  meeting 1e-5 there needs ~1e-7 relative — below float32's own epsilon. Measured on real
  weights: fp32 gives `1.068e-04` (relative `1.026e-06`), float64 gives `1.990e-13`
  (relative `1.923e-15`); both are ≈8.6× their own machine epsilon, which is round-off, not
  a merge error. Raising the tolerance would have weakened the check; raising the precision
  does not, because a genuine fault (wrong `alpha/r`, missed `Conv1D` transpose, adapter
  never applied) is systematic and fails 1e-5 at any precision. `merge_parity.json` carries
  `deviation_from_spec` and both precisions' deltas, and a test asserts `MERGE_PARITY_TOL
  == 1e-5` so it cannot drift (CLAUDE.md trap 15).
- **Only the merged copy is fingerprinted, resolved by weights present.**
  `evaluate_transformation.fingerprint_dir` prefers `step_<n>/merged/` when it holds a
  `config.json` and falls back to the step root otherwise — read from the filesystem, not
  from the condition name, so a rename cannot send the evaluator at the PEFT adapter.
- **Batches are right-padded.** Left padding would put a pad token at position 0 for every
  example shorter than the longest in its batch — manufacturing exactly the shared
  first-token anchor design §16.2 forbids, and the sink is measured at position 0.
- **`--smoke` is CPU/fp32 by construction**, not merely by default: the E6B smoke must give
  the same numbers on a machine with no GPU, and bf16 on a 4-layer random model would make
  the merge-parity tolerance meaningless.
- **F0 needs no config.** `03` §4.1 lists it as "none / base distilgpt2"; it is the
  untrained drift reference, so the four shipped configs are F1–F4 and F0 is reached
  through `--base`.
- **`test_effective_batch.py` reads the shipped configs**, not a hand-made dict. A config
  that drifts is the realistic failure, and a test that builds its own input would never
  see it. It also asserts the four conditions match `e6_preregistration.yaml`'s declared
  factorial cell-for-cell.

### 1.6 The decisions slice (2026-07-30) — D1–D7 resolved, six defects fixed

This slice made the seven §2.7 decisions and fixed what stood between the repo and a real
pilot. **No frozen file was touched.**

**Six defects, all pre-existing and none found by the suite.** Four were found by reading
the code against the artefacts it produces; two — the corpus memory blowup and the
schedule-splicing resume — were found only by *running* E6A for the first time.

1. **`check_pilot_gate.criterion_4` could never pass.** It read `passed`, then
   `all_within_tolerance`; the frozen `run_parity_check` writes **`all_rows_pass`**
   (confirmed against three real reports in `nn_results/parity_reports/`). So it computed
   `bool(None → False)` and reported `met: False` on a *passing* parity report — and design
   §8.5 forbids Phase 2 without `proceed: true`. Now reads the producer's key, keeps the
   other two as fallbacks, summarises the `rows` list into `observed`, and returns
   `met: null` **with a reason** when no recognised verdict key is present rather than
   defaulting to `False`. The reason it survived a green suite: the test's `_parity()`
   helper built its fixture in the *reader's* shape. New tests build it in the
   **producer's** shape and assert the key names agree with `nnsight_engine.py` itself.
   (CLAUDE.md trap 16.)
2. **Nothing could produce that report for a student checkpoint.** `save_pretrained` writes
   no tokenizer (trap 7) and the frozen Neo harness loads model *and* tokenizer from the
   same `--model-name`, so it died before reaching the parity check. New
   `transformation_inheritance/run_pilot_parity.py` stages a **copy** of the checkpoint with
   the run's tokenizer beside it and dispatches to the frozen CLI unchanged — it never
   re-implements the comparison, and a test asserts that on the source. The checkpoint
   itself is left unmodified so `checkpoint_sha256.txt` stays valid.
3. **`resolve_threshold_rule` calibrated on the wrong corpus.** It scoped its null spread
   with a hard-coded `corpus_override=None` while the contrast beside it honoured the
   caller's `--corpus`, so a criterion could compare a value measured on one corpus against
   a spread computed on another. Invisible in production; reachable through `--corpus`,
   which the smoke run uses. (Trap 17.)
4. **`nnsight_e6b_smoke.py` asserted the spec, not the code.** It required
   `merge_parity.check_dtype == "float32"` — `03` §4.5's value — while the implementation
   records `float64` beside `spec_check_dtype: "float32"` precisely so the trap-15 deviation
   stays auditable. It now asserts both, plus that `MERGE_PARITY_TOL` is still 1e-5.
   (Trap 18.)

**A fifth defect, found by actually running E6A: the training corpus does not fit in RAM.**
This is the one that had never been reachable, because only the 5-step `--smoke` path had
ever run. TinyStories' train split is 2,119,719 stories → 457M tokens → **3.57M blocks** of
128. `load_tinystories_blocks` held every block twice as Python lists of Python ints — once
in `blocks`, once again as `manifest[i]["input_ids"]` — which measures **5,630 bytes per
block, ~20 GB for the split**. On top of that `prov.sha256_json` materialised the whole
corpus as one JSON string (**+8.8 GB peak, ~11.7 min**), and `write_block_manifest` then
built a 3.57M-row DataFrame. Peak ~29 GB against 15.9 GB of host RAM. It froze this
workstation twice before it was diagnosed.

Fixed as a pure **representation** change — same tokens, same order, same
`manifest_sha256`, same parquet schema:

- `load_tinystories_blocks(as_array=True)` folds packed blocks into `int32` chunks **as they
  are produced** (`_ARRAY_FOLD_ROWS = 50_000`). Converting at the end saves nothing — the
  list-of-lists *is* the peak; a first attempt that did so still measured 10.4 GB.
- `load_tinystories_blocks(manifest_input_ids=False)` drops the duplicate copy; the ids
  still reach the `05` §4 parquet, read from the block array.
- `prov.sha256_int_rows` streams the digest, reproducing `sha256_json`'s **exact bytes** so
  the value does not move — a "different but equally good" digest would silently partition
  artefacts into two incomparable sets under `05` §7.1.
- `write_block_manifest` writes parquet in `MANIFEST_CHUNK_ROWS` row groups.

Both flags default to the previous behaviour, so `datasets_loader` stays additive-only.
Measured after the fix: **~3 GB peak, digest ~1.3 min, packing ~17–30 min** (one-time per
split/seed). `tests/test_block_corpus_memory.py` (16) and
`tests/test_provenance_streaming.py` (9) assert the equivalences — including that the fold
size changes nothing observable and that the streamed digest equals `sha256_json` on numpy
rows, Python lists and generators alike. CLAUDE.md trap 19.

**A sixth defect, in the documented pilot → Phase 2 sequence.**
`torch.optim.lr_scheduler.LambdaLR.state_dict()` excludes the lambda by design, so
`load_checkpoint` restores `last_epoch` but not the schedule it came from — and the lambda
closes over `max_steps`. NEXT_STEPS §3 tells you to run the pilot at `--max-steps 2000` and
then re-run the same config in the same directory at 10,000, which would have produced
**2,000 steps of a 2,000-step cosine followed by 8,000 steps of a 10,000-step cosine**, with
a learning-rate jump at the join and nothing in the artefacts recording it. Design §1.6
pre-registers the schedule and says the hyperparameters are not to be improved; a spliced
schedule is not the registered one. `assert_resumable_schedule` now refuses the resume,
names both horizons, explains why, and gives the two ways forward (fresh run directory, or
`--resume none`). It runs **before** `write_run_config`, which would otherwise overwrite the
very field it reads. Three tests in `tests/test_resume_exactness.py`; the pre-existing
`test_the_schedule_horizon_is_part_of_the_run` already proved the horizon changes the
trajectory, which is what makes this a correctness issue rather than a cosmetic one.
CLAUDE.md trap 20.

**One additive extension, needed by D3 and D4.** `resolve_threshold_rule` gained
`anchor_side: below|above`. The pre-existing form is only `anchor − k·spread`, which is
right for `e6a_3` (both limbs read one boundary from either side of an anchor at 1.0) but
**wrong** for `e6a_4` and `e6b_2`, whose observed quantity is an absolute difference
anchored at 0: `0 − 2·spread` is negative, so "similar function" (`less_equal`) could never
be satisfied and "measurably different" (`greater_equal`) always would — two criteria that
silently cannot fail. Default `below` reproduces the previous behaviour byte-for-byte; an
absent field changes nothing and a *present but unrecognised* one is refused, not folded
(trap 13). `tests/test_threshold_rule.py` asserts all three properties.

**Two contract tests were re-pinned, deliberately.**
`test_shipped_preregistration_marks_the_absent_design_sections` and
`test_the_shipped_preregistration_is_incomplete_until_the_design_is_transcribed` asserted
that the *shipped* YAML yields `incomplete`. v4 resolves the decisions, so that invariant is
now false by design. They assert the new one — every decision made, dated, and argued; no
criterion left with neither a threshold nor a rule — and the refusal mechanism they used to
cover is asserted on an **injected** sentinel instead, which is strictly stronger because it
pins both directions (trap 19). The three smoke programs were updated for the same reason.

**Pre-flight hygiene.**

- Three real `checkpoint_metrics.csv` from instrument-verification runs were moved out of
  the results tree to `transformation_inheritance/_exploratory/` (see §2.8). Both
  aggregators discover inputs by `rglob` with no de-duplication, so anything left under
  `results/` is pooled into the paper's tables.
- **FLORES-200 is gated on the Hub**; XNLI is not. Measured on `datasets==4.8.4`:
  `facebook/flores` → `DatasetNotFoundError: ... is a gated dataset`, while
  `facebook/xnli` loads both the per-language and `all_languages` configs (5,010 test rows).
  Swapping in an ungated mirror does **not** work — `Muennighoff/flores200` and
  `gsarti/flores_101` are loading-script datasets and `datasets>=3.0` refuses those, and
  `openlanguagedata/flores_plus` is gated too. `load_flores_parallel` now raises with those
  instructions instead of an opaque HF error (additive, error path only). **E7's
  XNLI/patching half is unaffected; only FLORES retrieval needs the token.**

---

## 2. Compute-PC gates

### 2.1 / 2.2 The two bridge gates — **PASSED** ✓

These are the load-bearing claim that "the new E6/E7 instrument measures the same thing as
the frozen E1–E5 code." They previously skipped for want of reference artefacts. With
`nn_results/` present they **run and pass** on real GPT-2 + real datasets:

```bash
./.venv/Scripts/python.exe -m pytest tests/test_corpus_e1_bridge.py \
    tests/test_fingerprint_runner_bridge.py -q     # 2 passed
```

- `frozen_e1_corpus` reproduces the E1 `sample_manifest.csv` row-for-row.
- `compute_fingerprint` reproduces `bos_attention_stats_overall.csv` for **every**
  intervention within `METRIC_ATOL`/`METRIC_RTOL`.

Both discover their reference by searching the repo tree; `SINKS_E1_MANIFEST_CSV` /
`SINKS_BOS_STATS_CSV` still override. The fingerprint gate now checks the reference's
sibling `run_config.json` and only accepts a `gpt2`-produced artefact — see §1.3, defect 2.
`nn_results/` is **untracked**, so a fresh clone will see these skip again rather than fail.

### 2.3 Finish `BASELINE_HASHES.json` — **DONE** ✓

Both sentinels are gone, and a third problem was fixed while closing them.

- **`reference_results.gpt2_small_table1`** now carries all ten pooled `all_datasets` rows
  (`int_a`..`int_j`: `mean_bos_attention`, `stderr`, `n_examples`), transcribed **by script**
  from
  `nn_results/e1_e2/legacy_results/results_gpt2_e1_e2/table1_multiseed/seed_000/dataset_analysis/bos_attention_stats_overall.csv`
  — never from the paper. `_source` records that file's own sha256, its sibling
  `run_config.json`'s sha256, and the `model_name: gpt2` / `dtype: float32` / band / massive
  coords the artefact was produced under (`05` §7.1, hash before compare — trap 10).
  Baseline: `int_a` 0.563683 ± 0.002170, `int_f` 0.561443 ± 0.002129, n=300.
- **`environment`** captured live: torch 2.10.0+cu128, transformers 5.3.0, nnsight 0.7.0,
  numpy 2.4.3, scipy 1.17.1, pandas 3.0.1, datasets 4.8.4, peft, python 3.12.4, CUDA 12.8,
  NVIDIA GeForce RTX 2060.
- **The hashes are now EOL-canonical.** They were recorded over raw bytes on a CRLF working
  tree (`core.autocrlf=true`), and there was no `.gitattributes`. All twelve still matched
  *here* — the frozen-file gate was never actually broken on this checkout — but they would
  have failed on any Linux or `autocrlf=false` clone, where the checkout is LF, and that
  failure is indistinguishable from someone having edited a frozen file. `_sha256` in
  `tests/test_frozen_files.py` now normalises `\r\n` → `\n` before hashing, the manifest was
  regenerated with the same normalisation, `hash_convention` records it, the superseded
  raw hashes are retained under `frozen_files_raw_crlf` for audit, and `.gitattributes`
  (`* text=auto eol=lf`) stops it recurring. Stated trade-off: an EOL-only change to a
  frozen file is no longer detected — correct, since git itself creates those and no E1–E5
  number depends on a line terminator.

### 2.4 Full-suite sanity — **green** ✓

`python -m pytest tests/ -q` → **538 passed, 0 skipped**, and
`scripts/run_nnsight_rest_smoke_tests.ps1` exits 0 across all **eight** smoke programs
(`nnsight_e6b_smoke.py` joined in the WP6 slice).

### 2.5 / 2.6 Transcribe the design sections — **DONE** ✓

`files (7).zip` supplied `e6_prereg_v3` and `e7_prereg_v3`, and both are in place. Design
§8.7's five primary contrasts, §10.11's six, §10.12's six interpretation rows and §18's
criteria for E6A, E6B and E7 are all transcribed, each with a `source:` line naming its
section. The `PENDING_DESIGN_*` sentinels are gone.

`08_AGGREGATOR_EXTENSION_SPEC.md` and `DECISIONS_REQUIRED.md` were relocated into
`new_design_plans/`, and WP12 taught both aggregators every schema form v3 uses. See §1.4
for the decisions taken while wiring it up.

### 2.7 The seven open decisions — **MADE 2026-07-30** ✓

All seven are resolved in `e6_prereg_v4` / `e7_prereg_v4`, **before any E6 or E7 run
existed**. That ordering is the whole content of a pre-registration (`05` §6), so it is
recorded in each file's header in checkable terms rather than merely asserted: at decision
time `transformation_inheritance/results/` held only a public TinyStories-8M reference row
and one 16-step E6B run (both since moved to `_exploratory/`, see §2.8), no D0/D1/D2
checkpoint had been trained, and `crosslingual_semantics/results/` did not exist at all.

| ID | Entry | Resolution |
|---|---|---|
| D1 | `primary_topology_metric` | **Wasserstein.** §24's claim is that anchors are conserved *roles*, i.e. about where in depth the anchor sits — and Spearman would score a teacher-like profile shifted to the wrong depth as highly similar. The rival argument (sign alignment with c1/c2) is presentational. `e6a_c3` is now a **distance**: label the inverted sign in table 2 and the figures. |
| D2 | `e6a_3` | **Framing (B)**, `null_seed_spread`, null D0, `statistic: range`, anchor 1.0, `anchor_side: below`, k=2 on both limbs. "High" = within 2 null ranges of 1.0; "low" = more than 2 below. |
| D3 | `e6a_4` | **Framing (B)** on the *same* D0 spread, anchor 0.0, **`anchor_side: above`**, k=2. Both limbs observe \|P8M − D2\|. |
| D4 | `e6b_1`, `e6b_2` | `e6b_1` = **0.02 fixed**, anchored on SST-2's own 872-example binomial SE (~1.0 point; 0.02 ≈ 2 SE) — an *external* anchor, so calibrating against our own runs would have been circular. `e6b_2` = framing (B), null F1, `statistic: std`, anchor 0.0, `anchor_side: above`, k=2, reusing `onset_threshold_k`. |
| D5 | `e7_c6` | **Route C — exploratory, not tiered.** Route A was attempted and closed on evidence: arXiv:2412.15115v2 §3.1 was read and gives no per-language token share, no mixture table and no language list. Route B (Joshi et al.) rejected as describing NLP resources generally rather than Qwen2.5's corpus. `languages_a`/`languages_b` stay empty, the aggregator emits a `pending` row, per-language effects are still reported. |
| D6 | `criterion_3_combine` | **`any`.** Criterion 1 says "K0 **or** V0"; reading criterion 3 as a conjunction would put opposite quantifiers on adjacent criteria with nothing in the text marking the change, and would contradict §10.12's expectation that K0 and V0 differ. |
| D7 | `interpretation_matrix.thresholds` | `semantic_sensitivity` = **0.10** (a *reading* of §18 criterion 3 — same quantity, same scale). `unrelated_transfer` = **0.10 as judgement, with no design anchor** — the paper must say so. Calibrating it against the `random` norm condition was considered and deferred: the E7 aggregator has no `threshold_rule` support, `_condition_holds` reads literal floats, and the 64-assignment disjointness proof sweeps those literals, so it would mean changing the matrix evaluator, `check_threshold_declaration` and the exclusivity test together — the riskiest change available for a field that blocks no verdict. |

**Consequence to plan around:** framing (B) refuses a spread from fewer than three seeds, so
`e6a_3`, `e6a_4` and `e6b_2` report `met: null` and `go_no_go.json` reads `incomplete` until
D0 (and F1, for E6B) have run at seeds 0, 1 and 2. That is the registered rule working, not
a fault. The pilot therefore trains **D0 at three seeds**, not one.

`new_design_plans/DECISIONS_REQUIRED.md` is kept as the record of the argument on each side.
**Do not re-open a decision now that results are being produced** — an amendment after the
fact will read as tuning unless the ordering is defensible.

Because no shipped entry carries a `PENDING_DECISION_*` sentinel any more, the blocking
mechanism is asserted on an *injected* one
(`test_injecting_a_sentinel_into_the_shipped_file_still_blocks`,
`test_reopening_a_decision_blocks_the_verdict_again`), so deleting the reader still fails
the suite — CLAUDE.md traps 11 and 19.

#### 2.7.1 The P8M reference path — **verified on real weights** ✓

`08` §4 ran end to end on this workstation's RTX 2060 (torch 2.10+cu128), not only in the
stubbed tests:

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py   --config transformation_inheritance/configs/e6a_ce.yaml   --evaluate-public-reference roneneldan/TinyStories-8M   --device cuda --dtype float32 --sink-blocks 24
```

Real `roneneldan/TinyStories-8M` against the real `roneneldan/TinyStories-33M` teacher on
real TinyStories + frozen-E1 corpora: two `condition="P8M"` rows at `checkpoint_step=-1`,
`seed=-1`, band `[2,8)`, 9 mutual interventions, `status="ok"`,
`fingerprint_cosine_to_teacher` 0.2113 (in-domain) and 0.4880 (cross-domain). A second run
with `with_delta_ce=True` produced `validation_ce` 1.8979 and
`functional_cosine_to_teacher` 0.9683 — the numbers design §8.7 contrast 4's matched-loss
selection needs. `aggregate_transformation.load_all_metrics` discovers the rows and
`is_reference_condition` classifies them as seedless.

**The real runs must pass `--with-delta-ce`.** Without it the reference's `validation_ce`
is honestly empty and `e6a_c4a`/`e6a_c4b` report `no_data` rather than selecting a
checkpoint.

**These rows have been moved out of the results tree** — see §2.8. They used
`--sink-blocks 24`, which produces `tinystories_validation_sink_24`, *not* the
`tinystories_validation_sink_300` the pre-registration names. The real reference run must
not pass `--sink-blocks` at all.

#### 2.7.2 The E6B pipeline — **verified on real weights** ✓

Real `distilbert/distilgpt2` LoRA fine-tuned on real SST-2 for 16 steps and evaluated
against the untrained base on the same RTX 2060. Six `ok` rows across three corpora at steps
0 and 16: `validation_ce` 0.654134 → 0.654351, `task_accuracy` 0.617188 → 0.609375, and all
three `*_drift_from_base` columns populated (4e-6 … 1.1e-5) — which is what `--base` exists
to do. Merge parity 2.700e-13 in float64. Step 0 is saved before the first optimiser update,
so its drift sits at the `1 − cosine` floor rather than at zero.

Also moved to `_exploratory/` (§2.8): 16 steps is not a production F1 run.

### 2.8 Results-tree hygiene — **DONE** ✓

Both aggregators discover inputs with `rglob("checkpoint_metrics.csv")` from the results
root and **do not de-duplicate**, so anything left under `results/` is pooled into the
paper's tables whether or not it was a production run. Three verification runs were moved to
`transformation_inheritance/_exploratory/` (with a README recording their numbers, so
nothing is lost):

- `e6a/P8M/reference/` and `e6a/P8M/reference_dce/` — **both** carried
  `condition=P8M, seed=-1, checkpoint_step=-1` on the *same* corpus, so a production
  aggregation would have loaded two rows for one seedless condition, one of them with an
  empty `validation_ce`. Design §8.7 contrast 4's matched-loss selection reads exactly that
  column, so the duplicate was not harmless.
- `e6b/F1/seed0/` — the 16-step run above, which would have pooled with real F1 seeds.

`transformation_inheritance/results/` now contains no `checkpoint_metrics.csv`. Verified:
`load_all_metrics` finds 0 rows there.

### 2.9 Public-pair screening — **RUN on real models** ✓

`07` P6's "cheapest evidence in the project", and the first production artefact in the repo.
No training. Real gpt2 / distilgpt2 / Qwen2.5 base+instruct at 0.5B and 1.5B, fp32 on the
RTX 2060, on the frozen `e1_100x40` corpus. **6 rows, 0 failed.**

| pair | fingerprint cosine | topology Wasserstein | band mismatch | mutual keys |
|---|---|---|---|---|
| gpt2 → distilgpt2 | 0.9801 | 0.0105 | 0.091 | 10 |
| Qwen2.5-0.5B base → instruct | 1.0000 | 0.0069 | 0.000 | 7 |
| Qwen2.5-1.5B base → instruct | 0.9999 | 0.0022 | 0.000 | 7 |

`qwen3_base_instruct` was not run — it is conditional in design §9.9 and is recorded as
`not run` rather than as a failure. Artefacts: `results/screening/screening_fingerprints.csv`
and `screening_summary.json`.

**These are observations, not verdicts.** No §18 criterion reads this table; it is the
screening step that decides whether the expensive experiments are worth running. Qwen's
7 mutual interventions versus GPT-2's 10 is the expected consequence of `int_i`/`int_j`
being unavailable when massive coordinates cannot be resolved (`02` §3.4) — recorded as
failures, not silently dropped.

---

### 2.10 E6A training — **started, measured, deliberately stopped**

The first real E6A run in the project. D0/seed 0 trained to **step 51 of 2,000** and was
stopped once the rate was known; the artefacts are in
`transformation_inheritance/_exploratory/e6a_D0_seed0_partial_50steps/` and the results tree
is empty. What it established:

- **The corpus memory fix works on the real split.** Peak RSS ~3.1 GB against the ~29 GB the
  unfixed path needed (§1.6). The 3.57M-block corpus builds in ~17–30 min and hashes in
  ~1.3 min.
- **6.67 s/step**, measured (27 steps in 180 s) at bf16, batch 16 × accum 4 = 8,192
  tokens/step. **Superseded by §2.13**: roughly 5.2 s of that was the per-micro-batch
  `epoch_order` shuffle, so the table below overstates every figure by ~3.5×. Kept as the
  historical record of what was measured at the time.

  | unit | cost |
  |---|---|
  | one 2,000-step pilot run | 3.7 h + ~20 min corpus build |
  | the full 5-run pilot | ~20 h |
  | one 10,000-step Phase 2 run | 18.5 h |
  | Phase 2, all nine runs | **~6.9 GPU-days**, E6A alone |

- **There is no cheap arm.** `need_attention` is hard-wired true because `L_ATTN` is logged
  as a metric for every condition, so D0 pays the teacher forward and the attention JSD just
  like D2. Budget all conditions equally.

The design's own estimate is 4–9 h per 10k run on a 4080 SUPER, so this card is roughly 2–4×
slower — the pilot is a scheduling decision, not a code one. Nothing scientific was produced:
51 steps measures the instrument, not the experiment.

---

### 2.11 The 2,000-step pilot — **ran, gate says `hold`/`incomplete`, four defects found**

The full five-run pilot ran on a 4080 SUPER (2026-07-30/31): D0/D1/D2 at seed 0 plus D0 at
seeds 1 and 2, the P8M reference, evaluation at steps 0/250/500/1000/2000 on both corpora,
aggregation, and the gate. **Measured cost:** 0.2065 steps/s (4.85 s/step), 4.7 GB peak VRAM,
~2.7 h train + ~31 min evaluate per run — so Phase 2 (9 × 10,000 steps) is ~5 GPU-days.

**The pipeline is clean.** 52 rows read, 52 used, 0 excluded, `n_failed: 0` on all five runs,
step-0 rows byte-identical across conditions (the shared-init property holds), parity all ten
interventions pass at max deviation 7.45e-09.

**What the numbers say.** CE 10.87 → 2.54–2.60 (teacher KL 22.7 → 2.51, top-1 agreement 55%),
so distillation works. But the sink **decreases**: 0.0114 → 0.0019–0.0028 in-domain. The
reason is upstream of the students — the **teacher's own** `baseline_sink` is **0.0860** and
the fully-trained public TinyStories-8M's is **0.0277**, both far below criterion 2's 0.15
bar. No student can inherit what the teacher does not have, and that is a premise decision
about the teacher, not a code fix. **Do not rewrite the 0.15 threshold now that the numbers
exist** (§2.7).

Two further readings worth carrying into the paper: fingerprint cosine to the teacher *falls*
with training (0.749 → 0.45–0.58) and the students converge on the **public 8M's** fingerprint
rather than the 4-layer teacher's; and `functional_cosine_to_teacher` ≈ 0.96 is carried almost
entirely by `int_g`/`int_h`, the two whole-subsystem ablations. On the surgical subset the
students score 0.108–0.131 while the public 8M scores **0.814** — so the subset discriminates
rather than merely being noisier. `evaluate_transformation.py` now writes
`functional_cosine_{coarse,surgical}_to_teacher` as **exploratory** columns; the pre-registered
primary metric is unchanged.

**Four defects were found by auditing the archive against the code, and all four are fixed.**
None changes a published number and none touches a pre-registration:

1. **The gate read the wrong rows** — criterion 2's `0.03460` was the step-0 *cross-domain*
   value (CLAUDE.md trap 23). Corrected it reports 0.00282 at the pilot step / 0.00993 over
   trained steps. Criterion 3's numbers were right by luck (`.iloc[0]` happened to be the
   in-domain row) and are byte-identical after the fix.
2. **`L_ATTN` was 64.5× under-scaled** (trap 22). **The pilot's D2 is not the registered
   D2**; its attention term was 0.022% of its loss. `run_config.json` now records
   `attn_reduction`. D0/D1 carry `attn_weight = 0` and are bit-identical under the fix, so
   **only D2 needs retraining.**
3. **The fingerprint cache overwrote itself per corpus** (trap 24) — all 33 archived
   `fingerprint.json` files hold `e1_100x40`. The CSV retains every headline number; what was
   lost is the per-corpus record, including the teacher's on the primary corpus.
4. **Criterion 4 passed on 3 examples where the design registers 5** (trap 25). It now reports
   `met: null` with a remedy, and `run_pilot_parity.py` supplies five.

Re-running the corrected gate on the archive **flips no verdict**: criterion 2 still false,
criterion 3 still true via `fingerprint_l1` 0.300487, criterion 4 becomes `met: null`, so
`decision` moves `hold → incomplete` and `proceed` stays `false`.

**Operator actions before any further training** (nothing here can be done in a smoke
checkout):

- Move `results/e6a/D2/seed0/` to `_exploratory/e6a_D2_seed0_underscaled_attn/` with a README
  recording its numbers. **Keep D0 (all three seeds) and D1** — their objectives were
  implemented as registered and D0's three seeds are the framing-(B) null. Both aggregators
  `rglob` without de-duplication (§2.8), so a mis-scaled D2 left in the tree is pooled into
  the paper's tables.
- Retrain **D2 seed 0 only**, in a **fresh run directory** (`assert_resumable_schedule` refuses
  reuse, correctly). Expect `l_attn` ≈ 0.32 at step 2,000 and a weighted contribution ≈ 1.4%
  of the loss — that is the check that the objective is active. Compare its held-out `l_attn`
  against D1's on the same scale.
- Re-run `run_pilot_parity.py` (five examples), then `evaluate_transformation.py --with-delta-ce`
  on the new D2, then `aggregate_transformation.py` and `check_pilot_gate.py`.
- **Use one clean commit per confirmatory run.** The pilot's artefacts carry four training SHAs
  and `git_dirty: true` on evaluation and aggregation. That is an auditability gap, *not* a
  confound: `git diff` across those commits shows only runbook documents and result artefacts,
  so no training code differed between D0, D1 and D2.

**Do not launch the nine Phase 2 runs.** `proceed: false` under every corrected reading.

---

### 2.12 The E6A-GPT2 arm — **code-complete, awaiting a pre-flight measurement**

§2.11's finding is that the TinyStories teacher has no in-domain sink, so its arm cannot
test inheritance. A **second arm** distils a teacher that does have one. Same design, same
§1.6 hyperparameters, different instrument:

| | `e6a` (existing) | `e6a_gpt2` (new) |
|---|---|---|
| teacher | TinyStories-33M, 4L×768 | **gpt2**, 12L×768 |
| student | TinyStories-8M config, random, 8L×256 | **distilgpt2 config, random, 6L×768** |
| public reference | TinyStories-8M | **distilgpt2** |
| conditions | D0/D1/D2 | **G0/G1/G2** |
| corpus | `tinystories_validation_sink_300` | **`openwebtext_validation_sink_300`** |
| layer map | `{0:1, 1:3, 2:5, 3:7}` | **`{1:0, 3:1, 5:2, 7:3, 9:4, 11:5}`** |
| teacher's sink, `e1_100x40` | 0.0860 (2.5× floor) | **0.5630 (16.5× floor, 73% of cells > 0.2)** |

**Isolation is by `experiment_id`.** Both aggregators `rglob` without de-duplicating, so the
arms would otherwise share contrasts, the matched-loss reference and — worst — the
framing-(B) null spread. Results go to `results/e6a_gpt2/`; the aggregator and the gate take
`--experiment`. `check_pilot_gate --experiment e6a` on the existing archive was verified to
produce **byte-identical** criteria to before the split.

**Decisions, all taken 2026-08-01 before any gpt2 run existed** and recorded in
`configs/e6a_gpt2_preregistration.yaml` (`e6a_gpt2_prereg_v1`). `e6_preregistration.yaml` is
**not touched** — this is a new registration, not an amendment.

| ID | Resolution |
|---|---|
| — | §8.7/§18 transcribed from `e6_prereg_v4` with D→G, P8M→PDG2 and the corpus id. Every threshold the design states is unchanged. D1/D2/D3 inherited verbatim. |
| G1 | **New criterion 2b**: a condition must reach **50% of the teacher's own sink** on the scored corpus, reported *beside* the unchanged 0.15. The teacher's sink is a property of the instrument — measurable before any student trains — so fixing a fraction now is not fitting to a result. 0.50 is recorded as judgement with no external anchor; **the paper must say so.** Emitted for `e6a_gpt2` only. |
| G2 | Corpus = `Skylion007/openwebtext`, a deterministic 400,000-document prefix for training and a disjoint 8,000-document window for validation. WikiText-103 was **rejected on the record**: the GPT-2 paper says WebText excluded Wikipedia, which would repeat the exact OOD confound this arm exists to avoid. The cap is sized for `max_steps: 10000`, not the pilot — the packing shuffle runs over `range(len(documents))`, so a smaller cap is a *different block sequence* and `--max-steps 2000` would stop being a truncation of the full run. |
| G3 | Student = randomly-initialised distilgpt2 config; reference = the real distilgpt2. **Recorded limitation:** distilgpt2 was layer-initialised from gpt2 and distilled, so it is a *differently distilled* comparand, not the independence control §8.1 describes — a weaker basis for §20's Narrative B than TinyStories-8M was. |

**Do the pre-flight before spending GPU-days.** gpt2's sink on 128-token OpenWebText blocks
is **unmeasured**; its 0.5630 is on 40-token E1 windows, and that is exactly the transfer
where the TinyStories teacher collapsed (2.53× floor cross-domain → 0.86× in-domain). Two
commands, ~40 min, no training — see RUNBOOK 01 Phase G0. If gpt2 lands near the floor,
**stop and report**; that is registered as `e6a_gpt2_stop_6` and the answer is never to move
the threshold.

**Cost — the estimate below is superseded by §2.13; read that first.** VRAM is **measured
6.50 GB** at batch 16 × 128 × accum 4 (real gpt2 teacher bf16 + random distilgpt2 student
fp32, eager), so a 24 GB card is nowhere near binding. The time estimate that stood here —
"12–20 s/step on a 3090, 7–11 h per 2,000-step run" — was derived by scaling the pilot's
4.85 s/step, and **~80% of that figure was a CPU-side shuffle, not GPU work** (§2.13). After
the fix the loop is GPU-bound: 1.96 s/step measured on an RTX 2060, so expect roughly
**0.4 s/step on a 4090 → ~1.1 h per 10,000-step run**. `runtime_estimate.json` after step 100
still carries the real rate and `peak_vram_bytes`; run G0 seed 0 first and read it before
launching the rest.

### 2.13 The training loop was CPU-bound — **fixed 2026-08-02, byte-identically**

The E6A-GPT2 Phase 2 runs (G0 on a 4090, G2 on a 5090, `--max-steps 10000`) showed **12%
and 3% GPU utilisation**. The cause was not the model, the batch size or the card.

**`epoch_order` was being recomputed once per gradient-accumulation micro-batch.** It is a
pure function of `(n_blocks, seed, epoch)` that builds `list(range(n_blocks))` and shuffles
it in Python; the loop then read sixteen elements and discarded the rest. `epoch` advances
only every `n_blocks / batch_size` micro-batches — ~194,000 for the gpt2 arm — so inside a
10,000-step run (40,000 micro-batches) it never advances at all and the *identical* 3.1M
element shuffle was rebuilt 40,000 times, single-threaded, with the GPU idle.

Measured on this workstation (RTX 2060 / Ryzen 5 5600):

| | measured |
|---|---|
| `epoch_order` at 3.1M blocks (OpenWebText) | 1.09 s → **4.36 s/step** at accum 4 |
| `epoch_order` at 3.57M blocks (TinyStories) | 1.30 s → 5.21 s/step |
| GPU-side cost of one gpt2→distilgpt2 step, 16×128 × accum 4 | 1.84–2.00 s |

So the §2.11 pilot's "4.85 s/step on a 4080 SUPER" was ~80% shuffle, and it was never a GPU
cost. Scaling the GPU term to a 4090 (~0.4 s/step) against a ~3.2 s/step shuffle predicts
11% utilisation, which is the 12% that was observed.

**Four more defects in the same loop**, all found by reading it against what it produces:

1. **88 GPU→CPU synchronisations per optimiser step.** Four `.item()` calls in
   `compute_losses` and three per mapped layer pair in `attn_js_loss` (six pairs on the
   gpt2 arm), × 4 micro-batches. Each drains the CUDA queue, so kernel launches could never
   run ahead. Now **5**, via stacked `.tolist()`.
2. **The per-pair breakdown was computed four times per step and three copies discarded** —
   `_average_components` keeps only `rows[-1]`, and `evaluate` ignores it entirely.
3. **`L_ATTN` was built inside the autograd graph even at weight 0** (D0/D1/G0/G1), where
   design §8.6 makes it a logged metric that never reaches `backward`. Measured **11%** of
   step time.
4. **`evaluate` ran four forwards per batch where two suffice** — it re-ran both models
   without attentions purely to get logits `compute_losses` already had. Both models are
   built with `attn_implementation="eager"`, so `output_attentions` only decides whether the
   probabilities are *returned*; verified numerically rather than assumed
   (`torch.equal(logits_with, logits_without)` is `True`, max |Δ| exactly `0.0`, on real
   gpt2 in bf16 and a random distilgpt2 in fp32).

**Every change is value-preserving, and that was gated rather than argued.** Goldens were
captured *before* any edit — an offline `--smoke` run and two real CUDA runs (real gpt2
teacher → randomly-initialised distilgpt2 student, 20 steps, G2 weights and G0 weights) —
and re-captured after. All **12 artefacts** (`train_log.jsonl`, `eval_log.jsonl` and the
step-0/step-20 `checkpoint_sha256.txt` of each) are **byte-identical**, including the fp32
weights after 20 real optimiser updates. Consequences:

- **The G0 seed-0 and G2 seed-0 runs already spent on the 4090/5090 stay valid** and pool
  with everything produced afterwards. If either is still in flight, resume it in place with
  `--resume auto --max-steps 10000`; the trajectory is unchanged.
- `per_device_batch_size: 16` / `grad_accum: 4` are **untouched** — §1.6 pre-registers them,
  and after the fix the GPU is ~96% busy at that size anyway. Peak VRAM is unchanged at
  6.50 GB, so a 24 GB card was never the constraint.
- TF32, autocast and `torch.compile` were considered and **rejected**: all change numerics.

**Measured result**, real gpt2→distilgpt2 at 3.1M blocks, 40 steps, RTX 2060:

| | before | after |
|---|---|---|
| s/step | 6.957 | **1.957** |
| GPU utilisation, mean / median | 29.0% / 29% | **95.8% / 99%** |
| peak VRAM | 6.495 GB | 6.495 GB |

1.96 s/step is the isolated GPU-side cost, so the loop is now GPU-bound — there is no
remaining CPU stall to remove at this batch size. Expect a 4090 to land near **0.4 s/step**
(~1.1 h per 10,000-step run, against ~10 h before) and a 5090 lower still.

**Corpus packing, the other single-threaded phase.** Two additive changes, neither of which
moves a token or a digest:

- `datasets_loader._pack_documents` gained `tokenizer_batch_size` (default `None` =
  one call per document, exactly as before), so a fast tokenizer batches across cores.
  Offered on the **OpenWebText loader only** — `load_tinystories_blocks` is deliberately
  untouched, because that arm's packed records are already on disk and the only safe number
  of ways to produce a `manifest_sha256` is one (trap 19).
- New `common/block_corpus_cache.py` persists a packed corpus under a key covering
  everything that can change its content — and **nothing that cannot**: `n_blocks` is
  excluded because packing is a prefix operation, and `tokenizer_batch_size` is excluded
  because it is proven not to change the output, so the trainer (which batches) and a corpus
  provider (which does not) share one entry instead of each re-streaming the window.
  **Every hit re-derives `prov.sha256_int_rows` over the blocks it just read and raises on
  a mismatch.** Wired into `train_distillation.py` and `corpus_providers.openwebtext_corpus`
  behind `--corpus-cache <dir>|none` (default `<repo>/.corpus_cache`, gitignored), and
  `run_config.json` gained a `corpus_cache` block recording the key and whether each corpus
  was packed or read. This also removes the 400,000-document stream-and-discard that every
  `evaluate_transformation.py` invocation paid to reach the OpenWebText validation window.

**Tests: 640 passed** (was 538), all nine offline smoke programs exit 0. The 102 new
assertions are in `tests/test_epoch_order_cache.py` (the memoised order *is* the order,
including the wraparound index expression, and the cached array is not mutable),
`tests/test_batched_tokenisation_equivalence.py` (batched == unbatched blocks, manifest and
digest at six batch sizes, on the stub **and on the real gpt2 tokenizer**), and
`tests/test_block_corpus_cache.py` (warm hit == cold pack, prefix serving, every key field,
and the refusals — a corrupted block file raises rather than being served). The suite caught
one defect in this work: forcing batched tokenisation inside `openwebtext_corpus` required
every caller's tokenizer to accept a list, for a pack too small to benefit. It was reverted
there and kept in the trainer.

**Operational note, unrelated to the code:** the `C:` drive on this workstation reached
**0 bytes free** during the benchmark round, which surfaces as an opaque torch
`"unexpected pos"` error from the safetensors/zip writer rather than as `ENOSPC`. 8 GB of
that was this session's own scratch; it has been released. Watch free space before a long
run — a checkpoint write that fails at step 9,000 costs the run.

### 2.14 Canonical GPT-2-medium -> GPT-2-small and the alignment bridge — **code-complete**

The paper-facing scale arm is now prospectively isolated as `e6a_gpt2_medium_small`:

| field | registered value |
|---|---|
| teacher | pinned `openai-community/gpt2-medium`, 24 layers / 16 heads / 1024 hidden |
| student | random canonical `openai-community/gpt2` config, 12 / 12 / 768 |
| public reference | pinned public GPT-2-small weights, condition `PG2S`, never an initializer |
| conditions | G0 CE; G1 CE+logit-KD; separately named `G2-aligned` |
| layer map | every second teacher layer, `{1:0, 3:1, ..., 23:11}` |
| attention method | teacher-to-student cosine-soft alignment at temperature 1.0, attached gradients, followed by the unchanged E6A JSD-form divergence and reduction |

The method is registered and documented as an **AMAD-style JSD extension**, never exact
AMAD: it borrows the soft head alignment but intentionally does not replace E6A JSD with
AMAD Variant 1's normalized-MSE objective or the paper's KL variants. Existing configs omit
`loss.head_alignment` and therefore remain on legacy index-JSD. G0/G1 optimization weights, data, optimizer, schedule,
checkpoints, seeds, and evaluator design are unchanged.

`e6a_gpt2_alignment_bridge` is a second prospective experiment on the equal-head
GPT-2 -> DistilGPT-2 pair. It freshly runs G0/G1/G2-legacy/G2-aligned, with the two G2
configs differing only in the alignment method. No completed `e6a_gpt2` artifact is reused
or reinterpreted, and the direct aligned-minus-legacy estimand has uncertainty but no
post-hoc success threshold.

Both arms pin teacher, student-config, tokenizer, and public-reference revisions; startup
asserts the exact model geometry and layer map. The evaluator forwards all revision pins.
The primary arm refuses the head-index carrier metric honestly: 16-vs-12
`carrier_jaccard_to_teacher` is null with a warning, while permutation-invariant fingerprint
and topology measures remain confirmatory. Pilot-gate dispatch is isolated by experiment id,
uses OpenWebText, includes the teacher-relative criterion 2b, and compares the first
condition with the last (`G2-aligned`) even in the four-condition bridge.

Verification on 2026-08-03: **685 pytest tests passed**. The new offline end-to-end smoke
preserved exact depth/head geometry at scaled width, proved identical per-seed initialization,
trained primary and bridge G2 conditions, evaluated all primary rows, emitted the unequal-head
carrier warning, aggregated under the new preregistration, exercised the arm-specific gate,
and proved no rows leak into original `e6a`.

What remains is real compute, not implementation: teacher preflight, 2,000-step pilot,
parity/gate, then fresh-root 10,000-step runs for all three primary and all four bridge
conditions at three seeds. Runtime, peak VRAM, and checkpoint storage for GPT-2-medium are
unmeasured; run G0 seed 0 first and inspect step-100 telemetry. The authoritative command
sequence is `RUNBOOK/03_E6A_GPT2_MEDIUM_SMALL.md`.

---

## 3. What is left

**No work package is left. Every one in the master-plan DAG — WP0–WP12 — is shipped**, and
`e6b_drift.csv`, `e6b_early_warning.csv`, `e6b_factorial.csv` and `e6b_go_no_go.json` carry
real rows in the offline smoke rather than the header-only placeholders they held since
WP10. §2.3 is closed and §2.7's decisions are made, so **what remains is GPU time and one
Hugging Face credential** (FLORES's gate — §1.6).

`COMMANDS.md` is the runnable version of everything in this section, in dependency order,
with the resumability and flag caveats inline, plus the **measured** per-run costs on this
hardware (§2.10). The commands below are kept for context.

**Scheduling reality on the validation box (12 GB RTX 2060, 15.9 GB RAM):** E6A Phase 2 is
~6.9 GPU-days, the 5-run pilot ~20 h. E6B and E7 are on top. A 4080 SUPER should be roughly
2–4× faster. FLORES additionally needs a Hugging Face token (§1.6).

E6B stays *scientifically* conditional (`00` §6 puts it outside the minimum publishable
version) — but it is no longer conditional on code.

### What E6B (WP6) needs from the compute PC

4 conditions × 3 seeds × 3 epochs on SST-2 (~67k train rows), `distilbert/distilgpt2`.
Small next to E6A, and `screen_public_pairs.py` should run **first** — `07` P6 calls it the
cheapest evidence in the project and it needs no training at all.

```bash
# 0. the free evidence: does a public distilled / instruction-tuned pair share a
#    fingerprint? No training, one command (03 §5 / design §9.9).
./.venv/Scripts/python.exe transformation_inheritance/screen_public_pairs.py

# 1. the 2x2, three seeds each
for c in f1 f2 f3 f4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_sentiment_adaptation.py \
      --config transformation_inheritance/configs/e6b_$c.yaml --seed $s
  done
done

# 2. fingerprint every checkpoint AGAINST THE UNTRAINED BASE — `--base` is what fills
#    the three *_drift_from_base columns, and design §9.8's early-warning analysis is
#    nothing without them. Absent it they stay empty (never 0.0) with a recorded reason.
for c in F1 F2 F3 F4; do
  for s in 0 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6b/$c/seed$s \
      --base distilbert/distilgpt2 --dtype bfloat16
  done
done

# 3. the same aggregator as E6A; it writes e6b_go_no_go.json alongside go_no_go.json
./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py
```

- **`e6b_go_no_go.json` reports `incomplete` until decision D4 is made** (`e6b_1`,
  `e6b_2`), no matter what the numbers say. §18 states the E6B criteria as a conjunction,
  so `combine: all` is in the YAML and `e6b_3`'s early-warning quantifiers are the design's
  own and need no decision.
- **A LoRA run's checkpoint is `step_<n>/{adapter,merged}/`** and only `merged/` is
  fingerprinted — the frozen GPT-2 harness cannot resolve module paths through a PEFT
  wrapper. `fingerprint_dir()` picks it automatically; you do not pass anything.
- **`merge_parity.json` must say `passed: true` at every checkpoint.** A failure aborts the
  run, and at float64 round-off is ~1e-13, so it means a systematic merge fault — check the
  `alpha/r` scale and the `Conv1D` transpose, **not** the tolerance (§1.5).
- **Watch the label-token report** in `run_config.json`. If `" positive"`/`" negative"`
  tokenise to more than one token under a different tokenizer, scoring runs over the full
  span; the count is measured at startup, never assumed.

### What E6 (WP5/WP10) still needs from the compute PC

- **Budget host RAM and ~20 minutes of corpus packing per run.** Every training run repacks
  the TinyStories train split: 2,119,719 stories → 3.57M blocks, ~17–30 min of tokenisation
  and ~3 GB of RSS *before the first optimiser step*. That is after the §1.6 memory fix;
  without it the same build peaks near 29 GB and will freeze a 16 GB host. Watch RSS on the
  first run of a fresh checkout.
- **E6A training itself.** ~1.5–3 GPU-days for 3 conditions × 3 seeds × 10k steps. Only
  the 5-step `--smoke` path has run here. The seed-0 pilot, end to end:
  ```bash
  # 1. train the three conditions to the 2,000-step pilot gate
  for c in e6a_ce e6a_logit_kd e6a_logit_attention_kd; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
  done

  # 1b. D0 at seeds 1 and 2 — REQUIRED by framing (B). e6a_3/e6a_4 calibrate on D0's
  #     across-seed spread and resolve_threshold_rule REFUSES fewer than three seeds,
  #     so with seed 0 alone both stay met: null. See §2.7.
  for s in 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
      --config transformation_inheritance/configs/e6a_ce.yaml --seed $s --max-steps 2000
  done

  # 2. fingerprint every checkpoint (resumable; re-run it freely after an interruption)
  for c in D0 D1 D2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6a/$c/seed0 \
      --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
  done
  for s in 1 2; do
    ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
      --run-dir transformation_inheritance/results/e6a/D0/seed$s \
      --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
  done

  # 2b. the seedless public reference (08 §4) — design §8.7 contrast 4 and criterion e6a_4
  #     need it, and --with-delta-ce is REQUIRED or its validation_ce is empty and the
  #     matched-loss selection has nothing to target. Run it once, not once per condition.
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --config transformation_inheritance/configs/e6a_ce.yaml \
    --evaluate-public-reference roneneldan/TinyStories-8M \
    --with-delta-ce --dtype bfloat16

  # 3. contrasts, matched loss, go/no-go  (also writes e6b_go_no_go.json)
  ./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py

  # 4. the parity report criterion 4 reads — nothing else produces one for a checkpoint
  ./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
    --run-dir transformation_inheritance/results/e6a/D0/seed0 --step 2000

  # 5. the gate
  ./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
    --seed 0 --pilot-step 2000 \
    --parity-report transformation_inheritance/results/e6a/D0/seed0/parity/step_2000/parity_report.json
  ```
  All four criteria now have inputs. `run_pilot_parity.py` stages a copy of the checkpoint
  with the run's tokenizer beside it — `save_pretrained` writes none, and the frozen Neo
  harness loads model and tokenizer from the same path (trap 7) — then calls the frozen
  `run_parity_check` on the design's **five** examples, with the manual runner, swap
  directions and massive coordinates all built by the frozen Neo module's own functions. The
  frozen CLI would run three (trap 25). Without `--parity-report` criterion 4 stays
  `met: null` and the gate cannot proceed. **Do not launch Phase 2 without `proceed: true`.**
- **`go_no_go.json` reads `incomplete` until D0 has three seeds**, no matter what the
  numbers say — not because a decision is open (all seven are made, §2.7) but because
  `e6a_3`/`e6a_4` calibrate on D0's across-seed spread and the rule refuses fewer than
  three. `pending_decisions_blocking` should be empty; the signal to read is `n_unknown`
  and each criterion's `threshold_rule.reason`.
- **Framing (B) was taken for D2/D3/D4**, so the pilot runs before the thresholds resolve.
  That is by design: the rule is the pre-registration, the number is a result, and
  `go_no_go.json` records both — the realised `spread`, the `null_seed_values` it came from,
  and the derived `threshold`. Quote the rule and the number together.
- **Watch `evaluation_summary.json`** after each evaluation: `n_failed > 0` with
  `n_written == 0` means every unit failed, and the exit code is non-zero. A few failed
  units are written as rows with `status` set and are excluded from contrasts by the 2%
  rule in the aggregator — check `aggregate_summary.json["exclusions"]` before reading a
  table.
### What E7 (WP8/WP9/WP11) still needs from the compute PC

- **FLORES-200 is GATED and needs a credential before anything downloads.** Measured
  2026-07-30 on `datasets==4.8.4`: `load_dataset("facebook/flores", "eng_Latn")` raises
  `DatasetNotFoundError: ... is a gated dataset on the Hub`. Accept the terms at
  <https://huggingface.co/datasets/facebook/flores>, then `hf auth login` (or set
  `HF_TOKEN`) once per machine. **Do not go looking for a mirror** — `Muennighoff/flores200`
  and `gsarti/flores_101` are loading-script datasets and `datasets>=3.0` refuses those
  outright, and `openlanguagedata/flores_plus` is gated too; all three were tried.
  `load_flores_parallel` now raises with these instructions rather than an opaque HF error.
  **XNLI is not gated** (both the per-language and `all_languages` configs load, 5,010 test
  rows), so the patching half of E7 is unaffected.
- **Real FLORES / XNLI manifests.** `prepare_flores_manifest.py` and
  `prepare_xnli_manifest.py` need downloads. Check the printed `join_strategy` on the XNLI
  run: if it says `all_languages_structural`, the shards carried no `promptID` and the
  alignment came from the structurally aligned config — record that in the paper's data
  section rather than claiming a promptID join.
- **The Phase 1 E7 study**, in order. Every step is resumable; re-run any of them freely
  after an interruption.
  ```bash
  # 0. manifests (downloads; writes patch_controls.csv and partitions.json beside them)
  ./.venv/Scripts/python.exe crosslingual_semantics/prepare_flores_manifest.py \
    --tokenizer Qwen/Qwen2.5-0.5B --n 300 --seed 42
  ./.venv/Scripts/python.exe crosslingual_semantics/prepare_xnli_manifest.py \
    --tokenizer Qwen/Qwen2.5-0.5B --n 600 --seed 42

  CFG=crosslingual_semantics/configs/e7_qwen05_base.yaml

  # 1. representations (04 §3) — derived tensors only, streamed to .npy memmaps
  ./.venv/Scripts/python.exe crosslingual_semantics/extract_sink_representations.py \
    --config $CFG --langs all

  # 2. retrieval (04 §4) — the language probe is the expensive part; --probe-layers
  #    restricts it without touching the retrieval tables
  ./.venv/Scripts/python.exe crosslingual_semantics/evaluate_crosslingual_retrieval.py \
    --reps crosslingual_semantics/results/reps/qwen05_base \
    --manifest crosslingual_semantics/results/manifests/flores_devtest.json

  # 3. the baseline multilingual fingerprint + its §10.5 controls (04 §3.4)
  for v in matched unmatched length_matched; do
    ./.venv/Scripts/python.exe crosslingual_semantics/run_multilingual_fingerprint.py \
      --config $CFG --variant $v
  done

  # 4. the 04 §5.5 Phase 1 patch smoke — 50 examples, 4 screening layers — FIRST.
  #    It writes runtime_estimate.json; check it before committing to the full screen.
  ./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
    --config $CFG --stage smoke

  # 5. screen the four layers on the 200 DEV examples -> layer_selection.json
  ./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
    --config $CFG --stage screen

  # 6. evaluate the five-layer window on the disjoint 400 TEST examples
  ./.venv/Scripts/python.exe crosslingual_semantics/run_cross_language_patching.py \
    --config $CFG --stage window

  # 7. contrasts, claim gate, go/no-go  (--retrieval supplies criteria e7_1 and e7_2)
  ./.venv/Scripts/python.exe crosslingual_semantics/aggregate_crosslingual.py \
    --retrieval crosslingual_semantics/results/retrieval
  ```
  Then repeat steps 1–6 with `e7_qwen05_instruct.yaml`, `e7_qwen15_base.yaml` and
  `e7_qwen15_instruct.yaml` — the §6.4 claim gate needs **two model sizes**, and contrast
  `e7_c5` needs both variants, or they report `no_data`.
- **A non-zero exit from `run_cross_language_patching.py` means an identity control
  failed.** That is not a warning: `invalid_units.json` names the run units, the aggregator
  excludes their rows, and the affected (language, object, layer) sites must be
  re-measured. Do not aggregate around it.
- **Watch the failure rate.** `aggregate_summary.json` carries
  `max_failure_rate_observed`; any contrast above 2% is written with
  `status="excluded"` and its reason rather than computed. `--allow-high-failure` is a
  deliberate override, not a default.
- **`--stage window` refuses to run without `layer_selection.json`**, and the dev/test
  partitions are asserted disjoint at startup. That refusal is the audit trail for
  "selected on dev, reported on test" — do not work around it.

### Minimum publishable path (master plan §6) — **code-complete**

`WP0–WP5 ✓ → WP10 ✓ → WP12 ✓` for E6 and `WP7 ✓ → WP8 ✓ → WP9 ✓ → WP11 ✓ → WP12 ✓` for
E7. Both halves of design §21's minimal publishable version are code-complete, the §2
bridge gates are green, and the pre-registrations are transcribed. What remains is GPU
time, §2.7's seven decisions, and §2.3. WP6 (E6B) and the 3B confirmation stay
conditional.

---

## 4. Hard rules that still apply

- The 12 files in `BASELINE_HASHES.json` are **frozen** — new code calls into them.
- `common/nnsight_engine.py` and `common/datasets_loader.py` are **additive-only**.
- Never fabricate numbers; failures are written as rows with `status`/`warning`.
- Every output artefact carries provenance (git sha, `manifest_sha256`, seed, dtype,
  device, registry version) — see `05_SCHEMAS_AND_CONTRACTS.md`.
- New tests must be Windows-safe at teardown (release model + `gc.collect()` or
  `TemporaryDirectory(ignore_cleanup_errors=True)`).
- The identity patch is a **continuous** correctness monitor, not only a test
  (`04` §5.4): any production E7 row with `status="identity_violation"` invalidates its
  run unit.
- Thresholds and criteria live in `configs/e6_preregistration.yaml`, never in analysis
  code. Changing one after seeing results requires an entry in its `amendments:` list with
  a timestamp and a reason (`05` §6).
