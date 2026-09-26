# WP10 — `evaluate_transformation.py` + `aggregate_transformation.py`

## Context

E6A's trainer (WP5) is finished and produces checkpoints, but **nothing turns those
checkpoints into numbers**. `check_pilot_gate.py` already reads a
`checkpoint_metrics.csv` that no code writes, so criteria 2 and 3 of the pilot gate
report `met: null` and `proceed` can never become true. WP10 is the missing link and,
per `00_MASTER_PLAN.md` §6, the last piece of the minimum publishable E6 path
(`WP0–WP5 ✓ → WP10`).

Two deliverables, per `03_MODULE_SPEC_e6_transformation.md` §6–7:

- `evaluate_transformation.py` — checkpoint → `FingerprintRecord` → one CSV row per
  `(run_id, step, corpus_id)`, resumable, written to the schema in `05` §2 exactly.
- `aggregate_transformation.py` — all `checkpoint_metrics.csv` → contrasts, matched-loss
  selection, drift tables, figures, `go_no_go.json`.

**The one thing this cannot do honestly.** `03` §7 requires `e6a_contrasts.csv` to carry
"the five primary contrasts (design §8.7)" and `go_no_go.json` to carry design §18's
criteria "transcribed as literal code before any results exist — that is the
pre-registration." *The design document is not in this repo* — only the spec pack that
cites it. The single criterion whose full text I do have is the worked example in `05` §6
(`e6a_1`: D2 improves teacher fingerprint similarity over D0 by ≥ 0.15). Inventing the
rest would fabricate a pre-registration, which CLAUDE.md rule 4 forbids. Resolution
(confirmed with the user): the criteria live in a declarative YAML, sourced entries are
filled, unsourced entries are explicit `PENDING_DESIGN_18` sentinels that emit
`met: null` + reason and force `decision: "incomplete"`. Pasting §8.7/§18 later changes
only the YAML.

---

## Files

### New

| Path | What |
|---|---|
| `transformation_inheritance/evaluate_transformation.py` | `03` §6 |
| `transformation_inheritance/aggregate_transformation.py` | `03` §7 |
| `transformation_inheritance/configs/e6_preregistration.yaml` | contrasts + go/no-go, declarative |
| `tests/test_evaluate_transformation.py` | resume / force / failures-are-rows |
| `tests/test_matched_loss_selection.py` | earliest-checkpoint rule, 0.05-nat refusal |
| `tests/test_aggregate_contracts.py` | hash + registry guards, 2% failure-rate refusal |
| `tests/nnsight_e6_eval_smoke.py` | offline end-to-end: train → evaluate → aggregate |

### Modified (all additive, none frozen)

| Path | Change | Why |
|---|---|---|
| `common/fingerprint_runner.py` | new kwarg `delta_ce_per_item=False` (default reproduces current behaviour byte-for-byte); when true, per-item ΔCE lists land in `record.provenance["delta_ce_per_item"]`. Flag joins `_cache_key`. | `07` P10 requires bootstrapping the 300-block subset's sampling error; `_delta_ce` currently returns means only. Record schema and existing cache keys unchanged. |
| `transformation_inheritance/train_distillation.py` | `write_run_config` gains `teacher`, `teacher_revision`, `student_config_from`, `config_path` | `run_config.json` records no teacher today, so the evaluator cannot find the model it must compare against. Adding fields is permitted (`05` §5: "A column may be added"); no Phase 2 run has started. `--teacher`/`--config` remain as CLI fallbacks for run dirs written before this. |
| `scripts/run_nnsight_rest_smoke_tests.ps1` | append the E6-eval smoke | it is the only thing that executes `nnsight_*_smoke.py` (pytest does not collect them) |
| `NEXT_STEPS.md`, `CLAUDE.md` | status + decisions | as instructed |

---

## `evaluate_transformation.py`

```bash
python transformation_inheritance/evaluate_transformation.py \
  --run-dir transformation_inheritance/results/e6a/D2/seed0 \
  --steps all --engine nnsight --with-delta-ce --resume
```

Flags: `--run-dir`, `--steps all|0,250,500`, `--engine nnsight`, `--with-delta-ce`,
`--resume` (default) / `--force`, `--config`, `--teacher`, `--cache-dir`,
`--allow-band-mismatch`, `--device`, `--dtype`, `--smoke`.

Flow per checkpoint, reusing existing code rather than adding any:

1. **Arch** from the checkpoint's own `config.json` `architectures[0]`
   (`GPTNeoForCausalLM`→`neo`, `GPT2LMHeadModel`→`gpt2`, `Qwen2ForCausalLM`→`qwen`,
   `OPTForCausalLM`→`opt`); unknown → abort naming the mapping.
2. `fingerprint_runner.load_handle(arch, ckpt, engine=…, dtype=…)`, with
   `checkpoint_sha256` read from the checkpoint's `checkpoint_sha256.txt`.
3. `band = depth_band.normalised_depth_band(num_layers)`.
4. Corpora from `corpus_providers`: in-domain `tinystories_corpus(purpose="sink", 300)`,
   cross-domain `frozen_e1_corpus`, and for E6B `sst2_prompt_corpus(split="validation")`.
   `--smoke` substitutes `synthetic_corpus` (fully offline).
5. ΔCE per design-delta D3 — `tinystories_corpus(purpose="ppl", 300)` at every checkpoint,
   the 2,000-block set only at step 0 and the final step. Both reported; the 300-block
   subset carries a bootstrap CI from the new `delta_ce_per_item` path via
   `inheritance_metrics.bootstrap_ci`, labelled `sampling_over_corpus_items`.
6. Teacher fingerprint computed **once** per `(corpus_id, band)`, cached under
   `<results>/e6a/teacher/fingerprints/`; comparison keys are
   `mutual_interventions(teacher_handle, student_handle)` — never a hard-coded list (D2).
7. Distances via `inheritance_metrics`: `fingerprint_report`, `topology_report`,
   `carrier_report`, `functional_report` → the `*_to_teacher` columns of `05` §2.
8. One row per `(run_id, step, corpus_id)` appended to
   `<run_dir>/checkpoint_metrics.csv`, `05` §2 columns verbatim, nested values as JSON
   strings, plus added `status` / `warning` / `n_failed` (adding columns is allowed;
   renaming is not).

**Guards, run continuously not just in tests** (`06` §4): `raw["int_a"] > 1e-8` (already
inside `compute_fingerprint`); `manifest_sha256` and `intervention_registry_version` must
match between teacher and student or the comparison raises (`05` §7.1); band depth
mismatch > 0.10 via `depth_band.band_agreement_report` raises unless
`--allow-band-mismatch`, which is then recorded in the row.

**Resumability.** The spec names the unit `(run_id, step, corpus_id, intervention)`. It is
implemented at `(run_id, step, corpus_id)` granularity with the intervention set recorded
per unit in `<run_dir>/eval_units.jsonl`, because `NNsightEngine.run_all` computes the
whole a–j battery in one forward sweep per item — splitting resumption below that would
*increase* total forwards, not reduce them. A unit is skipped when its CSV row exists and
its ledger entry matches on intervention set, `manifest_sha256`, registry version, band
and `checkpoint_sha256`; any mismatch recomputes. This deviation gets a comment in the
module and a line in `NEXT_STEPS.md`. `--force` recomputes and rewrites the unit's row in
place (no duplicate rows).

**Failures are data.** A checkpoint that fails to load, a corpus that fails wholesale, an
OOM — each writes a row with `status` set and metric columns as empty sentinels, then the
run continues to the next unit.

---

## `aggregate_transformation.py`

Reads every `checkpoint_metrics.csv` under `--results`, writes to `<results>/aggregate/`.

Pre-registration lives in `configs/e6_preregistration.yaml`:

```yaml
contrasts:
  - id: e6a_c1
    text: "D2 vs D0, fingerprint cosine to teacher, equal step"
    metric: fingerprint_cosine_to_teacher
    condition_a: D2
    condition_b: D0
    family: e6a_primary
  # … the mechanically-derivable D-pairings, then:
  - id: e6a_c5
    status: PENDING_DESIGN_8_7        # emits met/observed null + reason, never a number
go_no_go:
  - id: e6a_1
    text: "D2 improves teacher fingerprint similarity over D0 by >= 0.15"
    metric: fingerprint_cosine_to_teacher
    threshold: 0.15                   # sourced: 05 §6
    direction: greater_equal
  - id: e6a_2
    status: PENDING_DESIGN_18
```

Outputs:

| File | Notes |
|---|---|
| `e6a_contrasts.csv` | per seed and pooled; `paired_seed_contrast` carries `min_attainable_p` and its descriptive-not-inferential note |
| `e6a_matched_loss.csv` | equal-step vs matched-loss with realised CE gap |
| `table2_inheritance_components.csv` | topological / mechanistic / functional as separate columns — no composite score (`02` §4 forbids one) |
| `e6b_drift.csv`, `e6b_early_warning.csv`, `e6b_factorial.csv`, `table3_clean_vs_corrupt.csv` | built now; header-only + recorded `reason: "no e6b runs found"` until WP6 lands |
| `figures/fig2_trajectories.pdf`, `fig3_matched_loss.pdf`, `fig4_drift.pdf` | Agg backend, `dpi=200, bbox_inches="tight"` — `emergence_dynamics_analysis.py` exposes no importable style helper, so its conventions are matched literally and that is recorded |
| `go_no_go.json` | `05` §6 shape; any `PENDING_*` criterion ⇒ `met: null` and `decision: "incomplete"` |

**Matched-loss selection** (`03` §6.4): for each condition the **earliest** checkpoint
whose `validation_ce` is closest to D0's final CE; record the realised gap; if the gap
exceeds **0.05 nats** the row is written with `comparable: false` and a reason and is
excluded from the contrast — flagged, never silently compared. Same logic serves E6B's
matched-accuracy comparison.

**Statistics** — all via `inheritance_metrics`, nothing reimplemented:
`paired_seed_contrast` (exact sign-flip, `min_attainable_p = 0.25` at n=3),
`bootstrap_ci` / `hierarchical_bootstrap` labelled `uncertainty_kind:
"sampling_over_corpus_items"` and never as training uncertainty, `bh_correct` within each
`family` declared in the YAML.

**Refusals** (`05` §7.2): rows whose `n_failed / n_items` exceeds 2% are excluded from
contrasts unless `--allow-high-failure`; mismatched `manifest_sha256`,
`intervention_registry_version` or `n_keys_used` raise rather than pool.

---

## Tests

Following `06` §5 — invariants, determinism, and agreement with the frozen instrument;
never that a scientific quantity takes a particular value.

- `test_evaluate_transformation.py` — a second `--resume` run adds no rows and performs no
  forwards (assert via the ledger + a call counter); `--force` recomputes and replaces in
  place; a deliberately broken checkpoint writes a `status` row instead of raising; a
  registry-version mismatch raises.
- `test_matched_loss_selection.py` — earliest-not-nearest tie-breaking; realised gap
  recorded; a 0.06-nat gap yields `comparable: false`, a 0.04-nat gap does not.
- `test_aggregate_contracts.py` — `PENDING_*` criteria give `met: null` and
  `decision != "continue"`; 3% failure rate refuses without the override and passes with
  it; BH correction applied per family; no composite score column anywhere.
- `tests/nnsight_e6_eval_smoke.py` — real end-to-end on random tiny GPT-Neo models,
  CPU/fp32, no network: `train_distillation --smoke` → `evaluate_transformation --smoke`
  → `aggregate_transformation`, asserting the CSV carries every `05` §2 column and that
  `check_pilot_gate.py` can now read it. Windows-safe teardown (release model +
  `gc.collect()`, `ignore_cleanup_errors=True`).

## Verification

```bash
./.venv/Scripts/python.exe -m pytest tests/ -q          # expect 154 passed + new, 2 skipped
powershell -File scripts/run_nnsight_rest_smoke_tests.ps1 -Python ./.venv/Scripts/python.exe
```

Both must be green, with `tests/test_frozen_files.py` proving no frozen file moved. Then
the manual end-to-end above is run once by hand and its `pilot_gate.json` inspected —
criteria 2 and 3 must flip from `met: null` to real booleans, which is the concrete proof
that WP10 closed the gap. Real thresholds still cannot be judged until the compute PC runs
the §2 bridge gates.

## Doc updates

`NEXT_STEPS.md`: WP10 moves to the implemented table; record the resumability-granularity
deviation, the `delta_ce_per_item` and `run_config.json` additive changes, and the
`PENDING_DESIGN_8_7` / `PENDING_DESIGN_18` sentinels as a blocking input the user must
supply. `CLAUDE.md`: add `evaluate_transformation.py` / `aggregate_transformation.py` to
the module map and note that the pre-registration YAML is the only place thresholds live.
