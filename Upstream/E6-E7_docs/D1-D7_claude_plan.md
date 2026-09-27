# Resolve D1–D7, fix the blockers, run smoke + pilot

## Context

Every work package (WP0–WP12) is shipped and the suite is green (**467 passed**, verified
this session — `NEXT_STEPS.md` §2.4 still says 466). What stands between the repo and
paper-ready experiments is not code volume: it is eight undecided pre-registration
sentinels, an unfinished `BASELINE_HASHES.json`, and three defects that only bite on first
contact with real runs.

`getting-paper-ready.md` was checked claim by claim. Two of its four findings are correct
as stated, one is **wrong on this checkout but right about the future**, and one is correct
and now scoped away by your D7 choice:

| Claim | Verdict |
|---|---|
| 1. Preregs still v3, 8 sentinels | **TRUE.** e6: `primary_topology_metric`, `e6a_3`, `e6a_4`, `e6b_1`, `e6b_2`. e7: `e7_c6`, `criterion_3_combine`, `interpretation_matrix`. |
| 2. All 12 frozen files report DRIFT | **FALSE here.** I re-hashed all 12: every one is a **RAW-MATCH**, all contain CRLF, `core.autocrlf=true`, and `test_frozen_files.py` passes. That chat was working on an LF-normalised copy. **But the portability risk is real** — there is no `.gitattributes`, the test hashes raw bytes, and on any box with `autocrlf=false` all 12 would fail, indistinguishable from a real edit. Fix warranted; diagnosis was not. |
| 3. `environment` empty | **TRUE**, and now capturable. |
| 4. E7 has no `threshold_rule` support | **TRUE** — 0 occurrences in `aggregate_crosslingual.py` vs 7 in the E6 one. Moot given D7 = fixed. |

Two defects it missed, both found by inspection this session:

- **`check_pilot_gate.criterion_4` cannot ever pass.** The frozen `run_parity_check` writes
  `{all_rows_pass, rows, reference, atol, rtol}` ([nnsight_engine.py:1259](common/nnsight_engine.py#L1259), confirmed against
  three real reports in `nn_results/parity_reports/`). `criterion_4` reads `passed`, then
  falls back to `all_within_tolerance` — **neither key exists** — so it computes
  `bool(None→False)` and reports `met: False` on a *passing* report
  ([check_pilot_gate.py:211-213](transformation_inheritance/check_pilot_gate.py#L211)). The pilot gate can never say `proceed: true`, and §8.5 forbids
  launching Phase 2 without it.
- **Nothing can produce that report for a student checkpoint.** The Neo harness's
  `--verify-parity` path does `AutoTokenizer.from_pretrained(args.model_name)`
  ([intervention_analysis_neo.py:937](cross_scale_and_architecture/neo/intervention_analysis_neo.py#L937)) and a checkpoint dir holds no tokenizer (trap 7).

Also: this box is an **RTX 2060, 12 GB** — not the 16 GB 4080 the design assumes. bf16
works (measured 3.25 vs 0.73 TFLOPS fp32, so the configs' `precision: bfloat16` is the
right choice, not merely tolerable).

## Decisions taken

| ID | Resolution |
|---|---|
| D1 | `primary_topology_metric: topology_wasserstein_to_teacher` — depth-location sensitive; matches §24's "anchors as conserved roles". |
| D2 | `e6a_3` framing **B**, `kind: null_seed_spread`, `null_condition: D0`, `anchor: 1.0`, `k: 2.0`, `statistic: range`. |
| D3 | `e6a_4` framing **B**, same null, `anchor: 0.0`, `k: 2.0`, `statistic: range`, **`anchor_side: above`** (see fix 3). |
| D4 | `e6b_1: 0.02` (SST-2 binomial SE anchor); `e6b_2` framing B, `null_condition: F1`, `anchor: 0.0`, `k: 2.0`, `statistic: std`, `anchor_side: above`. |
| D5 | **Route C** — `e7_c6` registered exploratory, tiers left empty, sentinel deleted. The Qwen2.5 report §3.1 gives no per-language token share and no named language list (I read it: arXiv 2412.15115v2 pp. 3–4). |
| D6 | `criterion_3_combine: any`. |
| D7 | `semantic_sensitivity: 0.10`, `unrelated_transfer: 0.10`, both fixed; `unrelated_transfer` recorded as judgement with no design anchor. |

---

## Part 1 — Code fixes (four, all additive, each with a test)

**Fix 1 — `criterion_4` key mismatch.** `transformation_inheritance/check_pilot_gate.py`.
Read `all_rows_pass` first, keep `passed`/`all_within_tolerance` as fallbacks, and build
`observed` from the `rows` list (max `max_abs_metric_deviation`, `n_rows`, plus `atol`/`rtol`
when present). New `tests/test_pilot_gate.py` case: feed a real-shaped report and assert
`met is True`; feed a failing one and assert `met is False`. Without this the gate is a
vacuous always-fail — trap 3's shape in a fourth place.

**Fix 2 — a producer for the pilot parity report.** New
`transformation_inheritance/run_pilot_parity.py`: copies the teacher tokenizer into (a copy
of) the pilot checkpoint dir, then dispatches to the frozen Neo `--verify-parity` driver and
reports where `parity_report.json` landed. **Dispatch, never reimplement** (rule 3) — it
calls the frozen harness, it does not re-derive parity. No frozen file is touched.

**Fix 3 — `resolve_threshold_rule` gains `anchor_side`.**
`transformation_inheritance/aggregate_transformation.py`. Today the threshold is only
`anchor − k·spread`. That is correct for `e6a_3` (both limbs read the same boundary from two
sides of an anchor at 1.0) but **wrong for `e6a_4` and `e6b_2`**, whose observed quantity is
an *absolute difference* anchored at 0: `0 − 2·spread` is negative, making "similar" never
satisfiable and "measurably different" trivially true. Add optional
`anchor_side: below|above` (default `below`, byte-identical to today) giving
`anchor + k·spread`. New `tests/test_threshold_rule.py` cases: default unchanged; `above`
gives `+k·spread`; an unrecognised value is **refused**, not defaulted (trap 13).

**Fix 4 — portable frozen-file hashing.** `tests/test_frozen_files.py` `_sha256` normalises
`\r\n` → `\n` before hashing; regenerate the 12 hashes in `BASELINE_HASHES.json` with the
same normalisation; add `.gitattributes` with `* text=auto eol=lf`. Record in the JSON that
normalisation is applied and why. **Stated trade-off:** an EOL-only change to a frozen file
stops being detected — correct, because git itself creates those, and the test then measures
content identity instead of checkout accident. Existing raw hashes preserved under
`frozen_files_raw_crlf` so nothing is lost.

## Part 2 — Pre-registration edits

`transformation_inheritance/configs/e6_preregistration.yaml` — resolve D1, D2, D3, D4:
delete each `status:`/`reason:` pair, fill the value or the `threshold_rule:` block, and add
one `amendments:` entry of `kind: decision` per decision with today's date (2026-07-30) and
the reasoning. Bump to `e6_prereg_v4`.

`crosslingual_semantics/configs/e7_preregistration.yaml` — resolve D5, D6, D7. For D7
replace the value in the **declaration block and all six rows together**
(`check_threshold_declaration` compares them as sets and `_condition_holds` reads only the
rows — trap 13's shape). Bump to `e7_prereg_v4`.

Both files: replace the `"TRANSCRIPTION — fill in date"` / `"CORRECTION — fill in date"`
placeholders with real dates. An undated amendment list reads as one written afterwards.

## Part 3 — `BASELINE_HASHES.json`

- `frozen_files` — regenerated LF-normalised (fix 4).
- `reference_results.gpt2_small_table1` — transcribe the 10 pooled `all_datasets` rows
  (`int_a`..`int_j`: `mean_bos_attention`, `stderr`, `n_examples`) from
  `nn_results/e1_e2/legacy_results/results_gpt2_e1_e2/table1_multiseed/seed_000/dataset_analysis/bos_attention_stats_overall.csv`,
  whose sibling `run_config.json` I confirmed is `model_name: gpt2, dtype: float32`
  (trap 10, hash before compare). From the file, never the paper.
- `environment` — captured from this venv via the `_capture_command` already in the file.

## Part 4 — Pre-flight hygiene

1. **Quarantine stale runs.** Both aggregators discover inputs by `rglob` with no dedup
   ([aggregate_transformation.py:113](transformation_inheritance/aggregate_transformation.py#L113)). Three real `checkpoint_metrics.csv` sit in the tree:
   `e6a/P8M/reference/` and `e6a/P8M/reference_dce/` — **both** would load, giving two
   seedless P8M rows for one condition, one with an empty `validation_ce`, straight into
   contrast 4's matched-loss selection — plus `e6b/F1/seed0/` from the earlier 2060 run.
   Move all three to `results/_exploratory/` after transcribing their numbers into
   NEXT_STEPS.
2. **Prove FLORES/XNLI load** before any GPU booking. `datasets==4.8.4` raises
   `RuntimeError: Dataset scripts are no longer supported` for script repos (verified
   against the installed package), and `load_flores_parallel` calls
   `load_dataset("facebook/flores", ...)` with no fallback. One download attempt each
   settles it; if FLORES fails the fix is an **additive** loader accepting a parquet mirror
   (rule 2 permits, rule 3 unaffected). I will report before changing anything.

## Part 5 — What I run

1. `pytest tests/ -q` — the Phase 0 gate, green before any real-model work.
2. `scripts/run_nnsight_rest_smoke_tests.ps1` — all eight offline smoke programs.
3. `screen_public_pairs.py` — no training, `07` P6's cheapest evidence.
4. **E6A pilot:** D0/D1/D2 at seed 0 to 2,000 steps, **plus D0 at seeds 1 and 2** so
   framing B's calibration can resolve (the aggregator refuses a spread from <3 seeds).
   I measure the true step rate from the first ~100 steps and **report the projection
   before committing to the remaining runs.**
5. Evaluate every checkpoint **`--with-delta-ce`** — framing B's `e6a_3` limb 1 needs
   `functional_cosine_to_teacher`, which only ΔCE populates. Then the P8M reference
   (`--with-delta-ce`, once), `aggregate_transformation.py`, `run_pilot_parity.py`, and
   `check_pilot_gate.py --parity-report ...`.
6. Report `pilot_gate.json` and `go_no_go.json` verbatim — including a `no` verdict. A null
   result is data (rule 4); I will not tune anything to move it.

E6B, E7 and Phase 2 are **documented, not run** — that was your scope call.

## Part 6 — Documentation

- **`COMMANDS.md`** (new): environment bootstrap, the gates, and every command for the full
  paper-ready program — Phase 1 pilot, E6A Phase 2, E6B, E7 across all four Qwen configs —
  with the resumability, `--with-delta-ce`, `--base`, `--parity-report` and
  `runtime_estimate.json` checkpoints called out inline, plus the PowerShell stderr
  workaround (trap 8).
- **`NEXT_STEPS.md`**: §2.3 closed, §2.7 closed with the seven resolutions, §2.4 corrected
  to 467, a new §1.6 recording this slice's decisions, the missing §2.7.2 E6B numbers, and
  real pilot results.
- **`CLAUDE.md`**: rule 4's "seven decisions are open" paragraph rewritten to "resolved in
  v4, amendments recorded"; traps 16–18 added for the three defects above.

## Verification

- `python -m pytest tests/ -x -q` — must be ≥467 + the new cases, 0 skipped.
- `python tests/test_frozen_files.py` standalone, and re-verified after a simulated
  LF checkout (hash both ways, assert both match).
- `pytest tests/test_e7_interpretation_exclusivity.py -q` — the 64-assignment disjointness
  sweep plus `check_threshold_declaration`, after D7's six-row edit.
- `pytest tests/test_aggregate_contracts.py tests/test_e7_aggregate_contracts.py tests/test_threshold_rule.py -q`
  — these assert that a YAML threshold change moves the verdict, so they are what proves the
  v4 edits are actually read.
- `scripts/run_nnsight_rest_smoke_tests.ps1` exits 0 across all eight programs.
- End-to-end: `pilot_gate.json` shows four criteria with `met` non-null, and
  `go_no_go.json` carries resolved `threshold_rule` blocks with realised numbers and an
  empty `pending_decisions_blocking`.

## Assumptions and risks

- **No frozen file is edited.** Fixes 1–4 touch `check_pilot_gate.py`,
  `aggregate_transformation.py`, `tests/`, configs and docs — none of the 12.
- **Framing B leaves `go_no_go.json` incomplete if D0 lands <3 usable seeds.** That is the
  registered rule behaving correctly, not a failure to work around.
- **E7's 0.5B fp32 vs 1.5B bf16 asymmetry stands as shipped** (`04` §7: only the contrast's
  *direction* replicates across them). On 12 GB, fp32 at 1.5B with all-layer capture is the
  riskier change; I will flag it in COMMANDS.md rather than alter a config silently.
- **Pilot wall-clock is unmeasured.** The models are tiny (8M student, 33M teacher) so the
  bottleneck is launch overhead, not FLOPs, and the 2060 should land near the design's
  per-run estimate — but I will measure and report rather than assert.
