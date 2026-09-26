# WP12 — Aggregator Extension + v3 Pre-registrations

## Context

`NEXT_STEPS.md` §2.5 and §2.6 have been blocking since WP10/WP11 shipped: the design
document holding §8.7, §10.11, §10.12 and §18 was not in the repository, so both
pre-registration YAMLs carry `PENDING_DESIGN_*` sentinels and both go/no-go files report
`decision: "incomplete"` no matter what the results say.

`files (7).zip` supplies exactly what was missing — four files, inspected in full:

| File | What it is |
|---|---|
| `08_AGGREGATOR_EXTENSION_SPEC.md` | a new work package (WP12): the additive aggregator changes the v3 YAMLs need |
| `DECISIONS_REQUIRED.md` | seven scientific decisions (D1–D7) the design does **not** make |
| `e6_preregistration.yaml` (v3) | the real §8.7 contrasts and §18 criteria |
| `e7_preregistration.yaml` (v3) | the real §10.11 contrasts, §10.12 matrix, §18 criteria, claim gate |

The v3 YAMLs use six schema forms today's aggregators do not understand (`kind:
corpus_pair`, `kind: source_level`, `absolute_difference`, `compound`, `seed_consistency`,
`matched_loss_reference`, `combine`, `evaluation: claim_gate_component`, `threshold_rule`).
Dropped in as-is they would either crash or silently ignore fields — the second being
worse, because a criterion that ignores `seed_consistency: all` passes on one strong seed
and two weak ones, which is precisely the correctness bug `08` §5.1 identifies.

**Outcome:** the design sections are transcribed and evaluable; the only thing still
holding both verdicts at `incomplete` is the seven D1–D7 decisions, which stay PENDING by
your instruction. Nothing is invented (CLAUDE.md rule 4).

### Confirmed scope

- **D1–D7 stay PENDING.** No threshold, tier or metric choice is written by me.
- **WP6 (E6B trainer) is out of scope** for this pass.
- **Verification includes a real GPU run** of the new P8M reference path.

---

## 1. Relocate the design files

| From (in zip) | To |
|---|---|
| `08_AGGREGATOR_EXTENSION_SPEC.md` | `new_design_plans/08_AGGREGATOR_EXTENSION_SPEC.md` |
| `DECISIONS_REQUIRED.md` | `new_design_plans/DECISIONS_REQUIRED.md` |
| `e6_preregistration.yaml` | `transformation_inheritance/configs/e6_preregistration.yaml` (replaces v1) |
| `e7_preregistration.yaml` | `crosslingual_semantics/configs/e7_preregistration.yaml` (replaces v1) |

Extract to `files (7)/`, copy, then add `files (7)*` to [.gitignore](.gitignore) so the
staging folder and zip do not enter the tree.

**One edit to the delivered YAMLs, and only one.** Every `status: PENDING_DECISION_*` entry
carries a `decision_required:` block but no `reason:` key. Both aggregators and
`test_aggregate_contracts.py::test_shipped_preregistration_marks_the_absent_design_sections`
read `reason:` — without it the artefact would report a pending criterion with an empty
explanation, and that test fails. I add a one-line `reason:` to each PENDING entry pointing
at `new_design_plans/DECISIONS_REQUIRED.md` §D*n*. **No threshold, tier or number is
touched**; `decision_required:` stays verbatim.

---

## 2. E6 aggregator — [transformation_inheritance/aggregate_transformation.py](transformation_inheritance/aggregate_transformation.py)

Every field below is optional; a pre-registration omitting it produces byte-identical
output to today. That invariant is what the four unchanged test files assert.

**2.1 Per-entry corpus selection.** v3 contrasts carry `corpus_id:`, which the aggregator
ignores today — and `aggregate()` pre-filters with `DEFAULT_CORPUS_PREFIXES`, which *drops
`e1_100x40` before contrasts run*, making `e6a_c5` impossible. Fix: pass the unfiltered
`e6a` frame to `build_contrasts`; `evaluate_contrast` applies `select_corpus` per entry
(entry `corpus_id` → that corpus; absent → today's default prefixes). `--corpus` keeps its
current meaning and now overrides every entry, recorded as `corpus_override` in the row.
`build_matched_loss`/`build_table2`/figures keep receiving the corpus-selected frame, so
no existing number moves.

**2.2 `kind: corpus_pair`** (`08` §2.1) — pairs two *corpora* within each condition.
Per seed × condition: `diff = metric(corpus_a) − metric(corpus_b)` at the selected
checkpoint, then `im.paired_seed_contrast` over the pooled diffs. Records `n_conditions`
and the per-condition breakdown in `per_seed_json`.

**2.3 `absolute_difference: true`** (§2.2) — `|diff|` per seed *before* the threshold test
and before `paired_seed_contrast`. New column `mean_abs_diff` beside `mean_diff`; both
recorded.

**2.4 Single-arm entries.** `e6a_3`'s sub-criteria give `condition_a` with no
`condition_b`. `evaluate_contrast` gains that path (value of `metric` for `condition_a`
alone), which the v1 YAML's own comment already described but the code never implemented.

**2.5 `compound: {all_of: [...]}`** (§2.3) — `met` is the AND of the sub-criteria; any
`None` limb makes the compound `None`, not `False`. Each limb reports its own `observed`,
`threshold`, `direction`, `met` so a failure names the limb. `status: PENDING_*` on the
parent still short-circuits.

**2.6 `matched_loss_reference: <condition>`** (§3) — matched-loss resolved against *that*
condition's CE rather than the global reference. Implemented by memoising
`build_matched_loss(frame, reference_condition=...)` per reference. The 0.05-nat refusal is
unchanged and the realised gap is reported per contrast, not only in
`e6a_matched_loss.csv`.

**2.7 `P8M` as a seedless condition** (§4) — detected by `checkpoint_step == -1` and the
seed sentinel. `diff_seed = metric(P8M) − metric(student, seed)`; row carries
`n_seeds = 3`, `pairing: "reference_vs_seeds"`, `test: "descriptive_reference_contrast"`
and an explicit note that the uncertainty is over student seeds only and carries none from
P8M. The paired-seed permutation label is **not** reused.

**2.8 `seed_consistency: all`** (§5.1) — the correctness fix. `met` requires *every*
per-seed value to clear the threshold, not the pooled mean. Reports `n_seeds_meeting` and
`per_seed_met`. Absent ⇒ today's pooled-mean behaviour.

**2.9 `e6b_go_no_go` with `combine: all`** (§5.2) — a second criteria block evaluated over
the E6B frame, written to `e6b_go_no_go.json`. Header-only-with-reason until WP6 produces
rows, matching how the E6B tables already behave.

**2.10 `threshold_rule: {kind: null_seed_spread, …}`** (§7) — computes the spread across
seeds *within* `null_condition` at the selected checkpoint and derives the threshold. The
rule is the pre-registration; the realised number is written into `go_no_go.json` beside
it. Two guards, both refusals with a reason: fewer than three seeds in the null condition,
and a frame carrying the treatment arm without the null arm.

**2.11 Pending-decision sweep.** New `pending_decisions` key listing every `PENDING_*`
sentinel outside the criteria list (`primary_topology_metric`, PENDING contrasts,
`e6b_go_no_go` criteria). `decision = "incomplete"` when `n_unknown` **or**
`pending_decisions` is non-empty. Kept as a *separate* key so
`pending_preregistration`'s existing exact-equality assertion is untouched. This is what
makes `08` §12's acceptance criterion — `incomplete` while any `PENDING_DECISION_*`
remains — actually hold.

**Not implemented:** `08` §6 (`topology_area_diff_to_teacher` as a CSV column) is
explicitly conditional on D1 resolving to that metric. D1 is PENDING, so §6 is skipped and
recorded as a follow-up. `im.topology_report` already returns `area_diff_normalised`, so it
is a two-line change if D1 lands there.

---

## 3. E7 aggregator — [crosslingual_semantics/aggregate_crosslingual.py](crosslingual_semantics/aggregate_crosslingual.py)

**3.1 `kind: source_level`** (`08` §8.1) — a one-group mean of `margin_delta` for rows
matching `source_condition`/`objects`/`norm_condition`, *not* a paired difference. Same
`im.hierarchical_bootstrap` + `im.bootstrap_two_sided_p` as every other contrast;
`value_b` and `relative_difference` are `null`; the mean lands in `absolute_difference` so
`observed_pattern` reads it uniformly.

**3.2 `observed_pattern` gains four keys** (§8.2, §11): `r0_effect` (`e7_c7`),
`k0_unrelated_effect` (`e7_c8k`), `v0_unrelated_effect` (`e7_c8v`) and
`retrieval_top1_over_chance`. Without all four, matrix rows 1/3/4/5/6 evaluate to `None`
and `matched_row` is `null` whatever the data says.

**3.3 Retrieval threaded into the matrix** (§11) —
`build_interpretation_matrix(prereg, contrast_frame, retrieval=None)`, threaded from
`aggregate()`. `retrieval_top1_over_chance` is computed by a helper **extracted from
`evaluate_criterion`'s retrieval branch and called by both**, so criterion `e7_1` and row 5
cannot disagree. `retrieval=None` ⇒ the key is `None`, today's behaviour.

**3.4 `evaluation: claim_gate_component`** (§9) — `e7_4`/`e7_5` return the corresponding
`evaluate_claim_gate` component's verdict instead of recomputing it. `build_go_no_go`
excludes ids listed in `claim_gate.satisfies_criteria` from `n_met` while still reporting
them, and adds `n_met_excluding_gate_components` so both readings are visible.

**3.5 `go_no_go_combine: all`** (§5.2) — `continue` only when every criterion is `met` and
the gate is met. Default `any` reproduces today exactly. Any `None` still forces
`incomplete`.

**3.6 `criterion_3_combine`.** `08` specifies no handling for this key, but the YAML makes
it govern how `e7_3k`/`e7_3v` fold into the decision, so it cannot be ignored. Resolution,
recorded in the code and in `NEXT_STEPS.md`: while it carries `status: PENDING_DECISION_*`
it is a pending decision (§3.7); once resolved, `any`/`all` combines the two into one
composite verdict counted once, with both limbs still reported.

**3.7 Pending-decision sweep** — same mechanism as §2.11, covering `e7_c6`,
`criterion_3_combine` and `interpretation_matrix`. Forces `decision: "incomplete"` **and**
`supported: null`.

---

## 4. `--evaluate-public-reference` — [transformation_inheritance/evaluate_transformation.py](transformation_inheritance/evaluate_transformation.py)

`08` §4: `roneneldan/TinyStories-8M` must be fingerprinted against the same teacher,
corpora, band, dtype and `intervention_registry_version` and written with
`condition = "P8M"`, `checkpoint_step = -1` and a seed sentinel.

- New `evaluate_public_reference(...)`, callable two ways: inside `evaluate_run` (sharing
  the already-built `corpora`, `teacher_handle` and `teacher_records` — §4's "reuse the
  same corpora objects, not rebuild them"), or standalone from a run dir or an E6A config.
- Writes its own `checkpoint_metrics.csv` under `results/e6a/P8M/reference/`, which
  `load_all_metrics`'s `rglob` already picks up.
- Reuses `build_row`, `guard_comparison`, `blank_row`, the ledger and `upsert_row`
  unchanged — no new fingerprint path, no new failure semantics.
- CLI: `--evaluate-public-reference <hf-id>`, `--reference-condition` (default `P8M`),
  `--reference-out`.

---

## 5. Tests

**New** (`08` §12 names the first six):

| File | Proves |
|---|---|
| `tests/test_seed_consistency.py` | §5.1's own case: seeds `[0.30, 0.05, 0.05]` vs threshold 0.15 is `met: true` without the flag, `met: false` with it |
| `tests/test_corpus_pair_contrast.py` | in-domain minus cross-domain per condition; `n_conditions` and the per-condition breakdown are recorded |
| `tests/test_compound_criterion.py` | AND semantics; a `None` limb ⇒ `None` not `False`; the failing limb is named |
| `tests/test_source_level_contrast.py` | one-group mean, `value_b`/`relative_difference` null, CI from the same hierarchical bootstrap |
| `tests/test_e7_interpretation_exclusivity.py` | exhaustive 2ⁿ sweep over the **shipped** v3 matrix: no assignment matches more than one row. Not exhaustiveness — `04` §6.3 permits "matches none" |
| `tests/test_claim_gate_no_double_count.py` | `e7_4`/`e7_5` are not counted in `n_met`; `n_met_excluding_gate_components` present |
| `tests/test_pair_specific_matched_loss.py` | `matched_loss_reference` targets the named condition's CE; realised gap per contrast; 0.05-nat refusal intact |
| `tests/test_threshold_rule.py` | §7's two guards refuse with a reason; the realised threshold reaches `go_no_go.json` |
| `tests/test_public_reference_rows.py` | P8M row shape (`condition`, step `-1`, seed sentinel), corpora identity by `manifest_sha256`, `reference_vs_seeds` pairing label |

Every test asserts invariants and relationships, never that a scientific quantity takes a
value (`06` §5).

**Must pass unchanged** (`08` §12): `test_aggregate_contracts.py`,
`test_matched_loss_selection.py`, `test_evaluate_transformation.py`.

**One documented exception to `08` §12.** `test_e7_aggregate_contracts.py` cannot pass
entirely unchanged: `test_the_shipped_preregistration_is_incomplete_until_the_design_is_transcribed`
asserts `pending_preregistration >= {"e7_3","e7_4"}` and
`interpretation_matrix_status == "PENDING_DESIGN_10_12"` — both are **v1 sentinel
identifiers**, and v3 renames `e7_3` → `e7_3k`/`e7_3v`, makes `e7_4` a claim-gate
component, and restyles the matrix sentinel as `PENDING_DECISION_THRESHOLD`. Two
assertions in that one test are updated to v3's ids; the other 20 tests in the file are
untouched. The *behavioural* invariant §12 is protecting — shipped YAML ⇒ `incomplete`,
`supported: null` — is preserved and still asserted.

**Smoke scripts updated** for the same reason (they assert v1 ids):
`tests/nnsight_e6_eval_smoke.py` (v3 has no PENDING *contrasts*, so the "pending" assertion
moves to the go/no-go and `pending_decisions`; it will also pass the discovered synthetic
corpus id via `--corpus`, since the shipped YAML names real corpora the smoke does not
build) and `tests/nnsight_e7_pipeline_smoke.py` (v3 sentinel ids). Both must still exit 0.

---

## 6. Verification

1. `./.venv/Scripts/python.exe -m pytest tests/ -q` — 285 existing + ~9 new files, 0 skipped.
2. `powershell -File scripts/run_nnsight_rest_smoke_tests.ps1` — all seven smoke programs
   exit 0 (trap 8: verdict from `$LASTEXITCODE`).
3. Dry run of both aggregators against the v3 YAMLs on the smoke fixtures — must complete
   without error and report `decision: "incomplete"` with the `PENDING_DECISION_*` entries
   named (`08` §12).
4. **Real GPU run** (RTX 2060, torch 2.10+cu128, CUDA confirmed available):
   ```
   ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
     --evaluate-public-reference roneneldan/TinyStories-8M \
     --config transformation_inheritance/configs/e6a_ce.yaml \
     --device cuda --dtype float32 --sink-blocks 24
   ```
   Downloads TinyStories-8M and the TinyStories-33M teacher (~200 MB) into `.hf_cache`.
   Proves `08` §4 end to end on real weights: a P8M row with a real
   `fingerprint_cosine_to_teacher`, correct band, matching `manifest_sha256`. If a download
   fails, that is reported as a failure of step 4 — not worked around, and not reported as
   a pass.
5. `python -m pytest tests/test_frozen_files.py` — no frozen file touched. `nnsight_engine.py`
   and `datasets_loader.py` are not modified at all by WP12.

---

## 7. Documentation

- **`CLAUDE.md`** — WP12 in the pipeline tables; replace hard rule 4's `PENDING_DESIGN_*`
  paragraph (the design sections *are* transcribed now) with the `PENDING_DECISION_*`
  regime and a pointer to `new_design_plans/DECISIONS_REQUIRED.md`; add `08` to the spec
  table; add an eleventh trap — *a pre-registration field the aggregator does not read is
  worse than one that crashes*, which is why every new form is optional-but-asserted.
- **`NEXT_STEPS.md`** — §2.5/§2.6 close as *transcribed*; a new §2.7 lists the seven open
  decisions with their blocking effect; WP12 joins §1's table with its decisions recorded
  in a §1.4; §3 keeps WP6 as the only remaining work package; the §6 verification results
  are recorded as run, including whether the GPU step passed.
