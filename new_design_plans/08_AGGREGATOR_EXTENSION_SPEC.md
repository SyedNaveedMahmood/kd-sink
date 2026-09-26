# 08 — Aggregator Extension Spec (WP12)

Additive changes to `transformation_inheritance/aggregate_transformation.py` and
`crosslingual_semantics/aggregate_crosslingual.py` required by `e6_prereg_v3` and
`e7_prereg_v3`.

**Invariant:** every field below is optional. A pre-registration that omits it must
produce byte-identical output to today. `tests/test_aggregate_contracts.py` and
`tests/test_e7_aggregate_contracts.py` must pass unchanged before any new test is added.

Section numbers match the `requires:` lines in the two YAMLs.

---

## 1. What is already correct — do not touch

- `build_interpretation_matrix` refusing to pick a row when several match. That refusal is
  right; v2's matrix was the defect, not the code. v3's rows are pairwise disjoint (proved
  exhaustively in §10).
- `_condition_holds`'s `float(value)` and word-only operator vocabulary. v3 uses literal
  numbers and word operators; no change needed.
- `evaluate_criterion` reading a singular `contrast`. v3 splits E7 criterion 3 into
  `e7_3k`/`e7_3v` rather than widening the field.
- `go_no_go` being a list. v3 keeps it a list and puts the conjunction in a sibling key.

---

## 2. E6: three new contrast/criterion forms

### 2.1 `kind: corpus_pair` (needed by `e6a_c5`)

```yaml
kind: corpus_pair
conditions: [D0, D1, D2]
corpus_a: tinystories_validation_sink_300
corpus_b: e1_100x40
```

Today `evaluate_contrast` pairs two conditions within one corpus. This pairs two corpora
within each condition. Per seed, per listed condition:
`diff = metric(corpus_a) − metric(corpus_b)` at the selected checkpoint, then the existing
`paired_seed_contrast` over the per-seed diffs pooled across conditions.

Record `n_conditions` and the per-condition breakdown in `per_seed_json` so a single
condition driving the pooled effect is visible.

Entries with no `kind` key must keep taking the existing condition-pair path.

### 2.2 `absolute_difference: true` (needed by `e6a_2`, `e6a_4`, `e6b_1`)

Take `|diff|` per seed **before** the threshold test and before `paired_seed_contrast`.
Record both signed and absolute means; the CSV gains `mean_abs_diff` alongside
`mean_diff`. Absent or false ⇒ unchanged.

### 2.3 `compound: {all_of: [...]}` (needed by `e6a_3`, `e6a_4`)

A criterion whose `met` is the AND of several sub-criteria, each of which is an ordinary
criterion body. Report every sub-criterion's `observed`, `threshold`, `direction` and
`met` in the output so a failure names which limb failed. If any limb is `None`, the
compound is `None`, not `False`.

`status: PENDING_*` on the parent still short-circuits the whole thing, as today.

---

## 3. Pair-specific matched loss (needed by `e6a_c2`, `e6a_c4a`, `e6a_c4b`)

`build_matched_loss` matches every condition against one `reference_condition` (D0's final
CE). Design §8.7 contrast 4 needs D1's checkpoint nearest **P8M's** CE, which is a
different reference per contrast.

Add an optional `matched_loss_reference: <condition_id>` to a contrast/criterion. When
present, `selection: matched_loss` resolves `step_b` (and `step_a` where the condition has
multiple checkpoints) against that condition's CE rather than the global reference.

The existing `matched_loss_max_gap_nats: 0.05` refusal applies unchanged, and the realised
gap must be reported per contrast, not only in `e6a_matched_loss.csv`.

Absent ⇒ current behaviour.

---

## 4. `P8M` as a seedless condition (needed by `e6a_c4a/c4b`, `e6a_4`)

`roneneldan/TinyStories-8M` is fingerprinted through `evaluate_transformation.py` against
the same teacher, corpora, band, dtype and `intervention_registry_version`, and written to
`checkpoint_metrics.csv` with `condition = "P8M"`, a single `checkpoint_step` (use `-1` to
mark "not a training step"), and `seed` set to a sentinel.

The pairing problem: P8M has one measurement; D1/D2 have three seeds. Design §15.1 makes
the training seed the unit of replication, and P8M contributes none.

**Required handling:** treat P8M as a fixed reference value, not as a paired arm. For each
student seed, `diff_seed = metric(P8M) − metric(student, seed)`. Report the contrast with
`n_seeds = 3` but set `pairing: "reference_vs_seeds"` in the row, and state in the artefact
that the uncertainty is over student seeds only and carries no uncertainty from P8M, which
is a single public checkpoint. Do **not** silently reuse the paired-seed permutation test
label — set `test: "descriptive_reference_contrast"`.

`--evaluate-public-reference` on `evaluate_transformation.py` is the cleanest way to emit
these rows; it must reuse the same corpora objects, not rebuild them.

---

## 5. `seed_consistency` and `combine`

### 5.1 `seed_consistency: all` (E6, needed by every `e6a_*` criterion)

**This is a correctness fix, not a feature.** Design §18: "At least one of the following
is observed **across all three seeds**." `evaluate_criterion` sets `met` from `mean_diff`
and records `all_seeds` unused, so a criterion can pass on one strong seed and two weak
ones.

When `seed_consistency: all`, `met` requires **every** per-seed value to satisfy the
threshold, not the pooled mean. Report `n_seeds_meeting` and `per_seed_met` alongside.
When absent, keep today's pooled-mean behaviour so existing tests are untouched.

Add `tests/test_seed_consistency.py`: a synthetic frame with seeds `[0.30, 0.05, 0.05]`
against threshold 0.15 must be `met: true` without the flag and `met: false` with it.

### 5.2 `combine` (E7 `go_no_go_combine: all`, E6 `e6b_go_no_go.combine: all`)

E6A's §18 is a disjunction; E7's and E6B's are conjunctions. Today both decision rules are
at-least-one.

E7: read the sibling key `go_no_go_combine` (default `any`, today's behaviour).
`all` ⇒ `decision: "continue"` only when every criterion is `met: true` **and** the claim
gate is met. `any` unchanged. Any `None` still forces `incomplete`.

E6B: `e6b_go_no_go.combine` behaves the same over its own criteria list.

---

## 6. `topology_area_diff_to_teacher` (optional; needed only if that metric is chosen)

`inheritance_metrics.topology_area_diff` exists but is not surfaced as a column in
`checkpoint_metrics.csv`. If `primary_topology_metric` resolves to it, add it to the
record and the CSV; otherwise skip this section. Adding a column is permitted
(`05` §5: a column may be added, never renamed).

---

## 7. `threshold_rule` — calibrated thresholds (needed only under framing B)

If `e6a_3`/`e6a_4`/`e6b_2` are resolved by calibration rather than fixed numbers:

```yaml
threshold_rule:
  kind: null_seed_spread
  metric: functional_cosine_to_teacher
  null_condition: D0          # same condition, across seeds
  statistic: range            # range | std
  k: 2.0
  anchor: 1.0                 # "high" = within k*spread of anchor
```

Compute the spread across seeds **within** `null_condition` at the selected checkpoint,
derive the threshold, and **write the realised numeric threshold into `go_no_go.json`**
alongside the rule. The rule is the pre-registration; the number is a result.

Two guards: refuse if the null condition has fewer than three seeds (a spread from two
points is not a spread), and refuse if the rule is evaluated on a frame that also supplies
the contrast's treatment arm without the null arm present.

---

## 8. E7: `kind: source_level` and four new observed keys

### 8.1 `kind: source_level` (needed by `e7_c8k`, `e7_c8v`)

```yaml
kind: source_level
source_condition: same_label_en
objects: [K0_prerope]
norm_condition: direct
metric: margin_delta
```

A **one-group mean**, not a paired difference: the mean `margin_delta` over rows matching
the filter. `margin_delta` is already `patched − baseline`, so this is the unrelated
source's own causal effect against no patch — which is not expressible as any
parallel-vs-control contrast.

Statistics: the same hierarchical bootstrap (resample semantic ids, then languages) for
the CI, and `bootstrap_two_sided_p` against zero. `value_b` is `null`;
`relative_difference` is `null`. Keep `absolute_difference` as the column carrying the
mean so `observed_pattern` reads it uniformly.

### 8.2 `observed_pattern` gains four keys

```python
"r0_effect":                  _value("e7_c7"),
"k0_unrelated_effect":        _value("e7_c8k"),
"v0_unrelated_effect":        _value("e7_c8v"),
"retrieval_top1_over_chance": <see §11>,
```

Without all four, interpretation rows 1, 3, 4, 5 and 6 evaluate to `None` and
`matched_row` is `null` whatever the data shows.

---

## 9. `evaluation: claim_gate_component` (needed by `e7_4`, `e7_5`)

`e7_4`/`e7_5` are §18 criteria 4 and 5 and are also the two halves of the claim gate.
Evaluating them independently would double-count them in `n_met`.

Give them `evaluation: claim_gate_component` and `claim_gate_field:
min_model_sizes|min_languages`. `evaluate_criterion` returns the corresponding component's
verdict from `evaluate_claim_gate` rather than recomputing it, and `build_go_no_go`
excludes any id listed in `claim_gate.satisfies_criteria` from `n_met` while still
reporting it in `criteria`. Add `n_met_excluding_gate_components` so both readings are
visible.

---

## 10. Exclusivity test (mandatory)

`tests/test_e7_interpretation_exclusivity.py`: load the pre-registration, collect the
distinct keys used across all rows' conditions, and enumerate all `2^n` truth assignments
by constructing synthetic observed values just above and just below each key's threshold.
Assert **no assignment matches more than one row.**

v3 passes this: 64 assignments, 0 overlaps. Run it in CI — a future edit to one row is
exactly how exclusivity silently breaks, and the failure mode is a permanent
`matched_row: null` that looks like a null result.

The test must not assert exhaustiveness. §10.12 has no residual category and `04` §6.3
permits "matches none".

---

## 11. Retrieval into the interpretation matrix

`build_interpretation_matrix(prereg, contrast_frame)` has no access to the retrieval frame,
which `build_go_no_go` already receives. Row 5 ("retrieval is high but patches have no
causal effect") and row 6 both condition on it.

Add `retrieval=None` as a third argument, thread it from `main`, and compute
`retrieval_top1_over_chance` the same way criterion `e7_1` does — `max` over layers for
objects `[K0_prerope, V0]` at `alignment: cosine`. Reusing e7_1's reduction is required,
not incidental: if the two disagree, row 5 and the gate can contradict each other.

`retrieval=None` ⇒ the key is `None` and rows 5/6 report `None`, which is today's
behaviour for a missing input.

---

## 12. Acceptance

- `tests/test_aggregate_contracts.py`, `tests/test_e7_aggregate_contracts.py`,
  `tests/test_matched_loss_selection.py`, `tests/test_evaluate_transformation.py` pass
  **unchanged**.
- New: `test_seed_consistency.py`, `test_corpus_pair_contrast.py`,
  `test_compound_criterion.py`, `test_source_level_contrast.py`,
  `test_e7_interpretation_exclusivity.py`, `test_claim_gate_no_double_count.py`.
- A dry run of both aggregators against the v3 YAMLs on the existing offline smoke
  fixtures completes without error and reports `decision: "incomplete"` while the
  `PENDING_DECISION_*` entries remain — that is the correct state until the decisions in
  `DECISIONS_REQUIRED.md` are made.
- `nnsight_e6_eval_smoke.py` and `nnsight_e7_pipeline_smoke.py` still exit 0.
