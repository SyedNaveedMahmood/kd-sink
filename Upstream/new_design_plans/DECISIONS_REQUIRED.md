# DECISIONS_REQUIRED.md

Seven decisions the design document does not make. Each blocks a `PENDING_DECISION_*`
entry. **All must be made before any results are inspected** — a threshold chosen after
seeing numbers is not a pre-registration (`05` §6).

Fill the `CHOSEN` line, paste the value into the YAML, delete the entry's `status`, and
copy the row into that file's `amendments:` as `kind: decision` with a date and reason.

None of these has a right answer derivable from the design. Where a recommendation is
given, its reasoning is stated so you can reject it on the reasoning rather than on trust.

---

## D1 — E6 primary topology metric  (`e6_prereg_v3: primary_topology_metric`)

§8.7 says "topological similarity". §6.3 defines three measures and §8.7 names none.
Both computable forms are already registered as `e6a_c3` / `e6a_c3b`, so both get computed
either way — this decides which the paper reports as primary.

| Option | Argument for | Argument against |
|---|---|---|
| `topology_spearman_to_teacher` | The only one of the three that is a *similarity*, so `e6a_c3`'s sign runs the same way as `e6a_c1`/`e6a_c2`. Presentational. | Rank correlation is blind to *where* in depth the profiles differ — it would score a teacher-like shape shifted to the wrong depth as highly similar. |
| `topology_wasserstein_to_teacher` | Sensitive to depth location, which is what "the anchor moved" actually means. Closer to the paper's claim. | It is a distance, so the contrast's sign inverts relative to the other two and needs care in the figure. |
| `topology_area_diff_to_teacher` | Simplest to explain. | Not currently a column (needs `08` §6); dominated by the other two. |

**Recommendation: Wasserstein**, on the grounds that the paper's claim in §24 is about
anchors being *conserved roles* rather than co-varying profiles, and depth location is the
substance of that. This reverses the v2 suggestion, which was argued on sign convention —
a presentational reason that should not decide a scientific measure.

```
CHOSEN: ______________________    REASON: ______________________
```

---

## D2 — E6A criterion 3 thresholds  (`e6a_3`: "functional high while mechanistic low")

No numeric boundary anywhere in the design. **v2's 0.90/0.60 was justified by §6.4's
intervention-residual bands; that justification is withdrawn** — those bands are ratios
`A_j/A_baseline` for one intervention, while these metrics are cosines between vectors of
such ratios. Mapping a band edge onto a cosine is a scale error.

**Framing A — fixed numbers.** Pick two, record them as judgement. Honest and arbitrary;
say so in the paper.

**Framing B — calibrated null.** Define "high" and "low" against the metric's own noise:
compute the across-seed spread of each metric *within a single condition* (D0 vs D0 across
seeds 0/1/2) from the pilot, then "high" = within *k* spreads of 1.0, "low" = more than
*k* spreads below. Pre-registers a **rule**; the number is a result. Computed from data
that does not involve the D1/D2 comparison, so it does not leak the outcome. Needs
`08` §7.

**Recommendation: B with k = 2**, matching `onset_threshold_k` and the clean-run bar
already used for E6B drift, so one calibration convention governs the whole paper. The
reviewer question "where did 0.90 come from?" then has an answer.

Cost of B: the pilot must run before the thresholds resolve, so `go_no_go.json` stays
`incomplete` through Phase 1. That is fine — the pilot gate is a separate mechanism.

```
CHOSEN framing: ____     high = ____________     low = ____________
REASON: ______________________
```

---

## D3 — E6A criterion 4 thresholds  (`e6a_4`: P8M "similar function, different fingerprints")

Same situation, same two framings, same withdrawal. Note the two limbs run **opposite
ways**: "similar function" is a *small* absolute difference, "measurably different
fingerprints" is a *large* one. That opposition is the entire content of the criterion —
it is what would support §20's Narrative B (convergent reconstruction rather than
inheritance).

If you take framing B, use the **same** null spread as D2 so the paper has one notion of
"more than noise".

```
CHOSEN framing: ____     similar ≤ ____________     different ≥ ____________
REASON: ______________________
```

---

## D4 — E6B thresholds  (`e6b_1`, `e6b_2`)

**`e6b_1` "comparable task accuracy".** Unlike D2/D3 this has a real external anchor:
SST-2 validation is 872 examples, so the binomial SE at ~90% accuracy is ~1.0 point and a
2-point band is about 2 SE. That is a statement you can defend in text. Confirm 0.02 or
set your own.

**`e6b_2` "differs consistently".** "Consistently" is read as `seed_consistency: all` —
a reading of the word, not a number. The magnitude is undecided, and framing B applies
with no extra work: `onset_threshold_k: 2.0` already defines a clean-run mean + 2 SD bar
for E6B drift. Reusing it costs nothing.

```
e6b_1 CHOSEN: ____________     e6b_2 CHOSEN: ____________
REASON: ______________________
```

---

## D5 — E7 language tiers  (`e7_c6`: high- vs lower-resource)

The design names the contrast and never tiers `en zh ar de hi sw tr`. **v2 proposed
[en, zh, de] vs [ar, hi, sw, tr] citing Qwen2.5's pretraining mix — that citation was not
checked and the proposal is withdrawn as unsupported.**

| Route | Strength | Weakness |
|---|---|---|
| **A. Qwen2.5 technical report** (arXiv:2412.15115, design §23 ref 9) — tier on a reported per-language token share or named language list | Strongest, because the claim is about *this* model family | Requires reading it first. Do not assume what it says; it may not report per-language shares at all, in which case this route is closed |
| **B. External taxonomy** (e.g. Joshi et al. 2020) | Citable, model-agnostic, uncontroversial | Describes NLP resources in general, not Qwen2.5's corpus — a weaker match to the claim |
| **C. Declare exploratory** — report per-language effects, no tier split, state in the paper that §10.11 contrast 6 could not be pre-registered for want of a defensible tiering | Costs nothing, entirely honest | Loses one of the six §10.11 contrasts |

**Recommendation: check A first; fall back to C rather than B.** A contrast tiered on a
general-purpose taxonomy invites the reviewer question "why is this the right split for
Qwen?", and C answers that question by not making the claim.

This is the single easiest place in the project to accidentally tier on the outcome.
Decide before you look at per-language effects.

```
CHOSEN route: ____     languages_a: ____________     languages_b: ____________
CITATION: ______________________
```

---

## D6 — E7 criterion 3 combination  (`criterion_3_combine`)

§18 criterion 3 says "parallel patches outperform unrelated same-label patches by at least
0.10" and names no object, so it is registered as `e7_3k` (K0) and `e7_3v` (V0).

`any` — either object clears the bar. Parallel to criterion 1's explicit "K0 **or** V0",
and consistent with §10.12's expectation that K0 and V0 behave differently, which is the
whole point of contrast `e7_c3`.
`all` — both must. Stricter.

**Recommendation: `any`**, as the more faithful reading of the surrounding text.

```
CHOSEN: ____     REASON: ______________________
```

---

## D7 — E7 interpretation-matrix thresholds  (`interpretation_matrix.thresholds`)

`retrieval_high: 2.0` is **not** a decision — it is §18 criterion 1's own bar, reused so
row 5 cannot contradict the gate.

**`semantic_sensitivity`** has a defensible anchor: §18 criterion 3's 0.10 correct-label
margin. Both are the same parallel-minus-control difference on the same scale, so setting
them equal is a reading rather than an invention.

**`unrelated_transfer`** has no anchor. It is a *different quantity* — an absolute patch
effect, not a difference of two — and the 0.10 currently in the file is a placeholder that
exists only so the matrix parses and the exclusivity test runs.

**Recommendation: calibrate it against the `random` norm condition.** "K0 transfers
safely" then means an effect exceeding what a norm-matched random vector produces at the
same site. Those rows are already measured on every run unit, so this costs no compute,
and it is the natural null for "did anything transfer at all". Same shape as framing B.

Whatever you choose, replace **both** values consistently in all six rows and re-run
`tests/test_e7_interpretation_exclusivity.py` — the disjointness proof assumes one
threshold per driver, and editing rows individually is how it breaks.

```
semantic_sensitivity CHOSEN: ____     unrelated_transfer CHOSEN: ____
REASON: ______________________
```

---

## Order

D5 first — it may need a paper read, and route C is available if the report gives you
nothing. D1, D6, D7 are readings of the design and take minutes. D2, D3, D4 can be
resolved as *rules* now (framing B) and left to compute their numbers from the pilot,
which lets Phase 1 start immediately.

Nothing here needs a GPU.
