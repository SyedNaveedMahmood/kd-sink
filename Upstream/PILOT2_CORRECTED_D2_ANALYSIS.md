# E6A Pilot 2 — analysis of the corrected-D2 run

**Archive:** `corrected-d2-pilot-results/` · **Run:** `git_sha 3dc6471d`, `git_dirty: false`
**Date of run:** 2026-07-31 · **Hardware:** RTX 4080 SUPER, 0.2065 steps/s, 4.68 GB peak VRAM

Every number below is read from the archive or recomputed from it. Where I recompute
something the pipeline did not write, it is labelled. Nothing is estimated or extrapolated
unless the sentence says so.

---

## 1. Verdict

| Gate | Pilot 1 | Pilot 2 (corrected D2) |
|---|---|---|
| `e6a_pilot_1` CE falls | ✅ met | ✅ met |
| `e6a_pilot_2` sink > 0.15 | ❌ (read the wrong rows) | ❌ **met: false** — the subject of §3 |
| `e6a_pilot_3` D0 vs D2 differ ≥ 0.10 | ✅ (Δ L1 = 0.300) | ✅ met, **smaller margin** (Δ L1 = 0.160) |
| `e6a_pilot_4` parity on 5 examples | ⚠️ passed on 3 examples | ✅ **met on 5**, worst dev 1.118e-08 vs atol 1e-5 |
| **`pilot_gate.json`** | `hold`, proceed false | **`hold`, proceed false**, 3/4 met, 0 unknown |
| `go_no_go.json` | `stop` (0 met, 0 unknown) | **`incomplete`** (0 met, **1 unknown** — see §5.2) |

**All three code fixes are confirmed working in production.** The one criterion that still
fails does so for a reason the fixes made *measurable for the first time*, and it is not a
code fault: **the teacher has no in-domain attention sink to inherit.**

---

## 2. The fixes landed — evidence from the artefacts

| Fix | Evidence in this archive |
|---|---|
| **Trap 22** — `L_ATTN` reduction | `run_config.json → attn_reduction: "sum_over_keys_mean_over_queries"`; `train_log` per-pair breakdown carries both `n_valid: 2113536` and `n_query_rows: 32768` (ratio exactly 64.5) |
| **Trap 23** — gate row selection | `pilot_gate.json` carries `corpus_id: tinystories_validation_sink_300`, `step_selection: "max over trained checkpoints (step > 0)"`, and three readings per condition; criterion 3 carries `baseline_sink_denominator` |
| **Trap 24** — per-corpus cache | **12** fingerprint files, **6 per corpus**, in `…/step_<n>/<corpus_id>/`. The teacher now has a record for **both** corpora |
| **Trap 25** — parity example count | `parity_report.json → n_sentences: 5`; `parity_context.json → sentences_source: run_pilot_parity.PILOT_PARITY_SENTENCES` |
| Clean provenance | one `git_sha` (`3dc6471d`) across `run_config`, `aggregate_summary`, `go_no_go` and `pilot_gate`; `git_dirty: false` throughout |

The retrained D2 is a **controlled A/B against the old D2**: identical
`initial_state_sha256` (`a6c83ead…`) and identical `block_manifest_sha256` (`85014934…`).
Same weights at step 0, same data in the same order — the reduction is the only variable.
That matters for §4.

**The reduction fix costs nothing.** 0.2065 steps/s and 4.68 GB peak, identical to the old
run: the JSD tensor was always computed, only the denominator changed.

---

## 3. Why `e6a_pilot_2` still failed

### 3.1 The arithmetic

Threshold **0.15**. Corrected D2 on the pre-registered corpus: **0.002411** at step 2,000
and **0.006893** at its best trained checkpoint (step 250). That is **62×** and **22×**
short. All three conditions sit in the same place:

| | at step 2000 | max over trained steps | argmax | max incl. step 0 |
|---|---|---|---|---|
| D0 | 0.002823 | 0.009932 | 250 | 0.011407 |
| D1 | 0.001942 | 0.009663 | 250 | 0.011407 |
| **D2 corrected** | **0.002411** | **0.006893** | 250 | 0.011407 |

No reading of "by step 2000" comes within 20× of the bar. So the verdict does not depend on
the wording question the trap-23 fix exposed.

### 3.2 The metric has a sequence-length floor, and the pilot starts on it

`compute_bos_attention_metric` averages attention paid to position 0 **by second-half query
tokens**, over heads and the depth band. Under uniform causal attention, query *i* puts
`1/(i+1)` on key 0, so the expected value of the metric is

```
floor(L) = ( H_L − H_{L/2} ) / ( L − L/2 )   ≈   2·ln2 / L   =   1.386 / L
```

| corpus | tokens | predicted floor | measured at step 0 (random init) | agreement |
|---|---|---|---|---|
| `tinystories_validation_sink_300` | 128 | **0.010770** | **0.011407** | 1.059× |
| `e1_100x40` | 40 | **0.034040** | **0.034598** | 1.016× |

A prediction from first principles matching measurement to 1.6% and 5.9%. Two consequences:

1. **The step-0 value is not a sink.** It is the uniform-attention floor. The trap-23 defect
   in pilot 1 was reporting exactly this artefact (0.034598, the 40-token floor) as a result.
2. **The two corpora are not comparable on raw sink.** The cross-domain corpus's floor is
   3.16× higher purely because its windows are 40 tokens rather than 128. Every raw
   cross-domain sink in the project carries that factor.

Normalising by the floor gives the only scale-free reading of this metric:

| model | in-domain (×floor) | cross-domain (×floor) |
|---|---|---|
| **TEACHER TinyStories-33M (4L)** | **0.864** | **2.526** |
| random init (step 0) | 1.059 | 1.016 |
| D0 (CE only) @2000 | 0.262 | 0.682 |
| D1 (logit KD) @2000 | 0.180 | 0.710 |
| D2 old (under-scaled) @2000 | 0.174 | 0.672 |
| **D2 corrected @2000** | **0.224** | **0.888** |
| P8M public TinyStories-8M (8L, converged) | 0.308 | 0.813 |
| *GPT-2 small, frozen E1 reference* | — | **16.56** |

### 3.3 The teacher has no in-domain sink — this is the actual answer

**`baseline_sink` of `roneneldan/TinyStories-33M` on `tinystories_validation_sink_300` =
0.009309.** This number did not exist before this run: pilot 1's cache overwrote the
teacher's in-domain record with the cross-domain one (trap 24), so the only teacher value
anyone had was the cross-domain 0.085975. **Fixing the audit trail is what produced the
scientific finding.**

- **0.15 / 0.009309 = 16.1×.** The criterion asks a student to exceed its teacher sixteenfold.
- **0.009309 / 0.010770 = 0.864× the uniform floor.** The trained teacher puts *less* mass on
  position 0 than random attention would. There is no sink here — there is mild
  position-0 *avoidance*.
- `frac_cells_above_0_2 = 0.0000`: not one of the teacher's 4×16 (layer, head) cells puts
  >0.2 attention on position 0 in-domain. Its strongest carrier head reaches 0.0559.
- Per-layer, in-domain: `[0.0015, 0.0043, 0.0113, 0.0123]` — rising with depth, but the peak
  is still an order of magnitude below the bar.

And the teacher fails 0.15 on the cross-domain corpus too (0.085975; the bar is **1.7×**
higher). **There is no corpus in this pilot on which the teacher meets criterion 2.** So the
criterion is unsatisfiable by inheritance from this teacher regardless of corpus choice or
training length — no corpus-reselection argument can rescue it, and none should be attempted
(§7).

### 3.4 Where 0.15 comes from, and why it does not transfer

The frozen E1 reference in `BASELINE_HASHES.json` records GPT-2 small at
`int_a = 0.563683 ± 0.002170` (n=300) on `e1_100x40` — **16.6× that corpus's floor**. Against
that, 0.15 is ~27% of GPT-2's own baseline: a sensible "the sink has emerged" bar for a
GPT-2-class model on 40-token windows.

Transferred unchanged to 128-token TinyStories blocks it becomes **13.9× the uniform floor**,
i.e. it demands that 15% of every second-half query's attention mass land on the single first
token.

**0.15 is not a hard bar in general — it is hard only for this teacher.** All six models in
`results/screening/screening_fingerprints.csv` clear it comfortably on the *same* `e1_100x40`
corpus the teacher is measured on:

| model | `baseline_sink` | ×floor (÷0.03404) | frac cells > 0.2 |
|---|---|---|---|
| gpt2 | 0.563006 | **16.5×** | **0.729** |
| distilgpt2 | 0.463267 | 13.6× | 0.708 |
| Qwen2.5-0.5B base / instruct | 0.428838 / 0.434566 | 12.6× / 12.8× | 0.667 / 0.676 |
| Qwen2.5-1.5B base / instruct | 0.380932 / 0.387198 | 11.2× / 11.4× | 0.771 / 0.780 |
| *the 0.15 threshold* | 0.15 | *4.4×* | — |
| **TinyStories-33M teacher** | **0.085975** | **2.53×** | **0.031** |
| TinyStories-33M teacher, in-domain 128-tok | 0.009309 | 0.86× | 0.000 |
| TinyStories-8M public, in-domain 128-tok | 0.003317 | 0.31× | 0.000 |

Every screened public model sits at 11–17× floor with 67–78% of its (layer, head) cells above
0.2. The TinyStories teacher sits at 2.5× floor with 3.1% of cells, and below the floor
in-domain. **The teacher is the only model in the project measured below the bar.** So the
sink signature is strong and ubiquitous in ordinary pretrained LMs, weak and
out-of-distribution-only in the 33M TinyStories teacher, and absent in the 8M architecture.
Criterion 2's threshold was transcribed from the design without renormalising for sequence
length or checking the chosen teacher against it. That is an observation to report, **not** a
licence to rewrite the number now that results exist.

### 3.5 Why the sink *falls* during training — and why more steps will not reverse it

The trajectory 0.0114 → 0.0024 is not a failure to learn. The student begins at the uniform
floor (an artefact of near-constant attention logits at random init) and training concentrates
attention onto content tokens, so position 0's share falls below uniform. The teacher sits at
0.86× floor; the student ends at 0.22×, i.e. it over-suppresses position 0 by ~4× relative to
its teacher. Direction of travel: toward the teacher's regime, past it.

The design's prescribed remedy is "extend seed 0 to 5,000 steps". The strongest available
evidence on what that would find is **P8M**: `roneneldan/TinyStories-8M` is the same
architecture on the same data trained to convergence (validation CE 1.9356 vs our 2.5189),
and it measures **0.003317 in-domain — 0.31× floor, still 45× short of 0.15**. Caveat: P8M's
recipe and step count are a third party's, so this bounds *the architecture on this data*,
not strictly our recipe. But it is a measurement, not an extrapolation, and it points the
same way as our monotonic decline.

Two further quantities move *toward* the teacher while the magnitude does not (§4), which is
the substantive scientific result of this pilot.

---

## 4. What the corrected attention objective actually did

### 4.1 The objective is unambiguously active

The reduction ratio `n_valid / n_query_rows = 2113536 / 32768 = 64.5` is exact and constant
(fixed 128-token blocks, full causal support on all four mapped pairs), so an old-scale
`l_attn` can be converted exactly. D0 and D1 carry `attn_weight = 0`, so their weights are
untouched by the fix and their logged values convert to what corrected code would report:

| run | eval `l_attn` @2000 | on the corrected scale |
|---|---|---|
| D0 (no attn term) | 0.005213 | 0.336209 |
| D1 (no attn term) | 0.005079 | **0.327566** |
| D2 old (attn at 1/64.5) | 0.004964 | 0.320172 |
| **D2 corrected** | — | **0.202505** |

- **D2 is 38.2% below the D1-equivalent.** Under the old reduction it was 2.3% below.
- D2's own trajectory: **0.324319 → 0.202505, a 37.6% reduction** across 2,000 steps. The old
  run moved 1.3%.
- Share of the objective: **0.211% → 0.908%** of total loss (step 1 → step 2000), against
  0.003% → 0.022% before. Per-pair JSD at the end: `mixed 0.2489 / local 0.2015 /
  mixed 0.1778 / local 0.1808`.

By any reading, D2 now optimises the term design §1.6 registers for it. The proposed
micro-pilot bar ("≥ 15–20% below D1") is cleared without needing a separate micro-pilot.

### 4.2 What moved, and what is inside seed noise

D1 and D2 have one seed each, so the honest yardstick for a D2-vs-D1 difference is **D0's
across-seed range at step 2,000** (three seeds). Ratios below are |effect| ÷ that range.

**In-domain (`tinystories_validation_sink_300`), step 2000:**

| metric | D0 range | D2c − D1 | ×noise | D2c − D2old | teacher |
|---|---|---|---|---|---|
| `carrier_jaccard_to_teacher` | 0.00705 | **+0.0963** | **13.7×** | +0.0994 | 1.0 |
| `carrier_concentration` (Gini) | 0.02067 | **+0.1318** | **6.4×** | +0.1267 | **0.6804** |
| `topology_wasserstein_to_teacher` | 0.01102 | **−0.0173** | **1.6×** | −0.0203 | 0.0 |
| `baseline_sink` | 0.00043 | **+0.00047** | **1.1×** | +0.00054 | 0.009309 |
| `fingerprint_l1_to_teacher` | 0.18929 | −0.1278 | 0.7× | −0.1405 | 0.0 |
| `topology_spearman_to_teacher` | 0.03529 | −0.0235 | 0.7× | −0.0235 | 1.0 |
| `validation_ce` | 0.06391 | −0.0197 | 0.3× | −0.0232 | — |
| `functional_cosine_to_teacher` | 0.00327 | −0.0013 | 0.4× | +0.0056 | 1.0 |
| `fingerprint_cosine_to_teacher` | 0.11019 | **−0.0016** | **0.01×** | −0.0014 | 1.0 |

**Cross-domain (`e1_100x40`), step 2000:** the same pattern, and stronger on magnitude —
`baseline_sink` +0.00606 (**3.0× noise**), `carrier_concentration` +0.1296 (**4.1×**),
`carrier_jaccard` +0.0894 (**3.7×**), `topology_spearman` +0.1971 (**1.7×**); `fingerprint_cosine`
−0.0109 (0.2×, inside noise).

### 4.3 The reading

**Attention-map matching transfers the sink's *topology and carrier identity*; it does not
transfer its *magnitude*, and it does not touch the causal fingerprint.**

Supporting detail beyond the table:

- **Carrier concentration converges on the teacher's value.** D2 corrected reaches Gini
  **0.6634** in-domain against the teacher's **0.6804**; old D2 reached 0.5367 and D1 0.5316.
  Cross-domain D2c is 0.4732 against the teacher's 0.4627 — it slightly overshoots. Gini is
  scale-free, so this is a statement about *how* the (small) position-0 mass is distributed
  across (layer, head) cells, not how much of it there is.
- **The carrier heads are the teacher's heads.** D2 corrected's top in-domain carriers are
  heads **4, 13, 7** (strengths 0.0213, 0.0194, 0.0129 at normalised depths 0.71, 1.0, 1.0);
  the teacher's top four are heads **4, 13, 12, 7** (0.0559, 0.0368, 0.0365, 0.0325). Three of
  the teacher's top four, at comparable normalised depth, with ~38% of its magnitude. This is
  what the 13.7×-noise Jaccard jump is made of. (The equivalent old-D2 record was destroyed by
  trap 24, so head-identity cannot be compared against it directly — only via the Jaccard
  summary in the CSV.)
- **The fingerprint is untouched.** `fingerprint_cosine_to_teacher` moved by 0.0016, i.e.
  1.4% of one seed range. The interventions still disagree wholesale: D2's in-domain
  fingerprint is `int_c 0.5625, int_g 4.4793, int_h 2.4698, int_i 1.0427` against the
  teacher's `0.2391, 0.3009, 0.6609, 0.3261`. The teacher's position-0 mass collapses when
  the first positional embedding is removed (to 24%) or the massive key coordinates are
  zeroed (to 33%); the student's does not move (104%). **The student's residual position-0
  mass has a different causal support from the teacher's** — that is the mechanistic
  non-inheritance result, and it survives the fix intact.
- **The CE improvement is not established.** −0.0197 nats vs D1 is 0.3× of D0's own seed
  range. D2 currently has the lowest CE of the three (2.5189 < 2.5386 < 2.5696) but that
  ordering is inside noise and must not be reported as an effect.

One honest limitation on all of §4.2: the D2c − D2old column *is* a clean single-variable A/B
(identical init and data order), so the fix's effect on these artefacts is causally
established. Its **generality across seeds is not** — D1 and D2 have one seed each, and the
D2-vs-D1 column is where the noise ratios apply.

---

## 5. The other criteria

### 5.1 `e6a_pilot_3` passes with a smaller margin — and that is informative

Δ between D0 and D2 at the pilot step, in-domain: `fingerprint_l1` **0.15997** (was 0.30049),
`fingerprint_cosine` 0.09761 (was 0.09616). Still ≥ 0.10 on L1, so met.

The margin shrank because the corrected D2 moved **toward** the teacher in L1 (1.2117 →
1.0712) and therefore toward D0 (0.9112). Ordering in L1 distance-to-teacher:

```
D0 0.9112  <  D2 corrected 1.0712  <  D1 1.1989  ≈  D2 old 1.2117
```

Attention matching partially undoes the L1 damage logit-KD does — but **plain CE remains the
most teacher-like student on the fingerprint**, in both cosine (0.5484 vs 0.4508) and L1. If
that convergence continues to 10,000 steps, criterion 3 could fail: D0 and D2 would be
mechanistically indistinguishable. That would be a *result* (attention KD buys no mechanistic
difference), not a bug — the criterion exists precisely to detect that the arms are
distinguishable at all. Note also the diagnostic the fix now records: these are ratios over a
`baseline_sink` denominator of 0.0028/0.0024, so their absolute stability is poor.

### 5.2 `go_no_go.json` moved `stop` → `incomplete`, and this is not an improvement

`e6a_2` flipped from a computed value to **`no_data`**: *"realised CE gap 0.0507 exceeds 0.05
nats; the comparison is refused rather than made across mismatched losses (03 §6.4)"*.

D2's final CE (2.518927) is now **0.050717** below the D0 reference (2.569644) — over the
0.05 matched-loss window by **0.0007, i.e. 1.4%**. The next D0 checkpoint down is 2.9207 at
step 1,000, so no D0 checkpoint lies within 0.05 of D2's final loss. This is a **resolution
artefact of the pilot's five-checkpoint grid**, not a scientific finding, and it is the
matched-loss guard behaving correctly. At 10,000 steps with eight checkpoints the grid is
finer. `decision: "incomplete"` here means "one criterion has no comparable data", not "the
evidence got worse".

### 5.3 `e6a_4` — one limb now met

| limb | observed | threshold (calibrated on D0's 3-seed spread) | met |
|---|---|---|---|
| similar function: \|P8M − D2\| functional cosine ≤ t | **0.005365** (was 0.010940) | 0.006532 | ✅ **now met** |
| measurably different fingerprints: \|P8M − D2\| cosine ≥ t | 0.196940 (was 0.198383) | 0.220381 | ❌ (89.4% of bar) |

The corrected D2's function is now as teacher-like as the independently-trained public model's,
inside the calibrated null band — the design's "similar function" half of contrast 4 is
established. The second limb asks the fingerprint gap to exceed **twice D0's across-seed
range** (2 × 0.11019); the observed gap is 1.79 seed-ranges. That bar is calibrated on D0
alone, so running D2 at seeds 1 and 2 would tighten the *estimate* without moving the *bar*.

### 5.4 `e6a_1` — fails, but the effect is smaller than the null's own noise

Observed **−0.09761** against a +0.15 requirement: D2 is *less* teacher-like than plain CE.
But |−0.0976| is 0.89 of D0's across-seed range (0.11019), so with one D2 seed this contrast
resolves in neither direction. Report it as unresolved at pilot scale, not as a refutation.

### 5.5 `e6a_3` limb 0 — the structural issue is unchanged

Threshold 0.993468 = 1.0 − 2 × 0.003266 (D0's functional-cosine range). D0's own values are
0.9673/0.9672/0.9704 — **the null condition fails the threshold calibrated on it**, because
the anchor is 1.0 while the realised spread (0.0033) is ~10× smaller than the distance from
the anchor (0.033). No condition can meet this limb. Already documented; it is a property of
framing (B) anchored at 1.0, and it must be discussed in the paper rather than amended now.

---

## 6. Secondary findings

- **Distillation is working and is not the problem.** teacher KL 22.75 → **2.4756** (D2
  corrected, the lowest of the three), top-1 agreement 0 → **0.5591**, top-5 0.844 (pilot 1).
  CE 10.87 → 2.52.
- **Carrier concentration rises much faster with the corrected objective.** In-domain:
  0.0742 → 0.4614 (step 250) → 0.5290 → 0.6302 → 0.6634. Old D2: 0.0742 → 0.1296 → 0.2933 →
  0.4707 → 0.5367. The teacher is 0.6804.
- **The corrected D2 has the highest sink of any student on the corpus where the teacher has
  one.** Cross-domain: D2c 0.030176 (0.888× floor) > P8M 0.027662 (0.813×) > D1 0.024155 >
  D0 0.023213. That is +30% on D0 and 35% of the teacher's 0.085975. The effect is 3.0× the
  seed-noise range, so it is real at this seed. **Attention matching transfers sink magnitude
  where there is magnitude to transfer.**
- **`frac_cells_above_0_2 = 0.0000` for every student, on both corpora, at every step.** No
  student ever develops a single head putting >0.2 attention on position 0. The teacher only
  reaches 3.12% of cells, cross-domain only. The carrier-head signature of a sink is absent
  from the student population entirely.
- **The surgical/coarse split behaves as predicted.** In-domain at step 2,000, all-key vs
  coarse-only vs surgical-only functional cosine:

  | condition | all | coarse (`int_g`,`int_h`) | surgical |
  |---|---|---|---|
  | D0 | 0.9673 | 0.9737 | 0.1080 |
  | D1 | 0.9620 | 0.9684 | 0.1307 |
  | D2 old | 0.9551 | 0.9614 | 0.1164 |
  | **D2 corrected** | 0.9607 | 0.9670 | **0.1326** |
  | **P8M public 8M** | 0.9660 | 0.9704 | **0.8144** |

  The headline ~0.96 is carried entirely by the two whole-subsystem ablations. On the surgical
  subset the students sit at 0.11–0.13 while the converged public 8M reaches **0.81** — so the
  subset discriminates sharply rather than being noisier, and it says the public model shares
  the teacher's surgical response profile while none of our 2,000-step students do. *Method
  note: only D2's row is a pipeline output (`evaluate_transformation_v3`); the others I
  recomputed offline from each row's `delta_ce_json` against the teacher's record. The two
  agree exactly on D2 (0.1326 both ways), so the column is method-consistent.*
- **Follow-up, minor:** the two new columns are written per row but no aggregate table
  surfaces them — `table2_inheritance_components.csv` is long-format over a fixed
  `component`/`metric` list. D0/D1/P8M were not re-evaluated with `--force`, so their rows
  carry the columns empty.

---

## 7. What this means, and what must not be done

**The engineering question is closed.** Training, evaluation, the intervention battery,
manual-vs-NNsight parity on the registered five examples, provenance and the corrected
attention objective all work. Three of four pilot criteria are met on honest evidence.

**The scientific premise is the open question, and it is now quantified.** E6A asks whether a
student inherits its teacher's attention sink. The teacher's in-domain sink is 0.009309 —
**0.86× the uniform-attention floor**, with no carrier head above 0.2 — and it does not meet
0.15 on either corpus. Criterion 2 cannot be satisfied by inheritance from this teacher at any
step count. The converged public 8M's 0.003317 is the best available evidence on what 5,000 or
10,000 steps would find.

What the pilot *did* establish is narrower and more interesting than the registered criterion:
with the objective at its registered scale, attention-map distillation transfers **which heads
carry position-0 mass and how unevenly it is spread** (13.7× and 6.4× the seed-noise range,
three of the teacher's top four heads reproduced), while leaving **total magnitude** and the
**causal fingerprint** essentially unchanged. That is a publishable mechanistic claim about
what attention distillation does and does not carry — but it is not the claim §8.5 criterion 2
tests.

**Do not rewrite the 0.15 threshold, do not re-scope criterion 2 to the cross-domain corpus,
and do not renormalise the metric by the sequence-length floor in the pre-registration.**
Results exist now; every one of those edits would read as tuning (CLAUDE.md rule 4, NEXT_STEPS
§2.7), and none of them rescues the criterion anyway — the teacher is below 0.15 on both
corpora. The floor derivation in §3.2 and the teacher measurement in §3.3 belong in the
paper's *reporting* of a pre-registered criterion that failed, with the reason stated.

The decision that remains is yours and it is about the instrument, not the code. Two coherent
options, both supported by what is now measured:

1. **Report E6A as a pre-registered null**, with §4.3's topology-and-carrier-identity transfer
   as its positive content and §3.2–3.4 as the reason criterion 2 failed. Cheapest, honest,
   and the failure is itself informative: a teacher can transmit *which heads* carry
   position-0 mass without transmitting a sink, because it has none to transmit.
2. **Change the teacher to one that has a sink on the corpus being scored.** The screening
   table already establishes the candidates on real weights: gpt2 → distilgpt2 sit at 16.5×
   and 13.6× floor with ~71–73% of cells above 0.2 and a pair fingerprint cosine of 0.9801
   (NEXT_STEPS §2.9). That is a distillation pair with a strong sink at *both* ends, which is
   what the E6A hypothesis needs and TinyStories cannot supply. Cost: a new pre-registration
   entry, dated, for a *new* experiment arm — not an amendment to this one.

`proceed: false` stands either way — **do not launch the nine Phase 2 runs on this
configuration.**

---

## Appendix — provenance and method

| item | value |
|---|---|
| run commit | `3dc6471d0fabd7d6028df7e6ca135b55e322da5d`, `git_dirty: false` |
| `attn_reduction` | `sum_over_keys_mean_over_queries` |
| loss weights | `ce 0.45 / kd 0.45 / attn 0.10` (unchanged, pre-registered), `T = 2.0` |
| layer map | `{0:1, 1:3, 2:5, 3:7}`; bands: student `[2,8)`, teacher `[1,4)` (depth-normalised, trap 1) |
| init / data | `initial_state_sha256 a6c83ead…`, `block_manifest_sha256 85014934…` — identical to old D2 |
| evaluation | 10 units written, 0 skipped, 0 failed, 1814 s, `evaluate_transformation_v3` |
| aggregation | 52 rows read, 52 used, 0 excluded, `e6_prereg_v4`, no pending decisions |
| parity | 5 sentences, `all_rows_pass: true`, worst `int_i` 1.118e-08 vs atol 1e-5 (894× margin), band `[3,7)` fp32 |

**Pipeline outputs vs my recomputations.** Everything in §1, §2, §3.1, §3.3, §4.1, §5 and §6
except the surgical/coarse table for D0/D1/D2-old/P8M is read directly from the archive. The
uniform-floor values (§3.2), the ×floor normalisations, the 64.5 rescale of D0/D1's `l_attn`,
the seed-noise ratios (§4.2) and the surgical/coarse values for conditions other than D2 are
computed by me from archived quantities; each is reproducible from the files named above, and
the surgical split is cross-checked against the pipeline's own D2 value.

**Sources outside this archive**, all in the repository: the D0/D1/P8M rows and pilot-1
comparisons come from `pilot-run-results-only.zip`; GPT-2's `int_a = 0.563683 ± 0.002170`
(n=300) from `BASELINE_HASHES.json → reference_results.gpt2_small_table1`, captured on
`sample_size 100 / cut_length 40`, band `[3,11)`, fp32; the six screened public models from
`transformation_inheritance/results/screening/screening_fingerprints.csv` (fp32, `e1_100x40`,
`status: ok`, 0 failed). GPT-2 appears twice at 0.563683 (frozen E1 driver) and 0.563006
(screening) — two independent measurements of the same quantity agreeing to 0.12%, which is
also a seam check on the new instrument.

**D0, D1 and P8M were not retrained** — correctly: `attn_weight = 0` makes `0.0 * l_attn`
leave their gradients bit-identical, so the fix cannot have changed their checkpoints. Their
rows in this aggregation are pilot-1 measurements, and their `run_config.json` carries no
`attn_reduction` key, which is the intended marker of a pre-fix run directory.
