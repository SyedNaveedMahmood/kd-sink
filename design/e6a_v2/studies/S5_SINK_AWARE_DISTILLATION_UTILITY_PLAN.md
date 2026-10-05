# S5 extension: Sink-Aware Distillation Utility

Status: **draft implementation and analysis design; no implementation or research execution authorized by this file**.

Prepared by Codex (GPT-6), 2026-10-05 UTC. Repository HEAD at review: `dceba4976c06399cffabdd50e80c41c94137a5e6`. Review includes the existing, uncommitted Stage06 runtime-source repair. This document records proposals separately from already approved S1 requirements. Committing it does not make it a preregistration or approve new scientific choices.

Source: the complete user attachment, **“REPLY TO REV 3 — Experiment: Sink-Aware Distillation Utility”**, including its purpose, interpretation, impact, and artifact requirements. Original attachment SHA-256: `4122d4d3342accf1316d8281ba020d6fd52314275706d76b5e5edd34c595e25c`. Its local filename was `Pasted text.txt`; its machine-specific attachment path is not an experiment dependency. References to “SinkWithoutPlumbing” in that text are motivation supplied by the researcher, not newly verified results from this implementation.

## 1. Purpose and relationship to the project

The experiment asks whether directly supervising position-0 attention mass improves the usefulness of knowledge distillation, and whether a visually strong sink can inflate an impression of internal transfer without a corresponding behavioral benefit. It also asks whether those relationships change during 10,000 optimizer updates.

This is an extension of [S5: sink supervision](S5_SINK_SUPERVISION.md), using the four existing conditions from [S1](S1_LONGITUDINAL_KD.md). It should extend the current S5 analysis and shared evaluator. It requires no C7 condition, new student architecture, new training objective, or additional training seed by default. S1 remains the source of training runs; S5 remains their analysis/evaluation study. S4 supplies separate mechanism probes if subsequently requested; this experiment alone does not identify a transferred circuit. S2, S3, and S6 are not prerequisites for its scientific contrasts.

| Condition | Existing objective | What its comparison can test |
|---|---|---|
| C1, LogitKD | `B = 0.5 CE + 0.5 KD` | Behavioral distillation baseline without an attention target |
| C2, Full AttnKD-JSD | `B + (1/9) JSD_soft(full)` | Adding the complete attention target |
| C5, NoSink | `B + (1/9) JSD_soft(conditional non-sink)` | Replacing full attention supervision with non-sink supervision |
| C6, SinkOnly | `B + (1/9) JSD_soft(binary sink/rest)` | Adding only the sink-versus-rest attention target to B |

Three distinct questions must remain visible in every report:

1. **Pattern:** does the student develop a sink and match the teacher's attention distributions?
2. **Distillation utility:** do clean next-token loss, teacher KL, or agreement improve under one supervision recipe?
3. **Inference dependence:** how do deletion and relocation change predictions and target loss in the resulting model?

A change in output sensitivity is not automatically a change in predictive usefulness. A change in training supervision is not an intervention that removes all routes through which a sink can form. A correspondence between these measurements is not a causal mediation analysis.

## 2. Current state and execution boundary

As recorded in [NEXT_STEPS](../../../NEXT_STEPS.md) and [Stage06](../stages/06_S1_S3_INTEGRATION.md), no S1 scientific campaign has run. The attachment's “completed 10k S1 runs” are required future inputs, not artifacts whose existence this design assumes. This session reviewed repository records and implementation; it did not inventory external run directories or evaluate a checkpoint.

The existing production root `2a11da9bb71957a4d6b3a2f93a34bd67491dd9d21d70577a7a10858930a8943e` is superseded and non-launchable. Its source identity points at calibration-era code. A repair is present in the working tree, but an executable-source milestone and replacement root/configs remain pending. Existing measured data, calibration, hardware, and environment evidence is preserved. This plan does not resolve that blocker or certify launch readiness.

The authorized deliverable for this session is this design and its documentation records. No trainer, objective, evaluator, configuration, protocol lock, or test implementation is changed. No scientific task is marked complete. Future implementation follows an explicit S5 scope and the sequential stage rules; planning this extension does not complete Stage08 science or start the Stage09 release campaign.

### Source identity when implementing later

The pending `s1-runtime-critical-v1` policy covers **all** `src/sinklab/**`, `pyproject.toml`, and `uv.lock`. Adding an analysis function under `src/sinklab` therefore changes that tree even if it does not affect training numerics.

- Finish the separately scoped Stage06 source/protocol repair before treating an S1 checkout as launchable.
- Prefer an isolated analysis checkout/worktree for S5 implementation and post-hoc evaluation. Keep the immutable training source and analysis source as separate identities in derived records.
- A new analysis executable may read a completed run from an older approved source. It must record that fact and validate compatibility; it must not claim to be that training executable.
- If S5 collection is instead integrated into production before any S1 run, amend and freeze the executable source, metric versions, evaluation contract, resource evidence, and replacement production root before launching the entire affected comparison.
- Do not silently refresh an approved root to accept a changed package during ongoing paired training. Do not weaken the runtime-source guard to accommodate analysis work.

## 3. Code audit and concrete gaps

These observations describe the reviewed working tree, not generic desired architecture.

| Existing file/interface | Reusable capability | Work needed for this experiment |
|---|---|---|
| `src/sinklab/objectives.py`: `attention_auxiliary`, `compose_objective` | Exact C1/C2/C5/C6 losses, method IDs, teacher detachment, NoSink/SinkOnly invariances | Preserve their scientific definitions and coefficients; regression coverage only |
| `src/sinklab/models.py`: `AttentionFeatures`, `GPT2Adapter` | Pre-dropout probabilities, scores, Q/K/V, explicit causal edits | Reuse clean features for supplementary metrics; no new model adapter |
| `src/sinklab/metrics.py`: `behavioral_item`, `aggregate_behavior` | CE/PPL, teacher KL/agreement, accuracy, causal effects with sums/counts | Expose already recorded fields through S5; define new decomposition primitives |
| `metrics.py`: `attention_similarity` | Head-mean full and renormalized sink-excluded JSD/MSE with denominators | Preserve v1 semantics; full-minus-conditional is not additive sink attribution |
| `metrics.py`: `sink_profile`, depth helpers | Native sink scalar, layer/head table, depth distance and correlation | Extract profiles/topology into S5, retaining unavailable-value reasons |
| `src/sinklab/evaluate.py`: `evaluate_panel` | RNG-neutral clean/delete/relocate records on dense/full panels; topology and attention similarities already persisted | Add a separately versioned clean-attention supplement only when new scalars are absent |
| `evaluate.py`: `RecordStore`, `_key` | Sealed, immutable completed item records and idempotent retry | A read-only reader must not create directories; version/key design must accept legacy and supplement schemas explicitly |
| `src/sinklab/s5_analysis.py`: `extract_s1_record`, `join_s5` | Verifies aggregates/items; joins C1/C2/C5/C6 by seed/device/panel; five condition contrasts | Current fixed `MEASURES` omits PPL, teacher KL/agreement, relative effects, accuracy, topology, endpoint NLL-only input, temporal summaries, and richer provenance |
| `src/sinklab/analysis.py` | Pairing, missing cadence, optional matched-CE selection primitives | Add exact-grid AUC/late-window summaries and metric-specific completeness |
| `src/sinklab/training_entry.py` | Production identity, mapped teacher evaluation, tensor-content evaluation digest | Read/reuse identities; do not call training to load a checkpoint for analysis |
| `src/sinklab/train.py`, `checkpoint.py` | Structured training log; verified retained/full checkpoint manifests and protected `final-010000` | Build input inventory from actual artifacts; do not assume `run.json` exists |
| `src/sinklab/cli.py` | Explicit independent commands | No S5 command exists; add narrow audit/evaluate/analyze/plot commands later |
| `tests/unit/test_stage08_s5.py` | Basic verified joins, missing conditions, incompatible initialization, duplicate replica rejection | Extend coverage for requested metrics, provenance, versions, decomposition and incomplete grids |

Important details to address in the plan:

- S5's current `run_manifest` is a supplied seven-field dictionary. Its initialization and device fields are not themselves linked to a checkpoint identity by `extract_s1_record`. A production analysis must verify that link.
- The trainer does not currently write the `run.json` described in the software contract. Analysis must normalize verified checkpoint/log/lock evidence into a derived input manifest, never fabricate a historical run manifest.
- The production evaluator uses `_model_digest` over tensor names, dtypes, shapes and contents as `checkpoint_hash`. This differs from SHA-256 of a `.safetensors` file and from the initialization artifact's hash scheme. Store each digest with its kind; never compare unlike hashes as if interchangeable.
- The production `panel_hash` currently identifies the bundled frozen panels document. Distinguish panels with `(bundle digest, panel name, ordered item IDs)`, not digest alone.
- `evaluation_mode="full"` means the evaluator's full metric mode; it can refer to `owt_dense64` or `owt_full300`. It does not imply a 300-item panel.
- `behavior_only=True` on `owt_lm2000` persists clean NLL/counts/accuracy and does not retain teacher KL or attention fields. Its input schema needs a separate extraction path.
- `attention_similarity` averages heads before conditional renormalization. If all non-sink probabilities underflow, it raises an error. An arbitrary uniform distribution must not be substituted.
- `join_s5` currently requires a complete quartet and a fixed finite measure set. Preserve that legacy interface; implement a versioned utility result that can distinguish missing measurements from mathematically undefined ratios without dropping otherwise valid evidence.
- The normal production call requests clean/delete/relocate, not explicit no-op or key-1 controls. The intervention primitive supports them, but planned controls are not evidence that these records already exist. Inventory their actual coverage.

## 4. Scientific invariants inherited from S1

The controlling sources are [DECISIONS](../DECISIONS.md), [OBJECTIVES](../OBJECTIVES.md), [DATA_AND_PROVENANCE](../DATA_AND_PROVENANCE.md), [MODEL_AND_INTERVENTION_CONTRACTS](../MODEL_AND_INTERVENTION_CONTRACTS.md), [TRAINING_AND_CHECKPOINTING](../TRAINING_AND_CHECKPOINTING.md), and [HARDWARE_AND_EXECUTION](../HARDWARE_AND_EXECUTION.md), with sealed researcher amendments taking precedence over stale prose.

| Item | Required setting |
|---|---|
| Teacher/student | Frozen GPT-2-large, 36 layers/20 heads/1280 width; GPT-2-medium initialized from configuration, 24 layers/16 heads/1024 width |
| Pairing | Same seed-0 FP32 initialization, prepared corpus, tokenizer, ordered effective batches and schedule across the four conditions |
| Primary training horizon | 10,000 optimizer updates; no early stopping based on this study's outcomes |
| Effective batch | 64 sequences × 128 input tokens; common measured microbatch 4 × accumulation 16 |
| Counts | 8,192 input tokens and 8,128 shifted targets/update; 81,920,000 input tokens and 81,280,000 targets at 10k |
| Numerics | Locked eager backend, FP32 student/master/Adam state, BF16 autocast, FP32 probability/loss reductions; fixed checkpointing policy |
| Training base | CE/logit KD weights 0.5/0.5; training KD temperature 2 and one `T²` factor |
| Attention coefficient | 1/9 for C2/C5/C6; scale 1 for all three; no new scale calibration or outcome-driven coefficient changes |
| Primary S5 training block | Seed 0, RTX 4080 SUPER, exact approved UUID and environment for all C1/C2/C5/C6 |
| Teacher layer map | `[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35]` corresponding to student `[0..23]` |
| Dense evaluations | `0,100,...,10000`, 101 observations per condition on the same 64 blocks |
| Retained/full evaluations | `0,100,250,500,1000,2000,5000,7500,10000`, nine per condition on the same 300 blocks |
| Independent panel role | `owt_lm2000`: the following 2,000 validation blocks, clean evaluation at 0 and 10k; disjoint block IDs from full300, not necessarily independent documents |
| Persistence | Existing retained weights and protected final full checkpoint remain available; analysis never prunes or rewrites them |

The four primary run IDs are `s1-c1-seed0-rtx4080super`, `s1-c2-seed0-rtx4080super`, `s1-c5-seed0-rtx4080super`, and `s1-c6-seed0-rtx4080super`, as assigned in the reviewed S1 plan. Resolve these against the replacement approved production plan before use. C1/C2 RTX3090 bridge jobs are additional hardware observations of seed 0, not additional seeds or replacements for missing 4080 members.

C5 removes key 0 **before** both alignment and divergence using conditional probabilities from non-sink scores. It does not delete key 0 during training forward propagation. C6 uses `[P(0), sum(P(k>0))]` for both alignment and divergence. It never normalizes a singleton column or uses full-map alignment to select its binary target. A sink may emerge in C5 through shared parameters and the behavioral objective; this is a scientifically valid outcome.

## 5. Estimands, comparison directions, and claims

All differences use `condition A minus condition B` with the direction in the field name and caption. Lower CE, teacher KL, or attention divergence means better performance on that measurement. Higher agreement means closer predictions. Sink amplitude is neither intrinsically better nor worse.

| Priority | Contrast | Question |
|---|---|---|
| Primary | `C5_minus_C2` | What changes when the full target is replaced by conditional non-sink supervision? |
| Primary | `C6_minus_C1` | What does adding only sink/rest supervision contribute beyond LogitKD? |
| Reference | `C2_minus_C1` | Does full attention supervision help relative to LogitKD in this setting? |
| Secondary | `C5_minus_C1` | What does conditional non-sink supervision contribute beyond LogitKD? |
| Secondary | `C6_minus_C2` | What is lost or gained by using only sink/rest instead of the full target? |

Proposed endpoint hierarchy: report the primary contrast at step 10k on full300 for the aligned set of clean CE, teacher KL/agreement, structure, and causal effects. Report clean CE/PPL/accuracy on lm2000 at 10k as a mandatory separate corroborating endpoint. Show the same starting-state measurements at step 0. Dense64 supplies trajectories and temporal summaries. Do not substitute a different panel when an endpoint is missing or select the panel with the most favorable result. This hierarchy must enter the analysis lock before outcome inspection.

### Statistical and interpretive limits

- The mandatory sample is one trained seed. Report actual condition values and within-seed contrasts descriptively. There is no across-seed SD, population confidence interval, reproducibility claim, or hypothesis-test p-value by default.
- “C5 does not harm quality,” “C5 is equivalent to C2,” and “C6 gives no benefit” require a separately justified practical margin and inferential design. A small point estimate, nonsignificant comparison, or overlapping plot is insufficient. No such margin is currently approved; the software must leave equivalence/noninferiority decisions disabled.
- The existing 0.05-nat matched-CE selection tolerance is not an equivalence margin. The fingerprint denominator guard and numerical validation tolerances are also not effect-size thresholds.
- If optional seeds are later approved and complete, retain every seed's contrast and then report sample mean/SD and sign counts over actual distinct training seeds. Do not silently recruit partial additional campaigns into a headline estimate.
- Optional item uncertainty would be conditional on these fixed trained models. Packed blocks can share source documents, so treating all blocks as independent may understate uncertainty. A block/document resampling unit, pairing, dedicated RNG seed, intervals and multiplicity policy require a separate prospective decision; bootstrap is disabled in the initial implementation. Tokens, time points, and hardware replicas never become training replicates.
- Conditional and binary attention objectives carry different information and gradient geometry. The common coefficient does not equalize their gradient norms, compute cost, or information budgets. C5 versus C2 is a comparison of these complete recipes, not an isolated intervention on one parameter.
- Sink-only supervision plus CE/logit KD does not test whether sink mass alone can teach language modeling. Better conditional attention matching is not proof of useful computation transfer. A decomposition of divergence does not measure optimization effort or identify a mechanism.

## 6. Required measurement inventory

All logs use natural logarithms and explicit units. Extract measurements from verified per-item records and reproduce saved aggregates before joining conditions. A training loss field is never a replacement for a clean held-out metric.

| Group | Required measurements | Existing source or additional work |
|---|---|---|
| Clean behavior | CE, fixed-block PPL, next-token accuracy | Clean `behavior` sums/counts and `aggregate_behavior`; separate lm2000 NLL-only schema |
| Teacher matching | `KL(teacher || clean student)` at T=1; teacher top-1 agreement | **Clean operation's** `teacher_kl_nats` and `teacher_top1_agreement_fraction`; available on dense/full |
| Sink strength | Native-layer scalar S; layer/head table; per-layer amplitudes | `clean_structure`, already saved |
| Attention matching | Head-mean full JSD/MSE and conditional sink-excluded JSD/MSE | `mapped_attention_similarity`, already saved per mapped layer |
| Topology | Amplitude profiles; 16-point normalized depth profiles; weighted depth Wasserstein; guarded Spearman | `topology`, already saved; retain null on zero mass/constant profiles |
| Delete/relocate | Signed ΔCE; relative ΔCE%; self-KL; absolute target log-probability change; flip fraction; clean/edited accuracy | Separate operation `behavior` sums/counts and aggregates |
| Causal heterogeneity | Per-item signed ΔCE; median/p90 absolute item change | Derive from the stored sums/counts; preserve paired item IDs |
| Teacher reference | Mapped-layer teacher causal effects and corresponding sink summary | Verify actual teacher-role records; label all-native teacher sink summaries distinctly |
| Optimization audit | Steps, CE/KD/active auxiliary losses, LR, counters, elapsed training time, throughput, memory | `train.jsonl`; audit only, not a common cross-condition objective score |
| New attention decomposition | Binary mass JSD, exact conditional remainder, direct key-0 contribution and closure diagnostics | New versioned supplement from clean teacher/student features; legacy scalar records are insufficient |

`behavioral_item` computes teacher KL against the edited logits for intervention rows. Use its clean row for **clean** teacher matching; do not mix delete/relocate teacher KL into that endpoint.

The teacher-role evaluator can store a sink scalar over all native teacher layers even when the causal scope is the mapped 24. Use the corresponding entries in the layer/head table to derive the mapped reference, and label the all-36 scalar separately. Do not conflate the causal scope with the structure aggregation scope.

### Denominators and reduction order

- Behavior: logits at `q=0..126` predict token `q+1` at length128. Require both real query and target masks. Sum NLL/KL/absolute effects and counts across items, then divide by total valid targets. PPL is `exp(aggregate CE)`.
- Primary sink scalar S: for real length `n`, average key-0 probability over `q=ceil(n/2)..n-1`, then items, heads and native student layers. At n=128 this is queries64..127.
- Attention comparison/decomposition: normalized pre-dropout probabilities, valid causal edges, query `q>=1`, all 24 mapped layer pairs. Average native heads separately before comparison; no 20-to-16 head identity assumption.
- At n=128 each layer/item has 127 attention queries, 8,255 full valid edges and 8,128 non-sink valid edges. These edge counts are unrelated to the separate 8,128 shifted targets per **training update**.
- JSD: sum keys, then average queries within each item/layer and equally average items/layers. MSE: divide each item/layer's sum of squared errors by its own valid cell count, then equally average items/layers. Save numerators/counts and the reduction ID.
- The current S5 `_weighted_attention` pools query or cell counts across items/layers. It agrees with the proposed equal-item reduction on the fixed unpadded S1 panels. Validate this assumption; variable-length reuse requires an explicitly versioned reduction and must not silently inherit equivalence.
- Attention query127 contributes a structural observation but has no next-token target inside the block. Sink S, attention divergences and behavioral effects intentionally use different query rules; record them.

Undefined topology correlations, ratios or overflowed PPL values are `unavailable` with reasons. Missing required records are `missing`; invalid calculations are `failed`. A valid zero is a numeric zero. These states must not be collapsed.

## 7. Attention matching decomposition

### 7.1 Preserve the existing common descriptive measurements

For each example, mapped layer and valid query, let `p` and `s` be teacher and student distributions after separately averaging their native heads. Compute:

- `D_full = JSD(p, s)` over all valid keys.
- `D_cond = JSD(u, v)`, where `u=p[k>0]/sum(p[k>0])` and `v=s[k>0]/sum(s[k>0])`.
- Full and conditional cell-mean MSE with their distinct denominators.

`D_full - D_cond` is **not** the sink's additive contribution. Removing key0 changes normalization, and for MSE also the valid-cell denominator. It is legitimate to show both measures and ranking reversals; it is invalid to call their difference a percentage of transfer caused by the sink.

The order “mean heads, then condition” is part of this descriptive estimand. It differs from averaging per-head conditional distributions, which are used inside C5's training objective. Changing per-head sink scores can change the residual weighting of the head-mean distribution. Do not apply the C5 training-loss invariance test to this descriptive head-mean metric as if the two were identical.

### 7.2 Proposed exact decomposition of the common full-map JSD

At a single query, define:

```text
a = p[0]                     b = s[0]
rT = sum(p[k>0])             rS = sum(s[k>0])
u = p[k>0] / rT              v = s[k>0] / rS
rbar = (rT + rS) / 2
alpha = rT / (rT + rS)
w = alpha*u + (1-alpha)*v

D_mass = JSD([a,rT], [b,rS])
D_shape = 0.5*rT*KL(u || w) + 0.5*rS*KL(v || w)
        = rbar * weighted_JSD_alpha(u, v)

D_full = D_mass + D_shape
```

This is the chain rule obtained by first grouping key0 versus the remaining keys in both distributions and their mixture. Expand each non-sink KL term as `p_k log(p_k/m_k) = rT*u_k[log(rT/rbar) + log(u_k/w_k)]`, sum over non-sink keys, and repeat for the student. The binary mass terms and conditional terms give the identity above. Both components are nonnegative in exact arithmetic. The mixing weight depends on the two residual masses; using a fixed 1/2 inside the conditional remainder generally breaks the identity.

`D_mass` measures disagreement in **sink-versus-rest allocation**. `D_shape` measures the residual-mass-weighted conditional disagreement. `D_cond` remains a separate symmetric comparison that gives non-sink shape equal importance regardless of its mass. Report all three; the exact remainder alone can become small when both distributions put almost everything on key0.

For the literal contribution of the key-0 coordinate to full JSD, also store:

```text
m = (p+s)/2
d_k = 0.5*p_k*log(p_k/m_k) + 0.5*s_k*log(s_k/m_k)
D_key0 = d_0
D_other_columns = sum(d_k for k>0)
D_full = D_key0 + D_other_columns
```

These two partitions answer different questions. `D_mass` includes both sink and complementary residual mass terms; `D_key0` is the literal column contribution. Neither `D_shape` nor `D_other_columns` is equal to `D_cond` in general. Give every field its full name and units.

### 7.3 Attribution of an observed similarity improvement

For condition X relative to C1, define a descriptive gain on exactly matched items/layers/queries:

```text
G_full(X; C1)  = D_full(C1)  - D_full(X)
G_mass(X; C1)  = D_mass(C1)  - D_mass(X)
G_shape(X; C1) = D_shape(C1) - D_shape(X)
G_full = G_mass + G_shape
```

The same identity applies to another prespecified pair. Because the reduction is linear and common, it also holds after averaging. Positive gain means reduced divergence. Keep signed terms: an improvement in mass matching can offset worse conditional shape matching. Do not clip negative gains or force contributions to lie between 0% and 100%.

Default reporting uses absolute gains in nats/query, not a ratio such as `G_mass/G_full`. Such a ratio is unstable when gains cancel or are near zero. No new denominator floor is invented for it. “How much apparent matching improvement” is answered by the additive signed nats, with `D_cond` alongside them. It does not establish a fraction of behavioral improvement mediated by the sink or a fraction of optimizer effort spent on it.

### 7.4 Numerical implementation contract

Implement one pure, no-gradient descriptive primitive in `metrics.py`, with a distinct `s5-attention-decomposition-v1` ID. It accepts normalized probabilities, valid support, and optionally the corresponding scores for stable recovery. All reductions follow the same support and head-mean convention as the common metric.

Use zero-safe `x log x`/KL terms. Sum explicit residual entries rather than subtracting a rounded `p[0]` from 1. If one residual mass is exactly zero, the weighted shape remainder is zero by its limiting definition, while the symmetric conditional comparison is undefined. If both residual masses are zero, both distributions are pure sinks, the full/mass/shape divergence is zero, and conditional shape is unavailable. Never invent a conditional distribution for those probability-only cases.

When finite original scores are available but probability underflow erased residual mass, a new score-derived variant can compute native-head log-softmax values with the exact causal mask, then use `logsumexp(log_probabilities, head)-log(H)` to obtain the **head mean** in log space. Normalize the non-sink part of that head mean, not each head separately. Record the score-derived numerical variant explicitly and verify agreement with the probability path on ordinary inputs. Do not silently rewrite the existing v1 `attention_similarity` definition or its cached values.

Use FP32 production reductions and an independent FP64 reference in tests. Store row-sum error, minimum component, support counts, full-versus-components closure residual and probability-versus-score discrepancy. Exact analytic fixtures use tight FP64 tolerances; production FP32/BF16 acceptance tolerances must be measured on fixed engineering inputs and frozen before scientific use. A small diagnostic residual is not silently assigned to the sink term. Material negatives, nonfinite results, or failed closure produce failed diagnostics.

Do not add epsilon pseudomass to the scientific distributions or accept an arbitrary uniform fallback. If valid-record coverage is reduced by undefined conditional distributions, expose counts and withhold a supposedly complete conditional endpoint unless its coverage matches the registered rule.

## 8. Temporal summaries and outcome interpretation

Primary reporting includes all four individual trajectories on dense64 and full300 retained-state points in separate views. Never pool nested dense64/full300 observations as extra samples.

For a metric `m` on the exact dense grid `t_i=100*i`, the proposed time-normalized area is:

```text
AUC_mean(m) = [sum_i (t_(i+1)-t_i)*(m_i+m_(i+1))/2] / 10000
```

This is a defined trapezoidal summary of observed scheduled samples, not a claim that intervening measurements were observed. Require the complete 101-point grid for a complete primary AUC. Do not bridge a missing scheduled time with a longer trapezoid. Report a coverage map and, if useful, a separately labeled observed-interval partial area with its actual duration; it must not replace the complete AUC. The additional step250 full300 observation does not enter dense64 AUC.

Also report the mean of the 21 dense64 observations at steps8000..10000, requiring all 21. Compute paired temporal contrasts on matched support. Do not select a “best” checkpoint, estimate an onset threshold, or extend only favorable conditions. Newly computed decomposition metrics normally have only nine retained-state full300 observations; do not give them a fictitious 101-point trajectory or dense AUC.

| Observed pattern | Supported interpretation | Claim to avoid |
|---|---|---|
| C5 has lower S and a small observed clean-loss difference from C2 | Direct sink supervision may offer limited descriptive benefit under this recipe; report the exact quality gap | “No harm” or equivalence without a margin/design |
| C5 has better conditional matching while C2 has better full matching | The choice of attention metric changes the ranking; quantify mass/shape gains | Proof that the optimizer wasted a measured share of computation |
| C6 increases S with little observed gain over C1 | Strong structural transfer can accompany limited measured behavioral gain | Universal uselessness of sinks or sink-only language learning |
| C5 worsens clean CE/KL or C6 improves them | Sink/rest supervision may help this KD recipe; report tradeoffs and timing | Rejecting the experiment because the original narrative failed |
| Delete/relocate self-KL or flips grow but signed ΔCE stays small | Output dependence grows with little mean signed target-loss effect; inspect cancellation | Predictions are unchanged or the sink has no function |
| Removing the sink increasingly worsens CE | Evidence of predictive dependence under these interventions at those checkpoints | Complete circuit identification or all-settings necessity |
| C5 develops a sink naturally, or C2 fails to create one | The training recipes did not yield the anticipated pattern contrast | A failed engineering acceptance gate |

Every interpretation stays bounded to GPT-2-large → random GPT-2-medium, the fixed OWT recipe, context128, registered objectives, actual seed coverage, tested interventions and observed horizon. Ten thousand updates are not full pretraining convergence. Later mechanistic or cross-domain claims require the relevant separately scoped study.

## 9. Input audit and provenance contract

Before models are loaded, a read-only audit must produce a content-addressed `s5_utility_inputs` manifest from explicitly supplied run directories and the analysis plan. Fail closed on scientific identity conflicts; report missing files as missing rather than guessing paths from an external historical drive.

Required evidence and checks:

1. **Run selection:** exactly one selected run per C1/C2/C5/C6, one explicit seed, expected method IDs, common training device/UUID block, reviewed production-plan membership, no duplicates or engineering fixtures.
2. **Scientific locks:** verify envelopes and transitive protocol/artifact/environment/hardware/calibration digests, approved variants, tokenizer/teacher/student config identities, layer map, coefficients, precision/backend, common 4×16 schedule and initialization artifact.
3. **Actual run binding:** map checkpoint identity `protocol_hash/data_hash/init_hash/hardware_hash/calibration_hash/model_hash/device_role/gpu_uuid` to normalized analysis fields. Compare actual stored values against locks and supplied plan. A caller-written hash-shaped string is insufficient evidence.
4. **Training lineage:** check steps/counters, saved order snapshot, scheme/seed, consumed-order evidence and schedule state where available. Reconcile logs with the verified checkpoint boundary. Resumed logs can contain replayed updates beyond an earlier recovery point; detect duplicate segments rather than count them twice. If segment lineage cannot be resolved from evidence, training-trace audit remains incomplete; do not invent an order audit.
5. **Checkpoint identity:** verify `COMPLETE`, manifest digest and file hashes. For retained/final evaluation, recompute the evaluator's tensor-content digest from loaded weights to bind evaluation rows to that state. Step0 should additionally match the prepared initialization under its own hash convention. Final state must be step10000 and resumable, but its existence alone does not prove final evaluation completed.
6. **Dense-only states:** most dense checkpoints are not retained. Authenticate their immutable recorded model digests and run provenance, while stating that tensor-level re-verification cannot occur without replay. Never claim all 101 states were reloaded.
7. **Panels:** verify bundle digest, named member, exact ordered IDs/token masks and expected cardinality against the frozen corpus/panels. Check dense64 is the first64 of full300; lm2000 uses the separate 2000 following blocks. Use deterministic ID-based pairing with explicit order handling, not row-number coincidence.
8. **Records:** verify every envelope, record key, aggregate key/filename, operation, model role, scope, step, precision, method/metric/intervention version and item support. Recompute behavioral aggregates from source sums/counts. Snapshot aggregate payload digests because an incomplete aggregate can later be replaced after retry.
9. **Teacher:** verify teacher artifact revision/weights and available teacher-role record identity. Existing student keys alone do not fully name the teacher; bind them through the parent protocol/artifact lock. Repeated teacher records are repeated measurements of one model.
10. **Analysis identity:** record analysis source commit/tree, dependency lock, declared formulas/reduction/version IDs, UTC creation, analysis-plan digest, all input hashes and operator selection. If critical analysis code is dirty, do not call the result frozen production analysis.

The primary analysis requires compatible source protocols and full scientific identities across the quartet. Different condition-specific resolved-config hashes are expected, so compare their shared fields and validate each variant rather than demanding byte-identical configs. A changed root or source variant requires an explicit compatibility amendment; the tool does not infer equivalence from equal seeds or a few matching hyperparameters.

An eventual `run.json` can be consumed if it is present and verifiable, but absence of that unimplemented writer must not cause fake historical records. Store the derived inventory outside the original run directory. Preserve both artifact file hashes and canonical payload/tensor hashes with unambiguous labels.

## 10. Missing metrics, checkpoint recovery, and controls

The default analysis consumes existing records without GPU execution. It emits a matrix by condition, step, panel, operation and metric family with `complete`, `missing`, `failed`, `unavailable`, or `not_requested` status, plus reasons and expected/observed counts.

| Audit result | Next action |
|---|---|
| All requested legacy metrics exist | Extract and analyze them; no checkpoint loading |
| A new decomposition scalar is absent, but retained weights are available | Schedule an explicit clean-attention supplementary evaluation for that condition/step/panel |
| Behavioral/causal records are missing at a retained state | Offer an explicit shared-evaluator re-evaluation job with a new derived output namespace and original parent identities |
| A metric is missing at an unretained dense state | Mark it unavailable; deterministic replay is separately scoped and budgeted, never automatic |
| Only a partial quartet or horizon exists | Produce an incomplete coverage report and observed descriptive rows; withhold a complete primary quartet/trajectory claim |
| Source hashes, versions, masks or devices conflict | Reject the affected comparison; do not repair provenance labels to make it pass |

### Bounded supplementary evaluation

Proposed default scope, after explicit operator authorization: four conditions × nine retained states × full300 clean items. This is **10,800 student/teacher item pairs** for the new decomposition, with 24 mapped-layer summaries per pair. It does not require new optimizer updates. Dense supplementary collection is not assumed available retrospectively.

Use the existing adapters and `rng_neutral` lifecycle, `use_cache=False`, eager attention, frozen teacher and original FP32 saved weights with the registered evaluation autocast. Load only one student checkpoint at a time. Prefer verified `weights-<step>` files; if using `final-010000`, verify its full manifest but load only its model weights for forward evaluation. `load_checkpoint` currently deserializes optimizer state for full checkpoints, so add a narrow verified model-only load path or use verified safetensors directly; do not instantiate `Trainer` or restore/advance the optimizer for evaluation.

All derived records go to a separate operator-owned output directory with a parent-run/checkpoint link and a supplement schema ID. The input run directory, original evaluation records and protected final checkpoint must remain byte-identical. A single evaluation command selects one condition, seed, checkpoint step and named panel; there is no automatic next condition. Failed items retain evidence and require explicit retry.

Supplementary evaluation startup/progress must show S5 analysis ID, parent S1 run/condition/seed, checkpoint step, GPU/UUID, precision, fixed evaluation batch, panel, completed/expected item pairs, elapsed time, ETA and memory. Reuse tqdm for interactive progress and bounded UTC JSONL events for non-interactive use. Resume counts verified completed records, reports newly performed work separately, and never treats a cached item as a new observation. Analysis-only commands report their input coverage without inventing GPU or training-throughput measurements.

Evaluate all four conditions with the same supplementary device, precision, implementation and numerical policy. A different **evaluation** device from the training GPU requires a fixed, measured evaluation assignment and labels for both; it does not authorize training migration. For the minimal plan, use the reviewed 4080 block. Existing S1 training profiles do not establish new evaluator headroom, so first measure a bounded engineering case. Use a fixed evaluation batch, initially the existing one-item streaming pattern, and stop on OOM rather than silently change the scientific computation.

Teacher feature reuse can be added only if needed after measuring cost, with keys including teacher weights/config, item tokens/mask, panel, map, precision/backend and metric version. The initial implementation can recompute teacher features and avoid a new bulk cache. Retain small scalar summaries, not full attention or vocabulary tensors.

### Control coverage

Keep the existing primary key0 delete/relocate metrics and scopes. Audit explicit no-op and key1 positional-control coverage required by the shared contracts. If absent, report it and plan a separately bounded derived evaluation; no control is inferred from a successful unit test.

For key1-versus-key0 deletion, both controls must use the same restricted prediction-query mask `q>=2` with valid shifted targets; at n=128 that is q2..126. The primary key0 metrics keep their original mask. Persist removed attention mass so unequal intervention magnitudes remain visible. Extend evaluation keys to include source/destination key and scoring-mask ID before collecting these controls; an operation label alone is insufficient. No new dose sweep, S4 probe battery, or long-context study is included by default.

The existing FP32/BF16 sensitivity contract also remains applicable. Reuse compatible measured diagnostics; qualify new metrics on fixed engineering inputs. Numerical discrepancies cannot be used as practical equivalence margins. If required controls/diagnostics are unavailable, report the incomplete gate and restrict causal claims accordingly.

## 11. Proposed schemas and output artifacts

Use explicit dictionaries or small dataclasses within the existing package. Avoid an experiment registry, database, orchestration service, distributed runner or second trainer.

### Analysis plan

A future `protocols/s5_utility_analysis_draft.json` should contain: analysis ID/version/status; parent approved S1 root; four run IDs; one seed and training device block; source/analysis identities; panel hierarchy/cadences; contrast directions; reduction and metric versions; decomposition numerical variant; supplementary evaluation scope/device/budget; handling of missing data; descriptive inference mode; optional features disabled; and approval/amendment records. Seal an approved analysis lock only after those values are reviewed. No such lock is created by this planning session.

### Normalized analysis rows

Each row identifies source study S1 and derived analysis S5, run/condition/method/seed, training device/UUID, evaluation device/UUID, optimizer step, panel name/bundle/member digest, scope/map, checkpoint tensor and file digests where available, all source lock hashes, source aggregate/item-bundle hashes, metric schema/version, reduction ID, item IDs/counts/target counts, and status/reason.

Store measurements in named groups rather than a single all-finite dictionary: `behavior`, `teacher_matching`, `structure`, `attention_similarity`, `causal_delete`, `causal_relocate`, `decomposition`, and `audit`. Each value carries units and availability. Keep per-item numerators/counts needed for paired effects. Missing new decomposition does not erase a valid legacy CE measurement. Complete primary claims still require all their registered inputs.

Supplement keys include parent run/checkpoint, item token/mask identity, teacher identity, exact map/scope, clean operation, numerical policy, metric/reduction versions, analysis-plan digest and evaluation-source identity. Generic store support must parameterize new schema/version keys while retaining the exact legacy key behavior. Do not globally bump `METRIC_VERSION` and thereby make old records unreachable, or mix old/new definitions under one key.

### Proposed derived artifact layout

```text
<external-analysis-dir>/
  analysis-plan-<digest>.json
  inputs-<digest>.json
  coverage.json
  audit.json
  supplementary/                 # only explicitly requested evaluations
  item-measures.jsonl
  condition-measures.csv
  paired-contrasts.csv
  temporal-summaries.csv
  decomposition.csv
  figures/                       # PDF and SVG, optional PNG previews
  figure-manifest.json
  report.md
  artifact-manifest.json
```

Artifact manifest lists input/output digests, algorithm/source versions, statuses and creation time. Every figure names its source row IDs, source artifact digests, exact query/filter, units and caption. Large outputs remain external; commit only small sanitized evidence and documentation. Table exports cannot turn null into zero or omit unfavorable rows.

## 12. Planned code changes

The following are future edits, not work performed by this document.

| File | Planned responsibility |
|---|---|
| `src/sinklab/metrics.py` | Pure head-mean attention decomposition with counts, exact algebraic closure, stable zero handling and explicit version; preserve legacy behavior |
| `src/sinklab/evaluate.py` | Read-only record access; explicit supplement namespace; clean-feature supplementary pass sharing model/mode/RNG handling; versioned control keys if needed |
| `src/sinklab/s5_analysis.py` | Input audit and normalized identity validation; grouped metric extraction including NLL-only endpoints; matched quartets, contrasts, coverage and derived report serialization |
| `src/sinklab/analysis.py` | Reusable exact-grid temporal summaries and pairing checks; conditional uncertainty only if separately approved |
| `src/sinklab/checkpoint.py` | Only if necessary: verified model-only load for full checkpoints, preserving existing full-resume semantics |
| `src/sinklab/cli.py` | Thin dispatch for the four explicit S5 utility commands below; no training fallback |
| `src/sinklab/plotting.py` (new) | Small deterministic Matplotlib figure/table renderer over frozen derived records; no computation from unverified raw data |
| `tests/unit/test_stage08_s5.py` | Extend legacy extraction/join regression and input provenance rejection |
| `tests/unit/test_s5_utility.py` (new) | New schemas, measures, reductions, completeness, contrasts, temporal and interpretation guards |
| `tests/unit/test_attention_decomposition.py` (new) | Independent numerical references and analytic decomposition cases |
| `tests/unit/test_metrics_evaluate.py`, `test_train_resume.py` | Existing evaluator and RNG/training parity gates; supplement immutability and repeatability |
| `tests/integration/test_s5_utility_cli.py` (new) | Explicit flags, offline use, missing artifacts, no model load for analysis, immutable inputs, traceable output |
| `tests/gpu/test_s5_utility_gpu.py` (new) | Opt-in bounded real-shape numerical/headroom check; not scientific acceptance of an outcome |
| `protocols/s5_utility_analysis_draft.json`, study docs, journals, reports | Later schema/decision freeze, measured evidence and truthful capability/coverage status |

Keep `extract_s1_record`/`join_s5` as compatibility interfaces or explicitly version their replacement. New grouped records must not make existing tests accept incomplete legacy inputs. Factor helpers only where audit/evaluation/analysis genuinely share them. Avoid a large general provenance subsystem for this one experiment.

`train.py`, `make_objective_loss`, optimizer/scheduler/data-order logic, and the C1/C2/C5/C6 definitions require no planned scientific change. If the audit discovers a correctness defect there, record it as a separate prerequisite with affected runs and amendment/rerun requirements; do not conceal a trainer repair inside an analysis feature.

### Proposed command surface — not implemented

```text
sinklab s5-utility-audit --inputs <paths.json> --analysis-plan <plan.json> --seed 0 --out-dir <external-audit-dir>
sinklab s5-utility-evaluate --inputs <verified-inputs.json> --analysis-plan <approved-plan.json> --condition C5 --seed 0 --step 10000 --panel owt_full300 --device-role rtx4080super --out-dir <external-supplement-dir>
sinklab s5-utility-analyze --inputs <verified-inputs.json> --analysis-plan <approved-plan.json> --seed 0 --out-dir <external-analysis-dir>
sinklab s5-utility-plot --analysis-manifest <artifact-manifest.json> --out-dir <external-figures-dir>
```

`--inputs` maps explicit local paths to the four selected runs, locks, panels, corpus and optional supplements. Audit may operate against a draft plan and report missing inputs. Scientific evaluation/analysis requires the approved plan; tiny fixture tests use an explicit fixture mode and cannot emit production-ready status. Analyze reads all four conditions as one comparison, but it does not train/evaluate them. Evaluate handles one specified checkpoint only. Plot only consumes the verified analysis artifact. Unknown fields, implicit seed lists and automatic downloads fail. Use clear exit codes for complete, incomplete coverage and invalid inputs, frozen in CLI tests.

## 13. Figure and table plan

1. **Primary endpoint table:** four conditions and five prespecified contrasts at10k. Separate full300 behavior/structure/causal columns from lm2000 clean metrics. Include units, seed, device, panel and missing-value reasons; no automatic “equivalent” label.
2. **Dense trajectories:** shared x-axis optimizer updates, with the corresponding input-token scale documented. Separate panels for CE, teacher KL/agreement, S, delete ΔCE/self-KL, and complementary relocation effects. Show raw measurements; no smoothing that changes conclusions.
3. **Attention matching:** full and conditional JSD/MSE trajectories on identical support; retained-state mass/shape decomposition shown at its actual nine time points. Lower divergence is labeled explicitly.
4. **Gain decomposition:** signed C1-relative gains in full JSD split into mass and shape terms, with conditional JSD beside them. Use diverging bars when components oppose one another; no forced positive “percent explained.”
5. **Pattern and dependence:** matched-step S versus deletion effect, with condition/step labels. This is descriptive, not a regression claiming that S causes utility. Pair signed loss effects with self-KL or absolute effects to expose cancellation.
6. **Layer profile appendix:** student native-layer/head tables and mapped teacher profile, with amplitudes preserved and normalized-depth distance separately labeled. Never align 20 teacher heads to16 student heads by index.
7. **Coverage/provenance table:** complete expected observations, missing/failed controls, supplementary evaluation provenance, source versions, actual seeds and hardware roles. An incomplete output cannot look like a finished scientific result.

Generate standalone PDF/SVG scientific artifacts using the existing Matplotlib dependency. Include raw table files and caption/source manifests. Fixed-seed bootstrap bands, significance markers, causal onset lines and matched-CE figures are absent unless their separate analysis choices have been approved.

## 14. Implementation sequence and exit gates

Each item is a future task. No checkbox below records completed implementation. Implement one bounded milestone at a time; keep CORE/S1/S5 append-only journals and small evidence reports with each milestone.

- [ ] **SU0 — Freeze scope and resolve prerequisite status.** Recheck Stage06 and artifact availability. Review this draft's new analysis choices, preserving original S1 approvals. Record whether collection will be post-hoc in an analysis checkout or integrated before all affected runs. Exit: explicit analysis scope/decision record; production training still obeys its own readiness gate.
- [ ] **SU1 — Input schemas and read-only audit.** Add strict plans/manifests, legacy schema readers, checkpoint/lock binding, panel-member verification and expected coverage inventory. Use local fixtures. Exit: corrupt/mixed/missing evidence is handled correctly; audit creates no source directories and cannot launch a model or training.
- [ ] **SU2 — Complete reuse-only measurement extraction.** Extract all existing behavior/teacher/structure/topology/causal fields and separate lm2000 endpoints; implement matched primary/secondary contrasts and exact-grid summaries. Exit: independent hand-calculated fixture results and legacy regression pass; partial data remains explicitly incomplete.
- [ ] **SU3 — Exact decomposition primitive.** Implement the formulas, stable variants and units/counts; validate independent FP64 references and FP32 limits. Exit: all algebraic identities/invariances pass without a desired experiment outcome being hard-coded.
- [ ] **SU4 — Bounded derived evaluation and controls.** Add one-checkpoint clean supplement, model-only load, immutable namespaces, resume/retry and source identities. Scope missing causal controls separately. Exit: CPU model/evaluator parity, RNG preservation and source immutability pass; real GPU qualification remains a separate unchecked gate until actually run.
- [ ] **SU5 — Reporting and reproducible artifacts.** Add CLI analysis/plot paths, tables, figures, hashes and coverage reports. Exit: every point/row is traceable; figures faithfully include adverse and missing outcomes; no scientific output is generated from absent jobs.
- [ ] **SU6 — Qualify the execution path.** Run full CPU regression and offline clean-wheel gate. Under a separate bounded operator scope, measure new real-size GPU metrics, numerical errors, headroom/time/disk and restoration. Exit: measured evidence for the selected evaluator role; missing GPU evidence remains blocked, not skipped-as-passed.
- [ ] **SU7 — Execute the approved analysis.** After the four approved S1 runs and required records exist, audit first, run only authorized missing supplementary evaluations, verify the complete quartet/coverage, then produce frozen analysis artifacts. Exit: actual coverage, limitations and protocol deviations recorded separately from software capability.
- [ ] **SU8 — Scientific review and handoff.** Review claims against margins/seed limits/numerical controls and the contrary-outcome table; record analysis source, all artifacts and remaining gaps. Exit: a reproducible scoped S5 report. This does not automatically satisfy unrelated Stage09 release gates.

Suggested milestone subjects: `feat(s5): audit utility analysis inputs`, `feat(s5): extract utility contrasts and temporal summaries`, `feat(s5): add exact attention divergence decomposition`, `feat(s5): evaluate retained-state attention supplements`, and `feat(s5): render traceable utility reports`. Commit code/tests/journals/evidence together at their actual passing gates; never mark the study scientifically complete because fixture tests pass.

## 15. Required validation plan

### Mathematics and reductions

- Verify identical distributions produce zero full/mass/shape/coordinate divergence. Use a direct independent reference, not the production helper to compute expected answers.
- Same non-sink conditional distribution with different sink masses: `D_shape=0`, `D_cond=0`, `D_full=D_mass`; the literal key0 term need not equal the binary mass term.
- Equal sink mass `a=b`: `D_mass=0`, `D_full=(1-a)*D_cond`.
- Equal sink mass but different conditional shapes: nonzero conditional divergence; invariance of the binary mass metric to redistribution at fixed mass.
- Unequal residual masses: demonstrate the correct weighted conditional mixture and reject an erroneous fixed-1/2 remainder.
- Test both pure sinks, one pure sink, zero entries, near-saturated probabilities and finite-score underflow. Undefined conditional values have reasons and coverage counts, not fabricated zeros.
- Test key0-plus-other and mass-plus-shape closure per query, item, layer and aggregate; signed gain closure with positive and negative components.
- Test right padding, future-key exclusion, q0 exclusion, length2 q1 conditional zero, length1 rejection, differing20/16 heads and exact layer map. Test invariance to head permutations within each model, not fake head-index homology.
- Demonstrate mean-then-condition differs from condition-then-mean on a designed multihead example; preserve the specified choice.
- Test JSD query mean versus MSE cell mean, fixed128 counts, equal-item versus pooled reductions on variable lengths, and disjoint behavioral/attention query rules.

### Records, provenance, completeness and claims

- Reject mismatched init/order/protocol/tokenizer/teacher/map/numerics/device evidence, unapproved roots, duplicate run IDs, duplicate hardware replicas, duplicate JSON keys, corrupted hashes and mislabeled engineering inputs.
- Verify current caller-supplied identity fields against real synthetic checkpoint manifests; test unlike hash-kind rejection and panel-name distinction even when bundle hashes match.
- Verify clean teacher KL uses the clean row and T=1; training KD/T² and intervention teacher KL never leak into the clean comparison.
- Test PPL from aggregate CE, signed loss cancellation versus absolute effects/flips, relative denominator guards, undefined topology, and NLL-only endpoint extraction.
- Test the complete dense/full/LM grid, missing step0/10k, missing interior dense step, extra250 exclusion from dense AUC, missing late-window observation, and no filling of absent decomposition checkpoints.
- Test legacy v1 cache reads and new supplement reads independently; changed versions/numerical variants cannot collide. Verify output reruns are idempotent and source files are byte-identical.
- Test one-seed reports omit across-seed uncertainty, seed1729 cannot appear as a production replicate, bridge rows do not increase N, and no equivalence/onset flag can be emitted without the separately approved design.
- Include fixture datasets representing C5 harm, C6 benefit, increased causal dependence, natural NoSink sinks and incomplete quartets. All scientifically valid outcomes must render correctly.

### Evaluation, resource and operational checks

- Tiny GPT-2 feature parity; no-op parity; model state/modes/hooks and Python/NumPy/Torch CPU/CUDA RNG restoration on success and injected failure.
- Supplement insertion leaves subsequent training trajectory unchanged in the existing tiny resume/evaluation parity harness if collection is integrated; offline evaluation never creates optimizer state or updates model weights.
- Model-only loading from protected final checks the full manifest but does not deserialize optimizer state. Interrupted output writes preserve valid original records and require explicit retry.
- Read-only audit/analyze/plot path cannot invoke training, launch a GPU evaluation, download data, create a missing source record directory, or delete a checkpoint.
- Offline wheel/install with Upstream absent; all supplementary inputs are explicit external artifacts. No new runtime/test import from either `Upstream/` or `upstream/`.
- Real full-size teacher/student test at128 on the intended GPU, fixed engineering inputs, finite metrics, measured closure/parity tolerances, memory headroom, walltime and disk growth. Test data and thresholds are selected before seeing S1 scientific outcomes.
- If extra controls are collected, key0/key1 comparisons share q>=2 shifted-target masks, expose removed mass, and have distinct immutable record keys.

Planned commands after the named tests exist; **none was executed as part of this design**:

```powershell
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_attention_decomposition.py tests/unit/test_s5_utility.py tests/unit/test_stage08_s5.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit/test_metrics_evaluate.py tests/unit/test_analysis_primitives.py tests/unit/test_train_resume.py tests/integration/test_s5_utility_cli.py
.\.venv\Scripts\python.exe -m pytest -q tests/unit tests/integration -m "not gpu and not network"
.\.venv\Scripts\python.exe -m pytest -q tests/integration/test_clean_wheel.py
.\.venv\Scripts\python.exe -m pytest -q tests/gpu/test_s5_utility_gpu.py --device-role rtx4080super
```

The final GPU command additionally needs explicitly supplied verified local model/fixture inputs and an external evidence directory under the existing GPU harness conventions. Test collection/passing counts, environment, command, exit status, failures/skips and evidence paths must be recorded at execution time. Missing required GPU inputs/hardware block that gate.

## 16. Resource scope and feasibility

The minimal utility comparison is CPU analysis over scalar records. Existing expected student coverage for the quartet is 404 dense panel evaluations, 36 full300 evaluations and8 lm2000 endpoints. With three operations on dense/full and one on lm2000, this is 125,968 student item-operation records before supplements; teacher and optional control records are additional. Inventory actual records rather than assuming these counts were produced.

The proposed new decomposition pass is 10,800 clean teacher/student item pairs across retained full300 states. Its cost must be measured; no GPU-hour or disk estimate is asserted by this document. Begin with an explicitly authorized, bounded engineering checkpoint/item sample. Project time from measured per-item work plus checkpoint loading, teacher handling, hashing and serialization. Record uncertainty and whether caching is included. Do not hash multi-gigabyte checkpoint files repeatedly inside an item loop.

Use one resident student checkpoint and one pinned teacher if measured to fit. Keep only current features and scalar outputs. Profiles from a different evaluator policy do not authorize a different residence/chunking strategy without parity and headroom validation. No automatic checkpoint download, deletion, training replay, additional seed or GPU migration is included.

## 17. Decisions to freeze before scientific use

| Decision | Proposed/default disposition | Authority/status |
|---|---|---|
| Study identity and source conditions | Extend S5, reuse S1 C1/C2/C5/C6 | Existing study definition; matches attachment |
| Architecture, seed, objectives, batch, data, retention | Preserve Section4 settings | Existing researcher-approved S1 contract; production source blocker still open |
| Primary hardware block | Seed0 RTX4080 SUPER quartet | Existing reviewed plan; validate replacement locks |
| Endpoint/panel hierarchy and five contrasts | Section5; no panel substitution | Proposed S5 analysis specification to freeze |
| Added similarity decomposition | Common head-mean JSD mass/shape and coordinate partitions | New descriptive measurement proposal; needs version/analysis approval |
| Supplement scope | Nine retained states on full300; one explicit checkpoint per execution | Proposed bounded evaluation; operator scope and measured resources required |
| New metric numerical variants/tolerances | FP32 production, independent FP64 reference; measured engineering tolerances | Pending implementation/measurement; no invented scientific threshold |
| Formal quality-equivalence/noninferiority claim | Disabled | Practical margin and inferential design unresolved |
| Item bootstrap, matched-CE analysis, doses, extra seeds, S4/S6 extensions | Disabled in this implementation's initial scientific scope | Separate prospective approval/budget if requested |
| Reporting missing/contrary outcomes | Always explicit; no result-based acceptance gate | Existing project requirement |

The remaining decisions do not prevent implementation of formulas, schema validation and local engineering tests under a later coding instruction. They do prevent silently presenting proposed values as an approved scientific protocol or launching unscoped computation. Timing must be honest: an analysis frozen after outcome inspection is post-hoc even if its formulas are sound; existing immutable training outputs retain their original provenance.

## 18. Definition of done

**Design complete:** the requested experiment is mapped to S5; concrete implementation gaps, exact estimands/formulas, provenance, versioning, controls, tests, resource scope, interpretation and unresolved decisions are documented. This file satisfies that planning deliverable only.

**Capability complete:** SU1–SU6 implementations and applicable gates actually pass with evidence, the source/version identities are frozen, and the approved analysis configuration is executable without training or modifying input artifacts. GPU skips do not establish GPU capability.

**Scientific experiment complete:** the selected four approved S1 runs and required checkpoint/record coverage exist; SU7–SU8 are executed under approved scope; all primary results and missingness are traceable; the report includes contrary findings and confines its claims to the actual seed/panel/horizon/numerical evidence. A zero or adverse benefit is a completed experiment, not a failed experiment.

The next implementation action, after a coding instruction, is SU0/SU1: reconcile the runtime-source prerequisite and define the strict input audit against tiny local fixtures. No scientific outcome or additional training launch is an implementation acceptance criterion.
