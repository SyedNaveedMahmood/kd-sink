# KD-SINK: Mechanistic Validation and Developmental Causal Experiments

**Status:** Proposed research plan — **not executed, preregistered, or approved**  
**Prepared:** 2026-10-08  
**Repository:** https://github.com/SyedNaveedMahmood/kd-sink  
**Repository state reviewed:** `main` at `dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c`  
**Scope:** The three unresolved claims from the paper discussion: (i) teacher–student mechanism differences, (ii) how route correspondence changes during student training, and (iii) why sink-deletion sensitivity grows.  
**Boundary:** A design document only. **Do not run experiments, download checkpoints, resume training, change locked protocols, or edit the repository without separate researcher approval.**

---

## 0. Executive decision

The existing findings justify **a route-specific difference** and **a developmental dissociation between sink mass and sink dependence**. They do **not** yet justify either (a) entirely different teacher/student circuits, (b) increasing mechanistic divergence with time, or (c) an identified causal mediator of increasing dependence.

The optimal sequence is **read-only audit first → selective-route confirmation → within-checkpoint attention-path decomposition → causal sensitivity/rescue → only then, if needed, controlled training continuation**. This obtains the maximum information from retained seed-0 checkpoints before requesting expensive new training.

### Current evidence versus the stronger target claims

| Claim | Existing measured basis | Current defensible wording | Evidence needed to strengthen |
|---|---|---|---|
| Mechanism difference | S4: teacher Q-bias intervention ΔCE `0.05599`, EPE transport ΔCE `0.11781`; C2 at 10k Q-bias ΔCE `0.00090`, EPE ΔCE `0.00001` | **The tested Q-bias and EPE routes have different measured intervention responses.** | Verify effective perturbation strengths, intervention selectivity, matched scope/dose, numerical floors, and held-out-panel stability. |
| Diverges over training | S4 contains **five** checkpoint batteries per S1 condition, not zero longitudinal probe data | **Whether relative route responses converge, diverge, or stay stable is unresolved.** | Reanalyze all five checkpoints against the fixed teacher, with prespecified scale-aware fingerprints and a confirmatory panel. |
| Why dependence grows | C2 S5 Full300: step 500 `S=0.33185`, delete ΔCE `0.01241`; step 10k `S=0.33196`, delete ΔCE `0.37335` | **Similar sink mass can accompany ~30× greater deletion penalty.** | Account for head-value difference, output projection, downstream response, and clean-loss geometry using exact local identities and causal interventions. |

**Important nuance:** Existing S4 *already* contains a sparse longitudinal route-probe series at steps `0, 100, 500, 2000, 10000`. The missing evidence is a **complete, calibrated causal comparison**, not the total absence of time-varying probe data.

### Priority / compute overview

| Work package | Type | Models / steps | Main answer | Priority |
|---|---|---|---|---|
| **E0: S4 trajectory and artifact audit** | Reanalysis only | All S4 recorded states + teacher | Is an apparent route trend already visible? | **P0 / do first** |
| **E1: Controlled route-fingerprint replication** | New seed-0 inference, new approval | Teacher; C2/C3/C6; C1/C5 controls; 500/2000/10000 | Are teacher/student route differences robust to scope, dose and controls? | **P0** |
| **E2: Exact attention-output pathway accounting** | New seed-0 inference, new approval | C2 + C6; C5/C1 controls; 500/2000/10000 | Does local deletion perturbation itself grow, and where? | **P0** |
| **E3: Downstream sensitivity + rescue** | New seed-0 inference, new approval | Selected layers/models from E2 | Does growing downstream sensitivity explain the effect beyond local perturbation amplitude? | **P0/P1** |
| **E4: Controlled checkpoint continuation** | New training and separate approval | Branched C2 checkpoint at 500 | Does preventing a candidate pathway's adaptation suppress later dependence? | **P1 conditional** |
| **E5: Independent-seed / architecture replication** | New data/training/approval | Prospective campaign | Does the causal explanation generalize? | **P2** |

A successful E0–E3 campaign can substantially strengthen the manuscript **without** E4. E4 is needed only if the manuscript wishes to make a stronger *developmental causal mechanism* claim rather than a local explanation and intervention-supported pathway account.

---

## 1. Non-negotiable repository and scientific constraints

1. **D24 applies to all new S1 checkpoint inference:** seed `0` only. Recorded seed-1/2 *numbers* may be reanalyzed, but their missing model tensors must not be silently reconstructed or requested under the old follow-up approval. Do not count prompt items, probe RNG, devices, or checkpoints as independent training seeds.
2. **No implicit approval:** D24 permits the seed-0 population of follow-ups; it does *not* approve a new scientific study, new panel, extra probe battery, or training continuation. The new work needs a prospectively signed/hashed **new analysis/intervention lock** specifying exact panel, checkpoint lists, algorithm, acceptance gates, and device; any D24 change needs an explicit amendment.
3. **Do not overwrite S4/S5/S7:** Keep their locked definitions, sealed receipts and result records unchanged. Add new provenance and a distinct study identifier only after approval; never retrofit old results into a new preregistration.
4. **Student/teacher architecture mismatch:** teacher is GPT-2-large, 36 layers/20 heads/1280 width; student is GPT-2-medium, 24 layers/16 heads/1024 width. A common head index, neuron index or parameter coordinate is **not** a homologous component. Use model-local probes and separately report *mapped-24-teacher-layer* versus *native-36-teacher-layer* scopes.
5. **Intervene inside actual computation:** S1 attention editing acts on normalized causal attention probabilities **before** value aggregation, not on displayed attention. Match masks, q>=1 eligibility (q>=2 when matching key-1 controls), model mode, precision, and input identities.
6. **Use an independently specified panel for new confirmatory inference:** start with the fixed OpenWebText Full300 for historical comparability, but reserve a **disjoint, checked and sealed** confirmatory item set (e.g. a predetermined subset of the existing disjoint LM2000 corpus if licensing/provenance and overlap checks pass). Do not tune probe definitions or select layers on confirmation examples.
7. **Report clean CE and teacher output metrics separately from intervention sensitivity.** A high deletion ΔCE is not evidence that sink supervision improved training utility.
8. **New code must be independently validated** by CPU fixture/unit tests, FP32 eager-attention parity, GPU numerical gates and artifact-source identity checks. The repository includes infrastructure, not a completed implementation of the proposed new causal-analysis tools.
9. **Preserve provenance:** source model SHA, historical original protocol root, D24 follow-up SHA, analysis-lock SHA, source commit, exact code tree, full panel hash, model and device identity, seed, step, measured precision, fixed scopes, intervention strengths, numerical floors, and item IDs.

### Readiness gate — must be GREEN before any new inference

- [ ] Confirm availability and SHA-256 integrity of **seed-0** retained checkpoints, especially C2/C6 and C1/C5, at designated steps.
- [ ] Confirm teacher identity and fixed tokenizer/panel; audit saved source roots and the hardware role/bridge limitations.
- [ ] Confirm the intended study identifier and checkpoint list are admitted by `sinklab.followup_policy.admit_s1_followup(...)` **before** model loading.
- [ ] Create and approve a separate analysis/intervention protocol amendment; preserve D24 and S4 protocol unchanged.
- [ ] Freeze discovery and holdout panel membership, route doses, layer scopes, numerical floors, statistical analysis and decision rules.
- [ ] Validate GPU memory/profiling on minimal synthetic inputs; establish no-op and restore parity.
- [ ] Confirm available disk capacity and external artifact locations. Checkpoint weights are stored outside the Git repository; repository tests alone cannot demonstrate scientific completion.

---

## 2. Core research questions and competing hypotheses

**RQ1 — Route identity:** Do teacher and student depend on the **same tested functional routes** for their attention sinks?

- **H1a (inheritance):** When route perturbations are made comparable in locus, effective dose and layer scope, teacher and student show similar sink and output response curves.
- **H1b (route-specific noninheritance):** Teacher responds meaningfully to specific Q-bias/EPE probes but C2/C3/C6 do not, *despite verified effective interventions*, with effects distinguishable from matched controls and numerical floor.
- **H1c (underdetermined):** Raw teacher/student ΔCE differences disappear after scope/dose control, or probe effects are too weak/nonselective to distinguish routes.

**RQ2 — Developmental trend:** Does correspondence with the teacher's **tested route-response fingerprint** increase, decrease, or remain unchanged from step 500 to 10k?

- **H2a:** Convergence (student responses move towards teacher profile).
- **H2b:** Divergence (student responses move away).
- **H2c:** Stable difference / mixed route-dependent change / nonmonotonic trajectory.

**RQ3 — Why late dependence?** Is the growing deletion penalty primarily associated with (i) the **local pre-projection value difference**, (ii) greater **post-projection attention-output perturbation**, (iii) greater **downstream response to the perturbation**, or (iv) the changing loss/output-distribution geometry?

- **H3a (value/output-amplitude explanation):** Local projected deletion perturbations become much larger while equal-norm downstream sensitivity stays similar.
- **H3b (downstream-gain explanation):** Local perturbation stays similar but matched-norm downstream logit/KL effects increase.
- **H3c (both):** Both the local perturbation and downstream amplification change.
- **H3d (loss-geometry explanation):** Token-level clean distributions/loss gradients account for much of the CE increase without comparably increased output-distribution displacement.
- **H3e (unexplained):** No tested combination accounts for most of the change or responses depend strongly on layer interactions.

Do **not** word any of these alternatives as discovered in advance.

---

## 3. E0 — Read-only mechanistic trajectory audit (no new inference)

### Rationale

S4 already measured the same 10 probe types for all C0–C6 at `0/100/500/2000/10000`, plus a fixed teacher. Before adding new experiments, recover **the actual time series of route response**. This is required to prevent the paper from incorrectly claiming no longitudinal mechanistic observations exist.

### Data and exact source

- `reports/results/S4_RESULTS_20261007.md`, and its referenced hashed per-probe summary files under external `S4_attempt02`.
- `reports/results/S1_C0_C6_RESULTS_20261007.md` and `S5_RESULTS_20261007.md` for matched trajectory context.
- Existing report verification: `scripts/report_stage08_s4.py` and `scripts/report_stage08_s5.py`.
- Frozen teacher S4 reference and the original S4 probe control seed are unchanged.

### Analysis

1. Verify report/source hashes and the exact 35 student states plus teacher state; no selected-only summary.
2. Extract each probe's **absolute** ΔS, relative ΔS where nondegenerate, ΔCE, self-KL, absolute log-prob change and flip fraction. Keep all 10 probes; report each of five random K-coordinate controls, not only their average.
3. Tabulate complete C2/C3/C6 trajectories and C1/C5 low-sink controls. Flag step-0 values as a **randomly initialized student**, not mature circuit evidence.
4. Graph teacher's constant reference on each time-series panel. Report whether Q-bias and EPE response differences **increase, decrease or remain flat**; do not interpret a single endpoint difference as developmental divergence.
5. Analyze route response jointly with sink mass, clean CE and standard attention deletion sensitivity. Record signed changes and nonmonotonicity.
6. **Exploratory fingerprint:** a vector of separately standardized, unit-annotated probe responses. Fix the recipe on a discovery subset, show **each component** and the unscaled values. Do not let a cosine score with near-zero student responses masquerade as inherited function. Any composite distance is exploratory until independently confirmed.

### Deliverables and decision

- `E0_S4_ROUTE_TRAJECTORIES.csv`, `E0_TEACHER_STUDENT_FINGERPRINTS.csv`, full provenance JSON, plot with teacher line and 5 checkpoint observations, and an explicit **converge / diverge / mixed / insufficient** conclusion.
- **Decision gate:** If the existing probes show convergence or mixed patterns, the paper may not advertise increasing rebellion. Proceed with route-specific mechanisms and delayed dependence instead.
- **Claim permitted:** 'Measured route-intervention responses differ and have a documented checkpoint trajectory.' **Not:** 'The circuits grow more different over time.'

---

## 4. E1 — Calibrated teacher–student route-intervention comparison

**Status:** Requires newly approved seed-0 checkpoint inference; proposed extension, **not** an S4 rerun.

### Purpose

Rule out five alternative explanations of the teacher/student Q-bias and EPE contrasts: different parameter scales, different model performance, different fraction of sink mass actually disrupted, unmatched layer scopes, and nonselective perturbation effects.

### Models, checkpoints, panels

- **Teacher:** frozen GPT-2-large once per unique approved panel/probe/scope combination.
- **Primary student:** C2 seed0 steps `500, 2000, 10000`.
- **Independent high-sink controls:** C3 and C6 seed0, same steps, subject to validated artifacts/resources.
- **Low-sink controls:** C1 and C5 seed0 at `500, 10000` (expand only if required by a locked analysis).
- **Discovery:** registered Full300 (only for compatibility, not fresh generalization); **confirmation:** prospective disjoint panel with exactly documented tokenization and document disjointness.
- **Inference:** FP32, eager attention, `use_cache=False`; fixed identical inputs for teacher and students. GPT-2 absolute-position probes only; no fabricated GPT-NeoX equivalents.

### Interventions (matched within the same conceptual locus)

| Probe | Existing S4 comparator | New controlled version | Key limitation |
|---|---|---|---|
| Q-bias | Zero Q bias in all native layers | Continuous coefficient `b_Q'=(1−α)b_Q` at α=`0,.25,.5,.75,1`; mapped-teacher and native-teacher scopes separately | Bias norms/effects differ; matched α alone is **not** matched effective dose. |
| EPE | Layer-0 legacy-form EPE output transport | Continuous α-scaled transport; same per-example conservation identity, with fixed definition and verified hook output | EPE definition is a *probe*, not an exhaustive EPE circuit. |
| Position-0 anchor | `wpe[0]←wpe[1]` | Same anchor edit as positive control, with effect-on-S and effect-on-output recorded | Broad positional change; not sink-specific. |
| K-input coordinates | Model-local top-3 abs-E0 | Repeat model-local top-3 and five independent seeded matched-cardinality random sets; assess dose by continuous row scaling | Coordinates across widths are **not homologous**. |
| Neutral/negative controls | No-op and random K controls | α=0, reapply original parameter tensors, matched-norm random output perturbations (see E3) | A small effect is not a proof of absent circuit. |

For each intervention log actual **parameter change norm**, pre- and post-intervention sink mass, proportion of sink removed, layer/head sink profiles, ΔCE, self-KL, target-log-prob change, flips, and clean/reference outputs. Correct model-specific masks and effective scope are mandatory.

**Dose comparability rule:** Compare native α curves **and** interpolate only over *overlapping observed changes in sink mass or layer-wise attention outputs*. If no overlap exists, mark `dose_unmatched` rather than extrapolating, silently matching to outcome loss, or treating raw CE effects as homologous. Validate a nominal probe's *actual Q/K/EPE activation change* as an edit-success check.

### Outcome rules

- **Route-specific difference supported** only when: (i) teacher positive control is measurably responsive, (ii) model-local edit is correct and not at numerical noise floor, (iii) student difference persists over comparable scope and overlapping effective dose, (iv) confounding from generic positional/parameter damage and random controls is addressed, and (v) result holds on the frozen holdout panel.
- **No route-difference conclusion** if effects converge after calibration, the teacher route is nonresponsive under matched scope, or intervention dose cannot be meaningfully compared.
- **No universal circuit noninheritance claim under either outcome.**

### Engineering / validation

- Extend `src/sinklab/probes.py` with separately versioned **new** probe definitions; retain the original S4 behavior exactly.
- Unit tests: α=0 exact no-op; α=1 matches registered S4 for compatible scope; tensor restoration identical after exceptions; Q-bias touches Q only; K-coordinate edit touches K input rows only; EPE batch transport agrees with single-item reference and conserves the intended two-position **sum**, not norms.
- Add an independently reviewed interface for choosing teacher mapped-24 scope versus full-36 scope. No teacher-head matching is claimed.
- Verify no architecture- or device-dependent fused backend accidentally bypasses the edit.

---

## 5. E2 — Exact attention-output factorization across training

**Status:** New seed-0 inference under a separately approved mechanism-analysis lock.

### Mechanistic identity

For one head in **evaluation mode with attention dropout disabled**, valid query `i ≥ 1`, and an *isolated deletion at that layer*:

\[
  o_i= a_{i0}v_0 + (1-a_{i0})\bar v_{i,\neg0}, \qquad
  \bar v_{i,\neg0}=\sum_{j>0}\frac{a_{ij}}{1-a_{i0}}v_j .
\]

The repository implements stable conditional-softmax redistribution; compute `\bar v` from masked scores so that no division by `1-a_{i0}` is required when sink mass rounds to one. The **exact local head-output perturbation** is

\[
  \delta o_i=o_i^{\rm del}-o_i
  =a_{i0}(\bar v_{i,\neg0}-v_0).
\]

For the same layer's multi-head output, the immediate post-projection change before dropout is

\[
  \delta h_{\ell,i}
  =W_{O,\ell}\,\mathrm{concat}_h(\delta o_{\ell,h,i}),
\]

with `W_O` understood in the library's actual Conv1D parameter layout. The `W_O` bias cancels in the difference. At a fixed clean forward and an **isolated single-layer edit**, this construction should reproduce the attention-module output difference to FP32 numerical tolerance. **Do not sum these isolated terms and claim they equal an all-layer nonlinear intervention**; intervening at an earlier layer changes later hidden states, attention and values.

### What to measure, per model / step / layer / head

1. Sink mass `a_i0`, conditional mean non-sink `\bar v`, sink value `v_0`.
2. Value contrast: `||\bar v−v_0||_2`; actual local head delta: `||a_i0(\bar v−v_0)||_2`.
3. Post-output projection: `||\delta h_{\ell,i}||_2`, and `||\delta h|| / max(||h_{\ell,i}||, eps)`.
4. Orientation/cancellation across heads (e.g. norm of projected sum vs sum of projected norms), not just factorized scalar averages.
5. **Single-layer** deletion ΔCE, self-KL, flip fraction and logit change, all with matching exact token masks.
6. Primary case: C2 step `500` vs `10000` (add `2000`); supporting cases C6 and C3; low-sink C5/C1 controls.
7. Teacher reference as separate static circuit comparison, not a fictitious teacher training-time trajectory.

### Causal check and controls

- Implement one new transactional **attention-output trace/injection interface** through `src/sinklab/models.py` (do not modify original S4's result schema). Capture the native `A`, `V`, pre-`W_O`, post-`W_O` and pre-residual output **without** detaching live activation when testing a perturbation.
- Apply deletion in **one layer at a time** while upstream layers are clean, then separately inject the algebraically reconstructed `\delta h` at exactly that layer in a clean run. The two routes should agree on logits/behavior within a preregistered FP32 parity tolerance.
- Compare q>=1 support; q=0 cannot redistribute. Separate second-half queries for primary sink statistics and all valid shifted next-token targets for behavior.
- Match panel, no-op, precision, and exact parameter restore. Stream summary scalars: do not dump massive full-vocabulary logits or all head attention matrices to disk.
- Before scientific analysis, empirically calibrate the numeric tolerance on a **synthetic FP32 fixture**, lock it, and fail if no-op or constructed-vs-direct parity breaks.

### Interpretation of possible findings

| Observation | Conclusion allowed | Conclusion not allowed |
|---|---|---|
| Value contrast and projected local delta grow markedly | **The local attention/value/output-path perturbation grows** | The value path is the unique developmental cause of all-layer ΔCE. |
| Local projected delta stays similar while final ΔCE rises | **Later computation or output-loss geometry contributes to increasing sensitivity** | Downstream Jacobian definitely changed without measuring it. |
| Both grow | **Both local perturbation and downstream response are candidate explanations** | A numerically exact percentage mediation without a proper causal design. |
| Local single-layer effects small but all-layer effect large | **Layer interactions or distributed effects are important** | Each layer has no function. |

---

## 6. E3 — Separate local perturbation amplitude, downstream gain and loss geometry

**Status:** New seed-0 inference, contingent on successful E2 trace/injection parity.

### Question

Why is step-10k C2 much more sensitive to sink removal than step-500 C2 when measured `S` barely changes? Is the later network receiving a **larger local edit**, responding **more strongly to a comparable edit**, or merely producing a **larger signed target CE** for comparable output changes?

### E3A — Equal-norm residual injection (within each checkpoint)

1. From clean input at a selected layer, calculate the **sink-directed** actual post-projection deletion perturbation `\delta h_{\ell,i}` (E2).
2. Define `u_{\ell,i}=\delta h / ||\delta h||` for nondegenerate vectors and use **the same prespecified relative residual-norm dose** per checkpoint, not the raw magnitude of `\delta h`.
3. In separate clean forward passes inject `\eta\,||h||\,u` at that attention-output junction, for a frozen grid of `\eta` values calibrated on synthetic/discovery examples (e.g. `0, .01, .03, .1`; final grid requires approval). Ensure residual output scaling and layer scopes are identical *within each model*.
4. Negative controls: deterministic random directions of the **same local norm**, orthogonalized directions when feasible, a dose-zero no-op, and a preselected non-sink-direction perturbation. Keep controls' RNG separate from training seeds.
5. Compare induced **logit displacement, full-vocabulary self-KL, flips, absolute target-log-probability change and ΔCE** at steps 500/2000/10000.
6. Compute an empirical, small-dose **downstream sensitivity curve**. Record the fraction of positions whose natural sink-directed perturbation is too small to normalize; do not filter them silently.

**Interpretation:** Larger late-stage response to an equal-relative-norm sink-directed injection supports changed *local downstream sensitivity*. Different local residual bases and directions across checkpoints mean this is **not** a perfectly shared latent intervention across time; random-direction controls and raw perturbation norms must remain visible. An additional Jacobian-vector product check may estimate `||J_{\ell}u||` on a preselected small item set without forming the full Jacobian.

### E3B — Layer-local causal rescue and multi-layer interactions

1. Run a clean pass, a single-layer deletion pass, and a **clean-attention-output restored** pass for a preselected set of early/middle/late layers, then test clean-logit parity after restoring the edited module output.
2. Record each layer's **single-layer deletion** effect and, separately, one **all-layer deletion + one-layer clean-output rescue** effect. The latter is an interaction-dependent conditional effect, **not** an additive mediated fraction.
3. For a small, prespecified subset, build the exact telescoping sequence: `F_0` clean; `F_k` has layers `0…k−1` deleted; `F_24` all-layer deleted. Then

   `CE(F_24) − CE(F_0) = Σ_{k=0}^{23} [CE(F_{k+1}) − CE(F_k)]`.

   This identity holds for that **chosen layer order**, but the increments may change with order and cannot be interpreted as independent layer contributions. Check one alternate order on discovery items to expose order dependence.
4. Predefine coarse layer groups from existing native-layer scopes, then choose any finer-grained candidate layers using **discovery** only; confirm on the held-out panel without re-selection.
5. Recompute all causal outputs under live edited computation rather than posthoc modifying attention tensors in reports.

### E3C — Exact next-token loss-geometry account

For fixed clean logits `z`, edited logits `z+\delta z`, clean distribution `p`, and target `y`, use the **exact identity**

\[
  \Delta\ell=-\delta z_y+\log\left(\sum_v p(v)e^{\delta z_v}\right).
\]

Save the clean probability assigned to target `y`, clean target NLL, target-logit change, log-normalizer change, output self-KL and edited NLL. This distinguishes:

- bigger **logit / distribution** changes (output sensitivity),
- different **target-aligned CE** consequences at a changing clean model baseline,
- cancellation among positive/negative token-level ΔNLL values.

The identity does **not** prove causal mediation. It prevents incorrectly describing every rise in ΔCE as evidence of greater value-path change.

### Decision table for the paper

| E2/E3 result | Most defensible claim |
|---|---|
| Projected local deletion delta grows; equal-norm downstream response stable | 'Increasing sink dependence is primarily consistent with larger local value/output-path perturbations in tested layers.' |
| Projected delta stable; equal-norm response grows; clean-loss identity checked | 'Downstream computation becomes more sensitive to comparable sink-directed perturbations.' |
| Projected delta and equal-norm sensitivity both grow | 'Both changes contribute; their joint and layer-interaction effects matter.' |
| CE rises without comparable increases in self-KL/logit displacement | 'Increasing signed target loss is substantially shaped by changing prediction geometry.' |
| No coherent accounting, or large order interactions | 'The developmental mediator remains unresolved; structural and functional dissociation is still the primary finding.' |

The intended claim is **local computational explanation with causal perturbation tests**, *not* a formal percentage mediation unless a substantially stronger identification design is separately approved.

---

## 7. E4 — Optional intervention on learning dynamics (new training only)

**Do not start E4 automatically.** It requires a fresh researcher-approved experiment, new locks, device allocation, validated resume implementation under the changed runtime, and explicit compute budget. Reusing the old S1 training lock to continue under modified source is prohibited.

### Why it may be needed

E2/E3 show where and how a learned checkpoint responds. To claim **which adaptation caused later dependence to develop**, intervene on the *training process* while controlling the alternative effects of freezing parameters or changing capacity.

### Proposed continuation from a verified C2 seed-0 step-500 state

Use the **same initial checkpoint, optimizer/RNG state, data order and nominal schedule**, but register **new branch identities** with separate scientific approvals. Test branch equivalence at step 500 before diverging.

| Branch | Controlled manipulation | Interpretation |
|---|---|---|
| B0 | Resume unchanged to step 10k | Same-checkpoint continuation reference; require parity with original training where possible. |
| B1 | Freeze selected attention value-projection (`V`) parameters in prespecified layers | Test whether changes in those parameters are necessary for the late effect under this constrained continuation. |
| B2 | Freeze corresponding output-projection (`W_O`) parameters | Test output-projection adaptation separately. |
| B3 | Freeze a **matched trainable-parameter budget** outside these pathways (e.g. prespecified Q/K control slices) | Control for generic capacity loss due to freezing; matching parameter count alone does not guarantee equal difficulty. |
| B4 (optional) | Freeze selected downstream blocks/layer subsets with matched-scope controls | Test downstream-adaptation hypothesis; expensive and potentially confounded. |

**Causal caveat:** Freezing a module changes optimization, the ability to reduce clean CE, and often the attention distribution. If sink mass, clean CE, gradient norms, learning rate history, or model quality change sharply, a smaller deletion penalty **does not** cleanly prove that the frozen route caused original late reliance. Monitor all these mediating/confounding changes and report them.

### Required endpoints

Steps `500` (branch origin), `1000`, `2000`, `5000`, `7500`, `10000`: attention mass, attention JSD, clean CE, teacher KL, single-layer and all-layer deletion effects, self-KL, flips, and E2 local perturbation scalars. Run the same data schedule and frozen panels. Maintain checkpoint-restorable optimizer state; log all changed trainable parameters and compare with explicit clean-training controls.

**Minimum scientific upgrade:** independent branch replication is desirable but **not currently authorized by D24**; this is a *new training campaign* requiring its own approval, not an exception inferred from old seed-0 checkpoint-access rights. If feasible, approve independent restarts or multiple matched branch points prospectively. Without replication, keep claims model- and seed-specific.

### Falsification criterion

If freezing the hypothesized path does **not** reduce late dependence while matched clean performance/sink structure are preserved, the claim that its adaptation is necessary is weakened. If it does reduce dependence **but also strongly impairs clean modeling**, interpretation remains ambiguous rather than confirmed.

---

## 8. Analysis plan and statistical validity

### Primary estimands

- `ΔCE = CE_edit − CE_clean` (nats per valid shifted next-token target).
- `SelfKL = KL(p_clean || p_edit)` at temperature 1, **full vocabulary**.
- `flip_fraction = P(argmax p_clean ≠ argmax p_edit)` over matched target positions.
- `S = mean attention mass at key 0`, with correct later-query/head/native-layer averaging.
- `ΔS = S_edit − S_clean`, **plus** relative ΔS only for non-negligible baseline S.
- `D_route(t)` = explicitly specified, scale-aware teacher–student response distance across the frozen selective route battery, always **with per-probe component plots**.
- `local_value_delta`, `local_projected_delta`, equal-norm injection response, and any layer-specific conditional rescue from E2/E3.

### Comparisons and time points

1. **Temporal primary:** C2 seed0 at 500 versus 10k (same model lineage; 2000 secondary).
2. **Independent objective control:** C6 high-sink at the same steps; C1/C5 low-sink at the registered control steps.
3. **Mechanistic primary:** teacher vs C2 on **same items**, with route-edit success, layer scope and effective dose reported separately. C3/C6 test robustness across high-sink objectives.
4. **Developmental trend:** compare 500→10k and report the whole `0/100/500/2000/10000` trajectory; never claim that the trend is smooth, monotonic or possesses a sharp onset if the observed values do not show it.
5. **Descriptive statistical uncertainty:** paired item-level differences, optionally a **document-clustered** bootstrap (if document identifiers allow), with 95% intervals conditional on these *fixed* trained models. Seed0 is one training sample, not three because of C2/C3/C6, nor 300 because of documents, nor 5 because of checkpoints.
6. **No automatic equivalence inference:** a small Q-bias ΔCE or a nonsignificant comparison does not establish route absence. An equivalence claim requires a **predeclared practical-effect margin**, derived from measurement floor, editing dose, and the scientific effect of interest, plus a sufficiently tight interval *conditional on these models*.
7. **Prevent double dipping:** any model/layer/dose selected on discovery items must be frozen before holdout evaluation. Exploratory reanalysis of existing S4/S5 is **not preregistered**, even if the new analysis script is locked today.
8. **Multiple outcomes:** nominate self-KL and ΔCE as complementary primary behavioral endpoints; treat flip fraction and other diagnostics as supporting. Do not search for whichever outcome favors the story.

### Provenance and computational cost planning

Report an item-forward equivalent count **before** GPU booking rather than asserting unsupported hours or costs. Inference cost depends on scope, full-vocabulary KL computation, whether attention intermediates are materialized, dtype, checkpoint transfer and GPU. Estimate workload as:

`#models × #steps × #panels × #items × #intervention conditions × #layer scopes`,

then measure time/GPU memory on a fixed synthetic and tiny-real preflight. E0 needs no GPU. E1 is a modest probe-dose sweep; E2/E3 layerwise scans can multiply forward passes by 24 and require explicit profiling. E4 is a separate full continuation-training campaign.

---

## 9. Decision rules: what the paper can claim

| Experimental outcome | Strongest defensible manuscript wording | Wording to avoid |
|---|---|---|
| E1 confirms teacher Q-bias/EPE responsiveness, weak student effects after matched scope/dose | 'Students do not reproduce the measured reliance on these **particular teacher routes**.' | 'The student has an entirely different circuit' / 'no teacher mechanism transfers'. |
| E0 + E1 confirm growing route-response distance on holdout | 'Measured route-response correspondence **diverges** over these checkpoints and probes.' | 'All internal mechanisms become more rebellious'. |
| E0 + E1 show convergence | 'The student becomes more similar on the tested route fingerprints despite retaining other differences.' | Inventing divergence for the narrative. |
| E0 + E1 show mixed/nonmonotonic behavior | 'Different aspects of route correspondence evolve differently.' | A universal increasing/decreasing similarity law. |
| E2 identifies larger projected local delta late; E3 shows stable equal-norm downstream response | 'A larger value/output perturbation provides a **proximate explanation** for late sensitivity in tested layers.' | 'We have proven that value learning uniquely causes the entire developmental change.' |
| E2 projected delta similar, E3 shows greater matched-norm logit/KL effect late | 'Late computation shows greater downstream sensitivity to sink-directed perturbations.' | 'Teacher circuit is inherited' or 'formal mediation established'. |
| E4 targeted freeze suppresses late dependence **without materially changing clean performance and sink structure**, with adequate replication | 'Adaptation of this pathway contributes causally to development of reliance under this controlled training intervention.' | Universal necessity, uniqueness, exact mediation without additional assumptions. |
| Any gate fails | 'Mechanistic interpretation remains open; our robust claim is the structural/dependence dissociation.' | Hiding null, reversed or confounded results. |

**Recommended core scientific conclusion regardless of E1–E4 outcome:**

> Attention distillation can establish teacher-like sink allocation early while a student's sensitivity to disrupting that allocation changes substantially during training. Whether the resulting computation reproduces the teacher's routes, becomes progressively more similar to them, and benefits prediction are distinct, separately testable questions.

---

## 10. Implementation plan mapped to the actual repository

**Existing, inspected components** (do not change their historical contracts):

- [`src/sinklab/models.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/models.py): GPT-2 eager adapter, feature extraction, transactional attention edits; new E2/E3 module-output injection needs a versioned extension.
- [`src/sinklab/interventions.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/interventions.py): stable key-0 delete/redistribute and relocate operators; don't alter original semantics.
- [`src/sinklab/probes.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/probes.py): existing S4 position, Q-bias, EPE and K-coordinate probes; add separate calibrated probe versions, not a silent S4 edit.
- [`src/sinklab/metrics.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/metrics.py), [`src/sinklab/evaluate.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/evaluate.py): use established full-vocabulary metrics/record identity and natural-log units.
- [`src/sinklab/followup_policy.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/src/sinklab/followup_policy.py): enforce D24 before loading model; fixed S4 steps are not a blanket authorization to repurpose S4 run identity.
- [`scripts/run_stage08_s4.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/scripts/run_stage08_s4.py) and [`scripts/report_stage08_s4.py`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/scripts/report_stage08_s4.py): model for source preflight, sealed manifests, per-item coverage, read-only reporting. Reuse patterns with provenance, not source output directories.
- [`design/e6a_v2/S1_CHECKPOINT_FOLLOWUP_POLICY.md`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/design/e6a_v2/S1_CHECKPOINT_FOLLOWUP_POLICY.md), [`design/e6a_v2/MODEL_AND_INTERVENTION_CONTRACTS.md`](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/design/e6a_v2/MODEL_AND_INTERVENTION_CONTRACTS.md): source-of-truth follow-up policy and intervention definitions.

**Proposed new files, only after study approval** (paths are **proposed**, not existing):

```text
design/e6a_v2/studies/MECHANISTIC_FOLLOWUP_PROPOSAL.md
protocols/mechanistic_followup_analysis_approved.json  # after approval only
src/sinklab/mechanism_trace.py                    # E2 exact local accounting
src/sinklab/mechanism_injection.py                # E3 reversible insertion/rescue
scripts/report_mechanistic_e0.py                  # strict read-only audit
scripts/run_mechanistic_e1.py                    # calibrated route dose experiment
scripts/run_mechanistic_e2_e3.py                 # trace, intervention, rescue
scripts/report_mechanistic_results.py            # sealed aggregates and plots
tests/unit/test_mechanism_trace.py
tests/unit/test_mechanism_injection.py
tests/unit/test_calibrated_route_probes.py
tests/integration/test_mechanistic_identity.py
tests/gpu/test_mechanistic_parity_gpu.py
```

### Engineering gates for each prospective change

- [ ] CPU synthetic exact algebra with 2+ heads and masks; include q0, right padding and sink mass near 1.
- [ ] Intervention changes live logits in nondegenerate fixture; editing returned attention alone must not pass.
- [ ] Clean/no-op and dose-zero parity; fixed dtype/mode and `use_cache=False`.
- [ ] Direct deletion equals reconstructed one-layer injection within locked FP32 tolerance.
- [ ] Parameter edits and forward hooks restore **exactly**, including on raised exceptions.
- [ ] Original S4 behavior and tests remain unchanged; old sealed records rehash identically.
- [ ] Fingerprint aggregation is deterministic and rejects missing/failed/incompatible results.
- [ ] All experimental results carry source/analysis/protocol/file hashes and a completion/failure receipt.
- [ ] Run a small GPU preflight only after approval; skip and mark blocked if the required hardware/artifacts are unavailable.

---

## 11. Minimal final scientific outputs

**Figure A — Mechanistic route development:** teacher fixed response line and C2/C3/C6 checkpoint curves for Q-bias, EPE and K-local perturbations, with both **ΔS and self-KL/ΔCE** on separate, labeled axes. Includes C1/C5 controls and mapped-vs-native teacher scopes.

**Figure B — Why sink deletion becomes consequential:** C2 at 500/2000/10k showing sink mass, `||\bar v−v_0||`, projected attention-output difference, single-layer deletion effect, equal-norm downstream response and full-model deletion ΔCE. Show which component actually changes.

**Figure C — Causal pathway localization:** selected layerwise direct deletion, clean-output restoration and all-layer conditional rescue, with a clear warning that effects are nonlinear/order-dependent and not additive mediation fractions.

**Table A — Claim–evidence–failure matrix:** every paper assertion paired with source hashes, condition/seed/step/panel, effect units, controls, measured uncertainty and limitations. Report null and contrary results.

**Machine-readable package:** locked protocol, generated scripts/tests, audited input manifest, per-item paired records (or manifest references), `summary.json`, `results.csv`, plot data, test outputs, `COMPLETE` or `BLOCKED` receipt. External model weights remain outside Git.

---

## 12. Researcher decision gates and stopping rules

1. **Run E0 first.** If route fingerprints converge, do **not** claim growing divergence; choose the stronger delayed-dependence story.
2. Approve **E1–E3** only after E0 source hash audit, compute profiling and a lock of intervention recipes/panels/statistics. Prioritize C2 (primary), C6 (high-sink control) and C5 (low-sink control). Add C3/C1 if budget permits.
3. **Stop** if live intervention/restore/FP32 parity fails, if model identity is not verified, or if relevant effective-dose ranges do not overlap. Document `BLOCKED/INCONCLUSIVE`; don't reinterpret failures as null mechanism evidence.
4. If E2/E3 explain the effect locally but leave developmental necessity unresolved, publish a **proximate pathway explanation**, not a claimed training-time causal mediator.
5. Seek E4 approval only if a precise, falsifiable mediator hypothesis remains and the anticipated gain justifies retraining. A new branch is **not** an S1 replicate unless independently approved and correctly identified.
6. For a cross-seed or cross-architecture claim, initiate a wholly separate prospective replication protocol; current D24 does not license new seed-1/2 checkpoint inference.

### Proposed top-tier paper claim ladder

- **Level 1 (already supported):** Structural sink acquisition precedes strong deletion sensitivity in the observed training trajectories; high sink reliance is not synonymous with superior clean predictive utility.
- **Level 2 (E0/E1 success):** Specific teacher route responses are not fully reproduced by selected students under controlled causal probes; any convergence/divergence is explicitly **route- and checkpoint-conditional**.
- **Level 3 (E2/E3 success):** Growing dependence is traced to measured changes in local attention-value/output perturbation and/or downstream response, supported by exact local intervention identities and live causal injections.
- **Level 4 (E4 and prospective replication success):** A specified pathway's *adaptation during training* causally contributes to the later dependence under an approved controlled continuation design.

**A top-tier submission should target the highest level actually supported by completed controls—not the strongest preselected story.**

---

## Source documents inspected (repository links, version pinned)

- [S1 three-seed results](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/reports/results/S1_C0_C6_RESULTS_20261007.md)
- [S4 mechanistic probes: complete 35-state seed-0 series](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/reports/results/S4_RESULTS_20261007.md)
- [S5 sink-supervision trajectories](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/reports/results/S5_RESULTS_20261007.md)
- [S7 distillation utility and exact attention JSD decomposition](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/reports/results/S7_RESULTS_20261007.md)
- [S4 study contract](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/design/e6a_v2/studies/S4_MECHANISTIC_INHERITANCE.md)
- [Metric definitions and inference/replication limits](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/design/e6a_v2/METRICS_AND_ANALYSIS.md)
- [D24 checkpoint-follow-up limitations](https://github.com/SyedNaveedMahmood/kd-sink/blob/dd3fcc33132c8dc1c4d9f18466654b1dc5f4703c/design/e6a_v2/S1_CHECKPOINT_FOLLOWUP_POLICY.md)

**Caution about source accessibility:** The GitHub repository contains code, locked designs, reports, and artifact references, but large scientific weights and sealed item-level result trees are stored in externally referenced run directories. Their *presence and usability on a future experiment machine* must be verified; no new experiments were executed to prepare this plan.
