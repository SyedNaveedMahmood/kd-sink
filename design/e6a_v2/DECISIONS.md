# Decision register and freeze policy

Status: proposed amendment, not approved preregistration. A citation can justify a proposal; it cannot establish optimal hyperparameters, available memory, or adequate power for this experiment. The researcher may approve this register as a whole or amend individual IDs. Before production, store approval, values and their canonical SHA-256 in a protocol lock. Implementation can proceed on explicit proposals; unresolved scientific choices must not be filled silently.

## Fixed requirements and observed facts
- U01: S1 uses frozen GPT-2-large -> randomly initialized GPT-2-medium, 10,000 optimizer updates, C0-C6, scientifically multi-seed but operationally one explicitly supplied seed and condition per invocation.
- U02: evaluate at step0 and every100 updates; final full resume checkpoint; independent objectives; effective64 sequences x128 input tokens; common paired microbatch; no mid-run retuning.
- U03: production-size REL, including profiling, runs only on3090. Other jobs use approved4080/3090 allocation. New code must have no runtime/test dependency on Upstream.
- F01: audited repository contains uppercase `Upstream/`. Large/medium is36/24 layers,20/16 heads,1280/1024 width, not the old16/12-head pair.
- F02: old large/medium configuration used mean-head JSD; the paper-facing medium/small path used cosine-soft AMAD-style JSD. They are not the same alignment.
- F03: inspected `amad_js_attn_loss` has differentiable cosine-soft weights but no separately parameterized alignment module. Published Jin A2D uses a different trainable construction. TinyBERT originally matches pre-softmax scores. The requested probability-MSE and causal multilayer MiniLMv2-style recipes are explicit adaptations.

## Proposed decisions requiring approval
| ID | Recommendation | Reason, alternative, or limitation |
|---|---|---|
| D01 | Keep S1-S6; S3 has a separate optional budget | S1 already compares three auxiliary families; equal-head replication is additional evidence |
| D02 | S1 training seeds0,1,2; calibration seed1729 excluded | Three seeds are a minimum plan, not a power guarantee; each run is explicitly invoked |
| D03 | C2=`cosine_soft_jsd_v2`, teacher-to-soft-student, temperature1 | Closest inspected paper-facing implementation; do not call it exact Jin A2D |
| D04 | C3=head-mean post-softmax probability MSE; C4=causal QQ/KK/VV,64 relation heads, all mapped layers | Both are named adaptations, not exact published replications |
| D05 | B=0.5CE+0.5KD shared by C1-C6; auxiliary coefficient1/9 | Holds behavioral objective weights fixed; preserves legacy .45/.45/.10 ratios after division by.9, but not absolute gradients/clipping |
| D06 | Training-only initial-gradient calibration for C3/C4 relative to C2; C2/C5/C6 scale1 | Raw MSE/JSD/REL coefficients are not commensurate; this is a proposed control, not a proven optimum |
| D07 | Dense64 OWT; full300; endpoint NLL-only2000 | Separates affordable trajectories from more precise retained-state evaluation; cost must be measured |
| D08 | Primary dense delete/relocate over all24 student layers and mapped24 teacher layers | Avoids a mid-band-only null; teacher all36 and depth scopes are secondary |
| D09 | Retain weights0,100,250,500,1000,2000,5000,7500,10000; rolling full saves every500 | Step250 adds a full evaluation; it does not replace200/300 dense observations |
| D10 | AdamW5e-4, betas.9/.95, eps1e-8, decay.1, clip1; warmup500; cosine to10% at10k, constant floor thereafter | Starting recipe inherited from upstream with explicit future extension semantics |
| D11 | Student dropout.1; teacher eval; auxiliary targets before attention dropout | Probability divergences require normalized targets; this deliberately amends legacy training-time attention-return semantics |
| D12 | Standard Pythia160M/410M, training seeds1234,1,2, all verified native steps | Extra training-seed checkpoints exist; availability/revisions still need inventory; this is an external observational baseline |
| D13 |3090:C1-C4;4080:C0,C1,C2,C5,C6; each3 seeds |27 physical runs provide same-device primary contrasts; alternative all21 on3090; onlyREL-on3090 split remains confounded |
| D14 | Optional S3 GPT-2-small -> random DistilGPT-2 config, C0-C4, fixed-index JSD/MSE |15 extra runs; equal heads allow an index convention, not proof of head homology |
| D15 | S4 model-local top3 EPE coordinates and specified legacy-form EPE transport at fixed checkpoints | New teacher must be characterized; do not reuse medium coordinate IDs in large/student widths |
| D16 | Paired-seed descriptive estimates; no automatic binary onset or equivalence claim | Practical no-function/onset thresholds are not established by the sources |
| D17 | S6 SST2-validation sentences/GSM8K-test questions/HumanEval prompts,100 each,max128; same-doc40-token sensitivity; optional OWT512/1024 | Text probes are not task accuracy; long context is a separate cost and distribution-shift analysis |
| D18 | Explicit normalization/hash partition and nested panel ranking in DATA_AND_PROVENANCE | Makes selection reproducible rather than leaving numeric rules to a coding agent |

## Values that must be measured, not invented
- M01: tested dependency versions, model/tokenizer/dataset revisions and licenses, actual artifact checksums and download availability.
- M02: safe device/condition batch maxima, common microbatch/accumulation, activation-checkpoint policy, measured allocated/reserved VRAM, RAM, disk and time.
- M03: calibration raw losses/gradient norms and frozen MSE/REL scales; reject degenerate gradients rather than inventing a scale.
- M04: new-teacher sink/probe preflight, numerical BF16/FP32 diagnostics, all mandatory CPU/GPU smoke and resume evidence.
- M05: approved27-versus21-run budget, optional S3/long-context scope, external checkpoint/cache budget; practical margins only if equivalence/onset claims are desired.

## Gates and deviations
Production requires approval/amendment of D01-D18, relevant M01-M05 evidence and valid locks. Unset practical margins disable binary onset/equivalence analyses; descriptive trajectories can proceed. S3 remains disabled until approved. Missing public checkpoints remain missing, not substituted by latest weights.

Pilots test correctness, OOM, speed, finite gradients and whether the teacher instrument is informative on a separate calibration panel. Do NOT require a student to inherit a sink or preserve the old paper's dissociation to pass. Do not tune on final evaluation panels. A bug fix affecting scientific outputs requires a versioned amendment, list of affected runs and rerun of the complete affected paired comparison. Negative or contrary outcomes are valid results.
