# Decision register and freeze policy

Status: partial researcher amendment approved 2026-09-28; the full protocol remains draft and production is blocked. The recorded approval source and digest are in `protocols/s1_researcher_amendment_20260928.json`. A citation can justify a proposal; it cannot establish optimal hyperparameters, available memory, or adequate power. Before production, store all approvals and measured values in a sealed protocol lock. Unresolved choices must not be filled silently.

## Fixed requirements and observed facts
- U01: S1 uses frozen GPT-2-large -> randomly initialized GPT-2-medium, 10,000 optimizer updates, C0-C6. Seed 0 is the mandatory single-seed campaign; seeds 1 and 2 are optional, each explicitly launched per condition. Single-seed results cannot support across-seed variance or reproducibility claims.
- U02: evaluate at step0 and every100 updates; final full resume checkpoint; independent objectives; effective64 sequences x128 input tokens; common paired microbatch; no mid-run retuning.
- U03: RTX3090 is eligible for all S1 C0-C6; RTX4080 SUPER is eligible only for conditions with measured production-shape full-cycle/headroom evidence and an approved fixed-role plan. C0/C1/C2/C5/C6 have real 4080 profile evidence; C3 is pending_4080_profile; C4/REL is3090-only. A final hardware lock must encode condition/device eligibility and one common paired schedule. New code has no runtime/test dependency on Upstream.
- F01: audited repository contains uppercase `Upstream/`. Large/medium is36/24 layers,20/16 heads,1280/1024 width, not the old16/12-head pair.
- F02: old large/medium configuration used mean-head JSD; the paper-facing medium/small path used cosine-soft AMAD-style JSD. They are not the same alignment.
- F03: inspected `amad_js_attn_loss` has differentiable cosine-soft weights but no separately parameterized alignment module. Published Jin A2D uses a different trainable construction. TinyBERT originally matches pre-softmax scores. The requested probability-MSE and causal multilayer MiniLMv2-style recipes are explicit adaptations.

## Proposed decisions requiring approval
| ID | Recommendation | Reason, alternative, or limitation |
|---|---|---|
| D01 (amended) | S1 mandatory C0-C6; S2/S4/S5/S6 remain separately scoped, S3 optional | The mandatory campaign consists of seven independent S1 seed0 commands, not an automatic study sweep |
| D02 (amended) | Mandatory S1 seed0 C0-C6; optional seeds1 and2 as separately invoked complete campaigns; calibration seed1729 excluded | Single-seed primary reporting is descriptive only; no across-seed variance or reproducibility claim |
| D03 | C2=`cosine_soft_jsd_v2`, teacher-to-soft-student, temperature1 | Closest inspected paper-facing implementation; do not call it exact Jin A2D |
| D04 | C3=head-mean post-softmax probability MSE; C4=causal QQ/KK/VV,64 relation heads, all mapped layers | Both are named adaptations, not exact published replications |
| D05 | B=0.5CE+0.5KD shared by C1-C6; auxiliary coefficient1/9 | Holds behavioral objective weights fixed; preserves legacy .45/.45/.10 ratios after division by.9, but not absolute gradients/clipping |
| D06 | Training-only initial-gradient calibration for C3/C4 relative to C2; C2/C5/C6 scale1 | Raw MSE/JSD/REL coefficients are not commensurate; this is a proposed control, not a proven optimum |
| D07 (amended) | Dense64 first validation blocks; full300 first validation blocks; NLL-only2000 following validation blocks | Keeps the historical sink region distinct from the following PPL region; dense64 nests in full300 |
| D08 | Primary dense delete/relocate over all24 student layers and mapped24 teacher layers | Avoids a mid-band-only null; teacher all36 and depth scopes are secondary |
| D09 (amended) | Retain weights0,100,250,500,1000,2000,5000,7500,10000; rolling full saves every500 absolute updates; protect full10k and each approved extension endpoint | Step250 adds a full evaluation; it does not replace200/300 dense observations. A protected10k full state survives later failures/extensions |
| D10 (amended) | AdamW5e-4, betas.9/.95, eps1e-8, decay.1, clip1; warmup500; cosine to10% at10k, constant floor thereafter without schedule stretch | Starting recipe inherited from upstream; extension must load the protected10k full state with explicit lineage and preserve optimizer/RNG/order counters |
| D11 | Student dropout.1; teacher eval; auxiliary targets before attention dropout | Probability divergences require normalized targets; this deliberately amends legacy training-time attention-return semantics |
| D12 | Standard Pythia160M/410M, training seeds1234,1,2, all verified native steps | Extra training-seed checkpoints exist; availability/revisions still need inventory; this is an external observational baseline |
| D13 (corrected) | RTX3090 eligible C0-C6; RTX4080 SUPER may train a condition after its real production-shape profile/headroom gate and approved fixed-role plan. C0/C1/C2/C5/C6 have measured 4080 candidate evidence, C3 remains pending_4080_profile, and C4/REL is3090-only | Seven unique seed0 condition jobs remain mandatory, with optional complete seed1/2 campaigns. Prefer same-device primary contrasts; a cross-device contrast needs an explicit bridge/replica or a recorded hardware confound. No migration or retuning during a run |
| D14 | Optional S3 GPT-2-small -> random DistilGPT-2 config, C0-C4, fixed-index JSD/MSE |15 extra runs; equal heads allow an index convention, not proof of head homology |
| D15 | S4 model-local top3 EPE coordinates and specified legacy-form EPE transport at fixed checkpoints | New teacher must be characterized; do not reuse medium coordinate IDs in large/student widths |
| D16 | Paired-seed descriptive estimates; no automatic binary onset or equivalence claim | Practical no-function/onset thresholds are not established by the sources |
| D17 | S6 SST2-validation sentences/GSM8K-test questions/HumanEval prompts,100 each,max128; same-doc40-token sensitivity; optional OWT512/1024 | Text probes are not task accuracy; long context is a separate cost and distribution-shift analysis |
| D18 (amended) | Pinned OpenWebText single train stream: training source indices [0,400000), validation [400000,408000), disjoint training-only calibration [408000,416000); audited whitespace collapse, seeded document shuffle, GPT-2 tokens and post-document EOS, greedy128 packing; first300/following2000 validation regions | Restores historical corpus behavior; manifest hashes and source ownership make it auditable. Calibration window is new, not historical |

## Values that must be measured, not invented
- M01: tested dependency versions, model/tokenizer/dataset revisions and licenses, actual artifact checksums and download availability.
- M02: complete profile evidence for every condition/device selected by the reviewed plan, one common microbatch/accumulation, activation-checkpoint policy, measured VRAM/RAM/disk/time. RTX3090 C0/C5/C6 and RTX4080 C3 remain unmeasured; the latter is not currently eligible for a reviewed 4080 job. Existing 4080 C0/C1/C2/C5/C6 and3090 C1-C4 evidence remain valid engineering measurements but are not by themselves a final hardware lock.
- M03: calibration raw losses/gradient norms and frozen MSE/REL scales; reject degenerate gradients rather than inventing a scale.
- M04: new-teacher sink/probe preflight, numerical BF16/FP32 diagnostics, all mandatory CPU/GPU smoke and resume evidence.
- M05: seven mandatory S1 seed0 jobs plus optional seven-job seed1/2 campaigns, optional S3/long-context scope, external checkpoint/cache budget; practical margins only if equivalence/onset claims are desired.

## Gates and deviations
The 2026-09-28 researcher amendment resolves D01, D02, D07, D09, D10, D13 and D18 only for the stated S1 scope; other proposed choices and measured gates still require approval/verification. Production requires a complete approved protocol and valid artifact, environment, hardware and calibration locks. Unset practical margins disable binary onset/equivalence analyses; descriptive trajectories can proceed only after production gates. S3 remains disabled. Missing public checkpoints remain missing, not substituted by latest weights.

Pilots test correctness, OOM, speed, finite gradients and whether the teacher instrument is informative on a separate calibration panel. Do NOT require a student to inherit a sink or preserve the old paper's dissociation to pass. Do not tune on final evaluation panels. A bug fix affecting scientific outputs requires a versioned amendment, list of affected runs and rerun of the complete affected paired comparison. Negative or contrary outcomes are valid results.
