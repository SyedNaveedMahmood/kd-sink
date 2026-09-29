# Sequential implementation worklist

Stage 06 production launch readiness is restored under the new runtime-bound root; no S1 scientific campaign has run. Read AGENTS.md first. Work in order from the first incomplete task. A checkbox becomes checked ONLY when its implementation AND required tests actually pass with report/journal evidence. Missing hardware/network/approval is BLOCKED,not passed. Stage capability,validation and scientific coverage are separate. Optional studies remain disabled until approval.

## Stage00 - Foundation
Work order: [00_FOUNDATION](design/e6a_v2/stages/00_FOUNDATION.md).
- [x]00.1 Inspect tree/licenses and document reuse; leave Upstream unchanged.
- [x]00.2 Minimal package,strict configuration and explicit single-seed CLI.
- [x]00.3 Draft/approval locks,canonical hashing and templates.
- [x]00.4 Resolve/test dependency versions and lock environment.
- [x]00.5 T00/offline no-Upstream build tests,report and journals.
- [x] Stage00 evidence reviewed and milestone committed.

## Stage01 - Data and provenance
Work order: [01_DATA_PROVENANCE](design/e6a_v2/stages/01_DATA_PROVENANCE.md).
- [x]01.1 Deterministic document split/dedup/packing/manifests.
- [x]01.2 Shared CPU FP32 initialization artifacts per seed.
- [x]01.3 Effective-update order,cursors,prefixes and extension.
- [x]01.4 Frozen evaluation/calibration panels with synthetic tests.
- [x]01.5 T01/regression,usage and artifact evidence.
- [x] Stage01 evidence reviewed and milestone committed.

## Stage02 - Models and interventions
Work order: [02_MODEL_INTERVENTIONS](design/e6a_v2/stages/02_MODEL_INTERVENTIONS.md).
- [x]02.1 Native-faithful GPT-2/GPT-NeoX adapters.
- [x]02.2 Pre-dropout probabilities and Q/K/V,pure recomputation paths.
- [x]02.3 Delete/relocate/no-op before value aggregation.
- [x]02.4 Independent references and restoration checks.
- [x]02.5 T02/T07/regression and measured tolerances.
- [x] Stage02 evidence reviewed and milestone committed.

## Stage03 - Objectives
Work order: [03_OBJECTIVES](design/e6a_v2/stages/03_OBJECTIVES.md).
- [x]03.1 CE/KD masks,temperature and shared base.
- [x]03.2 Cosine-soft JSD and head-mean probability MSE.
- [x]03.3 Causal QQ/KK/VV and exact chunked gradients.
- [x]03.4 NoSink/SinkOnly invariance and gradient tests.
- [x]03.5 Separate training-only gradient calibration.
- [x]03.6 S3 index variants,T03/T04/regression.
- [x] Stage03 CPU evidence reviewed and milestone committed; real GPU validation remains Stage06.

## Stage04 - Trainer and resume
Work order: [04_TRAINER_RESUME](design/e6a_v2/stages/04_TRAINER_RESUME.md).
- [x]04.1 Single-run trainer,accumulation,precision and schedule.
- [x]04.2 Isolated profiles/common-batch solver and mock tests.
- [x]04.3 Sparse/rolling/protected full checkpoints,atomic saves.
- [x]04.4 Exact resume/extension/interruption and compatibility.
- [x]04.5 tqdm/elapsed/ETA/GPU/memory/non-TTY logs.
- [x]04.6 T04/T05/T06/T10 CPU evidence; realGPU pending.
- [x] Stage04 CPU evidence reviewed and milestone committed; realGPU validation remains Stage06.

## Stage05 - Evaluation
Work order: [05_EVALUATION](design/e6a_v2/stages/05_EVALUATION.md).
- [x]05.1 Exact behavioral/causal full-vocabulary metrics.
- [x]05.2 Structure/topology/fingerprint primitives.
- [x]05.3 Dense/full cadence and RNG-neutral evaluator.
- [x]05.4 Versioned per-item records,caches and resume.
- [x]05.5 T07/T08/T10/cancellation/training-parity CPU evidence.
- [x] Stage05 CPU evidence reviewed and milestone committed; realGPU validation remains Stage06.

## Stage06 - S1/S3 and hardware
Work order: [06_S1_S3_INTEGRATION](design/e6a_v2/stages/06_S1_S3_INTEGRATION.md).
- [x]06.1 Independent S1/S3 configs and variant checks.
- [x]06.2 Explicit reviewed seed-0 nine-physical-job plan for seven unique conditions, with C1/C2 same-seed hardware bridges; no default sweeps.
- [x]06.3 Actual3090/4080 model/teacher/intervention/objective smoke.
- [x]06.4 Measured profiles,common microbatch,calibration/resources.
- [x]06.5 ActualGPU resume/precision tests and readiness locks; runtime-source binding repaired and resealed.
- [x] Stage06 required evidence reviewed and milestone committed; old production root is superseded and non-launchable.

Stage06 predecessor status: **production_ready at its original runtime source**, with scientific coverage **none**; the D19 successor is pending. Researcher choices D01-D18 and the corrected amendment remain sealed; on 2026-09-28 the researcher additionally approved `fingerprint_denominator_floor=1e-8` before training. This is solely a numerical guard for reported S_I/S, not an effect-size threshold: if `abs(baseline_sink)<1e-8`, the ratio is unavailable while baseline/probed sinks and their absolute difference are retained. `configs/production/s1_jobs_seed0.json` fixes nine physical seed-0 jobs for seven unique C0-C6 condition/seed pairs: RTX4080 SUPER C0/C1/C2/C5/C6 and RTX3090 C1/C2/C3/C4. C1/C2 are same-seed hardware bridges. The measured common microbatch is 4 Ãƒâ€” accumulation16 (64 sequences, 8,192 inputs and 8,128 shifted targets per update). RTX3090 C0/C5/C6 and RTX4080 C3 remain unmeasured/unused; C4/REL is3090-only. The verified external artifact tree, exact two-role environment, seed1729 16-batch calibration (`s_MSE=68.00580071126464`, `s_REL=0.120179255876581`), unchanged four component locks and nine root-bound configs all pass transitive validation. The predecessor production root is `910961fcc53edaed0df48e3139dbb7ca2e058bf7480b67dab27bae0bcd90f26f`, bound to runtime source `00b70dbd727e090c2f79ff48236b3b4ad9cfd136`. No 10k production run or Stage09 work has started. See `reports/stage06.json`.

The old root `2a11da9bb71957a4d6b3a2f93a34bd67491dd9d21d70577a7a10858930a8943e` is `superseded_nonlaunchable_due_to_runtime_source_provenance_defect`. The four measured component locks and 3090 calibration remain valid evidence. The distinct production runtime source milestone, newly sealed root and nine configs passed validation. A 4080 preflight confirmed refusal before model load when runtime provenance fails; no training started.

**D19 hardware-class amendment sealed (2026-09-28):** The researcher prospectively authorizes exact-model NVIDIA GeForce RTX 4080 SUPER peers using NodiPC's measured C0/C1/C2/C5/C6 reference evidence. The source milestone `6d5f43aee093008a86bf203e2443884a0069763c` was pushed; the successor production root is `48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791`. NodiPC, NaveedPC, Adrita-PC and future exact-model peers may start fresh approved jobs after locked software, scientific artifact and source validation. Each run records its actual UUID and may resume only on that same UUID. NaveedPC's C2/C5 former per-card headroom failures remain recorded; permission is the explicit D19 reference transfer, not a new per-card pass. Common microbatch 4 x accumulation 16 remains fixed; production OOM aborts without numerical retuning or migration. C3 and C4 remain RTX3090-only in the nine-job plan. The predecessor root above remains historical for work started under it, including researcher-reported NodiPC C0; local C0 completion evidence was not assessed. Real NaveedPC successor-root C1/C2/C5 preflight passed, C3/C4 refused, and no scientific training was started by this amendment. See `reports/stage06.json`.

**S1 C1 production launch (2026-09-28, observed 15:57 UTC):** On explicit operator instruction, one seed-0 C1 job began on NaveedPC under the D19 successor root. Its external run directory is `D:\KD-SINK-stage06-runs\s1-c1-seed0-rtx4080super`. Step-0 and step-100 evaluations completed; training resumed and reached observed update 116 with finite loss. The run remains in progress, with no completion claim or other condition launched. The early wall-clock ETC was about 5-6 hours remaining at observation; see `reports/stage06.json` and the external structured logs. Future audit must verify the protected final checkpoint and full evaluation coverage before marking C1 complete.

**S1 C5 conditional queue (2026-09-28, observed 18:54 UTC):** The operator authorized one seed-0 C5 job on this NaveedPC only after C1 finishes successfully. An external single-use watcher is queued at `D:\KD-SINK-stage06-runs\launcher\queue_c5_after_c1.py`, with status in `queue-c5-after-c1.status.json`. It requires C1's normal 10,000-update result, complete finite sequential update log, checksum-verified final checkpoint with the actual NaveedPC UUID, and all registered evaluations before launching C5. Failure cancels the queue. C1 was at update 5,000; C5 had not started. No other condition is queued.

**D20 C3 RTX4080 SUPER prospective successor (2026-09-29):** NodiPC's real C3 mb8 cycle completed but failed headroom; the separate mb4 ? accumulation16 cycle and checkpoint load passed. Sealed evidence `9b608a1a8da5af520435cd34dd45548ea29f3d85cf67977d1245276979bcb703` and researcher decision `15f9cab8d2d691f14aca3b249b0a5d1458626c59c38ff47ce338f68ad9a532f5` extend the exact-model 4080 class to C3 while keeping C4 3090-only. Runtime source `56260519be5e31790a7ed7abc078e2a285c0cf18`, hardware lock `fc410cd3d01505d0cd7b9ffdda98caaf070bfd2524ca7ac7841a0c78e60aa593`, and successor root `fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552` are prospective. The nine-job plan moves only the unstarted C3 assignment to the 4080 class and records `C4-C3-seed0` as a hardware confound. D19 remains valid for already-started C1 and the preapproved C5-after-C1 queue on NaveedPC; those jobs do not need to pull D20 mid-run. No C3 completion or other scientific outcome is claimed.

**D20 first root superseded:** `fd5199ae6f90985ce8c87539f3e76b6d810c5420ccb2d0f6d6ee300a27cef5f4` is archived and nonlaunchable because its preflight used the new plan for the original artifact approval. The corrected inventory reproduces the original sealed digest; replacement root `fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552` passed real NodiPC C3 preflight with artifacts/source/environment verified and `training_started=false`; C4 was refused. The single C3 launch is now authorized.

## Stage07 - S2
Work order: [07_S2_PRETRAINING](design/e6a_v2/stages/07_S2_PRETRAINING.md).
- [x]07.1 Immutable public checkpoint/seed inventory capability; one public step0 entry is now resolved and verified.
- [x]07.2 Single Pythia-checkpoint evaluator/manifests on local CPU fixtures.
- [x]07.3 Opt-in trajectories,bounded cache/download,resume on local CPU fixtures.
- [x]07.4 Local tests and approved real network/GPU evidence.
- [x] Stage07 capability evidence reviewed and milestone committed; S2 scientific coverage remains absent.

Stage07 status: external capability acceptance COMPLETE using one immutable public Pythia-160m step0 checkpoint on the RTX 4080 SUPER. Its hashes, tokenizer, native rotary/masks, evaluator, restoration, resume and storage were measured. Full checkpoint coverage and the full-trajectory storage/download budget still require approval and measurement; no S2 scientific trajectory or coverage exists. See `reports/stage07.json`. Stage06 is now production-ready, without scientific coverage.

## Stage08 - S4/S5/S6
Work order: [08_S4_S5_S6](design/e6a_v2/stages/08_S4_S5_S6.md).
- [x]08.1 Anchor/route battery and model-local coordinate controls (CPU capability; real teacher calibration pending).
- [x]08.2 S5 reanalysis-only joins and contrasts (CPU capability; no production S1 records yet).
- [x]08.3 Domain/length panels and optional long contexts (CPU capability; optional long contexts disabled without approval/memory validation).
- [x]08.4 Study tests,new-teacher checks and approvedGPU smoke (RTX 4080 SUPER engineering qualification; S5 analysis has no GPU kernel).
- [x] Stage08 capability evidence reviewed and milestone committed; scientific coverage remains absent.

Stage08 status: capability and external qualification COMPLETE on the RTX 4080 SUPER using a pinned GPT-2-large teacher and a frozen engineering calibration panel. All ten fixed S4 probes responded at the predeclared numerical reporting floor; that observation is not a scientific transfer claim. Paired 40/128-token S6 GPU smoke passed on agent-authored engineering inputs. S5 remains read-only with no production C1/C2/C5/C6 records; none were fabricated. Optional 512/1024 contexts remain disabled. No S4/S5/S6 scientific coverage exists. See `reports/stage08.json`. Stage06 is now production-ready; Stage07 external capability acceptance is complete without scientific coverage. Stage09 has not started.

## Stage09 - Analysis and release
Work order: [09_ANALYSIS_RELEASE](design/e6a_v2/stages/09_ANALYSIS_RELEASE.md).
- [ ]09.1 Prespecified paired analyses and completeness checks.
- [ ]09.2 Traceable figure/table generation with raw seeds.
- [ ]09.3 Complete tests and offline clean-wheel/no-Upstream gate.
- [ ]09.4 Reports,manifests,remaining decisions,production commands.
- [ ]09.5 Release milestone and truthful handoff; no upstream deletion.
- [ ] Stage09 evidence reviewed and milestone committed.

## Separate research execution opt-in
- [ ] Researcher approves/amends D01-D18 and relevant M01-M05.
- [x] Protocol/environment/artifact/hardware/calibration locks frozen for S1 seed0 production readiness.
- [ ] Operator explicitly launches each approved study/condition/seed.
- [ ] Actual completed coverage audited before paper claims.
- [ ] Separate authorization obtained before eventual Upstream deletion.
