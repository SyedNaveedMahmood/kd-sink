# Sequential implementation worklist

D24 prospective S1 checkpoint-follow-up policy is sealed and CPU-validated: all new checkpoint-dependent work is seed0-only. D25 prospectively freezes S6 external data inputs and records upstream license uncertainty without inferring a license. S4/S6 remain designated-seed0 follow-ups. D26 seals the exact compatible mixed-root C1/C2/C5/C6 seed0 S5 reuse-only join after protocol/environment review. The user explicitly authorized the independent S4/S5/S6 scientific work in the latest Stage08 execution request; no training, S2 rerun, or Stage09 is authorized. Current campaign status is recorded in `reports/stage08_scientific_execution_20261005.json`: S5 is complete and independently hash-verified; S4 and S6 are pending their separate GPU runs. A checkbox becomes checked ONLY after its implementation and required tests/output audits actually pass with report/journal evidence. Missing hardware/network/approval is BLOCKED, not passed. Stage capability, validation, and scientific coverage are separate.

## Prospective D24 - S1 checkpoint follow-ups (2026-10-04)

- [x] Seal the researcher rule in `protocols/s1_researcher_amendment_d24_seed0_followups_20261004.json`; preserve every historical training root/config and the sealed original decision register.
- [x] Restrict S4 to C0-C6/seed0 at0,100,500,2000,10000 (35 logical states); S6 to C0-C6/seed0 at0,500,2000,10000 (28). No seed1/2 checkpoint readiness requirement.
- [x] Guard S4/S6 and new shared-evaluator S1 checkpoint work before model access. Keep original training identity plus the separate D24 policy hash; preserve registered in-training record keys and S2.
- [x] Preserve S5 reuse of recorded seed1/2 results; restrict any new S5 model inference to seed0. Label S4/S6 as designated-seed0 follow-ups, without across-training-seed reproducibility claims.
- [x] CPU synthetic/full regression: 310 passed, 0 failed, 0 skipped; one external astor deprecation warning. Offline wheel validation included. Historical preservation/seal checks passed. See `reports/d24_seed0_followups_validation.json` and appended CORE/S1/S4/S5/S6 journals.

D24 follow-up policy SHA: `46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6`. This is not a replacement for any S1 training root. The changed execution-critical source tree is not approved for training/resume under the old runtime lock; that lock remains unchanged and must continue to fail closed. The 2026-10-04 Adrita-PC preparation audit verified all 35 S4 and 28 S6 seed0 checkpoint states and the pinned local S1 OWT inputs. S4 is ready for separately authorized execution. S5 is blocked because the four required records retain three protocol roots and the current join enforces a single root. S6 was blocked at that audit because approved SST-2/GSM8K/HumanEval revisions, license statuses and local inputs were not yet frozen; D25 and the current S6 preparation report supersede that readiness status. The C3 seed0 transferred-log gap is accepted only under `protocols/s1_c3_seed0_transferred_log_exception.json`; no prefix was reconstructed. See `reports/stage08_adrita_preparation.json` and the appended CORE/S4/S5/S6 entries. No training, scientific evaluation, GPU code, profiling/calibration, downloads or Stage09 work was launched. Next: resolve S5 root compatibility before S5 readiness; separately authorize any S4/S5/S6 scientific execution. D25 has frozen the S6 sources and panels, and optional long contexts remain disabled.

## Prospective D25 - S6 external datasets and license status (2026-10-04)
- [x] Seal a narrow prospective amendment and source lock; D24, historical S1 roots/results, and S2 remain unchanged.
- [x] Pin exact SST-2, GSM8K, and HumanEval source commits/files, config/split/fields, source/card hashes, tokenizer identity/revision/hash, and upstream license evidence. SST-2 is `upstream_ambiguous` (`other` in the pinned GLUE card); no license was inferred. GSM8K and HumanEval record upstream-stated MIT metadata. This is not a legal determination or redistribution grant.
- [x] Freeze 100 eligible unique documents per domain under the existing deterministic content-hash rule; pair the same documents at max40/max128, right pad with EOS, and require at least two real tokens. Answers/solutions/tests are excluded from source projections and model inputs. Raw snapshots, selection IDs, and panels are outside Git at `D:\KD-SINK-central\analysis\S6_D25`.
- [x] Keep 512/1024 contexts disabled. S6 preparation is READY for separately authorized evaluation, not scientifically run.
- [x] Focused CPU tests: 73 passed; full offline CPU/unit/integration tests: 327 passed, 0 failed, 0 skipped; one external `astor` deprecation warning. Fresh read-only log/checkpoint audit verified all seven logs, 35 S4 states, and 28 S6 states. C3's exact sealed transferred-log-prefix exception remains the only log exception.

D25 SHA-256: `6c42ffebf8fa6b1a49787ec7468040939b6fd360c6132e2df10bf2b212780704`; source-lock SHA-256: `173b54c9dea5d19ad34a1174540173f40e1ad491ed4c81c9375f58e0be6d5235`. Frozen panel hashes, source evidence, and verification receipts are in `reports/stage08_s6_d25_preparation.json`; source tokens and raw text remain outside Git. S5's earlier mixed-root blocker remains open and is not changed by D25. No training, evaluation, GPU work, S4/S5 science, or Stage09 was executed.

## Prospective D26 - S5 seed0 mixed historical roots (2026-10-05)

- [x] Seal the researcher-approved, exact four-run S5 compatibility rule before reading S5 metric outcomes; retain all source roots unchanged.
- [x] Verify common comparison-critical protocol sections, seed/init/data/order/panel identity, architecture and teacher map, training schedule, objective registry, evaluation policy, calibration, numerical precision/backend, package map and 4080 runtime.
- [x] Add a narrow D26 validator, exact-root join admission, and tests for a valid mixed-root join plus rejection of a substantive schedule mismatch.
- [x] Reverify all four source logs and complete S1 evaluation records, reaggregate all Dense64 and retained Full300 C1/C2/C5/C6 seed0 records, and seal the complete S5 analysis output. Source/audit SHA evidence is in `reports/stage08_scientific_execution_20261005.json`.

D26 SHA-256: `888b21509000b570c83c2574f5a6bfbc3d1ec2dc197e88d841d085bac222f182`; critical-invariant SHA-256: `133d07d6513c7d25f7963afa76b01d0942e1658af90c585cd06fbc58cca9599e`. The four actual original roots are preserved in the amendment and `reports/stage08_s5_d26_compatibility.json`. Root/environment review found three physical RTX 4080 SUPER UUIDs; the S5 output must retain each per-condition UUID and hardware-lock hash as a hardware confound. The compatibility decision changes only join admission; it does not change the scientific estimand or authorize new S5 inference.

S5 execution completed on 2026-10-05 (UTC): all four D26 seed0 inputs passed fresh training/provenance and 59,206-file evaluation audits; 440 source panel/step rows reaggregated; 101 Dense64 and 9 Full300 steps joined with no missing entries. Audit SHA-256 is `09b3663cda822c7b5b3b697bbc3a44c4849b63e85f24b0a98abfce06c99dfa7c`; all four manifested files passed an independent SHA-256 check. No model was loaded and no inference was run. S4/S6 production drivers and their CPU tests are recorded in the same execution report; their scientific runs have not yet started at this milestone.

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

**Adrita-PC 4080 host readiness (2026-09-29 local):** The transferred ZIP matched its SHA-256 receipt and its 27 extracted files matched the committed scientific inventory. This machine's exact RTX 4080 SUPER UUID `GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf` passed the locked 101-package environment, five full-size GPU acceptance tests, runtime-source refusal test and D19 successor-root no-training preflight for C0/C1/C2/C5/C6; C3/C4 were refused. D19 transfers NodiPC's measured profiles, so this is a class-policy readiness result, not a per-card headroom pass. The common 4x16 schedule and same-UUID resume remain fixed. No condition was assigned or launched on Adrita-PC; NaveedPC C1 and the conditional C5 queue remain bound there. See `reports/stage06_adrita_4080_readiness.json`.

**S1 C6 production launch (2026-09-28, observed 19:46 UTC):** On explicit operator instruction, one C6 seed-0 10,000-update job started on Adrita-PC under the D19 successor root. Its external run directory is `E:\KD-SINK-stage06-runs\s1-c6-seed0-rtx4080super`. Step-0 dense64/full300 student and teacher plus LM2000 completed. Step-100 dense64/full300 student and teacher completed; training resumed and reached observed update 123 with finite loss and steady GPU memory. The job is running, with no completion claim. See `reports/stage06.json` and external structured logs. Audit the protected final checkpoint and required evaluation coverage after the job finishes.

**S1 C6 completion audit (2026-09-29, 06:15 UTC):** Adrita-PC C6 seed0 finished normally at 10,000 optimizer updates. All update losses and gradient norms were finite; cumulative counts are 81,920,000 input tokens and 81,280,000 shifted targets. All 222 required student/teacher evaluation completion events across 112 panel/step pairs are present, including the final LM2000 endpoint. The protected `final-010000` full checkpoint passed content-hash verification and contains optimizer, scheduler, RNG, data-order and exact counters. This verifies one C6 physical job, not completion of the wider S1 campaign. See `reports/stage06.json` and the external run directory.

**D20 C3 RTX4080 SUPER prospective successor (2026-09-29):** NodiPC's real C3 mb8 cycle completed but failed headroom; the separate mb4 ? accumulation16 cycle and checkpoint load passed. Sealed evidence `9b608a1a8da5af520435cd34dd45548ea29f3d85cf67977d1245276979bcb703` and researcher decision `15f9cab8d2d691f14aca3b249b0a5d1458626c59c38ff47ce338f68ad9a532f5` extend the exact-model 4080 class to C3 while keeping C4 3090-only. Runtime source `56260519be5e31790a7ed7abc078e2a285c0cf18`, hardware lock `fc410cd3d01505d0cd7b9ffdda98caaf070bfd2524ca7ac7841a0c78e60aa593`, and successor root `fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552` are prospective. The nine-job plan moves only the unstarted C3 assignment to the 4080 class and records `C4-C3-seed0` as a hardware confound. D19 remains valid for already-started C1 and the preapproved C5-after-C1 queue on NaveedPC; those jobs do not need to pull D20 mid-run. No C3 completion or other scientific outcome is claimed.

**D20 first root superseded:** `fd5199ae6f90985ce8c87539f3e76b6d810c5420ccb2d0f6d6ee300a27cef5f4` is archived and nonlaunchable because its preflight used the new plan for the original artifact approval. The corrected inventory reproduces the original sealed digest; replacement root `fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552` passed real NodiPC C3 preflight with artifacts/source/environment verified and `training_started=false`; C4 was refused. The single C3 launch is now authorized.

**S1 C5 Adrita-PC production launch (2026-09-29, observed 07:07 UTC):** On explicit operator instruction, one additional physical C5 seed-0 job started on Adrita-PC under the prospective D20 root, with the fixed 4x16 schedule and actual GPU UUID recorded. D20 C5 production validation and full no-training preflight passed before launch. Step-0 dense64/full300 student and teacher plus LM2000 completed; step-100 dense64/full300 student and teacher completed and training resumed to observed update 126. All observed losses and gradient norms are finite, update/token counters are sequential and exact, and GPU reservation is steady. The card had about 747 MiB free at observation, so later OOM remains an operational risk; no batch retuning is authorized. The separate NaveedPC D19 C5 queue status was not audited here, and this same-seed physical run is not an independent seed. C5 completion remains unverified. See `reports/stage06.json` and the external Adrita run logs.

**S1 C5 Adrita-PC interruption and recovery (2026-09-29):** The first C5 worker stopped after original update 3193 when Windows logged an unclean reboot at local 16:02:59. No Python traceback was found; the reboot trigger is unknown. The complete rolling-003000 checkpoint passed content and full-state verification with the approved C5/D20/Adrita identity, optimizer, scheduler, RNG, order and exact counters. Original logs through 3193 were archived under `E:\KD-SINK-stage06-Adrita\c5-interruption-20260929`; the active log was reduced to its byte-exact update-3000 prefix for replay and the old PID 35588 writer lock was archived after confirming no worker held it. Resume is prepared on the same GPU and 4x16 schedule; C5 is not complete. See the Stage06 report and CORE/S1 journals.

**S1 C5 Adrita-PC resumed health (2026-09-29, observed 18:21 UTC):** The same C5 seed-0 D20 command resumed from verified rolling-003000 on the same GPU UUID and 4x16 schedule. After cached step-3000 evaluation, it replayed all 193 interrupted-tail updates exactly for loss, objective components, gradient norm and learning rate, then passed the old stop point to update 3211. The active log has one sequence of updates 1..3211 with finite loss/gradient and exact token counters. Fresh step-3100 and step-3200 student/teacher dense64 evaluations completed; all 386 regenerated step-3100 cache records matched the archived interrupted records by SHA-256. The worker remains running toward 10,000; no completion claim. See `reports/stage06.json` and the external health audit.

**Adrita-PC C5/C6 seed0 transfer archives (2026-10-01):** Created independent ZIP64 archives outside Git at `E:\KD-SINK-transfer`. Each archive contains the complete source run directory, a file-level `TRANSFER_SHA256SUMS.txt`, and a `TRANSFER_MANIFEST.json`; each adjacent `.sha256` sidecar matches the finished ZIP. C5 preserves D20 root `fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552` and C6 preserves its original D19 root `48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791`. Both source/archive counts and uncompressed byte totals match; CRC, internal SHA list, and final-010000 checks passed. See the per-run archive receipts in `reports/stage06.json`.

**S1 C3 NodiPC launch (2026-09-29):** The one authorized seed-0 C3 run started under the corrected D20 root in `F:\KD-SINK-stage06-runs\s1-c3-seed0-rtx4080super`. Step-0 and step-100 evaluations completed, and updates 1-106 had finite loss and gradient norms. Windows then rebooted unexpectedly with no recorded training exception or bugcheck code. The original log and stale writer lock were archived under `F:\KD-SINK-stage06-ops\interrupted\`; the step-0 full checkpoint passed checksum and identity verification. The same run resumed from step 0 and replayed all 106 updates with zero mismatches in loss components, gradient norm, learning rate, and token counters; it then advanced to observed update 111. No C3 completion claim is made. The D19 C1 run and preapproved C5 queue on NaveedPC retain their original root and identity.

**S1 C3 interruption audit (2026-09-29):** The resumed C3 worker later stopped after update 2,571 during another period of unexpected NodiPC reboots. Updates 1-2,571 are sequential and finite, and step-2,500 dense64 student/teacher evaluations completed. `rolling-002500` passed full checksum and exact D20/UUID identity verification; the 71 later updates were not checkpointed. No normal completion record exists. Further same-run resume is held while the repeated host resets are investigated. Preserve the current log, checkpoints and stale writer lock as interruption evidence; do not mark C3 complete or start another condition.

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

Current authorized execution campaign update (2026-10-05): the preceding paragraph is the historical capability milestone. S5 is now COMPLETE for the D26-approved seed0 C1/C2/C5/C6 reuse-only scope. S4 (35 seed0 states plus the fixed teacher reference) and S6 (28 seed0 states, D25 panels, contexts 40/128) are staged as separate incremental GPU runs under `D:\KD-SINK-central\analysis\stage08_scientific_20261005`; neither had started at this report update. Latest statuses and hashes are in `reports/stage08_scientific_execution_20261005.json`. No training, S2 rerun, optional long-context work, or Stage09 is authorized.

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

## Prospective D21 C3-only seed replications (2026-09-29)

The researcher authorized C3-only seeds 1 and 2 and separate OpenWebText repacking/panels per seed. This supersedes D02's complete-campaign default for these two optional C3 jobs only. The D20 seed-0 C3 process resumed from verified step 2,500, completed update 10,000, and passed final-checkpoint, sequential-update, and 222-aggregate verification. Unexpected host resets remain a risk for later jobs.

- [x] Add tested D21 source support for exact per-seed provenance and a fail-closed one-shot queue.
- [x] Prepare and verify the two real seed-specific corpus/panel/order artifact sets when host memory is available.
- [ ] Seal and push the prospective D21 root, validate both real 4080 preflights, then arm seed-1 after verified seed-0 completion.
- [ ] Launch seed-2 only after seed-1 final checkpoint and required evaluation coverage verify.

The external queue status and run records determine actual launches; a prepared or armed queue is not scientific coverage.

## Prospective D22 C0/C2 seed replications (2026-10-01)

The researcher authorized four optional RTX 4080 SUPER jobs: C0 seed 1, C2
seed 1, C0 seed 2, and C2 seed 2. C0 and C2 share the same sealed artifacts
within each seed; seed 1 and seed 2 retain their separate D21 repacks/panels.

- [x] Add focused D22 source validation without changing objective/trainer/model/evaluator behavior.
- [x] Seal and push the D22 runtime source milestone.
- [x] Seal the exact D21 successor and generate four root-bound configs.
- [x] Pass all four real NodiPC production preflights with training disabled.
- [ ] Complete the external fail-closed serial queue and verify each final checkpoint, 10,000 updates, and 222 evaluation aggregates.

Queue order is C0 seed 1, C2 seed 1, C0 seed 2, C2 seed 2. A failed job or
storage/GPU/provenance gate stops later jobs. Queue state and run artifacts,
rather than this checklist, establish scientific completion.

## D23 optional seed replications across C0-C6 (2026-10-01)

- [x] Add the prospective all-condition seed1/2 amendment without rewriting D21/D22.
- [x] Push runtime-source milestone `9cb1815d44c3b2f42ddd508e5f9c71cc2ea274ec`.
- [x] Seal D23 root `7b12e5a637c3a8b6ebe66d4bedbd202077fc7960bc98a443babf8bf68de8d8a7`, preserve the seed0 nine-job plan, and validate fourteen explicit optional configs.
- [x] Verify all local seed1/2 artifact hashes and pass both real C4/3090 preflights without starting training.
- [x] Pass focused (106) and full CPU (255) regressions, offline clean-wheel, lock/package checks, and validate both job plans.
- [ ] Commit/push the validated D23 successor and start only the serialized C4 seed1 then seed2 queue.
- [ ] Audit each completed run's final checkpoint, 10,000 updates, and all 222 registered evaluation aggregates before claiming scientific coverage.

The optional plan assigns C0/C1/C2/C3/C5/C6 to the RTX4080 SUPER primary role
and C4 to the RTX3090. It adds no C1/C2 hardware bridge jobs. Optional C4
seed1/2 accepts an exact-model RTX3090 with the locked 24,576 MiB software
environment, records its actual UUID, and resumes only on that UUID; seed0
remains tied to its reference UUID. No seed1/2 condition other than C4 is
scheduled by the current AbdullahPC queue. Stage09 remains out of scope.

## Adrita-PC C5 optional seeds 1 and 2 (2026-10-01)

The operator explicitly requested only C5 seed1 and seed2 on Adrita-PC. The
seed-specific corpus, panels, update order, and CPU-FP32 initialization were
recreated from the pinned local source and match the D23 sealed digests. Both
no-training D23 preflights passed on RTX 4080 SUPER UUID
`GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf`. The C5-only serial queue in
`scripts/launch_stage06_c5_seed12_queue.py` requires its clean commit on
`origin/main`; it starts seed2 only after seed1 passes the final checkpoint and all 222
registered evaluation checks. Queue state is recorded in
`E:\KD-SINK-stage06-runs\launcher\c5-seed12-20261001\queue-status.json` and
run logs remain outside Git. Neither run is scientific coverage until audited.
The first end-to-end check-only invocation caught a queue-only expectation for
a `seed` field absent from otherwise successful preflight output; that false
gate was removed while the explicit config and `--seed` checks remain. The
corrected check-only gate passed and the serial queue was armed at
2026-10-01T12:08:52Z. At the 2026-10-01T12:51:42Z health sample, seed1 had
100 finite, ordered updates with exact token counters; its step100 evaluation
was advancing and seed2 remained queued.
Neither run is complete or scientific coverage until its final audit passes.

- [x] Push and arm the C5 seed1/seed2 serial queue; confirm seed1 training is healthy and seed2 remains queued.
- [x] Verify C5 seed1 completion: 10,000 finite sequential updates, final checkpoint integrity, and all 222 registered evaluation aggregates.
- [x] Repair the shared completion gate for C5 and permit the queue to resume only from a verified completed prefix; focused tests pass.
- [x] Commit/push the repair, pass the C5 queue's check-only gates, and rearm seed2 on the same Adrita-PC GPU.
- [ ] Complete C5 seed2 and audit both runs' final checkpoints, 10,000 updates, and 222 registered evaluation aggregates each.

At 2026-10-02T03:50:32Z, Adrita-PC C5 seed1 had completed 10,000 updates and all 222 evaluation aggregates. The independent queue verifier confirmed the D23 identity, actual GPU UUID, protected final checkpoint hashes, sequential finite updates, and evaluation completeness. The old serial queue stopped after seed1 because its shared D22 verifier rejected C5 as an unsupported condition; seed2 was not started. The narrow verifier/queue-resume fix was committed and pushed (`76d6462`). The first post-push check-only command used the wrong artifact-root path and failed with `tokenizer path is not a directory`; the recorded path is `E:\KD-SINK-stage06-Adrita\artifacts\KD-SINK-stage06-production`. Rerunning with that path passed both seed preflights and verified the seed1 prefix. The queue was rearmed at 2026-10-02T04:20:51Z and launched only seed2. At 2026-10-02T04:49:27Z, seed2 had 123 finite sequential updates, exact counters (1,007,616 inputs; 999,744 shifted targets), and 9 complete evaluations through step100 with no failures. The RTX 4080 SUPER had 894 MiB free at 79% utilization; this is an early health sample, not a completion audit.

## S1 C5 interruption and resume (2026-09-30, observed 07:49 UTC)

The earlier C1 gate completed successfully and launched C5 on NaveedPC. C5 reached update 8,070, then an unexpected Windows restart interrupted it without an OOM or traceback. The original log is preserved byte-for-byte as `train.precrash-through-8070.jsonl`. The checksum-verified rolling step-8,000 checkpoint retained the same C5/seed/root/physical-GPU identity. On explicit operator request, the same lineage resumed from step 8,000 with unchanged microbatch 4 and accumulation 16. At the latest health sample it reached update 8,148 with finite loss and no runtime error. Completion remains unproven; the practical ETC was 16:20-16:40 Asia/Dhaka on 2026-09-30.

## D21 optional C3-only replications (2026-09-30T03:10:59Z)
D20 C3 seed0 final coverage verified. D21 successor root 95a3607791fab3911916bcab407f7d10dab3576f197ebe7b606c2f9e3ac8ab15 seals separately packed seed1/2 C3 artifacts, unchanged C3 science and two explicit single-seed jobs. The external one-shot queue launches seed1 only after seal push; seed2 requires verified seed1 completion. No other S1 condition is queued. Scientific completion remains per run.
