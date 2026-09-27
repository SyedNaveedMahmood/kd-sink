# Sequential implementation worklist

DESIGN ONLY: no experiment code,CPU/GPU acceptance or scientific campaign is marked complete. Read AGENTS.md first. Work in order from the first incomplete task. A checkbox becomes checked ONLY when its implementation AND required tests actually pass with report/journal evidence. Missing hardware/network/approval is BLOCKED,not passed. Stage capability,validation and scientific coverage are separate. Optional studies remain disabled until approval.

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
- [ ]06.1 Independent S1/S3 configs and variant checks.
- [ ]06.2 Explicit approved job plans; no default sweeps.
- [ ]06.3 Actual3090/4080 model/teacher/intervention/objective smoke.
- [ ]06.4 Measured profiles,common microbatch,calibration/resources.
- [ ]06.5 ActualGPU resume/precision tests and readiness locks.
- [ ] Stage06 required evidence reviewed and milestone committed; unavailableGPU marked BLOCKED.

## Stage07 - S2
Work order: [07_S2_PRETRAINING](design/e6a_v2/stages/07_S2_PRETRAINING.md).
- [ ]07.1 Immutable public checkpoint/seed inventory.
- [ ]07.2 Single Pythia-checkpoint evaluator/manifests.
- [ ]07.3 Opt-in trajectories,bounded cache/download,resume.
- [ ]07.4 Local tests and approved real network/GPU evidence.
- [ ] Stage07 evidence reviewed and milestone committed.

## Stage08 - S4/S5/S6
Work order: [08_S4_S5_S6](design/e6a_v2/stages/08_S4_S5_S6.md).
- [ ]08.1 Anchor/route battery and model-local coordinate controls.
- [ ]08.2 S5 reanalysis-only joins and contrasts.
- [ ]08.3 Domain/length panels and optional long contexts.
- [ ]08.4 Study tests,new-teacher checks and approvedGPU smoke.
- [ ] Stage08 evidence reviewed and milestone committed.

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
- [ ] Protocol/environment/artifact/hardware/calibration locks frozen.
- [ ] Operator explicitly launches each approved study/condition/seed.
- [ ] Actual completed coverage audited before paper claims.
- [ ] Separate authorization obtained before eventual Upstream deletion.
