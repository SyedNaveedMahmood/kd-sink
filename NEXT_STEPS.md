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
- [x]06.1 Independent S1/S3 configs and variant checks.
- [x]06.2 Explicit reviewed seed-0 nine-physical-job plan for seven unique conditions, with C1/C2 same-seed hardware bridges; no default sweeps.
- [x]06.3 Actual3090/4080 model/teacher/intervention/objective smoke.
- [ ]06.4 Measured profiles,common microbatch,calibration/resources.
- [ ]06.5 ActualGPU resume/precision tests and readiness locks.
- [ ] Stage06 required evidence reviewed and milestone committed; unavailableGPU marked BLOCKED.

Stage06 status: **BLOCKED pending one approved metric floor and the final protocol root/configs**. Researcher choices D01-D18 are approved with sealed amendments taking precedence. `configs/s1_jobs_seed0_reviewed.json` fixes nine physical seed-0 jobs for seven unique C0-C6 condition/seed pairs: RTX4080 SUPER C0/C1/C2/C5/C6 and RTX3090 C1/C2/C3/C4. C1/C2 are same-seed hardware bridges. The measured scheduled-pair profiles establish common microbatch4 × accumulation16 (64 sequences, 8,192 inputs and 8,128 shifted targets per update). RTX3090 C0/C5/C6 and RTX4080 C3 remain unmeasured/unused; C4/REL is3090-only. The transferred ZIP at `E:\KD-SINK-stage06-production-transfer.zip` matched its 4080 SHA-256 receipt and restored the verified scientific inventory to `E:\KD-SINK-stage06-production`; the original ZIP remains external. Both exact environment records match on all 101 lock-managed packages. The approved RTX3090 seed1729 16-batch C2/C3/C4 calibration passed with `s_MSE=68.00580071126464` and `s_REL=0.120179255876581`, raw evidence under the external artifact root. Four measured component locks are sealed in `protocols/`. The evaluator requires an explicitly locked positive fingerprint denominator floor, but D01-D18 and the committed contracts do not give a numerical value; do not invent one. Until it is approved and the transitive protocol root, nine configs and final validation pass, 06.4/06.5 and Stage06 remain open. No 10k production run or Stage09 work has started; scientific coverage is none. See `reports/stage06.json`.

## Stage07 - S2
Work order: [07_S2_PRETRAINING](design/e6a_v2/stages/07_S2_PRETRAINING.md).
- [x]07.1 Immutable public checkpoint/seed inventory capability; one public step0 entry is now resolved and verified.
- [x]07.2 Single Pythia-checkpoint evaluator/manifests on local CPU fixtures.
- [x]07.3 Opt-in trajectories,bounded cache/download,resume on local CPU fixtures.
- [x]07.4 Local tests and approved real network/GPU evidence.
- [x] Stage07 capability evidence reviewed and milestone committed; S2 scientific coverage remains absent.

Stage07 status: external capability acceptance COMPLETE using one immutable public Pythia-160m step0 checkpoint on the RTX 4080 SUPER. Its hashes, tokenizer, native rotary/masks, evaluator, restoration, resume and storage were measured. Full checkpoint coverage and the full-trajectory storage/download budget still require approval and measurement; no S2 scientific trajectory or coverage exists. See `reports/stage07.json`. Stage06 production-readiness status remains BLOCKED.

## Stage08 - S4/S5/S6
Work order: [08_S4_S5_S6](design/e6a_v2/stages/08_S4_S5_S6.md).
- [x]08.1 Anchor/route battery and model-local coordinate controls (CPU capability; real teacher calibration pending).
- [x]08.2 S5 reanalysis-only joins and contrasts (CPU capability; no production S1 records yet).
- [x]08.3 Domain/length panels and optional long contexts (CPU capability; optional long contexts disabled without approval/memory validation).
- [x]08.4 Study tests,new-teacher checks and approvedGPU smoke (RTX 4080 SUPER engineering qualification; S5 analysis has no GPU kernel).
- [x] Stage08 capability evidence reviewed and milestone committed; scientific coverage remains absent.

Stage08 status: capability and external qualification COMPLETE on the RTX 4080 SUPER using a pinned GPT-2-large teacher and a frozen engineering calibration panel. All ten fixed S4 probes responded at the predeclared numerical reporting floor; that observation is not a scientific transfer claim. Paired 40/128-token S6 GPU smoke passed on agent-authored engineering inputs. S5 remains read-only with no production C1/C2/C5/C6 records; none were fabricated. Optional 512/1024 contexts remain disabled. No S4/S5/S6 scientific coverage exists. See `reports/stage08.json`. Stage06 remains BLOCKED; Stage07 external capability acceptance is complete without scientific coverage. Stage09 has not started.

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
