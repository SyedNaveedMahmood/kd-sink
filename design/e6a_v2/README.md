# E6A v2 design pack

Current follow-up scope: [prospective D24](S1_CHECKPOINT_FOLLOWUP_POLICY.md) restricts all new S1 checkpoint-dependent work to seed0. S4/S6 are designated-seed0 follow-ups; S5 may reuse already-recorded seed1/2 numbers. S2 and historical training roots/results are unchanged. Earlier status statements retain their design context; current validation is in NEXT_STEPS and the D24 report.

Status: partially amended research protocol, **not yet a fully approved preregistration or production experiment**. The corrected 2026-09-28 S1 researcher amendment resolves the single-seed campaign, condition/device eligibility policy, OpenWebText recipe and exact extension semantics. Stage06 remains blocked on production data, selected-role profiling, calibration and locks.

## Study map
| Study | Question | New training? |
|---|---|---|
| S1 | How do sink pattern and causal effects develop during long distillation? | Seven mandatory seed0 jobs; seeds1/2 are optional complete campaigns |
| S2 | How do sinks develop in ordinary pretraining? | No; evaluate pinned Pythia trajectories |
| S3 | Does the finding survive equal-head, fixed-correspondence supervision? | Optional approved replication: C0-C4, three seeds |
| S4 | Which positional anchors and parameter-level routes are inherited? | No; retained S1/S3 states |
| S5 | What does full, excluded-sink, and sink-only supervision transfer? | No; reuse S1 C1/C2/C5/C6 |
| S6 | Are findings robust across text domains and context lengths? | No; reuse retained states |
| S7 | Does direct sink supervision improve distillation utility, and how much attention matching concerns sink mass? | No new training; reuse sealed S5/S1 evidence, with a separately locked clean-attention supplement |

S1's mandatory campaign has **seven unique C0-C6 seed0 jobs**, each assigned a fixed approved GPU role; bridge replicas can raise the physical-job count. Optional seed1/seed2 campaigns add seven unique jobs each and require separate launches. Single-seed results cannot establish across-seed variance or reproducibility. S3 adds15 runs only if explicitly approved. S2 remains evaluation-only. See [hardware](HARDWARE_AND_EXECUTION.md) for the eligibility and batch gates.

The later D21 researcher direction prospectively permits **C3-only** seed-1/2 replications with separately repacked corpus and panels for each seed. It leaves the mandatory seed-0 campaign and existing run identities intact. These optional C3 records do not form complete seven-condition seed campaigns, and separate evaluation panels do not support direct item-level pairing across seeds. See D21 in [decisions](DECISIONS.md).

## Read in this order
1. [Decisions and approval gates](DECISIONS.md), [source audit](SOURCES_AND_UPSTREAM_AUDIT.md).
2. [Objectives](OBJECTIVES.md), [data/provenance](DATA_AND_PROVENANCE.md), [models/interventions](MODEL_AND_INTERVENTION_CONTRACTS.md).
3. [Metrics/analysis](METRICS_AND_ANALYSIS.md), [training/resume](TRAINING_AND_CHECKPOINTING.md), [hardware](HARDWARE_AND_EXECUTION.md), [software/artifacts](SOFTWARE_AND_ARTIFACT_CONTRACTS.md).
4. Relevant file under `studies/`, then the current file under `stages/` and [test matrix](TEST_MATRIX.md).
5. Root `NEXT_STEPS.md` and the journals under `implementation_notes/`.

## Revised evidence standard
The manuscript's main 500-step study and single-seed longer-horizon appendix motivate a longitudinal replacement with mandatory seed0 and optional additional seeds. Preserve its distinction among pattern, circuit, and function; do not presume its conclusion survives longer training. Ten thousand updates at this batch is 81.92M input tokens, not evidence of full pretraining convergence. A resumable10k run supports later extension, but any claim about later development requires actual later measurements.

Dense evaluation and sparse checkpoint retention are different policies. Log fixed-panel measurements every 100 optimizer steps, retain designated model states, and keep rolling/full-final recovery checkpoints. Failed or missing evaluations cannot be reconstructed from a deleted state without replay; retain per-item observations and recovery metadata.

## Delivery boundaries
This pack specifies what to implement and test. It does not claim the objectives fit either GPU, that upstream's historical 13.8 GiB measurement applies to the new code, or that the tests have already been implemented. Resolve production choices through the approval and pilot gates. Do not launch multi-seed runs by default.
