# E6A v2 design pack

Status: research-backed implementation proposal, **not yet an approved preregistration or implemented experiment**. Prepared 2026-09-27 against repository commit `96a80e1e7ad581728c8befd697547f9b3f3a2e85`.

## Study map
| Study | Question | New training? |
|---|---|---|
| S1 | How do sink pattern and causal effects develop during long distillation? | Seven conditions, three explicitly launched seeds each |
| S2 | How do sinks develop in ordinary pretraining? | No; evaluate pinned Pythia trajectories |
| S3 | Does the finding survive equal-head, fixed-correspondence supervision? | Optional approved replication: C0-C4, three seeds |
| S4 | Which positional anchors and parameter-level routes are inherited? | No; retained S1/S3 states |
| S5 | What does full, excluded-sink, and sink-only supervision transfer? | No; reuse S1 C1/C2/C5/C6 |
| S6 | Are findings robust across text domains and context lengths? | No; reuse retained states |

S1 contains 21 distinct condition/seed combinations. The recommended two-GPU blocking plan has **27 physical runs**, because C1 and C2 are replicated across devices to support within-device comparisons. These duplicates are not extra seeds. S3 adds 15 runs only if explicitly approved. S2's recommended two sizes x three available training seeds is evaluation-only. See [hardware](HARDWARE_AND_EXECUTION.md) for lower-cost alternatives and limitations.

## Read in this order
1. [Decisions and approval gates](DECISIONS.md), [source audit](SOURCES_AND_UPSTREAM_AUDIT.md).
2. [Objectives](OBJECTIVES.md), [data/provenance](DATA_AND_PROVENANCE.md), [models/interventions](MODEL_AND_INTERVENTION_CONTRACTS.md).
3. [Metrics/analysis](METRICS_AND_ANALYSIS.md), [training/resume](TRAINING_AND_CHECKPOINTING.md), [hardware](HARDWARE_AND_EXECUTION.md), [software/artifacts](SOFTWARE_AND_ARTIFACT_CONTRACTS.md).
4. Relevant file under `studies/`, then the current file under `stages/` and [test matrix](TEST_MATRIX.md).
5. Root `NEXT_STEPS.md` and the journals under `implementation_notes/`.

## Revised evidence standard
The manuscript's main 500-step study and single-seed longer-horizon appendix motivate a longitudinal, multi-seed replacement. Preserve its distinction among pattern, circuit, and function; do not presume its conclusion survives longer training. Ten thousand updates at this batch is 81.92M input tokens, not evidence of full pretraining convergence. A resumable 10k run supports later extension, but any claim about later development requires actual later measurements.

Dense evaluation and sparse checkpoint retention are different policies. Log fixed-panel measurements every 100 optimizer steps, retain designated model states, and keep rolling/full-final recovery checkpoints. Failed or missing evaluations cannot be reconstructed from a deleted state without replay; retain per-item observations and recovery metadata.

## Delivery boundaries
This pack specifies what to implement and test. It does not claim the objectives fit either GPU, that upstream's historical 13.8 GiB measurement applies to the new code, or that the tests have already been implemented. Resolve production choices through the approval and pilot gates. Do not launch multi-seed runs by default.
