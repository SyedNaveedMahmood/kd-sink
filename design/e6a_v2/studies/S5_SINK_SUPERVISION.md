# S5 - Full, excluded-sink and sink-only supervision

ANALYSIS/EVALUATION OF S1; no new training. Compare C2 full JSD,C5 NoSink,C6 SinkOnly,with C1 LogitKD, preferably within a same-device seed0 block. Already-recorded seed1/2 numeric results from completed campaigns remain eligible for compatible paired and appropriately labeled multi-seed analyses with original provenance. Under prospective D24, any new model inference for S5 is seed0-only; missing seed1/2 values remain missing and cannot trigger checkpoint replay. A cross-role contrast needs an explicit bridge/replica or reported hardware confound. Same seeds/manifests; no retuned replicas.

Ask whether directly supervising sink mass accelerates its pattern, whether sink-only supervision is sufficient for the measured sensitivity/usefulness under this recipe, and whether non-sink structure explains any behavioral improvement. Analyze all dense steps and10k with common full/nonsink attention metrics,S,delete/relocate DeltaCE,self-KL,absolute log effects,flips and clean loss.

C5 removes sink information BEFORE alignment/divergence. C6 matches binary sink-vs-rest probabilities and calculates its alignment only from that binary information. Single-column renormalization is vacuous and full-map alignment leaks non-sink information; both are prohibited. Test explicit invariances/gradients in OBJECTIVES.

## Limits of necessity/sufficiency language
These manipulate training SUPERVISION, not universal necessity of sinks for learning/inference. NoSink tests removing a particular target under this loss/base/architecture; SinkOnly tests adding that target to CE+LogitKD. Shared parameters/other losses can still create sinks in C5; C6 can create a pattern without making it useful. Natural later emergence in C0/C1 is a valid result.

Conditional non-sink and binary targets change the auxiliary objective/information content. Report losses and calibration conventions rather than claiming identical information budgets. Keep the common behavioral base fixed; no posthoc matching by sink strength. Present CE alongside attention similarities to identify metrics dominated by key0.

No extra S5 training jobs or duplicated seed counts; every row points back to unique S1 run IDs and compatible versions.

Journal: `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S5.md`. Stage08, final analysis09.

Prospective scope: [D24](../S1_CHECKPOINT_FOLLOWUP_POLICY.md). Preserve each original source run/checkpoint/protocol identity and record the separate D24 follow-up policy SHA for new inference. Never relabel historical roots.
