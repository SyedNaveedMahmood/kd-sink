# S3 - Equal-head replication

ADDITIONAL TRAINING; disabled until D01/D14 budget approval. GPT-2-small teacher (12layers,12heads,768width) -> random student from DistilGPT-2 configuration (6layers,12heads,768width). Never initialize from pretrained DistilGPT-2. Map teacher[1,3,5,7,9,11] to student[0..5].

C0-C4 x3 explicit seeds x10k updates =15 EXTRA jobs, not part of S1's21/27 count. Shared GPT-2 data/panels,effective batch,schedule,dense cadence and checkpoint guarantees. Separate architecture-specific common batch preflight. Recommended allocation: all5 conditions on3090 to keep REL comparisons same-device. Alternative plans must respect REL-only3090 and control hardware confounding.

## Exact variants
C0/C1 remain CE/logit families. S3-C2=`index_jsd_v1`: fixed native-head index, normalized probability JSD, no cosine weights or learned adapter. S3-C3=`index_probability_mse_v1`: fixed-index headwise probability MSE, intentionally different from S1 head-mean MSE. S3-C4: causal QQ/KK/VV,64 relation heads (dimension12), all6 mapped layers. Freeze S3-specific MSE/REL calibration relative to S3-C2; do not blindly reuse S1 factors.

Equal heads enable an index convention, not proof that headj has the same function. Architecture and recipe also differ, so S3 is a robustness replication, not an isolated head-count-only causal manipulation. S1 remains the main within-pair method comparison. Save method IDs and use common head-mean descriptive metrics rather than claiming head homology.

## Gates
Same exact-resume,intervention,numerical and data tests, native teacher preflight, fixed-map checks, and index-loss identity/permutation sensitivity. Soft-alignment-specific invariance expectations must not be imposed on fixed-index objectives. Show every seed and trajectory with same primary endpoints. Contrary replication is valid.

Implementing the capability does not authorize launching15 jobs. Report S3 optional or incomplete until actual approved coverage exists.

Journal: `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S3.md`. Capability stage06; analysis09.
