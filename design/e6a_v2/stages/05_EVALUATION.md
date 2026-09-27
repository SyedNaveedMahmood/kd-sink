# Stage05 - Dense causal evaluation and records

Dependency:stage04. Read AGENTS,NEXT_STEPS,journals,METRICS_AND_ANALYSIS.md,DATA_AND_PROVENANCE.md,MODEL_AND_INTERVENTION_CONTRACTS.md.

## Tasks
- [ ]05.1 Implement exact CE/PPL,teacherKL,self-KL,absolute target log change,flips and accuracy using full-vocabulary streamed reductions.
- [ ]05.2 Implement sink profiles,common attention metrics,weighted normalized-depth topology and guarded fingerprint primitives.
- [ ]05.3 Implement dense/full/endpoint cadence and no-op/clean/delete/relocate runner,transactional edits and RNG-neutral evaluation.
- [ ]05.4 Implement item/versioned aggregate records,hash-safe caches,unique keys,failure states and idempotent evaluation resume.
- [ ]05.5 Pass T07/T08/T10, including cancellation examples and identical future training with evaluations inserted/removed.

## Exit gate
Simulated10k clock yields101 dense observations plus extra250 full evaluation without dropping200/300. All masks,counts,units,KL directions and reductions match references. Failed/missing observations remain visible. No ordinary runtime assumption that self-KL is a performance metric or meanDeltaCE=0 means unchanged predictions.

Run scoped and full CPU regressions,prepare FP32/BF16 diagnostic harness for stage06,write reports/stage05.json,append journals,NEXT_STEPS and tested milestone commits. No production trajectory fabricated from mock clock tests.
