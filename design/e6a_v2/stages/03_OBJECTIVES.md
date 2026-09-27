# Stage03 - Seven objectives and calibration

Dependency:stage02. Read AGENTS,NEXT_STEPS,CORE/S1/S3/S5 journals,OBJECTIVES.md,DECISIONS.md,MODEL_AND_INTERVENTION_CONTRACTS.md.

## Tasks
- [ ]03.1 Implement shifted CE and exact full-vocabulary temperature-scaled KD,common behavioral base and explicit inactive fields.
- [ ]03.2 Implement C2 cosine-soft JSD and C3 head-mean probability MSE,exact masks/reduction denominators and variant metadata.
- [ ]03.3 Implement C4 causal QQ/KK/VV relations and exact chunked gradients; production-size REL restricted to3090.
- [ ]03.4 Implement C5 conditional NoSink before alignment and C6 binary SinkOnly; verify direct gradient and information-leak invariances.
- [ ]03.5 Implement initial-gradient calibration on separate fixed training-side batches/no updates,raw norm evidence,frozen factors and degeneracy rejection.
- [ ]03.6 Implement S3 fixed-index JSD/MSE variants with distinct IDs and architecture-specific calibration; pass T03/T04 and regression.

## Exit gate
Every objective has independent hand/reference forward and gradient checks; teacher detached; C5 has no direct sink-score leakage; C6 nonvacuous; REL correct axes/scaling/masks; chunked and unchunked gradients agree. No false soft-mixture identity test. Production calibration constants are not fabricated.

Save stage03 test report,append relevant journals and user-visible decisions,check only completed tested tasks,commit milestones. Do not change lambdas/targets after seeing evaluation outcomes.
