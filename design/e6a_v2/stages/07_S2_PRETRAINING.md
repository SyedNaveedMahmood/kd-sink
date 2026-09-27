# Stage07 - Public pretraining trajectories

Dependency:common stages00-06 accepted within declared scope; unresolved realGPU readiness remains a production blocker. Read AGENTS,NEXT_STEPS,CORE/S2 journals,studies/S2_PRETRAINING_DEVELOPMENT.md,DATA_AND_PROVENANCE.md,METRICS_AND_ANALYSIS.md.

## Tasks
- [ ]07.1 Implement explicit size/training-seed/native-step inventory and immutable branch resolution; standard1234 is not seed0.
- [ ]07.2 Implement single-checkpoint evaluation using separate Pythia-tokenized manifests and native architecture/scopes.
- [ ]07.3 Add opt-in trajectory iteration,bounded downloads/cache,verified-result resume and missing-checkpoint reporting.
- [ ]07.4 Pass local fake-inventory/native-model tests; perform scoped real network/GPU parity only when authorized,never automatically download924 states.

## Exit gate
Training seeds are distinct from evaluator RNG; no latest-checkpoint substitution; hashes/masks/rotary handling correct; native-step and training-token axes explicit. Actual checkpoint availability and storage/download budget are locked before full evaluation.

Run T09 S2,T02/T07 architecture,T01 tokenizer,T10 cache cases and CPU regression. Network/GPU skips remain blockers for corresponding acceptance. Write reports/stage07.json,append S2/CORE notes,update truthful tasks and commit milestones. Do not claim full S2 production from one smoke test.
