# Stage01 - Frozen data and paired initialization

Dependency:stage00 accepted. Read AGENTS,NEXT_STEPS,CORE/S1 journals, DATA_AND_PROVENANCE.md and SOFTWARE_AND_ARTIFACT_CONTRACTS.md. Review unresolved decisions before coding.

## Tasks
- [ ]01.1 Implement explicit preparation commands, normalized document hashes,deduplication/split disjointness,EOS packing and content-addressed manifests.
- [ ]01.2 Create CPU FP32 random initialization artifacts per seed; all conditions/hardware replicas load identical tensors.
- [ ]01.3 Implement64-block effective-update ordering,epoch/cursor/RNG serialization,prefix hashes and extension-safe suffixes independent of microbatching.
- [ ]01.4 Implement dense64/full300/NLL2000/calibration/domain builders with exact fields,masks,count guards and offline synthetic fixtures.
- [ ]01.5 Pass T01 and T00 regression; document artifact commands. Actual corpus downloads require explicit scope.

## Exit gate
Synthetic manifests reproduce hashes,never overlap documents across partitions, preserve ordered64-block batches and detect corruption/tokenizer mismatch. Initialization matches across conditions. Verify8192 inputs/8128 targets. Production revisions/licenses remain measured fields until downloaded and verified.

Run CPU suite and scoped data tests; record commands/exit codes and skipped network work. Append journals,write reports/stage01.json,mark only implemented/tested tasks and commit milestones. No source-directory dependency or implicit full-data download.
