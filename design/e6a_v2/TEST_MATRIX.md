# Required test matrix and evidence gates

D24 regression: reject S1 seed1/2 and ambiguous seeds before model/store access for S4, S6 (including optional long contexts), S5 new inference and future checkpoint panels. Enforce S4/S6 fixed steps; readiness requires only 35/28 seed0 states. Verify separate policy/source hashes, unchanged historical record keys, retained S5 seed1/2 numeric reuse, and unchanged S2 tests. CPU synthetic tests do not authorize scientific runs.

This is a specification, not evidence that experiment tests have run. Every report gives exact commands, environment/device, exit status, collected/passed/failed/skipped counts and log paths. Missing required hardware tests are BLOCKED, not passed. No finite suite catches every error. Explain discrepancies before changing numerical tolerances.

## T00: Configuration, CLI and independence
Reject missing seed, unknown fields, incompatible S/C/variant, wrong dimensions, unapproved/stale locks and forbidden GPU. Default invocation never loops seeds/conditions. Tiny smoke needs no downloads. Build/install wheel offline with Upstream absent; inspect imports/resource paths/symlinks for hidden dependency. Preserve attribution separately.

## T01: Data/provenance
Document split disjointness/dedup, EOS packing/no forcedBOS, label shift/masks, deterministic block prefixes independent of microbatch, epoch/extension cursor, corruption detection, identical initialization content hashes across conditions, tokenizer mismatch failure. Verify8192 input/8128 target count per update. Insufficient domain rows fail; synthetic test manifests cannot alter production partition rules. Hash tensor contents rather than nondeterministic pickle containers.

## T02: Model math
Tiny GPT-2/GPT-NeoX native-versus-adapter FP32 no-edit forward and gradient parity; correct projection/QKV layout, native scale flags, rotary positions, pre-dropout probabilities, masks and padding. Demonstrate that editing only a returned attention tuple is insufficient; actual edited probabilities must enter V aggregation. Real model-scale no-op checks on both eligible GPUs with measured precision tolerances.

## T03: Objectives and gradients
Hand-calculated CE/KL; matched shifted masks; T^2 once; detached teacher; finite intended student gradients; reject all-invalid inputs. Primitive JSD symmetry/bounds/identity; JSD key-sum vs MSE cell-mean reduction. C2 student-head permutation invariance; do NOT require soft mixtures of identical multihd sets to yield zero. C3 head averaging and denominator. C4 QQ/KK/VV axes/repartition/scaling, own dimension sqrt, causal support,R-divisibility rejection; do not implement QK by mistake. Chunked/unchunked forward AND backward parity.

C5 remove sink before alignment; invariance to sink-score perturbation and zero DIRECT auxiliary derivative wrt original sink score; saturated sink remains stable. C6 binary complement, nonzero loss for different mass, invariance to nonsink redistribution; regression test rejects singleton normalized-column loss. Small double-precision finite-difference/gradcheck fixtures, teacher detached.

## T04: Calibration/effective batch
Compute norm of accumulated full-batch gradient, not mean microbatch norms. Fixed calibration state/data, dropout restoration, no updates, no final panels, shared production factors, nonfinite/zero gradients rejected. In deterministic toy models batch64 equals microbatch accumulation; clip/optimizer/scheduler exactly once. Test variable valid-target denominators, inactive losses and no duplicate tied-weight optimizer entries.

## T05: Hardware solver
Mock complete eligibility matrix, prohibit REL-on4080, largest common divisor64, missing-profile refusal, actual optimizer-state allocation, isolated OOM candidates. Log headroom/device UUID. Resume cannot retune and OOM cannot shrink mid-run. Real profiles include optimizer/evaluator/save, not forward-only claims. Long-context eval has separate memory checks.

## T06: Checkpoint and training state
Tiny uninterruptedN vsK+resume+(N-K): weights,moments,LR,RNG,next IDs,next loss and metric keys agree. Interrupt during accumulation, before/after atomicrename and near evaluation. Partial/corrupt save fallback, disk-full retains last good state, duplicate writer rejected. Final full saves even off periodic schedule. Stop-after preserves original schedule;10k extension retains floorLR/moments/data. Evaluation insertion leaves subsequent training unchanged. Cross-device parity is distinct from same-backend exact replay.

## T07: Causal edits
Analytic toyP,V with unequal values; delete/relocate affect logits, conserve support/nonnegativity/row sums and leaveq0. Testa0,a1,doses,padding,length2; length1 exception/exclusion explicit. No arbitrary uniform fallback. Common target mask; key1control usesq>=2 for both comparisons. Integer layer scope and mapped teacher subset exact. FP32 row tolerance separate from BF16 rounding. No-op causal metrics zero within recorded tolerance. All parameter/mode/hook edits restore on exceptions.

## T08: Metrics/statistics
Manual CE/PPL/self-KL/flips/absolute-delta examples; exp aggregateCE not meanPPL; opposite signed item losses cancel in mean but not absolute measure. Correct KL direction and temperature. Chunked full vocabulary equals unchunked. Denominator/overflow guards and null-vs-zero. Weighted depth Wasserstein, shifted-profile positive distance, constant Spearman unavailable. Paired seed joins, hardware replicas not extraN, checkpoints not independent replicates, missing points flagged. Matched-CE pairing deterministic and independent of sink outcomes.

## T09: Studies
S1 allC0-C6 tiny tests and real-device smoke; simulated10k clock has101 dense evaluations plus extra250 full; final protected. S2 local fake branch inventory/seed1234 mapping, immutableSHA,no-latest fallback,missing checkpoint/cache behavior,GPTNeoX and tokenizer parity; network tests opt-in. S3 random Distil config,index maps,method IDs,new calibration. S4 batched EPE transport, top3/random coordinates,correct Kslice,restoration, Pythia not-applicable. S5 reuse-only joins, no hidden training. S6 exact source fields,40/128/long masks,no answer leakage or code execution/task-score mislabeling.

## T10: Terminal/artifacts/release
TTY tqdm and bounded non-TTY logs show correct study/condition/seed/GPU/batch. Global progress advances per optimizer update. Resume counters/elapsed and evaluator ETA honest. Disabling terminal output does not affect RNG. Idempotent metric writes, explicit versions/units, figures traceable to runs/panels and missing/failed rows visible. Build paper outputs without Upstream. No secrets/weights tracked; offline clean-wheel test.

## Suggested command surface once implemented
CPU: `pytest -q tests/unit tests/integration -m 'not gpu and not network'`.
GPU: explicit `pytest -q tests/gpu --device-role rtx3090` and4080 role after these options are implemented. REL-on4080 eligibility rejection is a test; unexecuted REL training is not a passing GPU result.
Network: explicit opt-in small inventory/download test, never an import-time requirement.
Production: not a unit test, never default. Approval,locks,realGPU profiles,fullS1 smoke/resume and no-Upstream gate are mandatory.

Each stage links its subset of these tests and records remaining blockers. A test command that does not exist is incomplete work, not a passed test.
