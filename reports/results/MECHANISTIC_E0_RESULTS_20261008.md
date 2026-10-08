# Mechanistic E0: audited recorded route trajectories

E0 is complete as retrospective reanalysis of existing S4/S5 observations. This is neither a preregistration nor new model inference. Implementation and validated execution use the isolated `mechanistic-e0` worktree; executable commit `af736c2a34afacd170fbd2379f352637041b4672`.

The complete S4 audit rehashed and checked all 108,000 item records, 36 summaries and their 108,037-file inventory. It recomputed token-weighted behavioral effects and equal-item sink means, verified all 35 student seed0 states plus the fixed teacher, all ten probes and the exact frozen Full300 membership. Historical S1 roots, devices, model-local coordinates, checkpoint bindings, D24 and the original C3 exception remain in provenance. Checkpoint tensor files were not loaded or freshly audited by E0.

## Outcome and interpretation

The composite outcome is **insufficient** because no independently specified scaling recipe was supplied. Raw component endpoint directions are **mixed**: 269 gaps toward the teacher, 80 away and one unchanged from step500 to step10000. These 350 correlated probe/metric/condition observations are descriptive components, not independent replicates, a vote, or a statistical test.

For C2, Q-bias CE response moves slightly toward the teacher between 500 and 10000, while EPE CE response moves away. Both post500 series are nonmonotonic. C3 shows the same endpoint directions; C6 Q-bias moves toward the teacher monotonically over the three post500 states while EPE is nonmonotonic. C5 moves toward the teacher on both CE components, nonmonotonically. This does not support a blanket claim that the measured routes increasingly diverge.

Student observations average native24 layers; historical teacher observations average native36. Intervention scopes and effective doses differ, especially for Q/K probes, and EPE coordinates are model-local. Small responses are not proof of absent routes. There is one training seed and no equivalence margin, significance test, causal mediation, or circuit-homology claim.

## Complete Q-bias and EPE CE trajectories

Values are signed Delta CE in nats per next-token target, displayed to eight decimals. Step0 is random student initialization. Each series below uses the original fixed probe definition, and tiny rounded values should not be read as exact zero. Teacher responses are fixed at Q-bias `0.05599476` and EPE `0.11780708`. The raw CSV includes all ten probes, five response channels, guarded relative sink changes, scopes and identities. The figure's full teacher range compresses small Q/EPE student differences; use this table and the raw CSV to inspect those temporal differences.

| Condition | Probe | 0 | 100 | 500 | 2000 | 10000 |
|---|---|---:|---:|---:|---:|---:|
| C0 | q_bias_all | 0.00000000 | 0.00002930 | 0.00047155 | -0.00000945 | 0.00034195 |
| C0 | epe_transport_layer0 | 0.00016471 | 0.00000208 | 0.00003552 | 0.00105639 | 0.00000132 |
| C1 | q_bias_all | 0.00000000 | -0.00000388 | 0.00076236 | -0.00002117 | 0.00078088 |
| C1 | epe_transport_layer0 | 0.00016471 | 0.00000035 | 0.00007408 | 0.00064591 | 0.00003454 |
| C2 | q_bias_all | 0.00000000 | -0.00001720 | 0.00052326 | 0.00021022 | 0.00090280 |
| C2 | epe_transport_layer0 | 0.00016471 | -0.00003904 | 0.00022835 | 0.00174949 | 0.00000658 |
| C3 | q_bias_all | 0.00000000 | -0.00001412 | 0.00052312 | -0.00002029 | 0.00076504 |
| C3 | epe_transport_layer0 | 0.00016471 | -0.00000372 | 0.00057497 | 0.00172329 | 0.00007253 |
| C4 | q_bias_all | 0.00000000 | -0.00000965 | 0.00004371 | 0.00010456 | 0.00079012 |
| C4 | epe_transport_layer0 | 0.00016471 | 0.00000045 | 0.00000285 | 0.00060539 | -0.00000051 |
| C5 | q_bias_all | 0.00000000 | -0.00001619 | 0.00028838 | 0.00007656 | 0.00117351 |
| C5 | epe_transport_layer0 | 0.00016471 | 0.00000031 | -0.00015316 | 0.00048959 | 0.00000833 |
| C6 | q_bias_all | 0.00000000 | -0.00001305 | 0.00007405 | 0.00014614 | 0.00081081 |
| C6 | epe_transport_layer0 | 0.00016471 | -0.00001261 | 0.00020229 | 0.00206350 | 0.00004417 |

## Already-recorded deletion context

The pinned S5 audit and all 440 source rows were reverified. Twenty compatible Full300 context states for C1/C2/C5/C6 join the five S4 steps; C0/C3/C4 context remains explicitly unavailable. S4 FP32 clean/sink observations and original S5 BF16 observations remain separate. S4 file-byte and S5 tensor-content checkpoint digests identify different representations.

For C2, S5 sink mass is `0.33184929` at500 and `0.33195972` at10000, while deletion Delta CE rises from `0.01240937` to `0.37335116` nats/target. This reproduces the developmental dissociation motivating E2/E3; it does not identify its mechanism.

## Artifacts and verification

External bundle: `D:/KD-SINK-central/analysis/mechanistic_e0_20261008/attempt01`. COMPLETE receipt SHA-256: `a47b79f66bafb39a0a9a9d67274c88fd3d32dfdbd2be6918728aa6a84b0dff7d`. Its checksum manifest binds all CSV/JSON/SVG/PNG outputs. The bundle verifier recomputes analysis and table semantics without source/model access. A second stdlib calculation checked every raw row against separately verified historical S4/S5 reports, all 1750 component differences, 350 trends, 35 context rows and artifact/source/code hashes.

[Validation commands, hashes and failure history](../mechanistic_e0_validation_20261008.json) record 435 passing CPU/unit/integration tests, zero failures/skips, including offline clean-wheel isolation and real CLI/figure roundtrip. One existing third-party astor deprecation warning remains. Three initial roundtrip failures exposed CSV column ordering after canonical JSON sorting; explicit column ordering fixed the cause. Failing evidence was retained and numerical checks were not relaxed.

The historical S4 enclosing wrapper exit1 versus sealed runner COMPLETE discrepancy is retained verbatim in provenance. This E0 audit and both final derivation/bundle checks exited0. No source results, scientific locks, historical runtime trees, model weights, objectives or shared evaluator definitions were changed.

## Next stage

E1-E3 remain pending. Resolve exact checkpoint grids, document-disjoint confirmation membership, calibrated doses and native/mapped scopes, precision/device/numerical policy, E3 reference norms and layer selection, then seal the separate intervention scope before inference. E4/E5 remain conditional and separately scoped; step500 weights do not establish an exact full-state continuation origin. E0 completion does not mark Stage09 complete.
