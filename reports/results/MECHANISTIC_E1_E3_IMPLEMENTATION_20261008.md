# E1-E3 implementation and validation, 2026-10-08

E1-E3 software capabilities are implemented and validated in the isolated
`mechanistic-e0` branch at `E:/kd-sink-mechanistic-e0`. The executable source is
`5dd4fb9b98477bf0b70fbe1f3a9b694b310205fd`. These are engineering results;
**no inference on scientific teacher/student checkpoint states was run**.

E1 adds continuous Q-bias, model-local K top3/five random controls, EPE and
anchor probes, exact restoration, native/mapped scopes, actual parameter and
activation doses, and observed-dose comparison without extrapolation. E2
captures live attention computation, verifies stable isolated deletion algebra
against actual outputs/logits, and pools value contrasts, projection and head
cancellation. E3 adds equal-relative-norm sink/random/orthogonal/non-sink
controls, clean-output rescue, conditional all-layer rescue, order-dependent
telescopes and exact full-vocabulary target-loss geometry. The researcher-fixed
E2 grid is 15 students; E3 uses the entering residual before ln_1 as reference.

The single-state commands admit only an explicit seed0/phase/state/panel. They
bind original checkpoint and teacher identities, D24, prepared document-disjoint
panels, approved protocol/runtime and real production-shape qualification.
Fresh external bundles retain per-item scalar records, structured progress,
summaries, file hashes and COMPLETE; failed attempts remain unavailable. Reports
rehash/reaggregate and require complete matched grids. Original S4 behavior and
the original training/results/locks remain unchanged.

## Validation evidence

The full offline CPU/unit/integration suite passed **535 tests**, 0 failed or
skipped, in 202.27s. It includes old experiment regressions, independent numeric
references, adversarial protocol/checkpoint/panel/model-loading tests, failures
and restoration, real 128-token CLI roundtrips, and installation of a wheel with
the reference trees absent and networking disabled. One preexisting third-party
`astor` deprecation warning remains. Exact commands and failures are in the
CORE/S4 journals and [machine evidence](../mechanistic_e1_e3_validation_20261008.json).

Fresh final-source engineering artifacts are external:
`D:/KD-SINK-central/analysis/mechanistic_e1_e3_engineering_20261008/attempt_5dd4fb9`.
All six CPU/RTX 4080 SUPER bundles passed the runner and independent stdlib
hash/scalar audits. Each uses a tiny random GPT-2 (2 layers, 3 heads, width 24,
vocabulary 31), two 128-token inputs with real lengths 128/117, and 243 valid
shifted targets per operation. E1/E2/E3 contain 47/2/32 operations respectively.
All six JSON reports and 12 PNG/SVG exports were hash-verified; representative
figures were visually inspected.

| Engineering check | CPU maximum error | RTX4080 SUPER maximum error |
|---|---:|---:|
| Local/output/logit deletion parity, including E3 rescue | 9.34233e-8 | 1.84751e-7 |
| Exact per-target loss geometry | 4.44089e-16 | 0 |
| Equal-norm injection | 2.35329e-10 | 2.60622e-10 |

These measurements use explicit fixture tolerances, not approved scientific
thresholds. No large/medium production-shape headroom qualification, RTX3090
test or across-training-seed replication is claimed. The earlier engineering
attempt remains preserved. Its independent audit initially paired operations
by JSON key ordinal; that audit tool was corrected to join operation IDs,
without changing artifacts or tolerances. Initial E1 test and shell diagnostic
failures and their corrections are also retained in the journals.

## Remaining scientific prerequisites

The templates remain drafts. Production needs a researcher-approved hashed
phase protocol, explicitly selected document-disjoint confirmation blocks,
eta/layer/control/support/floor/tolerance choices, fixed device/runtime and
measured real GPT-2-large/medium production-shape qualification. Panel preparation
and admission verify the original OWT packing/tokenizer/document ownership;
they do not automatically choose confirmation membership. E3 fine-layer
selection must use discovery only, and discovery checks an alternate order.
Optional JVP analysis is not implemented. No training, E4/E5 or automatic
conclusion-conditioned acceptance gate is introduced.

See [execution contract and commands](../../design/e6a_v2/MECHANISTIC_E1_E3_EXECUTION.md).
Passing tests cannot guarantee absence of all bugs or establish the proposed
scientific conclusions.
