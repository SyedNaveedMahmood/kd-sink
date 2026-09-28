# Approval lock boundary

The checked `design/e6a_v2/templates/` files are drafts. No production lock is
provided here, and `sinklab validate --production` requires one explicit
approved `protocol.lock.json` plus an exact digest in its run config.

An approved lock has `schema_version`, `payload`, and `sha256`. The digest is
SHA-256 of the UTF-8 JSON payload with sorted keys, compact separators, and
non-ASCII characters preserved. It excludes the enclosing digest field.
`payload` requires status `approved`, `production_ready: true`, a full source
commit, study, exact condition variants and device roles, the approved protocol body, named
approval evidence, and SHA-256 digests of artifact, environment, hardware,
and calibration locks. Stage 00 validates this boundary; later stages must
validate the detailed scientific contracts and measured lock contents.
The approval record must explicitly cover D01 through D18. The checked
`approval_draft.json` cannot serve as approval. `reuse_ledger.json` starts
empty because Stage 00 borrows no source code.

`sinklab validate --config configs/s1_c2_draft.json --seed 0` checks the
draft selection only. Validation never starts training or data preparation.

## 2026-09-28 S1 amendment

`s1_researcher_amendment_20260928.json` seals the user-provided researcher's partial decision approval (D01/D02/D07/D09/D10/D13/D18) and source attachment digest. `s1_artifact_partial_v1.json` and `s1_environment_verified_v1.json` seal only the immutable components verified this session. `approval_draft.json` points to them but remains `production_ready: false`; its production artifact/environment/hardware/calibration lock digest fields are null. The researcher identity/time of original approval is not inferred from the attachment. The draft protocol template contains null production corpus/panels, common batch and calibration factors. None of these files passes `sinklab validate --production` or authorizes training.

The researcher approved D01-D18 in `s1_researcher_approval_20260928.json`, with the sealed amendment taking precedence. The reviewed seed-0 plan has nine physical jobs for seven unique C0-C6 conditions: 4080 C0/C1/C2/C5/C6 and 3090 C1/C2/C3/C4. C1/C2 are same-seed hardware bridges. `reports/stage06_reviewed_batch_candidate.json` independently derives common 4x16 from the committed measured profiles; unused 3090 C0/C5/C6 and 4080 C3 pairs are not blockers for this plan. C3 remains ineligible for a future 4080 assignment until measured, and C4 on 4080 is prohibited. The production lock must bind the real OpenWebText/panel/init digests, measured pair/UUID matrix, C3/C4 3090 calibration, tested two-role environment, reviewed plan and immutable source commit. The candidate proof and approval record alone are not final production locks.
