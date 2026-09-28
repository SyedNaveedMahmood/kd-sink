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

The production lock must additionally bind the prepared OpenWebText and frozen panel manifest digests, complete RTX3090 C0-C6 measured hardware plan, C3/C4 calibration, exact synced production environment and all remaining D01-D18 approvals. Historical 4080 4x16 evidence is not a current production hardware lock.
