# Approval lock boundary

The checked `design/e6a_v2/templates/` files remain drafts. The final S1 seed-0
production lock is `protocol.lock.json`; `sinklab validate --production`
requires that explicit lock and its exact digest in each run config.

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

## RTX 4080 SUPER exact environment evidence

`s1_environment_verified_v1.json` remains historical partial evidence. The
before-state inventory is `reports/stage06_4080_environment_before.json` and
the sealed, versioned after-state record is `s1_environment_4080_exact_v1.json`.
The active RTX 4080 SUPER `.venv` was synchronized with `uv sync --locked`;
the follow-up locked dry run would make no changes, all 101 installed
lock-managed distributions match their versions in `uv.lock`, and `pip check`
and the offline clean-wheel/no-Upstream test pass. The three prior extras
(`build`, `pyproject-hooks`, `wheel`) were removed. `pip` and `uv` remain as
bootstrap tooling outside the lock and are listed explicitly in the evidence.
The 20 additional lock entries are for platforms other than this Windows host.

This 4080 role record is not the final two-role environment lock. On the
approved RTX 3090, reproduce the same `uv.lock`, verify equality of the
lock-managed installed-distribution inventory, and record its OS, NVIDIA
driver, CUDA/cuDNN versions and GPU UUID before sealing that lock. Scientific
Python package versions must match across roles; hardware metadata may differ.

## RTX 3090 exact environment evidence

`s1_environment_3090_exact_v1.json` records the approved device UUID
`GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e` and a no-change locked sync.
Its 101 installed lock-managed package names and versions equal the sealed
4080 record exactly. The approved `uv.lock` SHA-256 is
`b71f143ff21a0ccdd45e995b006430399134bb1564b1210bdc70fb3c67f4c3d5`;
the capture utility checks the LF-normalized Windows checkout against the
committed lock bytes before sealing. `pip` and `uv` are recorded separately as
bootstrap tools. At capture time this record alone did not make Stage 06
production-ready: external artifacts, measured C3/C4 calibration and transitive
production locks were separate gates.

## Restored production inputs and measured component locks

The 4080 transfer ZIP and sidecar receipt matched SHA-256
`220e428b54ba9c1e175826d3bcc14ac04ed0e5e2dcf9a59990d546ce32299d11`.
Its 27-file scientific tree was restored at the external
`E:\KD-SINK-stage06-production` root. The repository inventory validator
recomputed all content and scientific metadata; every field matched the
committed inventory except the expected absolute-root relocation from F: to E:.
The approved RTX3090 calibration ran from immutable source commit
`a29bcebc6253a5300452594bbaabe4b8e082a463` and produced external sealed
raw evidence SHA-256 `ddce66f76c43be3406f672cd2d0b4fd79fb35257ba28355caee72520e4932165`.
It measured 16 positive finite norms per term and median factors
`s_MSE=68.00580071126464`, `s_REL=0.120179255876581`.

The canonical measured components are `artifact.lock.json`,
`environment.lock.json`, `hardware.lock.json` and `calibration.lock.json`.
They bind the real inventory, both exact-role environments, reviewed measured
4x16 plan and raw calibration respectively. They do **not** authorize a run by
themselves. `protocol.lock.json` remains absent until the researcher gives an
exact positive `fingerprint_denominator_floor` for the guarded S_I/S ratio.
The current approval records give a UTC approval date, not an exact UTC time;
the final validator records date precision explicitly rather than inventing a
timestamp. Once that last value is approved, the final protocol root and all
nine production configs can be sealed and validated without rerunning profiles
or calibration. Scientific training coverage remains none.

## Final Stage 06 S1 production root

On 2026-09-28, before S1 production training, the researcher explicitly
approved `fingerprint_denominator_floor=1e-8`. This only guards division in
the reported S_I/S fingerprint ratio; it is not an effect-size threshold. If
`abs(baseline_sink)<1e-8`, the ratio is unavailable, while baseline sink,
probed sink and their absolute difference remain reported. The value is frozen
before outcomes and must not be changed based on results. The earlier D01-D18
approval and corrected amendment remain intact.

The real final sealer verified the four unchanged component envelopes:
artifact `2e721e9726e666cb8dc08a9b7d606a6d6ff99b434011b40fe2cf41f8eb315ccd`,
environment `0266e734fa97357ed0609e722c8d492284b34a06f726ce37b7301e703a370025`,
hardware `c284abebf7b9915053c282d57555a1a28cf9ad8250f3c402a3f313207eaae138`,
and calibration `fa031af3de63832e054b89539d3867c2904c9b592b1e7a8b3e4ca3e3676c532f`.
The validated `protocol.lock.json` envelope SHA-256 and production root is
`2a11da9bb71957a4d6b3a2f93a34bd67491dd9d21d70577a7a10858930a8943e`.
It preserves calibration source commit
`a29bcebc6253a5300452594bbaabe4b8e082a463` and binds the exact
reviewed nine-job 4x16 seed-0 plan. The nine generated configs and plan under
`configs/production/` each pass production validation. Stage 06 is
`production_ready`, but scientific coverage is `none`; no 10,000-update S1
job has been launched.
