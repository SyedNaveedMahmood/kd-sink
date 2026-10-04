# D24: prospective S1 checkpoint follow-ups

Researcher approval: 2026-10-04. **All future checkpoint-dependent analyses using S1 trained models are seed0-only.** This applies to new evaluation/intervention/model inference, including S5, and any future S1 checkpoint-based study. It does not prohibit analysis of already-recorded numeric results.

| Follow-up | Source conditions | Training seeds | Checkpoint steps | Logical states |
|---|---|---|---|---|
| S4 | C0-C6 | 0 | 0,100,500,2000,10000 | 35 |
| S6 | C0-C6 | 0 | 0,500,2000,10000 | 28 |
| S5 new inference / future S1 checkpoint work | Approved study scope | 0 | Separately approved study scope | No execution authorized |

S4/S6 are **designated-seed0 follow-ups** and cannot support across-training-seed reproducibility claims. Prompts, domains, checkpoint times, controls and hardware replicas do not increase the number of training seeds. Report descriptive within-seed effects; any item uncertainty is conditional on these fixed models. Keep assigned device blocks and explicit hardware confounds/bridges.

## Scope and preservation

- Completed S1 seed1/2 scientific results, logs, evaluations, checkpoints, manifests and provenance remain unchanged. This amendment authorizes no deletion or retention-policy change.
- S5 may reuse already-recorded seed0/1/2 numbers under existing compatibility and pairing checks. Only actual distinct completed training seeds support appropriately labeled mean/sample SD. Missing numeric records stay missing; new inference is seed0-only.
- Original S1 training and registered in-training evaluation are unchanged. Future post-training checkpoint replay/new inference follows D24. No training/resume/continuation is authorized here.
- S2 is unchanged, including its public pretrained seed series. Optional S3 scope is unchanged and remains disabled without its own approval. Teacher-only instrument checks and probe-control RNG are not S1 training-seed replications.
- No scientific evaluation, intervention, profiling, calibration, model download, or optional long context is launched or authorized by this amendment. Existing panel/hardware/runtime/approval gates still apply.

## Separate sealed policy identity

`protocols/s1_researcher_amendment_d24_seed0_followups_20261004.json` is a canonical SHA-256 envelope. Its payload digest is the **follow-up policy root**, not a replacement S1 training root:

`46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6`

The existing D23 training root remains:

`7b12e5a637c3a8b6ebe66d4bedbd202077fc7960bc98a443babf8bf68de8d8a7`

Every checkpoint keeps its own original root, including roots older than D23. A follow-up records the source identity plus `followup_policy.amendment_sha256`; it never rewrites `protocol_sha256`. Old sealed locks, configs, records, historical reports and journal entries remain historical evidence. D24 supersedes their all-seed follow-up wording prospectively only.

The original approval also seals `DECISIONS.md` as a whole file. That register is preserved byte-for-byte under its existing LF-normalized hash; D24 is recorded here and in its own sealed amendment rather than editing or resealing the historical register. Checkpoint manifests use `protocol_hash`; evaluator identities use `protocol_sha256`. Both original schemas are accepted without rewriting either, and contradictory hashes are rejected.

The new implementation changes the execution-critical source tree. It is **not** a reseal of the D23 training runtime. Existing training locks continue to reject a different runtime tree. Do not train/resume from this changed checkout under an old runtime lock; this task does not authorize changing that binding.

## Enforcement and read-only readiness

`sinklab.followup_policy.admit_s1_followup` requires an original S1 condition, integer seed0, original protocol SHA and checkpoint step; S4/S6 additionally enforce the fixed steps above. It runs before model access at S4 and S6 entries. The shared evaluator defaults to checkpoint follow-up admission for S1, including S5 or future panels. The trainer explicitly marks its original in-training cadence `registered_training`; this context must never be used for checkpoint replay or post-training inference. Existing record keys for registered training and S2 remain unchanged. New follow-up keys include D24 separately, preventing collisions with old records.

S4 student calls supply `source_identity` and `step` from the verified checkpoint, independently of the probe-control seed. Teacher-only calibration remains distinct. New studies must use the same admission function before loading/using S1 checkpoints; metadata-only numerical reuse does not require weights.

`configs/s1_checkpoint_followups_seed0.json` is a requirements plan, not a launcher. `scripts/check_s1_followup_readiness.py --study S4 --checkpoint <original-checkpoint-dir> ...` (or `--study S6`) validates D24/the plan, SHA-verifies explicitly supplied eligible checkpoint files, and reports missing seed0 requirements. It reads bytes only, never deserializes tensors or runs a model. Seed1/2 checkpoints do not supply or add requirements. An empty inventory reports 35/28 missing states; no actual machine artifact inventory is implied by policy validation.

Multiple retained/final copies or hardware bridges remain separate candidates with original identities. The checker does not choose an authoritative run, change device assignments, or authorize execution. A full inventory still needs explicit run/device selection, compatible panels and follow-up runtime/execution approval. Missing seed1/2 weights are never S4/S6/future S1 follow-up blockers.

## Validation and reporting

CPU synthetic regressions cover sealed-policy tampering, strict seed/step admission before model/store access, original-root preservation, historical-vs-follow-up key separation, seed0 readiness with bridges, unchanged registered training/S2 behavior, and S5 reuse of sealed seed1/2 numeric records. See `reports/d24_seed0_followups_validation.json` for executed commands/results. No synthetic test establishes scientific S4/S6 coverage.
