# Stage00 - Foundations and strict configuration

Status:NOT IMPLEMENTED. Start here. Read root AGENTS.md,NEXT_STEPS.md and `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_CORE.md`, then DECISIONS.md,SOFTWARE_AND_ARTIFACT_CONTRACTS.md,SOURCES_AND_UPSTREAM_AUDIT.md.

## Tasks
- [ ]00.1 Inspect current tree/licenses; record any borrowed functions and immutable provenance. Do not modify/delete Upstream.
- [ ]00.2 Create the minimal installable sinklab package and strict typed configuration/CLI skeleton. One explicit study/condition/seed; unknown fields and hidden sweeps rejected.
- [ ]00.3 Implement draft-versus-approved protocol status, canonical hashing and approval/reuse templates. Draft configs cannot launch production.
- [ ]00.4 Resolve tested dependency versions using official docs and tiny-model compatibility tests; create real lockfiles, not invented version pins.
- [ ]00.5 Implement T00, an offline clean-wheel import test with Upstream absent, and stage-report/journal conventions.

## Exit gate
Package builds/imports offline; missing seed or approval fails safely; no default multiseed run or Upstream import. CPU configuration tests actually pass. GPU compatibility remains a later explicit gate, not a claim here.

Run implemented CPU suite: pytest -q tests/unit tests/integration -m 'not gpu and not network'. Record exact commands/results. Append CORE and affected-study notes including actual agent,user-visible communications,decisions,failures/skips,next task. Write reports/stage00.json. Check NEXT_STEPS only after implementation/tests pass. Commit significant tested milestones, then a stage completion commit. Never force-push/reset user changes or launch long training implicitly.
