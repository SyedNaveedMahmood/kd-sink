# Mechanistic follow-up: E0 source audit and route trajectories

Implementation scope authorized 2026-10-08: E0 only, one stage at a time.
Source proposal: [complete mechanism plan](../../KD_SINK_Mechanistic_Validation_Experiment_Plan.md).
This is retrospective analysis, not a preregistration, inference approval, or
an amendment to S1/S4/S5/S7. The identifier `mechanistic_followup/E0` is an
implementation namespace; a numbered scientific study has not been assigned.

E0 reads the complete S4 tree: C0-C6 seed0 at 0/100/500/2000/10000,
ten probes per state, a fixed teacher, 300 identical Full300 items. It checks
the caller-pinned independent audit, runner audit, run manifest, checksum
inventory, every record/summary seal, identities, item coverage and item-to-
aggregate arithmetic. Outputs preserve original protocol roots, model and
checkpoint hashes, actual training/inference devices, model-local coordinate
plans, scopes, control RNG seed, panels and historical exception/discrepancy.
It does not load checkpoint tensors, use a GPU or reconstruct missing data.

All ten probes and all five observations remain separate. Behavioral effects
use token-weighted sums/counts; sink mass uses equal-item means. Teacher scope
is its historical native36 scope; student scope is native24. E0 cannot invent
a mapped24 teacher reference or claim that these scopes/doses are comparable.
Step0 is random student initialization. Probe coordinate IDs are model-local
and can change with checkpoint. Controls are not training replications.

Required outputs: raw route trajectories CSV (350 student rows plus ten teacher
rows), teacher/student component differences CSV, observed endpoint/trajectory
directions JSON, trajectory context CSV, component plot and plot-data JSON, source provenance, checksum
manifest and sealed COMPLETE receipt. An independently specified positive
scaling recipe may enable an exploratory RMS fingerprint distance. There is
no default mixed-unit distance, binary equivalence margin or inference test.
Without a recipe, composite convergence/divergence is `insufficient`; individual
raw-component trends remain visible. Small effects are not route-absence proof.

The optional already-recorded S5 bundle supplies standard deletion/relocation
context for C1/C2/C5/C6 at the five S4 steps, after its pinned audit and all 440
source rows are reverified. Preserve signed effects and unavailable C0/C3/C4
context. S4 FP32 clean/sink observations and S5 BF16 observations remain
separate columns; their checkpoint digests describe different byte/tensor
representations and are not asserted equal. No new inference fills missing data.

The optional fingerprint recipe is sealed JSON: kind `e0-fingerprint-recipe-v1`,
status `exploratory`, explicit `rationale`, `discovery_description`, `components`
(each a unique probe/metric pair with positive `scale`). Its hash travels with
outputs. Scales are supplied by the researcher, never fitted to confirmation
items. RMS uses equal component weights; endpoint direction is descriptive,
not practical significance. Exact-zero change is `unchanged`, not equivalence.

Only fresh external output directories are accepted, outside the source and
any repository. Writes are exclusive. A COMPLETE marker is written last;
failures preserve an explicit FAILED/BLOCKED receipt and never produce COMPLETE.
Source inventory content/metadata stability is checked after reading. Completed
outputs are immutable and independently verifiable without source/model access.

Engineering fixtures use the same grid and semantic checks but may have fewer
items under an explicit fixture flag. All fixture outputs say engineering-only;
they cannot claim scientific coverage. Tests must accept contrary/null trends.

E1-E3 remain pending approved grids/panels/doses/scopes/numerical policy/device
and new intervention locks. E4 additionally needs a full step500 branch origin;
weights-only states are insufficient. E5 needs a separate replication protocol.
Existing training source guards and scientific locks remain unchanged.
