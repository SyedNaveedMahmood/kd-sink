# E1 calibrated route comparison: prospective candidate v1

Read MECHANISTIC_PROTOCOL_COMMON_v1.md; all approval/admission/statistical/failure
rules apply. Fixed grid: C2/C3/C6 at500/2000/10000, C1/C5 at500/10000:13 students
plus one static teacher per panel. Discovery Full300; proposed confirmation300.

Q bias is multiplied by1-alpha at selected Q slices only. K-input top3 and five
separately retained deterministic random sets multiply only their selected
input rows in the K columns of c_attn by1-alpha. Freeze alpha0/.25/.5/.75/1.
Coordinates are selected separately for each model from |p0+MLP0(p0)| without
LayerNorm, stable descending magnitude/index tie break. Random sets exclude top3,
are distinct, and use proposed control seed20260927 to preserve legacy comparators.
Across-model coordinates are not homologous.

Student native24, teacher native36 and teacher mapped24 are explicit separate
scopes. Mapped24 uses the original S1 map, not a learned correspondence. Report
scoped and global sink separately. None and exact Q-bias reapplication are no-op
controls. Every zero alpha must reproduce clean; nonzero parameter probes must
have the intended exact edited slices. If an original slice is zero, record
zero delivered dose rather than require an output change. Restore parameters
bitwise and remove hooks even on exceptions; qualification compares alpha1
against the original S4 operator. Do not change S4's old definitions.

EPE is the legacy layer0 MLP-output diagnostic. With a=(m0 dot u0), the transport
adds alpha*a*(u1-u0) to m0 and subtracts the same vector from m1; their sum is
conserved mathematically, with separately measured FP32 roundoff. Other positions
remain unchanged. This is activation transport, not EPE reconstruction under
LayerNorm. Anchor0-to1 uses p0+alpha*(p1-p0), with exact p1 copy at alpha1; this
is a broad position control. These diagnostic loci are separate from Q/K scopes.

Save actual parameter delta L2, layerwise Q/K/MLP/output activation deltas,
scope-matched output RMS, absolute and guarded relative sink changes and behavior.
Primary proposed matched-dose targets are sink-removed fractions.10/.25/.50
for Q/K routes, comparing teacher mapped24 with student native24. Use only exact
observations or adjacent observed alpha brackets; no extrapolation, loss-based
matching or selected branch of a nonmonotonic curve. Missing overlap or multiple
brackets yields unavailable/ambiguous. Retain every route/control and raw curve;
no aggregate over a favorable subset. Output RMS and parameter doses are
supporting diagnostics; no cross-model equality of bases or effective mechanism.

Prospective workload, including clean teacher references per student item:
432600 model forwards at proposed two300-item panels (374400 student route
forwards,50400 teacher route forwards,7800 teacher-reference forwards). Runtime
measurements must replace unsupported hour estimates. One command runs one state.
