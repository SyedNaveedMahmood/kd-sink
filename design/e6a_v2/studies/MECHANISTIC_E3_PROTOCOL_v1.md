# E3 downstream sensitivity/rescue: prospective candidate v1

Read MECHANISTIC_PROTOCOL_COMMON_v1.md. Propose the same15 students as E2 plus
teacher, avoiding outcome-dependent state selection. The researcher fixed the
reference norm as the clean residual entering attention before ln_1. Inject
at outgoing attention output before residual addition; never normalize to the
edited residual, normalized ln_1 output or attention output norm.

Propose eta=[0,.01,.03,.10]. For each layer/item, independently verify E2
isolated parity, then use the actual projected sink-deletion delta direction.
Compare one deterministic Gaussian direction, its orthogonal projection to
the sink direction, and projected non-sink value contrast (v_key2-v_key1)W_O.
Propose seed20260927 and keys[1,2], query_min2. All four directions use exactly
the same real causal q>=2 positions, with requested norm eta*||clean residual||.
Normalize in double, deliver FP32. Proposed norm floor1e-8 excludes undefined
directions/reference norms; retain every direction-specific/common-support
count and mask. Never interpret exclusion as zero sensitivity. eta0 must
reproduce clean even when nonzero-dose directions would be unavailable.

Prespecified coarse thirds: student[0..7],[8..15],[16..23]; teacher[0..11],
[12..23],[24..35]. Propose lower-median representatives student[3,11,19],
teacher[5,17,29], which also correspond under the original mapped24 map. These
are layer probes, not aggregate interventions over an entire third. Both panels
use the same prespecified layers; no discovery result chooses this primary set.

A fine-layer extension is specified prospectively but remains disabled in the
coarse candidate. After separately authorized E2 discovery, for students select
one common layer per student third maximizing the equally weighted mean across
C2 500/2000/10000 of that layer's discovery-item median relative projected
delta (all real q>=1); ties choose the lower layer. Teacher selection uses its
own discovery-item median per native third and the same lower-index tie rule.
Missing/failed support blocks selection. Never use behavior or confirmation
outcomes. Freeze selected layers and hashed complete E2 discovery receipts
before confirmation, then issue a new derived fine-phase candidate/approved
lock under explicit authority. Do not silently replace the coarse candidate
or claim that the unexecuted fine selection has already been frozen.

At each selected layer measure direct deletion, deletion plus clean donor
attention-output restoration, all-native-layer deletion, and all-layer deletion
plus that single clean donor. The isolated clean-output rescue must restore
clean logits; the all-layer rescue is conditional/nonlinear and need not do so.
No automatic mediated fraction, unique circuit or necessity conclusion.

Run complete forward layer-order telescopes and complete reverse order on
discovery. The conservative coarse candidate retains both on confirmation too,
so qualification/settings are identical across panels. Each telescope sums
ordered increments to the same all-layer deltaCE; individual increments can
depend on order and are not independent contributions.

For every valid shifted target and the full50257-token vocabulary, double
logsumexp verifies deltaNLL=-delta z_y+logsumexp(log p_clean+delta z).
Save target probability, target-logit and normalizer terms, signed positive
and negative loss changes, self-KL and raw/centered logit-displacement norms.
No top-k approximation, linearization or optional JVP is part of this protocol.

Prospective workload at two300-item panels:1194600 forwards (1098000 students,
87600 teacher,9000 teacher references): three layers, four eta/four directions,
E2 parity, five clean/rescue forwards and two complete telescopes per item. This
counts actual evaluator execution; scientific execution remains unauthorized.
