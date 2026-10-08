# E2 exact attention-output accounting: prospective candidate v1

Read MECHANISTIC_PROTOCOL_COMMON_v1.md. The researcher explicitly fixed15
students: C1/C2/C3/C5/C6 at500/2000/10000, plus the static teacher. Cover all24
student or36 teacher native layers and every native head. No head homology is
asserted. The15-state choice resolves the older proposal's inconsistent grid.

Capture actual entering pre-ln_1 residual, live Q/K/V, scores/probabilities,
pre/post-projection and outgoing attention output in FP32 eager eval. For each
query i>=1 with a causal nonzero-key support, construct conditional non-sink
probabilities by stable masked softmax of the same live scores excluding key0.
Never divide by1-a_i0 when it rounds to zero. Then
delta_o_i,h=a_i0,h*(vbar_non_sink_i,h-v0,h). q0 and padding have exactly zero
deletion delta. Concatenate heads in native order and multiply by GPT-2 Conv1D
W_O in [input,output] layout. The projection bias cancels and is not added.

Require live A@V equals captured preprojection, direct isolated deletion delta
equals reconstructed per-head/concatenated delta, projected delta equals direct
postprojection change, and adding the fixed clean-layer delta at the outgoing
attention junction reproduces direct deletion logits. Use matched masks and
all valid shifted next-token targets; q0 predicts the first shifted target but
its local attention edit remains absent. Require exact restoration on exceptions.

Report clean sink mass, v0/conditional value/contrast norms, each head's local
and projected contribution norms, per-layer projected norm relative to clean
output and cancellation ratio ||sum projected heads||/sum||projected heads||.
Also report per-head cosine with the double-precision projected-head sum, with
explicit support counts; zero/under-floor vectors have unavailable orientation.
This additive diagnostic does not alter historical metrics. Both all q>=1 and
the per-item later half ceil(real_length/2)..real_length-1 are reported. No
filtering by which support produces a favorable result. Proposed floor1e-8.

Local clean-layer accounting concerns one isolated deletion. It is not an
additive decomposition of nonlinear simultaneous all-layer deletion. The
E3 clean-output/conditional rescues test downstream computation separately.
Primary behavior is signed deltaCE and self-KL; factors are proximate diagnostics,
not proof of a unique training-time mediator.

Prospective workload at two300-item panels:721800 model forwards (648000
student isolated checks,64800 teacher isolated checks,9000 teacher references).
