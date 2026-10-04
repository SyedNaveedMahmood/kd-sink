# Model adapters and causal interventions

Prospective [D24](S1_CHECKPOINT_FOLLOWUP_POLICY.md) restricts every new evaluation/intervention using S1 trained checkpoints to seed0, including the positional controls and numerical sensitivity studies described below. S4 uses C0-C6 at0,100,500,2000,10000; S6 at0,500,2000,10000. Probe-control RNG seeds are not training seeds. S2 and separately approved optional S3 scope are unchanged.

## One attention computation, not a display-only hook
Implement a narrow native-PyTorch adapter for pinned Hugging Face GPT2LMHeadModel; GPTNeoXForCausalLM supports S2. Preserve pretrained projection layout, scaling, normalization, residual paths, activations, tied embeddings and architecture-specific positions. Assert layers/heads/width. Student weights are fresh from configuration, not pretrained.

Expose attention scores, normalized FP32 probabilities BEFORE attention dropout, and projected Q/K/V through explicit interfaces. Evaluation interventions replace probabilities BEFORE multiplication by V. Editing a returned attentions tuple after output computation is not causal editing. Test that changing probabilities changes downstream logits in a constructed nondegenerate example. Set use_cache=False for training/full-context evaluations. Do not silently use fused SDPA/Flash when explicit probability access is required.

Use eager attention in every paired condition, including C0/C1. A faster backend requires separately versioned forward/backward/intervention parity, not an unreported method change. The new runtime does not require NNsight; upstream is an independent reference, not a dependency.

Auxiliary collection must be pure: no global lists or hook side effects that duplicate losses when activation checkpointing recomputes blocks. Checkpointed blocks may return hidden outputs plus scalar auxiliary tensors; do not detach student losses into Python floats. Use explicit non-reentrant checkpointing and RNG preservation, validated against uncheckpointed gradients [R7]. Freeze one common checkpoint policy after profiling. No torch.compile in the baseline before separate validation.

## Masks and layer scopes
Right-padding only for variable-length evaluation; fixed training blocks have none. Valid edges satisfy real query/key and k<=q. Query0 stays unchanged because it has no alternative key. Invalid rows never contribute to denominators. Position0 is a slot, not a token ID/BOS assumption.

Primary S1 student scope is all24 native layers. Corresponding teacher scope is the24 layers in the distillation map. Report teacher all36 separately as secondary. Secondary student early/middle/late thirds are[0:8],[8:16],[16:24], with mapped teacher counterparts. The normalized-depth band0.25<=l/(L-1)<0.90 is a distinct secondary scope; save integer lists. S3 uses thirds of6 and its exact teacher map. S2 uses all native layers without a GPT-2 teacher mapping. Scope effects are not additive causal decompositions.

## Delete and relocate
Let normalized row P have sink mass P0 and intervention strength a in[0,1]; primary a=1.

Delete/redistribute: P'0=(1-a)P0; for k>=1, P'k=Pk+a*P0*r_k, where r is the conditional softmax of original scores over valid non-sink keys. At a=1 use conditional softmax directly. This avoids division by1-P0 when P0 rounds to1. Future keys stay zero.

Relocate: P'0=(1-a)P0; P'1=P1+a*P0; other entries unchanged. Apply only when key1 exists/is causal. Leave q0 unchanged. Neither operation changes V, positions, masks or labels. Preserve row sums, nonnegativity and support. A no-op and a=0 must match clean output.

Do not replace numerical failure by arbitrary uniform rows. Reject NaNs/invalid scores and mark failed evaluation. Check normalized FP32 row sums before casting (proposed atol2e-6); quantify BF16 rounding separately rather than imposing a false FP32 tolerance on BF16 tensors. Save max probability/logit errors and the actual numerical policy. Tolerances are validation settings, not scientific effect thresholds.

Dense evaluation is clean/delete/relocate on the fixed panel, with exact streamed vocabulary metrics. Teacher caches require identical revision, manifest, precision, scope, adapter and intervention hashes.

## Controls and numerical sensitivity
At retained states include key1 deletion as a positional control, scored on q>=2 for BOTH key0 and key1 comparisons. Report removed mass: low-mass deletion is not a magnitude-matched control. No-op is mandatory. Optional approved doses a={0,.25,.5,1} are evaluations, not new training conditions.

Run a fixed small panel in FP32 at step0,500,10000 and teacher on3090, using original FP32 weights where available, not merely upcast BF16 weights. Record clean and intervention BF16-vs-FP32 discrepancies. Effects at numerical-noise scale cannot establish fine rankings or no-function. Save paired item differences, not only separate means.

## S4 positional and route probes
First characterize GPT-2-large. A probe ineffective in this teacher cannot support noninheritance merely because it is also ineffective in the student. Do not assume the old medium circuit is universal.

Anchor probes: replace the position0 embedding by position1's embedding; remove all absolute positional embeddings. These are broad changes; measure both sink and behavior.

Route probes: zero Q bias in the specified layers; legacy-form EPE direction transport; ablate model-local K-input coordinates and matched-cardinality random controls. Different widths preclude copying teacher coordinate IDs into the student as homologous dimensions.

Proposed EPE diagnostic: E_j=p_j+MLP_0(p_j), deliberately without inserting LayerNorm into this diagnostic definition. u_j=E_j/||E_j||; reject zero norm. At layer0 MLP output m, for EACH example set a=<m_0,u_0>, m'_0=m_0+a(u_1-u_0), m'_1=m_1-a(u_1-u_0). This conserves the sum of those vectors, not their individual norms. Version and compare with [UP4]; this is a specific legacy-form probe, not an exhaustive circuit test.

Select top3 coordinates by |E0| independently per model/checkpoint, deterministic index tie-break; save coordinates/magnitudes. Select five deterministic random sets of3 outside them using dedicated control RNG. They are controls, not training seeds. GPT-2 Conv1D weights are[D,3D]: zero selected INPUT ROWS of the K slice only. Q/V and unrelated biases must be unchanged. Transposed Linear storage needs corresponding input-column edits. Test the exact parameter differences.

All parameter edits use transactional context managers and restoration in finally, including modes/hooks. Verify state hashes before/after; restore baseline between probes. GPT-NeoX rotary positions do not support identical absolute-embedding/EPE probes: record not_applicable, not a fabricated zero fingerprint.
