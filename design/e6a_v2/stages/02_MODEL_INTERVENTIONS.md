# Stage02 - Model adapters and causal primitives

Dependency:stage01. Read AGENTS,NEXT_STEPS,journals,MODEL_AND_INTERVENTION_CONTRACTS.md,OBJECTIVES.md and TEST_MATRIX.md.

## Tasks
- [ ]02.1 Implement narrow GPT-2/GPT-NeoX adapters preserving native projections,masks,scale flags,rotary/absolute positions,tied weights and gradients.
- [ ]02.2 Expose pre-dropout probabilities and Q/K/V through pure feature paths; activation recomputation cannot duplicate auxiliary side effects.
- [ ]02.3 Implement no-op/delete/relocate before V aggregation,integer scopes,q0 and padding exceptions,stable saturated-sink math.
- [ ]02.4 Build independent tiny/manual references and transactional restoration tests,including exceptions.
- [ ]02.5 Pass T02/T07 CPU forward/backward/mask/scope/numerical cases and earlier regression.

## Exit gate
Native/no-op parity on tiny GPT-2/GPT-NeoX,analytic changed logits from interventions,correct probability support and restoration. Do not accept editing a returned attention tuple after output computation as intervention. Full-size GPU parity remains stage06.

Record measured numerical errors rather than simply loosening tolerances. Append CORE/S1/S2 notes,stage02 report,NEXT_STEPS evidence and tested milestone commits. No long/model download runs without scope.
