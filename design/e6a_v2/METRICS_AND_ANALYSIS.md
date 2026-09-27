# Measurements, estimands and analysis

The manuscript distinguishes pattern, circuit and function [PAPER]. The revision must additionally distinguish OUTPUT SENSITIVITY from usefulness for prediction. Self-KL/flips indicate perturbation, not necessarily improved predictive loss. Near-zero signed mean Delta CE does not imply unchanged predictions.

## Behavioral definitions
Use natural logs and explicit counts. For valid next-token position i, y_i is its ground-truth target, p the clean distribution and pI the intervened distribution.
- CE=-sum log p(y_i)/N. PPL=exp(CE), NOT average item PPL. Label fixed-block/truncated-context PPL [R8]. Preserve log-PPL and flag exponent overflow rather than silently cap.
- Delta CE=CE_I-CE_clean; positive means worse. Relative Delta CE%=100*Delta CE/CE_clean; nonpositive/tiny denominator is null with reason. Keep raw values.
- PPL ratio=exp(Delta CE); relative PPL change=expm1(Delta CE). These are algebraic transforms, not independent evidence.
- Self-KL=mean_i sum_v p_i(v)[log p_i(v)-log pI_i(v)] atT=1, clean-to-intervened. Full vocabulary, stable FP32 log-softmax, token-chunked without top-k approximation.
- Absolute target logprob change=mean_i |log pI_i(y_i)-log p_i(y_i)|, NOT |mean Delta CE|. Keep per-item Delta CE and median/p90 absolute item change.
- Prediction flips=mean 1[argmax p_i != argmax pI_i]. Store fraction; percent is display only. Also clean/intervened next-token accuracy against targets.
- Teacher KL for evaluation=KL(teacher||student) atT=1; trainingKD T=2 withT^2 is a DIFFERENT field. Teacher top1 agreement is not ground-truth accuracy.

Clean/intervention masks and targets must match. Save item NLL sums/counts and paired effects; aggregate token-weighted CE from sums. Item-weighted summaries are separately labeled for variable lengths. Keep improvements and degradations. Significantly negative self-KL is a failure, not automatically clamped away.

## Structure
For each real length n, second-half queries are q=ceil(n/2)..n-1. Per-head/layer sink is mean P(q,0); average queries within item, items, then heads. Save the full layer/head table. Primary scalar S averages all native layers; a normalized-depth band is secondary. Forced q0 self-attention must not enter S.

Interpolate per-layer head-mean profiles on16 equally spaced normalized-depth points d=l/(L-1). Report amplitudes separately. Normalize nonnegative profiles to unit mass on this depth SUPPORT and compute weighted1D Wasserstein; do not use unweighted magnitudes as samples. Zero mass yields unavailable distance. Constant-profile Spearman is null. No20-to16 head identity claim.

Common descriptive attention similarities: head-mean JSD/MSE, full-map and sink-excluded, using the same layers/masks for all conditions includingC0. Do not rank fidelity by comparing differently scaled C2/C3/C4 optimization losses. Record raw optimization losses separately.

Fingerprints retain baseline S, probed S, absolute Delta S and guarded ratio S_I/S. Ratios near a numerically negligible baseline are unavailable; log the locked denominator floor and show absolute effects. Fingerprint cosine alone discards effect magnitude and cannot establish function transfer. Teacher probe responsiveness and each probe are reported separately.

## Cadence
Dense at0,100,...,10000: fixed64-block clean behavior/teacher matching/structure and all-layer delete/relocate metrics.101 points per run are not101 independent replicates. Every retained state receives full300 including extra250. Endpoint clean NLL2000 is separate. S4 route probes use fixed saved states, not every100. Never replace missing causal values with cheap proxies silently.

Persist item scalars and structural summaries, not all vocabulary logits/attention tensors. Scalar logs allow reaggregation but do not reconstruct a model for new interventions. Missing checkpoint observations remain missing unless replayed/re-evaluated.

## Primary comparisons, subject to approval
| Question | Paired comparison at10k | Device block |
|---|---|---|
| Attention versus logits | C2-C1 |4080 primary,3090 replication|
| Logits versus ordinary training | C1-C0 |4080|
| Objective family | C3-C2,C4-C2,C4-C3 and each versusC1 |3090|
| Remove sink target | C5-C2, contextualC1 |4080|
| Sink-only target | C6-C2,C6-C1 |4080|

All-3090 approved allocation uses the analogous same-device comparisons. Co-report primary S, deletion Delta CE and deletion self-KL; relocation is complementary and prespecified. Secondary: complete trajectories, time-normalized trapezoidal AUC, and mean8000..10000. AUC requires observed endpoints; missing intervals are explicitly flagged, not silently imputed. Show individual seeds. Every output names metric/time/scope/panel/device.

With3 seeds report per-seed values, mean/sampleSD, paired differences/sign counts. Prompts/tokens/checkpoints/hardware repeats are NOT extra training seeds. An optional paired item bootstrap (10,000 draws,95%) is conditional on the fixed trained models, not training-population uncertainty. New formal hypothesis tests require prospective power/multiplicity decisions; no posthoc pile of uncorrected p-values.

No justified practical onset/equivalence margin is given by these sources. Until researcher approval, show S(t),self-KL(t),Delta CE(t) without binary functional-onset or no-function declarations. A nondetection through10k is horizon-censored, not proof that dependence never develops. Cancellation with flips/absolute effects must not be described as complete irrelevance.

## Secondary matched-behavior analysis
Equal-step remains primary. An approved secondary match chooses saved checkpoints within seed/device by minimum absolute CE gap<=.05nats, then minimum step separation, then earlier steps; selection cannot use sink/function. Publish selected pairs and unmatched cases. Do not interpolate unmeasured causal results. Conditioning on achieved CE changes the estimand and is not a randomized equal-budget comparison.

## External and robustness interpretation
S1C0 is the controlled normal-training baseline. S2 differs in architecture, data, tokenizer, optimization and tokens; it is an observational developmental reference, not a causal no-KD arm. Plot native steps AND tokens seen, emphasize within-family development, and do not compare raw cross-tokenizer PPL as if interchangeable. Missing public checkpoints stay missing. S6 text probes do not establish task accuracy/pass@k.

Freeze panels before production. Do not use trajectories to tune coefficients, select a favorable headline checkpoint or extend only supporting conditions. Publish contrary outcomes and protocol/resource deviations with affected run IDs.
