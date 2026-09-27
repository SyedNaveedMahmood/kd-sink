# C0-C6: exact objective contracts

These are the proposed v2 definitions (D03-D06), not a claim of exact reproduction of the named papers. One explicit condition selects one pure loss function; no condition-specific changes to initialization, corpus, optimizer, dropout, or step count.

## Shapes, masks, precision
For a mapped layer, normalized pre-attention-dropout probabilities are P^T[B,Ht,Q,K] and P^S[B,Hs,Q,K]. S1 uses Ht=20, Hs=16 and Q=K=128. Valid keys satisfy k<=q and both token masks; attention-objective query rows are q>=1. Exclude padding and query 0; never include masked -infinity score entries in MSE/cosine. Average equally over valid examples, heads/relations as applicable, and mapped layers; define within-row reductions below. Compute softmax/log-softmax, normalization, divergences and loss accumulation in FP32. Teacher targets are detached.

S1 maps student s=0..23 to teacher `ceil((s+1)*36/24)-1`, explicitly:
`[1,2,4,5,7,8,10,11,13,14,16,17,19,20,22,23,25,26,28,29,31,32,34,35]`.
Never reverse this map accidentally. S3 maps six student layers to teacher `[1,3,5,7,9,11]`.

CE uses logits at q=0..126 to predict input[q+1], masking invalid targets. Logit KD uses exactly the same query/target mask. Let p_T=softmax(z_T/T), p_S=softmax(z_S/T), T=2. KD = T^2 * mean_valid sum_v p_T(v)[log p_T(v)-log p_S(v)]. Apply T^2 exactly once. No top-k vocabulary approximation.

Recommended common base B=0.5 CE+0.5 KD and lambda=1/9. This deliberately holds the behavioral loss fixed across C1-C6. It preserves legacy attention/base ratios but not absolute legacy gradients. Approval D05 is required.

| Code | Training objective |
|---|---|
| C0 | CE |
| C1 | B |
| C2 | B + lambda JSD_soft |
| C3 | B + lambda s_MSE MSE_mean |
| C4 | B + lambda s_REL REL |
| C5 | B + lambda JSD_soft_NoSink |
| C6 | B + lambda JSD_soft_SinkOnly |

Inactive loss components are not computed every training step. Common descriptive attention metrics are computed at scheduled evaluations, outside autograd. C0 must not depend on teacher outputs for its update.

## C2: cosine-soft attention JSD
For each example and mapped layer, flatten each head over valid q>=1/key cells, L2-normalize with a documented epsilon, and calculate c[h,j]=cos(vec(P^T_h),vec(P^S_j)). Set w[h,j]=softmax_j(c[h,j]/1.0). Form R_h=sum_j w[h,j]P^S_j. Compare each teacher head to R_h with JSD(P,R)=0.5 KL(P||M)+0.5 KL(R||M), M=(P+R)/2.

Sum over keys, then mean over valid queries, teacher heads, examples and mapped layers. **Do not divide JSD by the number of key cells.** Weights stay differentiable through student probabilities; teacher side is detached. Weights are computed per example, never across a microbatch. There are no additional trainable adapter parameters. This is `cosine_soft_jsd_v2`, an adaptation of UP3 with pre-dropout normalized probabilities.

Primitive JSD(P,P)=0. However, soft alignment of an identical multi-head set need not be zero because soft mixing can average distinct heads. Do not write a false zero-loss identity test for the whole C2 pipeline. Student-head permutation invariance is a valid test.

## C3: head-mean probability MSE
Average native heads separately: A_T=mean_h P^T_h, A_S=mean_h P^S_h. For each example/layer, MSE_mean=sum_valid(A_T-A_S)^2 / number_of_valid_query_key_cells. Then average examples/layers. This is a cell-mean objective, unlike JSD's key-sum reduction; record its denominator. No learned or data-dependent head alignment and no head truncation.

Call this **TinyBERT-inspired post-softmax head-mean MSE**. Original TinyBERT uses pre-softmax scores [R1]. Do not silently change this C3 to score MSE. Score-MSE would require a distinct named variant and protocol amendment.

## C4: causal MiniLMv2-style relations
Capture projected Q,K,V before native-head splitting (or concatenate native heads in their exact original channel order). For each X in {Q,K,V}, reshape [B,L,D] to [B,R,L,D/R], with R=64 in the proposed protocol. Widths 1280 and 1024 give relation dimensions 20 and 16. **Relation heads are not native attention heads.** Reject nondivisible widths; never silently pad/truncate/group native heads.

For each relation head, construct A_X=softmax_causal(X X^T / sqrt(D/R)), including the diagonal, excluding padding, q>=1. Match KL(A_X^T || A_X^S), summing keys and averaging queries, relation heads, the three X types and mapped layers. Use each model's own sqrt(D/R). REL uses temperature 1 and no extra KD T^2 factor. Capture no teacher gradients; do not detach student Q/K/V.

This is QQ+KK+VV, **not merely QK attention plus VV**. Causal masking and all-layer supervision differ from the original encoder recipe [R2]. No guarantee that matching Gram relations identifies an exact circuit or transfers function. Process layers/relation types in chunks; chunking must preserve exact reductions and gradients. Do not change R or drop relation types to recover from OOM.

## C5: exclude sink supervision without a hidden leak
For each head, remove k=0 and renormalize over valid k>=1 **before both cosine alignment and JSD**. Prefer deriving conditional probabilities with a fresh masked softmax from the non-sink attention scores. This is mathematically P(k)/(1-P(0)) but avoids cancellation when P(0) approaches one. Query 1 has one remaining key and contributes zero divergence; query 0 is excluded throughout. Include only remaining cells in cosine normalization.

The auxiliary loss must be invariant to changing the original k=0 score while keeping all other row scores fixed. Autograd must give zero direct derivative with respect to that sink score, within numerical tolerance. Shared parameters, earlier layers, CE, and logit KD can still alter the sink; this is not a promise of zero total model gradient affecting sink behavior. Never calculate alignment weights from the full map and then remove the sink afterward.

## C6: supervise sink mass without non-sink shape
For each head/query form a two-outcome distribution Z=[P(0), sum_{k>=1}P(k)] = [p_sink,1-p_sink]. Calculate cosine-soft weights using flattened Z only, mix student Z, and apply Bernoulli JSD over the two outcomes. Average queries q>=1, teacher heads, examples and layers.

Never renormalize a single retained sink column: it becomes one everywhere and gives a vacuous zero loss. C6 must change when sink mass changes and remain invariant to redistribution among non-sink keys at fixed sink mass. It supervises sink-vs-rest allocation; it does not identify a specific head or guarantee causal necessity/sufficiency. Full-map alignment would leak non-sink structure and is prohibited.

## Proposed loss-scale calibration (D06)
Before production, use calibration seed 1729 and a separate training-only panel of 16 fixed effective batches; no optimizer updates or final-evaluation data. Reset the same initial student per objective, disable dropout for the calibration measurement only, and compute each raw auxiliary gradient's global L2 norm over the same student parameter set (missing gradients count as zero). Use full effective-batch gradients, not an average of microbatch gradient norms. Chunk/recompute to limit memory.

For j in {MSE,REL}, set s_j=median_b(||grad JSD_soft_b|| / ||grad L_j,b||). Save raw norms, per-batch ratios, manifests, numerical settings, calibration seed and the final constants. Reject zero/nonfinite gradients; do not clamp a huge ratio without a new decision. C2,C5,C6 keep scale 1 to preserve their directly specified JSD comparisons. Use one frozen factor per objective/architecture pair, shared across production seeds, devices and checkpoints. No adaptive balancing during production.

This controls initial gradient magnitude only; it does not prove optimal or equally strong supervision throughout training. Log raw/weighted losses and sparse gradient diagnostics on calibration runs. A raw-coefficient or coefficient-sensitivity study is separately named and separately budgeted, never silently mixed into primary results.
