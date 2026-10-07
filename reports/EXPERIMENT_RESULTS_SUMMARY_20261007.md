# KD-SINK experiment results summary

**S1 coverage update (2026-10-07):** This consolidated report used the then-audited seven seed0 S1 runs. A later checksum-verified compact transfer established C0–C6 numeric records for seeds 0, 1, and 2. See [the dedicated three-seed S1 report](S1_C0_C6_RESULTS_20261007.md) for current S1 coverage, per-seed numbers, and the limits of the compact seed1/2 artifacts. The historical seed0 statements below retain their original evidence scope.

**S4 detail update (2026-10-07):** See [the dedicated S4 results report](S4_RESULTS_20261007.md) for the complete 35-state seed0 trajectory, all ten endpoint probe families, teacher reference, source identities, full-tree audit, and interpretation limits. The S4 results summarized below remain the same audited scientific records.

**S5 detail update (2026-10-07):** See [the dedicated S5 results report](S5_RESULTS_20261007.md) for the complete D26 seed0 reuse join, Full300 trajectory, Dense64 time summaries, all endpoint measures, source identities, and limits. The S5 results summarized below remain the same sealed scientific records.

**Evidence cutoff:** 2026-10-07

**Repository source:** e4cd7f2a25fcdcbcd85ca51b835c3a93cf71c69a

**Results machine:** DESKTOP-GPO5752 (Adrita-PC)
**Purpose:** consolidated report of the verified scientific results and explicit non-results for the current KD-SINK studies.

## Executive summary

The verified central S1 set contains the seven mandatory seed0 condition runs, C0–C6, each ending at optimizer step 10,000. At that endpoint, C2, C3, and C6 showed much larger first-position attention mass than C0, C1, C4, or C5. Deleting or relocating the first-position attention key produced larger next-token loss changes in C2/C3/C6 than in C0/C1/C5, although C4 shows that a small sink score does not mean zero sensitivity. These are descriptive contrasts from one training seed, not across-seed estimates or evidence that sink mass mediates the effects.

S4 completed the seed0 positional-anchor and parameter-route probe battery and passed an independent full-tree audit. Moving the position-0 embedding to position 1 reduced the measured sink score and changed model outputs, especially in the high-sink C2/C3/C6 runs. Removing all absolute position embeddings caused large loss increases in every condition; that broad intervention cannot identify a sink-specific mechanism.

S5 is a read-only reaggregation of existing C1/C2/C5/C6 seed0 evaluation records. It joined 101 Dense64 steps and nine Full300 checkpoints without new inference. S7 then added 36 clean-attention decompositions over those four conditions and nine checkpoints. At 10,000, C2 and C6 had lower teacher–student full-attention divergence than C1, mostly through lower sink-mass divergence. C5 had a lower full divergence than C1 but retained a much larger sink-mass discrepancy than C2. Clean next-token metrics were close across the four recipes; delete/relocate effects were appreciably larger for C2 and C6 than for C1 and C5.

S6 completed all 28 designated seed0 checkpoints and passed an independent audit. Across three frozen text domains, longer available context reduced pooled next-token cross-entropy for every condition. The strongest delete/relocate sensitivity generally appeared in C2, C3, and C6. These are causal language-modeling measurements on text prompts, not SST-2 classification accuracy, GSM8K answer accuracy, or HumanEval pass@k. HumanEval code was not executed.

S2 has no scientific pretraining trajectory result in the verified campaign record. S3 remains disabled and has no scientific runs. Capability tests or engineering smoke results for these studies are not experimental outcomes.

Every checkpoint-dependent follow-up summarized here used training seed0 under D24. The report makes no across-training-seed reproducibility, p-value, equivalence, noninferiority, or causal-mediation claim. Condition and hardware are partly confounded: the original S1/S5 records use three physical RTX 4080 SUPER UUIDs, and C4 was trained on an RTX 3090.

## Scope, terms, and study names

This report follows the current repository’s study names. A previous user brief called the sink-aware utility analysis “S5”; the repository now identifies that analysis as **S7**. Here, **S5** means the D26 reuse-only reaggregation of existing S1 C1/C2/C5/C6 numeric records, and **S7** means the separate sink-aware utility and clean-attention decomposition analysis.

The report uses only existing S1 logs, evaluation records, sealed Stage08 summaries/audits, and the completed S7 campaign outputs. No scientific command was run to prepare this report. Values are descriptive and are reported at the units and pooling level recorded by the source analysis:

- **ΔCE** is edited next-token cross-entropy minus clean cross-entropy, in nats per valid target; positive values indicate increased loss after the edit.
- **Sink S** is the recorded first-position attention-mass score. It is a structural measurement, not a quality score.
- **Full, mass, and shape JSD** are the S7 teacher–student attention divergences and its prespecified decomposition. Mass plus shape closes to full JSD within floating-point tolerance.
- S6 token-weighted pooled values combine the three 100-document domains by valid target count. Paired context contrasts instead give equal weight to each matched document within a domain.

## Coverage at a glance

| Study | Verified scientific coverage | Status and boundary |
|---|---|---|
| S1 | C0–C6, seed0; 10,000 updates each; Dense64 evaluations at 101 steps; Full300 at nine retained steps; LM2000 at steps 0 and 10,000 | Completed primary seed0 set. C3’s transferred training log omits updates 1–2,500 under the exact accepted historical exception described below. |
| S2 | No scientific checkpoint trajectory | Not run. Stage capability or engineering checks are not S2 results. |
| S3 | 0 of 15 optional C0–C4 × three-seed jobs | Disabled; no scientific result. |
| S4 | 35 student condition/checkpoint batteries plus one teacher reference; 300 Full300 items and 10 probes per battery | Complete; independent full-tree audit passed. D24 seed0-only follow-up. |
| S5 | Four source conditions C1/C2/C5/C6; 440 panel-step rows; 101 Dense64 and nine Full300 joined steps | Complete reuse-only reaggregation; no model loading or new inference. |
| S6 | Seven conditions × four checkpoints; 28 states; three domains × two contexts × 100 prompts × three operations; 50,400 item-operation records and 168 aggregates | Complete; independent audit passed. D24 seed0-only follow-up; optional 512/1024 contexts disabled. |
| S7 | C1/C2/C5/C6 × nine retained steps; 36 clean Full300 decomposition supplements, 300 items each | Complete; independent audit passed. D24 seed0-only follow-up. |

The central S1 index verifies exactly seven primary seed0 runs and explicitly does not claim full multi-seed coverage. This report therefore does not pool seed1/seed2 results or calculate across-seed means or sample SD. Optional historical C5/C6 seed1/seed2 records are not part of the audited input set used here.

## S1 — Longitudinal distillation

The primary study trained the randomly initialized GPT-2-medium student against the frozen GPT-2-large teacher for 10,000 optimizer updates. Each update used 8,192 input tokens and 8,128 shifted targets; the full run totals are 81,920,000 input tokens and 81,280,000 targets. C0–C6 are distinct objective recipes: C0 CE; C1 the shared CE/logit-KD base; C2 full cosine-soft attention JSD; C3 scaled post-softmax head-mean probability MSE; C4 scaled causal QQ/KK/VV relations; C5 conditional non-sink attention JSD; and C6 binary sink-versus-rest JSD.

### Run identity and training-log coverage

| Condition | Run ID | Training GPU UUID | Original protocol root SHA-256 | Training log |
|---|---|---|---|---|
| C0 | s1-c0-seed0-rtx4080super | GPU-72b4b307-b613-c35e-ea32-53f4431de9ee | 910961fcc53edaed0df48e3139dbb7ca2e058bf7480b67dab27bae0bcd90f26f | 10,000 update records |
| C1 | s1-c1-seed0-rtx4080super | GPU-f6ff547d-b5cb-c2b8-ca56-ef87e6c18af0 | 48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791 | 10,000 update records |
| C2 | s1-c2-seed0-rtx4080super | GPU-72b4b307-b613-c35e-ea32-53f4431de9ee | 910961fcc53edaed0df48e3139dbb7ca2e058bf7480b67dab27bae0bcd90f26f | 10,000 update records |
| C3 | s1-c3-seed0-rtx4080super | GPU-72b4b307-b613-c35e-ea32-53f4431de9ee | fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552 | 7,500 records, updates 2,501–10,000 only |
| C4 | s1-c4-seed0-rtx3090 | GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e | 95a3607791fab3911916bcab407f7d10dab3576f197ebe7b606c2f9e3ac8ab15 | 10,000 update records |
| C5 | s1-c5-seed0-rtx4080super | GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf | fccf4c14bc691e550c6304f4955b72037efb8d042b3e51afb70367920fff0552 | 10,000 update records; a recorded resume marker at step 3,000 |
| C6 | s1-c6-seed0-rtx4080super | GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf | 48c39a25f640b90a70056e3c8f7308b66b9d635876e22c56f76516a09d2c9791 | 10,000 update records |

C3’s missing updates 1–2,500 are a known transferred-log limitation accepted by the researcher and sealed for this exact run in protocols/s1_c3_seed0_transferred_log_exception.json (SHA-256 f021ac0441f775d45d19a8109d47c0f45a0b0bbfc025bbd2df35be688e291cd2). No prefix was reconstructed, spliced, or rerun. Its checkpoints, model payloads, identities, and evaluation records passed their own checks, so the log gap does not invalidate those results. C5’s second start marker is recorded in its source log; the run still has update coverage through step 10,000.

### Full300 endpoint results at step 10,000

The behavioral columns below come from each run’s student Full300 aggregate over 300 OWT items (38,100 valid next-token targets). Sink S comes from the independently audited S4 baseline probe at the same endpoint.

| Condition | Clean CE | Sink S | Delete ΔCE | Delete flip fraction | Relocate ΔCE | Relocate flip fraction |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 4.40020 | 0.00958 | 0.02369 | 0.0572 | 0.03057 | 0.0669 |
| C1 | 3.99449 | 0.01718 | 0.05122 | 0.0875 | 0.07951 | 0.1247 |
| C2 | 3.99338 | 0.33197 | 0.37335 | 0.3302 | 0.36634 | 0.3293 |
| C3 | 3.99322 | 0.39537 | 0.34646 | 0.3139 | 0.42655 | 0.3426 |
| C4 | 3.97000 | 0.02760 | 0.07630 | 0.1162 | 0.11740 | 0.1689 |
| C5 | 3.97833 | 0.02424 | 0.06999 | 0.1115 | 0.10096 | 0.1543 |
| C6 | 3.99913 | 0.35750 | 0.22727 | 0.2445 | 0.34805 | 0.3108 |

The endpoint clean CE values are close among C1–C6 on this panel; their small ordering is descriptive and is not a significance result. C2/C3/C6 had high Sink S and larger delete/relocate effects than C0/C1/C5. C4 had low Sink S with moderate effects, while C6 had a high sink score but less delete sensitivity than C2/C3. The pattern therefore does not reduce to a one-dimensional “more sink means more useful” rule. S1 also retained the full trajectory and intervention records; the endpoint table is only a compact summary.

## S2 — Ordinary-pretraining development

No scientific Pythia trajectory was evaluated in the verified campaign. The designed study would compare native Pythia-160M/410M checkpoints across documented pretraining seeds, but there is no complete checkpoint inventory or S2 outcome series in the central result set. Any step-0 parity/engineering smoke is implementation evidence only, not a scientific observation for S2. No S2 result should be compared numerically with S1.

## S3 — Equal-head replication

S3 remains optional and disabled pending its separate budget approval. None of its 15 planned C0–C4 × three-seed training jobs was launched, so there are no S3 scientific results. The presence of S3 objective code or CPU tests does not constitute a replication.

## S4 — Positional anchors and parameter-level routes

S4 evaluated C0–C6 at steps 0, 100, 500, 2,000, and 10,000, seed0 only. It completed 35 student batteries and one fixed teacher reference: 108,000 item-probe records in 108,037 manifested files (2,273,573,249 uncompressed bytes). Each battery used 300 Full300 items and 10 registered probes. The independent audit verified the record/file hashes, unique coverage, run/checkpoint identities, panel identity, and absence of tensor payloads.

At step 10,000, selected results were:

| Condition | Baseline Sink S | Position 0→1 ΔS | Position 0→1 ΔCE | Position 0→1 flips | Remove absolute positions ΔCE | Remove-position flips |
|---|---:|---:|---:|---:|---:|---:|
| C0 | 0.00958 | −0.00411 | 0.01048 | 0.0387 | 1.54196 | 0.6198 |
| C1 | 0.01718 | −0.01184 | 0.03335 | 0.0753 | 2.54268 | 0.7503 |
| C2 | 0.33197 | −0.30389 | 0.34776 | 0.3276 | 3.01766 | 0.8162 |
| C3 | 0.39537 | −0.37607 | 0.32814 | 0.3127 | 3.11616 | 0.8146 |
| C4 | 0.02760 | −0.02093 | 0.05538 | 0.1044 | 3.22328 | 0.8215 |
| C5 | 0.02424 | −0.01807 | 0.04897 | 0.0989 | 3.20949 | 0.8254 |
| C6 | 0.35750 | −0.33201 | 0.21235 | 0.2408 | 2.61612 | 0.7574 |

The top-three model-local K-input-coordinate ablation produced ΔCE from 0.0016 to 0.0377 across conditions. Its five matched-cardinality random-coordinate controls had mean ΔCE between 0.00010 and 0.00019 and mean flip fractions between 0.0065 and 0.0080. The Q-bias removal produced ΔCE between 0.00034 and 0.00117. The layer-0 EPE transport probe produced absolute ΔCE no larger than 0.000073 and flip fractions below 0.003. These are the measured responses on the specified panel, not proof that unresponsive routes are absent from the model.

Position 0→1 both moved the sink score and affected output behavior, particularly in C2/C3/C6. Removing every absolute position embedding caused large loss and flip changes even for low-sink conditions; it is a broad positional intervention, not a sink-only manipulation. The probes do not exhaustively identify an alternative circuit and do not establish causal mediation. S4 is a designated-seed0 follow-up and cannot support across-training-seed reproducibility claims.

## S5 — Existing S1 numeric reaggregation

Under D26, S5 freshly verified and joined only the existing C1/C2/C5/C6 seed0 records while preserving their distinct original protocol roots. The comparison-critical invariants passed. The source contained 440 condition/panel/step rows; each of the four runs contributed 59,206 evaluation JSON files (58,984 item records and 222 aggregates). The resulting joins have no missing steps: 101 Dense64 steps and nine Full300 steps (0, 100, 250, 500, 1,000, 2,000, 5,000, 7,500, and 10,000). No model was loaded and no inference or training was run.

Full300 endpoint differences at step 10,000 (left condition minus right condition):

| Contrast | Clean CE difference | Full attention JSD difference | Sink S difference | Delete ΔCE difference | Relocate ΔCE difference |
|---|---:|---:|---:|---:|---:|
| C2 − C1 | −0.00111 | −0.17841 | +0.31478 | +0.32213 | +0.28683 |
| C5 − C1 | −0.01616 | −0.04113 | +0.00706 | +0.01877 | +0.02145 |
| C5 − C2 | −0.01505 | +0.13729 | −0.30772 | −0.30336 | −0.26537 |
| C6 − C1 | +0.00464 | −0.16199 | +0.34032 | +0.17605 | +0.26854 |

Thus, on this seed and panel, C2’s full attention map was more similar to the teacher than C1’s, with much higher sink mass and larger delete/relocate effects. C5 had nearly C1-like sink mass and intervention costs despite modestly lower clean CE. C6 also had lower full attention divergence than C1 and much higher sink mass, but a smaller delete ΔCE than C2. Original GPU identity varied across these four condition runs; three physical RTX 4080 SUPER UUIDs were present. Treat the numerical contrasts as descriptive and retain the hardware confound.

## S6 — Cross-domain and context robustness

S6 evaluated seed0 checkpoints 0, 500, 2,000, and 10,000 for all seven conditions. It covered 100 items each from SST-2 validation, GSM8K test, and HumanEval prompts, paired at max40 and max128 tokens. The final tree contained 50,400 item-operation records and 168 aggregates; the independent audit verified all 28 states, exact panel membership, finite records, and the full tree hashes.

At step 10,000, token-weighted results pooled across the three 100-item domains were:

| Condition | Clean CE max40→128 | Delete ΔCE max40→128 | Relocate ΔCE max40→128 |
|---|---:|---:|---:|
| C0 | 5.9081→5.3018 | 0.0475→0.0371 | 0.0888→0.0567 |
| C1 | 5.0365→4.2563 | 0.1291→0.0872 | 0.2441→0.1413 |
| C2 | 4.9269→4.1533 | 0.5407→0.3769 | 0.5109→0.3523 |
| C3 | 4.9751→4.2177 | 0.4239→0.2877 | 0.5201→0.3234 |
| C4 | 4.9133→4.1392 | 0.2176→0.1263 | 0.3187→0.1602 |
| C5 | 4.9263→4.1934 | 0.1997→0.1316 | 0.2354→0.1213 |
| C6 | 4.9536→4.1951 | 0.3907→0.2724 | 0.5785→0.3647 |

The pooled max40/max128 clean evaluations contained 9,907 and 19,418 valid targets respectively. Longer context reduced pooled clean CE in every condition. Pooled max128 deletion/relocation costs were largest in C2/C3/C6, consistent with the Full300 probes, but domain content and target counts differ and are not additional seed replicates.

The paired per-document summaries show that the context change in intervention cost (ΔCE at max128 minus ΔCE at max40) was small for SST-2: delete means ranged from −0.0051 to +0.0002 and relocation from −0.0061 to −0.0019 across conditions. For GSM8K the corresponding ranges were −0.0812 to −0.0114 and −0.0917 to −0.0143; for HumanEval they were −0.0887 to +0.0278 and −0.2420 to −0.0274. These are equal-item paired descriptive means. They show that intervention effects changed with context and domain; they are not tests of a population-level context effect.

The frozen sources were nyu-mll/glue revision bcdcba79d07bc864c1c254ccfcedcce55bcc9a8c (sst2/validation/sentence), openai/gsm8k revision 740312add88f781978c0658806c59bc2815b9866 (main/test/question), and openai/openai_humaneval revision 7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544 (openai_humaneval/test/prompt). D25 records SST-2 license status as upstream_ambiguous because the pinned GLUE metadata says “other”; GSM8K and HumanEval record the upstream-stated MIT metadata. This is provenance information, not a legal determination or redistribution grant. The same frozen documents were paired across context lengths; answers, completions, and test code were not model inputs. HumanEval code execution and optional 512/1024 contexts remained disabled.

## S7 — Sink-aware utility and attention decomposition

S7 reused the D26-selected C1/C2/C5/C6 seed0 records and added exactly 36 clean Full300 attention supplements: four conditions × nine steps (0, 100, 250, 500, 1,000, 2,000, 5,000, 7,500, 10,000), 300 ordered items per state. The frozen execution source was commit 9574a37348cf0e249457ead44ae4e75009f54e9d; inference used Adrita-PC RTX 4080 SUPER GPU-2a5c25d0-1f73-919b-fd8b-f6f0df709aaf. It did not retrain S1 or rerun delete/relocate interventions. The independent audit found all 36 unique states, 300/300 items each, the approved S7 lock and D24 identities, finite values, and maximum decomposition closure error 3.400058e−16.

### Utility and dependence at step 10,000

| Condition | Clean CE | Teacher KL | Teacher top-1 agreement | Delete ΔCE | Relocate ΔCE |
|---|---:|---:|---:|---:|---:|
| C1 | 3.99449 | 1.13906 | 0.48724 | 0.05122 | 0.07951 |
| C2 | 3.99338 | 1.13300 | 0.48672 | 0.37335 | 0.36634 |
| C5 | 3.97833 | 1.12387 | 0.48890 | 0.06999 | 0.10096 |
| C6 | 3.99913 | 1.14540 | 0.48344 | 0.22727 | 0.34805 |

Clean CE and teacher matching were close on Full300. C5 had the lowest listed clean CE and teacher KL, while C6 had the highest; these are single-seed descriptive values without a declared practical-equivalence margin. C2 and C6 had larger delete/relocate effects than C1/C5, so structural similarity, clean utility, and inference dependence do not rank the recipes identically.

### Teacher–student attention decomposition at step 10,000

| Condition | Full JSD (nats) | Sink-mass JSD (nats) | Shape JSD (nats) |
|---|---:|---:|---:|
| C1 | 0.22881 | 0.16069 | 0.06812 |
| C2 | 0.05040 | 0.00968 | 0.04072 |
| C5 | 0.18768 | 0.14604 | 0.04164 |
| C6 | 0.06683 | 0.00732 | 0.05952 |

The corresponding endpoint contrasts were:

| Contrast | Full JSD difference | Sink-mass JSD difference | Shape JSD difference |
|---|---:|---:|---:|
| C2 − C1 | −0.17841 | −0.15101 | −0.02740 |
| C5 − C2 | +0.13727 | +0.13636 | +0.00092 |
| C6 − C1 | −0.16198 | −0.15337 | −0.00861 |

For C2 and C6 relative to C1, most of the reduced full-map divergence was in the sink-mass component. C5 and C2 had similar shape divergence at the endpoint, while C5 retained substantially higher sink-mass divergence. This is a mathematical decomposition of the observed attention divergence; it does not establish that sink matching causes the behavioral outcome.

Across the 101-point Dense64 trajectory, prespecified AUC mean differences were:

| Contrast | Sink S AUC mean difference | Full JSD AUC mean difference | Delete ΔCE AUC mean difference |
|---|---:|---:|---:|
| C2 − C1 | +0.31732 | −0.19704 | +0.15454 |
| C5 − C1 | +0.00469 | −0.04689 | +0.00626 |
| C5 − C2 | −0.31262 | +0.15015 | −0.14828 |
| C6 − C1 | +0.34030 | −0.17113 | +0.08736 |

Taken together, the measured answers remain separate: C2/C6 transfer more of the teacher’s first-position mass pattern; clean next-token utility is similar across the four recipes; and C2/C6 show greater inference sensitivity to sink deletion/relocation than C1/C5. Because this is one training seed, the conditions have a physical-GPU confound, and the measures are not a randomized mediation design, the results do not support across-seed reproducibility or causal-mediation claims.

## Limitations and interpretation boundary

1. The verified primary S1 results and all new checkpoint-dependent S4/S6/S7 evaluations summarized here are training seed0 only. D24 prospectively restricted those follow-ups to seed0. Items, checkpoints, domains, inference hardware, and probes do not create independent training seeds.
2. Three physical RTX 4080 SUPER training UUIDs occur in the S1/S5 quartet, and C4 used an RTX 3090. These condition/device differences limit clean attribution to objective choice.
3. The main S1 horizon is 10,000 optimizer updates, not full pretraining convergence. Effects are limited to the specified models, panels, scopes, contexts, interventions, precision, and horizon.
4. S4’s absolute-position removal is broad. Its large output changes do not isolate a sink mechanism. A small effect from Q-bias or EPE transport does not show that no other route exists.
5. S6 measures next-token language modeling over selected task text. It does not measure benchmark task accuracy, GSM8K arithmetic correctness, or HumanEval program correctness.
6. S2 and S3 have no scientific outcome series in this report. S4/S5/S6/S7 completion does not imply Stage09 was run.
7. No p-values, hypothesis tests, across-seed variance, equivalence margins, noninferiority claims, favorable-checkpoint selection, or mediation claims were introduced.

## Completion discrepancies retained in the record

- S4’s first attempt failed a configuration-byte integrity check before any model forward pass and wrote zero records. Attempt 2 used a narrow raw-config hash verifier correction, completed, and passed the independent audit. The enclosing command session returned exit code 1 despite the runner’s COMPLETE event and sealed audit; that wrapper discrepancy remains disclosed.
- S6’s runner later returned a stale aggregate-count failure expecting 504 aggregates. The completed production design requires 168 aggregates (28 states × three domains × two contexts); all 168 and all 50,400 item-operation records passed the independent audit. The failure receipt is preserved and not erased.
- C3’s accepted S1 training-log prefix gap is limited to absent updates 1–2,500 in the transferred log. The original training trace was not reconstructed or rewritten.

## Source records and audit trail

Primary local source tree: D:\KD-SINK-central\runs. S1 inventory: D:\KD-SINK-central\manifests\S1_MASTER_INDEX.json (file SHA-256 06434301ce3c8d8a0e520c9b9906fcf21cd9d9e8383da41292c13340297b6591). The index verifies 7/7 expected seed0 primary run archives and explicitly says that transfer coverage is not a multi-seed scientific coverage claim.

| Evidence | Location | File SHA-256 or sealed identity |
|---|---|---|
| Stage08 S4/S5/S6 execution record | reports/stage08_scientific_execution_20261005.json | d0fc9960180d0667c111e322261e79b8803c848bc5e20d44edc12eeea73dacac |
| S4 independent audit receipt | D:\KD-SINK-central\analysis\stage08_scientific_20261005\audits\S4_FINAL_INDEPENDENT_AUDIT.json | file 9412d951974b7e286f245e9146e19fe2a0e8a281787a295867a78f7675b64ef3; sealed audit 18a3a470e42b773850441693c81fe6d184de00bccd1c6e0c860c4c472086bd243 |
| S5 sealed reaggregation audit | D:\KD-SINK-central\analysis\stage08_scientific_20261005\S5\STAGE08_S5_AUDIT.json | 09b3663cda822c7b5b3b697bbc3a44c4849b63e85f24b0a98abfce06c99dfa7c |
| S5 Full300 joined output | D:\KD-SINK-central\analysis\stage08_scientific_20261005\S5\S5_JOINED_OWT_FULL300.json | 659beb093b84d00201391fa4723df31483e15ed34c93b9f143cd71641128d892 |
| S6 independent audit receipt | D:\KD-SINK-central\analysis\stage08_scientific_20261005\audits\S6_FINAL_INDEPENDENT_AUDIT.json | file 74df9b921909d5fe41700cd5b9b67f583243d0b689579a7bea222fa35b75ecb9; sealed audit 5a66ce30937cd3905140dc95dcc9be1729210d029060f0c08a15778b6179e1f2 |
| S6 run manifest | D:\KD-SINK-central\analysis\stage08_scientific_20261005\S6\S6_RUN_MANIFEST.json | 64a268fcbccb5422c97c392bfc534e8facec0f5185ecc7c20f05c6cfc37c20eb |
| S7 campaign record | reports/s7_campaign_20261006.json | bc90649f1af0a08f163b0dab32eb0fa537e32f4ffa78f2f1face9435290793a9 |
| S7 final analysis | D:\KD-SINK-central\analysis\S7_scientific_20261006\analysis\final_d40afe0\S7_ANALYSIS.json | 58b5f72249f24cb4e602a0a4596752093984073b36e1196334f4a7254197a79c |
| S7 independent audit | D:\KD-SINK-central\analysis\S7_scientific_20261006\audit\independent_final_audit.json | d24ccd2c913a8b7f54edb2f7ddf3d42a44bd225428a3fb8e0d02613ab2d97912 |
| Prospective D24 follow-up policy | protocols/s1_researcher_amendment_d24_seed0_followups_20261004.json | 46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6 |
| Prospective D25 S6 data/license policy | protocols/s1_researcher_amendment_d25_s6_external_datasets_20261004.json | 6c42ffebf8fa6b1a49787ec7468040939b6fd360c6132e2df10bf2b212780704 |
| Prospective D26 S5 compatibility rule | protocols/s1_researcher_amendment_d26_s5_mixed_roots_20261005.json | 888b21509000b570c83c2574f5a6bfbc3d1ec2dc197e88d841d085bac222f182 |
| Approved S7 analysis lock | protocols/s7_utility_analysis_approved.json | 696c9fc206efdb60b53120c411cc3089306048733829fc4762db41caf92917b7 |
| S7 artifacts archive | D:\KD-SINK-central\archives\S7_scientific_20261006.zip | 21,259,437 bytes; SHA-256 04ef7ac6d4cb0474cbe2bc7d104fc66938ac09057a5aaa9f2221ecb151262a18 |

The S7 ZIP was previously verified with ZIP CRC testing, internal file-hash comparison, extraction/hash verification, and matching outer sidecar. It contains 268 files and no model/checkpoint tensor payloads.

## Final statement

The completed studies support a bounded, descriptive account of how these seven seed0 S1 recipes behaved over the 10,000-update horizon, how selected interventions changed their outputs, and how the C1/C2/C5/C6 results decompose across sink mass and non-sink shape. They do not establish multi-seed reproducibility, a uniquely identified circuit, or benchmark task accuracy on S6 prompts. S2 and S3 remain without scientific results.
