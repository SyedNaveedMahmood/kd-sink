# Source audit and research rationale

Research date: September27,2026. Separate source-supported statements from proposals in DECISIONS. The user's pasted review notes are the revision requirements; original reviewer reports were not separately verified. The manuscript is evidence about the old experiment, not an instruction to preserve its conclusions.

## Manuscript
[PAPER] A sink without the plumbing: Attention distillation copies the sink pattern, not its function. (n.d.). [Unpublished manuscript supplied as SinkWithoutPlumbing(2).pdf; no public URL/DOI supplied].

Relevant: Section3 setup/losses; Section4 pattern/circuit/function; Limitations p5; AppendixA p8; AppendixC pp8/10; AppendixD pp8-10; AppendixE pp10-11. The main study is500steps; NoSink and longer-horizon/second-pair evidence are single-seed; SinkOnly was not run. The revised protocol tests these gaps rather than assuming persistence. AppendixA describes3090 while the environment table lists4080SUPER; neither establishes resource requirements for the new code. The manuscript's Jin/trainable-alignment wording differs from the inspected cosine-soft AMAD-style implementation.

## Repository and actual reading scope
[UP0] SyedNaveedMahmood. (2026). kd-sink [Source code, commit96a80e1e7ad581728c8befd697547f9b3f3a2e85]. GitHub. https://github.com/SyedNaveedMahmood/kd-sink/tree/96a80e1e7ad581728c8befd697547f9b3f3a2e85

The root contains uppercase Upstream/. Audited sources:
- [UP1] Upstream/transformation_inheritance/configs/e6a_gpt2_large_medium_preregistration.yaml, full; blobd6bd9574639653b2fc3031339201360db7045721. Model dimensions,layer map,mean-head JSD,historical13.8GiB measurement. Not evidence of new REL memory use.
- [UP2] Upstream/transformation_inheritance/configs/e6a_gpt2_large_medium_logit_attention_kd.yaml, full; blobc91e4533a55a4982741ac3281a55957f9cb05ddc. Optimizer,data and checkpoint ancestry.
- [UP3] Upstream/transformation_inheritance/train_distillation.py, opening/config/loss sections and complete amad_js_attn_loss section (source430-640 fetched); blob627ea31489805781de1b0cc4c115a370530a025f. Historical64.5x JSD reduction issue documented. Per-example cosine-soft weights are differentiable but have no separate parameterized adapter. New pre-dropout targets deliberately amend legacy training-time attention-return semantics. This is a targeted audit,not a claim to have read every line of the large trainer.
- [UP4] Upstream/common/residual_sink_analysis_legacy.py, source1-240; blob85f78ff294c0b19871c204c0424e86152893a936. EPE construction,coordinate selection and first-MLP transport. New per-example batched transport/top3 rule are explicit adaptations.

Located reuse candidates,not represented as fully audited: common/block_corpus_cache.py,datasets_loader.py,depth_band.py,fingerprint_runner.py,inheritance_metrics.py,nnsight_engine.py; transformation_inheritance/run_pilot_parity.py,evaluate_transformation.py,aggregate_transformation.py. The coding agent reads each actual function before borrowing. No transitive Upstream imports. Verify licenses; absence of a license is not permission to invent one.

## Primary references and their use
[R1] Jiao, X., Yin, Y., Shang, L., Jiang, X., Chen, X., Li, L., Wang, F., & Liu, Q. (2020). TinyBERT: Distilling BERT for natural language understanding. In Findings of the Association for Computational Linguistics: EMNLP2020 (pp.4163-4174). Association for Computational Linguistics. https://doi.org/10.18653/v1/2020.findings-emnlp.372

Section3.1/equation7 targets unnormalized pre-softmax attention. New C3 is explicitly a post-softmax head-mean probability-MSE adaptation. PDF: https://aclanthology.org/2020.findings-emnlp.372.pdf

[R2] Wang, W., Bao, H., Huang, S., Dong, L., & Wei, F. (2021). MiniLMv2: Multi-head self-attention relation distillation for compressing pretrained transformers. In Findings of the Association for Computational Linguistics: ACL-IJCNLP2021 (pp.2140-2151). Association for Computational Linguistics. https://doi.org/10.18653/v1/2021.findings-acl.188

Section3/equations5-8 define relations by concatenating native heads and repartitioning into relation heads; QQ,KK,VV are used. Causal masking and all-mapped-layer supervision here are explicit adaptations. PDF: https://aclanthology.org/2021.findings-acl.188.pdf

[R3] Jin, H., Son, S., Park, J., Kim, Y., Noh, H., & Lee, Y. (2024). Align-to-distill: Trainable attention alignment for knowledge distillation in neural machine translation [Preprint]. arXiv. https://doi.org/10.48550/arXiv.2403.01479

A2D's parameterized alignment differs from UP3's cosine-soft construction. Official implementation: https://github.com/ncsoft/Align-to-Distill . Do not attribute an unimplemented learned adapter to C2.

[R4] EleutherAI. (n.d.). Pythia: Interpreting transformers across time and scale [Repository/model documentation]. Retrieved September27,2026, from https://github.com/EleutherAI/pythia/blob/main/README.md

Official metadata specifies154 native checkpoints,logarithmic early steps then1000-step spacing through143000; standard un-suffixed seed1234 and extra training seeds for160M/410M;2,097,152 tokens/update. Verify each branch revision before execution. Model cards: https://huggingface.co/EleutherAI/pythia-160m-seed1 and https://huggingface.co/EleutherAI/pythia-410m-seed1 . Do not mix v0/deduped/standard runs. Cite the Pythia and PolyPythias research papers with verified metadata in the eventual manuscript as well as exact artifact revisions.

[R5] PyTorch contributors. (n.d.). Reproducibility [Documentation]. Retrieved September27,2026, from https://docs.pytorch.org/docs/stable/notes/randomness.html

Identical seeds do not ensure equality across releases/platforms/devices. Hardware overlap diagnoses discrepancies; it does not prove all objective-by-hardware interactions absent.

[R6] PyTorch contributors. (n.d.). Automatic mixed precision examples [Documentation]. Retrieved September27,2026, from https://docs.pytorch.org/docs/stable/notes/amp_examples.html

Accumulation/clipping/optimizer boundaries respect effective batches. Proposed precision is BF16 autocast with FP32 student/master/optimizer state,not blanket BF16 state.

[R7] PyTorch contributors. (n.d.). torch.utils.checkpoint [Documentation]. Retrieved September27,2026, from https://docs.pytorch.org/docs/stable/checkpoint.html

Explicit non-reentrant checkpointing requires gradient/RNG parity. Do not collect auxiliary hook side effects twice during recomputation.

[R8] Hugging Face. (n.d.). Perplexity of fixed-length models [Transformers documentation]. Retrieved September27,2026, from https://huggingface.co/docs/transformers/main/en/perplexity

Tokenization and context affect PPL. Label fixed-block PPL and do not compare cross-tokenizer values as interchangeable performance.

[R9] Zhao, T., Singh, K. Y., Appalaraju, S., Tang, P., Mahadevan, V., Manmatha, R., & Wu, Y. N. (2024). No head left behind: Multi-head alignment distillation for transformers. Proceedings of the AAAI Conference on Artificial Intelligence,38(7),7514-7524. https://doi.org/10.1609/aaai.v38i7.28583

Primary AMAD source for cosine-soft alignment. New C2 uses this style with a separately specified JSD target; it does not claim every published AMAD detail is reproduced.

## Reuse record
Record source path,commit/blob,license,function/lines,destination,literal-or-conceptual reuse,intentional changes,independent tests,reviewer,milestone commit. Keep fixtures/attribution outside Upstream. Historical result tables are not acceptance expectations for a new architecture. Unknown memory,scales,practical margins and budget remain measured/approval gates,not facts supplied by citations.
