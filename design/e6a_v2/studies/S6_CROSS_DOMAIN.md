# S6 - Domain and context robustness

EVALUATION ONLY. Retained S1 states0,500,2000,10000,all conditions/seeds; optional approved S3 endpoints. Additional retained states require recorded budget expansion, not favorable-result selection. No new training or tuning.

Domain panel:100 SST-2 validation sentences,100 GSM8K test questions,100 HumanEval prompts,max128 tokens,exact fields/manifests. No answers/completions/test code in inputs and no generated code execution. Report each domain and an explicitly equal-item pooled summary. These are language-modeling text probes, NOT classification accuracy,math-answer accuracy or code pass@k.

Run clean behavior/structure and all-layer delete/relocate effects. Compare max40 and max128 renderings of the same documents to test the earlier short-context limitation. They have different target counts/masks; no assumption that a null at40 remains null at128.

Optional approved long-context check:300 OWT windows at512 and1024,final10k states first. Verify model positional limit and do not extrapolate embeddings. These contexts exceed128-token student training; report clean degradation and valid-target counts alongside causal effects. Profile memory separately because attention grows quadratically. Use a separately validated fixed evaluation batch, not altered training batch. OOM must not silently truncate sequences.

## Comparability
Keep integer teacher/student scopes and numerical policy explicit per length/corpus. Preserve q0 exception and slot semantics. Token-weight variable-length CE; distinguish item/domain-weighted summaries. Differences may reflect length/content/clean performance, not solely sink mechanism. Include all registered domains and contrary findings; domains are not training-seed replicates.

Tests: deterministic selection/field extraction,no answer leakage,minimum2 real tokens,right padding,exact40/128 document matching,correct pooling,1024bound,no OOM truncation,edit restoration and no external code execution.

Journal: IMPLEMENTATION_NOTES_BY_CLAUDE_S6.md. Stage08; long-context production separately approved.
