# S2 - Ordinary pretraining development

EXTERNAL OBSERVATIONAL BASELINE; evaluation only. S1C0 remains the controlled non-distillation baseline. S2 asks whether similar structural/sensitivity/predictive development appears during independent pretraining, not whether distillation is its unique cause.

## Proposed inventory, approval D12
P1=EleutherAI/pythia-160m; P2=EleutherAI/pythia-410m. For each, standard un-suffixed training seed1234 and separately pretrained -seed1/-seed2 variants. Public documentation describes these seed series [R4]. A new evaluation RNG seed is not a new pretraining replicate.

Expected native steps:0,1,2,4,8,16,32,64,128,256,512, then1000..143000 in1000 increments:154 per trajectory if verified available. Two sizes x3 seeds means up to924 checkpoint evaluations, not a free addition. INVENTORY FIRST: resolve each requested branch to immutable commit, validate config/tokenizer/weight hashes and save availability. No latest-weight fallback, invented intermediate state or mixing standard/deduped/v0 runs. If coverage differs, approve an intersection/incomplete-coverage policy before outcome inspection.

## Measurements
Use tested GPT-NeoX adapter preserving rotary positions/native projection layout. Same clean/delete/relocate semantics over all native layers. Same frozen raw documents where appropriate, independently Pythia-tokenized manifests. Dense64 at every verified native checkpoint; proposed full300 anchors0,128,1000,10000,50000,100000,143000 where available.

Measure S,topology,cleanNLL/PPL,DeltaCE,self-KL,absolute log change,flips. TeacherKL not applicable. GPT-2-specific absolute-position/EPE fingerprints are not applicable. Native steps and documented pretraining tokens are both recorded; standard Pythia uses2,097,152 tokens/update [R4]. Equal numerical step10000 does not imply equal S1/S2 training exposure. Cross-tokenizer PPL comparisons are not direct performance comparisons.

## Resource behavior
Primitive command takes one explicit size/training-seed/checkpoint. Multi-checkpoint iteration requires a frozen inventory and operator opt-in, cache/download byte cap and resume from verified records. Load one checkpoint at a time, persist outcomes, release memory. Scoped cache cleanup only after successful verified evaluation and explicit policy; never delete global user caches. No optimizer state and no full-Pile download needed.

## Tests and interpretation
Local fake inventories exercise missing branches,seed1234 mapping,no-latest fallback,cache keys and restart. Actual small network/GPU parity is explicit opt-in and recorded blocked when unavailable. Test step0 and later models, correct token counts, causal masks,rotary positions,native/no-op output parity. Do not automatically fetch924 states during a coding session.

Architecture,data,tokenizer,schedule and scale differ from S1. Present within-family trajectories and qualitative developmental comparisons, not a randomized causal attribution. Missing observations remain missing. Cite Pythia/PolyPythias appropriately in the eventual manuscript using verified source metadata.

Journal: IMPLEMENTATION_NOTES_BY_CLAUDE_S2.md. Stage07 after common evaluator.
