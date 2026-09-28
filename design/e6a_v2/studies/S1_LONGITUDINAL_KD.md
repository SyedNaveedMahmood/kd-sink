# S1 - Longitudinal sink inheritance

PRIMARY STUDY. Frozen GPT-2-large (36layers,20heads,1280width) -> randomly initialized GPT-2-medium (24layers,16heads,1024width). C0-C6 at mandatory seed0, each launched separately for10,000 optimizer updates on RTX3090. Complete seed1/seed2 campaigns are optional and separately invoked. One seed permits descriptive within-seed contrasts only, not across-seed variance or reproducibility claims. Model/tokenizer/weights/config require verified immutable artifacts. No acceptance test presumes the old finding persists.

## Questions and contrasts
Does attention supervision accelerate first-position sink appearance compared with CE-only and LogitKD? Does predictive usefulness or output sensitivity develop later? Do probability and Q/K/V-relation objectives change that relationship? All C0-C6 paired contrasts for a given seed use RTX3090, including S5 full/NoSink/SinkOnly comparisons.

Use exact OBJECTIVES definitions, not interchangeable historical C2 meanings. Shared initialization/order/effective batches within seed, one common microbatch plan, no per-condition hyperparameter tuning/early stopping. Freeze protocol,environment,data,calibration and hardware before production.

## Measurement and storage
At0 and every100 updates run dense64 clean/delete/relocate: CE/PPL,teacherKL/agreement,sink/profile,common attention similarities,DeltaCE/relativeDeltaCE,self-KL,absolute target log change,flips and accuracy. All-layer student scope versus mapped teacher scope.101 dense points per run are repeated measurements, not independent replicates.

Retain analysis states0,100,250,500,1000,2000,5000,7500,10000 and run full300 evaluation there. Step250 is extra, not a replacement for200/300. Clean NLL2000 at0/10k. Rolling full recovery every500 and protected full final checkpoint preserve future continuation.

S1 has seven mandatory seed0 jobs on3090. Optional seed1 and seed2 add seven jobs each, for at most21 unique condition/seed combinations. No hardware replicas are planned. Approve the additional resource cost before any optional campaign.

## Launch and reporting gates
All C0-C6 unit/integration suites, exact resume and evaluation-RNG parity, actual probability intervention checks, no-Upstream independence, calibrated losses and realGPU common-plan smoke must pass. Teacher preflight determines whether sink/probe comparisons are informative; it must not become a search for a student result that supports the old title.

Primary endpoint is10k with individual-seed and complete longitudinal evidence.10k is81.92M input tokens atcontext128, not full pretraining convergence. Increased function is a valid central finding. Nondetection is bounded to the horizon,operations,contexts and uncertainty tested. Later extension requires approved comparison coverage and full-state continuation, not a new warmup or selective favorable-condition extension.

Journal: `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md`. Capability stages00-06, analysis/release09.
