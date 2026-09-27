# S1 - Longitudinal sink inheritance

PRIMARY STUDY. Frozen GPT-2-large (36layers,20heads,1280width) -> randomly initialized GPT-2-medium (24layers,16heads,1024width). C0-C6, seeds0/1/2 explicitly launched separately,10,000 optimizer updates. Model/tokenizer/weights/config are verified immutable artifacts. No acceptance test presumes the old finding persists.

## Questions and contrasts
Does attention supervision accelerate first-position sink appearance compared with CE-only and LogitKD? Does predictive usefulness or output sensitivity develop later? Do probability and Q/K/V-relation objectives change that relationship? C2-C1 and C1-C0 are paired within4080; C2/C3/C4 versusC1 and each other within3090. S5 analyzes full/NoSink/SinkOnly. The approved all-3090 alternative preserves same-device comparisons.

Use exact OBJECTIVES definitions, not interchangeable historical C2 meanings. Shared initialization/order/effective batches within seed, one common microbatch plan, no per-condition hyperparameter tuning/early stopping. Freeze protocol,environment,data,calibration and hardware before production.

## Measurement and storage
At0 and every100 updates run dense64 clean/delete/relocate: CE/PPL,teacherKL/agreement,sink/profile,common attention similarities,DeltaCE/relativeDeltaCE,self-KL,absolute target log change,flips and accuracy. All-layer student scope versus mapped teacher scope.101 dense points per run are repeated measurements, not independent replicates.

Retain analysis states0,100,250,500,1000,2000,5000,7500,10000 and run full300 evaluation there. Step250 is extra, not a replacement for200/300. Clean NLL2000 at0/10k. Rolling full recovery every500 and protected full final checkpoint preserve future continuation.

S1 has21 unique condition/seed combinations. The recommended two-GPU overlap plan has27 physical runs; duplicated C1/C2 devices are NOT extra seeds. Approve its resource cost explicitly.

## Launch and reporting gates
All C0-C6 unit/integration suites, exact resume and evaluation-RNG parity, actual probability intervention checks, no-Upstream independence, calibrated losses and realGPU common-plan smoke must pass. Teacher preflight determines whether sink/probe comparisons are informative; it must not become a search for a student result that supports the old title.

Primary endpoint is10k with individual-seed and complete longitudinal evidence.10k is81.92M input tokens atcontext128, not full pretraining convergence. Increased function is a valid central finding. Nondetection is bounded to the horizon,operations,contexts and uncertainty tested. Later extension requires approved comparison coverage and full-state continuation, not a new warmup or selective favorable-condition extension.

Journal: `implementation_notes/IMPLEMENTATION_NOTES_BY_CLAUDE_S1.md`. Capability stages00-06, analysis/release09.
