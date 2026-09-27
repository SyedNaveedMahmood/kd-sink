# Two-GPU execution and memory contract

Upstream's historical13.8GiB optimizer microstep [UP1] does not establish memory for the rewritten REL objective, FP32 master states or dense evaluator. No new hardware fit or wallclock measurement is supplied here.

## Recommended comparison blocks: approval D13
| Block | Device | Conditions, each seeds0/1/2 | Runs |
|---|---|---|---:|
| A | RTX3090,24GB | C1,C2,C3,C4 |12|
| B | RTX4080SUPER,16GB | C0,C1,C2,C5,C6 |15|

This is21 unique condition/seed combinations and27 physical runs. C1/C2 hardware repeats are not extra training seeds. Compare REL with methods trained on its own GPU; compare full/NoSink/SinkOnly within their own block. One duplicated reference does not eliminate all possible hardware-by-objective interactions.

Alternative: all21 S1 jobs on3090 and use4080 for S2/eligible S3/evaluation. A21-run split with onlyREL on3090 is explicitly hardware-confounded; researcher approval must acknowledge that limitation. Do not silently inflate the budget to27. Production-size REL, including profiling and S3 REL, is allowed only on the approved3090; CPU synthetic tests remain allowed.

## Profile once, freeze a COMMON batch
Independent jobs must not independently select different paired schedules. First profile all eligible assigned condition/device combinations using actual model sizes, context128, precision, optimizer and feature capture. Do not profile production REL on4080.

Try divisors of64 largest-first:64,32,16,8,4,2,1. Each candidate runs in a fresh isolated subprocess to avoid OOM allocator contamination. Complete warmup and at least two optimizer updates so Adam states exist; include scheduled evaluation with actual resident-model policy and a checkpoint serialization pass. Synchronize; record allocated/reserved peaks, free/total VRAM, other device processes, host RAM and walltime. Retain failed-candidate evidence. Forward-only fit is insufficient.

Choose the largest candidate passing EVERY required profile, equivalently the minimum of their safe maxima. Set common accumulation=64/microbatch and freeze a single hardware-plan hash for all paired S1 seeds/conditions/devices. Spare3090 memory does not authorize different microbatching. A separate architecture pair may have its own common plan.

Proposed headroom is max(1.5GiB,10% physical VRAM), approved and tested under the actual desktop/display workload. Both free memory and peak reservation matter. Evaluation may use a separately validated smaller fixed batch because it does not affect optimizer batching; its reductions must remain equivalent.

If microbatch1 fails, validate common activation-checkpointing/chunk policy and reprofile BEFORE production. Do not silently shorten context, reduce relation heads, drop Q/K/V types/layers, quantize teacher or adopt an8-bit optimizer. These change the numerical/scientific design. Missing profiles block production. CPU smoke mode remains available.

A mid-run OOM safely aborts; no automatic shrink/retry with changed accumulation. Diagnose competing processes and preserve evidence. A protocol-changing fix creates a new lock and reruns affected paired comparisons. One training process per GPU; no DDP/FSDP/DeepSpeed needed.

## Objective-preserving optimization
No-grad teacher; capture only mapped layers; discard unused attention; chunk Q/K/V relations and full-vocabulary metrics with exact reductions; pure checkpointed blocks with gradient parity; avoid retaining previous autograd graphs and per-layer .item() synchronization. Do not collect hook side effects twice during recomputation. Do not retain full clean/intervened vocabulary outputs for all items simultaneously. Cache keys include scientific and numerical inputs.

Log GPU name/UUID/VRAM, driver/runtime, exact library versions, role, parameter/compute/state precision, microbatch, accumulation, input/target counts, measured peaks and plan hash. Device eligibility uses the approved UUID/model role, not an ambiguous GPU0 across PCs.

## Terminal progress
Use tqdm rather than a new dashboard dependency. Startup banner: study/condition/method/seed/run ID, GPU/VRAM, microbatch x accumulation, effective batch, precision, total updates, checkpoint path and protocol hash. Main progress advances ONCE per optimizer update and shows elapsed time, updates/sec, tokens/sec, ETA, CE/KD/aux/total, LR and VRAM. Evaluation has a separate panel/intervention/scope/item progress bar.

Distinguish training-only ETA from total ETA including measured evaluation/save costs. Show estimating until enough observations exist. Use tqdm.write for saves/resumes/errors and persist UTC JSONL events. Non-TTY mode disables animated bars and emits bounded-frequency text (proposed every20 updates). Logging/progress must not change RNG. Restore cumulative counters on resume; no prompts/secrets in routine output.

## Budget gate
Measure a representative100-update engineering pilot including dense evaluation and full save. Extrapolate separately by objective with uncertainty; REL/evaluation may dominate. Estimate21/27-run and optional15-run S3 cost, Pythia downloads, retained-weight/full-resume disk and RAM. Do not assert the old5-6-hour figure applies. Feasibility is not conditioned on reproducing the desired dissociation.
