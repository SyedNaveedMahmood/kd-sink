# S1 condition/device execution and memory contract

Upstream's historical13.8GiB optimizer microstep [UP1] does not establish memory for the rewritten REL objective, FP32 master states or dense evaluator. No new hardware fit or wallclock measurement is supplied here.

## Corrected S1 eligibility policy and campaign

The corrected 2026-09-28 amendment requires **seven unique C0-C6 seed0 jobs**, each with a fixed device role in a reviewed plan. RTX3090 is policy-eligible for all seven; RTX4080 SUPER may run a condition only after real sequence-128 production-shape full-cycle/headroom evidence and a final lock. Existing 4080 evidence covers C0/C1/C2/C5/C6. C3 is `pending_4080_profile` and must not be assigned there yet. C4/REL remains3090-only. Optional complete seed1 and seed2 campaigns each add seven unique condition/seed jobs, separately invoked. No mid-run GPU migration or automatic retuning. S3 remains optional and its REL role is3090.

An all-3090 seven-job seed0 draft and a mixed nine-physical-job draft with C1/C2 3090 bridges are examples, not approved launch plans. Primary cross-condition contrasts should use the same device where practical. If a necessary contrast crosses roles, the reviewed plan must name an explicit bridge/replica or record that hardware confound. Replicas are not independent training seeds. The final hardware lock contains a condition/device/UUID eligibility matrix; its approved entries are backed by complete measured profiles.

## Profile once, freeze a COMMON batch
Independent jobs must not independently select different paired schedules. Profile every condition/device actually selected by the reviewed plan using actual model sizes, context128, precision, optimizer and feature capture. Prior3090 C1=16 and C2/C3/C4=8 safe maxima leave C0/C5/C6 unmeasured there. The 4080 full-cycle candidate maxima within the earlier bound are C0=8 and C1/C2/C5/C6=4; C3 has no 4080 profile. The prior two-device4x16 result remains an engineering candidate, not a final production lock. No common production microbatch or accumulation is locked.

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
Measure a representative100-update engineering pilot including dense evaluation and full save. Extrapolate separately by objective with uncertainty; REL/evaluation may dominate. Estimate the seven mandatory jobs, optional seven-job seed1/seed2 campaigns, optional15-run S3, Pythia downloads, retained-weight/full-resume disk and RAM. Do not assert the old5-6-hour figure applies. Feasibility is not conditioned on reproducing the desired dissociation.
