# Training lifecycle and exact continuation

## Independent command
Require one study, condition, seed, approved protocol, hardware lock and unique run directory. Future interface, not implemented code:

```sh
python -m sinklab train --study S1 --condition C4 --seed 0 --protocol protocol.lock.json --hardware-plan hardware.lock.json --run-dir runs/S1/C4/seed0/3090
```

No seed list or all-conditions default. Optional batch launching requires an explicit reviewed plan and opt-in. Tiny local smoke mode is clearly labeled engineering evidence, not a scientific run.

## Update semantics
Teacher frozen/eval/no gradients; student train mode. Approved numerical policy: FP32 student/master parameters and Adam states with BF16 autocast, FP32 probability/loss reductions. Do not cast the entire optimizer/student toBF16 just to fit memory. Frozen teacher BF16 storage is allowed; precision diagnostics reload original FP32 weights. Record parameter, compute and state dtypes separately. BF16 normally requires no scaler; any scaler used must be persisted and tested.

AdamW:lr5e-4, betas(.9,.95), eps1e-8, weight decay.1, clip norm1. Decay matrix weights, exclude biases/norm scales, record groups and avoid duplicate tied parameters. Student dropout follows approved config; teacher dropout off; auxiliary attention probabilities are before dropout.

For each update get exactly64 ordered blocks, split into the fixed common microbatch, zero gradients once, accumulate losses weighted by their share of the full effective-batch denominator, clip once, optimizer.step once, scheduler.step once. At128 tokens there are8192 inputs but8128 shifted targets. No per-microbatch clipping/stepping or double division by accumulation. Inactive objectives are not computed for training.

Initialize Python/NumPy/Torch with the seed and reuse the exact CPU initialization artifact. Save backend/determinism flags. Matching seed and schedule does not guarantee cross-device/library bit equality [R5]. Hardware replicas are diagnostic, not new independent seeds.

## Absolute clock and extension
`optimizer_step` counts successfully completed updates. Save/evaluate step0 before any update. Dense evaluation follows completed multiples of100. `--stop-after` stops without changing the registered10k schedule; it is not a replacement for max_steps.

Before update u>=1 set LR: peak*u/500 for u<=500; floor+(peak-floor)*(1+cos(pi*(u-500)/9500))/2 for500<u<=10000; floor thereafter, with floor=.1*peak. Advance scheduler state once after the optimizer update to prepare the next update. Test u=1,500,501,10000,10001 explicitly.

A later extension requires a stable explicit extension/lineage ID and loads the checksum-verified, protected `final-010000` full state. Do not reset moments, RNG, data cursor, scheduler clock or warmup; do not recompute old LRs with a newly enlarged cosine denominator. Every extension update uses the constant 10% peak-LR floor. Rolling full recovery remains every500 absolute updates; dense evaluations remain every100 absolute updates. Save an extension endpoint as its own protected full checkpoint; never overwrite the protected10k state. Approve comparison coverage before later outcomes; do not selectively extend favorable conditions.

## Evaluation neutrality and failures
Capture/restore model mode and Python/NumPy/Torch CPU/CUDA RNG around in-memory evaluation, including exceptions. Release training graphs before evaluation. No-grad/inference metrics must not consume training data or alter the next update. Test identical subsequent CPU trajectories with evaluation inserted/removed.

Persist records atomically keyed by run/step/panel/scope/intervention/version. Resume detects completed evaluations and avoids double-counting. Failed evaluations stay failed until explicitly retried from retained/recoverable states. Do not interpolate deleted-state metrics and call them observations.

## Storage classes
1. Retained analysis weights at0,100,250,500,1000,2000,5000,7500,10000, with exact parameter dtype/config/checksums. Weights-only states enable reanalysis but are not resumable optimizer checkpoints.
2. Rolling FULL recovery every500 updates and orderly interruption; retain two verified generations while active.
3. Protected FULL final10k checkpoint, plus protected final state of every approved extension. A save_pretrained weights directory alone is insufficient.

Full state: student and any actual trainable auxiliary parameters, optimizer groups/moments, scheduler state and immutable formula, scaler if used, Python/NumPy/Torch CPU/all relevant CUDA RNG, sampler epoch/cursor/generator, completed update/input/target counts, protocol/model/data/init/calibration/hardware hashes, code commit, precision/backend and parent lineage. Teacher weights need not be duplicated if pinned external artifacts are verified.

Save only at optimizer boundaries. A signal requests safe stop: finish effective batch or replay it from a prior committed state. Mid-accumulation resume is not supported unless partial gradients/cursors are explicitly implemented and tested; not required here.

Write a temporary directory on the same filesystem, flush, calculate checksums, write COMPLETE marker, then atomic rename. Never delete the last good checkpoint before verifying its replacement. Reject corrupt/partial saves, disk-full truncation, hash mismatch and duplicate writers. Avoid unrestricted loading of untrusted serialized state; use safe tensor weights and documented primitive metadata, with optimizer/RNG serialization validated under the pinned Torch policy.

## Resume acceptance
Uninterrupted tinyN updates must match K+resume+(N-K): model tensors, moments, LR, next block IDs, RNG, metric keys and next loss on the same deterministic backend. Test around saves and evaluation boundaries. Resume reuses the existing hardware lock; it never auto-tunes again. Changed condition/seed/tokenizer/protocol/data prefix/batch is a new run. Cross-device migration needs explicit provenance and cannot claim bitwise replay.

Estimate disk from actual serialized sizes. Keep sparse states and full final; scoped cleanup of rolling duplicates requires a verified final state. Never delete Upstream, user caches or prior experiments during ordinary cleanup.
