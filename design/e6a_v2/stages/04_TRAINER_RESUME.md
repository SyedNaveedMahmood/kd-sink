# Stage04 - Trainer, profiling solver and exact resume

Dependency:stage03. Read AGENTS,NEXT_STEPS,CORE/S1 journals,TRAINING_AND_CHECKPOINTING.md,HARDWARE_AND_EXECUTION.md,SOFTWARE_AND_ARTIFACT_CONTRACTS.md.

## Tasks
- [ ]04.1 Implement one single-condition/single-seed trainer,fixed64 effective batch,parameter groups,BF16/FP32 policy,one-update accumulation and absolute LR schedule.
- [ ]04.2 Implement isolated actual profile commands and common-batch solver; mock eligibility/maxima for CPU tests without calling them hardware measurements.
- [ ]04.3 Implement sparse analysis weights,rolling full saves,protected final full state,atomic checksums and duplicate-writer protection.
- [ ]04.4 Implement strict resume/extension,interruption,RNG/data/scheduler restoration,immutable compatibility checks and no mid-run retuning.
- [ ]04.5 Add tqdm/non-TTY event logs with study/condition/seed/GPU/progress/elapsed/ETA/loss/VRAM and separate evaluation progress.
- [ ]04.6 Pass T04/T05/T06/T10 CPU integration and previous regression. Real GPU capacity is stage06 evidence.

## Exit gate
Tiny interrupted/uninterrupted runs match state,cursor,RNG,LR,next loss. Save failures preserve prior good state. CLI cannot sweep silently or reinterpret stop-after as schedule change. Resume beyond10k retains floorLR/history. Logging does not change model randomness.

Record CPU test commands/status and realGPU blockers in reports/stage04.json and journals. Milestone commits include code/tests/NEXT_STEPS evidence together. No actual campaign launch implied.
