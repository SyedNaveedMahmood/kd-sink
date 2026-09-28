# Stage06 - S1/S3 integration and real-GPU readiness

Dependency:stage05. Read AGENTS,NEXT_STEPS,relevant journals,studies/S1_LONGITUDINAL_KD.md,studies/S3_EQUAL_HEAD_REPLICATION.md,HARDWARE_AND_EXECUTION.md,DECISIONS.md.

## Tasks
- [ ]06.1 Create independently runnable S1 C0-C6 configs and optional S3 C0-C4 configs,with strict architecture/variant IDs.
- [ ]06.2 Implement explicit reviewed seven-job seed0 S1 plan and optional separate seed1/seed2 plans; no default multiseed launchers. All S1 production jobs use3090.
- [ ]06.3 Run actual native/adapter/teacher/intervention preflight and assigned-condition smoke on3090/4080; test REL-on4080 refusal.
- [ ]06.4 Profile complete RTX3090 C0-C6 optimizer/evaluator/save cycles at128 tokens,freeze ONE common microbatch,measure calibration factors,and publish real VRAM/RAM/disk/time evidence.
- [ ]06.5 Run actualGPU resume and BF16/FP32 diagnostics; resolve approval locks and produce readiness report without launching long experiments by default.

## Exit gate
Mandatory S1 CPU/GPU checks and common-plan validation pass with real RTX3090 evidence for all seven conditions. Optional S3 capability is tested on tiny models; its realGPU/production scope remains disabled until approved. Missing3090 means BLOCKED,not passed; the agent may implement later CPU-only capabilities only with explicitly documented dependency limitations,never declare readiness.

Commands after implementation: pytest -q tests/gpu --device-role rtx3090 and correspondingrtx4080super; collect versions,counts,failures/skips. Run full CPU regression. reports/stage06.json distinguishes implementation,pilot validation and actual scientific coverage. Append CORE/S1/S3 notes and milestone commits. No student sink outcome is a pass criterion.
