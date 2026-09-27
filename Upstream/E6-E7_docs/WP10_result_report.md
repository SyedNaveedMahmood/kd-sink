What shipped
transformation_inheritance/evaluate_transformation.py — checkpoints → checkpoint_metrics.csv in the 05 §2 schema. Arch is read from each checkpoint's own config.json; band from normalised_depth_band; every number from compute_fingerprint, every distance from inheritance_metrics. Resume is keyed on intervention set + manifest hash + registry version + band + checkpoint sha, so a changed instrument invalidates rather than skips. Failed checkpoints become rows with status set and blank metrics, and the run continues.

transformation_inheritance/aggregate_transformation.py — contrasts (BH-corrected within family), matched-loss selection, table 2, all four E6B tables, three figures, go_no_go.json.

configs/e6_preregistration.yaml — every threshold and criterion, read by the aggregator, hard-coded nowhere.

Verification
pytest tests/ -q → 191 passed, 2 skipped (was 154; the same 2 compute-PC gates skip). test_frozen_files.py green.
All six smoke programs pass via scripts/run_nnsight_rest_smoke_tests.ps1, runner exit 0.
nnsight_e6_eval_smoke.py runs the real pipeline offline: 3 conditions trained → evaluated → aggregated → gated. Resume recomputed nothing (asserted by call counter, not row count), and the pilot gate's criteria 2 and 3 — unevaluable before this — now read real values.
Three things to flag
Design §8.7 and §18 are not in this repo, and WP10's contrasts/go-no-go are supposed to be literal transcriptions of them. Four contrasts were mechanically derivable; the fifth contrast and two criteria are PENDING_DESIGN_* sentinels that report met: null and force decision: "incomplete" regardless of results. Paste those sections into the YAML before the numbers exist — filling them afterwards is indistinguishable from tuning them. Recorded as blocking item §2.5 in NEXT_STEPS.

The smoke runner was broken. 2>&1 | Tee-Object under $ErrorActionPreference = "Stop" turns transformers' stderr progress bars into terminating NativeCommandErrors, so it aborted during E3 despite python exiting 0. Fixed by relaxing the preference around the native call and taking the verdict from $LASTEXITCODE. If it ever "passed" before, that was on a stack printing no progress bars.

Deviations, all recorded in NEXT_STEPS §1.2: resume granularity is (run_id, step, corpus_id) not (…, intervention) — run_all computes the whole battery in one sweep, so finer resumption would increase forwards; compute_fingerprint gained delta_ce_per_item (default byte-identical) because ΔCE means can't be bootstrapped; run_config.json gained the teacher fields it never recorded; --smoke now persists its synthetic teacher so the teacher-comparison path is actually exercised; and emergence_dynamics_analysis.py has no importable style helpers, so its conventions were matched literally.

Untracked nn_results/ (real GPU E1/E2 output) was already present and I left it alone.