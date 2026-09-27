What shipped
WP9 — extract_sink_representations.py (streams derived tensors to .npy memmaps, resumable, refuses a tokenizer-mismatched manifest), evaluate_crosslingual_retrieval.py (per-language centring, cosine + Procrustes, grouped language probe, mid-token control in the same row), run_multilingual_fingerprint.py, and four e7_qwen*.yaml configs.

WP11 — run_cross_language_patching.py (dev-screen → test-window layer protocol, identity abort, 05 §3 schema exactly), aggregate_crosslingual.py, and e7_preregistration.yaml.

Tests — 6 new files (87 tests) plus nnsight_e7_pipeline_smoke.py, registered as E7-PIPE in the runner.

Verification
pytest tests/ -q → 285 passed, 0 skipped (was 191 passed, 2 skipped)
All seven smoke programs exit 0; runner exits 0
test_frozen_files.py green — no frozen file touched
WP7's 30 invariant tests pass unchanged, proving the cross_example_patching extension was additive
Three pre-existing defects found and fixed
compute_fingerprint raised on every Qwen item. It recorded int_i/int_j as unavailable, then asked run_all to run the whole registry anyway. run_all gained an additive keys= argument; Qwen fingerprints now complete with (i)/(j) recorded as failures, as 02 §3.4 always specified.

The bridge gate was comparing GPT-2 against a GPT-Neo-1.3B reference. _reference_csv() took the first rglob hit; once nn_results/ existed that was Neo. int_f read 0.561 vs 0.127 — which looks exactly like a seam regression. It now matches on the reference's run_config.json. Against the correct artefact the seam agrees to ~1e-7, and both §2 compute-PC gates now pass on real GPT-2 + real datasets.

hierarchical_bootstrap re-ran a pandas groupby inside every replicate, making E7 aggregation take tens of minutes. Now derived once — verified bit-identical to the old implementation, 45× faster.

Two things needing your attention
§2.6 (blocking, no GPU): design §18's remaining two criteria, §10.11's language tiers, and §10.12's six interpretation rows are not in this repo. Following the WP10 precedent, e7_go_no_go.json reports decision: "incomplete" / supported: null regardless of results until you transcribe them. Do it before results exist.
§2.3: BASELINE_HASHES.json still has reference_results / environment as PENDING — the numbers now exist in nn_results/, so this is transcription work.
One scope note: nn_results/ is untracked, so a fresh clone will see the two bridge gates skip again rather than fail.