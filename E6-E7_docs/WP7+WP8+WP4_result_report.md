Results
Suite: 154 passed, 2 skipped (baseline was 55 passed, 2 skipped). All five offline smoke scripts — E3/E4/E5/E6/E7 — exit 0. test_frozen_files.py green; no frozen file touched, and nnsight_engine.py was not modified at all.

Environment: the checkout had no .venv and the global interpreter lacked nnsight/peft, so 20 tests were erroring. Created .venv and installed requirements.txt (nnsight 0.7.0, peft 0.20.0, torch 2.10.0+cu128, transformers 5.3.0), then confirmed the pre-existing 55 were green before writing new code.

WP7 — common/cross_example_patching.py. All nine invariants pass. Measured, not assumed:

identity and self-source patches give max |Δlogits| = 0.0 for K, V and R writes
patching one of two KV heads moves exactly query heads 0–3 (0.322) and leaves 4–7 at the fp32 solve residual (~9e-7) — proving the _repeat_kv mapping, not just the count
RoPE at position 0 deviates by 0.0; the forward/inverse rotation round-trips at positions 0, 1, 9
WP8 — common/paired_manifests.py, additive loaders, the two real corpus providers, and both prepare scripts. The XNLI fixture emits each language's shard in a different order, so a positional join fails loudly.

WP5 — transformation_inheritance/train_distillation.py + 3 configs + check_pilot_gate.py. D0/D1/D2 at one seed share a byte-identical initial state dict; resume is bit-exact.

Things worth your attention
A real bug the tests caught: load_checkpoint used a strict load_state_dict, which fails on every resume because save_pretrained drops GPT-Neo's tied lm_head. It now loads non-strictly and verifies the missing keys are tied — a blanket strict=False would let a truncated checkpoint resume with random layers.

HF's facebook/xnli per-language configs carry no promptID. load_xnli_aligned resolves a strategy at runtime — explicit id column if present, else the structurally aligned all_languages config — and raises if neither exists. No positional fallback. Check the printed join_strategy on the real run; if it says all_languages_structural, the paper's data section should say that rather than claiming a promptID join.

Two spec reconciliations, both recorded in NEXT_STEPS §1.1: Kmid is realised as Kmid_prerope/Kmid_postrope (new values, no renamed column), and K/V patching is Qwen/Neo/OPT only — GPT-2's fused c_attn raises rather than shipping an untested write path.

Three tolerances I set deliberately, none to make a failure disappear: the pre-_repeat_kv head threshold sits between a 0.32 signal and a 9e-7 fp32 residual; the scorer test uses 1e-6 because the scorer computes in fp32 by design; JSD is clamped to [0, log 2], its mathematical bounds, because fp32 cancellation gave −2e-8 for near-no-op patches and a negative divergence in a results CSV is indistinguishable from a measurement.

CLAUDE.md's one-WP-per-session rule was set aside at your instruction. Three of the four traps I added to CLAUDE.md came out of this slice.

Nothing has been committed — the branch is dirty with 4 modified and 17 new files.