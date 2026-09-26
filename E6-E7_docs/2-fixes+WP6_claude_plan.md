# Trap fixes + WP6 (E6B sentiment adaptation)

## Context

Two follow-ups from the WP12 session, then the last remaining work package.

**The push is already done.** `sink-inheritance-foundation` is 0 ahead / 0 behind
`origin`; WP12 is committed (`c65c2d0`) and pushed. The only uncommitted path is
`nn_results/` (2.5 GB, 2069 files), which **cannot** be pushed — five files exceed
GitHub's hard 100 MB cap, the largest being 276 MB — and `NEXT_STEPS.md` already documents
it as deliberately untracked. It goes into `.gitignore` so the decision lives in the repo
rather than only in prose.

**I rechecked both traps empirically. One is a real defect, one is not.**

**Trap B — `criterion_3_combine` is genuinely wrong**, and worse than I described. Probed
directly:

| YAML state | reported | actual verdict |
|---|---|---|
| `combine: any` | `combine='any'`, `status=ok` | `met=True` ✓ |
| `combine: all` | `combine='all'`, `status=ok` | `met=False` ✓ |
| `status:` deleted, **no** `combine:` | `combine='any'`, `status=ok` | `met=True` ✗ |
| `combine: ALL_` (typo) | `combine='all_'`, `status=ok` | `met=True` ✗ |
| `combine: enny` (typo) | `combine='enny'`, `status=ok` | `met=True` ✗ |

The last three are the same class of failure and the typo rows are the worst: the artefact
*records a value the code did not honour*, so it looks auditable and lies. `str(...) or
"any"` plus an `if combine == "all": ... else: any` branch silently resolves a decision
nobody made. `go_no_go_combine` (both aggregators) and `e6b_go_no_go.combine` share the
unknown-value half of the bug.

**Trap A — the `thresholds:` block is *not* wrongly implemented.** `08` §10 explicitly
instructs replacing the value "in all six rows", so literals-in-rows is the intended data
model and `_condition_holds` reading them is correct. The real gap is that **nothing checks
the declared `thresholds:` against the rows**: edit the block alone and every test passes
while the matrix still uses 0.10. That is an under-guard, so the fix is an assertion, not a
behaviour change.

**WP6** is the last work package (`00` §6 keeps it conditional, outside the minimum
publishable version). Per `03` §4–5 and `07` P6 it is the trainer, four configs,
`screen_public_pairs.py`, and three named tests. I verified `peft` 0.20.0 works with
`transformers` 5.3 on GPT-2 `Conv1D` and that `merge_and_unload()` returns a plain
`GPT2LMHeadModel` — so the merged checkpoint is fingerprintable by the frozen harness.

### Confirmed scope

- `nn_results/` → `.gitignore`.
- WP6 **includes the evaluator link** (drift + task columns).
- Verification is **offline smoke + one short real GPU run**.

---

## Part 1 — Trap fixes (own commit, pushed before WP6 starts)

**1.1 Refuse an unresolved or unrecognised `combine`** — [aggregate_crosslingual.py](crosslingual_semantics/aggregate_crosslingual.py)

`_combine_criterion_3`: an entry that exists, carries no `PENDING_*` status and names no
`combine:`/`value:` returns `met: None` with a reason naming D6, and is swept as a blocking
pending decision. An unrecognised value (anything not `any`/`all` after casefold) is
refused the same way rather than falling through to `any`. Same treatment for
`go_no_go_combine` in both aggregators and `e6b_go_no_go.combine` in the E6 one.

This is trap 11 applied to itself: a field the aggregator *half*-reads is as bad as one it
ignores.

**1.2 Assert the declared thresholds match the rows** — [aggregate_crosslingual.py](crosslingual_semantics/aggregate_crosslingual.py) + [tests/test_e7_interpretation_exclusivity.py](tests/test_e7_interpretation_exclusivity.py)

Compare `set(interpretation_matrix.thresholds.values())` against the set of literal
`value:`s used across the rows. Set-equality catches both drift directions (block edited
without rows, rows edited without block) and needs no new mapping in the YAML — the
existing `test_each_condition_key_uses_exactly_one_threshold` already pins one threshold
per key, and the two together are a complete guard. Emitted as
`threshold_declaration_consistent` in `interpretation_matrix.json` **and** asserted in the
test, per trap 11: ship the reader and the test that fails if the reader is deleted.

**1.3 New tests** — `tests/test_combine_refusals.py`, plus two cases added to the
exclusivity file. Each asserts the *relationship* between YAML and verdict, never a value.

Commit message: `Refuse unresolved and unrecognised combine values`.

---

## Part 2 — WP6

### 2.1 `transformation_inheritance/train_sentiment_adaptation.py`

Mirrors [train_distillation.py](transformation_inheritance/train_distillation.py)'s shape
throughout — `prepare_run` / `train` / `evaluate` / `write_run_config` /
`append_jsonl` / `write_runtime_estimate`, `--resume auto`, the `_smoke` asset directory —
so the two trainers stay diffable.

Reuses rather than reimplements:
- `cp.sst2_prompt_corpus(...)` ([corpus_providers.py](common/corpus_providers.py)) already
  builds the `{sentence}\nSentiment:` format with the sentence at position 0, records
  `label_span` for loss masking, and **already accepts a `corrupted_manifest`** mapping
  `source_index -> flipped_label`. `test_sst2_prompt_format.py` already pins the format.
- `seed_everything`, `rng_state`, `lr_lambda`, `latest_checkpoint`, `tied_weight_keys`
  from the distillation trainer (imported, not copied).

New, because nothing equivalent exists:
- `build_corruption_manifest(train_rows, rate=0.20, seed=0) -> DataFrame` with
  `03` §4.3's four constraints: exactly `round(rate*N)` flipped, class-stratified within 1,
  train-only (asserted), reproducible within seed and distinct across seeds.
- `startup_assertions`: `per_device_batch * grad_accum == 32` for every condition
  (`03` §4.4), and the label-token count check — if `" positive"`/`" negative"` are
  multi-token, train and score the full sequence.
- `merge_and_verify`: `merge_and_unload()`, save `merged/` beside `adapter/`, then reload
  both **in fp32 on CPU**, score five fixed parity sentences, assert max abs logit
  difference < 1e-5, write `merge_parity.json` stating that training was bf16 and that this
  is *not* a bf16 tolerance claim (`03` §4.5). `fan_in_fan_out=True` is set explicitly —
  peft otherwise emits a correction warning for `Conv1D`.
- `evaluate`: accuracy, NLL and ECE over 10 **equal-width** bins per checkpoint into
  `eval_log.jsonl`.

Outputs follow `03` §4.6 exactly.

### 2.2 Four configs — `configs/e6b_f1..f4.yaml`

F1 LoRA/clean, F2 full/clean, F3 LoRA/corrupt, F4 full/corrupt, matching the
`e6b.factorial.conditions` map already in `e6_preregistration.yaml`. Hyperparameters are
`03` §4.4's verbatim and pre-registered — LoRA r=8 α=16 dropout 0.05 lr 2e-4 batch 32
accum 1; full FT lr 5e-5 batch 16 accum 2; both wd 0.01, warmup ratio 0.06, cosine,
3 epochs, bf16, clip 1.0, max 64 input tokens. F0 is the untrained base and needs no
config — it is the drift reference.

### 2.3 `transformation_inheritance/screen_public_pairs.py`

`03` §5 / `07` P6: a thin driver over `fr.compute_fingerprint` for gpt2 vs distilgpt2 and
Qwen2.5 0.5B/1.5B base vs instruct. No training. Writes `screening_fingerprints.csv` in the
`FingerprintRecord` schema. One command, standalone — "the cheapest evidence in the whole
project".

### 2.4 Evaluator link — [evaluate_transformation.py](transformation_inheritance/evaluate_transformation.py)

Both halves are additive; an `e6a` run produces byte-identical output.

- **Task metrics**: `task_accuracy` / `task_nll` / `ece_10bin` are *read* from the row
  `read_eval_log` already returns, exactly as `validation_ce` is. The trainer measured them
  with the model's own scorer; recomputing here would be a second source of truth.
- **Drift**: a `--base` handle (F0 / `distilgpt2`) fingerprinted on the same corpora,
  dispatching to the existing `im.fingerprint_drift`, `im.topology_drift`,
  `im.carrier_drift`. Guarded by the same `guard_comparison` the teacher uses, so a base
  measured on a different corpus or registry raises rather than producing a plausible
  drift. Absent `--base`, the columns stay empty with a recorded reason — never zero,
  which would read as "no drift".

### 2.5 Tests (`06` §1 names the first three)

| File | Proves |
|---|---|
| `tests/test_corruption_manifest.py` | exactly 20% flipped; class-stratified within 1; validation untouched; distinct across seeds; reproducible within seed |
| `tests/test_lora_merge_parity.py` | merged vs unmerged logits agree < 1e-5, fp32 CPU, five parity sentences |
| `tests/test_effective_batch.py` | `per_device_batch * grad_accum == 32` for every shipped F-config |
| `tests/test_e6b_drift_columns.py` | drift dispatches to `inheritance_metrics`; no base ⇒ empty + reason, never 0.0; task metrics read from `eval_log` |
| `tests/nnsight_e6b_smoke.py` | offline end to end: train F1+F3 → merge+parity → evaluate (with base) → aggregate → `e6b_drift.csv` and `e6b_go_no_go.json` carry real rows |

`nnsight_e6b_smoke.py` joins `scripts/run_nnsight_rest_smoke_tests.ps1` as an eighth
program.

---

## Verification

1. `./.venv/Scripts/python.exe -m pytest tests/ -q` — 372 existing + the new files, 0 skipped.
2. `powershell -File scripts/run_nnsight_rest_smoke_tests.ps1` — all **eight** programs
   exit 0 (it now defaults to `.venv`; verdict from `$LASTEXITCODE`, trap 8).
3. Trap-fix regression: the shipped `e7_preregistration.yaml` still yields
   `decision: "incomplete"`, and a hand-built prereg with `status:` deleted and no
   `combine:` now reports `met: None` instead of `True`.
4. **Short real GPU run** on the RTX 2060: F1 (LoRA) on real `distilbert/distilgpt2` +
   a few hundred real SST-2 rows, one seed, ~200 steps → merge → `merge_parity.json`
   under 1e-5 → fingerprint the merged checkpoint against the F0 base → confirm
   `fingerprint_drift_from_base` is a real number. Proves the peft → merge → frozen-harness
   path on real weights. A download or peft failure is reported as a failed step, not
   worked around.
5. `pytest tests/test_frozen_files.py` — no frozen file touched; `nnsight_engine.py` and
   `datasets_loader.py` unmodified.

---

## Documentation and pushes

Two commits, each pushed after its own verification:

1. `Refuse unresolved and unrecognised combine values` — Part 1 + `.gitignore`.
2. `Implement WP6 sentiment adaptation and E6B evaluator link` — Part 2.

- **`CLAUDE.md`** — WP6 in the E6 pipeline table; a thirteenth trap for the
  half-read-field defect (`str(x) or "default"` on a pre-registration field silently
  resolves a decision and records a value it did not honour).
- **`NEXT_STEPS.md`** — WP6 moves from §3 into §1's table with a §1.5 decisions block;
  §3's "WP6 is the only work package left" becomes "everything is code-complete"; the E6B
  compute-PC recipe (4 conditions × 3 seeds, `--base` for drift, `screen_public_pairs.py`
  first) is added; the D4 dependency is restated — `e6b_1`/`e6b_2` are still open decisions
  and must be made before E6B results exist.
