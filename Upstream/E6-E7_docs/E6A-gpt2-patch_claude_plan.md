# E6A-GPT2 — a second distillation arm with a teacher that has a sink

## Context

The TinyStories arm failed pilot criterion 2 for a reason that is not a code fault: **the
teacher has no in-domain attention sink to inherit.** Measured on the corpus every criterion
is scored on, `roneneldan/TinyStories-33M` has `baseline_sink = 0.009309` — **0.86× the
uniform-attention floor**, with zero (layer, head) cells above 0.2. The 0.15 bar is 16.1× the
teacher's own value, and the teacher misses it on the cross-domain corpus too (0.085975), so
no corpus reselection or step count rescues it. The converged public 8M sits at 0.003317.

Meanwhile `results/screening/screening_fingerprints.csv` already establishes, on real weights,
that **gpt2 → distilgpt2 is a distillation pair with a strong sink at both ends**: 0.563006 and
0.463267 on `e1_100x40` — 16.5× and 13.6× that corpus's floor, with 73% and 71% of cells above
0.2. This plan runs the same registered E6A design with gpt2 as the teacher.

**Scope:** a *new, isolated arm* — `experiment_id: e6a_gpt2`, conditions G0/G1/G2, its own
pre-registration file, its own results subtree, 2,000-step pilot. The TinyStories arm and
`e6_preregistration.yaml` are **not touched**; `e6a_gpt2` is not an amendment to `e6a`.

**Decisions taken with the user before any run exists** (2026-08-01): OpenWebText (capped) as
the corpus, `e6a_gpt2` + G0/G1/G2 for isolation, criterion 2 = design §8.5's 0.15 verbatim
**plus** a new teacher-relative criterion 2b, and `distilbert/distilgpt2` as the public
reference with its non-independence recorded.

---

## The instrument

| | TinyStories arm (existing) | **gpt2 arm (new)** |
|---|---|---|
| teacher | `roneneldan/TinyStories-33M` 4L×768, 16H | **`gpt2`** 12L×768, 12H |
| student | `TinyStories-8M` config, random, 8L×256 | **`distilbert/distilgpt2` config, random, 6L×768** |
| public reference | `roneneldan/TinyStories-8M` | **`distilbert/distilgpt2`** |
| layer map | `{0:1, 1:3, 2:5, 3:7}` (4→8) | **`{1:0, 3:1, 5:2, 7:3, 9:4, 11:5}`** (12→6, evenly spaced, ends on the last layer) |
| training corpus | `tinystories_train` | **`openwebtext_train`** (capped, see below) |
| in-domain sink corpus | `tinystories_validation_sink_300` | **`openwebtext_validation_sink_300`** |
| cross-domain | `e1_100x40` | `e1_100x40` (unchanged — keeps E1 and the screening table comparable) |

Widths match (768/768) and every layer is global, so `attention_types()` already returns
all-`global` and `build_valid_masks` produces full causal masks — all four `L_ATTN` pairs will
be `global_global`, cleaner than the TinyStories arm's `mixed` pairs. `int_b` is *available* on
GPT-2 (it is excluded on Neo for want of a q-bias), so the fingerprint has 10 keys, not 9.

**OpenWebText, capped and split by document.** `Skylion007/openwebtext` ships only a `train`
split and is ~38 GB. 2,000 steps × 16 × 4 × 128 = **16.4M tokens**, so the loader streams a
**deterministic document prefix** and carves two disjoint windows: documents `[0, N_train)` for
training (default 60,000 ≈ 60M tokens, ~1–2 GB) and `[N_train, N_train + N_val)` for the sink /
ΔCE corpora. Disjointness is by construction and recorded, mirroring `SINK_BLOCKS_RESERVED` in
`corpus_providers`. Document order is then shuffled with the run seed *before* packing, exactly
as design §1.4 requires.

---

## Changes

### 1. `common/datasets_loader.py` — additive, and `load_tinystories_blocks` is not touched

Add `OPENWEBTEXT_HF_PATH`, `_load_openwebtext(split, max_documents, offset)` (streaming
prefix → materialised list, `text` field), and `load_openwebtext_blocks(...)` with the same
signature shape and the same `as_array` / `manifest_input_ids` flags.

The packing loop is a **faithful copy** in a new private `_pack_documents(...)`, *not* a
refactor of `load_tinystories_blocks`. Rationale: `manifest_sha256` joins every cached artefact
in the project (`05` §7.1, trap 19) and a body edit that moved it by one byte would silently
partition the existing arm's records. The copy is proved faithful by test, not by inspection —
see §9.

### 2. `common/corpus_providers.py` — `openwebtext_corpus(...)`

Mirrors `tinystories_corpus` exactly: same `purpose="sink"|"ppl"` disjoint-window logic, same
`_make_corpus` tail, corpus id `openwebtext_{split}_{purpose}_{n_blocks}` →
`openwebtext_validation_sink_300`.

### 3. `transformation_inheritance/train_distillation.py` — honour `data.dataset`

`build_block_dataset` hard-codes `dl.load_tinystories_blocks`. Add a module-level registry:

```python
BLOCK_LOADERS = {"roneneldan/TinyStories": dl.load_tinystories_blocks,
                 "Skylion007/openwebtext": dl.load_openwebtext_blocks}
```

resolved from `config["data"]["dataset"]` (already recorded in `run_config.json`). An
**unrecognised id raises**, naming the registry — never a silent fall-back to TinyStories
(trap 13). Absent key defaults to TinyStories, which is what every existing config means.

### 4. `transformation_inheritance/evaluate_transformation.py` — corpus set follows the run

`build_corpora` hard-codes `cp.tinystories_corpus` for both the in-domain sink corpus and the
two ΔCE corpora. Select the provider from the same registry, keyed on the run's
`dataset_name`; refuse an unrecognised value, default to TinyStories when absent.

### 5. `transformation_inheritance/check_pilot_gate.py` — `--experiment` and criterion 2b

- `--experiment` (default `e6a`) and `--conditions` (default `D0,D1,D2`) replace the hard-coded
  `EXPERIMENT_ID` / `CONDITIONS`; `run_dir` follows.
- `PRIMARY_CORPUS_ID` becomes a per-experiment map (`e6a` → tinystories…, `e6a_gpt2` →
  openwebtext…), still with `--corpus` override, still asserted against the arm's
  pre-registration by test.
- **New `criterion_2b`** — the student reaches ≥ `TEACHER_SINK_FRACTION = 0.50` of the
  *teacher's own* `baseline_sink` on the scored corpus. The teacher's value is read from the
  per-corpus fingerprint record
  `<run_dir>/fingerprints/teacher_*/step_na/<corpus_id>/fingerprint.json` — an artefact that
  only exists because of the trap-24 fix. Absent or ambiguous ⇒ `met: null` with a reason,
  never a defaulted `False` (trap 16). Reports `teacher_sink`, `required`, and each condition's
  `at_pilot_step` / `max_through_pilot_step` fraction.
- `proceed` requires all criteria met, so the gpt2 arm's gate is five criteria, not four.
  The TinyStories arm keeps four: criterion 2b is registered in the **new** pre-registration
  only and the gate emits it only when the arm's spec declares it.

### 6. `transformation_inheritance/aggregate_transformation.py` — `--experiment`

Two hard-coded filters (`usable["experiment_id"] == "e6a"` / `"e6b"`) become the flag's value;
add `"openwebtext_"` to `DEFAULT_CORPUS_PREFIXES`. Pointing `--results` at
`transformation_inheritance/results/e6a_gpt2` already scopes both discovery and the
`aggregate/` output directory, so the arms cannot pool.

### 7. `transformation_inheritance/run_pilot_parity.py` — architecture dispatch

Currently hard-wired to the frozen Neo harness. Read the architecture from the staged
checkpoint's `config.json` via the existing `ev.ARCH_BY_CLASS`, then dispatch operands:

| arch | frozen module | operands |
|---|---|---|
| `neo` | `cross_scale_and_architecture/neo/intervention_analysis_neo.py` | `make_manual_runner_neo`, `neo_swap_directions`, `identify_massive_coords_neo(model)` |
| `gpt2` | `common/intervention_analysis_legacy.py` | `make_manual_runner`, `gpt2_swap_directions`, `identify_massive_coords(model, tokenizer)` |

Same `run_parity_check`, same `PILOT_PARITY_SENTENCES` (5), same fp32-only constraint. Note the
GPT-2 massive-coords helper takes the tokenizer as a second argument — a real signature
difference, not an oversight. An unsupported arch raises rather than guessing.

### 8. Configs and the pre-registration

- `configs/e6a_gpt2_ce.yaml`, `e6a_gpt2_logit_kd.yaml`, `e6a_gpt2_logit_attention_kd.yaml` —
  conditions G0/G1/G2, the design §8.4 optimiser block **copied verbatim** (lr 5e-4, betas
  (0.9, 0.95), wd 0.1, 500 warmup, cosine to 10%, clip 1.0, batch 16 × accum 4, bf16,
  `max_steps: 10000`, checkpoints `[0,100,250,500,1000,2000,5000,10000]`), loss weights
  verbatim (`1.0` / `0.5+0.5` / `0.45+0.45+0.10`, T=2.0), plus
  `data: {dataset: Skylion007/openwebtext, train_documents: 60000, validation_documents: 4000,
  block_size: 128, eos_between: true}`.
- **New** `configs/e6a_gpt2_preregistration.yaml`, `e6a_gpt2_prereg_v1`, dated **2026-08-01**,
  written **before any gpt2 run exists** — the ordering that is the whole content of a
  pre-registration (`05` §6), recorded in the header in checkable terms.
  - The five §8.7 contrasts and the four §18 criteria transcribed from
    `e6_prereg_v4` with D→G and the corpus id substituted; every `source:` line preserved.
  - Framing (B) rules for `e6a_gpt2_3` / `e6a_gpt2_4` reused verbatim, null condition **G0**,
    k = 2 — so this arm also reads `incomplete` until G0 has three seeds.
  - `pilot_gate:` block declaring `sink_threshold: 0.15` (design §8.5 verbatim),
    `teacher_sink_fraction: 0.50` (**new, decided 2026-08-01, argued in `amendments:`**),
    `corpus_id`, and `mechanistic_delta_threshold: 0.10`.
  - `amendments:` records why the arm exists — the teacher measurement 0.009309 vs a 0.15 bar —
    and that distilgpt2 is layer-initialised from gpt2 and distilled, so `e6a_gpt2_4` is a
    *distilled* comparand, not an independence control. **The paper must say so.**
- `e6_preregistration.yaml` is **not edited**. Neither threshold in it moves.

### 9. Tests (each fails if its change is reverted)

- `tests/test_block_packing_equivalence.py` — monkeypatch `_load_tinystories` and
  `_load_openwebtext` to the *same* synthetic document list; assert `load_tinystories_blocks`
  and `load_openwebtext_blocks` return byte-identical blocks, identical manifests and identical
  `prov.sha256_int_rows`. This is what makes the copied packing loop safe. No download.
- `tests/test_openwebtext_corpus.py` — determinism under a fixed seed, sink/ppl windows
  disjoint, train/validation document windows disjoint, `corpus_id` literal,
  `n_blocks` cap honoured.
- `tests/test_pilot_gate.py` — `--experiment`/`--conditions` reach `run_dir`; criterion 2b met
  and not-met from a fixture built in the **producer's** shape (a per-corpus teacher
  fingerprint record); a missing teacher record ⇒ `met: null` + reason; the gpt2 arm's
  `PRIMARY_CORPUS_ID` equals the corpus its own pre-registration names.
- `tests/test_e6a_gpt2_configs.py` — the three configs' weights, layer map and optimiser block
  match `e6a_gpt2_preregistration.yaml` cell-for-cell (the pattern
  `test_effective_batch.py` already uses); the layer map is a bijection into `range(6)` with
  every teacher index `< 12`.
- `tests/test_aggregate_contracts.py` — `--experiment e6a_gpt2` sees only that arm's rows; an
  `e6a` row in the same tree cannot enter its contrasts.
- `tests/test_pilot_parity_staging.py` — arch dispatch picks the legacy GPT-2 operands for a
  `GPT2LMHeadModel` config and the Neo ones for `GPTNeoForCausalLM`; both frozen helper sets
  are importable under the names used; an unknown arch raises.
- `tests/nnsight_e6a_gpt2_smoke.py` — the whole arm offline on a tiny random GPT-2 pair
  (train G0/G1/G2 → evaluate → aggregate → gate), added to
  `scripts/run_nnsight_rest_smoke_tests.ps1`.

### 10. Documentation

- **CLAUDE.md** — the new configs in the E6 table; **the pre-registration list becomes three
  files**, not two; new **trap 26**: *a criterion's threshold is only meaningful if the
  teacher clears it — measure the teacher on the scored corpus before training an arm*, with
  the 0.009309-vs-0.15 measurement as the case.
- **NEXT_STEPS.md §2.12** — the arm, the decision record, the measured cost, the pre-flight
  gate.
- **RUNBOOK/01_PILOT.md** — a parallel "gpt2 arm" section with every command below; the
  existing TinyStories phases stay as they are.
- **COMMANDS.md** — the same commands in dependency order.

---

## Run order on the compute PC

**Step 0 — the pre-flight the TinyStories arm did not have (~40 min, no training).** This is
the step that decides whether to spend 30+ hours:

```bash
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference gpt2 --with-delta-ce --dtype bfloat16
./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
  --config transformation_inheritance/configs/e6a_gpt2_ce.yaml \
  --evaluate-public-reference distilbert/distilgpt2 --with-delta-ce --dtype bfloat16
```

Read `baseline_sink` and `frac_cells_above_0_2` for **gpt2 on `openwebtext_validation_sink_300`**.
The uniform floor at 128 tokens is 0.010770, so ×floor = sink ÷ 0.010770.

- **≥ 0.15 with carrier cells above 0.2** → proceed; criterion 2 is reachable and 2b is
  meaningful.
- **Well below 0.15** → stop and report. That is the same failure as TinyStories and no amount
  of training changes it. Do not adjust the threshold.

**Steps 1–5 — the pilot**, identical in shape to the TinyStories pilot:

```bash
for c in e6a_gpt2_ce e6a_gpt2_logit_kd e6a_gpt2_logit_attention_kd; do
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/$c.yaml --seed 0 --max-steps 2000
done
for s in 1 2; do   # G0 at three seeds — framing (B) refuses a spread from fewer
  ./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_gpt2_ce.yaml --seed $s --max-steps 2000
done

for c in G0 G1 G2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a_gpt2/$c/seed0 \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done
for s in 1 2; do
  ./.venv/Scripts/python.exe transformation_inheritance/evaluate_transformation.py \
    --run-dir transformation_inheritance/results/e6a_gpt2/G0/seed$s \
    --steps 0,250,500,1000,2000 --with-delta-ce --dtype bfloat16
done

./.venv/Scripts/python.exe transformation_inheritance/aggregate_transformation.py \
  --experiment e6a_gpt2 \
  --results transformation_inheritance/results/e6a_gpt2 \
  --preregistration transformation_inheritance/configs/e6a_gpt2_preregistration.yaml \
  --reference-condition G0

./.venv/Scripts/python.exe transformation_inheritance/run_pilot_parity.py \
  --run-dir transformation_inheritance/results/e6a_gpt2/G0/seed0 --step 2000

./.venv/Scripts/python.exe transformation_inheritance/check_pilot_gate.py \
  --experiment e6a_gpt2 --conditions G0,G1,G2 --seed 0 --pilot-step 2000 \
  --preregistration transformation_inheritance/configs/e6a_gpt2_preregistration.yaml \
  --parity-report transformation_inheritance/results/e6a_gpt2/G0/seed0/parity/step_2000/parity_report.json
```

**Cost, and it is the real constraint.** VRAM ≈ 7–9 GB estimated (the dominant term — the
50257-vocab KD softmax at batch 16 × 128 — is unchanged from the measured 4.68 GB run), so
24 GB is not binding. Time is: scaling the measured 4.85 s/step by ~4.2× student FLOPs and
~1.8× teacher forward puts a 3090 at roughly **12–20 s/step → 7–11 h per 2,000-step run, 33–55 h
for all five**. Do not trust that: `runtime_estimate.json` is written after step 100 with both
`units_per_second` and `peak_vram_bytes`. **Run G0 seed 0 first and read it before launching the
rest.** If the budget is tight, run seed 0 only and accept `go_no_go: incomplete` — that is the
framing-(B) rule working, and G0 seeds 1–2 can follow later.

---

## Verification

```bash
./.venv/Scripts/python.exe tests/test_frozen_files.py          # 12/12, unchanged
./.venv/Scripts/python.exe -m pytest tests/ -q                 # 538 + new, 0 failed
powershell -File scripts/run_nnsight_rest_smoke_tests.ps1      # now 9 programs, exit 0
```

Then, before any GPU time:

- `check_pilot_gate.py --experiment e6a` on the existing TinyStories archive must produce
  **byte-identical** output to the current `pilot_gate.json` — the arm split must not move a
  single existing number.
- `aggregate_transformation.py` with no `--experiment` must reproduce the current
  `go_no_go.json` for the TinyStories arm.
- The new smoke program exercises train → evaluate → aggregate → gate for G0/G1/G2 offline.
