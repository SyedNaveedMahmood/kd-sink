# CLAUDE.md — MechanisticAccountofSinks / Sink Inheritance Extension

## What this project is

Mechanistic interpretability research on attention sinks, targeting an ACL-level venue.
Experiments E1–E5 are **published/frozen**. We are adding E6 (sink inheritance under
distillation and fine-tuning) and E7 (cross-lingual semantic transport through the sink).

Design pack lives in `new_design_plans/`. Read the relevant spec before writing code.

| File | When to read it |
|---|---|
| `00_MASTER_PLAN.md` | always, first — gap analysis, work-package DAG |
| `01_REFACTOR_SPEC_existing_code.md` | any edit to existing files |
| `02_MODULE_SPEC_common.md` | new `common/` modules |
| `03_MODULE_SPEC_e6_transformation.md` | E6 training/eval |
| `04_MODULE_SPEC_e7_crosslingual.md` | E7 manifests/extraction/patching |
| `05_SCHEMAS_AND_CONTRACTS.md` | any file-writing code |
| `06_TEST_PLAN.md` | any test |
| `07_IMPLEMENTATION_PROMPTS.md` | the per-work-package prompts |
| `08_AGGREGATOR_EXTENSION_SPEC.md` | any aggregator change (WP12) |
| `DECISIONS_REQUIRED.md` | **before touching either pre-registration** — the seven open decisions |

`NEXT_STEPS.md` tracks what is built, what must run on the compute PC, and what is left.
Read it before starting a work package.

### New `common/` modules (WP0–WP8, all shipped)

| Module | Purpose |
|---|---|
| `depth_band.py` | normalised-depth band — the fix for trap 1 |
| `corpus_providers.py` | injectable `Corpus`; the E1 regression bridge |
| `fingerprint_runner.py` | any model + any corpus → `FingerprintRecord` |
| `inheritance_metrics.py` | fingerprint / topology / carrier / functional distances |
| `cross_example_patching.py` | source-capture → target-inject patching (E7 causal claims); also the capture-only `Qmean`/`Rlast`/`Rmean` objects WP9 extracts |
| `paired_manifests.py` | FLORES/XNLI parallel joins, patch controls, partitions |
| `provenance.py` | git sha, digests, versions — the mandatory artefact tail |
| `block_corpus_cache.py` | packed-corpus disk cache, digest re-verified on every hit (trap 27) |

### E6 pipeline (`transformation_inheritance/`, WP5 + WP6 + WP10 + WP12 shipped)

| Script | Purpose |
|---|---|
| `train_distillation.py` | E6A: distil a teacher into a random student. **Two arms**: `e6a` = TinyStories-33M → random 8M (D0/D1/D2); `e6a_gpt2` = gpt2 → random distilgpt2 (G0/G1/G2). `data.dataset` selects the corpus via `BLOCK_LOADERS` |
| `train_sentiment_adaptation.py` | E6B: LoRA / full fine-tuning of distilgpt2 on clean and 20%-corrupted SST-2 (F1–F4) |
| `screen_public_pairs.py` | fingerprint gpt2/distilgpt2 and Qwen base/instruct — no training, the cheapest evidence in the project |
| `evaluate_transformation.py` | checkpoints → `checkpoint_metrics.csv` (`05` §2), resumable; `--evaluate-public-reference` writes the seedless P8M rows (`08` §4); `--base` fills E6B's `*_drift_from_base` |
| `aggregate_transformation.py` | contrasts, matched-loss, drift, figures, `go_no_go.json`, `e6b_go_no_go.json` |
| `check_pilot_gate.py` | the four §8.5 pilot criteria → `pilot_gate.json`; criteria 2/3 name their corpus and step explicitly (trap 23), criterion 4 checks the example count (trap 25) |
| `run_pilot_parity.py` | stages a checkpoint + its tokenizer, then calls the **frozen** `run_parity_check` on the design's **five** examples (the frozen CLI's default is three — trap 25) → the `parity_report.json` the gate's criterion 4 reads |
| `configs/e6_preregistration.yaml` | **one of the three places a threshold or criterion lives** (`e6_prereg_v4`) — the TinyStories arm |
| `configs/e6a_gpt2_preregistration.yaml` | the second (`e6a_gpt2_prereg_v1`) — the gpt2 arm, incl. its `pilot_gate` block and criterion 2b |
| `configs/e6a_gpt2_{ce,logit_kd,logit_attention_kd}.yaml` | the gpt2 arm's G0/G1/G2; identical to the TinyStories configs except teacher, student, corpus and layer map |
| `configs/e6b_f1..f4.yaml` | the E6B 2×2: (LoRA vs full) × (clean vs corrupt), effective batch fixed at 32 |

**The two E6A arms are isolated by `experiment_id`**, because both aggregators `rglob` the
results tree and do not de-duplicate. Pass `--experiment e6a_gpt2` (aggregator, gate) and its
own `--preregistration`; results land under `results/e6a_gpt2/`. Nothing in
`e6_preregistration.yaml` moves — the gpt2 arm is a **new registration, not an amendment**.

### E7 pipeline (`crosslingual_semantics/`, WP8 + WP9 + WP11 + WP12 shipped)

| Script | Purpose |
|---|---|
| `prepare_flores_manifest.py` / `prepare_xnli_manifest.py` | the parallel joins, patch controls, dev/test partitions |
| `extract_sink_representations.py` | per-language `R0/K0/V0/Qmean/...` → `.npy` memmaps (`04` §3) |
| `evaluate_crosslingual_retrieval.py` | cosine + Procrustes retrieval, language probe (`04` §4) |
| `run_multilingual_fingerprint.py` | the per-language intervention battery (`04` §3.4) |
| `run_cross_language_patching.py` | screen on dev → window on test → `patching_per_example.csv` (`04` §5) |
| `aggregate_crosslingual.py` | six contrasts, claim gate, `e7_go_no_go.json` (`04` §6) |
| `configs/e7_qwen*.yaml` | the instrument: model, dtype, objects, screening layers |
| `configs/e7_preregistration.yaml` | **the other place a threshold or criterion lives** (`e7_prereg_v4`) |

`COMMANDS.md` at the repo root is the runnable version of all of this: environment
bootstrap, the Phase 0 gates, and every command from the pilot through E7, in dependency
order.

---

## Hard rules

### 1. Frozen files — never edit

The twelve files listed in `BASELINE_HASHES.json` are read-only for the life of this
project. New code **calls into** them; it never modifies them.

```
common/intervention_analysis.py
common/intervention_analysis_legacy.py
common/residual_sink_analysis.py
common/residual_sink_analysis_legacy.py
common/experiments_single_input.py
cross_scale_and_architecture/neo/intervention_analysis_neo.py
cross_scale_and_architecture/opt/intervention_analysis_opt.py
cross_scale_and_architecture/qwen/intervention_analysis_qwen.py
cross_scale_and_architecture/run_table1_multiseed.py
emergence_dynamics/emergence_dynamics_analysis.py
evaluation_robustness/evaluation_robustness_analysis.py
reproduce_paper/experiments_statistical.py
```

If a task seems to require editing one, stop and say so instead. There is almost always
a wrapper-based alternative, and `tests/test_frozen_files.py` will fail the build anyway.

### 2. Additive-only edits to `common/nnsight_engine.py` and `common/datasets_loader.py`

New functions, or new keyword arguments whose defaults reproduce current behaviour
exactly. Never change an existing signature's semantics.

`common/corpus_providers.py` follows the same discipline in practice: its providers are
joined on by `manifest_sha256`, so changing what one returns invalidates every cached
record built from it.

### 3. Dispatch, never reimplement

If you find yourself writing an attention computation, an intervention, a cross-entropy,
or a band reduction — stop. It already exists in a frozen file. Import and call it.

### 4. Never fabricate numbers

No placeholder results, no illustrative values in a results file, no "example" metrics.
Every number in an output artefact must come from code that actually ran. If a value
isn't available, write a sentinel and say so.

This extends to **pre-registered criteria**. Design §8.7, §10.11, §10.12 and §18 are now
transcribed — `e6_prereg_v3` and `e7_prereg_v3` — and every transcribed entry carries a
`source:` line naming its section. They live in three YAML files and nowhere else:

- `transformation_inheritance/configs/e6_preregistration.yaml` (`e6_prereg_v4`, TinyStories arm)
- `transformation_inheritance/configs/e6a_gpt2_preregistration.yaml` (`e6a_gpt2_prereg_v1`, gpt2 arm)
- `crosslingual_semantics/configs/e7_preregistration.yaml` (`e7_prereg_v4`)

**The seven open decisions (D1–D7) are now made**, on 2026-07-30, before any E6 or E7 run
existed. `e6_prereg_v4` and `e7_prereg_v4` carry them, each with a `decided:` date, a
`decision_id:`, a `decision_taken:` argument, and a matching `amendments:` entry of
`kind: decision`. `new_design_plans/DECISIONS_REQUIRED.md` is retained as the record of the
argument on each side. In summary:

| ID | Resolution |
|---|---|
| D1 | primary topology metric = **Wasserstein** (depth-location sensitive; §24's claim is about *where* the anchor sits). e6a_c3 is a *distance*, so its sign is inverted relative to c1/c2 — label it. |
| D2/D3 | `e6a_3`/`e6a_4` = **framing (B)**, a calibrated null: threshold = anchor ± k·(D0's across-seed spread), k=2. The RULE is registered; the NUMBER is a result. |
| D4 | `e6b_1` = **0.02 fixed** (SST-2's own binomial SE — an external anchor); `e6b_2` = framing (B) on F1's spread, reusing `onset_threshold_k`. |
| D5 | `e7_c6` = **route C, exploratory**. The Qwen2.5 report (arXiv:2412.15115v2 §3.1) was read and gives no per-language token share and no language list, so route A is closed and no defensible tiering exists. |
| D6 | `criterion_3_combine` = **`any`** (criterion 1 says "K0 or V0"; §10.12 expects them to differ). |
| D7 | `semantic_sensitivity` = 0.10 (a *reading* of §18 criterion 3); `unrelated_transfer` = 0.10 (**judgement, no design anchor — say so in the paper**). |

**Do not re-open one after seeing results.** A threshold rewritten once numbers exist is not
a pre-registration (`05` §6); an amendment must carry a date and a reason and will be read
as post-hoc unless the ordering is defensible.

Both aggregators still sweep for `PENDING_DECISION_*` sentinels: any that reappears is
reported in `pending_decisions`, and those that can change a verdict carry
`blocks_decision: true` and force `decision: "incomplete"` (and, for E7, `supported: null`).
Because no shipped entry exercises that path any more, the mechanism is asserted on an
*injected* sentinel — `test_injecting_a_sentinel_into_the_shipped_file_still_blocks` and
`test_reopening_a_decision_blocks_the_verdict_again` — so deleting the reader still fails
the suite (trap 11). Neither aggregator hard-codes anything;
`tests/test_aggregate_contracts.py` and `tests/test_e7_aggregate_contracts.py` assert that
a threshold change in the YAML changes the verdict.

A consequence worth planning around: framing (B) **refuses a spread from fewer than three
seeds**, so `e6a_3`, `e6a_4` and `e6b_2` report `met: null` — and `go_no_go.json` reads
`incomplete` — until D0 (and, for E6B, F1) have run at seeds 0, 1 and 2. That is the
registered rule behaving correctly, not a failure to work around.

### 5. Failures are data

No pipeline drops an example silently. Write the row with `status` and `warning` set.

### 6. Provenance on everything

Every output artefact carries: git sha, manifest sha256, seed, dtype, device,
registry version. See `05_SCHEMAS_AND_CONTRACTS.md`.

---

## Working style

- **One work package per session.** `/clear` between packages. The specs are written to
  be loaded one at a time.
- **Plan before code.** For any non-trivial WP, propose the plan and file list first and
  wait for confirmation.
- Complete, runnable, modular code with type hints and docstrings. No `TODO`, no
  `pass  # implement later`, no truncated functions.
- State assumptions, edge cases, and limitations in comments where the code makes a
  non-obvious choice.
- Ship the tests named in `06_TEST_PLAN.md` in the same change as the module.
- If a spec requirement is ambiguous or looks wrong, say so before implementing.

---

## Tests

Run before claiming a WP is done:

```bash
python -m pytest tests/ -x -q
```

The Phase 0 gate (`06_TEST_PLAN.md` §2) must be green before **any** real-model
experiment runs.

Never loosen a tolerance to make a test pass. If a tolerance genuinely needs revisiting,
say so explicitly and justify it.

Tests assert invariants, determinism, and agreement with the frozen instrument — never
that a scientific quantity takes a particular value. Do not write a test asserting a
hypothesis; a null result would then look like a bug.

---

## Environment

- Target hardware: 1× RTX 4080 SUPER, 16 GB VRAM. Assume nothing else fits alongside.
- Repo root is on Windows; paths in code must use `pathlib`, never hard-coded separators.
- Imports use the existing dual-import idiom (`from . import x` / `import x` fallback)
  because scripts put `common/` on `sys.path`. Do not add `pyproject.toml` or `setup.py`
  — that changes import resolution for every existing script.
- Long runs must be resumable and idempotent. E6A alone is 1.5–3 GPU-days.

---

## Twenty-seven known traps

1. **`compute_band(4, "scaled")` returns `(3, 4)`** — a single layer for the 4-layer
   TinyStories teacher, vs four layers for the 8-layer student. Use `depth_band.py` for
   all E6/E7 work. Legacy `compute_band` is for E1–E5 reproduction only.
2. **XNLI is not positionally aligned across languages.** Join on `promptID`. A
   positional join produces a manifest that looks fine and destroys every cross-language
   claim. Note that HF's `facebook/xnli` per-language configs expose *no* `promptID`;
   `datasets_loader.load_xnli_aligned` therefore falls back to the structurally aligned
   `all_languages` config and records which strategy it used. It never joins by position.
3. **Identity and self-source patches must be exact no-ops.** A patching bug produces
   smooth, plausible, entirely fictitious effect sizes rather than a crash. Run the
   identity control continuously in production, not only in the test suite.
4. **On Qwen, "the key" is two different tensors.** RoPE is applied *inside*
   `Qwen2Attention.forward`, so a patch object must say `K0_prerope` or `K0_postrope`.
   `cross_example_patching` refuses a bare `K0`/`Kmid` on RoPE architectures rather than
   picking one. At position 0 the rotation is the identity — verified numerically, never
   assumed, because it does not hold at `Kmid`.
5. **`*mid` patch positions resolve from the prompt, not prompt+candidate.** Label scoring
   appends candidate tokens of differing lengths; resolving `seq_len // 2` after the
   append silently moves the patch site for every candidate.
6. **`save_pretrained` drops tied weights.** GPT-Neo ties `lm_head` to `wte`, so a
   checkpoint's `state_dict` has fewer keys than the model's. A strict load fails on every
   resume, and a blanket `strict=False` lets a truncated checkpoint resume with
   randomly-initialised layers. Load non-strictly and *verify* the missing keys are tied.
7. **A checkpoint directory contains no tokenizer.** `save_pretrained` on a model writes
   weights and config only, so anything loading a checkpoint must be told where the
   tokenizer is — `load_handle(..., tokenizer_name=...)`. Without it, loading falls through
   to an HF id that does not resolve offline.
8. **Progress bars go to stderr, and PowerShell 5.1 treats that as an error.**
   `& python script.py 2>&1 | Tee-Object` under `$ErrorActionPreference = "Stop"` wraps
   each stderr line in a terminating `NativeCommandError` even when the process exits 0.
   Relax the preference around the native call and take the verdict from `$LASTEXITCODE`.
9. **An identity violation invalidates the run *unit*, not just the row.** This is trap 3's
   production half. `run_patched` only *detects* it (`status="identity_violation"`);
   `run_cross_language_patching.py` must abort the unit, stamp every one of its rows
   `unit_status="identity_violation"`, record it in `invalid_units.json` and exit non-zero.
   Identity is therefore scheduled **first** at every site, so a broken unit costs one
   example rather than the whole partition. Never aggregate around one.
10. **A reference artefact belongs to a model.** The repo tree can hold
    `bos_attention_stats_overall.csv` for gpt2, gpt2-medium/large, three GPT-Neo sizes and
    Qwen. Picking one by `rglob` order compares GPT-2's fingerprint against GPT-Neo's:
    most interventions land close enough to look fine and one diverges, which reads as a
    seam regression rather than a mismatched comparand. Check the sibling
    `run_config.json`'s `model_name` before comparing — `05` §7.1, hash before compare.
11. **A pre-registration field the aggregator does not read is worse than one that
    crashes.** `seed_consistency: all` is the case that made this concrete: ignored, a
    criterion passes on one strong seed and two weak ones and the artefact says nothing.
    Every field WP12 added is optional *and* asserted — a pre-registration omitting it
    produces byte-identical output, and a pre-registration naming it has a test proving it
    changes the verdict. When adding a YAML field, ship the test that fails if the reader
    is deleted.
12. **The E6 corpus filter runs before contrasts.** `aggregate()` used to hand
    `build_contrasts` a frame already narrowed by `DEFAULT_CORPUS_PREFIXES`, which drops
    `e1_100x40` — so design §8.7 contrast 5 (in-domain versus cross-domain) would have been
    computable on paper and permanently `no_data` in practice. Contrasts now see the
    unfiltered frame and select per entry; matched loss, table 2 and the figures still read
    the narrowed one, so no existing number moved.
13. **A half-read pre-registration field is worse than an unread one.** `str(spec.get("x")
    or "default")` on a decision key silently resolves the decision *and* records a value
    it did not honour — `combine: ALL_` folded as `any` while the artefact said `"all_"`,
    which looks like an audit trail and is not one. An absent key may default only where a
    spec names the default; a present-but-unrecognised one is always refused.
14. **`merge_and_unload()` mutates the wrapper in place** and returns the unwrapped model,
    so the two share tensors. Holding a reference to the PEFT model and comparing after
    merging compares a model against itself and reports `0.0` for any adapter, broken or
    not. Capture the unmerged logits **before** merging. This is trap 3's shape in a
    different module: the vacuous check that always passes.
15. **`03` §4.5's merge-parity bar is absolute, and real logits are not O(1).**
    distilgpt2's reach ~104, so 1e-5 absolute demands ~1e-7 relative — below float32's own
    epsilon, i.e. unreachable by *any* correct implementation. Measured: fp32 gives
    1.07e-04 (relative 1.03e-06), float64 gives 1.99e-13 — both ≈8.6× their own eps. The
    fix is to raise the working precision, never the tolerance: a real merge fault is
    systematic and fails 1e-5 at any precision. A tiny smoke model hides this entirely.
16. **A verdict read from a key the producer does not write is silently always-false.**
    `check_pilot_gate.criterion_4` looked for `passed`, then `all_within_tolerance`; the
    frozen `run_parity_check` writes **`all_rows_pass`**. So `bool(payload.get(..., False))`
    returned `False` on a *passing* parity report and the §8.5 gate could never reach
    `proceed: true`. It read as a scientific failure and would have been "fixed" by
    loosening something. Two rules follow: a missing verdict is `met: null` with a reason,
    never a defaulted `False`; and the test must build its fixture in the **producer's**
    shape — `_parity()` wrote `{"passed": ...}`, the shape the *reader* expected, which is
    why the defect survived a green suite.
17. **A calibrated threshold must be calibrated on the data it judges.**
    `resolve_threshold_rule` scoped its null spread with a hard-coded `corpus_override=None`
    while the contrast beside it honoured the caller's `--corpus`. The criterion then
    compared a value measured on one corpus against a spread computed on another — a
    threshold that *looks* calibrated to the number it judges and is not. Invisible in
    production (where the override is `None`) and only reachable through `--corpus`, which
    is exactly what the smoke run uses.
18. **A test that asserts the spec instead of the code contradicts the code.**
    `nnsight_e6b_smoke.py` asserted `merge_parity.check_dtype == "float32"` — `03` §4.5's
    value — while the implementation records `check_dtype: "float64"` beside
    `spec_check_dtype: "float32"` precisely so the trap-15 deviation stays auditable. Where
    an artefact deliberately records both the spec and the reality, assert **both**, and
    assert which is which.
19. **The E6A training corpus does not fit in RAM as Python objects.** TinyStories' train
    split is 2,119,719 stories → 457M tokens → **3.57M blocks** of 128. Held as lists of
    Python ints and *duplicated* into the block manifest, that is ~20 GB (measured: 5,630
    B/block); `prov.sha256_json` over it adds an ~8.8 GB spike and ~12 minutes, and
    `write_block_manifest` then builds a 3.57M-row DataFrame. Peak ~29 GB — it froze a
    16 GB host, and no test caught it because only the 5-step `--smoke` path ever ran.
    `load_tinystories_blocks(as_array=True, manifest_input_ids=False)` +
    `prov.sha256_int_rows` + chunked parquet bring it to ~3 GB with **the same tokens, the
    same order and the same `manifest_sha256`**. Two sub-traps worth naming: converting
    the list to an array *at the end* saves nothing (the list-of-lists IS the peak — blocks
    must be folded into int32 as they are produced), and a streaming digest must reproduce
    `sha256_json`'s exact bytes or it silently partitions artefacts into two incomparable
    sets under `05` §7.1. Related operational trap: **`Get-Process ... | tail -2` truncates
    the table** and made a live 3.7 GB run look dead, so a second run was launched on top of
    it — two processes packing the same corpus into one run directory. Count processes
    explicitly (`@(Get-Process python).Count`) before concluding anything died.
20. **`LambdaLR.state_dict()` does not include the lambda.** So `load_checkpoint` restores
    `last_epoch` but not the schedule that produced it, and the lambda here closes over
    `max_steps`. Resuming a run under a different horizon therefore splices two cosines: the
    LR jumps at the resume point and the trajectory is neither schedule. This was reachable
    through the **documented** workflow — the pilot runs `--max-steps 2000`, Phase 2 re-runs
    the same config in the same directory at 10,000 — so the production run would have been
    2,000 steps of one cosine followed by 8,000 of another, with nothing in the artefacts
    saying so. Design §1.6 pre-registers the schedule and forbids "improving" the
    hyperparameters; a spliced schedule is not the registered one.
    `assert_resumable_schedule` refuses it and names the fix (fresh run directory, or
    `--resume none`). Note it must run **before** `write_run_config`, which overwrites the
    field it reads.
21. **Resolving a decision deletes the only live instance of its refusal path.** Removing
    the last `PENDING_DECISION_*` sentinel from the shipped YAML means no test exercises
    the blocking mechanism any more, so the reader could be deleted and the suite would
    stay green — trap 11 wearing a different hat. When you resolve the last sentinel of a
    kind, add a test that *injects* one back.
22. **A loss's reduction *axes* are part of its definition, and getting them wrong rescales
    a pre-registered weight.** `03` §1.5 defines `L_ATTN` as a mean JSD over **layer, head
    and query** — a JSD sums over the key axis, because that is the axis the two
    distributions live on. `attn_js_loss` divided by the number of valid `(query, key)`
    *cells*, which is the same quantity divided by the mean valid keys per query: exactly
    `(S+1)/2`, i.e. **64.5× at S=128**. Nothing crashed, nothing looked wrong, and the
    logged `l_attn` ≈ 0.005 was internally consistent — but D2's registered `0.10 * L_ATTN`
    was 0.022% of its loss instead of ~1.4%, so the first E6A pilot's attention arm was
    inert and D2 came out indistinguishable from D1 (|Δ fingerprint cosine| = 1.3e-4). The
    fix raises the denominator, never the weight: the weight is pre-registered and the
    denominator was a defect. `run_config.json` now records `attn_reduction`, because
    without it a pre-fix run directory and a corrected one are indistinguishable. Note D0
    and D1 carry `attn_weight = 0`, so `0.0 * l_attn` leaves their checkpoints bit-identical
    — only D2 needs retraining, and only their logged metric rescales.
23. **`checkpoint_metrics.csv` has one row per `(step, corpus_id)`, so a criterion that
    names neither reads whichever row the writer ordered first.** `check_pilot_gate`
    criterion 2 filtered on `checkpoint_step <= pilot_step` and took `baseline_sink.max()`,
    which on the first real pilot returned the **step-0 cross-domain** value — a
    random-initialisation number reported as a trained in-domain one, identical across
    D0/D1/D2 because the three share an init, which is the tell. Criterion 3 took
    `.iloc[0]`. Both now select the corpus explicitly and refuse an ambiguous selection, and
    criterion 2 reports all three readings of "by step 2000" (`at_pilot_step`,
    `max_through_pilot_step` over *trained* steps, `max_including_step_0`) rather than
    resolving the wording silently. Step 0 is saved *before* the first optimiser update, so
    it can never satisfy a criterion about emergence.
24. **A cache path must carry every key field the caller *sweeps*.** `fingerprint_runner`
    wrote `<cache>/<run_id>/step_<n>/fingerprint.json` with no corpus in the path, while
    `evaluate_transformation.py` loops corpora **inside** one step — so the second corpus
    overwrote the first and the entire first pilot archive holds `e1_100x40` for all 33
    fingerprints, including the teacher's on the corpus every criterion is scored on. The
    `_cache_key` check means the survivor is never *read* as the wrong corpus, so no number
    was ever wrong; what was destroyed is the audit trail `05` §7.1 requires. Band and
    intervention-set collisions are left alone deliberately — nothing sweeps them, so they
    cost a recompute, not an artefact.
25. **A criterion's own text can name a quantity the code never checks.** Criterion 4 reads
    "on **5** examples"; the frozen `PARITY_SENTENCES` default is three and the frozen Neo
    CLI passes no `sentences=`, so the report said `n_sentences: 3` while the gate recorded
    `n_rows: 10` — *interventions*, not examples — and returned `met: true`. Two rules: when
    a criterion's wording contains a count, a threshold or a corpus, the code must read it
    from the artefact and refuse (`met: null`) when it falls short; and where a frozen driver
    cannot be given what the criterion needs, wrap it — `run_pilot_parity.py` now calls the
    frozen `run_parity_check` in-process with five sentences and every operand still built by
    the frozen module's own functions.
26. **A threshold is only meaningful if the *teacher* clears it — measure the teacher on the
    scored corpus before training the arm.** E6A ran five 2,000-step runs and a full gate
    before anyone measured `roneneldan/TinyStories-33M`'s own sink on
    `tinystories_validation_sink_300`. It is **0.009309** — 0.86× the uniform-attention
    floor (`(H_L − H_{L/2})/(L−L/2)` ≈ 1.386/L = 0.010770 at 128 tokens), with **zero** cells
    above 0.2. Criterion 2's 0.15 bar was 16.1× the teacher's own value and 1.7× above it
    even cross-domain, so no student could ever have met it, at any step count, on any
    corpus — and the criterion's failure said nothing about *inheritance*, which is what E6A
    exists to measure. Two consequences, both now shipped: the pre-flight fingerprint of the
    teacher is step 0 of an arm's runbook (`--evaluate-public-reference <teacher>`, ~30 min,
    no training), and the gpt2 arm registers a **teacher-relative** criterion 2b beside the
    absolute one. Note the corollary about the metric itself: `baseline_sink` at random init
    *is* the uniform floor (measured 1.016× at L=40, 1.059× at L=128), so raw sinks are not
    comparable across corpora of different sequence length — the 40-token corpus's floor is
    3.16× the 128-token one's, before any model has learned anything.
27. **A pure function of the epoch, called once per micro-batch, made the whole project
    CPU-bound.** `epoch_order(n_blocks, seed, epoch)` builds `list(range(n_blocks))` and
    shuffles it in Python; the training loop called it inside the gradient-accumulation
    loop, read `batch_size` elements, and threw the rest away. `epoch` advances only every
    `n_blocks / batch_size` micro-batches — ~194,000 for the gpt2 arm — so within a
    10,000-step run it **never advances** and the identical 3.1M-element shuffle was
    recomputed 40,000 times. Measured: **1.09 s** at 3.1M blocks (OpenWebText), 1.30 s at
    3.57M (TinyStories), ×4 accumulation = **4.4–5.2 s of single-threaded CPU per step**
    with the GPU idle. That is what the pilot's "4.85 s/step" was, and it is why the
    E6A-GPT2 runs showed **12% GPU utilisation on a 4090 and 3% on a 5090**. Measured after
    memoising it, real gpt2→distilgpt2 at 3.1M blocks on an RTX 2060: **6.96 → 1.96 s/step,
    GPU utilisation 29% → 96% (median 99%), identical peak VRAM** — 1.96 s/step is the
    isolated GPU-side cost, so the loop is now GPU-bound. Three lessons, all general:
    a hot loop must be profiled against the *production* `n_blocks` (every smoke path here
    used 64 blocks, where the shuffle costs 15 µs and is invisible); `.item()` in a loop is
    a pipeline stall, not a read (`compute_losses` plus `attn_js_loss` made **88** GPU→CPU
    syncs per optimiser step, now 5); and a metric computed under `torch.enable_grad` when
    its weight is zero — `L_ATTN` for D0/D1/G0/G1 — costs real time (measured 11%) for a
    graph nothing traverses. **Every fix here is value-preserving and was gated on
    byte-identical `train_log.jsonl`, `eval_log.jsonl` and checkpoint sha256 across a
    20-step real-CUDA run**, which is the only reason the already-spent G0/G2 seed-0 runs
    remain poolable. Never "optimise" a training loop without that gate: a 1-ULP drift is
    indistinguishable from a scientific result.
