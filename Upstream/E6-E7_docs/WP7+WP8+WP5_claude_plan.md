# WP7 + WP8 + WP5 — Sink Inheritance (E6/E7)

## Context

`sink-inheritance-foundation` has WP0–WP4 done and smoke-verified (`NEXT_STEPS.md` §1); the
two compute-PC bridge gates in §2 have been run and passed. §3 lists what remains. This
plan implements the next three work packages in the master-plan DAG order:

| WP | Closes | Deliverable |
|---|---|---|
| **WP7** | G4 (no cross-example patching) | `common/cross_example_patching.py` — the source-capture → target-inject machinery every E7 causal claim rests on |
| **WP8** | E7 data | `common/paired_manifests.py` + the two `prepare_*_manifest.py` scripts + the deferred FLORES/XNLI loaders and corpus providers |
| **WP5** | G7 (no training code at all) | `transformation_inheritance/train_distillation.py` + 3 configs + `check_pilot_gate.py` (E6A) |

Intended outcome: Phase 0 is complete except WP6/WP9–WP11, E7 has real parallel manifests,
and E6A can start training on the compute PC.

**Environment note (blocking, resolved by user decision).** This checkout has no `.venv`
and the global interpreter lacks `nnsight` and `peft`; `pytest tests/ -q` currently gives
17 failed / 3 errors, all `ModuleNotFoundError: No module named 'nnsight'`. Step 0 below
creates a fresh `.venv` and installs `requirements.txt` into it; **every** command after
that runs as `./.venv/Scripts/python.exe`.

---

## Step 0 — environment (do first, everything depends on it)

```bash
py -3.12 -m venv .venv
./.venv/Scripts/python.exe -m pip install --upgrade pip
./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests/ -q          # expect 55 passed, 2 skipped
```

`requirements.txt` pins `torch==2.10.0+cu128` — a multi-GB download; expect this step to
dominate wall-clock. Confirm the pre-existing suite is green **before** writing any new
code, so a later failure is attributable to new code rather than to the environment.

Append-only edit to `requirements.txt`: add `pyarrow` (WP5 writes the repo's first
`.parquet`; pandas needs it and it is currently only a transitive dep). No reordering, no
re-pinning — same rule WP0 followed.

---

## WP7 — `common/cross_example_patching.py`

Spec: `02_MODULE_SPEC_common.md` §6. Prompt P7. Tests: `06_TEST_PLAN.md` rows 31–32, 39.

### Design

Self-contained two-trace implementation that **reuses** rather than reimplements:

- `nnsight_engine._resolve`, `ArchSpec` paths (`k_path`/`v_path`/`o_path`/`blocks_path`/
  `attn_path`), `NNsightEngine._payload` (mandatory `attention_mask`, RoPE `position_ids`)
  and the `EditPlan("int_a")` clean plan — all from [nnsight_engine.py](common/nnsight_engine.py).
- Frozen `compute_bos_attention_metric` from
  [intervention_analysis_legacy.py:633](common/intervention_analysis_legacy.py#L633) for
  every sink number; `fingerprint_runner._per_head_bos` for carrier-head sink.
- Frozen Qwen RoPE/GQA primitives `build_rope_cos_sin` / `_rotate_half` / `_apply_rope` /
  `_repeat_kv` from
  [intervention_analysis_qwen.py:236-279](cross_scale_and_architecture/qwen/intervention_analysis_qwen.py#L236-L279),
  imported via the lazy `sys.path.insert` idiom already used at
  [fingerprint_runner.py:200-205](common/fingerprint_runner.py#L200-L205).

**Zero edits to `nnsight_engine.py`.** `run_arch_trace(capture_position0=...)` keeps
raising `NotImplementedError` — `test_arch_trace_ce.py::test_capture_position0_not_implemented`
asserts that, and a write path cannot be expressed through `run_arch_trace` anyway. This
module supersedes it; a comment in the new module says so.

### Public surface

```python
PATCH_REGISTRY_VERSION = "patch_v1"
PATCH_OBJECTS  = ("K0_prerope","K0_postrope","V0","R0",
                  "Kmid_prerope","Kmid_postrope","Vmid","Rmid")
NORM_CONDITIONS = ("direct","rescaled","random","identity")

@dataclass(frozen=True) class PatchSite:  object: str; layer: int; position: int
@dataclass(frozen=True) class PatchSpec:  site: PatchSite; norm_condition: str
                                          source_item_id: Optional[str]; seed: int
@dataclass          class SourceCache:    model_fingerprint: str; item_id: str
                                          tensors: Dict[str, torch.Tensor]
                                          resolved_positions: Dict[str, int]   # additive
                                          seq_len: int; provenance: dict

def model_fingerprint(handle) -> str
def capture_sources(handle, corpus, sites, *, batch_size=1, to_cpu=True,
                    dtype=torch.float32) -> Dict[str, SourceCache]
def run_patched(handle, target_item, patch_specs, source_caches, *,
                score_candidates=None, capture_sink=True, band=None,
                carrier_heads=None, capture_block_outputs=()) -> dict
# pure scorer, model-free, unit-testable:
def sequence_logprob(logits, prompt_len, candidate_ids) -> (total, normalised)
def correct_label_margin(scores, gold_index) -> (normalised, unnormalised)
```

### Decisions that need to be visible

1. **`Kmid` is realised as `Kmid_prerope` / `Kmid_postrope`.** `05` §3 lists `Kmid` as a
   `patch_object` value, but `02` §6.4 requires both RoPE variants to be reported and never
   mixed. Bare `K0`/`Kmid` are accepted only when `spec.rope_applied_inside_attn is False`
   (unambiguous there); on Qwen they raise, naming the two variants. This adds *values*, not
   columns — `05`'s "nothing is renamed" contract holds.
2. **`K*_postrope` patching uses inverse rotation.** Write
   `k_pre = R(target_pos)⁻¹ · k_post_source`, with `R⁻¹` = `k*cos − rotate_half(k)*sin`
   (10 lines, reusing the frozen `_rotate_half`). At position 0 this reduces to the pre-RoPE
   case; the test **measures** that max-abs difference rather than assuming it, per §6.4.
3. **Invariant 4 (pre-`_repeat_kv`) is proven by inverting `o_proj`,** not by reading
   `.input` (an nnsight surface this repo never uses). Patch one KV head's `V0`, capture
   `attn.output[0]` (already-used surface), recover `Δcontext = Δout @ inv(W_o).T`, reshape
   to `[n_heads, head_dim]`, and assert exactly heads `j*n_rep … j*n_rep+n_rep-1` are
   non-zero — which tests the `_repeat_kv` ordering itself, not just the head count.
4. **Batching.** The internal writer is batch-index-aware; `run_patched` drives it with a
   1-item batch (production default). A private `_traced_forward` accepting a multi-item
   batch exists so invariant 7 tests the *same* code path. Batched patching asserts equal
   lengths — padding would move position 0.
5. **`*mid` resolves from the prompt length, not prompt+candidate.** Scoring appends
   candidate tokens, so a naive `seq_len // 2` would shift the patch site per candidate.
   Resolved positions are recorded on every output row.
6. **Arch coverage.** `R*` works on all four archs. `K*`/`V*` require
   `qkv_layout == "separate"` (neo/opt/qwen); GPT-2's fused `c_attn` raises a clear
   `NotImplementedError` — E7 is Qwen-only and a fused-slice write path would be untested risk.
7. **`model_fingerprint`** does not exist anywhere in the repo. New: sha256 over canonical
   JSON of `{arch, model_name, model_revision, checkpoint_step, checkpoint_sha256, dtype,
   num_layers, num_heads, num_kv_heads, hidden}` plus a cheap deterministic weight digest
   (per sorted param: name, shape, first/last 8 values). Mismatch in `run_patched` **raises**
   (invariant 8).
8. **Sink capture** requests full `[H,S,S]` maps for all layers only when
   `capture_sink=True`, reduces immediately with the frozen metric, then frees. Worst case
   ~34 MB fp32 for a 24-layer Qwen at seq 160 — acceptable at `batch_size=1`, and the only
   way to dispatch to the frozen reducer (CLAUDE.md rule 3).
9. **Failures are rows.** Non-finite logits / OOM / tokenisation failure → `status` +
   `warning` in the returned dict, never an exception that drops the example (§6.5).

### Files

- `common/cross_example_patching.py` (new)
- `tests/test_cross_example_patching.py` (new) — the nine invariants of `02` §6.2, on a random
  Qwen2 from `create_tiny_qwen_checkpoint` (`n_head=8, n_kv_head=2` — already the default)
- `tests/test_patch_scoring.py` (new) — hand-computed log-probs on a toy logits tensor, margin
  sign convention, one scorer shared by baseline and patched
- `tests/nnsight_e7_smoke.py` (new) — standalone `main()` in the `nnsight_e*_smoke.py` house
  style: all patch objects × all norm conditions on a random Qwen2, CPU/fp32, no network
- `scripts/run_nnsight_rest_smoke_tests.ps1` — append E6 and E7 (E6 was never added)

---

## WP8 — parallel manifests

Spec: `02` §5, `04` §1–2. Prompt P8. Tests: `06` rows 33–35.

### Files

- `common/paired_manifests.py` (new) — `ParallelManifest`, `build_flores_manifest`,
  `build_length_matched_subset` (scipy `linear_sum_assignment`, ≤20% rel. diff),
  `build_xnli_manifest`, `assign_patch_controls`, `grouped_partition`, `verify_manifest`
- `common/datasets_loader.py` (**additive only**) — `load_flores_parallel`,
  `load_xnli_aligned`; `load_optional_flores` untouched (it is a distinct contract per
  `01` §6)
- `common/corpus_providers.py` — replace the two `NotImplementedError` stubs at
  [corpus_providers.py:323-336](common/corpus_providers.py#L323-L336) with real
  `flores_corpus` / `xnli_prompt_corpus`, built through the existing `_make_corpus` (the only
  place `manifest_sha256` is set) and preserving `semantic_ids` order exactly
- `crosslingual_semantics/prepare_flores_manifest.py`, `prepare_xnli_manifest.py` (new dir)
- `tests/test_parallel_manifest_alignment.py`, `tests/test_patch_controls.py`,
  `tests/test_grouped_partition.py` (new)

### The XNLI join, stated explicitly

CLAUDE.md trap 2 requires joining on `promptID`, never row index. **HF's `facebook/xnli`
per-language configs expose only `premise`/`hypothesis`/`label` — no `promptID`.**
`load_xnli_aligned` therefore resolves a join strategy at runtime and records it in
`provenance["join_strategy"]`:

1. an explicit id column (`promptID`/`pairID`) when the loaded shards have one → join on it;
2. otherwise the `all_languages` config, where one row *is* the translation set, so alignment
   is structural rather than positional;
3. neither available → raise. **A positional join is never used.**

`test_parallel_manifest_alignment.py` monkeypatches `load_dataset` (the pattern already used
by `test_tinystories_packing.py:48-53`) to return per-language shards **in shuffled order**
with ids attached, and asserts the manifest still aligns and gold labels agree across all
seven languages — a positional join fails that test loudly.

FLORES keeps the spec's sanctioned source-index join (`devtest` is line-aligned), keeps a row
only if **every** language is 12–160 tokens, and sets `provenance["valid"] = False` with a
non-zero exit below 200 surviving rows. Covariates emitted per `04` §1: first-token frequency
over the full devtest split, punctuation-at-position-0
(`unicodedata.category(ch).startswith("P")`), Unicode script, language family (small static
table for the eight languages).

Real data needs downloads → compute PC. All alignment/control/partition logic is unit-tested
offline against monkeypatched loaders.

---

## WP5 — `train_distillation.py` (E6A)

Spec: `03` §1–3. Prompt P5. Tests: `06` rows 25–27.

### Files

- `transformation_inheritance/train_distillation.py` (new)
- `transformation_inheritance/check_pilot_gate.py` (new)
- `transformation_inheritance/configs/e6a_ce.yaml`, `e6a_logit_kd.yaml`,
  `e6a_logit_attention_kd.yaml` (new) — identical except the three loss weights
- `tests/test_distillation_loss.py`, `tests/test_distillation_init.py`,
  `tests/test_resume_exactness.py` (new)

### Points the tests will police

- **Startup assertions (D6)** across teacher / public-8M config / student config: vocab size,
  `len(tokenizer)`, BOS/EOS ids, head count == 16. Abort on mismatch; record in
  `run_config.json["vocab_assertions"]`. Also read and record `config.attention_layers` and
  `config.window_size` from both models — never assume the global/local pattern.
- **Student is random-init from the TinyStories-8M *config*** (`AutoModelForCausalLM.from_config`),
  never `from_pretrained`. Seed `torch`/`numpy`/`random`/`cuda` before construction; record
  `param_count` and a sha256 of the initial state dict so D0/D1/D2 at one seed are provably
  byte-identical.
- **`attn_js_loss` masks by exclusion, never renormalisation** (`06` §3 calls this out as
  carrying unusual weight). `valid_mask[t,s]` = elementwise AND of both layers'
  causal/window masks; returns a per-pair breakdown so global-mapped and local-mapped pairs
  report separately. Layer map 0→1, 1→3, 2→5, 3→7.
- **`kd_loss` applies `T**2` exactly once**; teacher forward under `torch.no_grad()`, teacher
  `eval()` + `requires_grad_(False)`, bf16.
- **Step-0 checkpoint saved by an explicit call before the loop**, not a branch inside it.
- Checkpoints carry model + optimiser + scheduler + RNG state (`torch`, `cuda`, `numpy`,
  `random`) so `--resume auto` is bit-exact; `checkpoint_sha256.txt` alongside.
- `block_manifest.parquet` from the existing
  [`load_tinystories_blocks`](common/datasets_loader.py#L385) output, hash recorded — conditions
  sharing a seed must share it exactly.
- Hyperparameters exactly `03` §1.6 — pre-registered, not to be "improved".
- `runtime_estimate.json` after step 100 in the `05` §5 schema.
- **New:** a git-sha provenance helper (none exists in the repo — `git_sha` today is a literal
  in `BASELINE_HASHES.json`). Goes in a new `common/provenance.py`, not in `datasets_loader.py`
  or any frozen file, and is reused by WP7/WP8 artefacts too.
- **`--smoke`**: 5 steps, tiny random GPT-Neo teacher/student, CPU, fp32, no downloads, tiny
  WordLevel tokenizer built in-process (same recipe as `nnsight_smoke_utils._tiny_tokenizer`,
  re-expressed locally rather than importing from `tests/` into production code).

`check_pilot_gate.py` evaluates the four `03` §2 criteria and writes `pilot_gate.json` with an
explicit `proceed: true|false`; on gate-2 failure it prints the design's prescribed remedy
(extend seed 0 to 5,000 steps) rather than concluding failure.

---

## Conventions all new code follows

- Dual-import idiom (`from . import x` / `import x` fallback) in `common/`; the 3-line
  `sys.path.insert` prelude + `# noqa: E402` in `tests/`. No `pyproject.toml`/`setup.py`.
- `pathlib` everywhere; no hard-coded separators.
- Windows-safe teardown in every model-bearing test:
  `TemporaryDirectory(prefix=..., ignore_cleanup_errors=True)` with `del engine; gc.collect()`
  in a `finally` **inside** the context.
- Smoke scripts are standalone `main()` scripts (pytest does not collect `nnsight_*_smoke.py`);
  `test_*.py` files are pytest.
- No fabricated numbers; failures written as rows with `status`/`warning`; provenance
  (git sha, manifest sha256, seed, dtype, device, registry version) on every artefact.
- No test asserts a scientific quantity's value — invariants, determinism, and agreement with
  the frozen instrument only.

---

## Verification

```bash
# 0. environment (once)
./.venv/Scripts/python.exe -m pytest tests/ -q            # baseline: 55 passed, 2 skipped

# 1. full suite after each WP
./.venv/Scripts/python.exe -m pytest tests/ -x -q

# 2. WP7 specifically — the nine invariants + scoring
./.venv/Scripts/python.exe -m pytest tests/test_cross_example_patching.py tests/test_patch_scoring.py -q
./.venv/Scripts/python.exe tests/nnsight_e7_smoke.py

# 3. WP8 — alignment, controls, partitions (all offline)
./.venv/Scripts/python.exe -m pytest tests/test_parallel_manifest_alignment.py \
    tests/test_patch_controls.py tests/test_grouped_partition.py -q

# 4. WP5 — losses, init determinism, bit-exact resume, then a real 5-step smoke run
./.venv/Scripts/python.exe -m pytest tests/test_distillation_loss.py \
    tests/test_distillation_init.py tests/test_resume_exactness.py -q
./.venv/Scripts/python.exe transformation_inheritance/train_distillation.py \
    --config transformation_inheritance/configs/e6a_logit_attention_kd.yaml \
    --seed 0 --smoke --output-dir <scratch>

# 5. frozen-file guard must still be green
./.venv/Scripts/python.exe -m pytest tests/test_frozen_files.py -q

# 6. regression smokes for the earlier WPs
./.venv/Scripts/python.exe tests/nnsight_e6_smoke.py
```

Expected end state: the pre-existing 55 pass plus the new tests, still 2 skips (the
compute-PC bridge gates, which skip on this box because the reference CSVs live elsewhere),
and `test_frozen_files.py` green.

I will report actual counts and any test I could not run, rather than claiming success.

---

## Docs updated at the end

- **`NEXT_STEPS.md`** — move WP7/WP8/WP5 from §3 into §1 with commit refs and test names;
  correct §"Verification split" to reflect the real `.venv` bootstrap and the actual suite
  counts; record the WP8 XNLI join-strategy decision and WP7's `Kmid_prerope`/`Kmid_postrope`
  value split; note that E6A training itself still needs the compute PC; leave WP6/WP9–WP11
  in §3.
- **`CLAUDE.md`** — add a fourth known trap only if implementation surfaces one worth
  recording (current candidate: `*mid` patch positions resolve from the prompt, not
  prompt+candidate, or the E6A step-0-before-first-update rule). Note the new
  `common/provenance.py` and `common/cross_example_patching.py` in the module map if the
  file carries one. No change to the frozen-file list or the hard rules.

## Out of scope

WP6 (E6B / LoRA), WP9, WP10, WP11. Real FLORES/XNLI downloads and the E6A training run
itself (compute PC).
