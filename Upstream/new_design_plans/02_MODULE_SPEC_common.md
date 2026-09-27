# 02 — Module Spec: New `common/` Modules

Six new modules. Signatures are normative — E6/E7 scripts are written against them.
All are pure-Python + torch/numpy; only `fingerprint_runner` and `cross_example_patching`
touch models.

---

## 1. `common/depth_band.py` (WP1)

Fully specified in `01_REFACTOR_SPEC` §3.2. Summary of the public surface:

```python
DEPTH_BAND_DEFAULT = (0.25, 0.90)
DEPTH_BAND_VERSION = "depth_band_v1"

normalised_depth_band(num_layers, frac=DEPTH_BAND_DEFAULT, min_layers=1) -> (int, int, dict)
layer_depths(num_layers) -> np.ndarray
band_agreement_report(num_layers_a, num_layers_b, frac=...) -> dict
```

No model imports. ~80 lines.

---

## 2. `common/corpus_providers.py` (WP1) — closes G1

### 2.1 Purpose

The frozen harnesses bind the corpus to `sample_benchmark_datasets()`. E6/E7 need to
evaluate the same intervention battery on: TinyStories validation blocks, SST-2 prompts,
FLORES sentences per language, XNLI prompts, and the **frozen E1 cross-domain manifest**.
A `Corpus` is the injectable unit.

### 2.2 Types

```python
@dataclass(frozen=True)
class CorpusItem:
    item_id: str            # stable, deterministic; used as a join key everywhere
    text: str               # raw text, post-normalisation
    input_ids: List[int]    # exact ids fed to the model, no special tokens unless declared
    n_tokens: int
    meta: Dict[str, Any]    # language, label, semantic_id, split, source_index, ...

@dataclass(frozen=True)
class Corpus:
    corpus_id: str          # "tinystories_val_sink_300", "flores_devtest_ben_Beng_300", ...
    items: Tuple[CorpusItem, ...]
    tokenizer_name: str
    tokenizer_revision: Optional[str]
    add_special_tokens: bool
    cut_length: Optional[int]     # None if variable length
    seed: int
    manifest_sha256: str          # over the canonical JSON of all items
    provenance: Dict[str, Any]    # dataset name, revision, split, filter params
```

`manifest_sha256` is computed over `json.dumps` of
`[(item_id, input_ids, sorted(meta.items()))]` with `sort_keys=True, ensure_ascii=False`.
This hash goes into every `run_config.json` and is the mechanism by which two runs are
proven to have seen identical inputs.

### 2.3 Providers

```python
def frozen_e1_corpus(tokenizer, sample_size=100, cut_length=40, seed=DEFAULT_SEED) -> Corpus
    # Wraps the EXISTING sample_benchmark_datasets() verbatim.
    # Must reproduce E1's sample_manifest.csv row-for-row. This is the
    # regression bridge between old and new code paths.

def tinystories_corpus(tokenizer, split, n_blocks, block_size=128, seed=0,
                       purpose="sink") -> Corpus
    # purpose in {"sink" (300 blocks), "ppl" (2000 blocks)}; the two are DISJOINT
    # and both derived from the validation split with the same seed.

def sst2_prompt_corpus(tokenizer, split, n=None, max_input_tokens=64, seed=0,
                       corrupted_manifest=None) -> Corpus
    # meta carries: label, label_token_ids, label_span, corrupted (bool),
    # original_label. corrupted_manifest is only ever applied to train.

def flores_corpus(tokenizer, lang, semantic_ids, split="devtest") -> Corpus
    # semantic_ids come from paired_manifests; the corpus is a projection of the
    # already-joined parallel manifest onto one language. Order follows
    # semantic_ids exactly so that row i is the same sentence in every language.

def xnli_prompt_corpus(tokenizer, lang, semantic_ids, split="test") -> Corpus
    # meta carries premise, hypothesis, gold_label, label_candidate_ids.

def synthetic_corpus(tokenizer, kind, n, seed) -> Corpus
    # Wraps the existing build_degenerate_domains() controls.
```

### 2.4 Hard requirements

- **Determinism.** Same args → identical `manifest_sha256`, on any machine, across
  runs. Test this by constructing each corpus twice in one process and once in a
  subprocess.
- **No special tokens by default.** Matches E1–E5. `add_special_tokens=True` is
  permitted only where a task format needs it and must appear in `provenance`.
- **`frozen_e1_corpus` is a regression test, not just a provider.**
  `tests/test_corpus_e1_bridge.py` asserts its manifest equals the checked-in E1
  `sample_manifest.csv`.
- **Length filtering is explicit.** FLORES keeps rows where *every* language falls in
  [12, 160] tokens; the filter is applied at the parallel-join level in
  `paired_manifests`, not per-language here.
- Corpora are serialisable to and from JSONL so a Phase 2 run can reload the exact
  Phase 1 corpus: `Corpus.save(path)` / `Corpus.load(path)` with hash verification on
  load.

---

## 3. `common/fingerprint_runner.py` (WP2) — closes G2 and G8

### 3.1 Purpose

One function that, given *any* model handle and *any* `Corpus`, returns a complete
`FingerprintRecord`. This is the single seam between the frozen harnesses and all new
experiments.

### 3.2 Model handle

```python
@dataclass
class ModelHandle:
    arch: str                     # "gpt2"|"neo"|"qwen"|"opt"
    model: Any                    # nn.Module, already on device and in eval()
    tokenizer: Any
    engine: str                   # "manual" | "nnsight"
    nn_engine: Optional[Any]      # NNsightEngine when engine=="nnsight"
    model_name: str               # HF id OR local path
    model_revision: Optional[str]
    checkpoint_step: Optional[int]
    checkpoint_sha256: Optional[str]
    dtype: str
    device: str

def load_handle(arch, model_name_or_path, *, engine="nnsight", dtype="float32",
                revision=None, device=None, tokenizer_name=None) -> ModelHandle

def handle_from_module(arch, model, tokenizer, *, engine="nnsight", **prov) -> ModelHandle
    # G8: wraps an in-memory module (step-0 pre-update checkpoint, merged LoRA)
    # without a round-trip to disk. When engine=="nnsight" it wraps the module
    # with nnsight.LanguageModel(model, tokenizer=...) rather than
    # load_nnsight_model(), which is HF-id shaped.
```

`load_handle` dispatches to the frozen loaders (`load_neo`, `load_qwen`, GPT-2 path)
for `engine="manual"`, and to `load_nnsight_model(ARCH_SPECS[arch], ...)` for nnsight.
It must not reimplement dtype resolution — call `resolve_qwen_dtype` for Qwen.

### 3.3 The record

```python
@dataclass
class FingerprintRecord:
    # identity
    run_id: str; experiment_id: str; condition: str; seed: int
    model_name: str; model_revision: Optional[str]; checkpoint_step: Optional[int]
    arch: str; num_layers: int; num_heads: int; hidden_size: int; param_count: int
    corpus_id: str; manifest_sha256: str
    band: Tuple[int, int]; band_depth: Tuple[float, float]; band_version: str
    engine: str; dtype: str; device: str
    intervention_registry_version: str
    available_interventions: List[str]

    # topology
    baseline_sink: float
    per_layer_sink: List[float]                 # length num_layers
    per_head_sink: List[List[float]]            # [num_layers][num_heads]
    depth_profile_16: List[float]               # interpolated, always length 16
    frac_cells_above_0_2: float
    carrier_concentration: float                # Gini over layer-head cells, via _gini
    top_carrier_heads: List[Tuple[float, int, float]]   # (depth, head, strength) top-5

    # mechanism
    fingerprint: Dict[str, float]               # intervention key -> r_j
    raw_sink_by_intervention: Dict[str, float]  # intervention key -> A(M,D,j)

    # function  (None when not requested)
    delta_ce: Optional[Dict[str, float]]
    baseline_ce: Optional[float]

    # stability
    massive_coords: List[int]
    massive_coord_scope: str                    # "per_model" | "per_sentence"

    # bookkeeping
    n_items: int; n_failed: int
    failures: List[Dict[str, Any]]
    wallclock_s: float
    provenance: Dict[str, Any]
```

### 3.4 The entry point

```python
def compute_fingerprint(handle: ModelHandle,
                        corpus: Corpus,
                        *,
                        band=None,                      # from normalised_depth_band
                        interventions=None,             # None -> all available
                        with_delta_ce=False,
                        ce_corpus=None,                 # separate, usually larger
                        target_positions=(0, 1, 2, 3, 4),
                        batch_size=1,
                        progress=True,
                        cache_dir=None) -> FingerprintRecord
```

Behaviour:

1. Resolve `band` via `normalised_depth_band(num_layers)` when not given.
2. Resolve `massive_coords` using the arch's own frozen function
   (`identify_massive_coords` / `_neo` / per-sentence for Qwen). Respect
   `spec.massive_scope`.
3. For each `CorpusItem`, run the intervention battery. **Dispatch, do not reimplement:**
   - `engine="manual"` → the frozen `run_all_interventions` of the matching harness;
   - `engine="nnsight"` → `NNsightEngine.run_all`, or `run_arch_trace` per intervention
     when `with_delta_ce=True`.
4. Reduce each intervention's per-layer attention with the frozen
   `compute_bos_attention_metric(..., layer_start=band[0], layer_end=band[1])`.
5. `r_j = raw[j] / raw["int_a"]`. Guard `raw["int_a"] < 1e-8` → record a failure and set
   `r_j = nan` rather than dividing.
6. ΔCE, when requested, runs on `ce_corpus` (defaults to `corpus`) via
   `run_arch_trace(capture_token_ce=True)`, mean over non-padding next-token positions.
7. Failures are **recorded, never dropped** — `failures` carries `item_id`,
   `intervention`, exception class, message.

### 3.5 Cross-model helpers

```python
def available_interventions(handle) -> List[str]
    # Reads the harness INTERVENTIONS registry AND excludes structurally
    # inapplicable ones: "int_b" when q_proj.bias is None (Neo),
    # "int_e" when spec.wpe_path is None (Qwen/RoPE), etc.
    # This is design-delta D2: the mutually-defined set is computed, not hard-coded.

def mutual_interventions(*handles) -> List[str]
    # Intersection, preserving INTERVENTION_ORDER.
```

### 3.6 Caching

`compute_fingerprint` writes `<cache_dir>/<run_id>/step_<n>/fingerprint.json` and
returns the cached record when the cache key matches on
`(model_sha, corpus_sha, band, interventions, engine, dtype, registry_version)`.
Any mismatch recomputes. This is what makes `evaluate_transformation.py` resumable
(master plan §4.1).

### 3.7 Test

`tests/test_fingerprint_runner_bridge.py`: `compute_fingerprint(gpt2_handle,
frozen_e1_corpus(...))` reproduces the frozen `bos_attention_stats_overall.csv`
within `METRIC_ATOL`. If this fails, the seam is wrong and nothing downstream is
trustworthy.

---

## 4. `common/inheritance_metrics.py` (WP4) — closes G9

Pure analysis. **No model loading, no torch requirement beyond tensors in/out.**

### 4.1 Fingerprint distances

```python
INTERVENTION_CATEGORIES = {
    "strong_reduction": (0.00, 0.50),
    "partial_reduction": (0.50, 0.90),
    "negligible":       (0.90, 1.10),
    "increase":         (1.10, float("inf")),
}

def fingerprint_vector(record, keys) -> np.ndarray
def fingerprint_cosine(fa, fb, keys) -> float
def fingerprint_spearman(fa, fb, keys) -> float
def fingerprint_l1(fa, fb, keys, normalise=True) -> float
def category_of(r) -> str
def category_agreement(fa, fb, keys) -> float        # fraction of keys agreeing
def fingerprint_report(fa, fb, keys) -> dict         # all four, plus per-key deltas
```

`keys` is always passed explicitly and always comes from `mutual_interventions`.
No function may silently take a union or intersection of dict keys — that is how
D2 goes wrong in practice.

**NaN policy:** any key that is NaN in either fingerprint is dropped from all four
metrics, and the report records `n_keys_used` and `dropped_keys`. Downstream
aggregation refuses to compare two reports with different `n_keys_used`.

### 4.2 Topology

```python
def interp_depth_profile(per_layer_values, n_points=16) -> np.ndarray
    # d = i/(L-1); linear interpolation onto linspace(0,1,n_points).
    # L == 1 -> constant profile, flagged in a returned warning.

def topology_spearman(pa, pb) -> float
def topology_area_diff(pa, pb, normalise=True) -> float
def topology_wasserstein(pa, pb) -> float
    # 1-D Wasserstein between the profiles treated as MASSES over depth.
    # Both profiles are L1-normalised to sum 1 first; the function returns
    # (distance, normalisation_constants) so a degenerate all-zero profile
    # is reported rather than producing a spurious 0.
def topology_report(pa, pb) -> dict
```

### 4.3 Carrier heads

```python
def carrier_cells(per_head_sink, num_layers, num_heads) -> np.ndarray     # [L, H]
def carrier_concentration(cells) -> float                                 # reuse _gini
def top_carriers(cells, k=5) -> list[(depth, head, strength)]
def weighted_jaccard(cells_a, cells_b, *, depth_align=True, tol=0.10) -> float
    # Head-index comparison requires equal head counts -> assert.
    # Layers are matched by normalised depth within `tol`; when counts differ,
    # a many-to-one match is resolved by nearest depth and the mapping is returned.
def carrier_report(cells_a, cells_b, num_heads_a, num_heads_b) -> dict
```

Design §6.3 permits head-level comparison for the TinyStories pair because both have
16 heads. `weighted_jaccard` **must raise** on unequal head counts rather than
truncating — a silent truncation here would be an invisible confound.

### 4.4 Function

```python
def functional_vector(record, keys) -> np.ndarray       # delta_ce
def functional_cosine(ra, rb, keys) -> float
def functional_report(ra, rb, keys) -> dict
```

### 4.5 Drift (E6B)

```python
def fingerprint_drift(f_t, f_0, keys) -> float          # 1 - cosine
def topology_drift(p_t, p_0) -> float                   # wasserstein
def carrier_drift(c_t, c_0, ...) -> float               # 1 - weighted_jaccard
def drift_trajectory(records_by_step, base_record, keys) -> pd.DataFrame
def drift_onset(traj, threshold, column) -> Optional[int]
    # first step exceeding threshold; None if never
def clean_run_threshold(clean_trajs, column, k=2.0) -> float
    # mean + k*std across clean runs at matched steps -> the E6B early-warning bar
```

### 4.6 Statistics

```python
def paired_seed_contrast(values_a, values_b) -> dict
    # mean diff, per-seed diffs, exact permutation test over the 2^n sign
    # assignments. For n=3 there are 8 assignments -> minimum attainable
    # two-sided p is 0.25. The function MUST return
    # {"min_attainable_p": ..., "note": "descriptive, not inferential"}
    # so no downstream table can present p=0.25 as significance.

def hierarchical_bootstrap(df, group_cols, value_col, n_boot=10000, seed=0) -> dict
    # E7: resample semantic_ids, then languages, preserving all patch conditions
    # within a selected semantic_id.

def bh_correct(pvalues, alpha=0.05) -> (rejected, qvalues)
def bootstrap_ci(values, n_boot=10000, alpha=0.05, seed=0) -> (lo, hi)
```

The permutation-test note is not decoration. With three seeds a paired permutation
test cannot produce p < 0.25; the design's §15.1 calls it "a descriptive exact test"
and the code must enforce that framing so a reviewer never sees it presented otherwise.

### 4.7 Composite scores — deliberately absent

There is **no** `inheritance_score()` function. Design §4 forbids "a single aggregate
inheritance score presented without its components." Adding one, even privately, invites
its appearance in a table. Topological / mechanistic / functional / semantic are
reported as four separate columns, always.

---

## 5. `common/paired_manifests.py` (WP8)

### 5.1 Purpose

Build and validate the parallel joins that E7 depends on, and the fixed control
assignments for patching.

```python
@dataclass(frozen=True)
class ParallelManifest:
    manifest_id: str
    dataset: str                  # "flores" | "xnli"
    split: str
    languages: Tuple[str, ...]
    semantic_ids: Tuple[str, ...]
    rows: Dict[str, Dict[str, dict]]   # semantic_id -> lang -> row
    partitions: Dict[str, Tuple[str, ...]]   # "train"/"dev"/"test" -> semantic_ids
    seed: int
    sha256: str
    provenance: dict
```

```python
def build_flores_manifest(tokenizer, langs, *, split="devtest",
                          min_tokens=12, max_tokens=160, n=300, seed=42) -> ParallelManifest
    # Join by source index. Keep a row only if EVERY language passes the length
    # filter. If fewer than 300 rows survive, keep all and set
    # provenance["valid"]=False when fewer than 200 (design §10.3 step 5).

def build_length_matched_subset(manifest, reference_lang="eng_Latn",
                                max_rel_diff=0.20) -> ParallelManifest
    # Optimal bipartite matching on token count (scipy.optimize.linear_sum_assignment)
    # between English and each target language. Records the realised max relative
    # difference per language.

def build_xnli_manifest(tokenizer, langs, *, split="test", n=600, seed=42,
                        balanced=True) -> ParallelManifest

def assign_patch_controls(manifest, seed=42) -> pd.DataFrame
    # For every (semantic_id, target_lang): parallel_en, same_label_en,
    # different_label_en, random_en. Deterministic from seed. Constraints:
    #   - same_label_en   != the target's own semantic_id
    #   - different_label_en has a different gold label
    #   - random_en drawn independently of label
    #   - no semantic_id used as its own control
    # Returns one row per (semantic_id, target_lang, control_condition).

def grouped_partition(manifest, fractions=(0.0, 1/3, 1.0), names=("dev","test"),
                      seed=42) -> dict
    # Partitions by SEMANTIC ID so no sentence appears in two partitions in any
    # language. E7 uses 200 dev / 400 test (design §10.9) and 100/200 for
    # Procrustes (design §10.7); both are expressed through this one function.

def verify_manifest(manifest) -> dict
    # Asserts: every semantic_id present in every language; no duplicate ids;
    # token counts within declared bounds; partitions disjoint and exhaustive;
    # sha256 matches recomputation. Raises on failure.
```

### 5.2 The alignment trap

FLORES `devtest` rows are aligned by position within each language config. XNLI is
aligned by `promptID`/`pairID`, not by row index. `build_xnli_manifest` **must** join
on the id field and assert the premise/hypothesis pair is a translation set, not
assume positional alignment. `tests/test_parallel_manifest_alignment.py` checks this
by asserting gold labels agree across all languages for every semantic id — a
positional-alignment bug shows up immediately as label disagreement.

---

## 6. `common/cross_example_patching.py` (WP7) — closes G4

The single most delicate module in the project. Everything E7 claims causally rests on
it being correct.

### 6.1 Two-trace design

```python
@dataclass(frozen=True)
class PatchSite:
    object: str        # "K0" | "V0" | "R0" | "Kmid" | "Vmid" | "Rmid"
    layer: int
    position: int      # 0 for *0 sites; resolved per-example for *mid

@dataclass(frozen=True)
class PatchSpec:
    site: PatchSite
    norm_condition: str    # "direct"|"rescaled"|"random"|"identity"
    source_item_id: Optional[str]
    seed: int

@dataclass
class SourceCache:
    model_fingerprint: str      # arch + model_name + revision + dtype + weight sha
    item_id: str
    tensors: Dict[str, torch.Tensor]   # "K0@L5" -> [n_kv_heads, head_dim], fp32 CPU
    seq_len: int
    provenance: dict
```

```python
def capture_sources(handle, corpus, sites, *, batch_size=1,
                    to_cpu=True, dtype=torch.float32) -> Dict[str, SourceCache]

def run_patched(handle, target_item, patch_specs, source_caches, *,
                score_candidates=None,     # list[list[int]] label token ids
                capture_sink=True, band=None) -> dict
```

Returns baseline and patched: candidate log-probs, gold margin, prediction, position-0
attention, attention to positions 1–4, per-carrier-head sink, JSD between output
distributions, and `status`/`warning`.

### 6.2 Non-negotiable invariants

Each is a test in `tests/test_cross_example_patching.py`.

1. **Identity patch is a no-op.** `norm_condition="identity"` (write the target's own
   captured tensor back) reproduces baseline logits to `1e-6` in fp32. If this fails,
   the write path is wrong and every other result is void.
2. **Self-source is a no-op.** Patching with `source_item_id == target_item_id`
   reproduces baseline logits to `1e-6`.
3. **Locality.** A `K0@L5` patch changes activations at layer 5 position 0 and nothing
   at layers < 5. Assert by capturing block outputs at layers 0…5 before and after.
4. **Pre-repeat.** For Qwen, `V0` is patched before `_repeat_kv`. Test on a random
   Qwen2 config with `num_key_value_heads=2, num_attention_heads=8`: patching one KV
   head must alter exactly four query heads' outputs.
5. **Norm matching.** `norm_condition="random"` produces a Gaussian vector rescaled to
   `||target||` within `1e-6`. `"rescaled"` rescales the source to `||target||` within
   `1e-6`.
6. **Length independence.** Source and target may have different sequence lengths.
   Position-0 sites are unaffected; `*mid` sites resolve `position = seq_len // 2` **per
   example** and the resolved positions are recorded in the output row.
7. **No batch leakage.** With `batch_size > 1`, patching item *i* leaves items *j ≠ i*
   bit-identical to their unpatched forward. Default to `batch_size=1` for patching
   and treat >1 as an optimisation gated on this test.
8. **Model-fingerprint match.** `run_patched` raises if
   `source_cache.model_fingerprint != handle` fingerprint. Cross-model patching is not
   a supported operation and must fail loudly, not silently produce garbage.
9. **Shape and dtype.** Assert the tensor shape against
   `(n_kv_heads, head_dim)` for K/V and `(hidden,)` for R. Cast to the model's dtype at
   write time; never let a fp32 cache silently upcast a bf16 forward.

### 6.3 Scoring

Label scoring is **length-normalised sequence log-probability** over the candidate's
full token sequence (design §10.4), computed with teacher forcing over the candidate
appended to the prompt. Implement once, in this module, and reuse for both the baseline
and patched forward so a scoring bug cannot differentially affect them.

`CorrectLabelMargin = logP(gold) - max_{non-gold} logP(label)`, on the same
length-normalised scale. Record both normalised and unnormalised.

### 6.4 The RoPE decision (design-delta D4)

Qwen applies RoPE inside `Qwen2Attention.forward`, after `k_proj`. There is no envoy on
the post-RoPE tensor, so "patch K0" is ambiguous. Implement **both** and treat them as
distinct `object` values:

- `"K0_prerope"` — capture and patch `k_proj.output[..., 0, :]`. RoPE is then applied at
  the target's position 0. Since position 0's rotation is identity, pre- and post-RoPE
  coincide at position 0 — **verify this numerically rather than assuming it**, and if
  the model uses a non-zero position offset, the identity does not hold.
- `"K0_postrope"` — capture `k_proj.output`, apply the harness's existing
  `build_rope_cos_sin` + `_apply_rope` outside the trace to obtain the post-RoPE key,
  and patch by writing back a pre-RoPE vector that yields the desired post-RoPE value
  (inverse rotation). At position 0 this reduces to the previous case; at `Kmid` it does
  not.

For `Kmid`/`Vmid` the distinction is substantive and both must be reported. Record the
chosen variant in the `patch_object` column so no aggregation ever mixes them.

### 6.5 Failure handling

Non-finite logits, OOM, tokenisation failure → write the row with `status="failed"` and
a `warning`, never drop it. Design §12 is explicit: *"save failures rather than silently
dropping examples."* The aggregation step reports failure counts per condition and
refuses to compute a contrast whose failure rate exceeds 2% without an explicit override.
