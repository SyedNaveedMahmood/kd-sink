# 01 — Refactor Spec: Changes to Existing Code

**Rule:** additive only. If a change would alter any existing script's output for any
existing CLI invocation, it is wrong. Every edit below is either (a) a new function in
an existing module, (b) a new optional keyword with a default that reproduces current
behaviour, or (c) a new dataclass field with a default of `None`.

---

## 1. WP0 — Freeze the baseline first

Before any edit.

```bash
git switch -c sink-inheritance
git tag frozen-e1-e5
```

Then write `BASELINE_HASHES.json` at the repo root:

```json
{
  "git_tag": "frozen-e1-e5",
  "git_sha": "<full sha>",
  "frozen_files": {
    "common/intervention_analysis.py": "<sha256>",
    "common/intervention_analysis_legacy.py": "<sha256>",
    "common/residual_sink_analysis.py": "<sha256>",
    "common/residual_sink_analysis_legacy.py": "<sha256>",
    "common/experiments_single_input.py": "<sha256>",
    "cross_scale_and_architecture/neo/intervention_analysis_neo.py": "<sha256>",
    "cross_scale_and_architecture/opt/intervention_analysis_opt.py": "<sha256>",
    "cross_scale_and_architecture/qwen/intervention_analysis_qwen.py": "<sha256>",
    "cross_scale_and_architecture/run_table1_multiseed.py": "<sha256>",
    "emergence_dynamics/emergence_dynamics_analysis.py": "<sha256>",
    "evaluation_robustness/evaluation_robustness_analysis.py": "<sha256>",
    "reproduce_paper/experiments_statistical.py": "<sha256>"
  },
  "reference_results": {
    "gpt2_small_table1": { "int_a": 0.0, "int_b": 0.0, "...": 0.0 },
    "tolerances": { "metric_atol": 1e-5, "metric_rtol": 1e-4 }
  },
  "environment": {
    "torch": "", "transformers": "", "nnsight": "", "python": ""
  }
}
```

Populate `reference_results.gpt2_small_table1` by actually running the frozen GPT-2
dataset analysis and copying `bos_attention_stats_overall.csv`. Do not transcribe from
the paper.

Add `tests/test_frozen_files.py`: recompute each sha256 and assert equality. This test
runs in CI for the whole project lifetime and is the mechanical enforcement of the
frozen-baseline rule.

**The twelve files above are read-only from this point.** Everything else in this
document edits only `common/nnsight_engine.py` and `common/datasets_loader.py`.

---

## 2. Import-path note

The harnesses use a dual-import idiom (`from . import x` / `import x` fallback) because
scripts put `common/` directly on `sys.path`. Every new module must use the same idiom.
New packages (`transformation_inheritance/`, `crosslingual_semantics/`) must add
`common/` to `sys.path` in exactly the way the existing cross-arch scripts do — copy the
header from `cross_scale_and_architecture/neo/intervention_analysis_neo.py` verbatim.
Do not introduce `setup.py`, `pyproject.toml`, or a package rename. That would be a
non-additive change to how every existing script resolves imports.

---

## 3. G6 — the layer-band problem

### 3.1 The bug in context

```python
compute_band(4,  "scaled") -> (3, 4)    # TinyStories-33M teacher: ONE layer
compute_band(8,  "scaled") -> (3, 7)    # 8M student: four layers
compute_band(6,  "scaled") -> (3, 5)    # distilgpt2: two layers
compute_band(12, "scaled") -> (3, 11)   # gpt2: eight layers
compute_band(24, "scaled") -> (3, 23)   # Qwen 0.5B: twenty layers
```

For E1–E5 (12/24/32-layer models) this is fine. For a 4-layer teacher it collapses to a
single layer, and `layer_mode="fixed"` gives `(3,4)` — the same single layer. Any
teacher/student sink comparison computed this way compares layer 3 of 4 (depth 1.0)
against layers 3–6 of 8 (depth 0.43–0.86). The number would be produced without error
and would be meaningless.

### 3.2 Resolution — new module, legacy untouched

Create `common/depth_band.py`. **Do not modify `compute_band`.** E1–E5 reproduction
continues to call the legacy function; E6/E7 call the new one.

```python
# common/depth_band.py

DEPTH_BAND_DEFAULT = (0.25, 0.90)   # fractional [start, end) over layer index
DEPTH_BAND_VERSION = "depth_band_v1"

def normalised_depth_band(num_layers, frac=DEPTH_BAND_DEFAULT, min_layers=1):
    """0-indexed [start, end) chosen by fractional depth, never empty.

    start = floor(frac[0] * num_layers)
    end   = max(start + min_layers, ceil(frac[1] * num_layers))
    end   = min(end, num_layers)
    start = min(start, end - min_layers)

    Returns (start, end, meta) where meta records num_layers, frac, the
    realised depth interval, and DEPTH_BAND_VERSION.
    """

def layer_depths(num_layers):
    """Array of d = i / (num_layers - 1) for i in range(num_layers).
    num_layers == 1 -> array([0.0])."""

def band_agreement_report(num_layers_a, num_layers_b, frac=DEPTH_BAND_DEFAULT):
    """Realised depth intervals for two models and their absolute mismatch.
    Used as an assertion in every cross-model comparison."""
```

Realised bands under the default:

| model | L | band | depth interval |
|---|---|---|---|
| TinyStories-33M teacher | 4 | (1, 4) | 0.33 – 1.00 |
| 8M student | 8 | (2, 8) | 0.29 – 1.00 |
| distilgpt2 | 6 | (1, 6) | 0.20 – 1.00 |
| gpt2 | 12 | (3, 11) | 0.27 – 0.91 |
| Qwen2.5-0.5B | 24 | (6, 22) | 0.26 – 0.91 |

Note that `gpt2` at `(3, 11)` **coincides exactly with the legacy band**. That is the
reason `(0.25, 0.90)` was chosen and it must be asserted in
`tests/test_depth_band.py`: for `num_layers=12`, `normalised_depth_band` must equal
`compute_band(12, "scaled")`. This keeps E6B's distilgpt2/gpt2 screening on the same
footing as the frozen E1 results.

### 3.3 Required call-site discipline

Every E6/E7 metric that averages over layers takes an explicit `band` argument obtained
from `normalised_depth_band`, and every `run_config.json` records
`layer_band`, `layer_band_depth_interval`, and `layer_band_version`. Cross-model
comparisons call `band_agreement_report` and **raise** if the depth-interval mismatch
exceeds 0.10 in either endpoint, unless `allow_band_mismatch=True` is passed explicitly
and the reason is recorded.

Additionally, all topology comparisons (§`02` §4.4) interpolate the **full** per-layer
profile to 16 fixed depth points and do not use the band at all. The band is only for
the scalar sink-strength metric.

---

## 4. `common/nnsight_engine.py` — additive edit 1: `ArchSpec` capture surface (G5)

Add four fields, all defaulting to `None`, so every existing `ARCH_SPECS` entry is
unchanged:

```python
@dataclass(frozen=True)
class ArchSpec:
    ...
    v_path: Optional[str] = None            # rel. to attn; "v_proj" when separate
    o_path: Optional[str] = None            # rel. to attn; "o_proj" / "c_proj"
    rope_applied_inside_attn: bool = False  # True for qwen
    kv_grouped: bool = False                # True when num_kv_heads < num_heads
```

Populate:

```python
"gpt2":  ... o_path="c_proj",
"opt":   ... v_path="v_proj",  o_path="out_proj",
"neo":   ... v_path="attention.v_proj", o_path="attention.out_proj",
"qwen":  ... v_path="v_proj",  o_path="o_proj",
              rope_applied_inside_attn=True, kv_grouped=True,
```

`kv_grouped` is a static declaration; the actual repeat factor is read at runtime from
`config.num_attention_heads // config.num_key_value_heads`. The Qwen harness already
implements `_repeat_kv`, `build_rope_cos_sin`, `_apply_rope`, `_rotate_half` — reuse
them, do not rewrite.

**Guard:** `q_proj.output` / `k_proj.output` / `v_proj.output` for Qwen2 are shaped
`[B, S, num_heads*head_dim]` and `[B, S, num_kv_heads*head_dim]` respectively.
Extraction and patching must reshape to `[B, S, n_kv_heads, head_dim]` and operate
**before** `_repeat_kv`. Assert the trailing dimension against
`num_key_value_heads * head_dim` on every call; a silent shape mismatch here would
produce plausible-looking but wrong patch results.

---

## 5. `common/nnsight_engine.py` — additive edit 2: generic trace with CE (G3)

### 5.1 The gap

`run_gpt2_trace(plan: GPT2TracePlan, ...)` supports `capture_logits` /
`capture_token_ce` but is GPT-2 specific — it resolves `c_attn`, uses `GPT2TracePlan`,
and is called only from `evaluation_robustness`. E6A needs ΔCE for interventions
c,d,f,g,h,i,j on a **GPT-Neo** student. There is no path from `EditPlan` to logits.

### 5.2 New method — do not modify `run_intervention` or `run_gpt2_trace`

```python
class NNsightEngine:
    def run_arch_trace(self, plan: EditPlan, inputs, *,
                       massive_coords=None,
                       random_columns=None,
                       swap_dirs=None,
                       capture_attention="targets",   # "full"|"targets"|"none"
                       target_positions=(0,),
                       band=None,
                       capture_logits=False,
                       capture_token_ce=False,
                       capture_position0=None,        # None | dict spec, see 02 §6
                       ) -> dict:
        """Arch-generic counterpart of run_gpt2_trace.

        Applies exactly the same edit sequence as run_intervention (which it must
        share, not duplicate: factor the edit body into a private
        _apply_edits(plan, ...) generator used by both), and additionally saves
        LM-head logits and/or per-token CE.

        Returns a dict with keys among:
          attention        list[Tensor] or None
          target_attention Tensor [L, heads, len(target_positions)] or None
          logits           Tensor [B, S, V] or None
          token_ce         Tensor [B, S-1]  or None
          position0        dict of derived tensors (see 02 §6) or None
          provenance       dict
        """
```

Implementation constraints:

1. **Factor, do not fork.** Extract the edit body of `run_intervention`
   (`nnsight_engine.py:534–588`) into `_apply_edits(self, plan, spec, ...)` and call it
   from both `run_intervention` and `run_arch_trace`. `run_intervention` must then be
   verified byte-equivalent in behaviour by `tests/test_run_intervention_parity.py`,
   which runs the frozen GPT-2 and Neo fingerprints before and after the refactor and
   asserts equality at `METRIC_ATOL`.
2. **CE definition.** Reuse `evaluation_robustness.token_cross_entropy` exactly. Do not
   write a second CE. Import it; if the import direction is awkward, move nothing —
   duplicate the four-line function into `nnsight_engine` **only if** a test asserts the
   two implementations agree to `1e-7` on random tensors.
3. **`capture_attention="targets"`** must return `[num_layers, num_heads, n_targets]` of
   second-half-query mean attention, computed inside the trace so full `[H,S,S]` maps
   never leave it. This is what makes 300-block × 9-intervention × 72-checkpoint
   evaluation fit in memory.
4. **Access order.** NNsight raises `MissedProviderError` if envoys are accessed out of
   execution order. `lm_head` / `logits` must be saved after the final block. Keep the
   existing ordering comment.
5. **No `torch.no_grad()` wrapper.** The existing comment at `run_intervention`
   explains why; the same applies.

### 5.3 OPT caveat

`needs_output_attentions=True` for OPT. `run_arch_trace` must thread the same payload
handling as `_payload`. OPT is not required for E6/E7 but the method must not silently
break it — `tests/nnsight_e6_smoke.py` includes a random OPT case.

---

## 6. `common/datasets_loader.py` — additive edits

Existing functions are frozen in behaviour. Add:

```python
def load_flores_parallel(langs, split="devtest", min_tokens=12, max_tokens=160,
                         n=300, seed=42, tokenizer=None):
    """Join facebook/flores rows by source index across `langs`.
    Returns (rows, manifest_rows). See 04 §1 for the row schema."""

def load_xnli_aligned(langs, split="test", n=600, seed=42, tokenizer=None,
                      max_tokens=None, balanced=True):
    """Aligned XNLI examples across `langs`, balanced over the three labels."""

def load_tinystories_blocks(tokenizer, split, block_size=128, n_blocks=None,
                            seed=0, eos_between=True):
    """Tokenise without special tokens, join with one EOS, pack into exact
    block_size blocks. Returns (blocks, block_manifest) where block_manifest
    records the source story indices contributing to each block."""

def load_sst2_prompted(tokenizer, split, max_input_tokens=64,
                       template="{sentence}\nSentiment:",
                       labels=(" positive", " negative"), seed=0):
    """SST-2 in the E6B prompt format. Returns rows with input_ids,
    label_token_ids, label_span (start,end), and the raw sentence.
    Asserts both label strings tokenise consistently and records their
    token counts; multi-token labels are supported and scored in full."""
```

`load_optional_flores` already exists — **do not replace it**. `load_flores_parallel`
is a distinct function with a distinct contract (parallel join, not sampling).

The existing `DEFAULT_SEED`, `DEFAULT_SAMPLE_SIZE`, `DEFAULT_CUT_LENGTH` and
`_normalize_text` / `_truncate` helpers are reused by all four new loaders so that
tokenisation and normalisation are identical to E1–E5.

---

## 7. What is *not* changed, and why

| Tempting change | Why it is rejected |
|---|---|
| Refactor `dataset_analysis` to accept a corpus | It is in three frozen files. `fingerprint_runner.py` (WP2) provides the injectable path instead, calling the frozen `run_all_interventions` per sentence. |
| Make `INTERVENTIONS` a shared registry across archs | Each harness's registry differs in closure signature (`num_heads`, `geom`, `pos_enc`). Unifying them is a behaviour-affecting refactor of frozen files. `available_interventions()` reads them instead. |
| Add `pyproject.toml` / make it a package | Changes import resolution for every existing script. |
| Replace `compute_band` | E1–E5 reproduction depends on it. New module instead. |
| Add training deps to `requirements.txt` in place | Append only, never reorder or bump. `peft`, `pyyaml`, `scipy` are new; `datasets`, `transformers`, `torch`, `nnsight` already present — do not change their pins without rerunning the parity gate. |

---

## 8. Acceptance for WP0 + WP3

- `tests/test_frozen_files.py` passes.
- `tests/test_depth_band.py` passes, including the `num_layers=12` legacy-coincidence
  assertion.
- `tests/test_run_intervention_parity.py`: GPT-2-small and gpt-neo-125m fingerprints
  computed through `run_intervention` after the `_apply_edits` factoring match the
  pre-refactor values within `METRIC_ATOL=1e-5` / `METRIC_RTOL=1e-4`.
- Existing `tests/test_crfm_attention_config_parity.py`, `tests/test_e4_parity_tolerances.py`,
  `tests/nnsight_e3_smoke.py`, `_e4_`, `_e5_` all still pass unchanged.
- `python -m tests.nnsight_e6_smoke` passes on random gpt2/neo/qwen/opt models,
  exercising `run_arch_trace` with `capture_token_ce=True`.
