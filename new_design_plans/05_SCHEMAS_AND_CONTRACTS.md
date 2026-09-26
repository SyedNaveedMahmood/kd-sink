# 05 — Schemas and Contracts

Every schema is normative. A column may be added; none may be renamed or removed once
a Phase 2 run has started, because aggregation joins on these names.

---

## 1. `run_config.json` — every run directory

Design §13.1 plus the fields the gap analysis showed are required.

```json
{
  "experiment_id": "e6a",
  "condition_id": "D2",
  "run_id": "e6a_D2_seed0",
  "model_name": "results/e6a/D2/seed0/checkpoints/step_10000",
  "model_revision": null,
  "architecture": "GPTNeoForCausalLM",
  "arch_key": "neo",
  "parameter_count": 8000000,
  "num_layers": 8,
  "num_heads": 16,
  "hidden_size": 256,

  "training_seed": 0,
  "data_seed": 0,
  "evaluation_seed": 42,

  "tokenizer_name": "roneneldan/TinyStories-33M",
  "tokenizer_revision": null,
  "vocab_assertions": {"vocab_size": true, "bos": true, "eos": true, "heads": true},

  "dtype": "bfloat16",
  "device": "cuda:0",
  "engine": "nnsight",
  "attn_implementation": "eager",

  "transformers_version": "",
  "torch_version": "",
  "nnsight_version": "",
  "peft_version": null,
  "numpy_version": "",
  "scipy_version": "",

  "dataset_name": "roneneldan/TinyStories",
  "dataset_revision": null,
  "corpus_id": "tinystories_val_sink_300",
  "manifest_sha256": "",
  "block_manifest_sha256": "",

  "checkpoint_step": 10000,
  "checkpoint_sha256": "",

  "layer_band": [2, 8],
  "layer_band_depth_interval": [0.2857, 1.0],
  "layer_band_version": "depth_band_v1",

  "intervention_registry_version": "neo_v1",
  "available_interventions": ["int_a","int_c","int_d","int_e","int_f","int_g","int_h","int_i","int_j"],
  "patch_registry_version": null,

  "git_sha": "",
  "frozen_baseline_tag": "frozen-e1-e5",
  "created_utc": ""
}
```

### 1.1 Registry versions

Bump `intervention_registry_version` whenever the set or semantics of interventions
changes for that arch. Aggregation **refuses to compare records with different registry
versions** — this is the mechanism that prevents a mid-project change to intervention
(i) from silently contaminating a cross-checkpoint trajectory.

Initial values: `gpt2_v1`, `neo_v1`, `qwen_v1`, `opt_v1`, `patch_v1`.

---

## 2. `checkpoint_metrics.csv` — E6

Design §13.2 plus the columns aggregation actually needs.

| column | type | notes |
|---|---|---|
| `experiment_id` | str | e6a / e6b |
| `run_id` | str | |
| `condition` | str | D0/D1/D2/F0–F4 |
| `seed` | int | |
| `checkpoint_step` | int | |
| `corpus_id` | str | in-domain / cross-domain / sst2-val |
| `manifest_sha256` | str | |
| `validation_ce` | float | |
| `validation_ce_n_blocks` | int | 300 or 2000 (design-delta D3) |
| `task_accuracy` | float | E6B; null for E6A |
| `task_nll` | float | E6B |
| `ece_10bin` | float | E6B |
| `baseline_sink` | float | |
| `frac_cells_above_0_2` | float | |
| `carrier_concentration` | float | |
| `route_a_share` | float | where defined |
| `relocation_ratio_swap_epe` | float | |
| `fingerprint_json` | str | JSON dict of intervention → r_j |
| `delta_ce_json` | str | JSON dict of intervention → ΔCE |
| `depth_profile_16_json` | str | JSON list, length 16 |
| `top_carriers_json` | str | JSON list of [depth, head, strength] |
| `massive_coords_json` | str | JSON list |
| `n_keys_used` | int | mutual interventions actually compared |
| `fingerprint_cosine_to_teacher` | float | |
| `fingerprint_spearman_to_teacher` | float | |
| `fingerprint_l1_to_teacher` | float | |
| `category_agreement_to_teacher` | float | |
| `topology_wasserstein_to_teacher` | float | |
| `topology_spearman_to_teacher` | float | |
| `functional_cosine_to_teacher` | float | |
| `carrier_jaccard_to_teacher` | float | |
| `fingerprint_drift_from_base` | float | E6B |
| `topology_drift_from_base` | float | E6B |
| `carrier_drift_from_base` | float | E6B |
| `n_items` | int | |
| `n_failed` | int | |
| `wallclock_s` | float | |
| `engine` | str | |
| `dtype` | str | |
| `layer_band` | str | `"[2,8)"` |
| `intervention_registry_version` | str | |
| `git_sha` | str | |

Nested structures are stored as JSON strings rather than exploded columns so the CSV
stays joinable and the fingerprint's key set is always explicit. Aggregation parses
them; it never infers a key order from column position.

---

## 3. `patching_per_example.csv` — E7

Design §13.3 plus provenance.

| column | type |
|---|---|
| `model`, `model_revision`, `dtype`, `engine` | str |
| `target_language` | str |
| `semantic_id` | str |
| `partition` | str (`dev`/`test`) |
| `gold_label` | str |
| `source_condition` | str (`parallel_en`/`same_label_en`/`different_label_en`/`random_en`) |
| `source_semantic_id`, `source_label` | str |
| `patch_object` | str (`K0_prerope`/`K0_postrope`/`V0`/`R0`/`Kmid`/`Vmid`/`Rmid`) |
| `patch_layer` | int |
| `patch_position`, `source_patch_position` | int |
| `norm_condition` | str |
| `baseline_gold_logprob`, `patched_gold_logprob` | float |
| `baseline_margin`, `patched_margin` | float |
| `baseline_margin_unnorm`, `patched_margin_unnorm` | float |
| `baseline_prediction`, `patched_prediction` | str |
| `baseline_sink`, `patched_sink` | float |
| `baseline_attn_pos1_4`, `patched_attn_pos1_4` | float |
| `jsd_output` | float |
| `carrier_sink_delta` | float |
| `source_seq_len`, `target_seq_len` | int |
| `source_norm`, `target_norm`, `patched_norm` | float |
| `manifest_sha256`, `patch_registry_version`, `git_sha` | str |
| `status` | str (`ok`/`failed`/`nonfinite`/`oom`) |
| `warning` | str |

`status != "ok"` rows are written, never dropped.

---

## 4. Run-directory layout

```
transformation_inheritance/results/
  e6a/<condition>/seed<k>/
    run_config.json
    block_manifest.parquet
    train_log.jsonl
    eval_log.jsonl
    runtime_estimate.json
    pilot_gate.json                  # seed 0 only
    checkpoints/step_<n>/
      config.json  model.safetensors  optimizer.pt  scheduler.pt  rng_state.pt
      checkpoint_sha256.txt
    fingerprints/step_<n>/<corpus_id>.json
    checkpoint_metrics.csv
  e6b/<condition>/seed<k>/
    ... same, plus corruption_manifest.csv, merge_parity.json,
        checkpoints/step_<n>/{merged,adapter}/
  screening/
    screening_fingerprints.csv
  aggregate/
    e6a_contrasts.csv  e6a_matched_loss.csv
    e6b_drift.csv  e6b_early_warning.csv  e6b_factorial.csv
    table2_inheritance_components.csv  table3_clean_vs_corrupt.csv
    go_no_go.json  figures/

crosslingual_semantics/results/
  manifests/
    flores_devtest.json  flores_devtest_lenmatched.json
    xnli_test.json  patch_controls.csv  partitions.json
  reps/<model_tag>/<lang>/<object>.npy  index.json  run_config.json
  fingerprints/<model_tag>/<lang>.json
  retrieval/retrieval_by_pair.csv  retrieval_by_object.csv  language_probe.csv
  patching/<model_tag>/patching_per_example.csv
            layer_selection.json  runtime_estimate.json
  aggregate/
    patching_contrasts.csv  table4_patch_effects.csv
    interpretation_matrix.json  e7_go_no_go.json  failures.csv  figures/
```

---

## 5. `runtime_estimate.json`

Required by design §17; no unverified runtime may appear in the paper.

```json
{
  "unit": "optimizer_step",
  "measured_units": 100,
  "elapsed_s": 0.0,
  "units_per_second": 0.0,
  "projected_total_units": 10000,
  "projected_total_s": 0.0,
  "projected_finish_utc": "",
  "device": "NVIDIA GeForce RTX 4080 SUPER",
  "peak_vram_bytes": 0,
  "dtype": "bfloat16",
  "measured_utc": ""
}
```

Written after 100 optimiser steps (training) or 50 patch examples (E7). Emit it, then
check it before committing to Phase 2.

---

## 6. `go_no_go.json`

```json
{
  "experiment": "e6a",
  "criteria": [
    {"id": "e6a_1", "text": "D2 improves teacher fingerprint similarity over D0 by >= 0.15",
     "observed": 0.0, "threshold": 0.15, "all_seeds": [0.0,0.0,0.0], "met": false}
  ],
  "n_met": 0,
  "decision": "continue|downgrade|stop",
  "evaluated_utc": "",
  "git_sha": ""
}
```

Criteria are transcribed from design §18 as literal code before any results exist.
That is the pre-registration; changing a threshold after seeing results requires a
recorded amendment with a timestamp and a reason.

---

## 7. Cross-cutting contracts

1. **Hash before compare.** Any two records compared must agree on `manifest_sha256`,
   `intervention_registry_version`, and `n_keys_used`, or the comparison raises.
2. **Failures are data.** No pipeline drops an example silently. Every stage writes a
   failure count; downstream stages refuse contrasts above a 2% failure rate without an
   explicit override flag.
3. **Provenance is mandatory.** Every CSV carries `git_sha` and the manifest hash. A row
   without provenance cannot enter the paper.
4. **Determinism is testable.** Every manifest, corpus, and control assignment must
   produce an identical hash on reconstruction, verified in a subprocess (not just
   within one interpreter, which can hide global-RNG dependence).
5. **Nothing is renamed mid-project.** If a column must change meaning, add a new one
   and bump the relevant registry version.
