# `_exploratory/` — verification runs, deliberately outside `results/`

Both aggregators discover their inputs with `rglob("checkpoint_metrics.csv")` from the
results root and do **not** de-duplicate. Anything left under
`transformation_inheritance/results/` is therefore pooled into the paper's tables whether
or not it was a production run. These two runs were instrument verification, not
experiments, so they live here instead. Moved 2026-07-30.

Nothing under this directory is read by `aggregate_transformation.py`.

## `e6a_P8M_sink24/` — the `08` §4 public-reference path, verified on real weights

Real `roneneldan/TinyStories-8M` fingerprinted against the real
`roneneldan/TinyStories-33M` teacher on CUDA. Two subdirectories, and the reason they had
to be quarantined rather than merely ignored:

| dir | corpus | `validation_ce` | `fingerprint_cosine_to_teacher` | `functional_cosine_to_teacher` |
|---|---|---|---|---|
| `reference/` | `tinystories_validation_sink_24` | — | 0.211349 | — |
| `reference/` | `e1_100x40` | — | 0.488000 | — |
| `reference_dce/` | `tinystories_validation_sink_24` | 1.89786 | 0.211349 | 0.968349 |

Both carry `condition=P8M, seed=-1, checkpoint_step=-1` on the **same** corpus, so a
production aggregation would have loaded two rows for one seedless condition — one of them
with an empty `validation_ce`. Design §8.7 contrast 4's matched-loss selection reads exactly
that column, so the duplicate is not harmless.

They are also the wrong corpus for the pre-registration: `--sink-blocks 24` produced
`tinystories_validation_sink_24`, while `e6_preregistration.yaml` names
`tinystories_validation_sink_300`. **The real reference run must not pass `--sink-blocks`,
and must pass `--with-delta-ce`** — without it `validation_ce` is honestly empty and
`e6a_c4a`/`e6a_c4b` report `no_data` instead of selecting a checkpoint.

## `e6a_D0_seed0_partial_50steps/` — E6A training, started and stopped by choice

The first real E6A training run in the project, and the artefact that proved the corpus
memory fix works. Stopped at **step 51 of 2,000** once the step rate had been measured; the
full 5-run pilot was ~20 h of wall-clock and was deferred rather than run here.

| what | value |
|---|---|
| condition / seed | D0 (CE only) / 0 |
| steps logged | 51 (loss 10.878 → 9.852) |
| checkpoints | `step_0` only (the pre-first-update save) |
| corpus | 3.57M blocks, `manifest_sha256` `5253852405e6f0d2…` |
| peak RSS during the corpus build | ~3.1 GB (was ~29 GB before the fix) |
| measured rate | **6.67 s/step** on the RTX 2060 |

Quarantined rather than left in `results/` for a specific reason: `run_config.json` records
`max_steps: 2000`, and resuming it into a 10,000-step production run would splice two
different cosine schedules together. `assert_resumable_schedule` now refuses that, but the
cleanest state for a real pilot is an empty run directory.

Nothing here is a scientific result: 51 steps of a 2,000-step pilot measures the
instrument's speed, not the experiment.

## `e6b_F1_seed0_16step/` — the WP6 E6B pipeline, verified on real weights

Real `distilbert/distilgpt2` LoRA fine-tuned on real SST-2 for **16 steps** (a production
F1 run is 3 epochs over ~67k rows), evaluated against the untrained base. Six `ok` rows
across three corpora at steps 0 and 16:

| step | `validation_ce` | `task_accuracy` | `fingerprint_drift_from_base` |
|---|---|---|---|
| 0 | 0.654134 | 0.617188 | 9e-6 / 1.1e-5 / 4e-6 |
| 16 | 0.654351 | 0.609375 | 7e-6 / 1.0e-5 / 2e-6 |

Merge parity at these checkpoints was 2.700e-13 in float64 (`03` §4.5's bar is 1e-5
absolute; see CLAUDE.md trap 15 for why the check runs in float64 and why the tolerance did
not move). The three `*_drift_from_base` columns are populated, which is what `--base`
exists to do — the point the verification was making.

Step 0 is saved before the first optimiser update, so its drift sits at the `1 − cosine`
floor rather than at zero: the two assertions that catch a base which is secretly the
checkpoint, and a base which is secretly a different model.
