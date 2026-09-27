# -*- coding: utf-8 -*-
"""evaluate_transformation.py — checkpoints to fingerprints for E6 (WP10).

``03_MODULE_SPEC_e6_transformation.md`` §6. Turns a finished (or in-progress) training run
into the ``checkpoint_metrics.csv`` that ``check_pilot_gate.py`` and
``aggregate_transformation.py`` both read, written to ``05_SCHEMAS_AND_CONTRACTS.md`` §2.

    python transformation_inheritance/evaluate_transformation.py \\
      --run-dir transformation_inheritance/results/e6a/D2/seed0 \\
      --steps all --engine nnsight --with-delta-ce --resume

Nothing here computes attention, a cross-entropy, or a band reduction: every number comes
from :func:`fingerprint_runner.compute_fingerprint`, which dispatches into the frozen
instrument, and every distance from :mod:`inheritance_metrics`.

Four properties that are easy to get wrong, and are therefore explicit
----------------------------------------------------------------------
* **Resumability is a hard requirement** (master plan §4.1): E6A evaluation is 8
  checkpoints × 9 interventions × 300 blocks × 9 runs plus the 2,000-block ΔCE. See
  :func:`unit_is_complete` for the exact skip condition and the granularity note.
* **Comparisons are guarded, not assumed** (``05`` §7.1). Teacher and student must agree on
  ``manifest_sha256`` and ``intervention_registry_version``, the intervention set is
  :func:`fingerprint_runner.mutual_interventions` computed at runtime (design-delta D2),
  and a depth-band mismatch above ``BAND_AGREEMENT_TOL`` raises unless explicitly allowed.
* **ΔCE follows design-delta D3**: a fixed 300-block subset at every checkpoint for the
  trajectory, the full 2,000 blocks only at step 0 and the final step. The subset's
  sampling error is bootstrapped from per-item ΔCE, never presented as a point estimate.
* **Failures are data** (``05`` §7.2). A checkpoint that will not load, a corpus that fails
  wholesale, an OOM — each writes a row with ``status`` set and sentinel metrics, and the
  run continues. No unit is ever silently dropped.

``--smoke`` evaluates a ``train_distillation.py --smoke`` run directory using synthetic
corpora on CPU/fp32 with no downloads.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import inheritance_metrics as im  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import (  # noqa: E402
    BAND_AGREEMENT_TOL,
    band_agreement_report,
    normalised_depth_band,
)

#: v3 adds the two exploratory ``functional_cosine_{coarse,surgical}_to_teacher`` columns.
#: Bumped because the artefact shape gained keys, following the WP12 precedent; no existing
#: column was renamed or recomputed, so v2 rows stay readable beside v3 rows.
#: v4 adds explicit E6B-family dispatch and tokenizer-revision propagation for isolated
#: model-scale extensions. Existing experiment ids and metric definitions are unchanged.
EVALUATOR_VERSION = "evaluate_transformation_v4"

#: ``08`` §4 — a public reference model has no training step and no training seed. Both
#: are written as ``-1`` so ``aggregate_transformation.is_reference_condition`` can tell a
#: fixed reference from a third replicate without matching on the condition name, and so a
#: reference row can never be mistaken for step 0 of a run.
REFERENCE_STEP = -1
REFERENCE_SEED = -1
DEFAULT_REFERENCE_CONDITION = "P8M"

#: Shared with ``train_distillation.DEFAULT_CORPUS_CACHE`` — the same packed corpus serves
#: training and evaluation, so they must name the same directory or each pays the pack.
DEFAULT_CORPUS_CACHE = _REPO / ".corpus_cache"

#: HF architecture class -> the ``ARCH_SPECS`` key the fingerprint runner needs.
ARCH_BY_CLASS: Dict[str, str] = {
    "GPTNeoForCausalLM": "neo",
    "GPT2LMHeadModel": "gpt2",
    "Qwen2ForCausalLM": "qwen",
    "OPTForCausalLM": "opt",
}

#: design-delta D3: trajectory ΔCE subset, and the full set used only at the endpoints.
CE_TRAJECTORY_BLOCKS = 300
CE_ENDPOINT_BLOCKS = 2000

#: ``05`` §2 column order. Written verbatim; columns may be added, never renamed (§7.5).
METRIC_COLUMNS: Tuple[str, ...] = (
    "experiment_id", "run_id", "condition", "seed", "checkpoint_step", "corpus_id",
    "manifest_sha256", "validation_ce", "validation_ce_n_blocks", "task_accuracy",
    "task_nll", "ece_10bin", "baseline_sink", "frac_cells_above_0_2",
    "carrier_concentration", "route_a_share", "relocation_ratio_swap_epe",
    "fingerprint_json", "delta_ce_json", "depth_profile_16_json", "top_carriers_json",
    "massive_coords_json", "n_keys_used", "fingerprint_cosine_to_teacher",
    "fingerprint_spearman_to_teacher", "fingerprint_l1_to_teacher",
    "category_agreement_to_teacher", "topology_wasserstein_to_teacher",
    "topology_spearman_to_teacher", "functional_cosine_to_teacher",
    "carrier_jaccard_to_teacher", "fingerprint_drift_from_base",
    "topology_drift_from_base", "carrier_drift_from_base", "n_items", "n_failed",
    "wallclock_s", "engine", "dtype", "layer_band", "intervention_registry_version",
    "git_sha",
    # --- added by WP10 beyond `05` §2 (adding is permitted, renaming is not) ---
    "status", "warning", "checkpoint_sha256", "delta_ce_ci_json",
    "delta_ce_n_blocks", "band_depth_interval", "band_version", "evaluator_version",
    "dropped_keys_json", "measured_utc",
    # --- exploratory, added after the first pilot; see COARSE_INTERVENTIONS ---
    "functional_cosine_coarse_to_teacher", "functional_cosine_surgical_to_teacher",
)

#: The two interventions that ablate a whole sub-system rather than probing the sink:
#: ``int_g`` zeroes every MLP and ``int_h`` zeroes all positional embeddings. Their ΔCE is
#: one to two orders of magnitude larger than any other intervention's in every model
#: measured, so they dominate the norm of the ΔCE vector and
#: ``functional_cosine_to_teacher`` is close to 1 for any pair of models that are both
#: damaged by them — including a randomly-initialised one.
#:
#: The split is reported, **not** substituted: ``functional_cosine_to_teacher`` remains the
#: pre-registered primary metric (design §8.7 / §18), and these two columns are exploratory
#: (``05`` §5 permits adding a column). On the 2,000-step pilot the all-key cosine was
#: 0.955-0.967 for D0/D1/D2 while the surgical-only cosine was 0.108-0.131 — and the public
#: TinyStories-8M scored 0.814 on the same subset, so the subset discriminates rather than
#: merely being noisier.
COARSE_INTERVENTIONS: Tuple[str, ...] = ("int_g", "int_h")

#: Every metric column that must be blank on a failed unit, so a sentinel is never read as
#: a measurement. Identity/provenance columns are still filled — a failure is a row.
_SENTINEL_COLUMNS = tuple(
    c for c in METRIC_COLUMNS
    if c not in ("experiment_id", "run_id", "condition", "seed", "checkpoint_step",
                 "corpus_id", "status", "warning", "engine", "dtype", "git_sha",
                 "checkpoint_sha256", "evaluator_version", "measured_utc",
                 "intervention_registry_version"))


# ═══════════════════════════════════════════════════════════════════════════════
# Run discovery
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RunInfo:
    """Identity of a training run, read from its ``run_config.json``."""

    run_dir: Path
    run_id: str
    experiment_id: str
    condition: str
    seed: int
    teacher: Optional[str]
    teacher_revision: Optional[str]
    tokenizer_name: Optional[str]
    smoke: bool
    #: ``data.dataset`` as recorded by the trainer. Selects the in-domain corpus provider,
    #: so the run is scored on the corpus it was trained on. ``None`` in a run directory
    #: written before the E6A-GPT2 arm, which means TinyStories.
    dataset_name: Optional[str] = None
    tokenizer_revision: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)


def read_run_info(run_dir: Path) -> RunInfo:
    """Load ``run_config.json``; fall back to the directory layout when fields are absent.

    Run directories written before WP10 added the teacher fields still evaluate — the
    teacher then has to come from ``--config`` or ``--teacher``, and the caller says so.
    """
    path = Path(run_dir) / "run_config.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. evaluate_transformation.py evaluates a run directory "
            "written by train_distillation.py; point --run-dir at one.")
    raw = json.loads(path.read_text(encoding="utf-8"))
    run_dir = Path(run_dir)
    seed = raw.get("training_seed")
    if seed is None:
        seed = int(run_dir.name.replace("seed", "") or 0)
    condition = raw.get("condition_id") or run_dir.parent.name
    experiment_id = raw.get("experiment_id") or run_dir.parent.parent.name
    return RunInfo(
        run_dir=run_dir,
        run_id=raw.get("run_id") or f"{experiment_id}_{condition}_seed{seed}",
        experiment_id=str(experiment_id), condition=str(condition), seed=int(seed),
        teacher=raw.get("teacher"), teacher_revision=raw.get("teacher_revision"),
        tokenizer_name=raw.get("tokenizer_name"), smoke=bool(raw.get("smoke", False)),
        dataset_name=raw.get("dataset_name"),
        tokenizer_revision=raw.get("tokenizer_revision"),
        raw=raw)


def is_e6b_run(run: RunInfo) -> bool:
    """Whether ``run`` is a sentiment-adaptation arm, including scale extensions.

    The original arm predates ``experiment_family`` and is recognised by its historical
    id. New arms declare the family explicitly, so a new experiment id can remain isolated
    without losing SST-2 evaluation or base-drift measurements.
    """
    family = run.raw.get("experiment_family")
    if family is not None:
        return str(family).strip().lower() == "e6b"
    return run.experiment_id == "e6b"


def discover_checkpoints(run_dir: Path) -> List[Tuple[int, Path]]:
    """``[(step, path), ...]`` ascending, from ``checkpoints/step_<n>/``."""
    root = Path(run_dir) / "checkpoints"
    if not root.exists():
        return []
    found: List[Tuple[int, Path]] = []
    for child in sorted(root.iterdir()):
        if not child.is_dir() or not child.name.startswith("step_"):
            continue
        try:
            found.append((int(child.name.split("_", 1)[1]), child))
        except ValueError:
            continue
    return sorted(found)


def select_steps(available: Sequence[Tuple[int, Path]], spec: str
                 ) -> List[Tuple[int, Path]]:
    """Resolve ``--steps all`` or a comma-separated list against the discovered set."""
    if spec.strip().lower() == "all":
        return list(available)
    wanted = {int(part) for part in spec.split(",") if part.strip()}
    chosen = [(step, path) for step, path in available if step in wanted]
    missing = sorted(wanted - {step for step, _ in chosen})
    if missing:
        raise SystemExit(
            f"--steps requested {missing} but the run directory has no such checkpoint. "
            f"Available: {[s for s, _ in available]}")
    return chosen


def _arch_of_classes(classes: Sequence[str], source: str) -> str:
    for name in classes:
        if name in ARCH_BY_CLASS:
            return ARCH_BY_CLASS[name]
    raise ValueError(
        f"{source}: architectures {list(classes)} map to no known arch key. Known: "
        f"{sorted(set(ARCH_BY_CLASS))}. Add the mapping deliberately rather than guessing "
        "— the intervention registry differs per architecture.")


def fingerprint_dir(ckpt: Path) -> Path:
    """The directory inside a checkpoint that holds plain HF weights.

    An E6B LoRA checkpoint is ``step_<n>/{adapter,merged}/`` — the adapter is a PEFT
    artefact the frozen GPT-2 harness cannot resolve module paths through, so only the
    merged copy is ever fingerprinted (`03` §4.5). Every other condition writes the model
    at the step root. Detected by looking for the weights rather than by reading the
    condition name, so a renamed condition cannot send the evaluator to the adapter.
    """
    ckpt = Path(ckpt)
    merged = ckpt / "merged"
    if (merged / "config.json").exists():
        return merged
    return ckpt


def arch_of_checkpoint(ckpt: Path) -> str:
    """Map the checkpoint's own ``config.json`` architecture to an ``ARCH_SPECS`` key.

    Read from the checkpoint rather than from ``run_config.json`` so a mislabelled run
    config cannot silently fingerprint a model with the wrong arch registry.
    """
    config_path = fingerprint_dir(ckpt) / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"{config_path} not found; cannot infer architecture")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    return _arch_of_classes(payload.get("architectures") or [], str(config_path))


def arch_of_model_id(model_id: str, *, revision: Optional[str] = None) -> str:
    """The ``ARCH_SPECS`` key of a hub id or local directory, read from its own config.

    The public reference (``08`` §4) is not a checkpoint directory, so
    :func:`arch_of_checkpoint` cannot reach it; the architecture is still read from the
    model's own config rather than assumed to match the students'.
    """
    local = Path(model_id) / "config.json"
    if local.exists():
        payload = json.loads(local.read_text(encoding="utf-8"))
        return _arch_of_classes(payload.get("architectures") or [], str(local))
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(model_id, revision=revision)
    classes = getattr(config, "architectures", None) or []
    if not classes:
        classes = [type(config).__name__.replace("Config", "ForCausalLM")]
    return _arch_of_classes(classes, str(model_id))


def first_resolvable_arch(checkpoints: Sequence[Tuple[int, Path]]) -> Optional[str]:
    """Architecture of the first checkpoint whose config can be read, else ``None``.

    Used only for the teacher hint. Each checkpoint's own arch is still read individually
    inside the loop, so a run whose checkpoints somehow disagree cannot be fingerprinted
    with one checkpoint's registry applied to another's weights.
    """
    for _step, path in checkpoints:
        try:
            return arch_of_checkpoint(path)
        except Exception:   # best-effort hint; the real per-checkpoint read still raises
            continue
    return None


def checkpoint_sha(ckpt: Path) -> Optional[str]:
    path = Path(ckpt) / "checkpoint_sha256.txt"
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def resolve_tokenizer_source(run: RunInfo, ckpt: Path, *, smoke: bool,
                             override: Optional[str] = None) -> str:
    """The id or local path the tokenizer is loaded from, resolved once.

    ``save_pretrained`` on a checkpoint writes the *model* only, so the checkpoint
    directory has no tokenizer and ``load_handle`` must be told where to find one — both
    for the corpora and for the traced model. A smoke run keeps its word-level tokenizer
    under ``<run_dir>/_smoke/tokenizer``; a real run records an HF id.
    """
    if override:
        return override
    local = run.run_dir / "_smoke" / "tokenizer"
    if smoke and local.exists():
        return str(local)
    name = run.tokenizer_name
    if name and name not in ("unknown", "smoke_tokenizer"):
        return str(name)
    if local.exists():
        return str(local)
    for candidate in (Path(ckpt), fingerprint_dir(ckpt)):
        if (candidate / "tokenizer.json").exists():
            return str(candidate)
    raise SystemExit(
        f"run_config.json records tokenizer_name={name!r} and no local tokenizer was "
        "found next to the run. Pass --tokenizer explicitly.")


def load_tokenizer(source: str, *, revision: Optional[str] = None):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(source, revision=revision)


# ═══════════════════════════════════════════════════════════════════════════════
# Corpora
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class CorpusSet:
    """The corpora one checkpoint is fingerprinted on, plus the ΔCE corpora."""

    sink: Dict[str, Any]                 # corpus_id -> Corpus
    ce_trajectory: Optional[Any]
    ce_endpoint: Optional[Any]


#: ``run_config.json``'s ``dataset_name`` -> the in-domain corpus provider that reads it.
#: The evaluator must measure the sink on the corpus the run was *trained* on, so this
#: mirrors ``train_distillation.BLOCK_LOADERS`` one-for-one; the two are asserted equal by
#: ``tests/test_e6a_gpt2_configs.py`` so an arm can never train on one corpus and be scored
#: on another.
IN_DOMAIN_PROVIDERS = {
    "roneneldan/TinyStories": cp.tinystories_corpus,
    "Skylion007/openwebtext": cp.openwebtext_corpus,
}

#: What an absent ``dataset_name`` means — every run directory written before the E6A-GPT2
#: arm. A *present but unrecognised* value is refused, never folded into this default
#: (CLAUDE.md trap 13): scoring a run on the wrong corpus is silent and unrecoverable.
DEFAULT_DATASET = "roneneldan/TinyStories"


def resolve_in_domain_provider(dataset_name: Optional[str]):
    """The corpus provider for ``dataset_name``, or a refusal naming the registry."""
    resolved = str(dataset_name) if dataset_name else DEFAULT_DATASET
    provider = IN_DOMAIN_PROVIDERS.get(resolved)
    if provider is None:
        raise ValueError(
            f"run_config dataset_name {resolved!r} has no in-domain corpus provider. "
            f"Known datasets: {sorted(IN_DOMAIN_PROVIDERS)}. Add one to "
            "evaluate_transformation.IN_DOMAIN_PROVIDERS rather than scoring this run on "
            "a corpus it was not trained on.")
    return provider


def build_corpora(run: RunInfo, tokenizer, *, smoke: bool, seed: int,
                  n_sink_blocks: int = 300, with_delta_ce: bool = False,
                  block_size: int = 128, corpus_cache=None) -> CorpusSet:
    """In-domain, cross-domain (+ SST-2 for E6B) corpora and the two ΔCE corpora.

    ``smoke`` swaps every download-backed provider for :func:`synthetic_corpus`, which is
    dataset-free, so the whole evaluator is exercisable offline.

    ``corpus_cache`` is forwarded to providers that accept it — today only
    :func:`corpus_providers.openwebtext_corpus`, whose window sits behind 400,000 streamed
    documents. It is passed by *capability*, not by dataset name, so adding a cacheable
    provider needs no change here; ``tinystories_corpus`` does not accept it and its
    already-archived records are therefore untouched.
    """
    sink: Dict[str, Any] = {}
    if smoke:
        in_domain = cp.synthetic_corpus(tokenizer, "shuffled_natural", 4, seed,
                                        cut_length=16)
        cross = cp.synthetic_corpus(tokenizer, "random_zipf", 4, seed + 1, cut_length=16)
        sink[in_domain.corpus_id] = in_domain
        sink[cross.corpus_id] = cross
        ce_traj = cp.synthetic_corpus(tokenizer, "random_uniform", 4, seed + 2,
                                      cut_length=16) if with_delta_ce else None
        return CorpusSet(sink=sink, ce_trajectory=ce_traj, ce_endpoint=None)

    provider = resolve_in_domain_provider(getattr(run, "dataset_name", None))
    extra = ({"cache_root": corpus_cache}
             if corpus_cache is not None
             and "cache_root" in inspect.signature(provider).parameters else {})
    in_domain = provider(tokenizer, "validation", n_sink_blocks,
                         block_size=block_size, seed=seed, purpose="sink", **extra)
    sink[in_domain.corpus_id] = in_domain
    cross = cp.frozen_e1_corpus(tokenizer)
    sink[cross.corpus_id] = cross
    if is_e6b_run(run):
        sst2 = cp.sst2_prompt_corpus(tokenizer, "validation", seed=seed)
        sink[sst2.corpus_id] = sst2

    ce_traj = ce_end = None
    if with_delta_ce:
        # ΔCE reads the same dataset as the sink corpus, from the disjoint `ppl` window.
        ce_traj = provider(tokenizer, "validation", CE_TRAJECTORY_BLOCKS,
                           block_size=block_size, seed=seed, purpose="ppl", **extra)
        ce_end = provider(tokenizer, "validation", CE_ENDPOINT_BLOCKS,
                          block_size=block_size, seed=seed, purpose="ppl", **extra)
    return CorpusSet(sink=sink, ce_trajectory=ce_traj, ce_endpoint=ce_end)


# ═══════════════════════════════════════════════════════════════════════════════
# The unit ledger (resumability)
# ═══════════════════════════════════════════════════════════════════════════════
#
# `03` §6.3 names the unit of work `(run_id, step, corpus_id, intervention)`. It is
# realised at `(run_id, step, corpus_id)` granularity with the intervention set recorded
# per unit, because `NNsightEngine.run_all` computes the entire a-j battery in ONE forward
# sweep per corpus item. Resuming below that granularity would re-run the sweep per
# intervention and *increase* total forwards — the opposite of what the requirement is for.
# The intervention set is part of the ledger key, so adding an intervention still
# invalidates and recomputes every affected unit.


def unit_key(run_id: str, step: int, corpus_id: str) -> str:
    return f"{run_id}|{step}|{corpus_id}"


def read_ledger(run_dir: Path) -> Dict[str, Dict[str, Any]]:
    path = Path(run_dir) / "eval_units.jsonl"
    if not path.exists():
        return {}
    ledger: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        ledger[row["unit"]] = row          # later entries win: a --force rewrite
    return ledger


def append_ledger(run_dir: Path, row: Dict[str, Any]) -> None:
    path = Path(run_dir) / "eval_units.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, default=str) + "\n")


def unit_is_complete(ledger: Dict[str, Dict[str, Any]], key: str,
                     expected: Dict[str, Any]) -> bool:
    """True when the ledger records this unit finished under identical conditions.

    Every field of ``expected`` must match: intervention set, ``manifest_sha256``,
    registry version, band and ``checkpoint_sha256``. Any drift recomputes, which is what
    stops a resumed run from mixing rows measured with two different instruments.
    """
    row = ledger.get(key)
    if row is None or row.get("status") != "ok":
        return False
    return all(row.get(field_name) == value for field_name, value in expected.items())


# ═══════════════════════════════════════════════════════════════════════════════
# Row construction
# ═══════════════════════════════════════════════════════════════════════════════


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _finite(value: Any) -> Any:
    """NaN/inf -> ``None`` so the CSV carries an empty cell rather than the text 'nan'."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return number if math.isfinite(number) else None


def blank_row(run: RunInfo, step: int, corpus_id: str, *, status: str, warning: str,
              engine: str, dtype: str, checkpoint_sha256: Optional[str]
              ) -> Dict[str, Any]:
    """A failure row: identity and provenance filled, every metric an empty sentinel.

    ``05`` §7.2 — "no pipeline drops an example silently". A reader can tell this apart
    from a measurement because ``status != "ok"`` and every metric cell is empty.
    """
    row = {column: None for column in METRIC_COLUMNS}
    row.update({
        "experiment_id": run.experiment_id, "run_id": run.run_id,
        "condition": run.condition, "seed": run.seed, "checkpoint_step": step,
        "corpus_id": corpus_id, "status": status, "warning": warning,
        "engine": engine, "dtype": dtype, "checkpoint_sha256": checkpoint_sha256,
        "git_sha": prov.git_sha(), "evaluator_version": EVALUATOR_VERSION,
        "measured_utc": prov.utc_now(),
        "intervention_registry_version": fr.INTERVENTION_REGISTRY_VERSION,
    })
    for column in _SENTINEL_COLUMNS:
        row[column] = None
    return row


def add_drift_columns(row: Dict[str, Any], record, base_record, keys: Sequence[str]
                      ) -> Dict[str, Any]:
    """E6B's three ``*_drift_from_base`` columns (WP6).

    Drift is distance from the model's **own** starting point — F0, the untrained base —
    which is a different comparand from ``*_to_teacher`` and answers a different question:
    not "did it inherit?" but "how far did adaptation move it?". Design §9.8's early-warning
    analysis reads these, and `05` §2 has carried the columns since WP10 while nothing
    populated them.

    Absent a base record the columns stay ``None`` with a recorded reason. Never 0.0: zero
    drift is a *measurement*, and a run that simply had no comparand must not be readable
    as one that did not move.
    """
    import numpy as np

    if base_record is None:
        return row
    row["fingerprint_drift_from_base"] = _finite(
        im.fingerprint_drift(record, base_record, keys))
    row["topology_drift_from_base"] = _finite(
        im.topology_drift(record.depth_profile_16, base_record.depth_profile_16))
    try:
        row["carrier_drift_from_base"] = _finite(im.carrier_drift(
            np.asarray(record.per_head_sink), np.asarray(base_record.per_head_sink)))
    except ValueError as exc:
        row["carrier_drift_from_base"] = None
        row["warning"] = "; ".join(x for x in (row.get("warning", ""),
                                               f"carrier_drift: {exc}") if x)
    return row


def build_row(run: RunInfo, step: int, record, teacher_record, keys: Sequence[str], *,
              validation_ce: Optional[float], validation_ce_n_blocks: Optional[int],
              delta_ce_ci: Optional[Dict[str, Any]], delta_ce_n_blocks: Optional[int],
              checkpoint_sha256: Optional[str], warning: str = "",
              base_record=None, task_metrics: Optional[Dict[str, Any]] = None
              ) -> Dict[str, Any]:
    """One ``05`` §2 row from a student record and (optionally) the teacher's.

    Every distance is computed by :mod:`inheritance_metrics` over ``keys``, which is the
    mutually-defined intervention set (design-delta D2) — never a union, never a
    hard-coded list.
    """
    row = {column: None for column in METRIC_COLUMNS}
    row.update({
        "experiment_id": run.experiment_id,
        "run_id": run.run_id,
        "condition": run.condition,
        "seed": run.seed,
        "checkpoint_step": step,
        "corpus_id": record.corpus_id,
        "manifest_sha256": record.manifest_sha256,
        "validation_ce": _finite(validation_ce),
        "validation_ce_n_blocks": validation_ce_n_blocks,
        "baseline_sink": _finite(record.baseline_sink),
        "frac_cells_above_0_2": _finite(record.frac_cells_above_0_2),
        "carrier_concentration": _finite(record.carrier_concentration),
        "fingerprint_json": _json(record.fingerprint),
        "delta_ce_json": _json(record.delta_ce) if record.delta_ce is not None else None,
        "depth_profile_16_json": _json(record.depth_profile_16),
        "top_carriers_json": _json(record.top_carrier_heads),
        "massive_coords_json": _json(record.massive_coords),
        "n_items": record.n_items,
        "n_failed": record.n_failed,
        "wallclock_s": _finite(record.wallclock_s),
        "engine": record.engine,
        "dtype": record.dtype,
        "layer_band": f"[{record.band[0]},{record.band[1]})",
        "band_depth_interval": _json(list(record.band_depth)),
        "band_version": record.band_version,
        "intervention_registry_version": record.intervention_registry_version,
        "git_sha": prov.git_sha(),
        "status": "ok",
        "warning": warning,
        "checkpoint_sha256": checkpoint_sha256,
        "delta_ce_ci_json": _json(delta_ce_ci) if delta_ce_ci is not None else None,
        "delta_ce_n_blocks": delta_ce_n_blocks,
        "evaluator_version": EVALUATOR_VERSION,
        "measured_utc": prov.utc_now(),
    })

    # Task metrics are *read* from the trainer's eval_log, exactly as validation_ce is:
    # the trainer measured accuracy, NLL and ECE with the model's own scorer, and a second
    # implementation here would be a second source of truth for numbers `05` §2 columns
    # and design §18's E6B criterion 1 both depend on.
    for column in ("task_accuracy", "task_nll", "ece_10bin"):
        if task_metrics and task_metrics.get(column) is not None:
            row[column] = _finite(task_metrics[column])

    if teacher_record is None:
        row["n_keys_used"] = len(keys)
        row["dropped_keys_json"] = _json([])
        return add_drift_columns(row, record, base_record, keys)

    fp_report = im.fingerprint_report(record, teacher_record, keys)
    topo = im.topology_report(record.depth_profile_16, teacher_record.depth_profile_16)
    row.update({
        "n_keys_used": fp_report["n_keys_used"],
        "dropped_keys_json": _json(fp_report["dropped_keys"]),
        "fingerprint_cosine_to_teacher": _finite(fp_report["cosine"]),
        "fingerprint_spearman_to_teacher": _finite(fp_report["spearman"]),
        "fingerprint_l1_to_teacher": _finite(fp_report["l1_normalised"]),
        "category_agreement_to_teacher": _finite(fp_report["category_agreement"]),
        "topology_wasserstein_to_teacher": _finite(topo["wasserstein"]),
        "topology_spearman_to_teacher": _finite(topo["spearman"]),
    })

    # Carrier overlap needs equal head counts; `weighted_jaccard` raises rather than
    # truncate (02 §4.3). A 4-layer teacher vs 8-layer student is handled by depth
    # alignment inside it; unequal heads is a real incomparability and is recorded.
    try:
        row["carrier_jaccard_to_teacher"] = _finite(im.weighted_jaccard(
            np.asarray(record.per_head_sink), np.asarray(teacher_record.per_head_sink)))
    except ValueError as exc:
        row["carrier_jaccard_to_teacher"] = None
        row["warning"] = "; ".join(x for x in (row["warning"],
                                               f"carrier_jaccard: {exc}") if x)

    if record.delta_ce is not None and teacher_record.delta_ce is not None:
        func_keys = [k for k in keys if k != "int_a"]
        row["functional_cosine_to_teacher"] = _finite(
            im.functional_cosine(record, teacher_record, func_keys))
        # Exploratory decomposition of the same vector: the coarse whole-subsystem ablations
        # and everything else, computed over the same mutual key set so the two partitions
        # are disjoint and exhaustive. Empty when a partition has no key, never 0.0 -- a
        # cosine over nothing is not a measurement of similarity.
        coarse_keys = [k for k in func_keys if k in COARSE_INTERVENTIONS]
        surgical_keys = [k for k in func_keys if k not in COARSE_INTERVENTIONS]
        if coarse_keys:
            row["functional_cosine_coarse_to_teacher"] = _finite(
                im.functional_cosine(record, teacher_record, coarse_keys))
        if surgical_keys:
            row["functional_cosine_surgical_to_teacher"] = _finite(
                im.functional_cosine(record, teacher_record, surgical_keys))
    return add_drift_columns(row, record, base_record, keys)


def bootstrap_delta_ce(record, *, n_boot: int, alpha: float, seed: int
                       ) -> Optional[Dict[str, Any]]:
    """Percentile CI per intervention over the ΔCE corpus items (design-delta D3).

    Labelled ``sampling_over_corpus_items``: this is the uncertainty of estimating the mean
    ΔCE from a 300-block *subset* of the validation set. It is emphatically **not**
    model-training uncertainty, which only the seed axis can express (``07`` P10).
    """
    per_item = (record.provenance or {}).get("delta_ce_per_item")
    if not per_item:
        return None
    out: Dict[str, Any] = {"uncertainty_kind": "sampling_over_corpus_items",
                           "n_boot": n_boot, "alpha": alpha, "per_intervention": {}}
    for key, values in per_item.items():
        if not values:
            continue
        lo, hi = im.bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
        out["per_intervention"][key] = {
            "mean": float(np.mean(values)), "ci_lo": _finite(lo), "ci_hi": _finite(hi),
            "n_items": len(values)}
    return out if out["per_intervention"] else None


# ═══════════════════════════════════════════════════════════════════════════════
# CSV I/O
# ═══════════════════════════════════════════════════════════════════════════════


def read_metrics(path: Path):
    import pandas as pd

    if not Path(path).exists():
        return pd.DataFrame(columns=list(METRIC_COLUMNS))
    return pd.read_csv(path)


def write_metrics(path: Path, frame) -> Path:
    """Write with the ``05`` §2 column order, adding any column the frame gained."""
    import pandas as pd

    ordered = [c for c in METRIC_COLUMNS if c in frame.columns]
    extra = [c for c in frame.columns if c not in METRIC_COLUMNS]
    frame = frame[ordered + extra]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding="utf-8")
    return Path(path)


def upsert_row(frame, row: Dict[str, Any]):
    """Replace the row for this ``(run_id, step, corpus_id)`` or append it.

    Upsert rather than append so ``--force`` cannot leave two rows for one unit, which
    would double-count that checkpoint in every downstream mean.
    """
    import pandas as pd

    if len(frame):
        mask = ((frame["run_id"] == row["run_id"])
                & (frame["checkpoint_step"] == row["checkpoint_step"])
                & (frame["corpus_id"] == row["corpus_id"]))
        frame = frame[~mask]
    return pd.concat([frame, pd.DataFrame([row])], ignore_index=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Validation CE from the trainer's eval_log
# ═══════════════════════════════════════════════════════════════════════════════


def read_eval_log(run_dir: Path) -> Dict[int, Dict[str, Any]]:
    """``step -> eval row`` from ``eval_log.jsonl``, written by the trainer.

    Validation CE is *read*, never recomputed here: the trainer measured it with the same
    loss the model was trained under, and a second implementation would be a second source
    of truth for the number matched-loss selection depends on (``03`` §6.4).
    """
    path = Path(run_dir) / "eval_log.jsonl"
    if not path.exists():
        return {}
    rows: Dict[int, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            row = json.loads(line)
            rows[int(row.get("step", -1))] = row
    return rows


# ═══════════════════════════════════════════════════════════════════════════════
# Teacher
# ═══════════════════════════════════════════════════════════════════════════════


def load_teacher_handle(teacher: Optional[str], *, arch_hint: str, engine: str,
                        dtype: str, device: Optional[str], revision: Optional[str],
                        tokenizer_name: Optional[str],
                        tokenizer_revision: Optional[str] = None):
    """Load the teacher, or return ``(None, reason)`` when there is nothing to load.

    A smoke run synthesises its teacher in-process and has no id to reload, and a run
    config written before WP10 records none. Both cases produce student-only rows with the
    teacher columns empty and a recorded reason — never a fabricated similarity.
    """
    if not teacher:
        return None, ("no teacher recorded in run_config.json and none passed; "
                      "teacher-similarity columns are empty for this run")
    try:
        handle = fr.load_handle(arch_hint, teacher, engine=engine, dtype=dtype,
                                revision=revision, device=device,
                                tokenizer_name=tokenizer_name,
                                tokenizer_revision=tokenizer_revision)
    except Exception as exc:  # recorded, not fatal: the student rows are still valid
        return None, f"teacher {teacher!r} failed to load ({type(exc).__name__}: {exc})"
    return handle, ""


def guard_comparison(student_record, teacher_record, *, allow_band_mismatch: bool) -> None:
    """``05`` §7.1 hash-before-compare, plus the band-depth guard from ``06`` §4.

    Raises rather than returning a flag: a comparison across different corpora or
    different intervention registries is not a degraded measurement, it is a wrong one.
    """
    if student_record.manifest_sha256 != teacher_record.manifest_sha256:
        raise ValueError(
            "manifest_sha256 mismatch between student and teacher fingerprints "
            f"({student_record.manifest_sha256[:12]} vs "
            f"{teacher_record.manifest_sha256[:12]}). The two were measured on different "
            "corpora; the comparison is meaningless (05 §7.1).")
    if (student_record.intervention_registry_version
            != teacher_record.intervention_registry_version):
        raise ValueError(
            "intervention_registry_version mismatch: "
            f"{student_record.intervention_registry_version} vs "
            f"{teacher_record.intervention_registry_version}. Records from different "
            "registries must not be compared (05 §1.1).")
    report = band_agreement_report(student_record.num_layers, teacher_record.num_layers)
    if report["exceeds_tolerance"] and not allow_band_mismatch:
        raise ValueError(
            "normalised depth bands disagree by "
            f"{report['max_mismatch']:.3f} > {BAND_AGREEMENT_TOL} "
            f"(student {report['model_a']['depth_interval']}, teacher "
            f"{report['model_b']['depth_interval']}). Sink strengths would be averaged "
            "over non-comparable depth regions. Pass --allow-band-mismatch to record and "
            "proceed deliberately.")


# ═══════════════════════════════════════════════════════════════════════════════
# The evaluation loop
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_run(run_dir: Path, *, steps: str = "all", engine: str = "nnsight",
                 with_delta_ce: bool = False, resume: bool = True,
                 teacher: Optional[str] = None, cache_dir: Optional[Path] = None,
                 allow_band_mismatch: bool = False, device: Optional[str] = None,
                 dtype: str = "float32", smoke: bool = False,
                 n_sink_blocks: int = 300, tokenizer_name: Optional[str] = None,
                 bootstrap_n: int = 10000, bootstrap_alpha: float = 0.05,
                 public_reference: Optional[str] = None,
                 reference_condition: str = DEFAULT_REFERENCE_CONDITION,
                 reference_out: Optional[Path] = None,
                 base: Optional[str] = None,
                 corpus_cache=None,
                 progress: bool = True) -> Dict[str, Any]:
    """Fingerprint every selected checkpoint and write ``checkpoint_metrics.csv``.

    Returns a summary dict; all detail goes to the run directory. Never raises for a single
    failed unit — those become rows — but does raise for a mis-specified run (no
    checkpoints, unknown arch, guarded comparison violation).
    """
    run = read_run_info(run_dir)
    checkpoints = select_steps(discover_checkpoints(run.run_dir), steps)
    if not checkpoints:
        raise SystemExit(
            f"{run.run_dir}/checkpoints contains no step_<n> directories; run "
            "train_distillation.py first.")

    metrics_path = run.run_dir / "checkpoint_metrics.csv"
    frame = read_metrics(metrics_path)
    ledger = read_ledger(run.run_dir) if resume else {}
    eval_log = read_eval_log(run.run_dir)
    cache_dir = Path(cache_dir) if cache_dir else run.run_dir / "fingerprints"

    final_step = max(step for step, _ in discover_checkpoints(run.run_dir))
    teacher_id = teacher or run.teacher

    tokenizer_source = resolve_tokenizer_source(run, checkpoints[0][1], smoke=smoke,
                                                override=tokenizer_name)
    tokenizer = load_tokenizer(tokenizer_source, revision=run.tokenizer_revision)
    corpora = build_corpora(run, tokenizer, smoke=smoke, seed=run.seed,
                            n_sink_blocks=n_sink_blocks, with_delta_ce=with_delta_ce,
                            corpus_cache=corpus_cache)

    # The teacher's arch is taken from the first checkpoint that can be read, not blindly
    # from checkpoints[0]: one corrupt checkpoint must become a row, not abort the run
    # before any checkpoint has been evaluated.
    arch_hint = first_resolvable_arch(checkpoints)
    if arch_hint is None:
        teacher_handle, teacher_note = None, (
            "no checkpoint's config.json could be read, so the teacher architecture is "
            "unknown; teacher-similarity columns are empty for this run")
    else:
        teacher_handle, teacher_note = load_teacher_handle(
            teacher_id, arch_hint=arch_hint, engine=engine, dtype=dtype, device=device,
            revision=run.teacher_revision, tokenizer_name=tokenizer_source,
            tokenizer_revision=run.tokenizer_revision)
    teacher_records: Dict[str, Any] = {}

    # E6B's drift columns measure distance from the run's own starting point (F0), which is
    # a different comparand from the teacher and answers a different question. Recorded in
    # run_config.json as `base_model`, so a run directory is self-describing; `--base`
    # overrides. Absent, the drift columns stay empty with this note attached.
    base_id = base or run.raw.get("base_model")
    base_handle, base_note = (None, "")
    if base_id and arch_hint is not None:
        base_handle, base_note = load_teacher_handle(
            base_id, arch_hint=arch_hint, engine=engine, dtype=dtype, device=device,
            revision=run.raw.get("base_revision"), tokenizer_name=tokenizer_source,
            tokenizer_revision=run.tokenizer_revision)
        if base_handle is None:
            base_note = f"base model: {base_note}"
    elif is_e6b_run(run):
        base_note = ("no base model recorded or passed; the *_drift_from_base columns are "
                     "empty for this run (they are never written as 0.0, which would read "
                     "as 'did not move')")
    base_records: Dict[str, Any] = {}

    started = time.time()
    n_written = n_skipped = n_failed = 0

    for step, ckpt in checkpoints:
        ckpt_sha = checkpoint_sha(ckpt)
        try:
            arch = arch_of_checkpoint(ckpt)
            handle = fr.load_handle(arch, fingerprint_dir(ckpt), engine=engine,
                                    dtype=dtype,
                                    device=device, tokenizer_name=tokenizer_source,
                                    tokenizer_revision=run.tokenizer_revision,
                                    checkpoint_step=step, checkpoint_sha256=ckpt_sha,
                                    local_files_only=True)
        except Exception as exc:
            for corpus_id in corpora.sink:
                row = blank_row(run, step, corpus_id, status="checkpoint_load_failed",
                                warning=f"{type(exc).__name__}: {exc}", engine=engine,
                                dtype=dtype, checkpoint_sha256=ckpt_sha)
                frame = upsert_row(frame, row)
                append_ledger(run.run_dir, {"unit": unit_key(run.run_id, step, corpus_id),
                                            "status": row["status"],
                                            "measured_utc": row["measured_utc"]})
                n_failed += 1
            write_metrics(metrics_path, frame)
            if progress:
                print(f"  step {step}: FAILED to load ({type(exc).__name__}) — "
                      "rows written with status, continuing")
            continue

        keys = (fr.mutual_interventions(teacher_handle, handle) if teacher_handle
                else fr.available_interventions(handle))
        band_start, band_end, _band_meta = normalised_depth_band(
            handle.nn_engine.num_layers)

        for corpus_id, corpus in corpora.sink.items():
            key = unit_key(run.run_id, step, corpus_id)
            expected = {
                "interventions": list(keys),
                "manifest_sha256": corpus.manifest_sha256,
                "registry_version": fr.INTERVENTION_REGISTRY_VERSION,
                "band": [band_start, band_end],
                "checkpoint_sha256": ckpt_sha,
            }
            has_row = len(frame) and bool((
                (frame["run_id"] == run.run_id)
                & (frame["checkpoint_step"] == step)
                & (frame["corpus_id"] == corpus_id)).any())
            if resume and has_row and unit_is_complete(ledger, key, expected):
                n_skipped += 1
                continue

            ce_corpus, ce_blocks = _ce_corpus_for(corpora, step, final_step,
                                                  with_delta_ce)
            try:
                record = fr.compute_fingerprint(
                    handle, corpus, band=(band_start, band_end), interventions=keys,
                    with_delta_ce=with_delta_ce, ce_corpus=ce_corpus,
                    cache_dir=cache_dir, run_id=run.run_id,
                    experiment_id=run.experiment_id, condition=run.condition,
                    seed=run.seed, progress=False,
                    delta_ce_per_item=bool(with_delta_ce))
            except Exception as exc:
                row = blank_row(run, step, corpus_id, status="fingerprint_failed",
                                warning=f"{type(exc).__name__}: {exc}", engine=engine,
                                dtype=dtype, checkpoint_sha256=ckpt_sha)
                frame = upsert_row(frame, row)
                append_ledger(run.run_dir, {"unit": key, "status": row["status"],
                                            "measured_utc": row["measured_utc"]})
                n_failed += 1
                continue

            teacher_record = None
            warning = "; ".join(x for x in (teacher_note, base_note) if x)
            if teacher_handle is not None:
                teacher_record = teacher_records.get(corpus_id)
                if teacher_record is None:
                    teacher_record = fr.compute_fingerprint(
                        teacher_handle, corpus, interventions=keys,
                        with_delta_ce=with_delta_ce, ce_corpus=ce_corpus,
                        cache_dir=cache_dir, run_id=f"teacher:{teacher_id}",
                        experiment_id=run.experiment_id, condition="teacher",
                        seed=run.seed, progress=False,
                        delta_ce_per_item=bool(with_delta_ce))
                    teacher_records[corpus_id] = teacher_record
                guard_comparison(record, teacher_record,
                                 allow_band_mismatch=allow_band_mismatch)

            base_record = None
            if base_handle is not None:
                base_record = base_records.get(corpus_id)
                if base_record is None:
                    base_record = fr.compute_fingerprint(
                        base_handle, corpus, interventions=keys,
                        with_delta_ce=with_delta_ce, ce_corpus=ce_corpus,
                        cache_dir=cache_dir, run_id=f"base:{base_id}",
                        experiment_id=run.experiment_id, condition="base",
                        seed=run.seed, progress=False,
                        delta_ce_per_item=bool(with_delta_ce))
                    base_records[corpus_id] = base_record
                # Same hash-before-compare guard the teacher gets: a base measured on a
                # different corpus or registry would produce a plausible drift that means
                # nothing (05 §7.1).
                guard_comparison(record, base_record,
                                 allow_band_mismatch=allow_band_mismatch)

            eval_row = eval_log.get(step, {})
            row = build_row(
                run, step, record, teacher_record, keys,
                base_record=base_record, task_metrics=eval_row,
                validation_ce=eval_row.get("validation_ce"),
                validation_ce_n_blocks=eval_row.get("n_batches"),
                delta_ce_ci=bootstrap_delta_ce(record, n_boot=bootstrap_n,
                                               alpha=bootstrap_alpha, seed=run.seed),
                delta_ce_n_blocks=ce_blocks, checkpoint_sha256=ckpt_sha,
                warning=warning)
            frame = upsert_row(frame, row)
            append_ledger(run.run_dir, {"unit": key, "status": "ok",
                                        "measured_utc": row["measured_utc"],
                                        **expected})
            n_written += 1
            if progress:
                print(f"  step {step:>6} {corpus_id}: sink={row['baseline_sink']} "
                      f"cos_to_teacher={row['fingerprint_cosine_to_teacher']} "
                      f"n_keys={row['n_keys_used']}")

        write_metrics(metrics_path, frame)
        del handle

    write_metrics(metrics_path, frame)

    # `08` §4 — evaluated here, in the same process, so it reuses the corpora and teacher
    # objects already built rather than reconstructing them. Its rows go to their own
    # directory so a P8M row can never be mistaken for a checkpoint of this run.
    reference_summary = None
    if public_reference:
        reference_dir = (Path(reference_out) if reference_out
                         else run.run_dir.parent.parent / reference_condition
                         / "reference")
        reference_summary = evaluate_public_reference(
            public_reference, reference_dir, teacher=teacher_id,
            tokenizer_source=tokenizer_source, corpora=corpora,
            teacher_handle=teacher_handle, teacher_records=teacher_records,
            condition=reference_condition, experiment_id=run.experiment_id,
            engine=engine, dtype=dtype, device=device, with_delta_ce=with_delta_ce,
            smoke=smoke, seed=run.seed, n_sink_blocks=n_sink_blocks,
            cache_dir=cache_dir, allow_band_mismatch=allow_band_mismatch,
            teacher_revision=run.teacher_revision, dataset_name=run.dataset_name,
            revision=run.raw.get("public_reference_revision"),
            tokenizer_revision=run.tokenizer_revision,
            bootstrap_n=bootstrap_n, corpus_cache=corpus_cache,
            bootstrap_alpha=bootstrap_alpha, progress=progress)

    summary = {
        "run_id": run.run_id, "run_dir": str(run.run_dir),
        "metrics_csv": str(metrics_path), "n_written": n_written,
        "n_skipped": n_skipped, "n_failed": n_failed,
        "n_checkpoints": len(checkpoints), "teacher": teacher_id,
        "teacher_note": teacher_note, "base": base_id, "base_note": base_note,
        "wallclock_s": time.time() - started,
        "public_reference": reference_summary,
        "evaluator_version": EVALUATOR_VERSION,
    }
    prov.write_json(run.run_dir / "evaluation_summary.json",
                    {**summary, **prov.provenance_block()})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# The public reference model (08 §4)
# ═══════════════════════════════════════════════════════════════════════════════


def evaluate_public_reference(model_id: str, out_dir: Path, *, teacher: Optional[str],
                              tokenizer_source: str, corpora: Optional[CorpusSet] = None,
                              teacher_handle=None, teacher_records=None,
                              condition: str = DEFAULT_REFERENCE_CONDITION,
                              experiment_id: str = "e6a", engine: str = "nnsight",
                              dtype: str = "float32", device: Optional[str] = None,
                              revision: Optional[str] = None,
                              with_delta_ce: bool = False, smoke: bool = False,
                              seed: int = 0, n_sink_blocks: int = 300,
                              cache_dir: Optional[Path] = None,
                              allow_band_mismatch: bool = False,
                              teacher_revision: Optional[str] = None,
                              tokenizer_revision: Optional[str] = None,
                              dataset_name: Optional[str] = None,
                              bootstrap_n: int = 10000, bootstrap_alpha: float = 0.05,
                              corpus_cache=None,
                              progress: bool = True) -> Dict[str, Any]:
    """Fingerprint a public checkpoint as a seedless reference condition (``08`` §4).

    Design §8.7 contrast 4 compares the independently trained public 8M against D1 and D2,
    which needs the 8M measured *through the same instrument*: the same teacher, the same
    corpora, the same normalised-depth band, the same dtype and the same intervention
    registry. Anything else and the contrast measures the instrument rather than the model.

    ``corpora``/``teacher_handle``/``teacher_records`` are passed in when this runs
    alongside :func:`evaluate_run` so the reference reuses the objects already built rather
    than rebuilding them — §4's requirement, and also the only way the ``manifest_sha256``
    guard is guaranteed rather than merely expected. Called standalone it rebuilds them
    with identical parameters, and :func:`guard_comparison` still raises if the digests
    disagree.

    Design §8.1 is load-bearing here: the public 8M is an independent-convergence
    reference and never initialises a student. Nothing in this function writes weights.
    """
    out_dir = Path(out_dir)
    started = time.time()
    run = RunInfo(run_dir=out_dir, run_id=f"{experiment_id}_{condition}_reference",
                  experiment_id=experiment_id, condition=condition, seed=REFERENCE_SEED,
                  teacher=teacher, teacher_revision=teacher_revision,
                  tokenizer_name=tokenizer_source, smoke=smoke,
                  dataset_name=dataset_name,
                  tokenizer_revision=tokenizer_revision,
                  raw={"public_reference": model_id, "reference_revision": revision})

    tokenizer = load_tokenizer(tokenizer_source, revision=tokenizer_revision)
    if corpora is None:
        corpora = build_corpora(run, tokenizer, smoke=smoke, seed=seed,
                                n_sink_blocks=n_sink_blocks, with_delta_ce=with_delta_ce,
                                corpus_cache=corpus_cache)
    cache_dir = Path(cache_dir) if cache_dir else out_dir / "fingerprints"

    metrics_path = out_dir / "checkpoint_metrics.csv"
    frame = read_metrics(metrics_path)

    arch = arch_of_model_id(model_id, revision=revision)
    if teacher_handle is None and teacher:
        teacher_handle, teacher_note = load_teacher_handle(
            teacher, arch_hint=arch, engine=engine, dtype=dtype, device=device,
            revision=teacher_revision, tokenizer_name=tokenizer_source,
            tokenizer_revision=tokenizer_revision)
    else:
        teacher_note = "" if teacher_handle is not None else (
            "no teacher supplied; teacher-similarity columns are empty for the reference")
    teacher_records = {} if teacher_records is None else teacher_records

    handle = fr.load_handle(arch, model_id, engine=engine, dtype=dtype, device=device,
                            revision=revision, tokenizer_name=tokenizer_source,
                            tokenizer_revision=tokenizer_revision)
    keys = (fr.mutual_interventions(teacher_handle, handle) if teacher_handle
            else fr.available_interventions(handle))
    band_start, band_end, _meta = normalised_depth_band(handle.nn_engine.num_layers)

    n_written = n_failed = 0
    for corpus_id, corpus in corpora.sink.items():
        # The reference has one measurement, so ΔCE always uses the endpoint corpus when
        # one exists — there is no trajectory to subsample along.
        ce_corpus = (corpora.ce_endpoint or corpora.ce_trajectory) if with_delta_ce \
            else None
        ce_blocks = len(ce_corpus.items) if ce_corpus is not None else None
        try:
            record = fr.compute_fingerprint(
                handle, corpus, band=(band_start, band_end), interventions=keys,
                with_delta_ce=with_delta_ce, ce_corpus=ce_corpus, cache_dir=cache_dir,
                run_id=run.run_id, experiment_id=experiment_id, condition=condition,
                seed=REFERENCE_SEED, progress=False,
                delta_ce_per_item=bool(with_delta_ce))
        except Exception as exc:
            row = blank_row(run, REFERENCE_STEP, corpus_id,
                            status="fingerprint_failed",
                            warning=f"{type(exc).__name__}: {exc}", engine=engine,
                            dtype=dtype, checkpoint_sha256=None)
            frame = upsert_row(frame, row)
            n_failed += 1
            continue

        teacher_record = None
        if teacher_handle is not None:
            teacher_record = teacher_records.get(corpus_id)
            if teacher_record is None:
                teacher_record = fr.compute_fingerprint(
                    teacher_handle, corpus, interventions=keys,
                    with_delta_ce=with_delta_ce, ce_corpus=ce_corpus,
                    cache_dir=cache_dir, run_id=f"teacher:{teacher}",
                    experiment_id=experiment_id, condition="teacher", seed=seed,
                    progress=False, delta_ce_per_item=bool(with_delta_ce))
                teacher_records[corpus_id] = teacher_record
            guard_comparison(record, teacher_record,
                             allow_band_mismatch=allow_band_mismatch)

        row = build_row(
            run, REFERENCE_STEP, record, teacher_record, keys,
            # A public checkpoint has no training curve, so there is no validation CE from
            # a trainer's eval_log to attach. `03` §6.4's matched-loss selection needs one,
            # so it is measured here from the ΔCE corpus's own baseline rather than left
            # empty — and it is empty, honestly, when --with-delta-ce was not passed.
            validation_ce=record.baseline_ce,
            validation_ce_n_blocks=ce_blocks,
            delta_ce_ci=bootstrap_delta_ce(record, n_boot=bootstrap_n,
                                           alpha=bootstrap_alpha, seed=seed),
            delta_ce_n_blocks=ce_blocks, checkpoint_sha256=None,
            warning=teacher_note)
        frame = upsert_row(frame, row)
        n_written += 1
        if progress:
            print(f"  {condition} {corpus_id}: sink={row['baseline_sink']} "
                  f"cos_to_teacher={row['fingerprint_cosine_to_teacher']} "
                  f"n_keys={row['n_keys_used']}")

    write_metrics(metrics_path, frame)
    summary = {
        "run_id": run.run_id, "run_dir": str(out_dir), "condition": condition,
        "public_reference": model_id, "reference_revision": revision,
        "tokenizer_revision": tokenizer_revision,
        "metrics_csv": str(metrics_path), "checkpoint_step": REFERENCE_STEP,
        "seed": REFERENCE_SEED, "n_written": n_written, "n_failed": n_failed,
        "teacher": teacher, "teacher_revision": teacher_revision,
        "teacher_note": teacher_note,
        "corpora": sorted(corpora.sink),
        "manifest_sha256": {cid: c.manifest_sha256 for cid, c in corpora.sink.items()},
        "band": [band_start, band_end], "interventions": list(keys),
        "intervention_registry_version": fr.INTERVENTION_REGISTRY_VERSION,
        "dtype": dtype, "engine": engine,
        "wallclock_s": time.time() - started,
        "evaluator_version": EVALUATOR_VERSION,
        "note": "08 §4 — a seedless reference condition: one measurement, no training "
                "seed, checkpoint_step = -1. Contrasts against it are labelled "
                "reference_vs_seeds and carry no uncertainty from this checkpoint.",
    }
    prov.write_json(out_dir / "reference_summary.json",
                    {**summary, **prov.provenance_block()})
    del handle
    return summary


def _ce_corpus_for(corpora: CorpusSet, step: int, final_step: int,
                   with_delta_ce: bool) -> Tuple[Optional[Any], Optional[int]]:
    """design-delta D3: 300-block subset every checkpoint, 2,000 only at the endpoints."""
    if not with_delta_ce:
        return None, None
    if step in (0, final_step) and corpora.ce_endpoint is not None:
        return corpora.ce_endpoint, len(corpora.ce_endpoint.items)
    if corpora.ce_trajectory is None:
        return None, None
    return corpora.ce_trajectory, len(corpora.ce_trajectory.items)


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def reference_from_config(config_path: Path, out_dir: Path, *,
                          model_id: Optional[str] = None, **kwargs) -> Dict[str, Any]:
    """Evaluate the public reference from an E6A config, with no run directory.

    The reference is not produced by training, so requiring a finished run to measure it
    would make design §8.7 contrast 4 wait on 1.5–3 GPU-days that have nothing to do with
    it. The config supplies exactly what the instrument needs — teacher, tokenizer and the
    reference id — and every other parameter is the same one ``evaluate_run`` would pass.
    """
    import yaml

    config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    configured_reference = config.get("public_reference")
    reference = model_id or configured_reference
    if not reference:
        raise SystemExit(
            f"{config_path} records no `public_reference:` and none was passed. "
            "--evaluate-public-reference needs a model id (08 §4).")
    tokenizer_source = kwargs.pop("tokenizer_source", None) or config.get("tokenizer") \
        or config.get("teacher")
    if not tokenizer_source:
        raise SystemExit(f"{config_path} records neither `tokenizer:` nor `teacher:`; "
                         "pass --tokenizer explicitly.")
    requested_revision = kwargs.pop("revision", None)
    reference_revision = (requested_revision if requested_revision is not None else
                          config.get("public_reference_revision")
                          if reference == configured_reference else None)
    return evaluate_public_reference(
        reference, out_dir, teacher=kwargs.pop("teacher", None) or config.get("teacher"),
        tokenizer_source=str(tokenizer_source),
        teacher_revision=config.get("teacher_revision"),
        tokenizer_revision=(kwargs.pop("tokenizer_revision", None)
                            or config.get("tokenizer_revision")),
        # The in-domain corpus follows the config's own dataset, so a gpt2-arm config
        # measures its reference on OpenWebText and a TinyStories config on TinyStories.
        dataset_name=(config.get("data") or {}).get("dataset"),
        experiment_id=str(config.get("experiment_id", "e6a")),
        revision=reference_revision,
        **kwargs)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", default=None,
                        help="a run directory written by train_distillation.py "
                             "(omit only with --evaluate-public-reference --config)")
    parser.add_argument("--config", default=None,
                        help="an E6A config; with --evaluate-public-reference this "
                             "measures the public model without a run directory")
    parser.add_argument("--evaluate-public-reference", dest="public_reference",
                        nargs="?", const="", default=None,
                        help="also fingerprint the public reference model as a seedless "
                             "condition (08 §4); defaults to the config's "
                             "`public_reference:` when given no value")
    parser.add_argument("--reference-condition", default=DEFAULT_REFERENCE_CONDITION,
                        help="condition id written on the reference rows")
    parser.add_argument("--revision", default=None,
                        help="revision of --evaluate-public-reference (overrides config)")
    parser.add_argument("--reference-out", default=None,
                        help="where the reference's checkpoint_metrics.csv goes "
                             "(default <results>/<experiment>/<condition>/reference)")
    parser.add_argument("--steps", default="all",
                        help="'all' or a comma-separated list of checkpoint steps")
    parser.add_argument("--engine", default="nnsight", choices=("nnsight",),
                        help="fingerprint_runner implements engine='nnsight' only")
    parser.add_argument("--with-delta-ce", action="store_true",
                        help="also measure ΔCE per intervention (design-delta D3)")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True,
                        help="skip units already completed under identical conditions "
                             "(the default)")
    parser.add_argument("--force", dest="resume", action="store_false",
                        help="recompute every unit, replacing existing rows in place")
    parser.add_argument("--teacher", default=None,
                        help="teacher model id; overrides run_config.json['teacher']")
    parser.add_argument("--base", default=None,
                        help="E6B: the untrained base (F0) the *_drift_from_base columns "
                             "measure against; overrides run_config.json['base_model']")
    parser.add_argument("--tokenizer", default=None,
                        help="tokenizer id; overrides run_config.json['tokenizer_name']")
    parser.add_argument("--cache-dir", default=None,
                        help="fingerprint cache (default <run-dir>/fingerprints)")
    parser.add_argument("--allow-band-mismatch", action="store_true",
                        help="proceed and record when teacher/student depth bands differ "
                             "by more than the tolerance")
    parser.add_argument("--device", default=None)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--sink-blocks", type=int, default=300)
    parser.add_argument("--bootstrap-n", type=int, default=10000)
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic corpora, CPU/fp32, no downloads")
    parser.add_argument(
        "--corpus-cache", default=str(DEFAULT_CORPUS_CACHE),
        help=("directory for the packed-corpus cache, or 'none' to rebuild every time. "
              "Only providers that accept it use it — today OpenWebText, whose validation "
              "window sits behind 400,000 streamed documents that every evaluation "
              "otherwise re-reads."))
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    corpus_cache = (None if str(args.corpus_cache).lower() == "none"
                    else Path(args.corpus_cache))

    if args.run_dir is None:
        # Standalone reference mode: no run directory to evaluate, so a missing
        # --evaluate-public-reference is a mis-invocation rather than a default.
        if args.public_reference is None or not args.config:
            raise SystemExit(
                "--run-dir is required unless --evaluate-public-reference is passed "
                "together with --config <e6a config>.")
        # The default output directory follows the config's own experiment id, so a
        # gpt2-arm reference lands under results/e6a_gpt2/ and cannot be rglob'd into the
        # TinyStories arm's aggregation (`05` §7.1 / trap 12's failure in a new place).
        import yaml as _yaml

        _cfg = _yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
        out = Path(args.reference_out) if args.reference_out else (
            _REPO / "transformation_inheritance" / "results"
            / str(_cfg.get("experiment_id", "e6a"))
            / args.reference_condition / "reference")
        reference = reference_from_config(
            Path(args.config), out, model_id=args.public_reference or None,
            condition=args.reference_condition, engine=args.engine,
            dtype=args.dtype, device=args.device, with_delta_ce=args.with_delta_ce,
            revision=args.revision,
            smoke=args.smoke, n_sink_blocks=args.sink_blocks,
            cache_dir=Path(args.cache_dir) if args.cache_dir else None,
            allow_band_mismatch=args.allow_band_mismatch,
            bootstrap_n=args.bootstrap_n, teacher=args.teacher,
            tokenizer_source=args.tokenizer, corpus_cache=corpus_cache)
        print(f"E6 public reference {reference['public_reference']} "
              f"-> {reference['metrics_csv']}")
        print(f"  condition={reference['condition']} "
              f"step={reference['checkpoint_step']} seed={reference['seed']} "
              f"written={reference['n_written']} failed={reference['n_failed']}")
        if reference["teacher_note"]:
            print(f"  note: {reference['teacher_note']}")
        return 1 if reference["n_failed"] and not reference["n_written"] else 0

    public_reference = args.public_reference
    if public_reference == "":                      # `--evaluate-public-reference` bare
        import yaml

        if not args.config:
            raise SystemExit("--evaluate-public-reference with no model id needs "
                             "--config to read `public_reference:` from.")
        public_reference = yaml.safe_load(
            Path(args.config).read_text(encoding="utf-8")).get("public_reference")

    summary = evaluate_run(
        Path(args.run_dir), steps=args.steps, engine=args.engine,
        with_delta_ce=args.with_delta_ce, resume=args.resume, teacher=args.teacher,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
        allow_band_mismatch=args.allow_band_mismatch, device=args.device,
        dtype=args.dtype, smoke=args.smoke, n_sink_blocks=args.sink_blocks,
        tokenizer_name=args.tokenizer, bootstrap_n=args.bootstrap_n, base=args.base,
        public_reference=public_reference,
        reference_condition=args.reference_condition,
        reference_out=Path(args.reference_out) if args.reference_out else None,
        corpus_cache=corpus_cache)

    print(f"E6 evaluation {summary['run_id']} -> {summary['metrics_csv']}")
    print(f"  checkpoints={summary['n_checkpoints']} written={summary['n_written']} "
          f"skipped={summary['n_skipped']} failed={summary['n_failed']}")
    if summary["teacher_note"]:
        print(f"  note: {summary['teacher_note']}")
    if summary.get("public_reference"):
        reference = summary["public_reference"]
        print(f"  reference {reference['public_reference']} -> "
              f"{reference['metrics_csv']} "
              f"(written={reference['n_written']} failed={reference['n_failed']})")
    # A non-zero failure count is data, not a crash — but it must be visible in the exit
    # code so a batch driver does not sail past a run where every checkpoint failed.
    return 1 if summary["n_failed"] and not summary["n_written"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
