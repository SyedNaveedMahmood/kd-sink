# -*- coding: utf-8 -*-
"""fingerprint_runner.py — one seam from any model + any Corpus to a FingerprintRecord.

WP2, closing G2 (no programmatic return from the frozen harnesses) and G8 (no in-memory
module loading path). This is the single point where E6/E7 cross the boundary into the
frozen instrument: it **dispatches**, never reimplements. Attention comes from
``NNsightEngine.run_all`` / ``run_arch_trace``; every sink number is reduced by the frozen
``compute_bos_attention_metric`` over a depth-aware band from ``normalised_depth_band``;
massive coordinates and (d)/(e) swap directions come from the frozen per-arch helpers.

Scope of this slice (verify-on-compute-PC for real models):

* ``engine="nnsight"`` is fully implemented for all four archs. ``engine="manual"`` is not
  wired here — the manual path is the *parity reference*, and the nnsight path is
  metric-equivalent to it within ``METRIC_ATOL`` (that equivalence is exactly what
  ``tests/test_fingerprint_runner_bridge.py`` asserts against the frozen E1 CSV).
* Massive coordinates use the frozen EPE-0 set for GPT-2 (so the E1 bridge is exact) and
  the frozen residual-stream set otherwise; the source is recorded in ``provenance``.
  Qwen's per-sentence scope is honoured by recording ``massive_coord_scope`` from the
  spec; per-sentence recomputation is deferred and, when a coordinate set cannot be
  resolved, (i)/(j) are recorded as failures rather than fabricated.
* (d)/(e) need swap directions; for RoPE Qwen they are structurally absent (position-id
  edits, no MLP swap) and for archs without a frozen swap helper they are recorded as
  failures — never silently turned into no-ops.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

try:  # package import
    from . import inheritance_metrics as _im
except ImportError:
    import inheritance_metrics as _im

from depth_band import DEPTH_BAND_VERSION, layer_depths, normalised_depth_band
from intervention_analysis import compute_bos_attention_metric
from nnsight_engine import (
    ARCH_SPECS,
    INTERVENTION_ORDER,
    NNsightEngine,
    load_nnsight_model,
)

_REPO = Path(__file__).resolve().parents[1]

RAW_A_FLOOR = 1e-8   # (§3.4 step 5) below this, int_a is degenerate -> r_j = nan


# ═══════════════════════════════════════════════════════════════════════════════
# Model handle
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class ModelHandle:
    arch: str
    model: Any
    tokenizer: Any
    engine: str
    nn_engine: Optional[NNsightEngine]
    model_name: str
    model_revision: Optional[str]
    checkpoint_step: Optional[int]
    checkpoint_sha256: Optional[str]
    dtype: str
    device: str


def load_handle(arch, model_name_or_path, *, engine="nnsight", dtype="float32",
                revision=None, device=None, tokenizer_name=None,
                tokenizer_revision=None,
                checkpoint_step=None, checkpoint_sha256=None,
                local_files_only=False) -> ModelHandle:
    """Load a model from an HF id or local path into a :class:`ModelHandle`.

    Only ``engine="nnsight"`` is implemented in this slice (see module docstring). Dtype is
    resolved to a torch dtype for loading; the string is preserved on the handle.
    """
    if engine != "nnsight":
        raise NotImplementedError(
            "fingerprint_runner v1 (WP2) implements engine='nnsight' only. The manual "
            "harness path is the parity reference; the nnsight path is metric-equivalent "
            "to it within METRIC_ATOL. Use engine='nnsight'.")
    if arch not in ARCH_SPECS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCH_SPECS)}")
    spec = ARCH_SPECS[arch]

    torch_dtype = getattr(torch, dtype)
    if tokenizer_name is None:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, revision=revision, local_files_only=local_files_only)
    else:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_name, revision=tokenizer_revision,
            local_files_only=local_files_only)

    dev = device if device is not None else torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    lm = load_nnsight_model(spec, str(model_name_or_path), dtype=torch_dtype,
                            tokenizer=tokenizer, device=dev, revision=revision,
                            local_files_only=local_files_only)
    nn_engine = NNsightEngine(lm, spec)
    return ModelHandle(
        arch=arch, model=lm._model, tokenizer=tokenizer, engine="nnsight",
        nn_engine=nn_engine, model_name=str(model_name_or_path),
        model_revision=revision, checkpoint_step=checkpoint_step,
        checkpoint_sha256=checkpoint_sha256, dtype=dtype, device=str(dev))


def handle_from_module(arch, model, tokenizer, *, engine="nnsight",
                       model_name="in_memory", model_revision=None,
                       checkpoint_step=None, checkpoint_sha256=None,
                       dtype="float32", device=None) -> ModelHandle:
    """Wrap an already-instantiated ``nn.Module`` without a disk round-trip (G8).

    For ``engine="nnsight"`` the module is wrapped with ``nnsight.LanguageModel(model,
    tokenizer=...)`` directly (not ``load_nnsight_model``, which is HF-id shaped). Used for
    step-0 pre-update checkpoints and merged-LoRA models held in memory.
    """
    if engine != "nnsight":
        raise NotImplementedError("handle_from_module supports engine='nnsight' only.")
    if arch not in ARCH_SPECS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCH_SPECS)}")
    spec = ARCH_SPECS[arch]
    from nnsight import LanguageModel

    lm = LanguageModel(model, tokenizer=tokenizer)
    nn_engine = NNsightEngine(lm, spec)
    dev = str(device) if device is not None else str(next(model.parameters()).device)
    return ModelHandle(
        arch=arch, model=lm._model, tokenizer=tokenizer, engine="nnsight",
        nn_engine=nn_engine, model_name=model_name, model_revision=model_revision,
        checkpoint_step=checkpoint_step, checkpoint_sha256=checkpoint_sha256,
        dtype=dtype, device=dev)


# ═══════════════════════════════════════════════════════════════════════════════
# Cross-model intervention availability (design-delta D2)
# ═══════════════════════════════════════════════════════════════════════════════


def available_interventions(handle: ModelHandle) -> List[str]:
    """Interventions applicable to this model, in ``INTERVENTION_ORDER``.

    Excludes the structurally inapplicable ones, computed from the model rather than
    hard-coded (D2): ``int_b`` when the query projection has no bias (GPT-Neo), and
    ``int_e`` when there is no additive positional embedding to swap (Qwen/RoPE).
    """
    engine = handle.nn_engine
    spec = engine.spec
    has_q_bias = engine._q_bias(0) is not None
    out = []
    for key in INTERVENTION_ORDER:
        if key == "int_b" and not has_q_bias:
            continue
        if key == "int_e" and spec.wpe_path is None:
            continue
        out.append(key)
    return out


def mutual_interventions(*handles: ModelHandle) -> List[str]:
    """Intersection of available interventions across handles, preserving order."""
    if not handles:
        return []
    common = set(available_interventions(handles[0]))
    for handle in handles[1:]:
        common &= set(available_interventions(handle))
    return [key for key in INTERVENTION_ORDER if key in common]


# ═══════════════════════════════════════════════════════════════════════════════
# Frozen per-arch operands (dispatch, never reimplement)
# ═══════════════════════════════════════════════════════════════════════════════


def _massive_coords(handle: ModelHandle) -> Tuple[List[int], str]:
    model = handle.model
    if handle.arch == "gpt2":
        from intervention_analysis import identify_massive_coords
        return [int(c) for c in identify_massive_coords(model)], "epe0_gpt2"
    device = next(model.parameters()).device
    from residual_sink_analysis import identify_massive_coords as residual_coords
    return [int(c) for c in residual_coords(model, device)], "residual_stream"


def _swap_directions(handle: ModelHandle) -> Dict[str, tuple]:
    model = handle.model
    if handle.arch == "gpt2":
        from intervention_analysis import gpt2_swap_directions
        return gpt2_swap_directions(model)
    if handle.arch == "neo":
        neo_dir = _REPO / "cross_scale_and_architecture" / "neo"
        if str(neo_dir) not in sys.path:
            sys.path.insert(0, str(neo_dir))
        from intervention_analysis_neo import neo_swap_directions
        return neo_swap_directions(model)
    # Qwen: position-id edits, no MLP swap -> no swap dirs needed. OPT: no frozen helper.
    return {}


# ═══════════════════════════════════════════════════════════════════════════════
# The record
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class FingerprintRecord:
    # identity
    run_id: str
    experiment_id: str
    condition: str
    seed: int
    model_name: str
    model_revision: Optional[str]
    checkpoint_step: Optional[int]
    arch: str
    num_layers: int
    num_heads: int
    hidden_size: int
    param_count: int
    corpus_id: str
    manifest_sha256: str
    band: Tuple[int, int]
    band_depth: Tuple[float, float]
    band_version: str
    engine: str
    dtype: str
    device: str
    intervention_registry_version: str
    available_interventions: List[str]

    # topology
    baseline_sink: float
    per_layer_sink: List[float]
    per_head_sink: List[List[float]]
    depth_profile_16: List[float]
    frac_cells_above_0_2: float
    carrier_concentration: float
    top_carrier_heads: List[Tuple[float, int, float]]

    # mechanism
    fingerprint: Dict[str, float]
    raw_sink_by_intervention: Dict[str, float]

    # function
    delta_ce: Optional[Dict[str, float]]
    baseline_ce: Optional[float]

    # stability
    massive_coords: List[int]
    massive_coord_scope: str

    # bookkeeping
    n_items: int
    n_failed: int
    failures: List[Dict[str, Any]]
    wallclock_s: float
    provenance: Dict[str, Any]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)

    @staticmethod
    def from_json(text: str) -> "FingerprintRecord":
        obj = json.loads(text)
        obj["band"] = tuple(obj["band"])
        obj["band_depth"] = tuple(obj["band_depth"])
        obj["top_carrier_heads"] = [tuple(t) for t in obj["top_carrier_heads"]]
        return FingerprintRecord(**obj)


INTERVENTION_REGISTRY_VERSION = "nnsight-run_all-v1"


# ═══════════════════════════════════════════════════════════════════════════════
# Entry point
# ═══════════════════════════════════════════════════════════════════════════════


def _second_half_bos_per_layer(maps: List[torch.Tensor]) -> np.ndarray:
    """Per-layer BOS attention (2nd-half query mean to position 0), one value per layer."""
    out = []
    for attn in maps:
        seq = attn.shape[-1]
        half = seq // 2
        out.append(float(attn[:, half:, 0].mean()))
    return np.asarray(out, dtype=np.float64)


def _per_head_bos(maps: List[torch.Tensor]) -> np.ndarray:
    """[L, H] BOS attention per layer-head cell (2nd-half query mean to position 0)."""
    rows = []
    for attn in maps:
        seq = attn.shape[-1]
        half = seq // 2
        rows.append(attn[:, half:, 0].mean(dim=1).float().cpu().numpy())
    return np.asarray(rows, dtype=np.float64)


def _cache_key(handle: ModelHandle, corpus, band, interventions, with_delta_ce,
               delta_ce_per_item=False) -> dict:
    key = {
        "model_name": handle.model_name,
        "checkpoint_sha256": handle.checkpoint_sha256,
        "checkpoint_step": handle.checkpoint_step,
        "manifest_sha256": corpus.manifest_sha256,
        "band": list(band),
        "interventions": list(interventions),
        "engine": handle.engine,
        "dtype": handle.dtype,
        "with_delta_ce": bool(with_delta_ce),
        "registry_version": INTERVENTION_REGISTRY_VERSION,
    }
    # Only present when requested, so records cached before WP10 added the flag still hit.
    if delta_ce_per_item:
        key["delta_ce_per_item"] = True
    return key


def compute_fingerprint(handle: ModelHandle, corpus, *, band=None, interventions=None,
                        with_delta_ce=False, ce_corpus=None,
                        target_positions=(0, 1, 2, 3, 4), batch_size=1, progress=True,
                        cache_dir=None, run_id=None, experiment_id="E6",
                        condition="base", seed=0,
                        delta_ce_per_item=False) -> FingerprintRecord:
    """Compute a complete :class:`FingerprintRecord` for ``handle`` over ``corpus``.

    See module docstring for the dispatch contract and this slice's scope. ``band`` comes
    from :func:`depth_band.normalised_depth_band` when not given; every sink number is
    reduced by the frozen ``compute_bos_attention_metric`` over that band. Failures are
    recorded per (item, intervention), never dropped (§3.4 step 7).

    ``delta_ce_per_item`` (WP10, additive; default reproduces prior behaviour exactly)
    additionally stores the *per-item* ΔCE lists under
    ``provenance["delta_ce_per_item"]``. ``07`` P10 requires the 300-block ΔCE subset to
    carry a bootstrapped sampling error, and a mean alone cannot be bootstrapped. The
    record's schema is unchanged — the lists live in ``provenance`` — and the flag joins
    the cache key only when set, so records cached before WP10 still hit.
    """
    if handle.engine != "nnsight" or handle.nn_engine is None:
        raise NotImplementedError("compute_fingerprint requires an nnsight handle.")
    engine = handle.nn_engine
    spec = engine.spec
    num_layers = engine.num_layers
    cfg = handle.model.config
    num_heads = int(cfg.num_attention_heads)
    hidden = engine.hidden
    run_id = run_id or f"{handle.model_name}:{handle.checkpoint_step}"

    if band is None:
        start, end, band_meta = normalised_depth_band(num_layers)
        band = (start, end)
        band_depth = tuple(band_meta["depth_interval"])
    else:
        band = (int(band[0]), int(band[1]))
        depths = layer_depths(num_layers)
        band_depth = (float(depths[band[0]]),
                      float(depths[min(band[1] - 1, num_layers - 1)]))
    interventions = list(interventions) if interventions is not None \
        else available_interventions(handle)

    # --- cache lookup ---
    key = _cache_key(handle, corpus, band, interventions, with_delta_ce,
                     delta_ce_per_item)
    cache_path = None
    if cache_dir is not None:
        step = handle.checkpoint_step if handle.checkpoint_step is not None else "na"
        step_dir = Path(cache_dir) / _safe(run_id) / f"step_{step}"
        # The corpus id is part of the PATH, not only of the key. A caller that fingerprints
        # one checkpoint against several corpora -- which is exactly what
        # `evaluate_transformation.py` does, looping corpora inside one step -- otherwise
        # writes every record to the same file and each corpus destroys the previous one's
        # artefact. The `_cache_key` check below means the survivor is never *read* as the
        # wrong corpus, so no number was ever wrong; what was lost is the audit trail for
        # every corpus but the last, which `05` §7.1 requires to be there to compare against.
        # Other key fields (band, intervention set, the ΔCE flags) still share one file, and
        # that is left alone deliberately: nothing sweeps them within a run, and a collision
        # there costs a recompute rather than an artefact.
        cache_path = step_dir / _safe(corpus.corpus_id) / "fingerprint.json"
        legacy_path = step_dir / "fingerprint.json"
        for candidate in (cache_path, legacy_path):
            # The legacy layout is read (a pre-fix cache whose key still matches is a valid
            # record) but never written, so a re-run migrates itself.
            if candidate.exists():
                cached = json.loads(candidate.read_text(encoding="utf-8"))
                if cached.get("_cache_key") == key:
                    return FingerprintRecord.from_json(json.dumps(cached["record"]))

    started = time.time()

    # --- per-arch operands ---
    failures: List[Dict[str, Any]] = []
    try:
        massive_coords, massive_source = _massive_coords(handle)
    except Exception as exc:  # i/j become unavailable, recorded not fabricated
        massive_coords, massive_source = [], f"unavailable:{type(exc).__name__}"
        failures.append({"item_id": "*", "intervention": "int_i/int_j",
                         "error": type(exc).__name__, "message": str(exc)})
    try:
        swap_dirs = _swap_directions(handle)
    except Exception as exc:
        swap_dirs = {}
        failures.append({"item_id": "*", "intervention": "int_d/int_e",
                         "error": type(exc).__name__, "message": str(exc)})

    device = next(handle.model.parameters()).device
    swap_dirs = {k: tuple(t.to(device) for t in v) for k, v in swap_dirs.items()}

    # Interventions we can actually run given resolved operands.
    runnable = []
    for key_j in interventions:
        if key_j in ("int_i", "int_j") and not massive_coords:
            failures.append({"item_id": "*", "intervention": key_j,
                             "error": "NoMassiveCoords",
                             "message": "massive coordinates could not be resolved"})
            continue
        if key_j in ("int_d", "int_e") and spec.positional != "rope_ids" \
                and key_j not in swap_dirs:
            failures.append({"item_id": "*", "intervention": key_j,
                             "error": "NoSwapDirections",
                             "message": "swap directions unavailable for this arch"})
            continue
        runnable.append(key_j)

    # --- accumulate raw sink metrics over the corpus ---
    raw_sums = {k: 0.0 for k in runnable}
    per_layer_sum = np.zeros(num_layers, dtype=np.float64)
    per_head_sum = np.zeros((num_layers, num_heads), dtype=np.float64)
    n_ok = 0
    ls, le = band

    for item in corpus.items:
        inputs = _item_inputs(item, device)
        try:
            # `runnable` (plus the int_a baseline) rather than the whole registry: an
            # intervention whose operands could not be resolved was already recorded as a
            # failure above, and asking the engine to run it anyway would raise and take
            # the entire item -- and therefore the entire corpus -- down with it.
            results = engine.run_all(inputs, massive_coords=massive_coords or None,
                                     swap_dirs=swap_dirs, verbose=False,
                                     keys=sorted(set(runnable) | {"int_a"}))
        except Exception as exc:
            failures.append({"item_id": item.item_id, "intervention": "*",
                             "error": type(exc).__name__, "message": str(exc)})
            continue
        # int_a baseline topology
        maps_a = results["int_a"]
        per_layer_sum += _second_half_bos_per_layer(maps_a)
        per_head_sum += _per_head_bos(maps_a)
        for key_j in runnable:
            maps = results.get(key_j)
            if maps is None:
                failures.append({"item_id": item.item_id, "intervention": key_j,
                                 "error": "Missing", "message": "not returned by run_all"})
                continue
            raw_sums[key_j] += compute_bos_attention_metric(
                maps, num_layers, "mid", layer_start=ls, layer_end=le)
        n_ok += 1

    if n_ok == 0:
        raise RuntimeError(
            "Every corpus item failed the fingerprint forward; see the failures list. "
            f"First failure: {failures[0] if failures else 'n/a'}")

    raw_sink = {k: raw_sums[k] / n_ok for k in runnable}
    per_layer_sink = (per_layer_sum / n_ok).tolist()
    per_head_sink = (per_head_sum / n_ok)

    raw_a = raw_sink.get("int_a", float("nan"))
    fingerprint: Dict[str, float] = {}
    for key_j in runnable:
        if not np.isfinite(raw_a) or abs(raw_a) < RAW_A_FLOOR:
            fingerprint[key_j] = float("nan")
            failures.append({"item_id": "*", "intervention": key_j,
                             "error": "DegenerateBaseline",
                             "message": f"raw int_a={raw_a} < {RAW_A_FLOOR}; r_j=nan"})
        else:
            fingerprint[key_j] = raw_sink[key_j] / raw_a

    # --- topology derived from int_a ---
    baseline_sink = float(np.mean(per_layer_sink[ls:le]))
    cells = per_head_sink  # [L, H]
    depth_profile_16 = _im.interp_depth_profile(np.asarray(per_layer_sink), 16)
    if isinstance(depth_profile_16, tuple):
        depth_profile_16 = depth_profile_16[0]
    frac_above = float((cells > 0.2).mean())
    concentration = float(_im.carrier_concentration(cells))
    top_heads = [tuple(t) for t in _im.top_carriers(cells, k=5)]

    # --- optional functional cost ---
    delta_ce = None
    baseline_ce = None
    per_item_delta_ce: Optional[Dict[str, List[float]]] = None
    per_item_ce_ids: List[str] = []
    if with_delta_ce:
        delta_ce, baseline_ce, ce_failures, per_item = _delta_ce(
            engine, ce_corpus or corpus, runnable, massive_coords, swap_dirs, band,
            device, collect_per_item=delta_ce_per_item)
        failures.extend(ce_failures)
        if delta_ce_per_item:
            per_item_delta_ce = per_item["delta_ce"]
            per_item_ce_ids = per_item["item_ids"]

    record = FingerprintRecord(
        run_id=run_id, experiment_id=experiment_id, condition=condition, seed=seed,
        model_name=handle.model_name, model_revision=handle.model_revision,
        checkpoint_step=handle.checkpoint_step, arch=handle.arch,
        num_layers=num_layers, num_heads=num_heads, hidden_size=hidden,
        param_count=int(sum(p.numel() for p in handle.model.parameters())),
        corpus_id=corpus.corpus_id, manifest_sha256=corpus.manifest_sha256,
        band=(int(ls), int(le)), band_depth=(float(band_depth[0]), float(band_depth[1])),
        band_version=DEPTH_BAND_VERSION, engine="nnsight", dtype=handle.dtype,
        device=str(device),
        intervention_registry_version=INTERVENTION_REGISTRY_VERSION,
        available_interventions=list(interventions),
        baseline_sink=baseline_sink, per_layer_sink=per_layer_sink,
        per_head_sink=per_head_sink.tolist(),
        depth_profile_16=list(map(float, np.asarray(depth_profile_16).tolist())),
        frac_cells_above_0_2=frac_above, carrier_concentration=concentration,
        top_carrier_heads=top_heads, fingerprint=fingerprint,
        raw_sink_by_intervention=raw_sink, delta_ce=delta_ce, baseline_ce=baseline_ce,
        massive_coords=list(massive_coords), massive_coord_scope=spec.massive_scope,
        n_items=n_ok, n_failed=len(failures), failures=failures,
        wallclock_s=time.time() - started,
        provenance={"massive_coord_source": massive_source,
                    "target_positions": list(target_positions),
                    "with_delta_ce": bool(with_delta_ce),
                    "n_corpus_items": len(corpus.items),
                    "swap_dir_keys": sorted(swap_dirs.keys()),
                    "delta_ce_per_item": per_item_delta_ce,
                    "delta_ce_item_ids": per_item_ce_ids})

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(
            {"_cache_key": key, "record": json.loads(record.to_json())},
            ensure_ascii=False, indent=2), encoding="utf-8")
    return record


def _delta_ce(engine, ce_corpus, interventions, massive_coords, swap_dirs, band, device,
              *, collect_per_item: bool = False):
    """Mean ΔCE per intervention vs int_a over ce_corpus (run_arch_trace CE path).

    Returns ``(delta_ce, baseline_ce, failures, per_item)``. ``per_item`` is populated only
    when ``collect_per_item`` — the caller bootstraps it (``07`` P10). Items that fail are
    absent from the per-item lists *and* from the mean, and appear in ``failures``, so a
    bootstrap can never resample a value that was never measured.
    """
    from nnsight_engine import draw_random_wk_columns

    # (j) draws random Wk columns exactly as run_all does (same seed/stream).
    k = len(massive_coords) if massive_coords else 0
    random_columns = (draw_random_wk_columns(engine.num_layers, k, engine.hidden)
                      if k else None)

    ce_sums = {k_: 0.0 for k_ in interventions}
    per_item: Dict[str, List[float]] = {k_: [] for k_ in interventions if k_ != "int_a"}
    item_ids: List[str] = []
    base_sum = 0.0
    n = 0
    failures: List[Dict[str, Any]] = []
    for item in ce_corpus.items:
        inputs = _item_inputs(item, device)
        if inputs["input_ids"].shape[-1] < 2:
            continue
        try:
            base = engine.run_arch_trace(
                engine.plans["int_a"], inputs, capture_attention="none",
                capture_token_ce=True, band=band)
            base_ce = float(base["token_ce"].mean())
        except Exception as exc:
            failures.append({"item_id": item.item_id, "intervention": "int_a(ce)",
                             "error": type(exc).__name__, "message": str(exc)})
            continue
        base_sum += base_ce
        for key_j in interventions:
            if key_j == "int_a":
                continue
            try:
                out = engine.run_arch_trace(
                    engine.plans[key_j], inputs, capture_attention="none",
                    massive_coords=massive_coords or None,
                    random_columns=random_columns,
                    swap_dirs=swap_dirs.get(key_j), capture_token_ce=True, band=band)
                ce_sums.setdefault(key_j, 0.0)
                item_delta = float(out["token_ce"].mean()) - base_ce
                ce_sums[key_j] += item_delta
                if collect_per_item:
                    per_item.setdefault(key_j, []).append(item_delta)
            except Exception as exc:
                failures.append({"item_id": item.item_id, "intervention": f"{key_j}(ce)",
                                 "error": type(exc).__name__, "message": str(exc)})
        if collect_per_item:
            item_ids.append(item.item_id)
        n += 1
    empty = {"delta_ce": {}, "item_ids": []}
    if n == 0:
        return {}, None, failures, empty
    delta_ce = {k: ce_sums[k] / n for k in interventions if k != "int_a"}
    detail = ({"delta_ce": per_item, "item_ids": item_ids} if collect_per_item else empty)
    return delta_ce, base_sum / n, failures, detail


def _item_inputs(item, device) -> dict:
    ids = torch.tensor([list(item.input_ids)], dtype=torch.long, device=device)
    return {"input_ids": ids, "attention_mask": torch.ones_like(ids)}


def _safe(text: str) -> str:
    raw = str(text)
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in raw)
    if len(safe) <= 96:
        return safe
    # Local smoke-model ids can be absolute Windows paths. Keep cache components bounded
    # without creating collisions; short historical names retain their exact old path.
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]
    return f"{safe[:79]}_{digest}"
