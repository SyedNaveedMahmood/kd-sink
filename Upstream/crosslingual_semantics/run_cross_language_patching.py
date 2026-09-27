# -*- coding: utf-8 -*-
"""run_cross_language_patching.py — cross-language activation patching for E7 (WP11).

``04_MODULE_SPEC_e7_crosslingual.md`` §5. Captures English source activations once, then for
every target-language XNLI example runs a clean forward and one patched forward per
(source condition × norm condition), writing ``patching_per_example.csv`` exactly per
``05_SCHEMAS_AND_CONTRACTS.md`` §3.

    # 1. Phase 1 smoke study (04 §5.5), before anything larger
    python crosslingual_semantics/run_cross_language_patching.py \\
      --config crosslingual_semantics/configs/e7_qwen05_base.yaml --stage smoke

    # 2. screen the four layers on the 200 DEV examples, then
    python crosslingual_semantics/run_cross_language_patching.py ... --stage screen

    # 3. evaluate the selected five-layer window on the disjoint 400 TEST examples
    python crosslingual_semantics/run_cross_language_patching.py ... --stage window

The layer protocol is the entire defence against sweep overfitting
------------------------------------------------------------------
``--stage screen`` may only ever touch the **dev** partition, and ``--stage window`` only
the **test** partition; both are enforced, not documented. ``layer_selection.json`` records
the dev effect for *every* screened layer, the selection rule, the selected layer and the
resulting clamped window, so the dev/test separation is auditable from the artefacts alone.
``--stage window`` refuses to start without that file.

The identity control aborts, it does not warn
---------------------------------------------
``04`` §5.4: "if any identity row shows a non-zero margin change beyond 1e-5, the run is
invalid and must abort, not merely warn." So the identity condition is scheduled **first**
at every site, and a violation marks the whole site invalid, stops it before the remaining
norm conditions are paid for, records it in ``invalid_units.json``, and makes the process
exit non-zero. ``aggregate_crosslingual.py`` excludes invalid units from every contrast.

Two schema decisions, recorded rather than silent
-------------------------------------------------
1. **Unit of work is (model, target language, object, layer)**, not ``05`` §3's
   (…, norm condition). All four norm conditions at one site share a single baseline forward
   inside :func:`cross_example_patching.run_patched`; splitting them would recompute that
   baseline four times. ``norm_condition`` is still a column on every row and part of the
   ledger key, so adding or removing one still invalidates and recomputes the site. Same
   reasoning as WP10's ``(run_id, step, corpus_id)``.
2. **``source_condition="none"`` for the source-independent norms.** ``random`` draws a
   Gaussian matched to the *target's* norm and ``identity`` restores the target's own
   vector — neither reads a source at all. Crossing them with the four source conditions
   would write four byte-identical rows and pay four times for them. They are written once
   per site with ``source_condition="none"``: a new *value*, never a renamed column
   (``05`` §5), and the same discipline WP7 used for ``Kmid_prerope``.

``--smoke`` runs the whole path on a random tiny Qwen2 and a synthetic manifest, offline.
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common", _REPO / "crosslingual_semantics"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import corpus_providers as cp  # noqa: E402
import cross_example_patching as cep  # noqa: E402
import fingerprint_runner as fr  # noqa: E402
import paired_manifests as pm  # noqa: E402
import provenance as prov  # noqa: E402
from depth_band import normalised_depth_band  # noqa: E402
from extract_sink_representations import arch_of, load_config  # noqa: E402

RUNNER_VERSION = "run_cross_language_patching_v1"

#: Norm conditions whose value does not come from the source example (see decision 2).
SOURCE_INDEPENDENT_NORMS: Tuple[str, ...] = ("random", "identity")

#: Written with ``source_condition`` for those norms.
NO_SOURCE = "none"

STAGES: Tuple[str, ...] = ("smoke", "screen", "window")

#: dev is for *selecting*; test is for *reporting*. Enforced in :func:`resolve_stage`.
STAGE_PARTITION: Dict[str, str] = {"smoke": "dev", "screen": "dev", "window": "test"}

LEDGER_FILENAME = "patch_units.jsonl"
ROWS_FILENAME = "patching_per_example.csv"
INVALID_FILENAME = "invalid_units.json"
SELECTION_FILENAME = "layer_selection.json"

#: ``05`` §5: after 50 patch examples.
RUNTIME_ESTIMATE_AFTER = 50

#: ``05`` §3, verbatim, plus a clearly-marked WP11 tail (adding is permitted, renaming is
#: not — `05` §5).
ROW_COLUMNS: Tuple[str, ...] = (
    "model", "model_revision", "dtype", "engine",
    "target_language", "semantic_id", "partition", "gold_label",
    "source_condition", "source_semantic_id", "source_label",
    "patch_object", "patch_layer", "patch_position", "source_patch_position",
    "norm_condition",
    "baseline_gold_logprob", "patched_gold_logprob",
    "baseline_margin", "patched_margin",
    "baseline_margin_unnorm", "patched_margin_unnorm",
    "baseline_prediction", "patched_prediction",
    "baseline_sink", "patched_sink",
    "baseline_attn_pos1_4", "patched_attn_pos1_4",
    "jsd_output", "carrier_sink_delta",
    "source_seq_len", "target_seq_len",
    "source_norm", "target_norm", "patched_norm",
    "manifest_sha256", "patch_registry_version", "git_sha",
    "status", "warning",
    # --- added by WP11 beyond `05` §3 ---
    "model_tag", "model_variant", "run_unit_id", "site_id", "unit_status", "stage",
    "margin_delta", "gold_label_name", "layer_band", "seed", "runner_version",
    "measured_utc",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Stage / partition / layer resolution
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class StagePlan:
    """Everything the layer protocol decides, resolved once and recorded on every row."""

    stage: str
    partition: str
    layers: List[int]
    objects: List[str]
    norm_conditions: List[str]
    source_conditions: List[str]
    max_examples: Optional[int]
    selection: Optional[Dict[str, Any]] = None
    note: str = ""


def clamp_window(selected: int, width: int, num_layers: int) -> List[int]:
    """The contiguous ``width``-layer window centred on ``selected``, clamped (``04`` §5.3).

    Clamping shifts the window rather than truncating it, so a selection at layer 0 still
    reports ``width`` layers. When the model is shallower than the window, every layer is
    returned.
    """
    if width <= 0:
        raise ValueError("window width must be positive")
    if width >= num_layers:
        return list(range(num_layers))
    start = selected - width // 2
    start = max(0, min(start, num_layers - width))
    return list(range(start, start + width))


def resolve_stage(config, num_layers: int, *, stage: str,
                  selection: Optional[Dict[str, Any]] = None) -> StagePlan:
    """Turn (config, stage) into the concrete layers/objects/partition for this run.

    ``screen`` and ``smoke`` read ``patching.screen_layers`` and run on **dev**; ``window``
    reads the window out of ``layer_selection.json`` and runs on **test**. The partition is
    not a flag: allowing it to be overridden is exactly how a test-set number ends up being
    selected on.
    """
    if stage not in STAGES:
        raise ValueError(f"unknown stage {stage!r}; choose from {STAGES}")
    patching = config.raw.get("patching") or {}
    norms = list(patching.get("norm_conditions") or cep.NORM_CONDITIONS)
    sources = list(patching.get("source_conditions") or pm.PATCH_CONTROL_CONDITIONS)

    if stage == "window":
        if not selection:
            raise ValueError(
                f"--stage window needs {SELECTION_FILENAME}; run --stage screen first. "
                "Selecting the layer on the same examples it is reported on is the one "
                "thing the layer protocol exists to prevent (04 §5.3).")
        layers = [int(x) for x in selection["window"]]
        objects = list(patching.get("objects") or [])
        note = (f"window of {len(layers)} layers centred on layer "
                f"{selection['selected_layer']}, selected on the dev partition")
        max_examples = None
    else:
        layers = [int(x) for x in (patching.get("screen_layers") or [])]
        if stage == "smoke":
            smoke = patching.get("smoke") or {}
            objects = list(smoke.get("objects") or ["K0_prerope", "V0"])
            max_examples = int(smoke.get("n_examples", 50))
            note = f"04 §5.5 Phase 1 smoke study, {max_examples} dev examples"
        else:
            objects = list(patching.get("objects") or [])
            max_examples = None
            note = "04 §5.3 layer screening on the dev partition"

    if not layers:
        raise ValueError(f"config declares no layers for stage {stage!r}")
    out_of_range = [x for x in layers if not 0 <= x < num_layers]
    if out_of_range:
        raise ValueError(
            f"layers {out_of_range} are out of range for a {num_layers}-layer model. "
            "The screening schedule is model-depth-specific (04 §5.3: 5/11/17/22 for 24 "
            "layers, 6/13/20/26 for 28); fix `patching.screen_layers` in the config.")
    if not objects:
        raise ValueError(f"config declares no patch objects for stage {stage!r}")
    if "identity" not in norms:
        raise ValueError(
            "`identity` is missing from patching.norm_conditions. It is not one option "
            "among four — it is the continuous correctness monitor that makes every other "
            "number trustworthy (04 §5.4). Running without it is refused.")

    return StagePlan(stage=stage, partition=STAGE_PARTITION[stage], layers=layers,
                     objects=objects, norm_conditions=norms, source_conditions=sources,
                     max_examples=max_examples, selection=selection, note=note)


# ═══════════════════════════════════════════════════════════════════════════════
# Ledger
# ═══════════════════════════════════════════════════════════════════════════════


def unit_key(model_tag: str, lang: str, obj: str, layer: int) -> str:
    return f"{model_tag}|{lang}|{obj}|L{layer}"


def read_ledger(root: Path) -> Dict[str, Dict[str, Any]]:
    path = Path(root) / LEDGER_FILENAME
    ledger: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return ledger
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict) and "unit" in row:
            ledger[str(row["unit"])] = row
    return ledger


def append_ledger(root: Path, row: Dict[str, Any]) -> None:
    path = Path(root) / LEDGER_FILENAME
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def unit_is_complete(ledger: Dict[str, Dict[str, Any]], key: str,
                     expected: Dict[str, Any]) -> bool:
    row = ledger.get(key)
    if row is None or row.get("status") != "ok":
        return False
    return all(row.get(name) == value for name, value in expected.items())


# ═══════════════════════════════════════════════════════════════════════════════
# Controls and source capture
# ═══════════════════════════════════════════════════════════════════════════════


def read_patch_controls(path):
    """The ``assign_patch_controls`` table written by ``prepare_xnli_manifest.py``."""
    import pandas as pd

    frame = pd.read_csv(path, encoding="utf-8")
    required = {"semantic_id", "target_language", "control_condition",
                "source_semantic_id", "source_label", "gold_label", "status"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns {missing}")
    return frame


def controls_for(controls, lang: str, semantic_ids: Sequence[str],
                 source_conditions: Sequence[str]) -> Dict[str, Dict[str, Any]]:
    """``{semantic_id: {condition: row}}`` for one target language, ``ok`` rows only.

    ``status == "unassignable"`` rows are kept out of the patch schedule but their absence
    is reported by the caller — a semantic id that could not be given a control is a
    recorded gap, not a silent one.
    """
    wanted = set(semantic_ids)
    subset = controls[(controls["target_language"] == lang)
                      & (controls["semantic_id"].isin(wanted))
                      & (controls["control_condition"].isin(list(source_conditions)))]
    out: Dict[str, Dict[str, Any]] = {}
    for row in subset.to_dict("records"):
        if str(row.get("status", "ok")) != "ok":
            continue
        out.setdefault(str(row["semantic_id"]), {})[str(row["control_condition"])] = row
    return out


def capture_english_sources(handle, manifest: pm.ParallelManifest, sites, *,
                            semantic_ids: Sequence[str], source_lang: str = "en",
                            cache_path: Optional[Path] = None,
                            progress: bool = True) -> Dict[str, Any]:
    """Capture every English source once, cached to disk keyed by the model fingerprint.

    ``04`` §5.2: "``capture_sources`` for all needed English source examples at the required
    sites, **once**, cached to disk keyed by model fingerprint." A cache whose fingerprint or
    site set does not match is discarded and recaptured rather than reused — cross-model
    patching must fail loudly (WP7 invariant 8), and a partial site set would fail later at
    a much less legible place.
    """
    fingerprint = cep.model_fingerprint(handle)
    site_keys = sorted(site.key() for site in sites)
    ids = sorted(set(semantic_ids))

    if cache_path is not None and Path(cache_path).exists():
        try:
            blob = torch.load(cache_path, map_location="cpu", weights_only=False)
            if (blob.get("model_fingerprint") == fingerprint
                    and blob.get("site_keys") == site_keys
                    and set(blob.get("caches", {})) >= {f"{sid}:{source_lang}"
                                                        for sid in ids}):
                if progress:
                    print(f"  [sources] reusing {len(blob['caches'])} cached captures")
                return blob["caches"]
        except Exception:
            pass  # a corrupt cache is recaptured, never trusted

    corpus = cp.xnli_prompt_corpus(handle.tokenizer, source_lang, ids,
                                   split=manifest.split, manifest=manifest)
    if progress:
        print(f"  [sources] capturing {len(corpus.items)} {source_lang} examples "
              f"at {len(site_keys)} sites")
    caches = cep.capture_sources(handle, corpus, sites)
    if cache_path is not None:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_fingerprint": fingerprint, "site_keys": site_keys,
                    "caches": caches}, cache_path)
    return caches


# ═══════════════════════════════════════════════════════════════════════════════
# Rows
# ═══════════════════════════════════════════════════════════════════════════════


def _finite(value):
    """NaN/inf become an empty CSV cell, so a sentinel is never read as a measurement."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return value
    return None if not np.isfinite(number) else number


def build_specs(plan: StagePlan, site: cep.PatchSite,
                controls: Dict[str, Any], seed: int
                ) -> List[Tuple[cep.PatchSpec, str, Optional[Dict[str, Any]]]]:
    """The patch schedule for one (target example, site).

    Ordered so **identity comes first** — a violated identity control must abort the site
    before the run pays for the remaining conditions (``04`` §5.4).
    """
    schedule: List[Tuple[cep.PatchSpec, str, Optional[Dict[str, Any]]]] = []
    for norm in plan.norm_conditions:
        if norm not in SOURCE_INDEPENDENT_NORMS:
            continue
        schedule.append((cep.PatchSpec(site=site, norm_condition=norm,
                                       source_item_id=None, seed=seed),
                         NO_SOURCE, None))
    schedule.sort(key=lambda entry: entry[0].norm_condition != "identity")

    for norm in plan.norm_conditions:
        if norm in SOURCE_INDEPENDENT_NORMS:
            continue
        for condition in plan.source_conditions:
            control = controls.get(condition)
            if control is None:
                continue
            source_item_id = f"{control['source_semantic_id']}:en"
            schedule.append((cep.PatchSpec(site=site, norm_condition=norm,
                                           source_item_id=source_item_id, seed=seed),
                             condition, control))
    return schedule


def _row(identity: Dict[str, Any], plan: StagePlan, *, lang: str, semantic_id: str,
         gold_label: Any, gold_label_name: str, site: cep.PatchSite,
         patch_row: Dict[str, Any], source_condition: str,
         control: Optional[Dict[str, Any]], unit_id: str, site_id: str,
         unit_status: str, band: Tuple[int, int], seed: int) -> Dict[str, Any]:
    """One ``patching_per_example.csv`` row. Failures are rows too (``05`` §3)."""
    margin_delta = patch_row.get("margin_delta")
    return {
        **identity,
        "target_language": lang,
        "semantic_id": semantic_id,
        "partition": plan.partition,
        "gold_label": gold_label,
        "gold_label_name": gold_label_name,
        "source_condition": source_condition,
        "source_semantic_id": (control or {}).get("source_semantic_id"),
        "source_label": (control or {}).get("source_label"),
        "patch_object": site.object,
        "patch_layer": int(site.layer),
        "patch_position": patch_row.get("patch_position"),
        "source_patch_position": patch_row.get("source_patch_position"),
        "norm_condition": patch_row.get("norm_condition"),
        "baseline_gold_logprob": _finite(patch_row.get("baseline_gold_logprob")),
        "patched_gold_logprob": _finite(patch_row.get("patched_gold_logprob")),
        "baseline_margin": _finite(patch_row.get("baseline_margin")),
        "patched_margin": _finite(patch_row.get("patched_margin")),
        "baseline_margin_unnorm": _finite(patch_row.get("baseline_margin_unnorm")),
        "patched_margin_unnorm": _finite(patch_row.get("patched_margin_unnorm")),
        "baseline_prediction": patch_row.get("baseline_prediction"),
        "patched_prediction": patch_row.get("patched_prediction"),
        "baseline_sink": _finite(patch_row.get("baseline_sink")),
        "patched_sink": _finite(patch_row.get("patched_sink")),
        "baseline_attn_pos1_4": _finite(patch_row.get("baseline_attn_pos1_4")),
        "patched_attn_pos1_4": _finite(patch_row.get("patched_attn_pos1_4")),
        "jsd_output": _finite(patch_row.get("jsd_output")),
        "carrier_sink_delta": _finite(patch_row.get("carrier_sink_delta")),
        "source_seq_len": patch_row.get("source_seq_len"),
        "target_seq_len": patch_row.get("target_seq_len"),
        "source_norm": _finite(patch_row.get("source_norm")),
        "target_norm": _finite(patch_row.get("target_norm")),
        "patched_norm": _finite(patch_row.get("patched_norm")),
        "patch_registry_version": patch_row.get("patch_registry_version",
                                                cep.PATCH_REGISTRY_VERSION),
        "status": patch_row.get("status", "failed"),
        "warning": patch_row.get("warning", ""),
        "run_unit_id": unit_id,
        "site_id": site_id,
        "unit_status": unit_status,
        "stage": plan.stage,
        "margin_delta": _finite(margin_delta),
        "layer_band": f"[{band[0]},{band[1]})",
        "seed": int(seed),
        "runner_version": RUNNER_VERSION,
        "measured_utc": prov.utc_now(),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# The run
# ═══════════════════════════════════════════════════════════════════════════════


def run(config, manifest: pm.ParallelManifest, controls, out_dir: Path, *,
        stage: str = "screen", handle=None, langs: Optional[Sequence[str]] = None,
        resume: bool = True, device: Optional[str] = None,
        selection: Optional[Dict[str, Any]] = None, max_examples: Optional[int] = None,
        source_cache_path: Optional[Path] = None, progress: bool = True
        ) -> Dict[str, Any]:
    """Run one stage. Returns the summary; the caller decides the exit code."""
    import pandas as pd

    started = time.time()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    owns_handle = handle is None
    if owns_handle:
        handle = fr.load_handle(arch_of(config.model), config.model,
                                engine=config.engine, dtype=config.dtype,
                                revision=config.model_revision, device=device,
                                tokenizer_name=config.raw.get("tokenizer"))
    try:
        num_layers = handle.nn_engine.num_layers
        band_start, band_end, _meta = normalised_depth_band(num_layers,
                                                            frac=config.band_frac)
        band = (int(band_start), int(band_end))
        plan = resolve_stage(config, num_layers, stage=stage, selection=selection)
        if max_examples is not None:
            plan.max_examples = int(max_examples)

        partition_ids = list(manifest.partitions.get(plan.partition, ()))
        if not partition_ids:
            raise ValueError(
                f"manifest has no {plan.partition!r} partition. `04` §5.3 requires a "
                "200 dev / 400 test grouped split; run prepare_xnli_manifest.py, which "
                "writes it.")
        # The one refusal that makes the protocol real: dev and test must not overlap.
        other = "test" if plan.partition == "dev" else "dev"
        overlap = set(partition_ids) & set(manifest.partitions.get(other, ()))
        if overlap:
            raise ValueError(
                f"the {plan.partition!r} and {other!r} partitions share {len(overlap)} "
                "semantic ids; selecting a layer on ids it is then reported on is exactly "
                "what 04 §5.3 forbids.")
        if plan.max_examples is not None:
            partition_ids = partition_ids[:plan.max_examples]

        target_langs = [lang for lang in (langs or (config.raw.get("xnli") or {})
                                          .get("langs") or manifest.languages)
                        if lang != "en"]
        unknown = [lang for lang in target_langs if lang not in manifest.languages]
        if unknown:
            raise ValueError(f"languages {unknown} are not in the manifest")

        identity = {
            "model": config.model,
            "model_revision": config.model_revision,
            "dtype": config.dtype,
            "engine": config.engine,
            "model_tag": config.tag,
            "model_variant": config.variant,
            "manifest_sha256": manifest.sha256,
            "git_sha": prov.git_sha(),
        }
        fingerprint = cep.model_fingerprint(handle)
        expected = {
            "manifest_sha256": manifest.sha256,
            "model_fingerprint": fingerprint,
            "partition": plan.partition,
            "stage": plan.stage,
            "norm_conditions": list(plan.norm_conditions),
            "source_conditions": list(plan.source_conditions),
            "n_examples": len(partition_ids),
            "patch_registry_version": cep.PATCH_REGISTRY_VERSION,
            "runner_version": RUNNER_VERSION,
        }

        sites = [cep.PatchSite(obj, layer) for obj in plan.objects
                 for layer in plan.layers]
        source_ids = sorted({str(row["source_semantic_id"]) for row in
                             controls[controls["status"] == "ok"].to_dict("records")
                             if str(row["semantic_id"]) in set(partition_ids)})
        caches = capture_english_sources(
            handle, manifest, sites, semantic_ids=source_ids,
            cache_path=source_cache_path, progress=progress)

        ledger = read_ledger(out_dir) if resume else {}
        rows: List[Dict[str, Any]] = []
        invalid: List[Dict[str, Any]] = []
        n_units = n_skipped = n_failed_rows = 0
        n_examples_run = 0
        runtime_estimate: Optional[Dict[str, Any]] = None

        for lang in target_langs:
            lang_controls = controls_for(controls, lang, partition_ids,
                                         plan.source_conditions)
            unassignable = [sid for sid in partition_ids if sid not in lang_controls]
            corpus = cp.xnli_prompt_corpus(handle.tokenizer, lang, partition_ids,
                                           split=manifest.split, manifest=manifest)
            item_by_id = {str(item.meta.get("semantic_id", item.item_id)): item
                          for item in corpus.items}

            for obj in plan.objects:
                for layer in plan.layers:
                    key = unit_key(config.tag, lang, obj, layer)
                    if resume and unit_is_complete(ledger, key, expected):
                        n_skipped += 1
                        continue

                    unit_rows, unit_status, n_failed, n_ran, estimate = _run_unit(
                        handle, plan, identity, caches=caches, corpus_ids=partition_ids,
                        item_by_id=item_by_id, lang_controls=lang_controls, lang=lang,
                        obj=obj, layer=layer, unit_id=key, band=band, seed=config.seed,
                        manifest=manifest, progress=progress,
                        need_runtime=runtime_estimate is None
                        and n_examples_run < RUNTIME_ESTIMATE_AFTER,
                        already_run=n_examples_run)
                    rows.extend(unit_rows)
                    n_units += 1
                    n_failed_rows += n_failed
                    n_examples_run += n_ran
                    if estimate is not None and runtime_estimate is None:
                        runtime_estimate = estimate

                    if unit_status == "ok":
                        append_ledger(out_dir, {"unit": key, "status": "ok",
                                                "measured_utc": prov.utc_now(),
                                                **expected})
                    else:
                        append_ledger(out_dir, {"unit": key, "status": unit_status,
                                                "measured_utc": prov.utc_now()})
                        invalid.append({"unit": key, "target_language": lang,
                                        "patch_object": obj, "patch_layer": int(layer),
                                        "stage": plan.stage,
                                        "unit_status": unit_status,
                                        "detected_utc": prov.utc_now()})
                        if progress:
                            print(f"  [patch] {key}: INVALID ({unit_status}) — aborted")

            if unassignable and progress:
                print(f"  [patch] {lang}: {len(unassignable)} semantic ids had no "
                      "assignable control and were not patched")

        frame = pd.DataFrame(rows).reindex(columns=list(ROW_COLUMNS))
        rows_path = out_dir / ROWS_FILENAME
        if rows_path.exists() and resume:
            previous = pd.read_csv(rows_path, encoding="utf-8")
            if not frame.empty:
                previous = previous[~previous["run_unit_id"].isin(
                    set(frame["run_unit_id"]))]
            frame = pd.concat([previous, frame], ignore_index=True)
        frame.to_csv(rows_path, index=False, encoding="utf-8")

        if invalid:
            prov.write_json(out_dir / INVALID_FILENAME,
                            {"invalid_units": invalid,
                             "identity_tolerance": cep.IDENTITY_TOLERANCE,
                             "note": "04 §5.4 — an identity patch that moves the margin "
                                     "beyond the tolerance invalidates its run unit; the "
                                     "aggregator excludes these rows.",
                             **prov.provenance_block()})
        if runtime_estimate is not None:
            prov.write_json(out_dir / "runtime_estimate.json",
                            {**runtime_estimate, **prov.provenance_block()})

        summary = {
            **identity,
            "stage": plan.stage,
            "partition": plan.partition,
            "note": plan.note,
            "layers": plan.layers,
            "objects": plan.objects,
            "norm_conditions": plan.norm_conditions,
            "source_conditions": plan.source_conditions,
            "target_languages": target_langs,
            "n_examples": len(partition_ids),
            "n_units": n_units,
            "n_skipped": n_skipped,
            "n_rows": int(len(frame)),
            "n_failed_rows": n_failed_rows,
            "n_invalid_units": len(invalid),
            "layer_band": list(band),
            "num_layers": int(num_layers),
            "model_fingerprint": fingerprint,
            "rows_csv": str(rows_path),
            "runner_version": RUNNER_VERSION,
            "wallclock_s": round(time.time() - started, 3),
        }
        prov.write_json(out_dir / f"run_summary_{plan.stage}.json",
                        {**summary, **prov.provenance_block()})
        return summary
    finally:
        if owns_handle:
            del handle
            gc.collect()


def _run_unit(handle, plan: StagePlan, identity: Dict[str, Any], *, caches,
              corpus_ids: Sequence[str], item_by_id, lang_controls, lang: str,
              obj: str, layer: int, unit_id: str, band, seed: int,
              manifest: pm.ParallelManifest, progress: bool, need_runtime: bool,
              already_run: int):
    """One (model, language, object, layer) unit. Aborts on an identity violation."""
    site = cep.PatchSite(obj, layer)
    site_id = f"{identity['model_tag']}|{lang}|{obj}|L{layer}"
    unit_rows: List[Dict[str, Any]] = []
    unit_status = "ok"
    n_failed = 0
    n_ran = 0
    started = time.time()
    peak_vram = 0
    estimate: Optional[Dict[str, Any]] = None

    for semantic_id in corpus_ids:
        controls = lang_controls.get(semantic_id)
        item = item_by_id.get(semantic_id)
        if controls is None or item is None:
            continue

        candidates = item.meta.get("label_candidate_ids")
        gold_label = item.meta.get("gold_label")
        if not candidates or gold_label is None:
            unit_rows.append(_row(
                identity, plan, lang=lang, semantic_id=semantic_id,
                gold_label=gold_label,
                gold_label_name=str(item.meta.get("gold_label_name", "")),
                site=site,
                patch_row={"status": "failed",
                           "warning": "corpus item carries no label candidates",
                           "norm_condition": "", "target_seq_len": item.n_tokens},
                source_condition=NO_SOURCE, control=None, unit_id=unit_id,
                site_id=site_id, unit_status=unit_status, band=band, seed=seed))
            n_failed += 1
            continue

        schedule = build_specs(plan, site, controls, seed)
        specs = [entry[0] for entry in schedule]
        try:
            result = cep.run_patched(handle, item, specs, caches,
                                     score_candidates=candidates,
                                     gold_index=int(gold_label), capture_sink=True,
                                     band=band)
            patch_rows = result["patches"]
        except Exception as exc:
            # A whole-example failure is still data: one row per scheduled spec.
            patch_rows = [{"status": "failed",
                           "warning": f"{type(exc).__name__}: {exc}",
                           "norm_condition": spec.norm_condition,
                           "target_seq_len": item.n_tokens} for spec in specs]

        n_ran += 1
        if torch.cuda.is_available():
            peak_vram = max(peak_vram, int(torch.cuda.max_memory_allocated()))

        violated = False
        for (spec, source_condition, control), patch_row in zip(schedule, patch_rows):
            if patch_row.get("status") == "identity_violation":
                violated = True
            if patch_row.get("status") != "ok":
                n_failed += 1
            unit_rows.append(_row(
                identity, plan, lang=lang, semantic_id=semantic_id,
                gold_label=gold_label,
                gold_label_name=str(item.meta.get("gold_label_name", "")),
                site=site, patch_row=patch_row, source_condition=source_condition,
                control=control, unit_id=unit_id, site_id=site_id,
                unit_status=unit_status, band=band, seed=seed))

        if violated:
            # 04 §5.4: abort the unit. Identity is scheduled first, so nothing past this
            # example is computed, and every row already written is re-stamped invalid.
            unit_status = "identity_violation"
            for row in unit_rows:
                row["unit_status"] = unit_status
            break

    if need_runtime and n_ran:
        elapsed = time.time() - started
        total = already_run + n_ran
        per_example = elapsed / n_ran
        if total >= RUNTIME_ESTIMATE_AFTER or plan.max_examples is not None:
            estimate = {
                "unit": "patch_example",
                "measured_units": int(n_ran),
                "elapsed_s": round(elapsed, 3),
                "units_per_second": round(1.0 / per_example, 6) if per_example else None,
                "projected_total_units": int(len(corpus_ids) * len(plan.layers)
                                             * len(plan.objects)),
                "projected_total_s": round(per_example * len(corpus_ids)
                                           * len(plan.layers) * len(plan.objects), 3),
                "projected_finish_utc": "",
                "device": (torch.cuda.get_device_name(0) if torch.cuda.is_available()
                           else "cpu"),
                "peak_vram_bytes": int(peak_vram),
                "dtype": identity["dtype"],
                "measured_utc": prov.utc_now(),
                "stage": plan.stage,
            }

    if progress and unit_status == "ok":
        print(f"  [patch] {unit_id}: {len(unit_rows)} rows ({n_failed} not ok)")
    return unit_rows, unit_status, n_failed, n_ran, estimate


# ═══════════════════════════════════════════════════════════════════════════════
# Layer selection (04 §5.3)
# ═══════════════════════════════════════════════════════════════════════════════


def dev_effect_table(frame, *, treatment: str = "parallel_en",
                     control: str = "same_label_en", norm_condition: str = "direct"):
    """Per (object, layer) mean parallel-vs-control margin change on the dev rows.

    Paired **within target example** (``04`` §6.1): the two conditions are joined on
    (language, semantic id) before differencing, so an example that is missing one of them
    contributes to neither.
    """
    import pandas as pd

    usable = frame[(frame["status"] == "ok") & (frame["unit_status"] == "ok")
                   & (frame["partition"] == "dev")
                   & (frame["norm_condition"] == norm_condition)]
    if usable.empty:
        return pd.DataFrame(columns=["patch_object", "patch_layer", "effect",
                                     "n_pairs", "n_languages"])

    keys = ["patch_object", "patch_layer", "target_language", "semantic_id"]
    a = usable[usable["source_condition"] == treatment].set_index(keys)["margin_delta"]
    b = usable[usable["source_condition"] == control].set_index(keys)["margin_delta"]
    joined = a.to_frame("treatment").join(b.to_frame("control"), how="inner").dropna()
    if joined.empty:
        return pd.DataFrame(columns=["patch_object", "patch_layer", "effect",
                                     "n_pairs", "n_languages"])
    joined["paired"] = joined["treatment"] - joined["control"]
    joined = joined.reset_index()
    return (joined.groupby(["patch_object", "patch_layer"], as_index=False)
            .agg(effect=("paired", "mean"), n_pairs=("paired", "size"),
                 n_languages=("target_language", "nunique")))


def select_layer(frame, *, screen_layers: Sequence[int], window: int, num_layers: int,
                 primary_objects: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Pick the screened layer with the largest dev effect and build its window.

    The selection is over the **mean effect across the primary objects** at each screened
    layer, which is the only reading of ``04`` §5.3's "the layer with the largest
    parallel-vs-unrelated effect" that yields one layer for the whole run rather than one
    per object. Every screened layer's per-object effect is recorded either way, so a reader
    can check the choice rather than take it on trust.
    """
    table = dev_effect_table(frame)
    per_layer: List[Dict[str, Any]] = []
    if not table.empty:
        objects = list(primary_objects) if primary_objects else \
            sorted(table["patch_object"].unique())
        subset = table[table["patch_object"].isin(objects)]
        for layer in sorted({int(x) for x in screen_layers}):
            at_layer = subset[subset["patch_layer"] == layer]
            per_layer.append({
                "layer": int(layer),
                "effect": float(at_layer["effect"].mean()) if not at_layer.empty else None,
                "n_pairs": int(at_layer["n_pairs"].sum()) if not at_layer.empty else 0,
                "per_object": {row.patch_object: float(row.effect)
                               for row in at_layer.itertuples()},
            })

    scored = [row for row in per_layer if row["effect"] is not None]
    if not scored:
        return {
            "status": "no_effect",
            "screened_layers": [int(x) for x in screen_layers],
            "per_layer": per_layer,
            "selected_layer": None,
            "window": [],
            "reason": "no dev rows produced a usable parallel-vs-control pair; run "
                      "--stage screen first, and check run_summary_screen.json for "
                      "failures and invalid units",
        }

    best = max(scored, key=lambda row: row["effect"])
    return {
        "status": "ok",
        "screened_layers": [int(x) for x in screen_layers],
        "per_layer": per_layer,
        "selection_rule": "argmax over screened layers of the mean parallel_en minus "
                          "same_label_en margin change on the DEV partition, averaged "
                          "over the primary patch objects (04 §5.3)",
        "primary_objects": list(primary_objects) if primary_objects else None,
        "treatment_source": "parallel_en",
        "control_source": "same_label_en",
        "norm_condition": "direct",
        "partition_selected_on": "dev",
        "partition_to_report_on": "test",
        "selected_layer": int(best["layer"]),
        "selected_effect": best["effect"],
        "window_width": int(window),
        "window": clamp_window(int(best["layer"]), int(window), int(num_layers)),
        "num_layers": int(num_layers),
    }


def write_layer_selection(out_dir: Path, selection: Dict[str, Any]) -> Path:
    return prov.write_json(Path(out_dir) / SELECTION_FILENAME,
                           {**selection, "runner_version": RUNNER_VERSION,
                            **prov.provenance_block()})


def read_layer_selection(out_dir: Path) -> Optional[Dict[str, Any]]:
    path = Path(out_dir) / SELECTION_FILENAME
    if not path.exists():
        return None
    selection = json.loads(path.read_text(encoding="utf-8"))
    return selection if selection.get("status") == "ok" else None


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-language activation patching for E7 (04 §5).")
    parser.add_argument("--config", required=True)
    parser.add_argument("--manifest", default=None,
                        help="XNLI manifest JSON; defaults to the config's xnli.manifest")
    parser.add_argument("--controls", default=None,
                        help="patch_controls.csv; defaults to the manifest's sibling")
    parser.add_argument("--stage", choices=STAGES, default="screen")
    parser.add_argument("--out", default=None)
    parser.add_argument("--langs", nargs="+", default=None)
    parser.add_argument("--max-examples", type=int, default=None,
                        help="cap the examples per unit (the Phase 1 study sets its own)")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--force", dest="resume", action="store_false")
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    import pandas as pd

    args = build_parser().parse_args(argv)
    config = load_config(args.config)

    manifest_path = args.manifest
    if manifest_path is None:
        declared = (config.raw.get("xnli") or {}).get("manifest")
        if declared is None:
            print("No --manifest given and the config declares no xnli.manifest.")
            return 2
        manifest_path = Path(args.config).resolve().parents[1] / declared
    manifest_path = Path(manifest_path)
    manifest = pm.ParallelManifest.load(manifest_path)
    controls = read_patch_controls(
        args.controls or manifest_path.with_name("patch_controls.csv"))

    out = Path(args.out) if args.out else (_REPO / "crosslingual_semantics" / "results" /
                                           "patching" / config.tag)

    selection = None
    if args.stage == "window":
        selection = read_layer_selection(out)
        if selection is None:
            print(f"{out / SELECTION_FILENAME} is missing or records no usable selection. "
                  "Run --stage screen first: `04` §5.3 selects the layer on the dev "
                  "partition and reports it on the disjoint test partition.")
            return 2

    summary = run(config, manifest, controls, out, stage=args.stage,
                  langs=args.langs, resume=args.resume, device=args.device,
                  selection=selection, max_examples=args.max_examples,
                  source_cache_path=out / "source_cache.pt", progress=args.progress)

    print(f"{args.stage}: {summary['n_units']} units, {summary['n_rows']} rows, "
          f"{summary['n_failed_rows']} not ok, "
          f"{summary['n_invalid_units']} invalid units")
    print(f"  -> {summary['rows_csv']}")

    if args.stage in ("screen", "smoke"):
        patching = config.raw.get("patching") or {}
        frame = pd.read_csv(summary["rows_csv"], encoding="utf-8")
        chosen = select_layer(
            frame, screen_layers=summary["layers"],
            window=int(patching.get("window", 5)),
            num_layers=int(summary["num_layers"]),
            primary_objects=(patching.get("smoke") or {}).get("objects")
            if args.stage == "smoke" else ["K0_prerope", "V0"])
        write_layer_selection(out, chosen)
        if chosen["status"] == "ok":
            print(f"  selected layer {chosen['selected_layer']} (dev effect "
                  f"{chosen['selected_effect']:.5f}); window {chosen['window']}")
        else:
            print(f"  no layer selected: {chosen['reason']}")

    if summary["n_invalid_units"]:
        print(f"ABORTED UNITS: {summary['n_invalid_units']} run units failed the identity "
              f"control (04 §5.4). See {out / INVALID_FILENAME}. Those units are invalid "
              "and are excluded from every contrast.")
        return 1
    return 0 if summary["n_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
