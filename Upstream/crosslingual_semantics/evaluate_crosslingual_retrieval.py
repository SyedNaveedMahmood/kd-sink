# -*- coding: utf-8 -*-
"""evaluate_crosslingual_retrieval.py — does the sink carry semantics across languages? (WP9)

``04_MODULE_SPEC_e7_crosslingual.md`` §4. Reads the ``.npy`` representations written by
``extract_sink_representations.py`` and, for each (model, source language, target language,
layer, object), retrieves the matching semantic id by cosine similarity.

    python crosslingual_semantics/evaluate_crosslingual_retrieval.py \\
      --reps crosslingual_semantics/results/reps/qwen05_base \\
      --manifest crosslingual_semantics/results/manifests/flores_devtest.json \\
      --out crosslingual_semantics/results/retrieval

The one step that decides whether the number means anything
-----------------------------------------------------------
**Each language is centred separately, before similarity.** Without it, the language mean
dominates cosine similarity and "retrieval" measures script identity rather than semantics
— it would produce a large, smooth, entirely uninformative number. This is asserted in
``tests/test_retrieval_chance.py``, not merely commented here.

What every table carries
------------------------
* top-1, top-5, MRR, median rank, and **ratio to chance** (chance = 1/n_candidates), since
  the go/no-go bar in ``04`` §4 is "at least twice chance".
* Two alignment variants: raw cosine, and orthogonal Procrustes fitted on the manifest's
  ``procrustes_train`` partition and evaluated on the **disjoint** ``procrustes_test``
  partition. The disjointness is asserted before the fit, not assumed.
* The **matched middle-token control in the same row** (``Kmid``/``Vmid``/``Rmid``): ``04``
  §4 is explicit that a position-0 number without its control cannot support the claim, so
  ``retrieval_by_object.csv`` carries ``control_object`` and
  ``position0_minus_mid_top1_points`` beside every position-0 row.
* Self-pairs (source language == target language) are computed and written with
  ``is_self_pair=True`` as a continuous correctness monitor — after centring they must
  retrieve perfectly — and are excluded from every aggregate.

The language-identity probe is grouped 5-fold CV split **by semantic id, never by row**,
standardised on training folds only, reporting macro-F1 and balanced accuracy (``04`` §4).
Splitting by row would put the same sentence in train and test in different languages and
inflate the score.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO, _REPO / "common"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import paired_manifests as pm  # noqa: E402
import provenance as prov  # noqa: E402

RETRIEVAL_VERSION = "evaluate_crosslingual_retrieval_v1"

#: position-0 object -> its matched middle-token control (``04`` §4).
MID_CONTROL: Dict[str, str] = {
    "R0": "Rmid",
    "K0_prerope": "Kmid_prerope",
    "K0_postrope": "Kmid_postrope",
    "V0": "Vmid",
}

PAIR_COLUMNS: Tuple[str, ...] = (
    "model_tag", "model", "variant", "dtype", "object", "layer", "alignment",
    "source_language", "target_language", "is_self_pair", "partition", "n_candidates",
    "top1", "top5", "mrr", "median_rank", "chance", "top1_over_chance", "centring",
    "manifest_sha256", "retrieval_version", "git_sha",
)

OBJECT_COLUMNS: Tuple[str, ...] = (
    "model_tag", "model", "variant", "dtype", "object", "layer", "alignment", "partition",
    "n_pairs", "n_candidates", "top1", "top5", "mrr", "median_rank", "chance",
    "top1_over_chance", "control_object", "control_top1",
    "position0_minus_mid_top1_points", "manifest_sha256", "retrieval_version", "git_sha",
)

PROBE_COLUMNS: Tuple[str, ...] = (
    "model_tag", "model", "dtype", "object", "layer", "n_samples", "n_languages",
    "n_folds", "macro_f1", "balanced_accuracy", "chance_accuracy", "grouped_by",
    "status", "warning", "manifest_sha256", "retrieval_version", "git_sha",
)


# ═══════════════════════════════════════════════════════════════════════════════
# Loading
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class RepsBundle:
    """Every language's representations for one model, plus the identity to record."""

    root: Path
    run_config: Dict[str, Any]
    indexes: Dict[str, Dict[str, Any]]          # lang -> index.json
    arrays: Dict[Tuple[str, str], np.ndarray]   # (lang, object) -> [n, n_layers, dim]

    @property
    def languages(self) -> List[str]:
        return sorted(self.indexes)

    @property
    def objects(self) -> List[str]:
        return list(self.run_config.get("objects") or [])

    @property
    def layers(self) -> List[int]:
        return [int(x) for x in (self.run_config.get("layers") or [])]

    def semantic_ids(self, lang: str) -> List[str]:
        return list(self.indexes[lang]["semantic_ids"])

    def rows_for(self, lang: str, semantic_ids: Sequence[str]) -> np.ndarray:
        """Row indices for ``semantic_ids`` — an explicit join, never a positional one."""
        row_of = self.indexes[lang]["row_of"]
        missing = [sid for sid in semantic_ids if sid not in row_of]
        if missing:
            raise KeyError(f"{lang}: semantic ids absent from index.json: {missing[:5]}")
        return np.asarray([int(row_of[sid]) for sid in semantic_ids], dtype=np.int64)

    def matrix(self, lang: str, obj: str, layer: int,
               semantic_ids: Sequence[str]) -> np.ndarray:
        """``[n_ids, dim]`` fp64 for one (language, object, layer), joined by semantic id."""
        array = self.arrays[(lang, obj)]
        layers = self.layers
        if layer not in layers:
            raise KeyError(f"layer {layer} not extracted (have {layers})")
        rows = self.rows_for(lang, semantic_ids)
        return np.asarray(array[rows, layers.index(layer), :], dtype=np.float64)


def load_reps(root) -> RepsBundle:
    """Load one ``reps/<model_tag>/`` directory as memmaps — nothing is read into RAM yet."""
    root = Path(root)
    run_config_path = root / "run_config.json"
    if not run_config_path.exists():
        raise FileNotFoundError(
            f"{run_config_path} not found; --reps points at a model tag directory written "
            "by extract_sink_representations.py")
    run_config = json.loads(run_config_path.read_text(encoding="utf-8"))

    indexes: Dict[str, Dict[str, Any]] = {}
    arrays: Dict[Tuple[str, str], np.ndarray] = {}
    for index_path in sorted(root.glob("*/index.json")):
        index = json.loads(index_path.read_text(encoding="utf-8"))
        lang = str(index["language"])
        indexes[lang] = index
        for obj in index["objects"]:
            arrays[(lang, obj)] = np.load(index_path.parent / f"{obj}.npy", mmap_mode="r")
    if not indexes:
        raise FileNotFoundError(f"no per-language index.json under {root}")
    return RepsBundle(root=root, run_config=run_config, indexes=indexes, arrays=arrays)


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


def centre(matrix: np.ndarray) -> np.ndarray:
    """Remove the language mean. The single most important line in this module."""
    return matrix - matrix.mean(axis=0, keepdims=True)


def _unit(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    # A zero row has no direction; leaving it at zero makes its similarity to everything 0,
    # which ranks it last rather than producing a NaN that would silently poison the mean.
    norms[norms == 0.0] = 1.0
    return matrix / norms


def retrieval_metrics(source: np.ndarray, target: np.ndarray) -> Dict[str, float]:
    """Row *i* of ``source`` should retrieve row *i* of ``target``.

    Both are centred here, per language, before any similarity is computed. Ranks are
    1-based; ties are resolved pessimistically (a tie with the gold counts against it), so
    a degenerate all-identical representation cannot look like perfect retrieval.
    """
    if source.shape[0] != target.shape[0]:
        raise ValueError(f"unequal candidate counts {source.shape[0]} vs {target.shape[0]}")
    n = int(source.shape[0])
    if n == 0:
        raise ValueError("retrieval needs at least one candidate")

    similarity = _unit(centre(source)) @ _unit(centre(target)).T
    gold = np.diagonal(similarity).copy()
    # Pessimistic rank: how many candidates score at least as well as the gold one.
    ranks = (similarity >= gold[:, None]).sum(axis=1).astype(np.float64)

    chance = 1.0 / n
    top1 = float((ranks <= 1).mean())
    return {
        "n_candidates": n,
        "top1": top1,
        "top5": float((ranks <= 5).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "chance": chance,
        "top1_over_chance": float(top1 / chance) if chance > 0 else float("nan"),
    }


def procrustes_rotation(source_train: np.ndarray, target_train: np.ndarray) -> np.ndarray:
    """Orthogonal map ``R`` minimising ``||source_train @ R - target_train||`` (``04`` §4)."""
    from scipy.linalg import orthogonal_procrustes

    rotation, _scale = orthogonal_procrustes(centre(source_train), centre(target_train))
    return rotation


# ═══════════════════════════════════════════════════════════════════════════════
# Language-identity probe
# ═══════════════════════════════════════════════════════════════════════════════


def language_probe(features_by_lang: Dict[str, np.ndarray], semantic_ids: Sequence[str], *,
                   n_folds: int = 5, seed: int = 42) -> Dict[str, Any]:
    """Multinomial logistic regression, grouped 5-fold CV **by semantic id** (``04`` §4).

    The grouping is the point: splitting by row would place the *same sentence* in train and
    test in different languages, and the probe would then measure memorisation rather than
    language identity. Standardisation is fitted on the training folds only, for the same
    reason.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, f1_score
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import StandardScaler

    langs = sorted(features_by_lang)
    x = np.concatenate([features_by_lang[lang] for lang in langs], axis=0)
    y = np.concatenate([np.full(features_by_lang[lang].shape[0], i)
                        for i, lang in enumerate(langs)])
    groups = np.concatenate([np.asarray(semantic_ids) for _ in langs])

    n_groups = len(set(groups.tolist()))
    folds = min(int(n_folds), n_groups)
    if folds < 2 or len(langs) < 2:
        return {"status": "skipped", "n_samples": int(x.shape[0]),
                "n_languages": len(langs), "n_folds": folds,
                "macro_f1": float("nan"), "balanced_accuracy": float("nan"),
                "chance_accuracy": (1.0 / len(langs)) if langs else float("nan"),
                "warning": f"need >=2 languages and >=2 semantic-id groups; "
                           f"got {len(langs)} languages, {n_groups} groups"}

    true: List[np.ndarray] = []
    pred: List[np.ndarray] = []
    for train_idx, test_idx in GroupKFold(n_splits=folds).split(x, y, groups):
        scaler = StandardScaler().fit(x[train_idx])
        model = LogisticRegression(max_iter=2000, random_state=seed)
        model.fit(scaler.transform(x[train_idx]), y[train_idx])
        true.append(y[test_idx])
        pred.append(model.predict(scaler.transform(x[test_idx])))
    y_true = np.concatenate(true)
    y_pred = np.concatenate(pred)
    return {
        "status": "ok",
        "n_samples": int(x.shape[0]),
        "n_languages": len(langs),
        "n_folds": folds,
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "chance_accuracy": 1.0 / len(langs),
        "warning": "",
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Driver
# ═══════════════════════════════════════════════════════════════════════════════


def _identity(bundle: RepsBundle, manifest_sha: str) -> Dict[str, Any]:
    cfg = bundle.run_config
    return {
        "model_tag": cfg.get("model_tag", bundle.root.name),
        "model": cfg.get("model", ""),
        "variant": cfg.get("variant", ""),
        "dtype": cfg.get("dtype", ""),
        "manifest_sha256": manifest_sha,
        "retrieval_version": RETRIEVAL_VERSION,
        "git_sha": prov.git_sha(),
    }


def _pair_rows(bundle: RepsBundle, identity: Dict[str, Any], *, objects: Sequence[str],
               layers: Sequence[int], languages: Sequence[str],
               raw_ids: Sequence[str], procrustes: Optional[Tuple[Sequence[str],
                                                                  Sequence[str]]],
               progress: bool) -> List[Dict[str, Any]]:
    """One row per (object, layer, alignment, source lang, target lang)."""
    rows: List[Dict[str, Any]] = []
    for obj in objects:
        if progress:
            print(f"  [retrieval] {obj}")
        for layer in layers:
            cache = {lang: bundle.matrix(lang, obj, layer, raw_ids) for lang in languages}
            for source in languages:
                for target in languages:
                    metrics = retrieval_metrics(cache[source], cache[target])
                    rows.append({**identity, "object": obj, "layer": int(layer),
                                 "alignment": "cosine", "source_language": source,
                                 "target_language": target,
                                 "is_self_pair": source == target,
                                 "partition": "all",
                                 "centring": "per_language_over_evaluated_rows",
                                 **metrics})
            del cache

            if procrustes is None:
                continue
            train_ids, test_ids = procrustes
            train = {lang: bundle.matrix(lang, obj, layer, train_ids)
                     for lang in languages}
            test = {lang: bundle.matrix(lang, obj, layer, test_ids) for lang in languages}
            for source in languages:
                for target in languages:
                    rotation = procrustes_rotation(train[source], train[target])
                    aligned = centre(test[source]) @ rotation
                    metrics = retrieval_metrics(aligned, test[target])
                    rows.append({**identity, "object": obj, "layer": int(layer),
                                 "alignment": "procrustes", "source_language": source,
                                 "target_language": target,
                                 "is_self_pair": source == target,
                                 "partition": "procrustes_test",
                                 "centring": "per_language_over_evaluated_rows",
                                 **metrics})
            del train, test
    return rows


def _object_rows(pair_frame, identity: Dict[str, Any]):
    """Aggregate over cross-language pairs and attach the matched middle-token control."""
    import pandas as pd

    cross = pair_frame[~pair_frame["is_self_pair"]]
    if cross.empty:
        return pd.DataFrame(columns=list(OBJECT_COLUMNS))

    grouped = (cross.groupby(["object", "layer", "alignment", "partition"], as_index=False)
               .agg(n_pairs=("top1", "size"),
                    n_candidates=("n_candidates", "max"),
                    top1=("top1", "mean"), top5=("top5", "mean"),
                    mrr=("mrr", "mean"), median_rank=("median_rank", "mean"),
                    chance=("chance", "max"),
                    top1_over_chance=("top1_over_chance", "mean")))

    lookup = {(r.object, r.layer, r.alignment, r.partition): r.top1
              for r in grouped.itertuples()}
    control_object: List[Any] = []
    control_top1: List[Any] = []
    delta_points: List[Any] = []
    for row in grouped.itertuples():
        control = MID_CONTROL.get(row.object)
        control_object.append(control)
        value = lookup.get((control, row.layer, row.alignment, row.partition)) \
            if control else None
        control_top1.append(value)
        # `04` §4 / design §18: position 0 must beat the matched middle-token control by
        # >= 5 *points*, so the difference is reported in percentage points, not as a ratio.
        delta_points.append(None if value is None else (row.top1 - value) * 100.0)

    grouped["control_object"] = control_object
    grouped["control_top1"] = control_top1
    grouped["position0_minus_mid_top1_points"] = delta_points
    for key, value in identity.items():
        grouped[key] = value
    return grouped.reindex(columns=list(OBJECT_COLUMNS))


def evaluate(reps_root: Path, manifest: pm.ParallelManifest, out_dir: Path, *,
             objects: Optional[Sequence[str]] = None,
             layers: Optional[Sequence[int]] = None,
             languages: Optional[Sequence[str]] = None,
             probe_layers: Optional[Sequence[int]] = None,
             probe: bool = True, figures: bool = True, seed: int = 42,
             progress: bool = True) -> Dict[str, Any]:
    """Run every retrieval table for one extracted model. Returns the summary dict."""
    import pandas as pd

    started = time.time()
    bundle = load_reps(reps_root)
    out_dir = Path(out_dir)
    identity = _identity(bundle, manifest.sha256)

    stored_sha = bundle.run_config.get("manifest_sha256")
    if stored_sha and stored_sha != manifest.sha256:
        raise ValueError(
            f"representations were extracted against manifest {stored_sha[:12]}… but "
            f"--manifest is {manifest.sha256[:12]}…. `05` §7.1: hash before compare.")

    languages = list(languages) if languages else bundle.languages
    objects = list(objects) if objects else bundle.objects
    layers = [int(x) for x in layers] if layers else bundle.layers

    raw_ids = [sid for sid in manifest.semantic_ids
               if all(sid in bundle.indexes[lang]["row_of"] for lang in languages)]
    if not raw_ids:
        raise ValueError("no semantic id is present in every requested language")

    procrustes: Optional[Tuple[List[str], List[str]]] = None
    procrustes_note = ""
    train = [sid for sid in manifest.partitions.get("procrustes_train", ()) if sid in raw_ids]
    test = [sid for sid in manifest.partitions.get("procrustes_test", ()) if sid in raw_ids]
    if train and test:
        overlap = set(train) & set(test)
        if overlap:
            raise ValueError(
                f"procrustes_train and procrustes_test share {len(overlap)} semantic ids "
                "(e.g. " + ", ".join(sorted(overlap)[:3]) + "). `04` §4 requires them to be "
                "disjoint; a fit evaluated on its own training ids is not a measurement.")
        procrustes = (train, test)
    else:
        procrustes_note = ("manifest has no procrustes_train/procrustes_test partitions; "
                           "only the raw-cosine alignment was computed")

    pair_rows = _pair_rows(bundle, identity, objects=objects, layers=layers,
                           languages=languages, raw_ids=raw_ids, procrustes=procrustes,
                           progress=progress)
    pair_frame = pd.DataFrame(pair_rows).reindex(columns=list(PAIR_COLUMNS))
    object_frame = _object_rows(pair_frame, identity)

    probe_rows: List[Dict[str, Any]] = []
    if probe:
        wanted = [int(x) for x in probe_layers] if probe_layers else layers
        for obj in objects:
            for layer in wanted:
                if progress:
                    print(f"  [probe] {obj} L{layer}")
                features = {lang: bundle.matrix(lang, obj, layer, raw_ids)
                            for lang in languages}
                try:
                    result = language_probe(features, raw_ids, seed=seed)
                except Exception as exc:
                    result = {"status": "failed", "n_samples": 0,
                              "n_languages": len(languages), "n_folds": 0,
                              "macro_f1": float("nan"),
                              "balanced_accuracy": float("nan"),
                              "chance_accuracy": float("nan"),
                              "warning": f"{type(exc).__name__}: {exc}"}
                probe_rows.append({**identity, "object": obj, "layer": int(layer),
                                   "grouped_by": "semantic_id", **result})
                del features
    probe_frame = pd.DataFrame(probe_rows).reindex(columns=list(PROBE_COLUMNS))

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_path = out_dir / "retrieval_by_pair.csv"
    object_path = out_dir / "retrieval_by_object.csv"
    probe_path = out_dir / "language_probe.csv"
    pair_frame.to_csv(pair_path, index=False, encoding="utf-8")
    object_frame.to_csv(object_path, index=False, encoding="utf-8")
    probe_frame.to_csv(probe_path, index=False, encoding="utf-8")

    figure_note = "skipped"
    if figures:
        try:
            figure = plot_retrieval_heatmap(pair_frame, out_dir / "figures" /
                                            "fig5_retrieval_heatmap.pdf")
            figure_note = str(figure) if figure else "skipped: no data"
        except Exception as exc:  # a plotting failure must never lose a table
            figure_note = f"failed: {type(exc).__name__}: {exc}"

    summary = {
        **identity,
        "reps_root": str(reps_root),
        "manifest_id": manifest.manifest_id,
        "languages": languages,
        "objects": objects,
        "layers": layers,
        "n_semantic_ids": len(raw_ids),
        "n_procrustes_train": len(procrustes[0]) if procrustes else 0,
        "n_procrustes_test": len(procrustes[1]) if procrustes else 0,
        "procrustes_note": procrustes_note,
        "n_pair_rows": int(len(pair_frame)),
        "n_object_rows": int(len(object_frame)),
        "n_probe_rows": int(len(probe_frame)),
        "outputs": {"retrieval_by_pair": str(pair_path),
                    "retrieval_by_object": str(object_path),
                    "language_probe": str(probe_path),
                    "fig5_retrieval_heatmap": figure_note},
        "wallclock_s": round(time.time() - started, 3),
    }
    prov.write_json(out_dir / "retrieval_summary.json",
                    {**summary, **prov.provenance_block()})
    return summary


# ═══════════════════════════════════════════════════════════════════════════════
# Figure
# ═══════════════════════════════════════════════════════════════════════════════


def _figure_setup():
    """``Agg`` is selected here, never at import — matching ``aggregate_transformation``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    return path


def plot_retrieval_heatmap(pair_frame, path: Path) -> Optional[Path]:
    """Source × target top-1 heatmap at the best layer of each position-0 object."""
    frame = pair_frame[(pair_frame["alignment"] == "cosine")
                       & (~pair_frame["is_self_pair"])]
    objects = [obj for obj in MID_CONTROL if obj in set(frame["object"])]
    if frame.empty or not objects:
        return None

    plt = _figure_setup()
    langs = sorted(set(frame["source_language"]) | set(frame["target_language"]))
    fig, axes = plt.subplots(1, len(objects), figsize=(4.2 * len(objects), 4.0),
                             squeeze=False)
    for ax, obj in zip(axes[0], objects):
        subset = frame[frame["object"] == obj]
        # Report the layer where the object retrieves best — the layer sweep itself is
        # exploratory and lives in the CSV, not in this summary figure.
        by_layer = subset.groupby("layer")["top1"].mean()
        best = int(by_layer.idxmax())
        at_best = subset[subset["layer"] == best]
        grid = np.full((len(langs), len(langs)), np.nan)
        for row in at_best.itertuples():
            grid[langs.index(row.source_language), langs.index(row.target_language)] = \
                row.top1
        image = ax.imshow(grid, vmin=0.0, vmax=1.0, cmap="viridis")
        ax.set_title(f"{obj} @ L{best}")
        ax.set_xticks(range(len(langs)))
        ax.set_xticklabels(langs, rotation=90, fontsize=7)
        ax.set_yticks(range(len(langs)))
        ax.set_yticklabels(langs, fontsize=7)
        ax.set_xlabel("target")
        ax.set_ylabel("source")
        fig.colorbar(image, ax=ax, fraction=0.046, label="top-1")
    fig.suptitle("E7 cross-lingual retrieval (top-1, per-language centred)")
    saved = _save(fig, Path(path))
    plt.close(fig)
    return saved


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-lingual retrieval from extracted sink representations (04 §4).")
    parser.add_argument("--reps", required=True,
                        help="a reps/<model_tag>/ directory")
    parser.add_argument("--manifest", required=True, help="the FLORES manifest JSON")
    parser.add_argument("--out", default=None,
                        help="output directory (default crosslingual_semantics/results/"
                             "retrieval/<model_tag>)")
    parser.add_argument("--objects", nargs="+", default=None)
    parser.add_argument("--layers", nargs="+", type=int, default=None)
    parser.add_argument("--languages", nargs="+", default=None)
    parser.add_argument("--probe-layers", nargs="+", type=int, default=None,
                        help="restrict the language probe to these layers (it is the "
                             "expensive part; retrieval always covers every layer)")
    parser.add_argument("--no-probe", dest="probe", action="store_false", default=True)
    parser.add_argument("--no-figures", dest="figures", action="store_false", default=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quiet", dest="progress", action="store_false", default=True)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    manifest = pm.ParallelManifest.load(args.manifest)
    reps = Path(args.reps)
    out = Path(args.out) if args.out else (
        _REPO / "crosslingual_semantics" / "results" / "retrieval" / reps.name)

    summary = evaluate(reps, manifest, out, objects=args.objects, layers=args.layers,
                       languages=args.languages, probe_layers=args.probe_layers,
                       probe=args.probe, figures=args.figures, seed=args.seed,
                       progress=args.progress)
    print(f"Retrieval for {summary['model_tag']}: {summary['n_pair_rows']} pair rows, "
          f"{summary['n_object_rows']} object rows, {summary['n_probe_rows']} probe rows")
    if summary["procrustes_note"]:
        print(f"  note: {summary['procrustes_note']}")
    print(f"  -> {out}")
    return 0 if summary["n_pair_rows"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
