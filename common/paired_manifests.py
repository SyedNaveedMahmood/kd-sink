# -*- coding: utf-8 -*-
"""paired_manifests.py — parallel joins and patch controls for E7 (WP8).

``02_MODULE_SPEC_common.md`` §5. Builds and validates the cross-language joins E7 depends
on, plus the fixed control assignments for cross-example patching.

The alignment trap (§5.2, CLAUDE.md trap 2)
-------------------------------------------
FLORES ``devtest`` rows *are* aligned by position within each language config, so joining
on the source index is correct there. **XNLI is not.** It must be joined on an explicit id,
and where the loaded shards carry no id column the structurally aligned ``all_languages``
config is used instead — see ``datasets_loader.load_xnli_aligned``, which resolves the
strategy and records it. A positional join produces a manifest that looks entirely healthy
and silently destroys every cross-language claim, so there is no positional fallback
anywhere in this module.

``verify_manifest`` is the mechanical check: every semantic id present in every language,
no duplicates, token counts within the declared bounds, partitions disjoint and exhaustive,
and the recorded sha256 equal to a recomputation.

Pure data + tokeniser. No model loading.
"""

from __future__ import annotations

import json
import random
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

try:  # dual-import idiom: scripts put common/ on sys.path directly
    from . import provenance as _prov
except ImportError:  # pragma: no cover - exercised via direct-path imports
    import provenance as _prov

import datasets_loader as _dl

MANIFEST_SCHEMA_VERSION = "parallel_manifest_v1"

#: The eight FLORES languages of ``04`` §1.
FLORES_LANGS: Tuple[str, ...] = (
    "eng_Latn", "ben_Beng", "zho_Hans", "arb_Arab",
    "deu_Latn", "hin_Deva", "swh_Latn", "tur_Latn",
)

#: The seven XNLI languages of ``04`` §2.
XNLI_LANGS: Tuple[str, ...] = ("en", "zh", "ar", "de", "hi", "sw", "tr")

#: Language family, for the covariates ``04`` §1 requires. Small and explicit rather than
#: inferred: eight languages do not justify a dependency, and a wrong inference would be
#: an invisible confound in the high/low-resource contrast.
LANGUAGE_FAMILY: Dict[str, str] = {
    "eng_Latn": "Indo-European/Germanic", "deu_Latn": "Indo-European/Germanic",
    "ben_Beng": "Indo-European/Indo-Aryan", "hin_Deva": "Indo-European/Indo-Aryan",
    "zho_Hans": "Sino-Tibetan", "arb_Arab": "Afro-Asiatic/Semitic",
    "swh_Latn": "Niger-Congo/Bantu", "tur_Latn": "Turkic",
    "en": "Indo-European/Germanic", "de": "Indo-European/Germanic",
    "hi": "Indo-European/Indo-Aryan", "zh": "Sino-Tibetan",
    "ar": "Afro-Asiatic/Semitic", "sw": "Niger-Congo/Bantu", "tr": "Turkic",
}

#: XNLI prompt template (``04`` §2). The premise begins at **position 0** — no instruction
#: prefix, because a constant prefix would manufacture a shared first-token anchor and
#: invalidate every position-0 measurement (design §16.2).
XNLI_PROMPT_TEMPLATE = (
    "{premise}\n{hypothesis}\n"
    "Question: Is the second statement entailed by, neutral with respect to, or "
    "contradictory to the first?\nAnswer:"
)

#: Candidate label strings, scored by length-normalised sequence log-probability.
XNLI_CANDIDATES: Tuple[str, ...] = (" entailment", " neutral", " contradiction")

#: The four patch source conditions of ``04`` §5.1.
PATCH_CONTROL_CONDITIONS: Tuple[str, ...] = (
    "parallel_en", "same_label_en", "different_label_en", "random_en")


# ═══════════════════════════════════════════════════════════════════════════════
# The manifest
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass(frozen=True)
class ParallelManifest:
    """A cross-language join: one semantic id, one row per language, plus partitions."""

    manifest_id: str
    dataset: str                                   # "flores" | "xnli"
    split: str
    languages: Tuple[str, ...]
    semantic_ids: Tuple[str, ...]
    rows: Dict[str, Dict[str, dict]]               # semantic_id -> lang -> row
    partitions: Dict[str, Tuple[str, ...]] = field(default_factory=dict)
    seed: int = 42
    sha256: str = ""
    provenance: dict = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.semantic_ids)

    def row(self, semantic_id: str, lang: str) -> dict:
        return self.rows[semantic_id][lang]

    def token_counts(self, lang: str) -> List[int]:
        return [int(self.rows[sid][lang]["n_tokens"]) for sid in self.semantic_ids]

    def to_dict(self) -> dict:
        return {
            "schema": MANIFEST_SCHEMA_VERSION,
            "manifest_id": self.manifest_id,
            "dataset": self.dataset,
            "split": self.split,
            "languages": list(self.languages),
            "semantic_ids": list(self.semantic_ids),
            "rows": self.rows,
            "partitions": {k: list(v) for k, v in self.partitions.items()},
            "seed": self.seed,
            "sha256": self.sha256,
            "provenance": self.provenance,
        }

    def save(self, path) -> Path:
        return _prov.write_json(path, self.to_dict())

    @staticmethod
    def load(path) -> "ParallelManifest":
        """Load and re-verify. A hash mismatch raises rather than warning."""
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        manifest = ParallelManifest(
            manifest_id=payload["manifest_id"], dataset=payload["dataset"],
            split=payload["split"], languages=tuple(payload["languages"]),
            semantic_ids=tuple(payload["semantic_ids"]), rows=payload["rows"],
            partitions={k: tuple(v) for k, v in payload.get("partitions", {}).items()},
            seed=int(payload.get("seed", 42)), sha256=payload.get("sha256", ""),
            provenance=payload.get("provenance", {}))
        recomputed = compute_manifest_sha256(manifest)
        if manifest.sha256 and recomputed != manifest.sha256:
            raise ValueError(
                f"Manifest hash mismatch on load: stored {manifest.sha256[:16]}…, "
                f"recomputed {recomputed[:16]}…. The file has been edited or was written "
                "by a different schema version.")
        return manifest


def compute_manifest_sha256(manifest: ParallelManifest) -> str:
    """SHA-256 over the canonical content of the join, excluding its own hash.

    Partitions are included: two manifests with the same rows but different dev/test
    splits are different artefacts and must not compare equal (``05`` §7.1).
    """
    payload = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "dataset": manifest.dataset,
        "split": manifest.split,
        "languages": list(manifest.languages),
        "semantic_ids": list(manifest.semantic_ids),
        "rows": {sid: {lang: _canonical_row(manifest.rows[sid][lang])
                       for lang in sorted(manifest.rows[sid])}
                 for sid in manifest.semantic_ids},
        "partitions": {k: list(v) for k, v in sorted(manifest.partitions.items())},
        "seed": manifest.seed,
    }
    return _prov.sha256_json(payload)


def _canonical_row(row: dict) -> dict:
    """The hash-relevant fields of a row, in a fixed order."""
    return {k: row[k] for k in sorted(row) if k not in ("_transient",)}


def _with_hash(manifest: ParallelManifest) -> ParallelManifest:
    from dataclasses import replace
    return replace(manifest, sha256=compute_manifest_sha256(manifest))


# ═══════════════════════════════════════════════════════════════════════════════
# Covariates (04 §1)
# ═══════════════════════════════════════════════════════════════════════════════


def unicode_script(text: str) -> str:
    """Dominant Unicode script of a string, from the most common letter's name."""
    scripts: Counter = Counter()
    for ch in text:
        if not ch.isalpha():
            continue
        try:
            name = unicodedata.name(ch)
        except ValueError:
            continue
        scripts[name.split(" ")[0]] += 1
    return scripts.most_common(1)[0][0] if scripts else "UNKNOWN"


def starts_with_punctuation(text: str) -> bool:
    """``04`` §1's position-0 punctuation indicator, on the first character."""
    return bool(text) and unicodedata.category(text[0]).startswith("P")


def _first_token_frequencies(tokenizer, texts: Iterable[str]) -> Dict[int, int]:
    """Count first-token ids over a corpus, for the first-token-frequency covariate."""
    counts: Counter = Counter()
    for text in texts:
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        if len(ids):
            counts[int(ids[0])] += 1
    return dict(counts)


# ═══════════════════════════════════════════════════════════════════════════════
# FLORES
# ═══════════════════════════════════════════════════════════════════════════════


def build_flores_manifest(tokenizer, langs=FLORES_LANGS, *, split="devtest",
                          min_tokens: int = 12, max_tokens: int = 160,
                          n: int = 300, seed: int = 42) -> ParallelManifest:
    """Join FLORES by source index and keep rows every language passes the length filter.

    ``04`` §1: tokenise without special tokens, keep a row only when **every** language
    lands in ``[min_tokens, max_tokens]``, then sample ``n`` with ``seed``. Fewer than 200
    survivors sets ``provenance["valid"] = False``; the caller (the prepare script) is what
    exits non-zero. Below ``n`` but at or above 200, all survivors are kept and the realised
    count is recorded.

    Emits the per-language covariates the analysis needs later: first-token id and its
    frequency over the full split, a punctuation-at-position-0 indicator, Unicode script
    and language family.
    """
    langs = tuple(langs)
    raw, info = _dl.load_flores_parallel(langs, split=split)

    counts = set(info["rows_per_language"].values())
    if len(counts) != 1:
        raise ValueError(
            "FLORES language configs have different row counts "
            f"{info['rows_per_language']}; a source-index join is only valid when the "
            "configs are line-aligned.")

    frequencies = {lang: _first_token_frequencies(
        tokenizer, (row["text"] for row in raw[lang].values())) for lang in langs}

    shared = set(raw[langs[0]])
    for lang in langs[1:]:
        shared &= set(raw[lang])

    kept: Dict[str, Dict[str, dict]] = {}
    n_seen = 0
    for source_index in sorted(shared):
        n_seen += 1
        per_lang: Dict[str, dict] = {}
        ok = True
        for lang in langs:
            text = raw[lang][source_index]["text"]
            ids = [int(t) for t in tokenizer(text, add_special_tokens=False)["input_ids"]]
            if not min_tokens <= len(ids) <= max_tokens:
                ok = False
                break
            first = ids[0]
            per_lang[lang] = {
                "language": lang,
                "text": text,
                "input_ids": ids,
                "n_tokens": len(ids),
                "first_token_id": first,
                "first_token_frequency": int(frequencies[lang].get(first, 0)),
                "punctuation_at_position_0": starts_with_punctuation(text),
                "script": unicode_script(text),
                "language_family": LANGUAGE_FAMILY.get(lang, "unknown"),
                "source_index": int(source_index),
            }
        if ok:
            kept[f"flores_{split}_{source_index:05d}"] = per_lang

    survivors = sorted(kept)
    rng = random.Random(seed)
    if len(survivors) > n:
        chosen = sorted(rng.sample(survivors, n))
    else:
        chosen = survivors

    valid = len(chosen) >= 200
    manifest = ParallelManifest(
        manifest_id=f"flores_{split}_{len(chosen)}",
        dataset="flores", split=split, languages=langs,
        semantic_ids=tuple(chosen),
        rows={sid: kept[sid] for sid in chosen},
        partitions={}, seed=seed,
        provenance=_prov.provenance_block(extra={
            "loader": info,
            "min_tokens": min_tokens, "max_tokens": max_tokens,
            "requested_n": n, "realised_n": len(chosen),
            "n_candidates": n_seen, "n_survivors": len(survivors),
            "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
            "valid": valid,
            "schema": MANIFEST_SCHEMA_VERSION,
        }))
    return _with_hash(manifest)


def build_length_matched_subset(manifest: ParallelManifest,
                                reference_lang: str = "eng_Latn",
                                max_rel_diff: float = 0.20) -> ParallelManifest:
    """Length-matched secondary subset via optimal bipartite matching (``02`` §5.1).

    For each non-reference language, sentences are matched to reference sentences by token
    count with ``scipy.optimize.linear_sum_assignment``, and a semantic id is kept only if
    **every** language's matched partner is within ``max_rel_diff`` relative token-count
    difference. The realised maximum relative difference per language is recorded.

    Note what this does and does not claim: the matching is over token counts within the
    manifest, so it controls length as a covariate; it does not re-pair sentences across
    languages (the semantic join is untouched).
    """
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    if reference_lang not in manifest.languages:
        raise ValueError(f"reference_lang {reference_lang!r} is not in "
                         f"{manifest.languages}")
    ids = list(manifest.semantic_ids)
    reference = np.array([manifest.rows[sid][reference_lang]["n_tokens"] for sid in ids],
                         dtype=float)

    keep = set(ids)
    realised: Dict[str, float] = {}
    matches: Dict[str, Dict[str, str]] = {}
    for lang in manifest.languages:
        if lang == reference_lang:
            realised[lang] = 0.0
            continue
        other = np.array([manifest.rows[sid][lang]["n_tokens"] for sid in ids], dtype=float)
        cost = np.abs(reference[:, None] - other[None, :])
        rows_idx, cols_idx = linear_sum_assignment(cost)
        worst = 0.0
        lang_matches: Dict[str, str] = {}
        for r, c in zip(rows_idx, cols_idx):
            denom = max(reference[r], 1.0)
            rel = abs(reference[r] - other[c]) / denom
            lang_matches[ids[r]] = ids[c]
            if rel > max_rel_diff:
                keep.discard(ids[r])
            else:
                worst = max(worst, rel)
        realised[lang] = float(worst)
        matches[lang] = lang_matches

    chosen = tuple(sid for sid in ids if sid in keep)
    from dataclasses import replace
    subset = replace(
        manifest,
        manifest_id=f"{manifest.manifest_id}_lenmatched",
        semantic_ids=chosen,
        rows={sid: manifest.rows[sid] for sid in chosen},
        partitions={},
        provenance={**manifest.provenance,
                    "length_matched": {
                        "reference_lang": reference_lang,
                        "max_rel_diff": max_rel_diff,
                        "realised_max_rel_diff": realised,
                        "matches": matches,
                        "n_before": len(ids), "n_after": len(chosen)}},
        sha256="")
    return _with_hash(subset)


# ═══════════════════════════════════════════════════════════════════════════════
# XNLI
# ═══════════════════════════════════════════════════════════════════════════════


def build_xnli_manifest(tokenizer, langs=XNLI_LANGS, *, split="test",
                        n: int = 600, seed: int = 42, balanced: bool = True,
                        min_tokens: int = 12, max_tokens: int = 160,
                        template: str = XNLI_PROMPT_TEMPLATE,
                        candidates: Sequence[str] = XNLI_CANDIDATES,
                        instruction_by_lang: Optional[Dict[str, str]] = None
                        ) -> ParallelManifest:
    """Join XNLI on an explicit id (never row index) and build the prompted rows.

    ``04`` §2: ``n`` aligned semantic ids, balanced across entailment / neutral /
    contradiction when ``balanced``, after tokenisation and length filtering. Each row
    carries the full prompt, its token ids, the gold label and the candidate token ids for
    length-normalised scoring.

    ``instruction_by_lang`` implements the ``--translated-instruction`` robustness switch:
    for the named languages the English task wording is replaced by the supplied template.
    The prompt-language confound is otherwise a deliberate constant (design §16.3), and
    which languages were translated is recorded in ``provenance``.
    """
    langs = tuple(langs)
    raw, info = _dl.load_xnli_aligned(langs, split=split)

    candidate_ids = {c: [int(t) for t in tokenizer(c, add_special_tokens=False)["input_ids"]]
                     for c in candidates}
    for text, ids in candidate_ids.items():
        if not ids:
            raise ValueError(f"Candidate {text!r} tokenised to zero tokens")

    shared = set(raw[langs[0]])
    for lang in langs[1:]:
        shared &= set(raw[lang])

    instruction_by_lang = instruction_by_lang or {}
    frequencies = {lang: _first_token_frequencies(
        tokenizer, (row["premise"] for row in raw[lang].values())) for lang in langs}

    kept: Dict[str, Dict[str, dict]] = {}
    labels: Dict[str, int] = {}
    for key in sorted(shared):
        gold = {raw[lang][key]["label"] for lang in langs}
        if len(gold) != 1:
            # A genuine alignment failure: the same semantic id disagreeing on the gold
            # label across languages means the join is wrong. Skip and record, never
            # silently pick one.
            continue
        label = gold.pop()
        if not 0 <= label < len(candidates):
            continue

        per_lang: Dict[str, dict] = {}
        ok = True
        for lang in langs:
            row = raw[lang][key]
            tmpl = instruction_by_lang.get(lang, template)
            prompt = tmpl.format(premise=row["premise"], hypothesis=row["hypothesis"])
            ids = [int(t) for t in tokenizer(prompt, add_special_tokens=False)["input_ids"]]
            if not min_tokens <= len(ids) <= max_tokens:
                ok = False
                break
            premise_first = tokenizer(row["premise"],
                                      add_special_tokens=False)["input_ids"]
            first = int(premise_first[0]) if len(premise_first) else -1
            per_lang[lang] = {
                "language": lang,
                "premise": row["premise"],
                "hypothesis": row["hypothesis"],
                "text": prompt,
                "input_ids": ids,
                "n_tokens": len(ids),
                "gold_label": int(label),
                "gold_label_name": _dl.XNLI_LABEL_NAMES[label],
                "candidate_token_ids": [candidate_ids[c] for c in candidates],
                "candidates": list(candidates),
                "translated_instruction": lang in instruction_by_lang,
                "first_token_id": first,
                "first_token_frequency": int(frequencies[lang].get(first, 0)),
                "punctuation_at_position_0": starts_with_punctuation(row["premise"]),
                "script": unicode_script(row["premise"]),
                "language_family": LANGUAGE_FAMILY.get(lang, "unknown"),
                "source_index": int(row["source_index"]),
            }
        if ok:
            semantic_id = f"xnli_{split}_{key}"
            kept[semantic_id] = per_lang
            labels[semantic_id] = int(label)

    chosen = _select_balanced(sorted(kept), labels, n=n, seed=seed, balanced=balanced)

    manifest = ParallelManifest(
        manifest_id=f"xnli_{split}_{len(chosen)}",
        dataset="xnli", split=split, languages=langs,
        semantic_ids=tuple(chosen),
        rows={sid: kept[sid] for sid in chosen},
        partitions={}, seed=seed,
        provenance=_prov.provenance_block(extra={
            "loader": info,
            "min_tokens": min_tokens, "max_tokens": max_tokens,
            "requested_n": n, "realised_n": len(chosen),
            "n_survivors": len(kept), "balanced": balanced,
            "label_counts": dict(Counter(labels[sid] for sid in chosen)),
            "template": template,
            "translated_instruction_langs": sorted(instruction_by_lang),
            "candidates": list(candidates),
            "candidate_token_ids": {c: candidate_ids[c] for c in candidates},
            "tokenizer": getattr(tokenizer, "name_or_path", "unknown"),
            "valid": len(chosen) >= min(n, 200),
            "schema": MANIFEST_SCHEMA_VERSION,
        }))
    return _with_hash(manifest)


def _select_balanced(ids: Sequence[str], labels: Dict[str, int], *, n: int,
                     seed: int, balanced: bool) -> List[str]:
    """Sample ``n`` ids, class-balanced when asked, deterministically from ``seed``."""
    rng = random.Random(seed)
    if not balanced:
        pool = list(ids)
        rng.shuffle(pool)
        return sorted(pool[:n])

    by_label: Dict[int, List[str]] = {}
    for sid in ids:
        by_label.setdefault(labels[sid], []).append(sid)
    if not by_label:
        return []
    per_class = n // len(by_label)
    chosen: List[str] = []
    for label in sorted(by_label):
        pool = sorted(by_label[label])
        rng.shuffle(pool)
        chosen.extend(pool[:per_class])
    # Any remainder from n not dividing evenly is filled from what is left, in a
    # deterministic order, so the realised count is as close to n as the data allows.
    if len(chosen) < n:
        remaining = sorted(set(ids) - set(chosen))
        rng.shuffle(remaining)
        chosen.extend(remaining[:n - len(chosen)])
    return sorted(chosen)


# ═══════════════════════════════════════════════════════════════════════════════
# Patch controls (02 §5.1)
# ═══════════════════════════════════════════════════════════════════════════════


def assign_patch_controls(manifest: ParallelManifest, seed: int = 42,
                          source_lang: str = "en",
                          label_key: str = "gold_label"):
    """Assign the four English source conditions for every (semantic_id, target_lang).

    Constraints, all asserted by ``tests/test_patch_controls.py``:

    * ``parallel_en``        — the target's own semantic id, in English (the treatment);
    * ``same_label_en``      — a *different* id sharing the gold label;
    * ``different_label_en`` — an id with a different gold label;
    * ``random_en``          — drawn independently of label, and never the id itself;
    * no id is ever its own control (``parallel_en`` excepted — that is the treatment).

    Deterministic from ``seed``. Returns a ``pandas.DataFrame`` with one row per
    ``(semantic_id, target_lang, control_condition)``.
    """
    import pandas as pd

    if source_lang not in manifest.languages:
        raise ValueError(f"source_lang {source_lang!r} is not in {manifest.languages}")
    ids = list(manifest.semantic_ids)
    labels = {sid: int(manifest.rows[sid][source_lang][label_key]) for sid in ids}
    by_label: Dict[int, List[str]] = {}
    for sid in ids:
        by_label.setdefault(labels[sid], []).append(sid)
    for label in by_label:
        by_label[label].sort()

    targets = [lang for lang in manifest.languages if lang != source_lang]
    records: List[dict] = []
    for sid in ids:
        for lang in targets:
            # A per-(id, lang) stream so adding a language cannot shift another
            # language's assignments — the manifest must be extendable without
            # invalidating a completed run.
            rng = random.Random(f"{seed}|{manifest.manifest_id}|{sid}|{lang}")
            label = labels[sid]

            same_pool = [s for s in by_label.get(label, []) if s != sid]
            diff_pool = sorted(s for s in ids if labels[s] != label)
            random_pool = [s for s in ids if s != sid]

            assignments = {
                "parallel_en": sid,
                "same_label_en": rng.choice(same_pool) if same_pool else None,
                "different_label_en": rng.choice(diff_pool) if diff_pool else None,
                "random_en": rng.choice(random_pool) if random_pool else None,
            }
            for condition in PATCH_CONTROL_CONDITIONS:
                source = assignments[condition]
                records.append({
                    "manifest_id": manifest.manifest_id,
                    "semantic_id": sid,
                    "target_language": lang,
                    "gold_label": label,
                    "control_condition": condition,
                    "source_semantic_id": source,
                    "source_language": source_lang,
                    "source_label": labels[source] if source is not None else None,
                    "status": "ok" if source is not None else "unassignable",
                    "warning": ("" if source is not None
                                else f"no candidate available for {condition}"),
                    "seed": seed,
                })
    return pd.DataFrame.from_records(records)


# ═══════════════════════════════════════════════════════════════════════════════
# Partitions (02 §5.1)
# ═══════════════════════════════════════════════════════════════════════════════


def grouped_partition(manifest: ParallelManifest,
                      fractions: Sequence[float] = (0.0, 1 / 3, 1.0),
                      names: Sequence[str] = ("dev", "test"),
                      seed: int = 42) -> Dict[str, Tuple[str, ...]]:
    """Partition **by semantic id**, so no sentence appears in two partitions in any language.

    ``fractions`` are cumulative cut points over a seeded shuffle of the semantic ids;
    ``len(fractions) == len(names) + 1``. This one function serves both E7 splits: the
    200/400 dev/test split (``fractions=(0, 1/3, 1)`` over 600 ids) and the 100/200
    Procrustes split (``fractions=(0, 1/3, 1)`` over 300).

    Splitting by row rather than by semantic id is the classic leak here — the same
    sentence in another language would land on the other side of the split and the
    "held-out" set would not be held out at all.
    """
    if len(fractions) != len(names) + 1:
        raise ValueError(f"{len(fractions)} cut points cannot define {len(names)} "
                         "partitions; expected len(names) + 1")
    if list(fractions) != sorted(fractions):
        raise ValueError("fractions must be non-decreasing")
    if not (abs(fractions[0]) < 1e-12 and abs(fractions[-1] - 1.0) < 1e-12):
        raise ValueError("fractions must start at 0.0 and end at 1.0")

    ids = list(manifest.semantic_ids)
    random.Random(seed).shuffle(ids)
    total = len(ids)
    bounds = [int(round(f * total)) for f in fractions]
    out: Dict[str, Tuple[str, ...]] = {}
    for i, name in enumerate(names):
        out[name] = tuple(sorted(ids[bounds[i]:bounds[i + 1]]))
    return out


def with_partitions(manifest: ParallelManifest,
                    partitions: Dict[str, Tuple[str, ...]]) -> ParallelManifest:
    """Attach partitions and recompute the hash (partitions are part of the identity)."""
    from dataclasses import replace
    return _with_hash(replace(manifest, partitions=dict(partitions), sha256=""))


# ═══════════════════════════════════════════════════════════════════════════════
# Verification (02 §5.1)
# ═══════════════════════════════════════════════════════════════════════════════


def verify_manifest(manifest: ParallelManifest, *, min_tokens: Optional[int] = None,
                    max_tokens: Optional[int] = None) -> Dict[str, Any]:
    """Assert every structural property of a manifest. Raises on the first failure.

    Checks: no duplicate semantic ids; every id present in every language; token counts
    within the declared bounds; for XNLI, the gold label agreeing across all languages;
    partitions disjoint and drawn from the manifest's own ids; and the stored sha256 equal
    to a recomputation.
    """
    problems: List[str] = []
    ids = list(manifest.semantic_ids)

    if len(set(ids)) != len(ids):
        duplicates = [sid for sid, c in Counter(ids).items() if c > 1]
        problems.append(f"duplicate semantic ids: {duplicates[:5]}")

    if min_tokens is None:
        min_tokens = manifest.provenance.get("min_tokens")
    if max_tokens is None:
        max_tokens = manifest.provenance.get("max_tokens")

    for sid in ids:
        per_lang = manifest.rows.get(sid, {})
        missing = [lang for lang in manifest.languages if lang not in per_lang]
        if missing:
            problems.append(f"{sid} missing languages {missing}")
            continue
        if manifest.dataset == "xnli":
            gold = {per_lang[lang]["gold_label"] for lang in manifest.languages}
            if len(gold) != 1:
                problems.append(
                    f"{sid} has disagreeing gold labels across languages {sorted(gold)} — "
                    "this is what a positional join looks like")
        if min_tokens is not None and max_tokens is not None:
            for lang in manifest.languages:
                n_tokens = int(per_lang[lang]["n_tokens"])
                if not min_tokens <= n_tokens <= max_tokens:
                    problems.append(f"{sid}/{lang} has {n_tokens} tokens, outside "
                                    f"[{min_tokens}, {max_tokens}]")

    known = set(ids)
    seen: Dict[str, str] = {}
    for name, members in manifest.partitions.items():
        for sid in members:
            if sid not in known:
                problems.append(f"partition {name!r} contains unknown id {sid!r}")
            if sid in seen:
                problems.append(f"{sid} is in both {seen[sid]!r} and {name!r}")
            seen[sid] = name

    recomputed = compute_manifest_sha256(manifest)
    if manifest.sha256 and recomputed != manifest.sha256:
        problems.append(f"sha256 mismatch: stored {manifest.sha256[:16]}…, "
                        f"recomputed {recomputed[:16]}…")

    if problems:
        raise ValueError("Manifest verification failed:\n  - " + "\n  - ".join(problems))

    return {
        "manifest_id": manifest.manifest_id,
        "dataset": manifest.dataset,
        "n_semantic_ids": len(ids),
        "languages": list(manifest.languages),
        "partitions": {k: len(v) for k, v in manifest.partitions.items()},
        "token_range": {lang: [min(manifest.token_counts(lang)),
                               max(manifest.token_counts(lang))]
                        for lang in manifest.languages} if ids else {},
        "sha256": recomputed,
        "valid": bool(manifest.provenance.get("valid", True)),
    }
