# -*- coding: utf-8 -*-
"""test_prepare_manifests_cli.py — the two WP8 prepare scripts run end-to-end, offline.

The manifest *logic* is covered by ``test_parallel_manifest_alignment.py``,
``test_patch_controls.py`` and ``test_grouped_partition.py``. This file covers the wiring
the CLI adds on top — argument handling, artefact layout, exit codes — because those are
what actually run on the compute PC and a wiring bug there costs a download-sized retry.

``datasets.load_dataset`` and ``AutoTokenizer.from_pretrained`` are both monkeypatched, so
nothing here touches the network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "crosslingual_semantics"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import datasets_loader as dl  # noqa: E402
import paired_manifests as pm  # noqa: E402

import prepare_flores_manifest as flores_cli  # noqa: E402
import prepare_xnli_manifest as xnli_cli  # noqa: E402
import wp8_fake_data as fake  # noqa: E402


@pytest.fixture()
def offline(monkeypatch):
    """Patch the dataset loader and both scripts' tokenizer loaders."""
    tokenizer = fake.tiny_tokenizer(None)

    def _fake_from_pretrained(name, **kwargs):
        tokenizer.name_or_path = "wp8_fake_tokenizer"
        return tokenizer

    import transformers
    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained",
                        staticmethod(_fake_from_pretrained))
    monkeypatch.setattr(flores_cli, "_load_tokenizer",
                        lambda *a, **k: _fake_from_pretrained("x"))
    return tokenizer


def test_flores_cli_writes_every_artefact(monkeypatch, offline, tmp_path):
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(flores=fake.make_flores(n_rows=320)))
    out = tmp_path / "flores_devtest.json"
    code = flores_cli.main(["--n", "240", "--seed", "42", "--out", str(out)])
    assert code == 0, "240 surviving rows is above the 200-row validity floor"

    manifest = pm.ParallelManifest.load(out)
    assert len(manifest) == 240
    assert set(manifest.partitions) == {"procrustes_train", "procrustes_test"}
    assert len(manifest.partitions["procrustes_train"]) == 80
    assert len(manifest.partitions["procrustes_test"]) == 160

    subset = out.with_name("flores_devtest_lenmatched.json")
    assert subset.exists()
    pm.ParallelManifest.load(subset)

    report = json.loads(out.with_name("flores_devtest_report.json").read_text("utf-8"))
    assert report["manifest"]["sha256"] == manifest.sha256
    assert report["provenance"]["git_sha"]


def test_flores_cli_exits_non_zero_on_a_thin_manifest(monkeypatch, offline, tmp_path):
    """``04`` §1: fewer than 200 surviving rows must not be proceeded on."""
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(flores=fake.make_flores(n_rows=60)))
    out = tmp_path / "thin.json"
    code = flores_cli.main(["--n", "300", "--seed", "42", "--out", str(out)])
    assert code == 1
    # The artefact is still written, so the failure is inspectable rather than invisible.
    manifest = pm.ParallelManifest.load(out)
    assert manifest.provenance["valid"] is False


def test_xnli_cli_writes_manifest_controls_and_partitions(monkeypatch, offline, tmp_path):
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=300)))
    out = tmp_path / "xnli_test.json"
    code = xnli_cli.main(["--n", "180", "--seed", "42", "--max-tokens", "200",
                          "--out", str(out)])
    assert code == 0

    manifest = pm.ParallelManifest.load(out)
    assert len(manifest) == 180
    assert manifest.provenance["loader"]["join_strategy"] == "promptID"
    assert len(manifest.partitions["dev"]) == 60
    assert len(manifest.partitions["test"]) == 120

    import pandas as pd
    controls = pd.read_csv(out.with_name("patch_controls.csv"))
    targets = [lang for lang in manifest.languages if lang != "en"]
    assert len(controls) == 180 * len(targets) * len(pm.PATCH_CONTROL_CONDITIONS)
    assert set(controls["control_condition"]) == set(pm.PATCH_CONTROL_CONDITIONS)

    partitions = json.loads(out.with_name("partitions.json").read_text("utf-8"))
    assert partitions["sizes"] == {"dev": 60, "test": 120}
    assert partitions["manifest_sha256"] == manifest.sha256

    report = json.loads(out.with_name("xnli_test_report.json").read_text("utf-8"))
    assert report["join_strategy"] == "promptID"
    assert report["patch_controls"]["unassignable"] == 0


def test_xnli_cli_validates_the_translated_instruction_switch(offline):
    """A translated template that drops the premise would move it off position 0."""
    with pytest.raises(ValueError, match=r"\{premise\}"):
        xnli_cli.parse_translated_instructions(["de=Frage: {hypothesis}\nAntwort:"])
    with pytest.raises(ValueError, match="LANG=TEMPLATE"):
        xnli_cli.parse_translated_instructions(["de"])

    parsed = xnli_cli.parse_translated_instructions(
        ["de={premise}\n{hypothesis}\nFrage:\nAntwort:"])
    assert set(parsed) == {"de"}


def test_xnli_cli_records_a_translated_instruction_run(monkeypatch, offline, tmp_path):
    monkeypatch.setattr(dl, "load_dataset",
                        fake.fake_load_dataset(xnli=fake.make_xnli(n_rows=120)))
    out = tmp_path / "xnli_translated.json"
    code = xnli_cli.main([
        "--n", "60", "--seed", "42", "--max-tokens", "200", "--out", str(out),
        "--translated-instruction",
        "de={premise}\n{hypothesis}\nFrage:\nAntwort:",
        "tr={premise}\n{hypothesis}\nSoru:\nCevap:",
    ])
    assert code == 0
    manifest = pm.ParallelManifest.load(out)
    assert manifest.provenance["translated_instruction_langs"] == ["de", "tr"]
    sid = manifest.semantic_ids[0]
    assert manifest.rows[sid]["de"]["translated_instruction"] is True
    assert manifest.rows[sid]["en"]["translated_instruction"] is False
    # The premise still begins at position 0 in the translated condition.
    assert manifest.rows[sid]["de"]["text"].startswith(manifest.rows[sid]["de"]["premise"])
