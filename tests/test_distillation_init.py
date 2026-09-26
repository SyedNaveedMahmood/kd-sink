# -*- coding: utf-8 -*-
"""test_distillation_init.py — D0/D1/D2 start from identical weights and data (WP5).

``06_TEST_PLAN.md`` row 26. The E6A design pairs conditions **by seed and data order**, so
the whole D0-vs-D2 contrast rests on three runs at one seed differing in nothing but their
loss weights. That has to be provable, not assumed: this file asserts a byte-level sha256
of the initial state dict and of the block manifest.

It also pins the two startup guards that would otherwise fail silently: the student must be
randomly initialised from a config (never ``from_pretrained``), and a vocab/head mismatch
must abort rather than train.
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(REPO / "transformation_inheritance"))

import provenance as prov  # noqa: E402
import train_distillation as td  # noqa: E402

CONDITIONS = {
    "D0": {"ce_weight": 1.0, "kd_weight": 0.0, "attn_weight": 0.0},
    "D1": {"ce_weight": 0.5, "kd_weight": 0.5, "attn_weight": 0.0},
    "D2": {"ce_weight": 0.45, "kd_weight": 0.45, "attn_weight": 0.10},
}


def _config(condition):
    weights = dict(CONDITIONS[condition])
    weights.update({"temperature": 2.0, "layer_map": td.DEFAULT_LAYER_MAP})
    return {"condition": condition, "loss": weights,
            "data": {"block_size": 32}, "optim": {"max_steps": 1}}


def _setup(root, condition, seed):
    return td.prepare_run(_config(condition), seed=seed, output_dir=root,
                          max_steps=1, smoke=True)


def test_conditions_share_byte_identical_initial_weights_and_data():
    with tempfile.TemporaryDirectory(prefix="e6a_init_", ignore_cleanup_errors=True) as t:
        root = Path(t)
        setups = {c: _setup(root, c, seed=0) for c in CONDITIONS}

        shas = {c: s.initial_state_sha256 for c, s in setups.items()}
        assert len(set(shas.values())) == 1, shas

        manifests = {c: s.train_data.manifest_sha256 for c, s in setups.items()}
        assert len(set(manifests.values())) == 1, manifests

        # Byte-level, not just hash-level: compare the tensors themselves.
        reference = setups["D0"].student.state_dict()
        for condition in ("D1", "D2"):
            other = setups[condition].student.state_dict()
            assert set(other) == set(reference)
            for name in reference:
                assert torch.equal(reference[name], other[name]), (condition, name)

        # The weights differ between conditions, which is the only thing that may.
        assert setups["D0"].weights != setups["D2"].weights


def test_a_different_seed_gives_different_weights_and_data():
    with tempfile.TemporaryDirectory(prefix="e6a_seed_", ignore_cleanup_errors=True) as t:
        root = Path(t)
        a = _setup(root, "D2", seed=0)
        b = _setup(root, "D2", seed=1)
        assert a.initial_state_sha256 != b.initial_state_sha256
        assert a.train_data.manifest_sha256 != b.train_data.manifest_sha256


def test_the_student_is_randomly_initialised_not_loaded():
    """A ``from_pretrained`` student would make the whole experiment meaningless.

    Checked structurally (two seeds give different weights, so nothing fixed was loaded)
    and by inspecting ``build_student`` specifically — the teacher is legitimately loaded
    with ``from_pretrained`` elsewhere in the module, so a whole-module source scan would
    be both brittle and wrong.
    """
    import inspect

    # Strip the docstring first — it *names* from_pretrained in order to forbid it.
    student_source = inspect.getsource(td.build_student)
    student_code = student_source.replace(td.build_student.__doc__ or "", "")
    assert "from_config" in student_code
    assert "from_pretrained" not in student_code

    with tempfile.TemporaryDirectory(prefix="e6a_rand_", ignore_cleanup_errors=True) as t:
        root = Path(t)
        first = td.build_student(_setup(root, "D0", 0).student_config, seed=0)
        second = td.build_student(_setup(root, "D0", 0).student_config, seed=0)
        third = td.build_student(_setup(root, "D0", 0).student_config, seed=3)
        assert prov.sha256_state_dict(first.state_dict()) == \
            prov.sha256_state_dict(second.state_dict())
        assert prov.sha256_state_dict(first.state_dict()) != \
            prov.sha256_state_dict(third.state_dict())


def test_run_config_records_the_hashes_and_the_normalised_band():
    with tempfile.TemporaryDirectory(prefix="e6a_cfg_", ignore_cleanup_errors=True) as t:
        import json

        root = Path(t)
        setup = _setup(root, "D2", seed=0)
        manifest_path = td.write_block_manifest(
            setup.train_data, setup.run_dir / "block_manifest.parquet")
        path = td.write_run_config(setup, manifest_path)
        payload = json.loads(path.read_text(encoding="utf-8"))

        assert payload["initial_state_sha256"] == setup.initial_state_sha256
        assert payload["manifest_sha256"] == setup.train_data.manifest_sha256
        assert payload["block_manifest_sha256"]
        assert payload["git_sha"]
        assert payload["vocab_assertions"] == setup.vocab_report["checks"]

        # CLAUDE.md trap 1: the normalised band, not compute_band. The 8-layer student
        # gets a multi-layer band ([2, 8) — the value 05 §1's example records) and the
        # 4-layer teacher's band is recorded separately rather than reused.
        assert payload["layer_band"] == [2, 8]
        assert payload["layer_band"][1] - payload["layer_band"][0] > 1
        assert payload["num_layers"] == 8
        assert payload["teacher_layer_band"][1] > payload["teacher_layer_band"][0]
        assert payload["layer_band_version"] == "depth_band_v1"


def test_vocab_assertions_catch_a_mismatch():
    """``03`` §1.2 / design-delta D6: abort on mismatch, never train through it."""
    class _Cfg:
        def __init__(self, vocab_size, bos, eos, heads):
            self.vocab_size = vocab_size
            self.bos_token_id = bos
            self.eos_token_id = eos
            self.num_heads = heads
            self.attention_layers = ["global"] * 4
            self.window_size = 128

    class _Tok:
        vocab_size = 100
        bos_token_id = 1
        eos_token_id = 2

        def __len__(self):
            return 100

    good = vars()  # keep flake quiet about unused
    teacher = _Cfg(100, 1, 2, 16)
    student = _Cfg(100, 1, 2, 16)
    report = td.vocab_assertions(_Tok(), teacher, student, student)
    assert report["passed"], report

    wrong_vocab = _Cfg(99, 1, 2, 16)
    assert not td.vocab_assertions(_Tok(), wrong_vocab, student, student)["passed"]

    wrong_eos = _Cfg(100, 1, 3, 16)
    report = td.vocab_assertions(_Tok(), wrong_eos, student, student)
    assert report["checks"]["eos"] is False and not report["passed"]

    wrong_heads = _Cfg(100, 1, 2, 8)
    report = td.vocab_assertions(_Tok(), wrong_heads, student, student)
    assert report["checks"]["heads"] is False

    mean_report = td.vocab_assertions(
        _Tok(), wrong_heads, student, student, attention_head_alignment="mean")
    assert mean_report["passed"]
    assert mean_report["checks"]["heads"] is True
    assert mean_report["observed"]["head_counts_equal"] is False
    assert mean_report["observed"]["attention_head_alignment"] == "mean"

    # The observed values travel with the report so a failure is diagnosable.
    assert report["observed"]["teacher_heads"] == 8
    assert report["observed"]["student_heads"] == 16
    assert report["observed"]["teacher_attention_layers"] == ["global"] * 4
    assert good is not None


def test_prepare_run_aborts_on_a_failed_assertion(monkeypatch):
    def _bad(*_args, **_kwargs):
        return {"checks": {"vocab_size": False}, "observed": {}, "passed": False}

    monkeypatch.setattr(td, "vocab_assertions", _bad)
    with tempfile.TemporaryDirectory(prefix="e6a_abort_", ignore_cleanup_errors=True) as t:
        with pytest.raises(SystemExit, match="Startup assertions failed"):
            _setup(Path(t), "D2", seed=0)
