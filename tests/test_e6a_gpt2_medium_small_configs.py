# -*- coding: utf-8 -*-
"""Contracts for the canonical GPT-2-medium -> GPT-2-small E6A extension."""

from __future__ import annotations

import hashlib
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import aggregate_transformation as ag  # noqa: E402
import check_pilot_gate as gate  # noqa: E402
import evaluate_transformation as ev  # noqa: E402
import train_distillation as td  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
PRIMARY = "e6a_gpt2_medium_small"
BRIDGE = "e6a_gpt2_alignment_bridge"
MEDIUM_REVISION = "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"
SMALL_REVISION = "607a30d783dfa663caf39e06633721c8d4cfcd7e"
DISTIL_REVISION = "2290a62682d06624634c1f46a6ad5be0f47f38aa"

PRIMARY_FILES = {
    "G0": "e6a_gpt2_medium_small_ce.yaml",
    "G1": "e6a_gpt2_medium_small_logit_kd.yaml",
    "G2-aligned": "e6a_gpt2_medium_small_logit_attention_kd_aligned.yaml",
}
BRIDGE_FILES = {
    "G0": "e6a_gpt2_alignment_bridge_ce.yaml",
    "G1": "e6a_gpt2_alignment_bridge_logit_kd.yaml",
    "G2-legacy": "e6a_gpt2_alignment_bridge_logit_attention_kd_legacy.yaml",
    "G2-aligned": "e6a_gpt2_alignment_bridge_logit_attention_kd_aligned.yaml",
}
WEIGHTS = {
    "G0": (1.0, 0.0, 0.0),
    "G1": (0.50, 0.50, 0.0),
    "G2-legacy": (0.45, 0.45, 0.10),
    "G2-aligned": (0.45, 0.45, 0.10),
}


def _load(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _without_condition_and_weights(config):
    payload = deepcopy(config)
    payload.pop("condition")
    for key in ("ce_weight", "kd_weight", "attn_weight"):
        payload["loss"].pop(key)
    return payload


def test_primary_configs_are_isolated_pinned_and_use_the_canonical_models():
    for condition, name in PRIMARY_FILES.items():
        config = _load(name)
        assert config["condition"] == condition
        assert config["experiment_id"] == PRIMARY
        assert config["experiment_family"] == "e6a"
        assert config["teacher"] == "openai-community/gpt2-medium"
        assert config["teacher_revision"] == MEDIUM_REVISION
        assert config["student_config_from"] == config["public_reference"] == (
            "openai-community/gpt2")
        assert config["student_config_revision"] == SMALL_REVISION
        assert config["public_reference_revision"] == SMALL_REVISION
        assert config["tokenizer_revision"] == SMALL_REVISION
        assert config["student_init"] == "random"


def test_primary_conditions_differ_only_in_condition_and_registered_weights():
    configs = {condition: _load(name) for condition, name in PRIMARY_FILES.items()}
    baselines = [_without_condition_and_weights(configs[c]) for c in sorted(configs)]
    assert all(value == baselines[0] for value in baselines)
    for condition, config in configs.items():
        loss = config["loss"]
        assert (loss["ce_weight"], loss["kd_weight"], loss["attn_weight"]) == (
            WEIGHTS[condition])
        assert td.resolve_head_alignment(loss)["method"] == td.AMAD_JSD_HEAD_ALIGNMENT


def test_g0_and_g1_keep_the_existing_training_and_evaluation_settings():
    for condition, old_name in (("G0", "e6a_gpt2_ce.yaml"),
                                ("G1", "e6a_gpt2_logit_kd.yaml")):
        new = _load(PRIMARY_FILES[condition])
        old = _load(old_name)
        for key in ("data", "optim", "checkpoints", "seeds", "eval"):
            assert new[key] == old[key], key
        for key in ("ce_weight", "kd_weight", "attn_weight", "temperature"):
            assert new["loss"][key] == old["loss"][key], key


def test_primary_layer_map_is_the_every_second_24_to_12_last_to_last_map():
    mapping = _load(PRIMARY_FILES["G0"])["loss"]["layer_map"]
    assert list(mapping) == list(range(1, 24, 2))
    assert list(mapping.values()) == list(range(12))
    assert mapping[23] == 11


def test_model_contract_records_64_dimensional_heads_on_both_sides():
    contract = _load(PRIMARY_FILES["G0"])["model_contract"]
    assert contract["teacher"] == {
        "num_layers": 24, "num_heads": 16, "hidden_size": 1024}
    assert contract["student"] == {
        "num_layers": 12, "num_heads": 12, "hidden_size": 768}
    assert contract["teacher"]["hidden_size"] // contract["teacher"]["num_heads"] == 64
    assert contract["student"]["hidden_size"] // contract["student"]["num_heads"] == 64


def test_model_contract_refuses_missing_roles_or_geometry_fields():
    config = SimpleNamespace(num_layers=2, num_heads=4, hidden_size=32)
    complete = {
        role: {"num_layers": 2, "num_heads": 4, "hidden_size": 32}
        for role in ("teacher", "student", "public_reference")
    }
    assert td.model_contract_assertions(config, config, config, complete)["passed"]

    missing_role = deepcopy(complete)
    missing_role.pop("public_reference")
    with pytest.raises(ValueError, match="missing roles"):
        td.model_contract_assertions(config, config, config, missing_role)

    missing_field = deepcopy(complete)
    missing_field["student"].pop("hidden_size")
    with pytest.raises(ValueError, match="missing fields"):
        td.model_contract_assertions(config, config, config, missing_field)


def test_unequal_heads_are_allowed_only_by_the_explicit_aligned_method():
    class Tok:
        vocab_size = 100
        bos_token_id = 1
        eos_token_id = 2

        def __len__(self):
            return 100

    def cfg(heads):
        return SimpleNamespace(vocab_size=100, bos_token_id=1, eos_token_id=2,
                               num_heads=heads, num_layers=2)

    legacy = td.vocab_assertions(Tok(), cfg(16), cfg(12), cfg(12))
    aligned = td.vocab_assertions(
        Tok(), cfg(16), cfg(12), cfg(12),
        td.resolve_head_alignment({"head_alignment": {"method": "amad_jsd"}}))
    assert legacy["checks"]["heads"] is False and legacy["passed"] is False
    assert aligned["checks"]["heads"] is True and aligned["passed"] is True


def test_alignment_configuration_is_fixed_and_typos_are_refused():
    resolved = td.resolve_head_alignment(_load(PRIMARY_FILES["G2-aligned"])["loss"])
    assert resolved == {
        "method": "amad_jsd",
        "version": td.HEAD_ALIGNMENT_VERSION,
        "similarity": "cosine",
        "softmax_temperature": 1.0,
        "direction": "teacher_to_student",
        "divergence": "jsd",
        "alignment_weights_gradient": "attached",
    }
    with pytest.raises(ValueError, match="unknown"):
        td.resolve_head_alignment({"head_alignment": {"method": "amad-jds"}})
    with pytest.raises(ValueError, match="unsupported"):
        td.resolve_head_alignment(
            {"head_alignment": {"method": "amad_jsd", "divergence": "mse"}})


def test_bridge_directly_pairs_legacy_and_aligned_g2():
    legacy = _load(BRIDGE_FILES["G2-legacy"])
    aligned = _load(BRIDGE_FILES["G2-aligned"])
    left, right = deepcopy(legacy), deepcopy(aligned)
    left.pop("condition")
    right.pop("condition")
    left["loss"].pop("head_alignment")
    right["loss"].pop("head_alignment")
    assert left == right
    assert td.resolve_head_alignment(legacy["loss"])["method"] == "index"
    assert td.resolve_head_alignment(aligned["loss"])["method"] == "amad_jsd"
    assert legacy["loss"]["attn_weight"] == aligned["loss"]["attn_weight"] == 0.10


def test_bridge_g0_g1_keep_the_existing_training_objectives():
    for condition, old_name in (("G0", "e6a_gpt2_ce.yaml"),
                                ("G1", "e6a_gpt2_logit_kd.yaml")):
        bridge = _load(BRIDGE_FILES[condition])
        old = _load(old_name)
        for key in ("data", "optim", "checkpoints", "seeds", "eval"):
            assert bridge[key] == old[key]
        assert tuple(bridge["loss"][k] for k in
                     ("ce_weight", "kd_weight", "attn_weight")) == WEIGHTS[condition]
        assert td.resolve_head_alignment(bridge["loss"])["method"] == "index"


def test_bridge_legacy_g2_keeps_the_original_attention_objective_and_settings():
    bridge = _load(BRIDGE_FILES["G2-legacy"])
    old = _load("e6a_gpt2_logit_attention_kd.yaml")
    for key in ("data", "optim", "checkpoints", "seeds", "eval"):
        assert bridge[key] == old[key]
    for key in ("ce_weight", "kd_weight", "attn_weight", "temperature", "layer_map"):
        assert bridge["loss"][key] == old["loss"][key]
    assert td.resolve_head_alignment(bridge["loss"])["method"] == "index"
    assert td.resolve_head_alignment(old["loss"])["method"] == "index"


@pytest.mark.parametrize(
    "experiment,prereg_name,files,reference",
    [
        (PRIMARY, "e6a_gpt2_medium_small_preregistration.yaml", PRIMARY_FILES, "PG2S"),
        (BRIDGE, "e6a_gpt2_alignment_bridge_preregistration.yaml", BRIDGE_FILES, "PDG2"),
    ],
)
def test_preregistrations_are_prospective_consumable_and_match_gate(
        experiment, prereg_name, files, reference):
    path = CONFIG_DIR / prereg_name
    prereg = ag.load_preregistration(path)
    assert prereg["experiment_id"] == experiment
    assert set(prereg["conditions"]) - {reference} == set(files)
    assert gate.ARMS[experiment]["conditions"] == tuple(files)
    assert gate.ARMS[experiment]["corpus_id"] == prereg["pilot_gate"]["corpus_id"]
    assert gate.ARMS[experiment]["teacher_relative_gate"] is True
    assert prereg["pilot_gate"]["teacher_sink_fraction"] == gate.TEACHER_SINK_FRACTION
    assert not ag.collect_pending_decisions(prereg)


def test_both_preregistrations_pin_the_unchanged_source_file_hash():
    source = CONFIG_DIR / "e6a_gpt2_preregistration.yaml"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    for name in ("e6a_gpt2_medium_small_preregistration.yaml",
                 "e6a_gpt2_alignment_bridge_preregistration.yaml"):
        assert _load(name)["source_preregistration_sha256"] == digest


def test_standalone_public_reference_forwards_model_and_tokenizer_revisions(
        tmp_path, monkeypatch):
    captured = {}

    def fake(reference, out_dir, **kwargs):
        captured.update({"reference": reference, "out_dir": out_dir, **kwargs})
        return {"ok": True}

    monkeypatch.setattr(ev, "evaluate_public_reference", fake)
    ev.reference_from_config(
        CONFIG_DIR / PRIMARY_FILES["G0"], tmp_path / "reference",
        condition="PG2S")
    assert captured["reference"] == "openai-community/gpt2"
    assert captured["revision"] == SMALL_REVISION
    assert captured["tokenizer_revision"] == SMALL_REVISION
    assert captured["teacher_revision"] == MEDIUM_REVISION


def test_aligned_smoke_preserves_registered_depths_and_heads_and_backpropagates(tmp_path):
    """Production smoke path: exact 24/16 -> 12/12 geometry, scaled widths, no network."""
    config = deepcopy(td.load_config(CONFIG_DIR / PRIMARY_FILES["G2-aligned"]))
    config["data"]["block_size"] = 16
    setup = td.prepare_run(
        config, seed=0, output_dir=tmp_path, max_steps=1, smoke=True)
    assert (td._num_layers(setup.teacher_config),
            td._num_heads(setup.teacher_config)) == (24, 16)
    assert (td._num_layers(setup.student_config),
            td._num_heads(setup.student_config)) == (12, 12)
    assert setup.head_alignment["method"] == "amad_jsd"

    batch = setup.train_data.batch([0, 1], setup.device)
    loss, components = td.compute_losses(setup, batch, need_attention=True)
    loss.backward()
    assert torch.isfinite(loss)
    assert components["l_attn"] >= 0.0
    assert any(parameter.grad is not None for parameter in setup.student.parameters())
    assert all(parameter.grad is None for parameter in setup.teacher.parameters())
