"""Isolation and invariants for the GPT-2 large -> medium E6A arm."""

from __future__ import annotations

import copy
from pathlib import Path

import yaml

from transformation_inheritance import check_pilot_gate as gate
from transformation_inheritance import evaluate_transformation as evaluate


REPO = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
FILES = {
    "G0": "e6a_gpt2_large_medium_ce.yaml",
    "G1": "e6a_gpt2_large_medium_logit_kd.yaml",
    "G2": "e6a_gpt2_large_medium_logit_attention_kd.yaml",
}


def _load(name: str):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


def _without_condition_and_weights(config):
    payload = copy.deepcopy(config)
    payload.pop("condition")
    for key in ("ce_weight", "kd_weight", "attn_weight"):
        payload["loss"].pop(key)
    return payload


def test_the_three_conditions_are_isolated_and_differ_only_in_the_registered_loss():
    configs = {condition: _load(path) for condition, path in FILES.items()}
    assert {cfg["experiment_id"] for cfg in configs.values()} == {
        "e6a_gpt2_large_medium"}
    assert {condition: cfg["condition"] for condition, cfg in configs.items()} == {
        "G0": "G0", "G1": "G1", "G2": "G2"}
    common = [_without_condition_and_weights(config) for config in configs.values()]
    assert common[1:] == common[:-1]
    assert [configs[c]["loss"]["ce_weight"] for c in configs] == [1.0, 0.5, 0.45]
    assert [configs[c]["loss"]["kd_weight"] for c in configs] == [0.0, 0.5, 0.45]
    assert [configs[c]["loss"]["attn_weight"] for c in configs] == [0.0, 0.0, 0.10]


def test_model_identity_initialisation_and_revisions_are_frozen():
    config = _load(FILES["G0"])
    assert config["teacher"] == "gpt2-large"
    assert config["student_config_from"] == "gpt2-medium"
    assert config["student_init"] == "random"
    assert config["public_reference"] == "gpt2-medium"
    assert config["tokenizer"] == "gpt2"
    for key in ("teacher_revision", "student_config_revision",
                "public_reference_revision", "tokenizer_revision"):
        assert len(config[key]) == 40


def test_layer_and_head_alignment_are_explicit_36_to_24_invariants():
    loss = _load(FILES["G2"])["loss"]
    layer_map = loss["layer_map"]
    assert loss["attention_head_alignment"] == "mean"
    assert sorted(layer_map.values()) == list(range(24))
    assert sorted(layer_map) == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17,
                                 19, 20, 22, 23, 25, 26, 28, 29, 31, 32, 34, 35]
    assert max(layer_map) == 35


def test_data_optimisation_and_checkpoints_match_the_existing_gpt2_arm():
    new = _load(FILES["G0"])
    old = _load("e6a_gpt2_ce.yaml")
    comparable_data_keys = ("dataset", "train_split", "eval_split", "block_size",
                            "eos_between", "train_documents", "validation_documents")
    assert {key: new["data"][key] for key in comparable_data_keys} == {
        key: old["data"][key] for key in comparable_data_keys}
    assert new["data"]["corpus_id"] == "openwebtext_train"
    assert new["optim"] == old["optim"]
    assert new["checkpoints"] == old["checkpoints"]
    assert new["seeds"] == [0, 1, 2]


def test_preregistration_and_gate_name_the_new_arm_without_moving_thresholds():
    prereg = _load("e6a_gpt2_large_medium_preregistration.yaml")
    assert prereg["experiment_id"] == "e6a_gpt2_large_medium"
    assert prereg["conditions"]["PGM"].startswith("gpt2-medium")
    assert prereg["pilot_gate"]["sink_threshold"] == 0.15
    assert prereg["pilot_gate"]["mechanistic_delta_threshold"] == 0.10
    assert prereg["pilot_gate"]["teacher_sink_fraction"] == 0.50
    assert prereg["analysis"]["matched_loss_max_gap_nats"] == 0.05
    assert prereg["analysis"]["max_failure_rate"] == 0.02
    assert gate.ARMS["e6a_gpt2_large_medium"]["conditions"] == ("G0", "G1", "G2")
    assert gate.ARMS["e6a_gpt2_large_medium"]["corpus_id"] == (
        "openwebtext_validation_sink_300")


def test_existing_gpt2_arm_keeps_strict_head_alignment_by_default():
    for path in FILES.values():
        assert _load(path)["loss"]["attention_head_alignment"] == "mean"
    for path in ("e6a_gpt2_ce.yaml", "e6a_gpt2_logit_kd.yaml",
                 "e6a_gpt2_logit_attention_kd.yaml"):
        assert "attention_head_alignment" not in _load(path)["loss"]


def test_public_reference_revision_is_pinned_but_a_model_override_does_not_reuse_it(
        tmp_path, monkeypatch):
    captured = []

    def fake_reference(model, out, **kwargs):
        captured.append((model, kwargs["revision"]))
        return {"model": model, "out": str(out)}

    monkeypatch.setattr(evaluate, "evaluate_public_reference", fake_reference)
    config = CONFIG_DIR / FILES["G0"]
    evaluate.reference_from_config(config, tmp_path / "medium")
    evaluate.reference_from_config(config, tmp_path / "teacher", model_id="gpt2-large")
    evaluate.reference_from_config(
        config, tmp_path / "teacher-pinned", model_id="gpt2-large",
        revision="teacher-commit")

    assert captured == [
        ("gpt2-medium", "6dcaa7a952f72f9298047fd5137cd6e4f05f41da"),
        ("gpt2-large", None),
        ("gpt2-large", "teacher-commit"),
    ]
