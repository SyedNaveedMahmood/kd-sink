# -*- coding: utf-8 -*-
"""test_e6a_gpt2_configs.py — the gpt2 arm's configs match its pre-registration.

Following ``test_effective_batch.py``: these read the **shipped** files rather than
hand-made dicts, because a config that drifts from what was registered is the realistic
failure and a test that builds its own input would never see it.

What is pinned:

* the three conditions carry design §1.6's loss weights and nothing else differs between
  them — the arms are paired by seed and data order, so a "tuned" hyperparameter in one
  file makes the G0/G1/G2 contrast uninterpretable;
* the optimiser block is identical to the TinyStories arm's, so the two arms differ in
  teacher, student and corpus and in nothing else;
* the layer map is a valid 12 -> 6 map;
* the corpus literal, the conditions and the two pilot-gate thresholds agree with
  ``e6a_gpt2_preregistration.yaml``, and the trainer's and evaluator's dataset registries
  agree with each other — the failure that registry mismatch would cause is training on one
  corpus and scoring on another, which nothing downstream would report.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

yaml = pytest.importorskip("yaml")

REPO = Path(__file__).resolve().parents[1]
for _path in (REPO, REPO / "common", REPO / "transformation_inheritance"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import check_pilot_gate as gate  # noqa: E402
import evaluate_transformation as ev  # noqa: E402
import train_distillation as td  # noqa: E402

CONFIG_DIR = REPO / "transformation_inheritance" / "configs"
PREREG = CONFIG_DIR / "e6a_gpt2_preregistration.yaml"

#: condition -> (ce, kd, attn), design §1.6's table with D->G.
REGISTERED_WEIGHTS = {
    "G0": (1.0, 0.0, 0.0),
    "G1": (0.50, 0.50, 0.0),
    "G2": (0.45, 0.45, 0.10),
}
CONFIG_BY_CONDITION = {
    "G0": "e6a_gpt2_ce.yaml",
    "G1": "e6a_gpt2_logit_kd.yaml",
    "G2": "e6a_gpt2_logit_attention_kd.yaml",
}


def _load(name):
    return yaml.safe_load((CONFIG_DIR / name).read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def prereg():
    return yaml.safe_load(PREREG.read_text(encoding="utf-8"))


@pytest.mark.parametrize("condition,name", sorted(CONFIG_BY_CONDITION.items()))
def test_each_config_carries_its_registered_loss_weights(condition, name):
    loss = _load(name)["loss"]
    ce, kd, attn = REGISTERED_WEIGHTS[condition]
    assert (loss["ce_weight"], loss["kd_weight"], loss["attn_weight"]) == (ce, kd, attn)
    assert loss["temperature"] == 2.0


def test_the_three_configs_differ_only_in_the_loss_weights():
    """The conditions are paired by seed and data order; anything else breaks the pairing."""
    loaded = {c: _load(n) for c, n in CONFIG_BY_CONDITION.items()}
    for key in ("teacher", "student_config_from", "student_init", "public_reference",
                "tokenizer", "data", "optim", "checkpoints", "seeds", "eval",
                "experiment_id"):
        values = [loaded[c][key] for c in sorted(loaded)]
        assert all(v == values[0] for v in values), f"{key} differs across the conditions"
    maps = [loaded[c]["loss"]["layer_map"] for c in sorted(loaded)]
    assert all(m == maps[0] for m in maps)


def test_the_optimiser_block_is_the_tinystories_arms_verbatim():
    """§1.6 is pre-registered and shared; a "tuned" lr would make the arms incomparable."""
    gpt2 = _load("e6a_gpt2_ce.yaml")["optim"]
    tinystories = _load("e6a_ce.yaml")["optim"]
    assert gpt2 == tinystories
    assert _load("e6a_gpt2_ce.yaml")["checkpoints"] == _load("e6a_ce.yaml")["checkpoints"]


def test_the_layer_map_is_a_valid_12_to_6_map():
    layer_map = _load("e6a_gpt2_ce.yaml")["loss"]["layer_map"]
    assert sorted(layer_map.values()) == list(range(6)), "must cover every student layer once"
    assert all(0 <= t < 12 for t in layer_map), "teacher indices must be inside 12 layers"
    assert max(layer_map) == 11, "the map must end on the teacher's last layer"
    assert sorted(layer_map) == [1, 3, 5, 7, 9, 11], "evenly spaced, per design §8.3's form"


def test_the_arm_uses_gpt2_and_a_randomly_initialised_distilgpt2():
    cfg = _load("e6a_gpt2_ce.yaml")
    assert cfg["experiment_id"] == "e6a_gpt2"
    assert cfg["teacher"] == "gpt2"
    assert cfg["tokenizer"] == "gpt2"
    assert cfg["student_config_from"] == "distilbert/distilgpt2"
    assert cfg["public_reference"] == "distilbert/distilgpt2"
    assert cfg["student_init"] == "random", (
        "design §8.1: the public checkpoint is a reference and never initialises a student")


def test_the_configs_and_the_preregistration_name_the_same_conditions(prereg):
    registered = set(prereg["conditions"]) - {"PDG2"}
    assert registered == set(CONFIG_BY_CONDITION)
    for entry in prereg["contrasts"] + prereg["go_no_go"]:
        for key in ("condition_a", "condition_b"):
            value = entry.get(key)
            if value:
                assert value in set(CONFIG_BY_CONDITION) | {"PDG2"}, (entry["id"], value)


def test_the_preregistered_corpus_is_the_one_the_gate_and_the_provider_produce(prereg):
    corpus_id = prereg["corpora"]["in_domain"]
    assert corpus_id == "openwebtext_validation_sink_300"
    assert gate.ARMS["e6a_gpt2"]["corpus_id"] == corpus_id
    assert prereg["pilot_gate"]["corpus_id"] == corpus_id
    # Every scored entry names it, so no criterion can be measured on a different corpus.
    for entry in prereg["go_no_go"]:
        limbs = (entry.get("compound", {}).get("all_of")
                 or entry.get("compound", {}).get("any_of") or [entry])
        for limb in limbs:
            assert limb.get("corpus_id") == corpus_id, entry["id"]


def test_the_gate_constants_match_the_preregistration(prereg):
    """The thresholds live in the YAML; the code must not carry a different number."""
    pilot = prereg["pilot_gate"]
    assert gate.SINK_THRESHOLD == pilot["sink_threshold"] == 0.15
    assert gate.MECHANISTIC_DELTA_THRESHOLD == pilot["mechanistic_delta_threshold"] == 0.10
    assert gate.PARITY_EXAMPLES_REQUIRED == pilot["parity_examples"] == 5
    assert gate.TEACHER_SINK_FRACTION == pilot["teacher_sink_fraction"] == 0.50
    assert gate.ARMS["e6a_gpt2"]["conditions"] == ("G0", "G1", "G2")


def test_criterion_2b_is_registered_for_this_arm_only(prereg):
    """Adding it to `e6a` would amend a pre-registration that already produced a verdict."""
    tinystories = yaml.safe_load(
        (CONFIG_DIR / "e6_preregistration.yaml").read_text(encoding="utf-8"))
    assert "pilot_gate" not in tinystories, (
        "e6_preregistration.yaml must stay untouched; criterion 2b belongs to the new arm")
    assert prereg["pilot_gate"]["teacher_sink_fraction_decided"] == "2026-08-01"
    ids = {a.get("decision_id") for a in prereg["amendments"]}
    assert "G1" in ids, "the teacher-relative criterion must carry its decision record"


def test_the_tinystories_preregistration_is_untouched():
    """The whole point of a second arm: nothing in the first arm's registration moves."""
    tinystories = yaml.safe_load(
        (CONFIG_DIR / "e6_preregistration.yaml").read_text(encoding="utf-8"))
    assert tinystories["version"] == "e6_prereg_v4"
    assert tinystories["corpora"]["in_domain"] == "tinystories_validation_sink_300"
    by_id = {c["id"]: c for c in tinystories["go_no_go"]}
    assert by_id["e6a_1"]["threshold"] == 0.15
    assert by_id["e6a_2"]["threshold"] == 0.10


def test_the_trainer_and_the_evaluator_agree_on_the_dataset_registry():
    """Train on one corpus and score on another and nothing downstream would say so."""
    assert set(td.BLOCK_LOADERS) == set(ev.IN_DOMAIN_PROVIDERS)
    assert td.DEFAULT_DATASET == ev.DEFAULT_DATASET
    dataset = _load("e6a_gpt2_ce.yaml")["data"]["dataset"]
    assert dataset in td.BLOCK_LOADERS and dataset in ev.IN_DOMAIN_PROVIDERS


def test_an_unrecognised_dataset_is_refused_by_both():
    """A typo must not silently fall back to TinyStories (CLAUDE.md trap 13)."""
    with pytest.raises(ValueError, match="no block loader"):
        td.resolve_block_loader("Skylion007/openwebtxt")
    with pytest.raises(ValueError, match="no in-domain corpus provider"):
        ev.resolve_in_domain_provider("Skylion007/openwebtxt")

    # An ABSENT value is the pre-arm default, which is what every old run means.
    assert td.resolve_block_loader(None)[1] == "roneneldan/TinyStories"
    assert ev.resolve_in_domain_provider(None) is ev.IN_DOMAIN_PROVIDERS[
        "roneneldan/TinyStories"]
