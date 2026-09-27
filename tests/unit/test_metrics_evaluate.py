import copy
import math
import random

import numpy as np
import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.evaluate import RecordStore, cadence, evaluate_panel, precision_diagnostics, rng_neutral
from sinklab.metrics import (aggregate_behavior, behavioral_item, depth_support,
                             fingerprint, guarded_spearman, sink_profile,
                             weighted_depth_wasserstein)
from sinklab.models import GPT2Adapter


def test_manual_behavior_kl_direction_cancellation_and_chunking():
    clean = torch.tensor([[[math.log(.9), math.log(.1)], [0., 0.]],
                          [[math.log(.1), math.log(.9)], [0., 0.]]])
    edited = clean.flip(-1)
    ids = torch.tensor([[0, 0], [1, 0]])
    mask = torch.ones_like(ids)
    teacher = torch.tensor([[[math.log(.8), math.log(.2)], [0., 0.]],
                            [[math.log(.8), math.log(.2)], [0., 0.]]])
    a = behavioral_item(clean, edited, ids, mask, teacher=teacher, token_chunk=1)
    b = behavioral_item(clean, edited, ids, mask, teacher=teacher, token_chunk=100)
    assert a == pytest.approx(b)
    assert a["flip_count"] == 2
    assert a["valid_targets"] == 2
    aggregate = aggregate_behavior([a])
    assert aggregate["delta_ce_nats"] == pytest.approx(0, abs=1e-7)
    assert aggregate["prediction_flip_fraction"] == 1
    assert aggregate["absolute_target_logprob_change_nats"] > 2
    assert aggregate["self_kl_nats"] > 1
    assert aggregate["clean_ppl"] == pytest.approx(math.exp(aggregate["clean_ce_nats"]))
    # Teacher KL is teacher-to-edited student, not its reverse.
    expected = (.8 * math.log(.8 / .1) + .2 * math.log(.2 / .9) +
                .8 * math.log(.8 / .9) + .2 * math.log(.2 / .1)) / 2
    assert aggregate["teacher_kl_nats"] == pytest.approx(expected, abs=1e-6)


def test_structure_depth_and_fingerprint_guards():
    p = torch.zeros(1, 2, 4, 4)
    p[0, 0, 2:, 0] = torch.tensor([.2, .4])
    p[0, 1, 2:, 0] = torch.tensor([.6, .8])
    profile = sink_profile([p, p * .5], torch.ones(1, 4, dtype=torch.bool))
    assert profile["layer_head_sink"][0] == pytest.approx([.3, .7])
    assert profile["native_layer_mean"] == pytest.approx(.375)
    assert weighted_depth_wasserstein([1, 0, 0], [0, 0, 1]) > 0
    assert weighted_depth_wasserstein([0, 0], [1, 1]) is None
    assert guarded_spearman([1, 1, 1], [1, 2, 3]) is None
    assert fingerprint(0, .5, denominator_floor=1e-6)["ratio"] is None
    assert fingerprint(.5, .25, denominator_floor=1e-6)["ratio"] == .5
    assert len(depth_support([1, 2, 3])) == 16


def _adapter():
    torch.manual_seed(31)
    config = GPT2Config(vocab_size=23, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=2, n_head=4, resid_pdrop=.2, embd_pdrop=.2,
                        attn_pdrop=.2, _attn_implementation="eager")
    return GPT2Adapter(GPT2LMHeadModel(config))


def test_rng_neutral_restores_modes_even_on_failure():
    adapter = _adapter()
    py, np_state, cpu = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    with pytest.raises(RuntimeError), rng_neutral(adapter.model):
        random.random(); np.random.random(); torch.rand(3)
        assert not adapter.model.training
        raise RuntimeError("fault")
    assert adapter.model.training
    assert random.getstate() == py
    assert np.array_equal(np.random.get_state()[1], np_state[1])
    torch.testing.assert_close(torch.get_rng_state(), cpu, rtol=0, atol=0)


def test_panel_records_idempotent_and_restores_rng(tmp_path):
    adapter = _adapter()
    store = RecordStore(tmp_path)
    items = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
             {"id": "b", "input_ids": [4, 3, 2, 0], "attention_mask": [1, 1, 1, 0]}]
    before = torch.get_rng_state().clone()
    args = dict(adapter=adapter, items=items, panel="toy", panel_hash="a" * 64,
                checkpoint_hash="b" * 64, run_id="toy", step=0, store=store,
                operations=("clean", "none", "delete", "relocate"), terminal=False)
    first = evaluate_panel(**args)
    files = sorted(p.name for p in tmp_path.iterdir())
    second = evaluate_panel(**args)
    assert files == sorted(p.name for p in tmp_path.iterdir())
    assert first["operations"] == second["operations"]
    assert all(row["status"] == "complete" for row in first["operations"].values())
    assert first["operations"]["none"]["metrics"]["self_kl_nats"] == pytest.approx(0, abs=1e-6)
    assert first["operations"]["none"]["metrics"]["delta_ce_nats"] == pytest.approx(0, abs=1e-6)
    assert first["operations"]["none"]["metrics"]["prediction_flip_fraction"] == 0
    assert first["operations"]["delete"]["metrics"]["prediction_flip_fraction"] >= 0
    assert first["operations"]["delete"]["metrics"]["self_kl_nats"] >= -1e-6
    assert first["operations"]["clean"]["metrics"]["valid_targets"] == 5
    assert adapter.model.training
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    key = {"run_id": "r", "item_id": "x"}
    store.write(key, status="failed", value=None, error="injected")
    with pytest.raises(ValueError, match="explicit retry"):
        store.write(key, status="complete", value={"ok": True})
    store.write(key, status="complete", value={"ok": True}, retry_failed=True)


def test_failed_observation_visible_and_explicit_retry(tmp_path):
    adapter = _adapter()
    store = RecordStore(tmp_path)
    item = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}]
    original = adapter.forward
    def broken(**kwargs):
        raise RuntimeError("injected edit failure")
    adapter.forward = broken
    args = dict(adapter=adapter, items=item, panel="toy", panel_hash="a" * 64,
                checkpoint_hash="b" * 64, run_id="toy", step=100, store=store,
                operations=("clean", "delete"), terminal=False)
    result = evaluate_panel(**args)
    assert result["operations"]["clean"]["status"] == "complete"
    assert result["operations"]["delete"]["status"] == "incomplete"
    assert result["operations"]["delete"]["failed_item_ids"] == ["a"]
    adapter.forward = original
    assert evaluate_panel(**args)["operations"]["delete"]["status"] == "incomplete"
    assert evaluate_panel(**args, retry_failed=True)["operations"]["delete"]["status"] == "complete"


def test_teacher_topology_and_attention_descriptive_records(tmp_path):
    student, teacher = _adapter(), _adapter()
    records = evaluate_panel(adapter=student, teacher_adapter=teacher, teacher_map=(0, 1),
                             items=[{"id": "a", "input_ids": [1, 2, 3, 4],
                                     "attention_mask": [1, 1, 1, 1]}],
                             panel="toy", panel_hash="a" * 64, checkpoint_hash="b" * 64,
                             run_id="toy", step=0, store=RecordStore(tmp_path),
                             operations=("clean",), terminal=False)
    assert records["operations"]["clean"]["status"] == "complete"
    import json
    item_file = next(p for p in tmp_path.glob("*.json") if not p.name.startswith("aggregate"))
    value = json.loads(item_file.read_text())["payload"]["value"]
    assert len(value["topology"]["student_normalized_depth_16"]) == 16
    assert len(value["mapped_attention_similarity"]) == 2
    assert value["behavior"]["teacher_kl_sum_nats"] is not None


def test_cadence_clock_exact():
    dense = [step for step in range(10001) if "owt_dense64" in cadence(step)]
    full = [step for step in range(10001) if "owt_full300" in cadence(step)]
    assert len(dense) == 101
    assert 200 in dense and 300 in dense and 250 not in dense
    assert full == [0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000]
    assert cadence(250) == ("owt_full300",)


def test_precision_diagnostic_harness_is_rng_neutral():
    adapter = _adapter()
    ids = torch.tensor([[1, 2, 3, 4]])
    mask = torch.ones_like(ids)
    before = torch.get_rng_state().clone()
    report = precision_diagnostics(adapter, ids, mask)
    assert report["status"] == "diagnostic_only"
    assert report["fp32_noop_logit_max_abs"] < 1e-5
    assert report["bf16_probability_row_sum_max_abs"] < 1e-4
    torch.testing.assert_close(before, torch.get_rng_state(), rtol=0, atol=0)


def test_endpoint_records_only_clean_nll(tmp_path):
    result = evaluate_panel(adapter=_adapter(),
                            items=[{"id": "a", "input_ids": [1, 2, 3, 4],
                                    "attention_mask": [1, 1, 1, 1]}],
                            panel="owt_lm2000", panel_hash="a" * 64,
                            checkpoint_hash="b" * 64, run_id="toy", step=0,
                            store=RecordStore(tmp_path), operations=("clean",),
                            behavior_only=True, terminal=False)
    metrics = result["operations"]["clean"]["metrics"]
    assert set(metrics) == {"schema_version", "valid_targets", "clean_ce_nats",
                            "clean_log_ppl", "clean_ppl", "ppl_unavailable_reason",
                            "clean_accuracy_fraction", "units"}
    assert "self_kl_nats" not in metrics
