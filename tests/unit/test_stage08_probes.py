import json

import pytest
import torch
from transformers import GPT2Config, GPT2LMHeadModel, GPTNeoXConfig, GPTNeoXForCausalLM

from sinklab.evaluate import RecordStore
from sinklab.models import GPT2Adapter
from sinklab.probes import (ProbeError, apply_probe, coordinate_controls,
    epe_directions, evaluate_probe_battery, probe_plan, transport_epe_batch)


def _gpt2(width=16):
    torch.manual_seed(57 + width)
    return GPT2LMHeadModel(GPT2Config(vocab_size=29, n_positions=8, n_ctx=8,
        n_embd=width, n_layer=2, n_head=4, resid_pdrop=.2, embd_pdrop=.2,
        attn_pdrop=.2, _attn_implementation="eager"))


def test_batched_epe_exact_reference_and_sum_conservation():
    model = _gpt2()
    before = torch.get_rng_state().clone()
    directions = epe_directions(model)
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        model.eval()
        p = model.transformer.wpe.weight[:2]
        expected = p + model.transformer.h[0].mlp(p)
    torch.testing.assert_close(directions["e"], expected, rtol=0, atol=0)
    torch.manual_seed(11)
    m = torch.randn(3, 4, 16)
    mask = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]])
    out = transport_epe_batch(m, directions["u"][0], directions["u"][1], mask)
    assert not torch.allclose(torch.linalg.vector_norm(out[:, 0], dim=-1),
                              torch.linalg.vector_norm(m[:, 0], dim=-1))
    for batch in range(3):
        a = torch.dot(m[batch, 0], directions["u"][0])
        delta = directions["u"][1] - directions["u"][0]
        torch.testing.assert_close(out[batch, 0], m[batch, 0] + a * delta)
        torch.testing.assert_close(out[batch, 1], m[batch, 1] - a * delta)
        torch.testing.assert_close(out[batch, :2].sum(0), m[batch, :2].sum(0), atol=2e-6, rtol=0)
    with pytest.raises(ProbeError, match="two real"):
        transport_epe_batch(m, directions["u"][0], directions["u"][1],
                            torch.tensor([[1, 1, 1, 1], [1, 0, 0, 0], [1, 1, 1, 1]]))


def test_model_local_coordinates_k_slice_and_exception_restoration():
    models = [_gpt2(16), _gpt2(24)]
    for model, top in zip(models, ([0, 1, 2], [20, 21, 22]), strict=True):
        with torch.no_grad():
            model.transformer.wpe.weight[0].zero_()
            model.transformer.wpe.weight[0, top] = torch.tensor([9., 8., 7.])
            mlp = model.transformer.h[0].mlp
            for param in mlp.parameters():
                param.zero_()
        plan = probe_plan(model, control_seed=82)
        assert plan["k_top3_all"]["coordinates"] == top
        controls = coordinate_controls(model, control_seed=82)["random_sets"]
        assert len(controls) == len({tuple(x) for x in controls}) == 5
        assert all(len(x) == 3 and not set(x) & set(top) for x in controls)
        width = model.config.n_embd
        prior = {k: v.detach().clone() for k, v in model.state_dict().items()}
        mask = torch.ones(1, 4, dtype=torch.bool)
        with pytest.raises(RuntimeError, match="injected"):
            with apply_probe(model, "k_top3_all", plan=plan, mask=mask):
                for index, layer in enumerate(model.transformer.h):
                    before = prior[f"transformer.h.{index}.attn.c_attn.weight"]
                    weight = layer.attn.c_attn.weight
                    assert torch.count_nonzero(weight[top, width:2 * width]) == 0
                    torch.testing.assert_close(weight[:, :width], before[:, :width], rtol=0, atol=0)
                    torch.testing.assert_close(weight[:, 2 * width:], before[:, 2 * width:], rtol=0, atol=0)
                raise RuntimeError("injected")
        for name, tensor in model.state_dict().items():
            torch.testing.assert_close(tensor, prior[name], rtol=0, atol=0)
        with apply_probe(model, "q_bias_all", plan=plan, mask=mask):
            for index, layer in enumerate(model.transformer.h):
                assert torch.count_nonzero(layer.attn.c_attn.bias[:width]) == 0
                torch.testing.assert_close(layer.attn.c_attn.bias[width:],
                    prior[f"transformer.h.{index}.attn.c_attn.bias"][width:])


def test_epe_hook_restores_and_zero_direction_rejected():
    model = _gpt2()
    plan = probe_plan(model, control_seed=1)
    hooks = len(model.transformer.h[0].mlp._forward_hooks)
    with pytest.raises(RuntimeError):
        with apply_probe(model, "epe_transport_layer0", plan=plan,
                         mask=torch.ones(1, 4, dtype=torch.bool)):
            assert len(model.transformer.h[0].mlp._forward_hooks) == hooks + 1
            raise RuntimeError("fault")
    assert len(model.transformer.h[0].mlp._forward_hooks) == hooks
    with torch.no_grad():
        model.transformer.wpe.weight[:2].zero_()
        for param in model.transformer.h[0].mlp.parameters():
            param.zero_()
    with pytest.raises(ProbeError, match="zero"):
        epe_directions(model)


def test_probe_battery_records_nonresponse_guard_and_neox_not_applicable(tmp_path):
    model = _gpt2()
    adapter = GPT2Adapter(model)
    before = torch.get_rng_state().clone()
    item = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]}]
    progress = []
    kwargs = dict(adapter=adapter, items=item, store=RecordStore(tmp_path),
        checkpoint_sha256="a" * 64, panel_sha256="b" * 64, run_id="fixture",
        model_role="teacher", control_seed=3, denominator_floor=1e6,
        responsiveness_floor=1e6, provenance={"fixture": True},
        progress_callback=lambda done, total, item_id: progress.append((done, total, item_id)))
    first = evaluate_probe_battery(**kwargs)
    assert first["status"] == "complete"
    assert progress == [(1, 1, "a")]
    stored = json.loads(next(tmp_path.glob("*.json")).read_text(encoding="utf-8"))
    assert stored["payload"]["value"]["attention_mask"] == [[1, 1, 1, 1]]
    assert len(first["probes"]) == 10
    assert all(v["responsiveness"] == "nonresponsive" for v in first["probes"].values())
    assert all(v["fingerprint"]["ratio"] is None for v in first["probes"].values())
    files = sorted(p.name for p in tmp_path.iterdir())
    kwargs.pop("progress_callback")
    assert evaluate_probe_battery(**kwargs) == first
    assert files == sorted(p.name for p in tmp_path.iterdir())
    assert model.training
    torch.testing.assert_close(torch.get_rng_state(), before, rtol=0, atol=0)
    neox = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=29, hidden_size=16,
        num_hidden_layers=2, num_attention_heads=4, intermediate_size=32,
        max_position_embeddings=8, _attn_implementation="eager"))
    neox_result = evaluate_probe_battery(adapter=type("Adapter", (), {"model": neox})(),
        **{k: v for k, v in kwargs.items() if k != "adapter"})
    assert neox_result["status"] == "not_applicable"
    assert all(v["status"] == "not_applicable" for v in neox_result["probes"].values())


def test_d24_s4_student_records_keep_training_root_and_separate_policy(tmp_path):
    from sinklab.followup_policy import D24_SHA256
    source = {"study": "S1", "condition": "C6", "seed": 0,
              "run_id": "original", "protocol_sha256": "c" * 64}
    result = evaluate_probe_battery(adapter=GPT2Adapter(_gpt2()),
        items=[{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1] * 4}],
        store=RecordStore(tmp_path), checkpoint_sha256="a" * 64,
        panel_sha256="b" * 64, run_id="original", model_role="student",
        control_seed=1729, denominator_floor=1e-8, responsiveness_floor=1e-8,
        provenance={"fixture": True}, source_identity=source, step=10000)
    assert result["status"] == "complete"
    assert result["followup_policy"]["amendment_sha256"] == D24_SHA256
    for path in tmp_path.glob("*.json"):
        key = json.loads(path.read_text())["payload"]["key"]
        assert key["source_identity"] == source
        assert key["step"] == 10000
        assert key["followup_policy"]["source_protocol_sha256"] == "c" * 64
