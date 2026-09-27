import contextlib
import io
import random

import numpy as np
import pytest
import torch

from sinklab.checkpoint import (CheckpointError, latest_full, load_checkpoint,
                                save_checkpoint, verify_checkpoint, writer_lock)
from sinklab.hardware import HardwareError, Profile, build_batch_plan
from sinklab.train import Trainer, parameter_groups, update_lr


class TinyLM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(11, 7)
        self.drop = torch.nn.Dropout(0.2)
        self.linear = torch.nn.Linear(7, 11)

    def forward(self, ids):
        return self.linear(self.drop(self.embed(ids)))


def fixture(tmp_path, microbatch=8, dropout=0.2):
    torch.manual_seed(47)
    model = TinyLM()
    model.drop.p = dropout
    blocks = {str(i): [(i + j) % 11 for j in range(4)] for i in range(17)}
    identity = {k: "fixture" for k in ("protocol_hash", "data_hash", "init_hash", "hardware_hash",
                                  "calibration_hash", "model_hash", "backend")}
    identity.update(study="S1", condition="C0", seed=5, run_id="test", precision="fp32",
                    microbatch=microbatch, device_role="cpu")

    def loss(ids):
        logits = model(ids)
        loss = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 11), ids[:, 1:].reshape(-1))
        return loss, {"ce": float(loss.detach())}

    return Trainer(model=model, blocks=blocks, loss_fn=loss, identity=identity,
                   run_dir=tmp_path, device=torch.device("cpu"), microbatch=microbatch,
                   seed=5, engineering_fixture=True)


def run_silent(trainer, **kwargs):
    with contextlib.redirect_stdout(io.StringIO()):
        return trainer.train(terminal=False, text_interval=100, **kwargs)


def test_lr_and_parameter_groups():
    assert update_lr(1) == pytest.approx(1e-6)
    assert update_lr(500) == pytest.approx(5e-4)
    assert update_lr(501) < update_lr(500)
    assert update_lr(10000) == pytest.approx(5e-5)
    assert update_lr(10001) == pytest.approx(5e-5)
    model = TinyLM()
    groups = parameter_groups(model)
    assert len({id(p) for g in groups for p in g["params"]}) == len(list(model.parameters()))
    assert model.linear.bias in groups[1]["params"]


def test_exact_resume_next_loss_rng_and_order(tmp_path):
    full = fixture(tmp_path / "full")
    run_silent(full, stop_after=5)
    split = fixture(tmp_path / "split")
    run_silent(split, stop_after=2)
    resumed = fixture(tmp_path / "split")
    resumed.resume()
    run_silent(resumed, stop_after=5)
    for a, b in zip(full.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert full.order.snapshot() == resumed.order.snapshot()
    assert full.state.step == resumed.state.step == 5
    assert full.state.input_tokens == resumed.state.input_tokens == 1280
    assert full.state.target_tokens == resumed.state.target_tokens == 960
    assert full.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"]
    assert random.getstate() == torch.load(latest_full(resumed.run_dir / "checkpoints", identity=resumed.identity) / "state.pt", weights_only=True)["rng"]["python"]
    assert np.array_equal(torch.get_rng_state(), torch.load(latest_full(resumed.run_dir / "checkpoints", identity=resumed.identity) / "state.pt", weights_only=True)["rng"]["torch_cpu"])
    # Compare the entire next update, including its loss and optimizer moments.
    full_next = fixture(tmp_path / "full")
    full_next.resume()
    run_silent(full_next, stop_after=6)
    resumed_next = fixture(tmp_path / "split")
    resumed_next.resume()
    run_silent(resumed_next, stop_after=6)
    for a, b in zip(full_next.model.parameters(), resumed_next.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    a = full_next.optimizer.state_dict()["state"]
    b = resumed_next.optimizer.state_dict()["state"]
    for key in a:
        for field in a[key]:
            torch.testing.assert_close(a[key][field], b[key][field], rtol=0, atol=0)
    assert full_next.order.snapshot() == resumed_next.order.snapshot()
    import json
    def last_loss(path):
        return [json.loads(line)["loss"] for line in path.read_text().splitlines() if '"event":"update"' in line][-1]
    assert last_loss(full_next.run_dir / "train.jsonl") == last_loss(resumed_next.run_dir / "train.jsonl")


def test_atomic_failure_corruption_and_writer(tmp_path):
    trainer = fixture(tmp_path)
    run_silent(trainer, stop_after=1)
    root = tmp_path / "checkpoints"
    prior = latest_full(root, identity=trainer.identity)
    with pytest.raises(OSError), writer_lock(tmp_path):
        save_checkpoint(root, step=2, kind="rolling", model=trainer.model,
                        state=trainer._snapshot(), identity=trainer.identity, fail_at="before_rename")
    assert latest_full(root, identity=trainer.identity) == prior
    with writer_lock(tmp_path):
        with pytest.raises(CheckpointError, match="duplicate writer"):
            with writer_lock(tmp_path):
                pass
    (root / "rolling-000999").mkdir()
    assert latest_full(root, identity=trainer.identity) == prior
    assert verify_checkpoint(prior, identity=trainer.identity)["step"] == 1
    with pytest.raises(CheckpointError):
        verify_checkpoint(prior, identity={**trainer.identity, "seed": 7})


def test_mock_common_solver_rejects_missing_and_rel_4080():
    def p(condition, role, batch, passed=True):
        return Profile(condition, role, role + "-uuid", batch, passed, "mock_cpu", True, True, True,
                       100, 110, 1000, 1000, 100)
    required = {("C1", "rtx3090", "rtx3090-uuid"), ("C2", "rtx4080super", "rtx4080super-uuid")}
    profiles = [p("C1", "rtx3090", b, b <= 16) for b in (64, 32, 16, 8, 4, 2, 1)]
    profiles += [p("C2", "rtx4080super", b, b <= 8) for b in (64, 32, 16, 8, 4, 2, 1)]
    plan = build_batch_plan(profiles, required, production=False)
    assert (plan["microbatch"], plan["accumulation"], plan["evidence"]) == (8, 8, "mock_cpu")
    with pytest.raises(HardwareError, match="mock evidence"):
        build_batch_plan(profiles, required, production=True)
    with pytest.raises(HardwareError, match="missing"):
        build_batch_plan(profiles[:1], required, production=False)
    with pytest.raises(HardwareError, match="REL"):
        build_batch_plan([p("C4", "rtx4080super", 1)], {("C4", "rtx4080super", "rtx4080super-uuid")}, production=False)


def test_effective_batch_accumulates_once_and_stop_keeps_clock(tmp_path):
    big = fixture(tmp_path / "big", microbatch=64, dropout=0)
    small = fixture(tmp_path / "small", microbatch=8, dropout=0)
    run_silent(big, stop_after=1)
    run_silent(small, stop_after=1)
    for a, b in zip(big.model.parameters(), small.model.parameters()):
        torch.testing.assert_close(a, b, rtol=2e-6, atol=2e-7)
    assert {int(s["step"]) for s in big.optimizer.state.values()} == {1}
    assert {int(s["step"]) for s in small.optimizer.state.values()} == {1}
    assert small.optimizer.param_groups[0]["lr"] == update_lr(1)
    assert small.state.input_tokens == 64 * 4
    assert small.state.target_tokens == 64 * 3
    with pytest.raises(ValueError, match="extension ID"):
        run_silent(small, stop_after=10001)
