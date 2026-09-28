import contextlib
import io
import random
import hashlib
import shutil

import numpy as np
import pytest
import torch

from sinklab.checkpoint import (CheckpointError, latest_full, load_checkpoint,
                                save_checkpoint, verify_checkpoint, writer_lock)
from sinklab.hardware import HardwareError, Profile, build_batch_plan, validate_eligibility_matrix
from sinklab.train import TrainError, Trainer, parameter_groups, update_lr


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
    full_rng = torch.load(latest_full(full.run_dir / "checkpoints", identity=full.identity) / "state.pt",
                          weights_only=True)["rng"]
    resumed_rng = torch.load(latest_full(resumed.run_dir / "checkpoints", identity=resumed.identity) / "state.pt",
                             weights_only=True)["rng"]
    assert full_rng["python"] == resumed_rng["python"] == random.getstate()
    assert full_rng["numpy"] == resumed_rng["numpy"]
    torch.testing.assert_close(full_rng["torch_cpu"], resumed_rng["torch_cpu"], rtol=0, atol=0)
    assert len(full_rng["torch_cuda"]) == len(resumed_rng["torch_cuda"])
    for left, right in zip(full_rng["torch_cuda"], resumed_rng["torch_cuda"]):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
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
    with pytest.raises(OSError), writer_lock(tmp_path):
        save_checkpoint(root, step=2, kind="weights", model=trainer.model,
                        state=None, identity=trainer.identity, fail_at="after_rename")
    assert verify_checkpoint(root / "weights-000002", identity=trainer.identity)["step"] == 2
    assert latest_full(root, identity=trainer.identity) == prior
    with writer_lock(tmp_path):
        with pytest.raises(CheckpointError, match="duplicate writer"):
            with writer_lock(tmp_path):
                pass
    (root / "rolling-000999").mkdir()
    assert latest_full(root, identity=trainer.identity) == prior
    assert verify_checkpoint(prior, identity=trainer.identity)["step"] == 1
    with torch.no_grad():
        trainer.model.linear.bias.add_(1)
    with pytest.raises(CheckpointError, match="different model content"):
        save_checkpoint(root, step=1, kind="rolling", model=trainer.model,
                        state=trainer._snapshot(), identity=trainer.identity)
    with pytest.raises(CheckpointError):
        verify_checkpoint(prior, identity={**trainer.identity, "seed": 7})


def test_mock_common_solver_rejects_missing_and_rel_4080():
    def p(condition, role, batch, passed=True):
        return Profile(condition, role, role + "-uuid", batch, passed, "mock_cpu", True, True, True,
                       100, 110, 1000 if passed else 0, 1000, 100)
    required = {("C1", "rtx3090", "rtx3090-uuid"), ("C2", "rtx4080super", "rtx4080super-uuid")}
    profiles = [p("C1", "rtx3090", b, b <= 16) for b in (64, 32, 16, 8, 4, 2, 1)]
    profiles += [p("C2", "rtx4080super", b, b <= 8) for b in (64, 32, 16, 8, 4, 2, 1)]
    plan = build_batch_plan(profiles, required, production=False)
    assert (plan["microbatch"], plan["accumulation"], plan["evidence"]) == (8, 8, "mock_cpu")
    assert plan["eligibility_matrix"] == {"C1": {"rtx3090": "rtx3090-uuid"},
                                          "C2": {"rtx4080super": "rtx4080super-uuid"}}
    validate_eligibility_matrix(plan)
    with pytest.raises(HardwareError, match="disagrees"):
        validate_eligibility_matrix({**plan, "eligibility_matrix": {"C1": {"rtx3090": "wrong"}}})
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


def test_rng_neutral_evaluation_insertion_preserves_future_training(tmp_path):
    from sinklab.evaluate import RecordStore, evaluate_panel
    from sinklab.models import GPT2Adapter
    from transformers import GPT2Config, GPT2LMHeadModel
    baseline = fixture(tmp_path / "baseline")
    run_silent(baseline, stop_after=5)
    adapter = GPT2Adapter(GPT2LMHeadModel(GPT2Config(
        vocab_size=11, n_positions=4, n_ctx=4, n_embd=8, n_layer=1, n_head=2,
        resid_pdrop=.2, embd_pdrop=.2, attn_pdrop=.2, _attn_implementation="eager")))
    inserted = fixture(tmp_path / "inserted")
    run_silent(inserted, stop_after=2)
    with contextlib.redirect_stdout(io.StringIO()):
        evaluation = evaluate_panel(adapter=adapter, items=[{"id": "x", "input_ids": [1, 2, 3, 4],
                             "attention_mask": [1, 1, 1, 1]}], panel="fixture",
                             panel_hash="a" * 64, checkpoint_hash="b" * 64,
                             run_id="parity", step=2, store=RecordStore(tmp_path / "evaluation"),
                             operations=("clean", "delete", "relocate"), terminal=False)
    assert all(row["status"] == "complete" for row in evaluation["operations"].values())
    run_silent(inserted, stop_after=5)
    for a, b in zip(baseline.model.parameters(), inserted.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert baseline.order.snapshot() == inserted.order.snapshot()


def test_mid_accumulation_failure_replays_from_last_full_state(tmp_path):
    baseline = fixture(tmp_path / "baseline")
    run_silent(baseline, stop_after=3)
    broken = fixture(tmp_path / "interrupted")
    original, calls = broken.loss_fn, 0
    def fail_on_second_microbatch(ids):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected mid-accumulation interruption")
        return original(ids)
    broken.loss_fn = fail_on_second_microbatch
    with pytest.raises(RuntimeError, match="mid-accumulation"):
        run_silent(broken, stop_after=3)
    restored = fixture(tmp_path / "interrupted")
    restored.resume()
    assert restored.state.step == 0
    run_silent(restored, stop_after=3)
    for a, b in zip(baseline.model.parameters(), restored.model.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    assert baseline.order.snapshot() == restored.order.snapshot()


def test_simulated_10k_trainer_clock_keeps_250_full_and_final(tmp_path):
    from sinklab.evaluate import cadence
    trainer = fixture(tmp_path)
    observations, saves = [], []
    def fake_update():
        trainer.state.step += 1
        trainer.state.input_tokens += 256
        trainer.state.target_tokens += 192
        return 0., {"grad_norm": 0.}
    trainer._one_update = fake_update
    trainer._save = lambda kind: saves.append((kind, trainer.state.step))
    trainer.train(stop_after=10000, evaluate=lambda step: observations.append((step, cadence(step))),
                  terminal=False, text_interval=10001)
    dense = [step for step, panels in observations if "owt_dense64" in panels]
    assert dense == list(range(0, 10001, 100))
    assert (250, ("owt_full300",)) in observations
    assert ("final", 10000) in saves


def test_10k_full_save_precedes_failing_endpoint_evaluation(tmp_path):
    trainer = fixture(tmp_path)
    def fake_update():
        trainer.state.step += 1
        trainer.state.input_tokens += 256
        trainer.state.target_tokens += 192
        return 0., {"grad_norm": 0.}
    saves = []
    trainer._one_update = fake_update
    trainer._save = lambda kind: saves.append((kind, trainer.state.step))
    def fail_at_endpoint(step):
        if step == 10000:
            raise RuntimeError("injected endpoint evaluation failure")
    with pytest.raises(RuntimeError, match="endpoint evaluation failure"):
        run_silent(trainer, stop_after=10000, evaluate=fail_at_endpoint)
    assert ("final", 10000) in saves


def test_non_tty_banner_and_update_are_structured(tmp_path):
    import json
    trainer = fixture(tmp_path)
    output = io.StringIO()
    with contextlib.redirect_stdout(output):
        trainer.train(stop_after=1, terminal=False, text_interval=1)
    rows = [json.loads(line) for line in output.getvalue().splitlines()]
    banner, update = rows
    assert (banner["study"], banner["condition"], banner["seed"], banner["run_id"]) == ("S1", "C0", 5, "test")
    assert (banner["microbatch"], banner["accumulation"], banner["effective_batch"]) == (8, 8, 64)
    assert banner["gpu_name"] is None and banner["gpu_total_bytes"] is None
    assert update["step"] == 1 and update["loss"] > 0
    assert update["ce"] == pytest.approx(update["loss"], abs=1e-7)
    assert update["kd"] is None and update["attention"] is None and update["relation"] is None
    assert update["input_tokens"] == 256 and update["target_tokens"] == 192
    assert update["eta_train_seconds"] == 0 and update["tokens_per_second"] > 0
    assert banner["registered_horizon"] == 10000 and banner["total_updates"] == 1


def test_protected_10k_extension_exact_resume_optimizer_rng_data_lr_and_lineage(tmp_path):
    """A compact state at the absolute 10k clock exercises real full-state replay."""
    base = fixture(tmp_path / "base")
    run_silent(base, stop_after=1)
    for _ in range(9999):
        base.order.take_update()
    base.state.step = 10000
    base.state.input_tokens = 10000 * 64 * 4
    base.state.target_tokens = 10000 * 64 * 3
    for group in base.optimizer.param_groups:
        group["lr"] = update_lr(10000)
    anchor = base._save("final")
    anchor_sha = hashlib.sha256((anchor / "manifest.json").read_bytes()).hexdigest()
    with pytest.raises(TrainError, match="protected 10k state"):
        run_silent(base, stop_after=10001, extension_id="approved-test-extension")
    split_dir = tmp_path / "split"
    shutil.copytree(anchor, split_dir / "checkpoints" / anchor.name)
    base.resume(anchor)
    run_silent(base, stop_after=10002, extension_id="approved-test-extension")
    split = fixture(split_dir)
    split.resume()
    run_silent(split, stop_after=10001, extension_id="approved-test-extension")
    resumed = fixture(split_dir)
    resumed.resume()
    run_silent(resumed, stop_after=10002, extension_id="approved-test-extension")
    assert base.state == resumed.state or (base.state.step, base.state.input_tokens,
        base.state.target_tokens) == (resumed.state.step, resumed.state.input_tokens,
        resumed.state.target_tokens)
    for left, right in zip(base.model.parameters(), resumed.model.parameters()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    left_state, right_state = base.optimizer.state_dict(), resumed.optimizer.state_dict()
    for key in left_state["state"]:
        for field in left_state["state"][key]:
            torch.testing.assert_close(left_state["state"][key][field],
                                       right_state["state"][key][field], rtol=0, atol=0)
    assert base.order.snapshot() == resumed.order.snapshot()
    assert base.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"] == update_lr(10002)
    assert base.extension_id == resumed.extension_id == "approved-test-extension"
    assert base.extension_parent_sha256 == resumed.extension_parent_sha256 == anchor_sha
    assert hashlib.sha256((anchor / "manifest.json").read_bytes()).hexdigest() == anchor_sha
    assert (split_dir / "checkpoints" / "final-010000").is_dir()
    assert (split_dir / "checkpoints" / "final-010002").is_dir()
    full_state = torch.load(base.run_dir / "checkpoints" / "final-010002" / "state.pt", weights_only=True)
    resumed_state = torch.load(split_dir / "checkpoints" / "final-010002" / "state.pt", weights_only=True)
    assert full_state["rng"]["python"] == resumed_state["rng"]["python"]
    assert full_state["rng"]["numpy"] == resumed_state["rng"]["numpy"]
    torch.testing.assert_close(full_state["rng"]["torch_cpu"],
                               resumed_state["rng"]["torch_cpu"], rtol=0, atol=0)
    assert full_state["scheduler"] == resumed_state["scheduler"]
    with pytest.raises(ValueError, match="lineage changed"):
        run_silent(resumed, stop_after=10003, extension_id="different-extension")
    observed = []
    def cheap_update():
        resumed.order.take_update()
        resumed.state.step += 1
        resumed.state.input_tokens += 256
        resumed.state.target_tokens += 192
        return 0., {"grad_norm": 0.}
    resumed._one_update = cheap_update
    run_silent(resumed, stop_after=10500, extension_id="approved-test-extension",
               evaluate=lambda step: observed.append(step))
    assert observed == [10100, 10200, 10300, 10400, 10500]
    assert (split_dir / "checkpoints" / "rolling-010500").is_dir()
    assert (split_dir / "checkpoints" / "final-010500").is_dir()
    assert hashlib.sha256((anchor / "manifest.json").read_bytes()).hexdigest() == anchor_sha
