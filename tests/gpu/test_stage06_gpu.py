"""Bounded full-size S1 engineering checks on the explicitly selected GPU."""

import contextlib
import hashlib
import io
import json
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.interventions import AttentionIntervention
from sinklab.evaluate import RecordStore, evaluate_panel, precision_diagnostics
from sinklab.models import GPT2Adapter, ModelShape
from sinklab.train import make_objective_loss
from sinklab.train import Trainer
from sinklab.checkpoint import latest_full


TEACHER_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"


@pytest.fixture(scope="session")
def full_size_pair(selected_cuda, gpu_evidence):
    evidence, _ = gpu_evidence
    cached = try_to_load_from_cache("openai-community/gpt2-large", "config.json", revision=TEACHER_REVISION)
    if not isinstance(cached, str) or not Path(cached).is_file() or not (Path(cached).parent / "model.safetensors").is_file():
        evidence["status"] = "blocked_missing_pinned_teacher"
        pytest.skip("pinned GPT-2-large teacher is absent from local cache")
    torch.manual_seed(1729)
    teacher = GPT2LMHeadModel.from_pretrained(str(Path(cached).parent), local_files_only=True,
                                               attn_implementation="eager").to(selected_cuda).eval()
    student_config = GPT2Config(n_layer=24, n_head=16, n_embd=1024, _attn_implementation="eager")
    student = GPT2LMHeadModel(student_config).to(selected_cuda).eval()
    teacher.requires_grad_(False)
    evidence["teacher_revision"] = TEACHER_REVISION
    digest = hashlib.sha256()
    with (Path(cached).parent / "model.safetensors").open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    evidence["teacher_safetensors_sha256"] = digest.hexdigest()
    evidence["student_initialization"] = "random_from_config_seed1729"
    evidence["teacher_parameters"] = sum(p.numel() for p in teacher.parameters())
    evidence["student_parameters"] = sum(p.numel() for p in student.parameters())
    yield GPT2Adapter(teacher, ModelShape(36, 20, 1280)), GPT2Adapter(student, ModelShape(24, 16, 1024))
    del teacher, student
    torch.cuda.empty_cache()


def _ids(device):
    return torch.tensor([[15496, 11, 616, 1438, 318, 257, 1332, 13]], device=device)


def test_full_size_native_features_and_causal_edits(full_size_pair, selected_cuda, gpu_evidence):
    teacher, student = full_size_pair
    measurements = gpu_evidence[0]["measurements"]
    ids = _ids(selected_cuda)
    torch.cuda.reset_peak_memory_stats(selected_cuda)
    for name, adapter in (("teacher", teacher), ("student", student)):
        with torch.no_grad():
            native = adapter.model(input_ids=ids, use_cache=False, output_attentions=True)
            clean = adapter.forward(input_ids=ids)
            featured = adapter.forward_with_features(input_ids=ids)
            noop = adapter.forward(input_ids=ids, intervention=AttentionIntervention("none"), layer_scope=(0,))
            deleted = adapter.forward(input_ids=ids, intervention=AttentionIntervention("delete"), layer_scope=(0,))
            relocated = adapter.forward(input_ids=ids, intervention=AttentionIntervention("relocate"), layer_scope=(0,))
        logit_error = float((clean.logits - native.logits).abs().max())
        noop_error = float((noop.logits - native.logits).abs().max())
        probability_error = max(float((a.probabilities - b.float()).abs().max())
                                for a, b in zip(featured.attention, native.attentions, strict=True))
        delete_delta = float((deleted.logits - native.logits).abs().max())
        relocate_delta = float((relocated.logits - native.logits).abs().max())
        measurements[name] = {"native_logit_max_error": logit_error,
                              "noop_logit_max_error": noop_error,
                              "pre_dropout_probability_max_error": probability_error,
                              "delete_max_logit_change": delete_delta,
                              "relocate_max_logit_change": relocate_delta}
        assert logit_error == 0 and noop_error <= 1e-4
        assert probability_error <= 1e-4
        assert delete_delta > 0 and relocate_delta > 0
    measurements["peak_allocated_bytes_preflight"] = torch.cuda.max_memory_allocated(selected_cuda)


def test_eligible_condition_objective_smoke(full_size_pair, selected_cuda, gpu_evidence):
    teacher, student = full_size_pair
    ids = _ids(selected_cuda)
    teacher_map = tuple((s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1) for s in range(24))
    values = {}
    for condition in ("C0", "C1", "C2", "C5", "C6"):
        student.model.zero_grad(set_to_none=True)
        loss_fn = make_objective_loss(study="S1", condition=condition, student_adapter=student,
                                      teacher_adapter=None if condition == "C0" else teacher,
                                      teacher_map=teacher_map)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss, components = loss_fn(ids)
        loss.backward()
        grad = student.model.transformer.h[0].attn.c_attn.weight.grad
        values[condition] = {"loss": float(loss.detach()), "first_qkv_grad_norm": float(grad.float().norm()),
                             "components": components}
        assert torch.isfinite(loss) and grad is not None and torch.isfinite(grad).all()
        del loss, components, loss_fn
        student.model.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
    gpu_evidence[0]["measurements"]["bf16_objective_smoke"] = values


def test_full_size_precision_metrics_and_rng_neutrality(full_size_pair, selected_cuda, gpu_evidence, tmp_path):
    teacher, student = full_size_pair
    ids = _ids(selected_cuda)
    mask = torch.ones_like(ids, dtype=torch.bool)
    before_cpu = torch.get_rng_state().clone()
    before_cuda = torch.cuda.get_rng_state(selected_cuda).clone()
    diagnostic = precision_diagnostics(student, ids, mask)
    result = evaluate_panel(adapter=student,
                            items=[{"id": "synthetic-gpu-probe", "input_ids": ids[0].tolist(),
                                    "attention_mask": mask[0].tolist()}],
                            panel="engineering_probe", panel_hash="a" * 64,
                            checkpoint_hash="b" * 64, run_id="stage06-engineering", step=0,
                            store=RecordStore(tmp_path / "metrics"), layer_scope=(0,),
                            teacher_adapter=teacher, teacher_map=tuple(
                                (s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1)
                                for s in range(24)),
                            operations=("clean", "none", "delete", "relocate"),
                            precision="bf16", terminal=False)
    assert all(row["status"] == "complete" for row in result["operations"].values())
    metrics = result["operations"]
    assert metrics["none"]["metrics"]["self_kl_nats"] <= 1e-6
    assert metrics["none"]["metrics"]["prediction_flip_fraction"] == 0
    assert metrics["clean"]["metrics"]["teacher_kl_nats"] is not None
    assert diagnostic["fp32_noop_logit_max_abs"] <= 1e-4
    assert diagnostic["bf16_probability_row_sum_max_abs"] <= 1e-4
    assert torch.equal(before_cpu, torch.get_rng_state())
    assert torch.equal(before_cuda, torch.cuda.get_rng_state(selected_cuda))
    gpu_evidence[0]["measurements"]["precision"] = diagnostic
    gpu_evidence[0]["measurements"]["metric_operations"] = list(result["operations"])
    gpu_evidence[0]["measurements"]["metric_summary"] = {
        op: {key: row["metrics"][key] for key in
             ("clean_ce_nats", "edited_ce_nats", "self_kl_nats", "prediction_flip_fraction",
              "teacher_kl_nats")}
        for op, row in metrics.items()}
    gpu_evidence[0]["measurements"]["evaluation_rng_neutral"] = True


def test_cuda_bf16_checkpoint_resume_cursor_rng_and_next_loss(selected_cuda, gpu_evidence, tmp_path):
    class TinyLM(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = torch.nn.Embedding(19, 16)
            self.drop = torch.nn.Dropout(0.2)
            self.head = torch.nn.Linear(16, 19)

        def forward(self, ids):
            return self.head(self.drop(self.embed(ids)))

    def make_trainer(path):
        torch.manual_seed(71)
        model = TinyLM()
        blocks = {str(i): [(i + j) % 19 for j in range(6)] for i in range(73)}
        identity = {key: "engineering" for key in ("protocol_hash", "data_hash", "init_hash",
                    "hardware_hash", "calibration_hash", "model_hash", "backend")}
        identity.update(study="S1", condition="C0", seed=71, run_id="cuda-resume",
                        precision="bf16", microbatch=8, device_role="rtx4080super")
        def loss(ids):
            logits = model(ids)
            value = torch.nn.functional.cross_entropy(logits[:, :-1].reshape(-1, 19),
                                                      ids[:, 1:].reshape(-1))
            return value, {"ce": float(value.detach())}
        return Trainer(model=model, blocks=blocks, loss_fn=loss, identity=identity,
                       run_dir=path, device=selected_cuda, microbatch=8, seed=71,
                       engineering_fixture=True)

    def train_silent(trainer, stop):
        with contextlib.redirect_stdout(io.StringIO()):
            trainer.train(stop_after=stop, terminal=False, text_interval=100)

    full = make_trainer(tmp_path / "full")
    train_silent(full, 3)
    interrupted = make_trainer(tmp_path / "split")
    train_silent(interrupted, 1)
    resumed = make_trainer(tmp_path / "split")
    resumed.resume()
    train_silent(resumed, 3)
    max_error = max(float((a.detach() - b.detach()).abs().max())
                    for a, b in zip(full.model.parameters(), resumed.model.parameters()))
    assert max_error == 0
    assert full.order.snapshot() == resumed.order.snapshot()
    assert full.optimizer.param_groups[0]["lr"] == resumed.optimizer.param_groups[0]["lr"]
    assert full.state.input_tokens == resumed.state.input_tokens == 3 * 64 * 6
    assert full.state.target_tokens == resumed.state.target_tokens == 3 * 64 * 5
    full_state = torch.load(latest_full(full.run_dir / "checkpoints", identity=full.identity) / "state.pt", weights_only=True)
    resumed_state = torch.load(latest_full(resumed.run_dir / "checkpoints", identity=resumed.identity) / "state.pt", weights_only=True)
    assert torch.equal(full_state["rng"]["torch_cpu"], resumed_state["rng"]["torch_cpu"])
    assert all(torch.equal(a, b) for a, b in zip(full_state["rng"]["torch_cuda"], resumed_state["rng"]["torch_cuda"]))
    def last_loss(path):
        return [json.loads(line)["loss"] for line in path.read_text().splitlines()
                if '"event":"update"' in line][-1]
    assert last_loss(full.run_dir / "train.jsonl") == last_loss(resumed.run_dir / "train.jsonl")
    gpu_evidence[0]["measurements"]["tiny_cuda_resume"] = {
        "steps": 3, "model_max_error": max_error, "cursor_equal": True,
        "cpu_cuda_rng_equal": True, "lr_equal": True, "last_loss_equal": True}
