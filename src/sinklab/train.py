"""Single-condition trainer with an absolute optimizer clock and exact CPU resume."""

from __future__ import annotations

import json
import hashlib
import math
import random
import signal
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

import numpy as np
import torch
from tqdm import tqdm

from .checkpoint import (CheckpointError, latest_full, load_checkpoint, verify_checkpoint,
                         prune_rolling, save_checkpoint, writer_lock)
from .order import UpdateOrder, microbatches
from .provenance import canonical_json_bytes


class TrainError(ValueError):
    pass


REQUIRED_IDENTITY = {"study", "condition", "seed", "run_id", "protocol_hash", "data_hash",
                     "init_hash", "hardware_hash", "calibration_hash", "model_hash", "precision",
                     "backend", "microbatch", "device_role"}
RETAIN = {0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000}


def update_lr(update: int, peak: float = 5e-4) -> float:
    if type(update) is not int or update < 1 or peak <= 0:
        raise TrainError("positive absolute update and peak LR required")
    if update <= 500:
        return peak * update / 500
    floor = peak * 0.1
    if update <= 10000:
        return floor + (peak - floor) * (1 + math.cos(math.pi * (update - 500) / 9500)) / 2
    return floor


def parameter_groups(model: torch.nn.Module, *, weight_decay: float = 0.1) -> list[dict]:
    seen: set[int] = set()
    decay, no_decay = [], []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or id(parameter) in seen:
            continue
        seen.add(id(parameter))
        (no_decay if parameter.ndim < 2 or name.endswith("bias") or "norm" in name.lower() or "ln_" in name.lower()
         else decay).append(parameter)
    if not seen:
        raise TrainError("student has no trainable parameters")
    return [{"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0}]


def make_optimizer(model: torch.nn.Module) -> torch.optim.AdamW:
    return torch.optim.AdamW(parameter_groups(model), lr=5e-4, betas=(0.9, 0.95), eps=1e-8)


def _rng_state() -> dict:
    numpy = np.random.get_state()
    return {"python": random.getstate(), "numpy": (numpy[0], numpy[1].tolist(), *numpy[2:]),
            "torch_cpu": torch.get_rng_state(),
            "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def _restore_rng(state: dict) -> None:
    random.setstate(state["python"])
    n = state["numpy"]
    np.random.set_state((n[0], np.asarray(n[1], dtype=np.uint32), *n[2:]))
    torch.set_rng_state(state["torch_cpu"])
    if state["torch_cuda"]:
        torch.cuda.set_rng_state_all(state["torch_cuda"])


def _event(path: Path, row: dict) -> None:
    with path.open("ab") as stream:
        stream.write(canonical_json_bytes(row) + b"\n")


@dataclass
class TrainingState:
    step: int = 0
    elapsed_seconds: float = 0.0
    input_tokens: int = 0
    target_tokens: int = 0


class Trainer:
    """One already resolved run. ``loss_fn`` returns loss and optional active scalars.

    The fixture path may use shorter fixed blocks; production callers use 128 tokens.
    Model construction and shared random initialization happen before this boundary.
    """

    def __init__(self, *, model: torch.nn.Module, blocks: Mapping[str, list[int]],
                 loss_fn: Callable[[torch.Tensor], tuple[torch.Tensor, Mapping[str, float]]],
                 identity: dict, run_dir: Path, device: torch.device,
                 microbatch: int, seed: int, engineering_fixture: bool = False,
                 order_scheme: str = "v2-horizon-independent"):
        if (set(identity) not in (REQUIRED_IDENTITY, REQUIRED_IDENTITY | {"gpu_uuid"}) or
                identity["seed"] != seed or identity["microbatch"] != microbatch):
            raise TrainError("immutable single-run identity is incomplete or inconsistent")
        if identity["study"] not in {"S1", "S3"} or identity["condition"] not in {f"C{i}" for i in range(7)}:
            raise TrainError("one explicit supported study/condition required")
        if identity["condition"] == "C4" and not engineering_fixture and identity.get("device_role") != "rtx3090":
            raise TrainError("production C4 requires RTX 3090")
        if not engineering_fixture and (not isinstance(identity.get("gpu_uuid"), str) or
                                        not identity["gpu_uuid"].strip()):
            raise TrainError("production run identity requires the actual physical GPU UUID")
        if not engineering_fixture and (device.type != "cuda" or identity["precision"] != "bf16"):
            raise TrainError("production training requires pinned CUDA/BF16 plan")
        if not engineering_fixture and any(p.dtype != torch.float32 for p in model.parameters() if p.requires_grad):
            raise TrainError("production student trainable parameters must remain FP32")
        if not engineering_fixture and identity["study"] == "S1" and order_scheme != "upstream-owt-epoch-v1":
            raise TrainError("production S1 requires pinned upstream OWT block order")
        if microbatch not in (1, 2, 4, 8, 16, 32, 64) or not blocks:
            raise TrainError("microbatch must divide 64 and block manifest must be nonempty")
        lengths = {len(v) for v in blocks.values()}
        if len(lengths) != 1 or min(lengths) < 2 or (not engineering_fixture and lengths != {128}):
            raise TrainError("training blocks need one fixed sequence length (128 in production)")
        self.model, self.blocks, self.loss_fn = model.to(device), blocks, loss_fn
        self.identity, self.run_dir, self.device = identity, Path(run_dir), device
        self.microbatch, self.seed = microbatch, seed
        self.fixture = engineering_fixture
        self.optimizer = make_optimizer(model)
        self.order_scheme = order_scheme
        self.order = UpdateOrder(list(blocks), seed=seed, scheme=order_scheme)
        self.state = TrainingState()
        self.peak_lr = 5e-4
        self._started = False
        self.extension_id: str | None = None
        self.extension_parent_sha256: str | None = None
        self._extension_source_verified = False

    def _snapshot(self) -> dict:
        return {"schema_version": 1, "optimizer": self.optimizer.state_dict(),
                "scheduler": {"formula": "warmup500-cosine10000-floor-v1", "peak_lr": self.peak_lr,
                              "last_completed_update": self.state.step,
                              "next_lr": update_lr(self.state.step + 1, self.peak_lr)},
                "rng": _rng_state(), "order": self.order.snapshot(),
                "extension_id": self.extension_id,
                "extension_parent_sha256": self.extension_parent_sha256,
                "counters": vars(self.state).copy(),
                "backend": {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                            "cudnn_deterministic": torch.backends.cudnn.deterministic,
                            "cudnn_benchmark": torch.backends.cudnn.benchmark},
                "scaler": None}

    def resume(self, path: Path | None = None) -> None:
        path = path or latest_full(self.run_dir / "checkpoints", identity=self.identity)
        manifest, saved = load_checkpoint(path, model=self.model, identity=self.identity)
        if saved is None or saved["schema_version"] != 1 or saved["scaler"] is not None:
            raise TrainError("full unscaled trainer state required")
        sched = saved["scheduler"]
        if sched != {"formula": "warmup500-cosine10000-floor-v1", "peak_lr": self.peak_lr,
                     "last_completed_update": saved["counters"]["step"],
                     "next_lr": update_lr(saved["counters"]["step"] + 1, self.peak_lr)}:
            raise TrainError("scheduler formula or absolute clock mismatch")
        self.optimizer.load_state_dict(saved["optimizer"])
        self.order = UpdateOrder.resume(list(self.blocks), saved["order"],
                                        expected_scheme=self.order_scheme)
        self.state = TrainingState(**saved["counters"])
        self.extension_id = saved["extension_id"]
        self.extension_parent_sha256 = saved.get("extension_parent_sha256")
        if self.extension_parent_sha256 is not None:
            self._verify_extension_parent()
        self._extension_source_verified = (
            self.extension_parent_sha256 is not None or
            (self.state.step == 10000 and path.name == "final-010000" and manifest["kind"] == "final")
        )
        length = len(next(iter(self.blocks.values())))
        if (manifest["step"] != self.state.step or
                self.state.input_tokens != self.state.step * 64 * length or
                self.state.target_tokens != self.state.step * 64 * (length - 1)):
            raise TrainError("checkpoint step or token counters disagree")
        if self.order.presentations != self.state.step * 64:
            raise TrainError("data cursor and optimizer clock disagree")
        backend = saved["backend"]
        if backend != {"deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
                       "cudnn_deterministic": torch.backends.cudnn.deterministic,
                       "cudnn_benchmark": torch.backends.cudnn.benchmark}:
            raise TrainError("backend determinism flags changed")
        _restore_rng(saved["rng"])
        self._started = True

    def _verify_extension_parent(self) -> str:
        anchor = self.run_dir / "checkpoints" / "final-010000"
        manifest = verify_checkpoint(anchor, identity=self.identity)
        if manifest["step"] != 10000 or manifest["kind"] != "final":
            raise TrainError("extension requires protected full step-10000 checkpoint")
        digest = hashlib.sha256((anchor / "manifest.json").read_bytes()).hexdigest()
        if self.extension_parent_sha256 is not None and self.extension_parent_sha256 != digest:
            raise TrainError("extension parent checkpoint changed")
        return digest

    def _save(self, kind: str) -> Path:
        return save_checkpoint(self.run_dir / "checkpoints", step=self.state.step, kind=kind,
                               model=self.model, state=None if kind == "weights" else self._snapshot(),
                               identity=self.identity)

    def _one_update(self) -> tuple[float, dict]:
        next_step = self.state.step + 1
        for group in self.optimizer.param_groups:
            group["lr"] = update_lr(next_step, self.peak_lr)
        self.optimizer.zero_grad(set_to_none=True)
        samples = self.order.take_update()
        losses = []
        diagnostics: dict[str, float | None] = {"ce": None, "kd": None,
                                                "attention": None, "relation": None}
        for batch in microbatches(samples, self.microbatch):
            ids = torch.tensor([self.blocks[item["block_id"]] for item in batch],
                               dtype=torch.long, device=self.device)
            with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16,
                                enabled=self.device.type == "cuda" and self.identity["precision"] == "bf16"):
                loss, active = self.loss_fn(ids)
            if loss.ndim or not torch.isfinite(loss):
                raise TrainError("nonfinite or nonscalar objective")
            (loss * len(batch) / 64).backward()
            losses.append(float(loss.detach()) * len(batch) / 64)
            for name, value in active.items():
                if name not in diagnostics:
                    raise TrainError(f"unknown active objective component: {name}")
                diagnostics[name] = (diagnostics[name] or 0.) + float(value) * len(batch) / 64
        grad_norm = float(torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0))
        if not math.isfinite(grad_norm):
            raise TrainError("nonfinite accumulated gradient")
        self.optimizer.step()
        self.state.step = next_step
        self.state.input_tokens += 64 * len(next(iter(self.blocks.values())))
        self.state.target_tokens += 64 * (len(next(iter(self.blocks.values()))) - 1)
        return sum(losses), {"grad_norm": grad_norm, **diagnostics}

    def train(self, *, stop_after: int, extension_id: str | None = None,
              evaluate: Callable[[int], None] | None = None,
              stop_requested: Callable[[], bool] | None = None,
              text_interval: int = 20, terminal: bool | None = None) -> TrainingState:
        if type(stop_after) is not int or stop_after < self.state.step or (stop_after > 10000 and not extension_id):
            raise TrainError("stop-after is an absolute stop, not a schedule change; extension ID required past 10k")
        if stop_after > 10000:
            if self.state.step < 10000 or not self._started or not self._extension_source_verified:
                raise TrainError("extension must continue from the protected 10k state")
            parent = self._verify_extension_parent()
            if self.state.step > 10000 and self.extension_id is None:
                raise TrainError("resumed extension is missing its lineage ID")
            if self.extension_id is not None and self.extension_id != extension_id:
                raise TrainError("extension lineage changed on resume")
            self.extension_id = extension_id
            self.extension_parent_sha256 = parent
        if terminal is None:
            terminal = sys.stderr.isatty()
        self.run_dir.mkdir(parents=True, exist_ok=True)
        event_path = self.run_dir / "train.jsonl"
        with writer_lock(self.run_dir):
            if not self._started:
                random.seed(self.seed)
                np.random.seed(self.seed % 2**32)
                torch.manual_seed(self.seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(self.seed)
                self._started = True
                self._save("rolling")
                self._save("weights")
                if evaluate is not None:
                    evaluate(0)
            elif evaluate is not None and (self.state.step % 100 == 0 or self.state.step == 250):
                evaluate(self.state.step)  # idempotent cache completes interrupted evaluation
            banner = {"event": "start", "study": self.identity["study"],
                      "condition": self.identity["condition"], "seed": self.seed,
                      "run_id": self.identity["run_id"], "gpu": str(self.device),
                      "gpu_name": torch.cuda.get_device_name(self.device) if self.device.type == "cuda" else None,
                      "gpu_uuid": self.identity.get("gpu_uuid"),
                      "gpu_total_bytes": torch.cuda.get_device_properties(self.device).total_memory if self.device.type == "cuda" else None,
                      "microbatch": self.microbatch, "accumulation": 64 // self.microbatch,
                      "effective_batch": 64, "input_tokens_per_update": 64 * len(next(iter(self.blocks.values()))),
                      "target_tokens_per_update": 64 * (len(next(iter(self.blocks.values()))) - 1),
                      "precision": self.identity["precision"], "total_updates": stop_after,
                      "registered_horizon": 10000,
                      "extension_id": self.extension_id,
                      "extension_parent_sha256": self.extension_parent_sha256,
                      "checkpoint_path": str(self.run_dir / "checkpoints"),
                      "protocol_hash": self.identity["protocol_hash"], "resumed_step": self.state.step}
            _event(event_path, banner)
            if not terminal:
                print(json.dumps(banner, sort_keys=True), flush=True)
            progress = tqdm(total=stop_after, initial=self.state.step, disable=not terminal,
                            desc=f'{self.identity["study"]}/{self.identity["condition"]}/seed{self.seed}', unit="update")
            interrupted = False
            previous_handlers = {}
            if threading.current_thread() is threading.main_thread():
                def request_stop(signum, frame):
                    nonlocal interrupted
                    interrupted = True
                for signum in (signal.SIGINT, signal.SIGTERM):
                    previous_handlers[signum] = signal.getsignal(signum)
                    signal.signal(signum, request_stop)
            try:
                while self.state.step < stop_after:
                    if interrupted or (stop_requested is not None and stop_requested()):
                        break
                    started = time.perf_counter()
                    loss, diagnostics = self._one_update()
                    duration = time.perf_counter() - started
                    self.state.elapsed_seconds += duration
                    step = self.state.step
                    row = {"event": "update", "step": step, "elapsed_seconds": self.state.elapsed_seconds,
                           "eta_train_seconds": max(stop_after - step, 0) * self.state.elapsed_seconds / step,
                           "input_tokens": self.state.input_tokens, "target_tokens": self.state.target_tokens,
                           "tokens_per_second": self.state.input_tokens / max(self.state.elapsed_seconds, 1e-9),
                           "loss": loss, "lr": self.optimizer.param_groups[0]["lr"],
                           "memory_allocated_bytes": torch.cuda.memory_allocated(self.device) if self.device.type == "cuda" else 0,
                           "memory_reserved_bytes": torch.cuda.memory_reserved(self.device) if self.device.type == "cuda" else 0,
                           **diagnostics}
                    _event(event_path, row)
                    progress.update(1)
                    progress.set_postfix(loss=f"{loss:.4g}", lr=f'{row["lr"]:.3g}',
                                         tok_s=f'{row["tokens_per_second"]:.0f}', mem=row["memory_allocated_bytes"])
                    if not terminal and (step % text_interval == 0 or step == stop_after):
                        print(json.dumps(row, sort_keys=True), flush=True)
                    if step in RETAIN:
                        self._save("weights")
                    if step % 500 == 0:
                        self._save("rolling")
                        prune_rolling(self.run_dir / "checkpoints", identity=self.identity)
                    if step == 10000:
                        self._save("final")  # preserve continuation even if endpoint evaluation fails
                    if evaluate is not None and (step % 100 == 0 or step == 250):
                        evaluate(step)
                if self.state.step == 10000 or (self.state.step > 10000 and self.state.step == stop_after):
                    self._save("final")
                else:
                    self._save("rolling")
                    prune_rolling(self.run_dir / "checkpoints", identity=self.identity)
            finally:
                progress.close()
                for signum, handler in previous_handlers.items():
                    signal.signal(signum, handler)
        return self.state


def make_objective_loss(*, study: str, condition: str, student_adapter,
                        teacher_adapter=None, teacher_map: tuple[int, ...] = (),
                        mse_scale: float | None = None, rel_scale: float | None = None):
    """Bind existing pure objectives to one student/teacher adapter pair."""
    from .objectives import compose_objective

    if condition != "C0" and teacher_adapter is None:
        raise TrainError("distillation requires a pinned frozen teacher")
    if teacher_adapter is not None:
        teacher_adapter.model.eval()
        teacher_adapter.model.requires_grad_(False)
    if condition in {"C2", "C3", "C4", "C5", "C6"} and (
        len(teacher_map) != student_adapter.layer_count or
        len(set(teacher_map)) != len(teacher_map) or
        min(teacher_map) < 0 or max(teacher_map) >= teacher_adapter.layer_count
    ):
        raise TrainError("explicit one-to-one teacher layer map required")

    def loss_fn(ids: torch.Tensor):
        mask = torch.ones_like(ids, dtype=torch.bool)
        auxiliary = condition in {"C2", "C3", "C4", "C5", "C6"}
        with torch.no_grad():
            teacher = (teacher_adapter.forward_with_features(input_ids=ids) if auxiliary else
                       teacher_adapter.forward(input_ids=ids)) if teacher_adapter is not None else None
        student = (student_adapter.forward_with_features(input_ids=ids) if auxiliary else
                   student_adapter.forward(input_ids=ids))
        s_logits = student.outputs.logits if auxiliary else student.logits
        attention_pairs = ()
        relation_pairs = None
        if auxiliary:
            attention_pairs = tuple((teacher.attention[t].probabilities, student.attention[s].probabilities,
                                     teacher.attention[t].scores, student.attention[s].scores)
                                    for s, t in enumerate(teacher_map))
            if condition == "C4":
                def relation(feature):
                    return {k: getattr(feature, k) for k in ("query", "key", "value")}
                relation_pairs = (tuple(relation(teacher.attention[t]) for t in teacher_map),
                                  tuple(relation(f) for f in student.attention))
        result = compose_objective(study, condition, s_logits, ids, mask,
                                   teacher_logits=teacher.outputs.logits if auxiliary else
                                   (teacher.logits if teacher is not None else None),
                                   attention_pairs=attention_pairs, relation_pairs=relation_pairs,
                                   mse_scale=mse_scale, rel_scale=rel_scale)
        return result.total, {name: float(value.detach()) for name, value in result.active.items()}
    return loss_fn
