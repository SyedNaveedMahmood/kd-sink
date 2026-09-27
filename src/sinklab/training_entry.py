"""Strict one-run production entry. Actual GPU validation remains Stage 06."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .config import MODELS, resolve_config
from .data import load_corpus
from .panels import validate_owt_panels
from .hardware import DIVISORS, HardwareError, measure_candidate
from .initialization import load_initialization
from .models import GPT2Adapter, ModelShape
from .provenance import canonical_json_bytes, payload_digest, validate_protocol_lock, verify_envelope
from .train import Trainer, make_objective_loss
from .interventions import AttentionIntervention
from .evaluate import RecordStore, cadence, evaluate_panel


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _gpu_identity() -> tuple[str, str]:
    if not torch.cuda.is_available():
        raise HardwareError("approved training requires a CUDA GPU")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid", "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    first = result.stdout.splitlines()[0].split(", ", 1)
    if len(first) != 2:
        raise HardwareError("GPU name/UUID query failed")
    return first[0], first[1]


def _teacher_map(study: str) -> tuple[int, ...]:
    if study == "S1":
        return tuple((s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1) for s in range(24))
    return (1, 3, 5, 7, 9, 11)


def _model_digest(model: torch.nn.Module) -> str:
    """Stream exact current tensor content into the evaluation cache identity."""
    h = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(tensor.dtype).encode("ascii") + b"\0")
        h.update(canonical_json_bytes({"shape": list(tensor.shape)}))
        h.update(tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return h.hexdigest()


def run_approved_training(args) -> dict:
    raw, protocol, plan = _read(args.config), _read(args.protocol_lock), _read(args.hardware_plan)
    spec = resolve_config(raw, seed=args.seed, protocol_lock=protocol, production=True)
    lock, _ = validate_protocol_lock(protocol)
    evaluation_rules = lock["protocol"].get("evaluation")
    if not isinstance(evaluation_rules, dict) or not isinstance(
        evaluation_rules.get("fingerprint_denominator_floor"), (float, int)
    ) or not math.isfinite(evaluation_rules["fingerprint_denominator_floor"]) or evaluation_rules["fingerprint_denominator_floor"] <= 0:
        raise ValueError("approved protocol must lock fingerprint denominator floor")
    if plan.get("sha256") != payload_digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise HardwareError("hardware plan checksum mismatch")
    if plan.get("evidence") != "measured_gpu" or lock["hardware_lock_digest"] != plan["sha256"]:
        raise HardwareError("approved measured hardware lock required")
    if plan.get("microbatch") not in DIVISORS or plan.get("accumulation") != 64 // plan["microbatch"]:
        raise HardwareError("common batch schedule is malformed")
    gpu_name, gpu_uuid = _gpu_identity()
    role = "rtx3090" if "3090" in gpu_name else "rtx4080super" if "4080 SUPER" in gpu_name.upper() else None
    if role != spec.device_role or [spec.condition, role, gpu_uuid] not in plan["required"]:
        raise HardwareError("actual GPU name/UUID is absent from approved eligibility")
    if spec.condition == "C4" and role != "rtx3090":
        raise HardwareError("C4 production restricted to RTX 3090")
    config = GPT2Config(**_read(args.student_config))
    expected = MODELS[spec.study][1]
    if (config.n_layer, config.n_head, config.n_embd) != (expected["layers"], expected["heads"], expected["width"]):
        raise ValueError("student config differs from approved study architecture")
    config._attn_implementation = "eager"
    student = GPT2LMHeadModel(config)
    student.load_state_dict(load_initialization(args.initialization, config=config, seed=spec.seed), strict=True)
    teacher = GPT2LMHeadModel.from_pretrained(str(args.teacher_dir), local_files_only=True,
                                               attn_implementation="eager")
    teacher_shape = MODELS[spec.study][0]
    teacher = GPT2Adapter(teacher, ModelShape(teacher_shape["layers"], teacher_shape["heads"],
                                               teacher_shape["width"]))
    student = GPT2Adapter(student, ModelShape(expected["layers"], expected["heads"], expected["width"]))
    corpus_doc = _read(args.corpus)
    tokenizer_hash = corpus_doc["payload"]["tokenizer"]["files_sha256"]
    corpus, corpus_hash = load_corpus(args.corpus, tokenizer_sha256=tokenizer_hash)
    blocks = {b["id"]: b["token_ids"] for b in corpus["partitions"]["training"]["blocks"]}
    panel_doc = _read(args.panels)
    panel_hash = validate_owt_panels(panel_doc, corpus_doc)
    evaluation_blocks = {b["id"]: b["token_ids"] for b in corpus["partitions"]["evaluation"]["blocks"]}
    model_hash = hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()
    identity = {"study": spec.study, "condition": spec.condition, "seed": spec.seed,
                "run_id": args.run_dir.name, "protocol_hash": spec.protocol_digest,
                "data_hash": corpus_hash, "init_hash": _read(args.initialization)["payload"]["tensor_content_sha256"],
                "hardware_hash": plan["sha256"], "calibration_hash": lock["calibration_lock_digest"],
                "model_hash": model_hash, "precision": "bf16", "backend": "eager",
                "microbatch": plan["microbatch"], "device_role": role, "gpu_uuid": gpu_uuid}
    device = torch.device("cuda:0")
    student.model.to(device)
    teacher.model.to(device).eval().requires_grad_(False)
    loss = make_objective_loss(study=spec.study, condition=spec.condition, student_adapter=student,
                               teacher_adapter=None if spec.condition == "C0" else teacher,
                               teacher_map=_teacher_map(spec.study),
                               mse_scale=args.mse_scale, rel_scale=args.rel_scale)
    trainer = Trainer(model=student.model, blocks=blocks, loss_fn=loss, identity=identity,
                      run_dir=args.run_dir, device=device, microbatch=plan["microbatch"],
                      seed=spec.seed)
    if (args.run_dir / "checkpoints").exists():
        trainer.resume()
    store = RecordStore(args.run_dir / "evaluation")
    teacher_digest = _model_digest(teacher.model)
    def evaluate_step(step: int) -> None:
        for panel_name in cadence(step):
            ids = panel_doc["payload"][panel_name]
            items = [{"id": ident, "input_ids": evaluation_blocks[ident],
                      "attention_mask": [1] * 128} for ident in ids]
            result = evaluate_panel(adapter=student, items=items, panel=panel_name,
                                    panel_hash=panel_hash, checkpoint_hash=_model_digest(student.model),
                                    run_id=identity["run_id"], step=step, store=store,
                                    teacher_adapter=None if panel_name == "owt_lm2000" else teacher,
                                    teacher_map=_teacher_map(spec.study),
                                    operations=("clean",) if panel_name == "owt_lm2000" else
                                               ("clean", "delete", "relocate"),
                                    precision="bf16", behavior_only=panel_name == "owt_lm2000",
                                    denominator_floor=evaluation_rules["fingerprint_denominator_floor"],
                                    run_identity={"study": spec.study, "condition": spec.condition,
                                                  "seed": spec.seed, "model_sha256": model_hash,
                                                  "corpus_sha256": corpus_hash,
                                                  "protocol_sha256": spec.protocol_digest})
            if any(row["status"] != "complete" for row in result["operations"].values()):
                raise ValueError(f"evaluation incomplete: {panel_name} step {step}; inspect item records")
            if panel_name != "owt_lm2000":
                teacher_result = evaluate_panel(adapter=teacher, items=items, panel=panel_name,
                    panel_hash=panel_hash, checkpoint_hash=teacher_digest,
                    run_id=identity["run_id"], step=step, store=store,
                    layer_scope=_teacher_map(spec.study), operations=("clean", "delete", "relocate"),
                    precision="bf16", model_role="teacher",
                    denominator_floor=evaluation_rules["fingerprint_denominator_floor"],
                    run_identity={"study": spec.study, "condition": spec.condition,
                                  "seed": spec.seed, "model_sha256": teacher_digest,
                                  "corpus_sha256": corpus_hash, "protocol_sha256": spec.protocol_digest})
                if any(row["status"] != "complete" for row in teacher_result["operations"].values()):
                    raise ValueError(f"teacher evaluation incomplete: {panel_name} step {step}")
    result = trainer.train(stop_after=args.stop_after, extension_id=args.extension_id,
                           evaluate=evaluate_step)
    return {"action": "single_run_stopped", "study": spec.study, "condition": spec.condition,
            "seed": spec.seed, "run_id": identity["run_id"], "step": result.step,
            "input_tokens": result.input_tokens, "shifted_targets": result.target_tokens}


def run_profile_candidate(args) -> dict:
    """This command profiles exactly one candidate; launcher must isolate children."""
    raw = _read(args.config)
    spec = resolve_config(raw, seed=args.seed, production=False)
    if args.microbatch not in DIVISORS:
        raise HardwareError("candidate must divide 64")
    name, uuid = _gpu_identity()
    role = "rtx3090" if "3090" in name else "rtx4080super" if "4080 SUPER" in name.upper() else None
    if role != spec.device_role or (spec.condition == "C4" and role != "rtx3090"):
        raise HardwareError("actual GPU role differs from requested eligible role")
    expected = MODELS[spec.study][1]
    config = GPT2Config(**_read(args.student_config))
    if (config.n_layer, config.n_head, config.n_embd) != (expected["layers"], expected["heads"], expected["width"]):
        raise HardwareError("profile must use full study architecture")
    config._attn_implementation = "eager"
    student_model = GPT2LMHeadModel(config)
    student_model.load_state_dict(load_initialization(args.initialization, config=config, seed=spec.seed), strict=True)
    student = GPT2Adapter(student_model, ModelShape(expected["layers"], expected["heads"], expected["width"]))
    teacher_model = GPT2LMHeadModel.from_pretrained(str(args.teacher_dir), local_files_only=True,
                                                     attn_implementation="eager")
    shape = MODELS[spec.study][0]
    teacher = GPT2Adapter(teacher_model, ModelShape(shape["layers"], shape["heads"], shape["width"]))
    teacher.model.cuda().eval().requires_grad_(False)
    student.model.cuda()
    corpus_doc = _read(args.corpus)
    corpus, corpus_hash = load_corpus(args.corpus,
                                     tokenizer_sha256=corpus_doc["payload"]["tokenizer"]["files_sha256"])
    blocks = {b["id"]: b["token_ids"] for partition in corpus["partitions"].values()
              for b in partition["blocks"]}
    panel, _ = verify_envelope(_read(args.panels))
    if panel.get("corpus_sha256") != corpus_hash or len(panel.get("owt_dense64", [])) != 64:
        raise HardwareError("profile requires frozen dense64 panel from same corpus")
    training = {b["id"]: b["token_ids"] for b in corpus["partitions"]["training"]["blocks"]}
    if not all(i in blocks for i in panel["owt_dense64"]):
        raise HardwareError("profile panel references missing blocks")
    identity = {"study": spec.study, "condition": spec.condition, "seed": spec.seed,
                "run_id": args.profile_dir.name, "protocol_hash": spec.protocol_digest or "unapproved-profile",
                "data_hash": corpus_hash, "init_hash": _read(args.initialization)["payload"]["tensor_content_sha256"],
                "hardware_hash": "candidate-only", "calibration_hash": "candidate-only",
                "model_hash": hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest(),
                "precision": "bf16", "backend": "eager", "microbatch": args.microbatch,
                "device_role": role, "gpu_uuid": uuid}
    loss = make_objective_loss(study=spec.study, condition=spec.condition, student_adapter=student,
                               teacher_adapter=None if spec.condition == "C0" else teacher,
                               teacher_map=_teacher_map(spec.study),
                               mse_scale=args.mse_scale, rel_scale=args.rel_scale)
    trainer = Trainer(model=student.model, blocks=training, loss_fn=loss, identity=identity,
                      run_dir=args.profile_dir, device=torch.device("cuda:0"),
                      microbatch=args.microbatch, seed=spec.seed, engineering_fixture=True)
    import random
    import numpy as np
    random.seed(spec.seed)
    np.random.seed(spec.seed % 2**32)
    torch.manual_seed(spec.seed)
    torch.cuda.manual_seed_all(spec.seed)

    def evaluation_pass():
        was_training = student.model.training
        student.model.eval()
        try:
            with torch.inference_mode():
                for block_id in panel["owt_dense64"]:
                    ids = torch.tensor([blocks[block_id]], dtype=torch.long, device="cuda:0")
                    student.forward(input_ids=ids)
                    for operation in ("delete", "relocate"):
                        student.forward(input_ids=ids, intervention=AttentionIntervention(operation))
                    teacher.forward(input_ids=ids)
                    for operation in ("delete", "relocate"):
                        teacher.forward(input_ids=ids, intervention=AttentionIntervention(operation),
                                        layer_scope=_teacher_map(spec.study))
        finally:
            student.model.train(was_training)

    started = time.perf_counter()
    args.profile_dir.mkdir(parents=True, exist_ok=False)
    from .checkpoint import writer_lock
    try:
        with writer_lock(args.profile_dir):
            profile = measure_candidate(condition=spec.condition, device_role=role,
                                        device_uuid=uuid, microbatch=args.microbatch,
                                        train_update=trainer._one_update, evaluate=evaluation_pass,
                                        save=lambda: trainer._save("rolling"),
                                        headroom_bytes=max(1610612736, int(torch.cuda.get_device_properties(0).total_memory * .1)))
    except Exception as exc:
        failure = {"status": "failed", "evidence": "measured_gpu_candidate_failure",
                   "condition": spec.condition, "device_role": role, "gpu_uuid": uuid,
                   "microbatch": args.microbatch, "error_type": type(exc).__name__,
                   "error": str(exc), "profile_dir": str(args.profile_dir)}
        args.output.write_bytes(canonical_json_bytes(failure) + b"\n")
        raise
    report = {"profile": asdict(profile), "gpu_name": name, "wall_seconds": time.perf_counter() - started,
              "profile_dir": str(args.profile_dir), "evidence": "measured_gpu", "status": "complete"}
    args.output.write_bytes(canonical_json_bytes(report) + b"\n")
    return report
