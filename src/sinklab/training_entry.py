"""Strict one-run production entry. Actual GPU validation remains Stage 06."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path

import torch
from transformers import GPT2Config, GPT2LMHeadModel

from .config import MODELS, resolve_config
from .data import load_corpus
from .panels import validate_owt_panels
from .owt_compat import load_owt_corpus, validate_owt_panels as validate_upstream_owt_panels, RECIPE
from .hardware import (DIVISORS, HardwareError, authorize_production_device,
                       measure_candidate, validate_eligibility_matrix)
from .initialization import load_initialization
from .models import GPT2Adapter, ModelShape
from .provenance import canonical_json_bytes, payload_digest, validate_protocol_lock, verify_envelope
from .train import Trainer, make_objective_loss
from .interventions import AttentionIntervention
from .evaluate import RecordStore, cadence, evaluate_panel


def _read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_seed_replication_artifacts(repo: Path, root: Path, lock: dict,
                                      spec, args) -> None:
    """Verify the exact externally stored files bound by the D21 root."""
    from .stage06_readiness import D21_EVIDENCE_PATH
    evidence, evidence_sha = verify_envelope(_read(repo / D21_EVIDENCE_PATH))
    body = lock["protocol"]
    if evidence_sha != body["seed_replications_sha256"]:
        raise ValueError("seed replication evidence differs from the protocol")
    row = evidence["seeds"][str(spec.seed)]
    names = {
        "corpus": (f"corpus/seed{spec.seed}", f"owt-corpus-{row['corpus_sha256']}.json",
                   "corpus_file_sha256"),
        "panels": (f"panels/seed{spec.seed}", f"owt-panels-{row['panels_sha256']}.json",
                   "panels_file_sha256"),
        "order": (f"order/seed{spec.seed}", f"update-order-{row['order_sha256']}.json",
                  "order_file_sha256"),
        "initialization": (f"initialization/seed{spec.seed}",
                           f"init-seed{spec.seed}-{row['initialization_sha256']}.json",
                           "initialization_metadata_file_sha256"),
        "weights": (f"initialization/seed{spec.seed}",
                    f"init-seed{spec.seed}-{row['initialization_sha256']}.safetensors",
                    "initialization_weights_file_sha256"),
    }
    if any(root.is_relative_to(repo / name) for name in ("Upstream", "upstream")):
        raise ValueError("replication artifacts cannot depend on Upstream")
    for name, (folder, filename, hash_field) in names.items():
        path = root / folder / filename
        if not path.is_file() or _file_sha256(path) != row[hash_field]:
            raise ValueError(f"seed {spec.seed} {name} differs from D21 artifact evidence")
        if name in ("corpus", "panels", "initialization") and hasattr(args, name):
            if Path(getattr(args, name)).resolve() != path.resolve():
                raise ValueError(f"seed {spec.seed} {name} path differs from approved root")
    metadata, _ = verify_envelope(_read(root / names["initialization"][0] /
                                         names["initialization"][1]))
    if metadata.get("seed") != spec.seed or metadata.get("tensor_content_sha256") != row["initialization_sha256"]:
        raise ValueError("seed initialization identity differs from D21")
    order, order_sha = verify_envelope(_read(root / names["order"][0] /
                                             names["order"][1]))
    if order_sha != row["order_sha256"] or order.get("seed") != spec.seed:
        raise ValueError("seed update order identity differs from D21")


def _gpu_identity() -> tuple[str, str]:
    gpu = _gpu_metadata()
    return gpu["name"], gpu["uuid"]


def _gpu_metadata() -> dict:
    if not torch.cuda.is_available():
        raise HardwareError("approved training requires a CUDA GPU")
    result = subprocess.run(["nvidia-smi", "--query-gpu=name,uuid,memory.total,driver_version",
                             "--format=csv,noheader"],
                            capture_output=True, text=True, check=True)
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    fields = [part.strip() for part in lines[0].split(",")] if len(lines) == 1 else []
    if len(fields) != 4 or not fields[1]:
        raise HardwareError("one GPU with name/UUID/VRAM/driver is required")
    return {"name": fields[0], "uuid": fields[1], "vram_mib": int(fields[2].split()[0]),
            "driver": fields[3]}


def _check_runtime_environment(environment: dict, role: str, gpu: dict) -> None:
    import transformers
    expected = environment["devices"][role]
    installed = {row.metadata["Name"].lower().replace("_", "-"): row.version
                 for row in importlib.metadata.distributions()}
    locked = environment["lock_managed_distributions"]
    if (any(installed.get(name) != version for name, version in locked.items()) or
            set(installed) - set(locked) - {"pip", "uv"} or
            sys.version != expected["python"] or torch.__version__ != expected["torch"] or
            torch.version.cuda != expected["cuda_runtime"] or
            torch.backends.cudnn.version() != expected["cudnn"] or
            transformers.__version__ != expected["transformers"] or
            gpu["driver"] != expected["gpu"]["nvidia_driver"] or
            gpu["vram_mib"] != expected["gpu"]["total_vram_mib"]):
        raise HardwareError("actual software, numerical environment or GPU capacity differs from lock")


def preflight_approved_training(args) -> dict:
    """Validate a single prospective run without constructing models or training."""
    from .stage06_readiness import validate_final_lock_set
    from .runtime_provenance import validate_runtime_source
    lock_set = validate_final_lock_set(args.protocol_lock.parent)
    if _read(args.protocol_lock) != lock_set["protocol"]:
        raise ValueError("requested protocol lock differs from validated production root")
    raw, protocol, hardware_document = (_read(args.config), lock_set["protocol"],
                                        _read(args.hardware_plan))
    spec = resolve_config(raw, seed=args.seed, protocol_lock=protocol, production=True)
    lock, _ = validate_protocol_lock(protocol)
    source = validate_runtime_source(Path(__file__).resolve().parents[2],
                                     lock["production_runtime_source_commit"],
                                     lock["execution_critical_path_set_version"],
                                     loaded_package_dir=Path(__file__).resolve().parent)
    hardware, hardware_digest = verify_envelope(hardware_document)
    if (hardware_document != lock_set["hardware"] or
            hardware_digest != lock["hardware_lock_digest"] or
            hardware.get("evidence") != "measured_reference_plus_researcher_transfer"):
        raise HardwareError("approved reference-plus-transfer hardware lock required")
    plan = hardware["proof"]
    validate_eligibility_matrix(plan)
    if (plan.get("sha256") != payload_digest({k: v for k, v in plan.items() if k != "sha256"}) or
            (plan.get("microbatch"), plan.get("accumulation")) != (4, 16)):
        raise HardwareError("common batch schedule or reference proof changed")
    gpu = _gpu_metadata()
    identity = authorize_production_device(lock, hardware, condition=spec.condition,
                                           device_role=spec.device_role,
                                           gpu_name=gpu["name"], gpu_uuid=gpu["uuid"])
    environment, _ = verify_envelope(lock_set["environment"])
    _check_runtime_environment(environment, spec.device_role, gpu)
    result = {**identity, "gpu_driver": gpu["driver"], "gpu_vram_mib": gpu["vram_mib"],
              "protocol_sha256": protocol["sha256"], "hardware_sha256": hardware_digest,
              "runtime_source": source, "training_started": False}
    artifact_root = getattr(args, "artifact_root", None)
    if artifact_root is None and hasattr(args, "corpus"):
        artifact_root = (args.teacher_dir if spec.seed in (1, 2) else args.corpus).resolve().parent
        if spec.seed == 0:
            artifact_root = args.corpus.resolve().parents[1]
    if artifact_root is not None:
        from .stage06_artifacts import build_inventory
        artifact_root = Path(artifact_root).resolve()
        repo_root = Path(__file__).resolve().parents[2]
        if any(artifact_root.is_relative_to(repo_root / name)
               for name in ("Upstream", "upstream")):
            raise ValueError("production artifacts cannot depend on reference-only Upstream")
        if hasattr(args, "corpus"):
            expected = {"student_config": artifact_root / "student-config/config.json",
                        "teacher_dir": artifact_root / "teacher"}
            if spec.seed == 0:
                expected["corpus"] = next((artifact_root / "corpus").glob("owt-corpus-*.json"), None)
                expected["panels"] = next((artifact_root / "panels").glob("owt-panels-*.json"), None)
                expected["initialization"] = next((artifact_root / "initialization/seed0").glob("init-seed0-*.json"), None)
            if any(value is None or Path(getattr(args, key)).resolve() != value.resolve()
                   for key, value in expected.items()):
                raise ValueError("training inputs must use one verified production artifact root")
        inventory, _ = verify_envelope(build_inventory(artifact_root, repo_root))
        committed, _ = verify_envelope(_read(Path(__file__).resolve().parents[2] /
                                              "reports/stage06_production_artifact_inventory.json"))
        if {k: v for k, v in inventory.items() if k != "observed_external_root"} != {
                k: v for k, v in committed.items() if k != "observed_external_root"}:
            raise ValueError("production scientific artifact inventory differs from lock")
        result["scientific_artifacts_verified"] = True
        if spec.seed in (1, 2):
            if lock["protocol"].get("production_config_binding_schema") != 4:
                raise ValueError("optional C3 seed requires D21 production protocol")
            replication_root = getattr(args, "replication_root", None)
            if replication_root is None and hasattr(args, "corpus"):
                replication_root = args.corpus.resolve().parents[1]
            if replication_root is None:
                raise ValueError("optional seed preflight requires --replication-root")
            _check_seed_replication_artifacts(repo_root, Path(replication_root).resolve(),
                                              lock, spec, args)
            result["seed_replication_artifacts_verified"] = True
    return result


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


def _require_s1_data_lock(protocol: dict, corpus: dict, corpus_hash: str,
                          panel_hash: str, *, seed: int = 0) -> None:
    data = protocol.get("data", {})
    if not isinstance(data, dict) or data.get("recipe") != RECIPE:
        raise ValueError("approved S1 protocol must require upstream-compatible OWT recipe")
    required = {
        "dataset_revision": corpus["dataset"]["revision"],
        "tokenizer_revision": corpus["tokenizer"]["revision"],
        "tokenizer_files_sha256": corpus["tokenizer"]["files_sha256"],
    }
    if any(data.get(key) != value for key, value in required.items()):
        raise ValueError("S1 source or tokenizer differs from approved lock")
    if seed == 0:
        expected_corpus, expected_panels = data["production_corpus_sha256"], data["frozen_panels_sha256"]
    else:
        row = protocol["seed_replications"][str(seed)]
        expected_corpus, expected_panels = row["corpus_sha256"], row["panels_sha256"]
    if (("seed_replications" in protocol and corpus.get("seed") != seed) or
            corpus_hash != expected_corpus or
            panel_hash != expected_panels):
        raise ValueError("S1 seed, corpus or frozen panels differ from approved lock")


def run_approved_training(args) -> dict:
    preflight = preflight_approved_training(args)
    from .stage06_readiness import validate_final_lock_set
    lock_set = validate_final_lock_set(args.protocol_lock.parent)
    raw, protocol = _read(args.config), lock_set["protocol"]
    spec = resolve_config(raw, seed=args.seed, protocol_lock=protocol, production=True)
    lock, _ = validate_protocol_lock(protocol)
    hardware_digest = preflight["hardware_sha256"]
    plan = lock_set["hardware"]["payload"]["proof"]
    evaluation_rules = lock["protocol"].get("evaluation")
    if not isinstance(evaluation_rules, dict) or not isinstance(
        evaluation_rules.get("fingerprint_denominator_floor"), (float, int)
    ) or not math.isfinite(evaluation_rules["fingerprint_denominator_floor"]) or evaluation_rules["fingerprint_denominator_floor"] <= 0:
        raise ValueError("approved protocol must lock fingerprint denominator floor")
    if plan.get("sha256") != payload_digest({k: v for k, v in plan.items() if k != "sha256"}):
        raise HardwareError("hardware plan checksum mismatch")
    factors = lock["protocol"]["calibration"]
    if spec.condition == "C3":
        if args.mse_scale != factors["s_MSE"] or args.rel_scale is not None:
            raise ValueError("C3 requires the locked MSE factor and no REL factor")
    elif spec.condition == "C4":
        if args.rel_scale != factors["s_REL"] or args.mse_scale is not None:
            raise ValueError("C4 requires the locked REL factor and no MSE factor")
    elif args.mse_scale is not None or args.rel_scale is not None:
        raise ValueError("inactive objective scale must not be supplied")
    role, gpu_uuid = preflight["device_role"], preflight["gpu_uuid"]
    if spec.condition == "C4" and role != "rtx3090":
        raise HardwareError("C4 production is restricted to RTX 3090")
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
    if spec.study == "S1":
        corpus, corpus_hash = load_owt_corpus(args.corpus, tokenizer_sha256=tokenizer_hash)
    else:
        corpus, corpus_hash = load_corpus(args.corpus, tokenizer_sha256=tokenizer_hash)
    blocks = {b["id"]: b["token_ids"] for b in corpus["partitions"]["training"]["blocks"]}
    panel_doc = _read(args.panels)
    panel_hash = (validate_upstream_owt_panels(panel_doc, corpus_doc) if spec.study == "S1"
                  else validate_owt_panels(panel_doc, corpus_doc))
    if spec.study == "S1":
        _require_s1_data_lock(lock["protocol"], corpus, corpus_hash, panel_hash,
                              seed=spec.seed)
    evaluation_blocks = {b["id"]: b["token_ids"] for b in corpus["partitions"]["evaluation"]["blocks"]}
    model_hash = hashlib.sha256(canonical_json_bytes(config.to_dict())).hexdigest()
    identity = {"study": spec.study, "condition": spec.condition, "seed": spec.seed,
                "run_id": args.run_dir.name, "protocol_hash": spec.protocol_digest,
                "data_hash": corpus_hash, "init_hash": _read(args.initialization)["payload"]["tensor_content_sha256"],
                "hardware_hash": hardware_digest, "calibration_hash": lock["calibration_lock_digest"],
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
                      seed=spec.seed, order_scheme="upstream-owt-epoch-v1" if spec.study == "S1"
                      else "v2-horizon-independent")
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
    tokenizer_hash = corpus_doc["payload"]["tokenizer"]["files_sha256"]
    if spec.study == "S1":
        corpus, corpus_hash = load_owt_corpus(args.corpus, tokenizer_sha256=tokenizer_hash)
    else:
        corpus, corpus_hash = load_corpus(args.corpus, tokenizer_sha256=tokenizer_hash)
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
