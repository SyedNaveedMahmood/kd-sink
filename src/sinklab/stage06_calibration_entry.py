"""Run the frozen S1 raw-gradient calibration on the reviewed RTX 3090."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
import transformers
from transformers import GPT2Config, GPT2LMHeadModel

from .calibration import calibrate_initial_gradients
from .initialization import load_initialization
from .models import GPT2Adapter, ModelShape
from .objectives import METHOD_IDS, attention_auxiliary, relation_auxiliary
from .owt_compat import validate_calibration_blocks_export
from .provenance import canonical_json_bytes, seal_payload, verify_envelope
from .training_entry import _teacher_map


EXPECTED_3090_UUID = "GPU-a21766e4-bb31-9b79-5e8f-e58021e9708e"


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_calibration_inputs(args) -> tuple[dict, dict, dict]:
    """Reject wrong source, panel, model, initialization or common plan."""
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != args.source_commit:
        raise ValueError("calibration source commit differs from handoff commit")
    if subprocess.check_output(["git", "status", "--porcelain"], text=True).strip():
        raise ValueError("calibration requires a clean source worktree")
    candidate, _ = verify_envelope(json.loads(args.candidate_report.read_text(encoding="utf-8")))
    proof = candidate["proof"]
    if (candidate.get("kind") != "stage06-reviewed-batch-candidate-v1" or
            (proof["microbatch"], proof["accumulation"], proof["effective_sequences"]) != (4, 16, 64) or
            ["C4", "rtx3090", EXPECTED_3090_UUID] not in proof["required"]):
        raise ValueError("reviewed nine-job common batch proof is absent")
    panel, panel_digest = validate_calibration_blocks_export(
        json.loads(args.panel_export.read_text(encoding="utf-8")))
    frozen_panels, frozen_digest = verify_envelope(json.loads(args.panels.read_text(encoding="utf-8")))
    if (frozen_digest != panel["panels_sha256"] or
            frozen_panels["corpus_sha256"] != panel["corpus_sha256"] or
            frozen_panels["calibration16x64"] != [[block["id"] for block in batch]
                                                  for batch in panel["batches"]]):
        raise ValueError("calibration blocks differ from the frozen panel manifest")
    inventory, _ = verify_envelope(json.loads(args.artifact_inventory.read_text(encoding="utf-8")))
    if (inventory.get("kind") != "stage06-production-artifact-inventory-v1" or
            inventory["corpus"]["payload_sha256"] != panel["corpus_sha256"] or
            inventory["panels"]["payload_sha256"] != frozen_digest or
            inventory["calibration_export"]["payload_sha256"] != panel_digest or
            inventory["reviewed_plan_sha256"] != candidate["reviewed_plan_sha256"]):
        raise ValueError("calibration inputs differ from committed production artifact inventory")
    if panel["dataset_revision"] != "79d93d786212f7344586290adb811d4ae6a1762c":
        raise ValueError("calibration panel is not from the pinned OpenWebText revision")
    partial, _ = verify_envelope(json.loads(args.artifact_partial.read_text(encoding="utf-8")))
    if (_sha(args.teacher_dir / "model.safetensors") != partial["teacher"]["safetensors_sha256"] or
            _sha(args.teacher_dir / "config.json") != partial["teacher"]["config_sha256"] or
            _sha(args.student_config) != partial["student_config"]["config_sha256"] or
            panel["tokenizer_files_sha256"] != partial["tokenizer"]["files_sha256"]):
        raise ValueError("teacher, student config or tokenizer differs from pinned artifacts")
    if (inventory["teacher"]["weights_sha256"] != partial["teacher"]["safetensors_sha256"] or
            inventory["teacher"]["config_sha256"] != partial["teacher"]["config_sha256"] or
            inventory["student_config"]["sha256"] != partial["student_config"]["config_sha256"]):
        raise ValueError("inventory model hashes differ from pinned artifact evidence")
    init, _ = verify_envelope(json.loads(args.initialization.read_text(encoding="utf-8")))
    if init.get("seed") != 1729 or init.get("kind") != "gpt2-random-cpu-fp32-v1":
        raise ValueError("calibration requires seed-1729 CPU-FP32 initialization")
    if (inventory["calibration_initialization"]["tensor_content_sha256"] !=
            init["tensor_content_sha256"] or
            inventory["calibration_initialization"]["metadata_sha256"] != _sha(args.initialization)):
        raise ValueError("calibration initialization differs from committed inventory")
    return candidate, panel, {"digest": panel_digest, "partial": partial, "init": init}


def run_calibration(args) -> dict:
    candidate, panel, verified = verify_calibration_inputs(args)
    if not torch.cuda.is_available() or "3090" not in torch.cuda.get_device_name(0):
        raise ValueError("calibration requires the approved RTX 3090 as CUDA device 0")
    if "GPU-" + str(torch.cuda.get_device_properties(0).uuid) != EXPECTED_3090_UUID:
        raise ValueError("CUDA device 0 UUID differs from the approved RTX 3090")
    query = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"],
        text=True).splitlines()
    matches = [row.split(", ") for row in query if EXPECTED_3090_UUID in row]
    if len(matches) != 1 or "3090" not in matches[0][0]:
        raise ValueError("approved RTX 3090 UUID was not found")
    gpu_name, gpu_uuid, driver = matches[0]
    config = GPT2Config(**json.loads(args.student_config.read_text(encoding="utf-8")))
    if (config.n_layer, config.n_head, config.n_embd) != (24, 16, 1024):
        raise ValueError("calibration student must be GPT-2-medium configuration")
    config._attn_implementation = "eager"
    student_model = GPT2LMHeadModel(config)
    student_model.load_state_dict(load_initialization(args.initialization, config=config, seed=1729),
                                  strict=True)
    teacher_model = GPT2LMHeadModel.from_pretrained(
        str(args.teacher_dir), local_files_only=True, attn_implementation="eager")
    student = GPT2Adapter(student_model, ModelShape(24, 16, 1024))
    teacher = GPT2Adapter(teacher_model, ModelShape(36, 20, 1280))
    device = torch.device("cuda:0")
    student.model.to(device)
    teacher.model.to(device).eval().requires_grad_(False)
    mapping = _teacher_map("S1")
    if len(mapping) != 24 or mapping[-1] != 35:
        raise ValueError("S1 mapped teacher layers changed")

    def term(name: str):
        def raw(model: torch.nn.Module, cpu_ids: torch.Tensor):
            ids = cpu_ids.to(device, non_blocking=False)
            mask = torch.ones_like(ids, dtype=torch.bool)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                with torch.no_grad():
                    t = teacher.forward_with_features(input_ids=ids)
                s = student.forward_with_features(input_ids=ids)
                if name == "rel":
                    kinds = ("query", "key", "value")
                    teacher_layers = tuple({k: getattr(t.attention[index], k) for k in kinds}
                                           for index in mapping)
                    student_layers = tuple({k: getattr(feature, k) for k in kinds}
                                           for feature in s.attention)
                    loss = relation_auxiliary(teacher_layers, student_layers, mask)
                else:
                    method = METHOD_IDS["S1"]["C2" if name == "jsd" else "C3"]
                    loss = torch.stack([attention_auxiliary(
                        t.attention[index].probabilities, s.attention[layer].probabilities,
                        mask, method) for layer, index in enumerate(mapping)]).mean()
            return loss * ids.shape[0], ids.shape[0]
        return raw

    batches = [[torch.tensor([block["token_ids"] for block in batch[offset:offset + 4]],
                             dtype=torch.long) for offset in range(0, 64, 4)]
               for batch in panel["batches"]]
    evidence = calibrate_initial_gradients(
        student.model, batches, {name: term(name) for name in ("jsd", "mse", "rel")},
        architecture="s1_gpt2_large_to_random_medium", panel_manifest=verified["digest"],
        panel_role="training_calibration", seed=1729)
    result = seal_payload({"kind": "s1-c3-c4-raw-gradient-calibration-v1",
        "status": "measured_on_approved_rtx3090", "source_commit": args.source_commit,
        "gpu": {"name": gpu_name, "uuid": gpu_uuid, "driver": driver},
        "environment": {"python_torch": torch.__version__, "torch_cuda": torch.version.cuda,
                        "transformers": transformers.__version__},
        "candidate_batch_plan_sha256": candidate["proof"]["sha256"],
        "reviewed_plan_sha256": candidate["reviewed_plan_sha256"],
        "panel_export_sha256": verified["digest"],
        "frozen_panels_sha256": panel["panels_sha256"],
        "corpus_sha256": panel["corpus_sha256"],
        "student_config_file_sha256": _sha(args.student_config),
        "teacher_config_file_sha256": _sha(args.teacher_dir / "config.json"),
        "teacher_weights_file_sha256": _sha(args.teacher_dir / "model.safetensors"),
        "calibration_initialization_tensor_sha256": verified["init"]["tensor_content_sha256"],
        "calibration_initialization_metadata_sha256": _sha(args.initialization),
        "numerics": {"autocast": "cuda_bf16", "gradient_reduction": "fp32_global_l2",
                     "dropout": "disabled_during_measurement_only", "optimizer_updates": 0,
                     "effective_batch": 64, "microbatch": 4, "accumulation": 16,
                     "missing_gradients": "zero", "ratio_rule": "per_batch_jsd_over_raw_objective_median"},
        "raw_norms": {name: list(values) for name, values in evidence.raw_norms.items()},
        "ratios": {name: list(values) for name, values in evidence.ratios.items()},
        "factors": dict(evidence.factors), "c2_c5_c6_scale": 1})
    args.out_dir.mkdir(parents=True, exist_ok=True)
    target = args.out_dir / f"s1-calibration-{result['sha256']}.json"
    if target.exists():
        raise FileExistsError("calibration output already exists; refuse overwrite")
    target.write_bytes(canonical_json_bytes(result) + b"\n")
    return {"path": str(target), "sha256": result["sha256"],
            "s_mse": evidence.factors["mse"], "s_rel": evidence.factors["rel"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure frozen S1 C3/C4 raw gradients on RTX 3090")
    for flag in ("candidate-report", "panel-export", "panels", "artifact-inventory",
                 "artifact-partial", "teacher-dir",
                 "student-config", "initialization", "out-dir"):
        parser.add_argument(f"--{flag}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    args = parser.parse_args()
    print(json.dumps(run_calibration(args), sort_keys=True))


if __name__ == "__main__":
    main()
