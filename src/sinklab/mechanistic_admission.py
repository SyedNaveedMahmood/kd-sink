"""Prospective admission for a single scientific mechanistic inference job.

Operator supplies the externally approved envelope digest explicitly. No draft
can authorize itself, and historical S1/S4 locks are never rewritten here.
"""
from __future__ import annotations

import importlib.metadata
import platform
from pathlib import Path

import torch

from .calibrated_probes import require, validate_panel_split
from .followup_policy import D24_SHA256, admit_s1_followup
from .mechanistic_run import VERSION, read_json, sha256_file, validate_items, validate_settings
from .provenance import payload_digest, verify_envelope, SHA256_PATTERN, COMMIT_PATTERN

E1_GRID = tuple(f"{c}/step{s}" for c in ("C1", "C2", "C3", "C5", "C6")
                for s in ((500,10000) if c in {"C1","C5"} else (500,2000,10000)))
E2_GRID = tuple(f"{c}/step{s}" for c in ("C1", "C2", "C3", "C5", "C6") for s in (500,2000,10000))


def runtime_identity(repo, device):
    """Bind every package module, the new runner, existing pins and actual device."""
    repo = Path(repo)
    files = sorted(repo.joinpath("src/sinklab").rglob("*.py")) + [repo/"scripts/run_mechanistic.py", repo/"scripts/report_mechanistic.py", repo/"pyproject.toml", repo/"uv.lock"]
    code = {p.relative_to(repo).as_posix(): sha256_file(p) for p in files}
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "numpy", "scipy", "safetensors", "tqdm")}
    device = torch.device(device)
    hardware = {"device": str(device), "name": "CPU", "uuid": None, "total_memory_bytes": None}
    if device.type == "cuda":
        require(device.index is not None, "fixed CUDA device index required")
        props = torch.cuda.get_device_properties(device)
        hardware.update(name=props.name, uuid=str(props.uuid), total_memory_bytes=props.total_memory)
    return {"code_sha256": payload_digest(code), "code_files": code,
        "environment": {"python": platform.python_version(), "packages": versions, "torch_cuda": torch.version.cuda,
            "platform": platform.platform(), "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "tf32_cudnn": torch.backends.cudnn.allow_tf32}, "hardware": hardware,
        "precision": "float32", "backend": "eager", "context_length": 128, "use_cache": False}


def _pinned(reference):
    require(isinstance(reference, dict) and isinstance(reference.get("path"), str) and
            SHA256_PATTERN.fullmatch(reference.get("sha256", "")), "explicit file path and SHA-256 required")
    path = Path(reference["path"]).resolve()
    require(not any(p.lower() == "upstream" for p in path.parts), "reference-only directory cannot be a runtime source")
    require(sha256_file(path) == reference["sha256"], f"pinned source hash mismatch: {path.name}")
    return path


def _source(source, state):
    config_path = _pinned(source["config"])
    weights_path = _pinned(source["weights"])
    config = read_json(config_path)
    role = "teacher" if state == "teacher" else "student"
    expected = (36,20,1280) if role == "teacher" else (24,16,1024)
    require(tuple(config.get(k) for k in ("n_layer", "n_head", "n_embd")) == expected and
            config.get("vocab_size") == 50257 and config.get("n_positions") == 1024,
            "main GPT-2 architecture/config mismatch")
    if role == "student":
        condition, suffix = state.split("/")
        step = int(suffix.removeprefix("step"))
        policy = admit_s1_followup(source["identity"], study="mechanistic_followup", step=step)
        require(source["identity"]["condition"] == condition and source.get("initialization") == "configuration_random",
                "wrong source condition or pretrained student initialization")
        manifest_path = _pinned(source["manifest"])
        manifest = read_json(manifest_path)
        require(manifest["identity"] == source["identity"] and manifest["step"] == step and
                manifest.get("schema_version") == 1 and manifest.get("kind") in {"weights", "rolling", "final"} and
                manifest["identity_sha256"] == payload_digest(source["identity"]) and
                manifest["files"]["model.safetensors"] == source["weights"]["sha256"] and
                weights_path == manifest_path.parent/"model.safetensors" and
                (manifest_path.parent/"COMPLETE").read_text(encoding="ascii").strip() == source["manifest"]["sha256"],
                "source checkpoint manifest/COMPLETE/identity binding mismatch")
    else:
        require(source.get("model_id") == "openai-community/gpt2-large" and
                isinstance(source.get("revision"), str) and COMMIT_PATTERN.fullmatch(source["revision"]),
                "pinned frozen GPT-2-large teacher revision required")
        policy = {"role": "frozen_teacher", "static_reference": True}
    return {"role": role, "config_path": str(config_path), "weights_path": str(weights_path),
            "weights_sha256": source["weights"]["sha256"], "config_sha256": source["config"]["sha256"],
            "source_identity": source.get("identity"), "followup_policy": policy}


def admit_job(document, *, approved_sha256, phase, state, panel_name, seed, repo, device):
    """All protocol/scope/input/runtime gates precede deserialization/model access."""
    payload, digest = verify_envelope(document)
    require(isinstance(approved_sha256, str) and SHA256_PATTERN.fullmatch(approved_sha256) and
            digest == approved_sha256, "explicit externally approved protocol digest required")
    require(payload.get("schema_version") == VERSION and payload.get("status") == "approved" and
            payload.get("production_ready") is True and payload.get("approval", {}).get("authority") == "researcher" and
            bool(payload["approval"].get("record")), "draft or unapproved scientific protocol")
    require(type(seed) is int and seed == payload["seed"] == 0 and payload["phase"] == phase and
            payload["d24_sha256"] == D24_SHA256, "phase/seed0/D24 mismatch")
    require(panel_name in {"discovery", "confirmation"}, "explicit panel required")
    grid = payload["grid"]
    required_grid = E1_GRID if phase == "E1" else E2_GRID if phase == "E2" else None
    require(isinstance(grid, list) and grid and len(set(grid)) == len(grid) and "teacher" in grid and state in grid,
            "explicit unique student/teacher state grid required")
    if required_grid:
        require(set(grid) == {"teacher", *required_grid}, "researcher checkpoint grid differs")
    else:
        require(phase == "E3" and set(grid).issubset({"teacher", *E2_GRID}), "E3 requires selected E2 states")
    require(set(payload["sources"]) == set(grid), "source inventory differs from state grid")
    runtime = runtime_identity(repo, device)
    require(runtime == payload["runtime"], "code/environment/fixed-device runtime lock mismatch")
    qualification = read_json(_pinned(payload["qualification"]))
    require(qualification.get("status") == "qualified" and qualification.get("scientific_production_shape") is True and
            qualification.get("phase") == phase and qualification.get("runtime") == runtime and
            qualification.get("architectures") == ["gpt2-large", "gpt2-medium"] and
            qualification.get("settings_sha256") == payload_digest(payload["settings"]) and
            qualification.get("parity_passed") is True and qualification.get("headroom_passed") is True,
            "missing real production-shape phase qualification")
    panel = read_json(_pinned(payload["panel"]))
    require(panel.get("context_length") == 128 and panel.get("tokenizer", {}).get("model_id") == "gpt2" and
            SHA256_PATTERN.fullmatch(panel["tokenizer"].get("artifact_sha256", "")), "frozen GPT-2 tokenizer/panel identity required")
    validate_panel_split(panel["discovery"], panel["confirmation"])
    require(payload["panel_counts"] == {p: len(panel[p]) for p in ("discovery", "confirmation")}, "panel count mismatch")
    for name in ("discovery", "confirmation"):
        validate_items(panel[name], length=128, vocab_size=50257)
    from .mechanistic_panel import prepare_frozen_panel
    artifact_document = read_json(Path(repo)/"protocols/artifact.lock.json")
    prepared = prepare_frozen_panel(artifact_document=artifact_document,
        corpus_reference=panel["preparation"]["corpus"],panels_reference=panel["preparation"]["registered_panels"],
        confirmation_ids=[i["id"] for i in panel["confirmation"]])
    require(prepared == panel, "prepared panel differs from registered tokens/source-document ownership")
    artifact,_ = verify_envelope(artifact_document)
    locked_teacher = payload["sources"]["teacher"]
    require(locked_teacher["weights"]["sha256"] == artifact["teacher"]["weights_sha256"] and
            locked_teacher["config"]["sha256"] == artifact["teacher"]["config_sha256"] and
            locked_teacher["revision"] == artifact["teacher"]["revision"], "teacher differs from original S1 frozen reference")
    if state != "teacher":
        require(payload["sources"][state]["config"]["sha256"] == artifact["student_config"]["sha256"],
                "student differs from original random-initialization configuration")
    settings = payload["settings"]
    layers = 36 if state == "teacher" else 24
    current = settings[state]
    validate_settings(phase, current, layers)
    if phase == "E2":
        require(current["layers"] == list(range(layers)), "E2 must cover every native layer/head")
    if phase == "E1":
        scopes = {s["name"]: s["layers"] for s in current["scopes"]}
        require(scopes.get("native") == list(range(layers)), "native E1 scope required")
        if state == "teacher":
            mapping = payload["teacher_layer_map"]
            require(len(mapping) == 24 and len(set(mapping)) == 24 and all(type(i) is int and 0 <= i < 36 for i in mapping) and
                    scopes.get("mapped_teacher") == mapping, "explicit mapped-24 teacher scope required")
    if phase == "E3":
        selection = payload["selection"]
        require(selection["method"] in {"prespecified_coarse", "discovery_only"} and
                selection["selected_layers"][state] == current["layers"] and
                selection["selected_states"] == grid, "frozen E3 state/layer selection required")
        if selection["method"] == "discovery_only":
            discovery = read_json(_pinned(selection["discovery_evidence"]))
            require(discovery["panel"] == "discovery" and discovery["phase"] == "E2" and
                    discovery["panel_sha256"] == payload["panel"]["sha256"], "confirmation selection leakage")
        if panel_name == "discovery":
            require(len(current["orders"]) >= 2, "discovery requires an alternate deletion order")
    source = _source(payload["sources"][state], state)
    teacher = source if state == "teacher" else _source(payload["sources"]["teacher"], "teacher")
    identity = {"schema_version": VERSION, "phase": phase, "study": f"mechanistic_followup/{phase}",
        "run_id": payload["run_id"], "seed": seed, "state": state, "panel": panel_name,
        "panel_sha256": payload["panel"]["sha256"], "protocol_sha256": digest,
        "runtime_sha256": payload_digest(runtime), "context_length": 128,
        "source": source, "teacher_source": teacher, "engineering_only": False}
    return identity, current, panel[panel_name], source, teacher


def load_model(source, device):
    """Local pinned tensors only; no optimizer, training or network access."""
    from safetensors.torch import load_file
    from transformers import GPT2Config, GPT2LMHeadModel
    # Recheck immediately before load to catch stale admission references.
    require(sha256_file(source["weights_path"]) == source["weights_sha256"] and
            sha256_file(source["config_path"]) == source["config_sha256"], "source changed after admission")
    config = GPT2Config.from_dict(read_json(source["config_path"]))
    config._attn_implementation = "eager"
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        model = GPT2LMHeadModel(config)
    weights = load_file(source["weights_path"], device="cpu")
    result = model.load_state_dict(weights, strict=False)
    require(not result.unexpected_keys and set(result.missing_keys).issubset({"lm_head.weight"}) and
            (not result.missing_keys or model.config.tie_word_embeddings), "incomplete/unexpected model weights")
    if not result.missing_keys and model.config.tie_word_embeddings:
        require(torch.equal(weights["lm_head.weight"], weights["transformer.wte.weight"]), "conflicting tied embedding tensors")
    model.tie_weights()
    return model.float().to(device).eval().requires_grad_(False)
