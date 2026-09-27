"""Explicit, one-checkpoint-at-a-time Pythia evaluation (S2).

Inventory creation is offline. Hub resolution and downloads are separate opt-in
operations; neither import nor trajectory planning contacts the network.
"""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path
from typing import Callable, Iterable

from .data import _ids, document_hash, normalized_text, save_manifest, load_manifest
from .provenance import (COMMIT_PATTERN, SHA256_PATTERN, _no_duplicate_keys,
                         canonical_json_bytes, seal_payload, verify_envelope)


class PythiaError(ValueError):
    pass


TOKENS_PER_STEP = 2_097_152
SIZES = {"160m", "410m"}
TRAINING_SEEDS = {1234, 1, 2}


def expected_steps() -> tuple[int, ...]:
    return (0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
            *range(1000, 143001, 1000))


def repository(size: str, training_seed: int) -> str:
    if size not in SIZES or type(training_seed) is not int or training_seed not in TRAINING_SEEDS:
        raise PythiaError("unsupported Pythia size or training seed; standard is seed 1234")
    suffix = "" if training_seed == 1234 else f"-seed{training_seed}"
    return f"EleutherAI/pythia-{size}{suffix}"


def checkpoint_entry(*, size: str, training_seed: int, native_step: int,
                     revision: str, files: dict[str, dict]) -> dict:
    repo = repository(size, training_seed)
    if type(native_step) is not int or native_step not in expected_steps():
        raise PythiaError("unsupported native checkpoint step")
    if not isinstance(revision, str) or not COMMIT_PATTERN.fullmatch(revision):
        raise PythiaError("checkpoint revision must be an immutable 40-character commit")
    if not isinstance(files, dict) or "config.json" not in files or not any(
            name.endswith((".safetensors", ".bin")) for name in files):
        raise PythiaError("checkpoint requires config and weight file metadata")
    for name, meta in files.items():
        if (not isinstance(name, str) or not name or Path(name).is_absolute() or
                ".." in Path(name).parts or "\\" in name or
                not isinstance(meta, dict) or set(meta) != {"sha256", "bytes"} or
                not isinstance(meta["sha256"], str) or not SHA256_PATTERN.fullmatch(meta["sha256"]) or
                type(meta["bytes"]) is not int or meta["bytes"] < 1):
            raise PythiaError("invalid checkpoint file hash, size, or path")
    return {"size": size, "training_seed": training_seed, "repository": repo,
            "branch": f"step{native_step}", "revision": revision,
            "native_step": native_step, "training_tokens": native_step * TOKENS_PER_STEP,
            "files": files}


def create_inventory(entries: Iterable[dict]) -> dict:
    checked, keys = [], set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {
                "size", "training_seed", "repository", "branch", "revision",
                "native_step", "training_tokens", "files"}:
            raise PythiaError("inventory entry has missing or unknown fields")
        expected = checkpoint_entry(size=entry["size"], training_seed=entry["training_seed"],
            native_step=entry["native_step"], revision=entry["revision"], files=entry["files"])
        if entry != expected:
            raise PythiaError("inventory branch, repository, token count or metadata differs")
        key = (entry["size"], entry["training_seed"], entry["native_step"])
        if key in keys:
            raise PythiaError("duplicate checkpoint identity")
        keys.add(key)
        checked.append(entry)
    checked.sort(key=lambda e: (e["size"], e["training_seed"], e["native_step"]))
    return seal_payload({"kind": "pythia-inventory-v1", "entries": checked})


def load_inventory(path: Path) -> tuple[dict, str]:
    payload, digest = load_manifest(path, stem="pythia-inventory")
    if payload.get("kind") != "pythia-inventory-v1" or create_inventory(payload.get("entries", []))["payload"] != payload:
        raise PythiaError("invalid checkpoint inventory")
    return payload, digest


def select_checkpoint(inventory: dict, *, size: str, training_seed: int, native_step: int) -> dict:
    repository(size, training_seed)
    matches = [e for e in inventory["entries"] if
               (e["size"], e["training_seed"], e["native_step"]) ==
               (size, training_seed, native_step)]
    if len(matches) != 1:
        raise PythiaError(f"missing checkpoint: {size} training seed {training_seed} step {native_step}")
    return matches[0]


def resolve_one_hub_branch(*, size: str, training_seed: int, native_step: int,
                           api=None) -> dict:
    """Explicit network operation. Require content hashes; never resolve main/latest."""
    from huggingface_hub import HfApi
    repo = repository(size, training_seed)
    if type(native_step) is not int or native_step not in expected_steps():
        raise PythiaError("unsupported native checkpoint step")
    branch = f"step{native_step}"
    info = (api or HfApi()).model_info(repo_id=repo, revision=branch, files_metadata=True)
    files = {}
    names = {s.rfilename for s in info.siblings}
    use_safe = any(n == "model.safetensors" or n.startswith("model-") and n.endswith(".safetensors") for n in names)
    for sibling in info.siblings:
        name = sibling.rfilename
        weight = ((name == "model.safetensors" or
                   name.startswith("model-") and name.endswith(".safetensors")) if use_safe else
                  (name == "pytorch_model.bin" or
                   name.startswith("pytorch_model-") and name.endswith(".bin")))
        index = "model.safetensors.index.json" if use_safe else "pytorch_model.bin.index.json"
        if name == "config.json" or weight or name == index:
            lfs = getattr(sibling, "lfs", None) or {}
            digest = lfs.get("sha256") if isinstance(lfs, dict) else None
            count = getattr(sibling, "size", None)
            if digest is None:
                from huggingface_hub import hf_hub_download
                path = Path(hf_hub_download(repo_id=repo, filename=name, revision=info.sha))
                digest, count = _sha256(path), path.stat().st_size
            if digest is None or count is None:
                raise PythiaError(f"Hub did not supply immutable file hash/size: {name}")
            files[name] = {"sha256": digest, "bytes": count}
    return checkpoint_entry(size=size, training_seed=training_seed,
                            native_step=native_step, revision=info.sha, files=files)


def prepare_panel(rows: Iterable[dict], tokenizer, *, tokenizer_id: str,
                  tokenizer_revision: str, tokenizer_sha256: str,
                  source_panel_sha256: str, selected_document_hashes: list[str],
                  context: int = 128) -> dict:
    """Retokenize explicitly selected raw documents; never reuse GPT-2 IDs."""
    if context < 2 or not selected_document_hashes or len(set(selected_document_hashes)) != len(selected_document_hashes):
        raise PythiaError("nonempty unique source documents and context >=2 required")
    for digest in [tokenizer_sha256, source_panel_sha256, *selected_document_hashes]:
        if not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest):
            raise PythiaError("source and tokenizer SHA-256 required")
    if not tokenizer_id.startswith("EleutherAI/pythia-") or not COMMIT_PATTERN.fullmatch(tokenizer_revision):
        raise PythiaError("pinned Pythia tokenizer identity required")
    eos = tokenizer.eos_token_id
    if type(eos) is not int or eos < 0:
        raise PythiaError("Pythia tokenizer EOS required")
    selected = set(selected_document_hashes)
    found = {}
    for row in rows:
        norm = normalized_text(row["text"])
        digest = document_hash(norm)
        if digest in selected and digest not in found:
            ids = _ids(tokenizer, norm)[:context]
            if len(ids) < 2:
                raise PythiaError("selected document has fewer than two Pythia tokens")
            found[digest] = {"id": digest, "input_ids": ids + [eos] * (context - len(ids)),
                             "attention_mask": [1] * len(ids) + [0] * (context - len(ids))}
    if set(found) != selected:
        raise PythiaError("selected raw documents missing")
    return seal_payload({"kind": "pythia-panel-v1", "tokenizer": {"id": tokenizer_id,
        "revision": tokenizer_revision, "files_sha256": tokenizer_sha256,
        "eos_token_id": eos}, "source_panel_sha256": source_panel_sha256,
        "context": context, "items": [found[d] for d in selected_document_hashes]})


def load_panel(path: Path, *, tokenizer_sha256: str) -> tuple[dict, str]:
    payload, digest = load_manifest(path, stem="pythia-panel")
    if payload.get("kind") != "pythia-panel-v1" or payload["tokenizer"]["files_sha256"] != tokenizer_sha256:
        raise PythiaError("Pythia panel tokenizer mismatch")
    context, eos = payload["context"], payload["tokenizer"]["eos_token_id"]
    items = payload["items"]
    if not items or len({i["id"] for i in items}) != len(items):
        raise PythiaError("invalid Pythia panel items")
    for item in items:
        ids, mask = item["input_ids"], item["attention_mask"]
        n = sum(mask)
        if (len(ids) != context or len(mask) != context or not 2 <= n <= context or
                mask != [1] * n + [0] * (context - n) or ids[n:] != [eos] * (context - n)):
            raise PythiaError("invalid Pythia token IDs or right-padding mask")
    return payload, digest


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_snapshot(entry: dict, root: Path) -> str:
    root = Path(root)
    for name, meta in entry["files"].items():
        path = root / name
        if not path.is_file() or path.is_symlink() or path.stat().st_size != meta["bytes"] or _sha256(path) != meta["sha256"]:
            raise PythiaError(f"missing or altered checkpoint file: {name}")
    return hashlib.sha256(canonical_json_bytes({
        "revision": entry["revision"], "files": entry["files"]})).hexdigest()


def bounded_download(entry: dict, *, cache_root: Path, byte_cap: int,
                     downloader: Callable | None = None) -> Path:
    """Download one pinned state to an owned directory under a hard metadata cap."""
    if type(byte_cap) is not int or byte_cap < 1:
        raise PythiaError("positive cache byte cap required")
    root = Path(cache_root)
    if root.is_symlink():
        raise PythiaError("cache root may not be a symlink")
    root.mkdir(parents=True, exist_ok=True)
    used = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    needed = sum(m["bytes"] for m in entry["files"].values())
    target = root / f'{entry["size"]}-seed{entry["training_seed"]}-step{entry["native_step"]}-{entry["revision"]}'
    if target.exists():
        verify_snapshot(entry, target)
        return target
    if used + needed > byte_cap:
        raise PythiaError(f"cache cap exceeded: used {used}, required {needed}, cap {byte_cap}")
    if downloader is None:
        from huggingface_hub import snapshot_download
        downloader = snapshot_download
    downloader(repo_id=entry["repository"], revision=entry["revision"],
               allow_patterns=list(entry["files"]), local_dir=str(target))
    verify_snapshot(entry, target)
    actual = sum(p.stat().st_size for p in root.rglob("*") if p.is_file())
    if actual > byte_cap:
        raise PythiaError(f"cache cap exceeded after download: {actual} > {byte_cap}")
    return target


def evaluate_one(*, entry: dict, snapshot: Path, panel_path: Path,
                 tokenizer_sha256: str, store_root: Path, evaluator_seed: int,
                 precision: str = "fp32", device: str = "cpu", model_loader=None) -> dict:
    """Evaluate one verified native checkpoint; evaluator RNG is independent."""
    from .evaluate import RecordStore, evaluate_panel
    from .models import GPTNeoXAdapter
    from transformers import GPTNeoXForCausalLM
    import torch
    if type(evaluator_seed) is not int or evaluator_seed < 0:
        raise PythiaError("explicit nonnegative evaluator RNG seed required")
    if precision not in {"fp32", "bf16"}:
        raise PythiaError("unsupported precision")
    if device not in {"cpu", "cuda"} or (device == "cuda" and not torch.cuda.is_available()):
        raise PythiaError("explicit available CPU/CUDA device required")
    if precision == "bf16" and device != "cuda":
        raise PythiaError("BF16 evaluation requires CUDA")
    checkpoint_hash = verify_snapshot(entry, snapshot)
    panel, panel_hash = load_panel(panel_path, tokenizer_sha256=tokenizer_sha256)
    model = (model_loader or GPTNeoXForCausalLM.from_pretrained)(
        str(snapshot), local_files_only=True, attn_implementation="eager")
    try:
        source_weight_dtype = str(next(model.parameters()).dtype)
        model.to(device=device, dtype=torch.float32)
        adapter = GPTNeoXAdapter(model)
        identity = _run_identity(entry, panel, panel_hash, checkpoint_hash, device, precision)
        result = evaluate_panel(adapter=adapter, items=panel["items"], panel="pythia",
            panel_hash=panel_hash, checkpoint_hash=checkpoint_hash,
            run_id=f'S2-{entry["size"]}-seed{entry["training_seed"]}-step{entry["native_step"]}',
            step=entry["native_step"], store=RecordStore(store_root),
            layer_scope=list(range(adapter.layer_count)), operations=("clean", "delete", "relocate"),
            precision=precision, model_role="pythia", run_identity=identity,
            denominator_floor=1e-8, terminal=False)
        return {"size": entry["size"], "training_seed": entry["training_seed"],
                "evaluator_seed": evaluator_seed, "native_step": entry["native_step"],
                "training_tokens": entry["training_tokens"], "repository": entry["repository"],
                "branch": entry["branch"], "revision": entry["revision"],
                "checkpoint_sha256": checkpoint_hash, "tokenizer": panel["tokenizer"],
                "panel_sha256": panel_hash, "device": device, "precision": precision,
                "source_weight_dtype": source_weight_dtype,
                "compute_weight_dtype": str(next(model.parameters()).dtype),
                "evaluation": result}
    finally:
        del model


def _run_identity(entry: dict, panel: dict, panel_hash: str, checkpoint_hash: str,
                  device: str, precision: str) -> dict:
    return {"study": "S2", "condition": "ordinary_pretraining",
            "seed": entry["training_seed"], "model_sha256": checkpoint_hash,
            "corpus_sha256": panel_hash, "protocol_sha256": hashlib.sha256(
                canonical_json_bytes({"inventory_entry": entry,
                    "tokenizer": panel["tokenizer"], "device": device,
                    "precision": precision})).hexdigest()}


def _summary_path(store_root: Path, entry: dict, panel_hash: str,
                  device: str, precision: str) -> Path:
    name = hashlib.sha256(canonical_json_bytes({"entry": entry, "panel": panel_hash,
        "device": device, "precision": precision})).hexdigest()
    return Path(store_root) / f"pythia-result-{name}.json"


def _verified_result(store_root: Path, entry: dict, panel: dict, panel_hash: str,
                     device: str, precision: str) -> bool:
    """A summary alone cannot authorize resume; verify every underlying item."""
    from .evaluate import RecordStore, _key
    path = _summary_path(store_root, entry, panel_hash, device, precision)
    if not path.exists():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        summary, _ = verify_envelope(document)
        checkpoint_hash = hashlib.sha256(canonical_json_bytes({
            "revision": entry["revision"], "files": entry["files"]})).hexdigest()
        if summary != {"entry": entry, "panel_sha256": panel_hash,
                       "checkpoint_sha256": checkpoint_hash, "status": "complete",
                       "device": device, "precision": precision}:
            raise PythiaError("stale trajectory completion summary")
        identity = _run_identity(entry, panel, panel_hash, checkpoint_hash, device, precision)
        store = RecordStore(store_root)
        run_id = f'S2-{entry["size"]}-seed{entry["training_seed"]}-step{entry["native_step"]}'
        # Native layer count is fixed by the pinned config. Read the verified
        # aggregate key for scope rather than loading weights just to learn it.
        aggregates = list(Path(store_root).glob("aggregate-*.json"))
        matching = []
        for aggregate in aggregates:
            data, _ = verify_envelope(json.loads(aggregate.read_text(encoding="utf-8"),
                                                  object_pairs_hook=_no_duplicate_keys))
            key = data["key"]
            if (key.get("run_id"), key.get("panel_hash"), key.get("checkpoint_hash"),
                    key.get("precision"), key.get("run_identity")) == (
                    run_id, panel_hash, checkpoint_hash, precision, identity):
                matching.append(data)
        if len(matching) != 1 or any(v["status"] != "complete" for v in matching[0]["operations"].values()):
            raise PythiaError("missing complete trajectory aggregate")
        key = matching[0]["key"]
        if key["run_identity"] != identity or set(key["operations"]) != {"clean", "delete", "relocate"}:
            raise PythiaError("trajectory aggregate provenance mismatch")
        for item in panel["items"]:
            for op in key["operations"]:
                item_key = _key(run_id=run_id, step=entry["native_step"], panel="pythia",
                    panel_hash=panel_hash, checkpoint_hash=checkpoint_hash,
                    item_id=item["id"], scope=key["scope"], operation=op,
                    strength=0. if op in {"clean", "none"} else 1.,
                    precision=key["precision"], model_role="pythia",
                    evaluation_mode="full", run_identity=identity,
                    denominator_floor=key["fingerprint_denominator_floor"])
                row = store.read(item_key)
                if row is None or row["status"] != "complete":
                    raise PythiaError("missing completed trajectory item")
        return True
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise PythiaError(f"trajectory result verification failed: {exc}") from exc


def _write_summary(store_root: Path, entry: dict, panel_hash: str, checkpoint_hash: str,
                   device: str, precision: str) -> None:
    path = _summary_path(store_root, entry, panel_hash, device, precision)
    path.parent.mkdir(parents=True, exist_ok=True)
    document = seal_payload({"entry": entry, "panel_sha256": panel_hash,
        "checkpoint_sha256": checkpoint_hash, "status": "complete",
        "device": device, "precision": precision})
    if path.exists() and path.read_bytes() != canonical_json_bytes(document) + b"\n":
        raise PythiaError("immutable trajectory summary conflict")
    path.write_bytes(canonical_json_bytes(document) + b"\n")


def trajectory_plan(inventory: dict, *, size: str, training_seed: int,
                    native_steps: list[int]) -> dict:
    """Plan only requested states; expose missing entries, never choose latest."""
    repository(size, training_seed)
    if not native_steps or len(set(native_steps)) != len(native_steps):
        raise PythiaError("explicit unique native steps required")
    present, missing = [], []
    for step in native_steps:
        try:
            present.append(select_checkpoint(inventory, size=size, training_seed=training_seed, native_step=step))
        except PythiaError:
            missing.append(step)
    return {"entries": present, "missing_native_steps": missing,
            "estimated_download_bytes": sum(sum(m["bytes"] for m in e["files"].values()) for e in present)}


def run_trajectory(*, inventory: dict, size: str, training_seed: int,
                   native_steps: list[int], cache_root: Path, byte_cap: int,
                   panel_path: Path, tokenizer_sha256: str, store_root: Path,
                   evaluator_seed: int, device: str = "cpu", precision: str = "fp32",
                   downloader=None, model_loader=None,
                   evict_after_verified: bool = False) -> dict:
    """Opt-in bounded loop. Verified record stores allow idempotent restart."""
    plan = trajectory_plan(inventory, size=size, training_seed=training_seed, native_steps=native_steps)
    if plan["missing_native_steps"]:
        return {"status": "missing_checkpoints", **plan, "completed": []}
    panel, panel_hash = load_panel(panel_path, tokenizer_sha256=tokenizer_sha256)
    completed = []
    for entry in plan["entries"]:
        if _verified_result(store_root, entry, panel, panel_hash, device, precision):
            completed.append({"native_step": entry["native_step"],
                              "training_tokens": entry["training_tokens"],
                              "status": "verified_resume"})
            continue
        snapshot = bounded_download(entry, cache_root=cache_root, byte_cap=byte_cap,
                                    downloader=downloader)
        result = evaluate_one(entry=entry, snapshot=snapshot, panel_path=panel_path,
            tokenizer_sha256=tokenizer_sha256, store_root=store_root,
            evaluator_seed=evaluator_seed, device=device, precision=precision,
            model_loader=model_loader)
        completed.append({"native_step": entry["native_step"],
                          "training_tokens": entry["training_tokens"],
                          "status": "complete" if all(x["status"] == "complete" for x in result["evaluation"]["operations"].values()) else "incomplete"})
        if completed[-1]["status"] == "complete":
            _write_summary(store_root, entry, panel_hash, result["checkpoint_sha256"],
                           device, precision)
            if evict_after_verified and _verified_result(store_root, entry, panel,
                                                          panel_hash, device, precision):
                owned_root = Path(cache_root).resolve()
                candidate = snapshot.resolve()
                if snapshot.is_symlink() or candidate.parent != owned_root:
                    raise PythiaError("refusing to evict checkpoint outside owned cache root")
                shutil.rmtree(candidate)
        else:
            break
    return {"status": "complete" if len(completed) == len(plan["entries"]) else "incomplete",
            **plan, "completed": completed}
