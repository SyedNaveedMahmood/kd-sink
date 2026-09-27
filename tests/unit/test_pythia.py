"""T09 S2, T01 tokenizer, T02/T07 native model and T10 cache fixtures."""

import hashlib
import json
from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

from sinklab.data import document_hash, save_manifest
from sinklab.pythia import (PythiaError, TOKENS_PER_STEP, bounded_download,
    checkpoint_entry, create_inventory, evaluate_one, expected_steps, load_inventory,
    load_panel, prepare_panel, repository, resolve_one_hub_branch, run_trajectory, select_checkpoint,
    trajectory_plan, verify_snapshot)


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(tmp_path, *, step=0):
    torch.manual_seed(43)
    model = GPTNeoXForCausalLM(GPTNeoXConfig(vocab_size=31, hidden_size=16,
        num_hidden_layers=2, num_attention_heads=4, intermediate_size=32,
        max_position_embeddings=32, rotary_pct=.5, hidden_dropout=0,
        attention_dropout=0, _attn_implementation="eager"))
    snapshot = tmp_path / "snapshot"
    model.save_pretrained(snapshot, safe_serialization=True)
    files = {p.name: {"sha256": _hash(p), "bytes": p.stat().st_size}
             for p in snapshot.iterdir() if p.name == "config.json" or p.suffix == ".safetensors"}
    entry = checkpoint_entry(size="160m", training_seed=1234, native_step=step,
                             revision="a" * 40, files=files)
    inventory_doc = create_inventory([entry])
    inventory_path = save_manifest(inventory_doc, tmp_path, "pythia-inventory")
    class Tokenizer:
        eos_token_id = 0
        def encode(self, text, add_special_tokens=False):
            assert not add_special_tokens
            return [ord(ch) % 30 + 1 for ch in text]
    raw = [{"text": "abcd"}, {"text": "efghij"}]
    hashes = [document_hash(r["text"]) for r in raw]
    panel = prepare_panel(raw, Tokenizer(), tokenizer_id="EleutherAI/pythia-160m",
        tokenizer_revision="b" * 40, tokenizer_sha256="c" * 64,
        source_panel_sha256="d" * 64, selected_document_hashes=hashes, context=8)
    panel_path = save_manifest(panel, tmp_path, "pythia-panel")
    return entry, inventory_path, snapshot, panel_path


def test_inventory_seed_branch_tokens_immutable_and_missing(tmp_path):
    entry, path, _, _ = _fixture(tmp_path)
    payload, digest = load_inventory(path)
    assert digest == path.stem.split("-")[-1]
    assert repository("160m", 1234) == "EleutherAI/pythia-160m"
    assert repository("160m", 1) == "EleutherAI/pythia-160m-seed1"
    assert 143000 in expected_steps()
    assert entry["training_tokens"] == 0
    later = checkpoint_entry(size="410m", training_seed=2, native_step=1000,
        revision="f" * 40, files=entry["files"])
    assert later["training_tokens"] == 1000 * TOKENS_PER_STEP
    assert later["branch"] == "step1000"
    with pytest.raises(PythiaError, match="missing checkpoint"):
        select_checkpoint(payload, size="160m", training_seed=1234, native_step=1)
    assert trajectory_plan(payload, size="160m", training_seed=1234,
                           native_steps=[0, 1])["missing_native_steps"] == [1]
    with pytest.raises(PythiaError):
        repository("160m", 0)
    with pytest.raises(PythiaError):
        create_inventory([entry, entry])
    with pytest.raises(PythiaError):
        checkpoint_entry(size="160m", training_seed=1234, native_step=0,
                         revision="main", files=entry["files"])
    broken = json.loads(path.read_text())
    broken["payload"]["entries"][0]["training_seed"] = 0
    path.write_text(json.dumps(broken))
    with pytest.raises(ValueError):
        load_inventory(path)


def test_pythia_tokenized_panel_masks_and_provenance(tmp_path):
    _, _, _, path = _fixture(tmp_path)
    panel, _ = load_panel(path, tokenizer_sha256="c" * 64)
    assert panel["tokenizer"]["revision"] == "b" * 40
    assert panel["items"][0]["attention_mask"] == [1, 1, 1, 1, 0, 0, 0, 0]
    assert panel["items"][0]["input_ids"][:4] == [8, 9, 10, 11]
    with pytest.raises(PythiaError, match="tokenizer mismatch"):
        load_panel(path, tokenizer_sha256="e" * 64)


def test_single_native_model_causal_evaluation_and_verified_resume(tmp_path):
    entry, _, snapshot, panel = _fixture(tmp_path)
    kwargs = dict(entry=entry, snapshot=snapshot, panel_path=panel,
        tokenizer_sha256="c" * 64, store_root=tmp_path / "records", evaluator_seed=991)
    first = evaluate_one(**kwargs)
    files = sorted(p.name for p in (tmp_path / "records").iterdir())
    second = evaluate_one(**kwargs)
    assert files == sorted(p.name for p in (tmp_path / "records").iterdir())
    assert first["checkpoint_sha256"] == second["checkpoint_sha256"]
    assert first["training_seed"] == 1234 and first["evaluator_seed"] == 991
    assert first["training_tokens"] == 0
    assert all(x["status"] == "complete" for x in first["evaluation"]["operations"].values())
    assert first["evaluation"]["operations"]["clean"]["metrics"]["valid_targets"] == 8
    assert first["evaluation"]["operations"]["delete"]["metrics"]["self_kl_nats"] >= -1e-6
    (snapshot / "config.json").write_text("{}")
    with pytest.raises(PythiaError, match="altered"):
        evaluate_one(**kwargs)


def test_trajectory_missing_and_bounded_cache(tmp_path):
    entry, inventory_path, snapshot, panel = _fixture(tmp_path)
    inventory, _ = load_inventory(inventory_path)
    cache = tmp_path / "cache"
    needed = sum(m["bytes"] for m in entry["files"].values())
    calls = []
    def copy_download(*, repo_id, revision, allow_patterns, local_dir):
        calls.append((repo_id, revision, allow_patterns))
        target = Path(local_dir)
        target.mkdir()
        for name in allow_patterns:
            (target / name).write_bytes((snapshot / name).read_bytes())
    args = dict(inventory=inventory, size="160m", training_seed=1234,
        cache_root=cache, byte_cap=needed, panel_path=panel,
        tokenizer_sha256="c" * 64, store_root=tmp_path / "records",
        evaluator_seed=4, downloader=copy_download)
    missing = run_trajectory(**args, native_steps=[0, 1])
    assert missing["status"] == "missing_checkpoints" and missing["completed"] == []
    assert not calls
    with pytest.raises(PythiaError, match="cache cap"):
        bounded_download(entry, cache_root=cache, byte_cap=needed - 1,
                         downloader=copy_download)
    result = run_trajectory(**args, native_steps=[0])
    assert result["status"] == "complete" and result["completed"][0]["native_step"] == 0
    assert len(calls) == 1 and calls[0][1] == "a" * 40
    again = run_trajectory(**args, native_steps=[0])
    assert again["status"] == "complete" and len(calls) == 1
    assert again["completed"][0]["status"] == "verified_resume"
    assert verify_snapshot(entry, next(cache.iterdir()))
    item_file = next(p for p in (tmp_path / "records").glob("*.json") if len(p.stem) == 64)
    item_file.write_text("{}")
    with pytest.raises(PythiaError, match="result verification failed"):
        run_trajectory(**args, native_steps=[0])


def test_one_branch_resolver_never_uses_latest_or_conflates_seeds():
    class File:
        def __init__(self, name):
            self.rfilename = name
            self.lfs = {"sha256": "e" * 64}
            self.size = 12
    class Info:
        sha = "f" * 40
        siblings = [File("config.json"), File("model.safetensors")]
    class Api:
        def model_info(self, *, repo_id, revision, files_metadata):
            assert (repo_id, revision, files_metadata) == (
                "EleutherAI/pythia-410m-seed2", "step1000", True)
            return Info()
    entry = resolve_one_hub_branch(size="410m", training_seed=2,
                                   native_step=1000, api=Api())
    assert entry["revision"] == "f" * 40
    assert entry["training_tokens"] == 1000 * TOKENS_PER_STEP


def test_opt_in_eviction_only_after_verified_result(tmp_path):
    entry, inventory_path, snapshot, panel = _fixture(tmp_path)
    inventory, _ = load_inventory(inventory_path)
    cache = tmp_path / "owned-cache"
    def copy_download(*, repo_id, revision, allow_patterns, local_dir):
        target = Path(local_dir)
        target.mkdir()
        for name in allow_patterns:
            (target / name).write_bytes((snapshot / name).read_bytes())
    kwargs = dict(inventory=inventory, size="160m", training_seed=1234,
        native_steps=[0], cache_root=cache,
        byte_cap=sum(m["bytes"] for m in entry["files"].values()),
        panel_path=panel, tokenizer_sha256="c" * 64,
        store_root=tmp_path / "records", evaluator_seed=3,
        downloader=copy_download, evict_after_verified=True)
    assert run_trajectory(**kwargs)["status"] == "complete"
    assert not list(cache.iterdir())
    assert snapshot.is_dir()  # source checkpoint is outside the owned cache
    assert run_trajectory(**kwargs)["completed"][0]["status"] == "verified_resume"


def test_network_and_trajectory_commands_require_operator_flags():
    from sinklab.cli import main
    with pytest.raises(SystemExit) as resolved:
        main(["pythia-resolve-one", "--size", "160m", "--training-seed", "1234",
              "--native-step", "0", "--out", "entry.json"])
    assert resolved.value.code == 2
    with pytest.raises(SystemExit) as trajectory:
        main(["pythia-trajectory", "--inventory", "i.json", "--size", "160m",
              "--training-seed", "1234", "--native-steps", "0", "--panel", "p.json",
              "--tokenizer-sha256", "c" * 64, "--store-root", "records",
              "--evaluator-seed", "1", "--device", "cpu", "--precision", "fp32",
              "--cache-root", "cache", "--byte-cap", "100"])
    assert trajectory.value.code == 2
