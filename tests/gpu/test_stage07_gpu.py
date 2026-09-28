"""One pinned public Pythia checkpoint, engineering acceptance only."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pytest
import torch
from transformers import GPTNeoXForCausalLM

from sinklab.data import tokenizer_files_hash
from sinklab.interventions import AttentionIntervention
from sinklab.models import GPTNeoXAdapter, ModelShape
from sinklab.pythia import (_verified_result, load_inventory, load_panel,
                            run_trajectory, verify_snapshot)
from sinklab.provenance import verify_envelope


ROOT = Path(__file__).parent
INVENTORY = ROOT / "pythia-inventory-6eb15732026945d69e58b2c4f4b09f80a294d99a99f587de563fe5801e4a7528.json"
PANEL = ROOT / "pythia-panel-8ec4bfa9dd9dde6e4a795df518b19e801838da047690190c4ae95a596f7e7c78.json"
TOKENIZER_SHA = "bd13e7ce7dec89032a23791c486b1e294381d14484f447c1a0476c8bf92e236f"


@pytest.fixture(scope="module")
def public_checkpoint(selected_cuda, gpu_evidence):
    base = os.environ.get("STAGE07_ACCEPTANCE_ROOT")
    if not base:
        pytest.fail("STAGE07_ACCEPTANCE_ROOT must identify the owned external checkpoint cache")
    root = Path(base)
    inventory, inventory_sha = load_inventory(INVENTORY)
    assert len(inventory["entries"]) == 1
    entry = inventory["entries"][0]
    snapshot = root / "checkpoints" / f"160m-seed1234-step0-{entry['revision']}"
    checkpoint_sha = verify_snapshot(entry, snapshot)
    panel, panel_sha = load_panel(PANEL, tokenizer_sha256=TOKENIZER_SHA)
    assert tokenizer_files_hash(root / "tokenizer") == TOKENIZER_SHA
    assert panel["tokenizer"]["revision"] == entry["revision"]
    assert panel["source_panel_sha256"] == "29974142a3f3c9aead3d0e5f314da64b107aaf136a46df844a449a2535ad4745"
    gpu_evidence[0]["stage07_identity"] = {
        "repository": entry["repository"], "branch": entry["branch"],
        "revision": entry["revision"], "training_seed": entry["training_seed"],
        "native_step": entry["native_step"], "training_tokens": entry["training_tokens"],
        "inventory_sha256": inventory_sha, "panel_sha256": panel_sha,
        "tokenizer_sha256": TOKENIZER_SHA, "checkpoint_sha256": checkpoint_sha,
        "file_metadata": entry["files"], "context": panel["context"],
        "real_token_counts": [sum(x["attention_mask"]) for x in panel["items"]],
    }
    return root, entry, inventory, panel, panel_sha, snapshot


def test_real_native_rotary_mask_adapter_parity_and_restoration(public_checkpoint,
                                                                  selected_cuda, gpu_evidence):
    _, entry, _, panel, _, snapshot = public_checkpoint
    model = GPTNeoXForCausalLM.from_pretrained(
        str(snapshot), local_files_only=True, attn_implementation="eager"
    )
    source_dtype = str(next(model.parameters()).dtype)
    assert source_dtype == "torch.float16"
    model = model.to(device=selected_cuda, dtype=torch.float32).eval()
    adapter = GPTNeoXAdapter(model, ModelShape(12, 12, 768))
    item = panel["items"][0]
    ids = torch.tensor([item["input_ids"]], device=selected_cuda)
    mask = torch.tensor([item["attention_mask"]], device=selected_cuda, dtype=torch.bool)
    modules = [layer.attention for layer in model.gpt_neox.layers]
    before = [("forward" in module.__dict__, module.__dict__.get("forward")) for module in modules]
    try:
        with torch.inference_mode():
            native = model(input_ids=ids, attention_mask=mask, use_cache=False,
                           output_attentions=True)
            features = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
            noop = adapter.forward(input_ids=ids, attention_mask=mask,
                                   intervention=AttentionIntervention("none"),
                                   output_attentions=True)
            logit_error = float((native.logits - features.outputs.logits).abs().max())
            noop_error = float((native.logits - noop.logits).abs().max())
            attention_error = max(float((a - f.probabilities).abs().max())
                                  for a, f in zip(native.attentions, features.attention, strict=True))
            valid = features.attention[0].valid_edges
            assert valid.shape == (1, 1, 128, 128)
            assert not valid[..., 0, 1].any()  # causal
            assert not valid[..., sum(item["attention_mask"]):, :].any()  # padded queries
            assert not valid[..., :, sum(item["attention_mask"]):].any()  # padded keys
            assert logit_error == 0
            assert attention_error < 2e-5
            assert noop_error < 2e-4
            with pytest.raises(RuntimeError, match="restoration check"):
                with adapter.intervention_context(AttentionIntervention("delete"), mask):
                    raise RuntimeError("restoration check")
            assert [("forward" in module.__dict__, module.__dict__.get("forward"))
                    for module in modules] == before
            after = model(input_ids=ids, attention_mask=mask, use_cache=False)
            assert torch.equal(after.logits, native.logits)
            # A nonuniform position change must affect rotary attention.
            positions = torch.arange(ids.shape[1], device=selected_cuda)[None, :].clone()
            positions[:, 1] += 1
            shifted = model(input_ids=ids, attention_mask=mask,
                            position_ids=positions, use_cache=False)
            rotary_effect = float((shifted.logits[:, :sum(item["attention_mask"])] -
                                   native.logits[:, :sum(item["attention_mask"])]).abs().max())
            assert rotary_effect > 0
        gpu_evidence[0]["measurements"]["stage07_native_parity"] = {
            "logit_max_abs": logit_error, "attention_max_abs": attention_error,
            "noop_logit_max_abs": noop_error, "rotary_position_effect_max_abs": rotary_effect,
            "all_12_layers": True, "causal_and_padding_masks": True,
            "exception_restoration": True, "post_restore_logits_equal": True,
            "source_weight_dtype": source_dtype,
            "compute_weight_dtype": str(next(model.parameters()).dtype),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(selected_cuda),
        }
    finally:
        del model
        torch.cuda.empty_cache()


def test_real_single_checkpoint_resume_and_missing(public_checkpoint, selected_cuda,
                                                    gpu_evidence, tmp_path):
    root, entry, inventory, panel, panel_sha, _ = public_checkpoint
    store = tmp_path / "records"
    args = dict(inventory=inventory, size="160m", training_seed=1234,
                cache_root=root / "checkpoints", byte_cap=800_000_000,
                panel_path=PANEL, tokenizer_sha256=TOKENIZER_SHA,
                store_root=store, evaluator_seed=77, device="cuda", precision="fp32")
    missing = run_trajectory(**args, native_steps=[0, 1])
    assert missing["status"] == "missing_checkpoints"
    assert missing["missing_native_steps"] == [1] and not missing["completed"]
    first = run_trajectory(**args, native_steps=[0])
    assert first["status"] == "complete" and first["completed"][0]["status"] == "complete"
    resumed = run_trajectory(**args, native_steps=[0])
    assert resumed["completed"][0]["status"] == "verified_resume"
    copied = tmp_path / "verification_copy"
    shutil.copytree(store, copied)
    assert _verified_result(copied, entry, panel, panel_sha, "cuda", "fp32")
    summary = next(copied.glob("pythia-result-*.json"))
    summary.unlink()
    assert not _verified_result(copied, entry, panel, panel_sha, "cuda", "fp32")
    repaired = run_trajectory(**{**args, "store_root": copied}, native_steps=[0])
    assert repaired["completed"][0]["status"] == "complete"
    assert _verified_result(copied, entry, panel, panel_sha, "cuda", "fp32")
    aggregate = next(copied.glob("aggregate-*.json"))
    aggregate.unlink()
    with pytest.raises(ValueError, match="verification failed"):
        run_trajectory(**{**args, "store_root": copied}, native_steps=[0])
    shutil.copy2(next(store.glob("aggregate-*.json")), aggregate)
    item = next(p for p in copied.glob("*.json") if len(p.stem) == 64)
    item.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="verification failed"):
        run_trajectory(**{**args, "store_root": copied}, native_steps=[0])
    aggregate_payload, _ = verify_envelope(json.loads(next(store.glob("aggregate-*.json")).read_text()))
    key = aggregate_payload["key"]
    assert key["scope"] == list(range(12))
    assert key["step"] == 0 and key["panel"] == "pythia"
    assert key["panel_hash"] == panel_sha
    assert key["checkpoint_hash"] == gpu_evidence[0]["stage07_identity"]["checkpoint_sha256"]
    assert key["run_identity"]["seed"] == 1234
    assert key["run_identity"]["corpus_sha256"] == panel_sha
    assert key["run_identity"]["model_sha256"] == key["checkpoint_hash"]
    assert key["precision"] == "fp32" and key["model_role"] == "pythia"
    assert key["operations"] == ["clean", "delete", "relocate"]
    metrics = {op: {"valid_targets": row["metrics"]["valid_targets"],
                    "clean_ce_nats": row["metrics"]["clean_ce_nats"],
                    "delta_ce_nats": row["metrics"]["delta_ce_nats"],
                    "self_kl_nats": row["metrics"]["self_kl_nats"]}
               for op, row in aggregate_payload["operations"].items()}
    gpu_evidence[0]["measurements"]["stage07_evaluation"] = {
        "status": first["status"], "completed": first["completed"],
        "verified_resume": resumed["completed"], "missing_case": missing,
        "missing_summary_recomputed_required": True,
        "missing_aggregate_rejected": True, "corrupt_item_rejected": True,
        "record_count": len(list(store.glob("*.json"))), "metrics": metrics,
    }
