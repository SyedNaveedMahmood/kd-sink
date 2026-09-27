"""Bounded Stage 08 qualification on the pinned real GPT-2-large and RTX 4080.

This is engineering calibration and evaluation smoke, not S1/S4/S5/S6 scientific
coverage. The text panel is frozen in stage08_teacher_calibration_v1.json.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path

import pytest
import torch
from huggingface_hub import try_to_load_from_cache
from transformers import GPT2Config, GPT2LMHeadModel, GPT2TokenizerFast

from sinklab.evaluate import RecordStore, evaluate_panel
from sinklab.initialization import tensor_content_hash
from sinklab.interventions import AttentionIntervention
from sinklab.models import GPT2Adapter, ModelShape
from sinklab.probes import (PROBE_IDS, apply_probe, epe_directions,
                            evaluate_probe_battery, probe_plan, transport_epe_batch)
from sinklab.provenance import payload_digest, verify_envelope
from sinklab.s5_analysis import join_s5
from sinklab.s6 import prepare_s6_domains, render_domain_items


TEACHER_REVISION = "32b71b12589c2f8d625668d2335a01cac3249519"
TEACHER_WEIGHTS_SHA256 = "5f47f3e12f91cd33b662ce7e433b6150ad5512b5884a2cee961b50e9c3bbebce"
PANEL_PATH = Path(__file__).with_name("stage08_teacher_calibration_v1.json")
CONTROL_SEED = 20260927
DENOMINATOR_FLOOR = 1e-8
RESPONSIVENESS_FLOOR = 1e-6  # Engineering numerical reporting threshold, fixed before observing probes.


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edited_parameter_hash(model: GPT2LMHeadModel) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        if name == "transformer.wpe.weight" or (
            ".attn.c_attn." in name and (name.endswith(".weight") or name.endswith(".bias"))
        ):
            digest.update(name.encode("utf-8"))
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@pytest.fixture(scope="module")
def real_pair(selected_cuda, gpu_evidence):
    evidence, _ = gpu_evidence
    cached = try_to_load_from_cache("openai-community/gpt2-large", "config.json",
                                    revision=TEACHER_REVISION)
    if not isinstance(cached, str):
        pytest.fail("pinned GPT-2-large config missing from local cache; no network fallback")
    root = Path(cached).parent
    weights = root / "model.safetensors"
    tokenizer_file = root / "tokenizer.json"
    if not weights.is_file() or not tokenizer_file.is_file():
        pytest.fail("pinned GPT-2-large weights/tokenizer missing from local cache")
    actual_sha = _file_hash(weights)
    assert actual_sha == TEACHER_WEIGHTS_SHA256
    teacher = GPT2LMHeadModel.from_pretrained(str(root), local_files_only=True,
                                               attn_implementation="eager").to(selected_cuda).eval()
    teacher.requires_grad_(False)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(1729)
        student = GPT2LMHeadModel(GPT2Config(n_layer=24, n_head=16, n_embd=1024,
                                              _attn_implementation="eager")).eval()
    student_sha = tensor_content_hash(dict(student.state_dict()))
    student = student.to(selected_cuda)
    tokenizer = GPT2TokenizerFast.from_pretrained(str(root), local_files_only=True)
    evidence["stage08_models"] = {
        "teacher_repository": "openai-community/gpt2-large",
        "teacher_revision": TEACHER_REVISION,
        "teacher_weights_sha256": actual_sha,
        "tokenizer_json_sha256": _file_hash(tokenizer_file),
        "teacher_shape": [36, 20, 1280], "student_shape": [24, 16, 1024],
        "student_initialization": "configuration_only_cpu_fp32_seed1729",
        "student_tensor_content_sha256": student_sha,
        "weights_downloaded": 0,
    }
    try:
        yield (GPT2Adapter(teacher, ModelShape(36, 20, 1280)),
               GPT2Adapter(student, ModelShape(24, 16, 1024)), tokenizer, root)
    finally:
        del teacher, student
        torch.cuda.empty_cache()


def _teacher_items(tokenizer, tokenizer_sha256: str) -> tuple[list[dict], str, str]:
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    assert panel["kind"] == "stage08_teacher_engineering_calibration_v1"
    assert panel["status"] == "engineering_only_not_production_panel"
    assert panel["context"] == 128 and len(panel["texts"]) == 4
    source_sha = _file_hash(PANEL_PATH)
    items = []
    for text in panel["texts"]:
        tokens = tokenizer.encode(text, add_special_tokens=False)
        assert 40 < len(tokens) <= 128
        count = len(tokens)
        items.append({"id": hashlib.sha256(text.encode("utf-8")).hexdigest(),
                      "input_ids": tokens + [tokenizer.eos_token_id] * (128 - count),
                      "attention_mask": [1] * count + [0] * (128 - count)})
    panel_sha = payload_digest({"source_sha256": source_sha, "tokenizer_sha256": tokenizer_sha256,
                                "context": 128, "items": items})
    return items, panel_sha, source_sha


def test_real_teacher_s4_fixed_probe_battery_and_restoration(real_pair, gpu_evidence, tmp_path):
    teacher, student, tokenizer, _ = real_pair
    evidence = gpu_evidence[0]
    tokenizer_sha = evidence["stage08_models"]["tokenizer_json_sha256"]
    items, panel_sha, source_sha = _teacher_items(tokenizer, tokenizer_sha)
    teacher_plan = probe_plan(teacher.model, control_seed=CONTROL_SEED)
    student_plan = probe_plan(student.model, control_seed=CONTROL_SEED)
    teacher_epe = epe_directions(teacher.model)
    student_epe = epe_directions(student.model)
    assert teacher_plan["k_top3_all"]["coordinates"] == sorted(
        range(1280), key=lambda i: (-abs(float(teacher_epe["e"][0, i])), i))[:3]
    assert student_plan["k_top3_all"]["coordinates"] == sorted(
        range(1024), key=lambda i: (-abs(float(student_epe["e"][0, i])), i))[:3]
    assert all(0 <= i < 1280 for i in teacher_plan["k_top3_all"]["coordinates"])
    assert all(0 <= i < 1024 for i in student_plan["k_top3_all"]["coordinates"])
    assert len({tuple(teacher_plan[f"k_random{i}_all"]["coordinates"]) for i in range(5)}) == 5

    directions = teacher_epe
    sample = torch.arange(4 * 3 * 1280, device=next(teacher.model.parameters()).device,
                          dtype=torch.float32).reshape(4, 3, 1280) / 10000
    mask = torch.ones(4, 3, dtype=torch.bool, device=sample.device)
    batched = transport_epe_batch(sample, directions["u"][0], directions["u"][1], mask)
    reference = []
    for row in sample:
        delta = torch.dot(row[0], directions["u"][0]) * (directions["u"][1] - directions["u"][0])
        expected = row.clone()
        expected[0] = row[0] + delta
        expected[1] = row[1] - delta
        reference.append(expected)
    parity_error = float((batched - torch.stack(reference)).abs().max())
    sum_error = float((batched[:, :2].sum(1) - sample[:, :2].sum(1)).abs().max())
    assert parity_error <= 1e-5 and sum_error <= 1e-5

    # Transformers 5.3 installs its own persistent output-capture hooks lazily
    # on the first feature forward. Baseline hook accounting after that forward;
    # probe-owned hooks must still disappear after both normal and fault paths.
    warm_ids = torch.tensor([items[0]["input_ids"]], dtype=torch.long, device=sample.device)
    warm_mask = torch.tensor([items[0]["attention_mask"]], dtype=torch.bool, device=sample.device)
    with torch.inference_mode():
        teacher.forward_with_features(input_ids=warm_ids, attention_mask=warm_mask)
        native_logits = teacher.model(input_ids=warm_ids, attention_mask=warm_mask,
                                      use_cache=False).logits
        noop_logits = teacher.forward(input_ids=warm_ids, attention_mask=warm_mask,
            intervention=AttentionIntervention("none"), layer_scope=range(36)).logits
    noop_max_error = float((native_logits - noop_logits).abs().max())
    assert noop_max_error <= 1e-5
    modes = [module.training for module in teacher.model.modules()]
    hooks = [len(module._forward_hooks) for module in teacher.model.modules()]
    cpu_rng = torch.get_rng_state().clone()
    cuda_rng = torch.cuda.get_rng_state_all()
    parameter_sha = _edited_parameter_hash(teacher.model)
    real_mask = torch.tensor([items[0]["attention_mask"]], dtype=torch.bool,
                             device=sample.device)
    for injected_probe in ("k_top3_all", "epe_transport_layer0"):
        with pytest.raises(RuntimeError, match="injected restoration check"):
            with apply_probe(teacher.model, injected_probe, plan=teacher_plan,
                             mask=real_mask):
                raise RuntimeError("injected restoration check")
        assert _edited_parameter_hash(teacher.model) == parameter_sha
        assert [module.training for module in teacher.model.modules()] == modes
        assert [len(module._forward_hooks) for module in teacher.model.modules()] == hooks
    torch.cuda.reset_peak_memory_stats(sample.device)
    started = time.perf_counter()
    result = evaluate_probe_battery(adapter=teacher, items=items,
        store=RecordStore(tmp_path / "s4_probe_records"),
        checkpoint_sha256=TEACHER_WEIGHTS_SHA256, panel_sha256=panel_sha,
        run_id="stage08-real-teacher-calibration-v1", model_role="teacher",
        control_seed=CONTROL_SEED, denominator_floor=DENOMINATOR_FLOOR,
        responsiveness_floor=RESPONSIVENESS_FLOOR,
        provenance={"kind": "engineering_calibration_not_scientific_coverage",
                    "source_panel_sha256": source_sha, "tokenizer_sha256": tokenizer_sha,
                    "teacher_revision": TEACHER_REVISION, "device_role": "rtx4080super",
                    "precision": "fp32"})
    assert result["status"] == "complete" and set(result["probes"]) == set(PROBE_IDS)
    assert all(row["status"] == "complete" and row["complete_item_count"] == 4
               for row in result["probes"].values())
    raw_records = list((tmp_path / "s4_probe_records").glob("*.json"))
    assert len(raw_records) == 4 * len(PROBE_IDS)
    expected_masks = {item["id"]: [item["attention_mask"]] for item in items}
    for path in raw_records:
        payload, _ = verify_envelope(json.loads(path.read_text(encoding="utf-8")))
        key, value = payload["key"], payload["value"]
        assert key["checkpoint_sha256"] == TEACHER_WEIGHTS_SHA256
        assert key["panel_sha256"] == panel_sha and key["probe_id"] in PROBE_IDS
        assert key["provenance"]["source_panel_sha256"] == source_sha
        assert value["attention_mask"] == expected_masks[key["item_id"]]
    assert _edited_parameter_hash(teacher.model) == parameter_sha
    assert [module.training for module in teacher.model.modules()] == modes
    assert [len(module._forward_hooks) for module in teacher.model.modules()] == hooks
    torch.testing.assert_close(torch.get_rng_state(), cpu_rng, rtol=0, atol=0)
    for actual, expected in zip(torch.cuda.get_rng_state_all(), cuda_rng, strict=True):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    evidence["measurements"]["stage08_s4_teacher"] = {
        "status": result["status"], "panel_source_sha256": source_sha,
        "panel_sha256": panel_sha, "panel_item_count": len(items),
        "panel_real_token_counts": [sum(item["attention_mask"]) for item in items],
        "context": 128, "control_seed": CONTROL_SEED,
        "denominator_floor": DENOMINATOR_FLOOR,
        "responsiveness_floor": RESPONSIVENESS_FLOOR,
        "teacher_top3": teacher_plan["k_top3_all"]["coordinates"],
        "student_top3_model_local": student_plan["k_top3_all"]["coordinates"],
        "epe_norms": directions["norms"],
        "epe_batched_reference_max_abs_error": parity_error,
        "epe_vector_sum_max_abs_error": sum_error,
        "teacher_noop_logit_max_abs_error": noop_max_error,
        "parameter_hash_restored": True, "modes_hooks_rng_restored": True,
        "native_output_capture_hook_baseline": "warmed_before_probe; persistent Transformers hook excluded from probe leak check",
        "elapsed_seconds": time.perf_counter() - started,
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(sample.device),
        "probes": result["probes"],
        "scientific_coverage": False,
    }


def _engineering_domain_sources():
    paragraph = ("The observer read the notice and wrote a careful summary. "
                 "Several details were checked against the schedule before the group moved on. ")
    return {
        "sst2": [{"split": "validation", "sentence": f"Review {i}. " + paragraph * 8,
                  "label": "EXCLUDED"} for i in range(100)],
        "gsm8k": [{"split": "test", "question": f"Question {i}. " + paragraph * 8,
                    "answer": "EXCLUDED"} for i in range(100)],
        "humaneval": [{"split": "test", "prompt": f"def task_{i}(value):\n    \"\"\"" + paragraph * 8 + "\"\"\"\n",
                       "canonical_solution": "EXCLUDED", "test": "EXCLUDED"} for i in range(100)],
    }


def test_real_gpu_s6_standard_contexts_and_s5_missing_records(real_pair, gpu_evidence, tmp_path):
    teacher, student, tokenizer, _ = real_pair
    evidence = gpu_evidence[0]
    tokenizer_sha = evidence["stage08_models"]["tokenizer_json_sha256"]
    sources = _engineering_domain_sources()
    document = prepare_s6_domains(sources, tokenizer, tokenizer_id="gpt2-large-cached",
        tokenizer_revision=TEACHER_REVISION, tokenizer_sha256=tokenizer_sha,
        revisions={domain: "agent-authored-engineering-v1" for domain in sources},
        licenses={domain: "agent-authored-fixture" for domain in sources})
    assert "EXCLUDED" not in str(document)
    teacher_map = tuple((s + 1) * 36 // 24 - (0 if (s + 1) * 36 % 24 else 1)
                        for s in range(24))
    store = RecordStore(tmp_path / "s6_gpu_records")
    measurements = {}
    for domain in ("sst2", "gsm8k", "humaneval"):
        paired = {}
        for context in (40, 128):
            items, panel_sha = render_domain_items(document, tokenizer_sha256=tokenizer_sha,
                                                   domain=domain, context=context)
            item = items[0]
            assert item["valid_target_count"] == context - 1
            assert item["labels"] == item["input_ids"]
            result = evaluate_panel(adapter=student, teacher_adapter=teacher,
                teacher_map=teacher_map, items=[item], panel=f"stage08_s6_gpu_{domain}_{context}",
                panel_hash=panel_sha, checkpoint_hash=evidence["stage08_models"]["student_tensor_content_sha256"],
                run_id="stage08-random-student-gpu-smoke-v1", step=0, store=store,
                operations=("clean", "none", "delete", "relocate"), precision="fp32",
                denominator_floor=DENOMINATOR_FLOOR, terminal=False)
            assert all(row["status"] == "complete" for row in result["operations"].values())
            metrics = result["operations"]
            assert metrics["clean"]["metrics"]["valid_targets"] == context - 1
            assert metrics["none"]["metrics"]["self_kl_nats"] == 0
            paired[context] = {"panel_sha256": panel_sha, "item_id": item["id"],
                "valid_targets": context - 1,
                "clean_ce_nats": metrics["clean"]["metrics"]["clean_ce_nats"],
                "delete_delta_ce_nats": metrics["delete"]["metrics"]["delta_ce_nats"],
                "relocate_delta_ce_nats": metrics["relocate"]["metrics"]["delta_ce_nats"],
                "noop_self_kl_nats": metrics["none"]["metrics"]["self_kl_nats"]}
        assert paired[40]["item_id"] == paired[128]["item_id"]
        measurements[domain] = paired
    missing = join_s5([], device_role="rtx4080super", panel_sha256="0" * 64,
                      steps=[0], seeds=[0])
    assert missing["status"] == "incomplete" and len(missing["missing"]) == 4
    evidence["measurements"]["stage08_s6_gpu"] = {
        "status": "complete_engineering_smoke", "context_sizes": [40, 128],
        "domains": measurements, "source_manifest_sha256": document["sha256"],
        "student_state": "random_from_configuration_no_training",
        "scientific_coverage": False,
    }
    evidence["measurements"]["stage08_s5"] = {
        "status": "analysis_only_missing_real_S1_records",
        "missing_condition_count": len(missing["missing"]),
        "new_training_jobs": 0, "fabricated_production_records": 0,
    }
