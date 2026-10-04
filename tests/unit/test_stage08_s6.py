import copy
import importlib.util
from pathlib import Path

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.provenance import seal_payload
from sinklab.evaluate import RecordStore, _key
from sinklab.s6 import (DOMAIN_SOURCE_CONTRACTS, S6Error, equal_item_behavior, evaluate_optional_long_context,
                        evaluate_s6_domains, prepare_s6_domains,
                        render_domain_items, validate_optional_long_context,
                        validate_s6_domains, validate_s6_source_metadata)

ROOT = Path(__file__).resolve().parents[2]
PREP_SPEC = importlib.util.spec_from_file_location(
    "prepare_s6_d25_panels", ROOT / "scripts/prepare_s6_d25_panels.py")
PREP = importlib.util.module_from_spec(PREP_SPEC)
PREP_SPEC.loader.exec_module(PREP)


class Tokenizer:
    eos_token_id = 999

    def encode(self, text, *, add_special_tokens):
        assert not add_special_tokens
        return [ord(char) for char in text]


def panel_fixture():
    sources = {
        "sst2": [{"split": "validation", "sentence": f"sentence-{i}-" + "s" * 150,
                  "label": "LEAK"} for i in range(101)],
        "gsm8k": [{"split": "test", "question": f"question-{i}", "answer": "LEAK"} for i in range(101)],
        "humaneval": [{"split": "test", "prompt": f"def f_{i}(x): pass", "canonical_solution": "LEAK",
                       "test": "LEAK"} for i in range(101)]}
    metadata = source_metadata_fixture()
    return prepare_s6_domains(sources, Tokenizer(), tokenizer_id="tiny",
        tokenizer_revision="a" * 40, tokenizer_sha256="b" * 64,
        source_metadata=metadata, decision_sha256="c" * 64,
        source_lock_sha256="d" * 64)


def source_metadata_fixture():
    metadata = {}
    for domain, contract in DOMAIN_SOURCE_CONTRACTS.items():
        revision = "a" * 40
        license_status = "upstream_ambiguous" if domain == "sst2" else "upstream_stated"
        license_value = "other" if domain == "sst2" else "mit"
        metadata[domain] = {
            **contract,
            "revision": revision,
            "file_sha256": "e" * 64,
            "file_bytes": 1234,
            "row_count": 101,
            "model_input_fields": [contract["field"]],
            "source_uri": f"https://huggingface.co/datasets/{contract['repository']}/resolve/{revision}/{contract['file_path']}",
            "license": {
                "status": license_status,
                "upstream_value": license_value,
                "evidence_uri": f"https://huggingface.co/datasets/{contract['repository']}/blob/{revision}/README.md",
                "evidence_sha256": "f" * 64,
                "evidence_text": ["license evidence fixture"],
                "status_basis": "synthetic test fixture",
            },
        }
    return metadata


def test_same_document_contexts_masks_provenance_and_no_answer_leakage():
    document = panel_fixture()
    payload, _ = validate_s6_domains(document, tokenizer_sha256="b" * 64)
    assert payload["metric_label"] == "causal_language_modeling_not_domain_task_accuracy"
    for domain in ("sst2", "gsm8k", "humaneval"):
        short, short_hash = render_domain_items(document, tokenizer_sha256="b" * 64,
                                                domain=domain, context=40)
        long, long_hash = render_domain_items(document, tokenizer_sha256="b" * 64,
                                              domain=domain, context=128)
        assert len(short) == len(long) == 100 and short_hash != long_hash
        for a, b in zip(short, long):
            assert a["id"] == b["id"]
            assert a["input_ids"] == b["input_ids"][:40]
            for row in (a, b):
                mask = row["attention_mask"]
                assert sum(mask) >= 2
                assert mask == [1] * sum(mask) + [0] * (len(mask) - sum(mask))
                assert sum(mask[i] and mask[i + 1] for i in range(len(mask) - 1)) == sum(mask) - 1
                assert row["labels"] == [token if valid else -100
                                         for token, valid in zip(row["input_ids"], mask)]
                assert row["valid_target_count"] == sum(mask) - 1
                assert row["source_revision"] == "a" * 40
                assert row["source_file_sha256"] == "e" * 64
                assert row["source_license_evidence_sha256"] == "f" * 64
                if domain == "sst2":
                    assert row["source_license_status"] == "upstream_ambiguous"
                    assert row["source_upstream_license_value"] == "other"
                else:
                    assert row["source_license_status"] == "upstream_stated"
                    assert row["source_upstream_license_value"] == "mit"
                assert row["metric_label"] == "causal_language_modeling_not_domain_task_accuracy"
    assert "LEAK" not in str(document)
    damaged = copy.deepcopy(document["payload"])
    damaged["base_panel"]["domains"]["sst2"]["items"][0]["renderings"]["40"]["attention_mask"][0] = 0
    with pytest.raises(ValueError):
        validate_s6_domains(seal_payload(damaged), tokenizer_sha256="b" * 64)


def test_s6_license_metadata_accepts_explicit_uncertainty_without_inventing_license():
    metadata = source_metadata_fixture()
    metadata["humaneval"]["license"]["status"] = "upstream_unspecified"
    metadata["humaneval"]["license"]["upstream_value"] = None
    validated = validate_s6_source_metadata(metadata)
    assert validated["sst2"]["license"]["status"] == "upstream_ambiguous"
    assert validated["sst2"]["license"]["upstream_value"] == "other"
    assert validated["gsm8k"]["license"]["upstream_value"] == "mit"
    assert validated["humaneval"]["license"]["status"] == "upstream_unspecified"


@pytest.mark.parametrize("damage", ["missing", "status", "evidence", "revision", "field", "claimed"])
def test_s6_rejects_incomplete_or_conflicting_source_license_metadata(damage):
    metadata = source_metadata_fixture()
    if damage == "missing":
        del metadata["sst2"]["license"]
    elif damage == "status":
        metadata["sst2"]["license"]["status"] = "unknown"
    elif damage == "evidence":
        metadata["sst2"]["license"]["evidence_sha256"] = None
    elif damage == "revision":
        metadata["sst2"]["source_uri"] = "https://huggingface.co/datasets/nyu-mll/glue/resolve/main/file.parquet"
    elif damage == "field":
        metadata["gsm8k"]["field"] = "answer"
    else:
        metadata["sst2"]["license"]["status"] = "upstream_stated"
        metadata["sst2"]["license"]["upstream_value"] = None
    with pytest.raises(S6Error):
        validate_s6_source_metadata(metadata)


def test_d25_selector_is_repeatable_records_ids_and_never_returns_source_text():
    rows = [{"split": "test", "prompt": f"function-{i}-" + "x" * 70,
             "task_id": f"HumanEval/{i}", "test": "DO_NOT_SELECT_TEST_CODE",
             "canonical_solution": "DO_NOT_SELECT_COMPLETION"} for i in range(101)]
    a, eligible_a = PREP._selection(rows, domain="humaneval", field="prompt",
        id_field="task_id", tokenizer=Tokenizer())
    b, eligible_b = PREP._selection(rows, domain="humaneval", field="prompt",
        id_field="task_id", tokenizer=Tokenizer())
    assert eligible_a == eligible_b == 101
    assert a == b and len(a) == 100
    assert all("document_sha256" in row and "source_record_id" in row for row in a)
    assert "DO_NOT_SELECT_TEST_CODE" not in str(a)
    assert "DO_NOT_SELECT_COMPLETION" not in str(a)


def test_equal_item_lm_pool_differs_from_target_weighting():
    rows = [{"valid_targets": 2, "clean_nll_sum_nats": 2., "edited_nll_sum_nats": 3.,
             "self_kl_sum_nats": .1, "absolute_target_logprob_change_sum_nats": .2,
             "flip_count": 1, "clean_correct_count": 1, "edited_correct_count": 0},
            {"valid_targets": 8, "clean_nll_sum_nats": 24., "edited_nll_sum_nats": 28.,
             "self_kl_sum_nats": .4, "absolute_target_logprob_change_sum_nats": .8,
             "flip_count": 2, "clean_correct_count": 0, "edited_correct_count": 1}]
    result = equal_item_behavior(rows)
    assert result["clean_ce_nats"] == 2.
    assert result["clean_ce_nats"] != (2 + 24) / 10
    assert result["metric_label"] == "causal_language_modeling_not_domain_task_accuracy"
    assert result["clean_next_token_accuracy_fraction"] == .25
    with pytest.raises(S6Error, match="nonempty"):
        equal_item_behavior([])


def test_optional_long_context_requires_scope_memory_and_position_limit(tmp_path):
    model = GPT2LMHeadModel(GPT2Config(vocab_size=23, n_positions=1024, n_ctx=1024,
                                       n_embd=16, n_layer=1, n_head=4))
    with pytest.raises(S6Error, match="approved"):
        validate_optional_long_context(model=model, context=512, approval=None,
                                       memory_validation=None)
    approval = {"status": "approved", "context": 512}
    memory = {"context": 512, "batch_size": 1, "peak_bytes": 100, "limit_bytes": 200}
    assert validate_optional_long_context(model=model, context=512,
        approval=approval, memory_validation=memory)["status"] == "admissible_only"
    short_model = GPT2LMHeadModel(GPT2Config(vocab_size=23, n_positions=512, n_ctx=512,
                                             n_embd=16, n_layer=1, n_head=4))
    with pytest.raises(S6Error, match="positional"):
        validate_optional_long_context(model=short_model, context=1024,
            approval={"status": "approved", "context": 1024},
            memory_validation={"context": 1024, "batch_size": 1, "peak_bytes": 100,
                               "limit_bytes": 200})
    with pytest.raises(S6Error, match="exact real length"):
        evaluate_optional_long_context(adapter=type("Adapter", (), {"model": model})(),
            items=[{"id": "one", "input_ids": [1] * 500,
                    "attention_mask": [1] * 500}], context=512,
            approval=approval, memory_validation=memory,
            panel_sha256="a" * 64, checkpoint_sha256="b" * 64,
            run_id="fixture", step=10000, store=RecordStore(tmp_path),
            run_identity={"fixture": True}, denominator_floor=1e-8, precision="fp32")


@pytest.mark.parametrize("source_study", ["fixture", "S1"])
def test_domain_evaluation_wrapper_pairs_contexts_and_pools_items(tmp_path, monkeypatch, source_study):
    import sinklab.s6 as s6
    from sinklab.followup_policy import admit_s1_followup
    identity = ({"study": "S1", "condition": "C6", "seed": 0,
                 "protocol_sha256": "a" * 64, "model_sha256": "c" * 64,
                 "corpus_sha256": "d" * 64} if source_study == "S1" else {"fixture": True})
    document = panel_fixture()
    original_render = s6.render_domain_items
    def one_item(document, *, tokenizer_sha256, domain, context):
        items, digest = original_render(document, tokenizer_sha256=tokenizer_sha256,
                                        domain=domain, context=context)
        return items[:1], digest
    monkeypatch.setattr(s6, "render_domain_items", one_item)
    def fake_panel(**kwargs):
        scope = [0]
        for item in kwargs["items"]:
            n = sum(item["attention_mask"]) - 1
            behavior = {"valid_targets": n, "clean_nll_sum_nats": float(n),
                "edited_nll_sum_nats": 1.1 * n, "self_kl_sum_nats": .1 * n,
                "absolute_target_logprob_change_sum_nats": .1 * n,
                "flip_count": 0, "clean_correct_count": 0, "edited_correct_count": 0,
                "teacher_kl_sum_nats": None, "teacher_top1_agreement_count": None}
            for op in kwargs["operations"]:
                key = _key(run_id=kwargs["run_id"], step=kwargs["step"],
                    panel=kwargs["panel"], panel_hash=kwargs["panel_hash"],
                    checkpoint_hash=kwargs["checkpoint_hash"], item_id=item["id"],
                    scope=scope, operation=op, strength=0. if op == "clean" else 1.,
                    precision=kwargs["precision"], model_role="student",
                    evaluation_mode="full", run_identity=kwargs["run_identity"],
                    denominator_floor=kwargs["denominator_floor"],
                    followup_policy=admit_s1_followup(identity, study="S6", step=0)
                    if source_study == "S1" else None)
                kwargs["store"].write(key, status="complete", value={"behavior": behavior})
        return {"key": {"scope": scope}, "operations": {op: {"status": "complete"}
                for op in kwargs["operations"]}}
    monkeypatch.setattr(s6, "evaluate_panel", fake_panel)
    result = evaluate_s6_domains(adapter=object(), document=document,
        tokenizer_sha256="b" * 64, checkpoint_sha256="c" * 64,
        run_id="fixture", step=0, store=RecordStore(tmp_path),
        run_identity=identity, denominator_floor=1e-8, precision="fp32")
    assert result["status"] == "complete"
    assert result["pooled"]["40_delete"]["equal_item"]["item_count"] == 3
    assert result["pooled"]["40_delete"]["token_weighted"]["accuracy_definition"] == "next_token_argmax_fraction"
    assert result["paired"]["sst2_delete"][0]["targets_40"] == 39
    assert result["paired"]["sst2_delete"][0]["targets_128"] >= 39
    with pytest.raises(S6Error, match="fixed retained"):
        evaluate_s6_domains(adapter=object(), document=document,
            tokenizer_sha256="b" * 64, checkpoint_sha256="c" * 64,
            run_id="fixture", step=100, store=RecordStore(tmp_path / "invalid"),
            run_identity={"fixture": True}, denominator_floor=1e-8, precision="fp32")
