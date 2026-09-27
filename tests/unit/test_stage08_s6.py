import copy

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.provenance import seal_payload
from sinklab.evaluate import RecordStore, _key
from sinklab.s6 import (S6Error, equal_item_behavior, evaluate_optional_long_context,
                        evaluate_s6_domains, prepare_s6_domains,
                        render_domain_items, validate_optional_long_context,
                        validate_s6_domains)


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
    return prepare_s6_domains(sources, Tokenizer(), tokenizer_id="tiny",
        tokenizer_revision="a" * 40, tokenizer_sha256="b" * 64,
        revisions={name: "pinned-fixture" for name in sources},
        licenses={name: "fixture" for name in sources})


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
                assert row["source_revision"] == "pinned-fixture"
                assert row["source_license"] == "fixture"
                assert row["metric_label"] == "causal_language_modeling_not_domain_task_accuracy"
    assert "LEAK" not in str(document)
    damaged = copy.deepcopy(document["payload"])
    damaged["base_panel"]["domains"]["sst2"]["items"][0]["renderings"]["40"]["attention_mask"][0] = 0
    with pytest.raises(ValueError):
        validate_s6_domains(seal_payload(damaged), tokenizer_sha256="b" * 64)


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


def test_domain_evaluation_wrapper_pairs_contexts_and_pools_items(tmp_path, monkeypatch):
    import sinklab.s6 as s6
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
                    denominator_floor=kwargs["denominator_floor"])
                kwargs["store"].write(key, status="complete", value={"behavior": behavior})
        return {"key": {"scope": scope}, "operations": {op: {"status": "complete"}
                for op in kwargs["operations"]}}
    monkeypatch.setattr(s6, "evaluate_panel", fake_panel)
    result = evaluate_s6_domains(adapter=object(), document=document,
        tokenizer_sha256="b" * 64, checkpoint_sha256="c" * 64,
        run_id="fixture", step=0, store=RecordStore(tmp_path),
        run_identity={"fixture": True}, denominator_floor=1e-8, precision="fp32")
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
