import copy
import json

import pytest
from transformers import GPT2Config, GPT2LMHeadModel

from sinklab.evaluate import RecordStore, evaluate_panel
from sinklab.models import GPT2Adapter
from sinklab.s5_analysis import S5Error, extract_s1_record, join_s5


def _records(tmp_path, seed=0, execution_context="registered_training"):
    config = GPT2Config(vocab_size=23, n_positions=8, n_ctx=8, n_embd=16,
                        n_layer=2, n_head=4, _attn_implementation="eager")
    teacher = GPT2Adapter(GPT2LMHeadModel(config))
    student = GPT2Adapter(GPT2LMHeadModel(config))
    store = RecordStore(tmp_path / "records")
    items = [{"id": "a", "input_ids": [1, 2, 3, 4], "attention_mask": [1, 1, 1, 1]},
             {"id": "b", "input_ids": [3, 4, 5, 0], "attention_mask": [1, 1, 1, 0]}]
    rows = []
    for index, condition in enumerate(("C1", "C2", "C5", "C6")):
        run_id = f"s1-{condition}-seed{seed}"
        identity = {"study": "S1", "condition": condition, "seed": seed,
                    "model_sha256": "a" * 64, "corpus_sha256": "b" * 64,
                    "protocol_sha256": "c" * 64}
        result = evaluate_panel(adapter=student, teacher_adapter=teacher,
            teacher_map=[0, 1], items=items, panel="owt_dense64",
            panel_hash="d" * 64, checkpoint_hash=f"{index + 1}" * 64,
            run_id=run_id, step=100, store=store,
            operations=("clean", "delete", "relocate"), terminal=False,
            run_identity=identity, denominator_floor=1e-8,
            execution_context=execution_context)
        manifest = {"run_id": run_id, "condition": condition, "seed": seed,
                    "device_role": "rtx4080super", "initialization_sha256": "e" * 64,
                    "data_sha256": "b" * 64, "protocol_sha256": "c" * 64}
        rows.append(extract_s1_record(aggregate_path=result["record_path"],
                                     store_root=store.root, run_manifest=manifest))
    return rows


@pytest.mark.parametrize("seed", [1, 2])
def test_d24_keeps_s5_historical_nonzero_seed_records_usable(tmp_path, seed):
    rows = _records(tmp_path, seed=seed)
    before = {p: p.read_bytes() for p in (tmp_path / "records").glob("*.json")}
    result = join_s5(rows, device_role="rtx4080super", panel_sha256="d" * 64,
                     steps=[100], seeds=[seed])
    assert result["status"] == "complete"
    assert all(row["seed"] == seed for row in result["joined"])
    assert all(p.read_bytes() == content for p, content in before.items())
    assert all("followup_policy" not in json.loads(content)["payload"]["key"]
               for content in before.values())


def test_d24_s5_can_extract_new_seed0_followup_records(tmp_path):
    rows = _records(tmp_path, execution_context="checkpoint_followup")
    assert join_s5(rows, device_role="rtx4080super", panel_sha256="d" * 64,
                   steps=[100], seeds=[0])["status"] == "complete"


def test_s5_reuse_only_verified_records_and_contrasts(tmp_path):
    rows = _records(tmp_path)
    assert all(row["study"] == "S1" for row in rows)
    result = join_s5(rows, device_role="rtx4080super", panel_sha256="d" * 64,
                     steps=[100], seeds=[0])
    assert result["status"] == "complete" and result["missing"] == []
    joined = result["joined"][0]
    assert set(joined["source_run_ids"]) == {"C1", "C2", "C5", "C6"}
    assert joined["comparisons"]["C5_minus_C2"]["sink_s"] == pytest.approx(
        rows[2]["measures"]["sink_s"] - rows[1]["measures"]["sink_s"])
    assert joined["measures_by_condition"]["C1"]["clean_ce_nats"] > 0
    assert joined["source_aggregate_sha256"]["C1"] == rows[0]["source_aggregate_sha256"]
    missing = join_s5(rows[:-1], device_role="rtx4080super",
                      panel_sha256="d" * 64, steps=[100], seeds=[0])
    assert missing["status"] == "incomplete" and missing["joined"] == []
    assert missing["missing"] == [{"step": 100, "seed": 0, "condition": "C6"}]


def test_s5_rejects_mismatched_provenance_and_duplicate_replica(tmp_path):
    rows = _records(tmp_path)
    bad = copy.deepcopy(rows)
    bad[1]["initialization_sha256"] = "f" * 64
    with pytest.raises(S5Error, match="incompatible"):
        join_s5(bad, device_role="rtx4080super", panel_sha256="d" * 64,
                steps=[100], seeds=[0])
    with pytest.raises(S5Error, match="duplicate"):
        join_s5(rows + [copy.deepcopy(rows[0])], device_role="rtx4080super",
                panel_sha256="d" * 64, steps=[100], seeds=[0])
    with pytest.raises(S5Error, match="source provenance"):
        malformed = copy.deepcopy(rows)
        malformed[0]["source_aggregate_sha256"] = "not-a-hash"
        join_s5(malformed, device_role="rtx4080super", panel_sha256="d" * 64,
                steps=[100], seeds=[0])
