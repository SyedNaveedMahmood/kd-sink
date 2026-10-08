import pytest
from sinklab.mechanistic_panel import select_disjoint_confirmation


def source():
    blocks = [{"id": f"b{i}", "source_indices": [i]} for i in range(5)]
    docs = [{"source_index": i, "normalized_text_sha256": ("0" if i in (0, 2) else str(i))*64} for i in range(5)]
    return {"partitions": {"evaluation": {"blocks": blocks, "documents": docs}}}, {"owt_full300": ["b0"], "owt_lm2000": ["b0", "b2", "b4", "b3", "b1"]}


def test_selection_uses_source_text_identity_and_original_order():
    ids, receipt = select_disjoint_confirmation(*source(), count=2)
    assert ids == ["b4", "b3"]
    assert receipt["eligible_count"] == 3 and not receipt["outcomes_used"]
    assert receipt["excluded"][1]["reasons"] == ["normalized_text_overlap"]
    assert receipt["selected"][0]["lm2000_index"] == 2


def test_shared_packed_document_excluded():
    corpus, registered = source()
    corpus["partitions"]["evaluation"]["blocks"][4]["source_indices"] = [4, 0]
    ids, receipt = select_disjoint_confirmation(corpus, registered, count=2)
    assert ids == ["b3", "b1"]
    assert "source_document_overlap" in receipt["excluded"][-1]["reasons"]


@pytest.mark.parametrize("count", [0, -1, True, 4])
def test_no_relaxation_or_implicit_count(count):
    with pytest.raises(ValueError): select_disjoint_confirmation(*source(), count=count)


def test_duplicate_candidates_rejected():
    corpus, registered = source()
    registered["owt_lm2000"].append("b3")
    with pytest.raises(ValueError): select_disjoint_confirmation(corpus, registered, count=1)


def test_id_list_serialization_and_exclusive_creation(tmp_path):
    import json
    import runpy
    from pathlib import Path
    script=runpy.run_path(str(Path(__file__).parents[2]/"scripts/prepare_mechanistic_confirmation.py"))
    path=tmp_path/"ids.json"
    script["write_ids"](path,["a","b"])
    assert json.loads(path.read_text())==["a","b"]
    with pytest.raises(FileExistsError): script["write_ids"](path,["c"])
