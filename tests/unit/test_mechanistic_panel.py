import pytest
from sinklab.mechanistic_panel import panel_from_validated_corpus


def fixture():
    corpus={"partitions":{"evaluation":{"documents":[{"source_index":i,"normalized_text_sha256":str(i)*64} for i in range(4)],
        "blocks":[{"id":f"block{i}","token_ids":[i+1]*128,"source_indices":[i]} for i in range(4)]}}}
    registered={"owt_full300":["block0","block1"]}
    return corpus,registered


def test_explicit_confirmation_preserves_tokens_order_and_underlying_documents():
    corpus,registered=fixture()
    result=panel_from_validated_corpus(corpus,registered,["block3","block2"],tokenizer_sha256="a"*64)
    assert [r["id"] for r in result["confirmation"]]==["block3","block2"]
    assert result["confirmation"][0]["input_ids"]==[4]*128
    assert result["confirmation"][0]["document_ids"]==["owt:3"]
    assert result["confirmation"][0]["document_hashes"]==["3"*64]


@pytest.mark.parametrize("failure",["same_block","shared_document","same_text","unknown","duplicate","short"])
def test_independent_block_ids_do_not_bypass_document_disjointness(failure):
    corpus,registered=fixture();selection=["block2"]
    if failure=="same_block": selection=["block0"]
    elif failure=="shared_document": corpus["partitions"]["evaluation"]["blocks"][2]["source_indices"]=[0,2]
    elif failure=="same_text": corpus["partitions"]["evaluation"]["documents"][2]["normalized_text_sha256"]="0"*64
    elif failure=="unknown": selection=["missing"]
    elif failure=="duplicate": selection=["block2","block2"]
    elif failure=="short": corpus["partitions"]["evaluation"]["blocks"][2]["token_ids"]=[1]*127
    with pytest.raises(ValueError): panel_from_validated_corpus(corpus,registered,selection,tokenizer_sha256="a"*64)
