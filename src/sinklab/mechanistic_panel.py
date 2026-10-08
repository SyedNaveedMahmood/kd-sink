"""Prepare explicitly selected confirmation blocks from the frozen OWT source.

Selection is explicit or prospectively authorized by a deterministic source-only
rule. Outcomes never enter selection. Shared packed-block ownership matters.
"""
from .calibrated_probes import require, validate_panel_split
from .mechanistic_run import read_json, sha256_file
from .owt_compat import load_owt_corpus, panels_from_validated_corpus
from .provenance import verify_envelope


def select_disjoint_confirmation(corpus, registered, *, count):
    """First eligible LM2000 blocks in frozen order; no fallback or outcomes."""
    require(type(count) is int and count > 0, "explicit positive confirmation count required")
    blocks = {b["id"]: b for b in corpus["partitions"]["evaluation"]["blocks"]}
    documents = {d["source_index"]: d["normalized_text_sha256"]
                 for d in corpus["partitions"]["evaluation"]["documents"]}
    discovery = registered["owt_full300"]
    candidates = registered["owt_lm2000"]
    require(len(set(discovery)) == len(discovery) and len(set(candidates)) == len(candidates), "duplicate registered blocks")
    discovery_docs = {i for ident in discovery for i in blocks[ident]["source_indices"]}
    discovery_hashes = {documents[i] for i in discovery_docs}
    eligible, excluded = [], []
    for index, ident in enumerate(candidates):
        ownership = blocks[ident]["source_indices"]
        reasons = []
        if ident in discovery: reasons.append("block_overlap")
        if discovery_docs.intersection(ownership): reasons.append("source_document_overlap")
        if discovery_hashes.intersection(documents[i] for i in ownership): reasons.append("normalized_text_overlap")
        if reasons: excluded.append({"lm2000_index": index, "id": ident, "reasons": reasons})
        else: eligible.append({"lm2000_index": index, "id": ident})
    require(len(eligible) >= count, "insufficient document-disjoint LM2000 blocks; no fallback permitted")
    selected = eligible[:count]
    return [r["id"] for r in selected], {
        "rule": "first count eligible LM2000 blocks in original registered order; exclude any discovery document ID/text hash or block overlap",
        "candidate_count": len(candidates), "requested_count": count, "eligible_count": len(eligible),
        "selected": selected, "excluded": excluded, "outcomes_used": False,
        "approval_status": "candidate_count_requires_researcher_approval"}


def panel_from_validated_corpus(corpus, registered, confirmation_ids, *, tokenizer_sha256):
    require(isinstance(confirmation_ids,list) and confirmation_ids and len(set(confirmation_ids))==len(confirmation_ids),
            "explicit unique confirmation block IDs required")
    partition = corpus["partitions"]["evaluation"]
    blocks = {b["id"]:b for b in partition["blocks"]}
    docs = {d["source_index"]:d["normalized_text_sha256"] for d in partition["documents"]}
    result = {"context_length":128,"tokenizer":{"model_id":"gpt2","artifact_sha256":tokenizer_sha256}}
    for name, selected in (("discovery",registered["owt_full300"]),("confirmation",confirmation_ids)):
        items=[]
        for ident in selected:
            require(ident in blocks,"selected block absent from frozen evaluation partition")
            block=blocks[ident]
            require(len(block["token_ids"])==128,"production blocks must contain128 tokens")
            items.append({"id":ident,"input_ids":block["token_ids"],"attention_mask":[True]*128,
                "document_ids":[f"owt:{i}" for i in block["source_indices"]],
                "document_hashes":[docs[i] for i in block["source_indices"]]})
        result[name]=items
    validate_panel_split(result["discovery"],result["confirmation"])
    return result


def prepare_frozen_panel(*, artifact_document, corpus_reference, panels_reference, confirmation_ids):
    """Verify original source hashes/envelopes and every packing/tokenizer contract."""
    from .mechanistic_admission import _pinned
    artifact,artifact_sha = verify_envelope(artifact_document)
    require(corpus_reference["sha256"]==artifact["corpus"]["file_sha256"] and
            panels_reference["sha256"]==artifact["panels"]["file_sha256"],"original registered input pins differ")
    corpus_path,panels_path = _pinned(corpus_reference),_pinned(panels_reference)
    corpus,corpus_sha = load_owt_corpus(corpus_path,tokenizer_sha256=artifact["tokenizer"]["files_sha256"])
    registered,registered_sha = verify_envelope(read_json(panels_path))
    require(corpus_sha==artifact["corpus"]["payload_sha256"] and registered_sha==artifact["panels"]["payload_sha256"] and
            panels_from_validated_corpus(corpus,corpus_sha)["payload"]==registered,"registered Full300 source differs")
    result = panel_from_validated_corpus(corpus,registered,confirmation_ids,tokenizer_sha256=artifact["tokenizer"]["files_sha256"])
    require(len(result["discovery"])==300,"registered discovery must be exactly Full300")
    result["preparation"]={"artifact_lock_sha256":artifact_sha,"corpus":corpus_reference,
        "registered_panels":panels_reference,"selection":"explicit confirmation block IDs; authority in separate selection receipt",
        "source_validation":"complete original OWT packing/tokenizer/document ownership contract"}
    require(sha256_file(corpus_path)==corpus_reference["sha256"] and sha256_file(panels_path)==panels_reference["sha256"],
            "source changed during panel preparation")
    return result
