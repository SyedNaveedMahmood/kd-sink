"""Freeze the prospectively pinned D25 S6 text panels without evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyarrow.parquet as parquet

from sinklab.data import local_tokenizer, normalized_text, tokenizer_files_hash
from sinklab.panels import SALT, _rank
from sinklab.provenance import canonical_json_bytes, seal_payload, verify_envelope
from sinklab.s6 import (prepare_s6_domains, render_domain_items,
                        validate_s6_domains, validate_s6_source_metadata)


EXCLUDED_SOURCE_FIELDS = {"label", "answer", "canonical_solution", "test", "entry_point"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_envelope(path: Path) -> tuple[dict, str]:
    return verify_envelope(json.loads(Path(path).read_text(encoding="utf-8")))


def _selection(rows: list[dict], *, domain: str, field: str, id_field: str,
               tokenizer) -> tuple[list[dict], int]:
    candidates = {}
    eligible_count = 0
    for row_index, row in enumerate(rows):
        text = normalized_text(row[field])
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < 2:
            continue
        eligible_count += 1
        document_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        if id_field == "split_row_index":
            identifier = {"kind": "split_row_index", "value": row_index}
        else:
            identifier = {"kind": id_field, "value": row[id_field]}
        candidates.setdefault(document_sha, {"source_row_index": row_index,
            "source_record_id": identifier})
    ranked = sorted(candidates, key=lambda digest: (_rank(digest, domain), digest))
    if len(ranked) < 100:
        raise ValueError(f"{domain} has fewer than 100 unique eligible source documents")
    return [{"rank_sha256": _rank(digest, domain), "document_sha256": digest,
             **candidates[digest]} for digest in ranked[:100]], len(ranked)


def build_panels(*, repo_root: Path, source_root: Path, tokenizer_dir: Path,
                 output_dir: Path) -> dict:
    repo_root = Path(repo_root).resolve()
    source_root = Path(source_root).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir == repo_root or repo_root in output_dir.parents:
        raise ValueError("raw/frozen S6 panel output must stay outside the Git repository")
    if source_root == repo_root or repo_root in source_root.parents:
        raise ValueError("raw S6 dataset snapshots must stay outside the Git repository")
    if output_dir.exists():
        raise FileExistsError(f"refusing to overwrite frozen output: {output_dir}")

    lock, source_lock_sha = read_envelope(
        repo_root / "protocols/s6_external_dataset_sources_d25.json")
    decision, decision_sha = read_envelope(
        repo_root / "protocols/s1_researcher_amendment_d25_s6_external_datasets_20261004.json")
    if (lock.get("kind") != "s6-external-dataset-source-lock-v1" or
            decision.get("decision_id") != "D25" or
            decision.get("source_lock_sha256") != source_lock_sha or
            decision.get("d24_policy_sha256") !=
            "46351d8e32ef1ef6238af18c11676e45e3d439e0e46942d1e61e35b8001851e6"):
        raise ValueError("D25/source-lock seal or D24 lineage mismatch")
    selection_policy = lock["selection"]
    if (selection_policy.get("items_per_domain") != 100 or
            selection_policy.get("source_input_fields_only") is not True or
            selection_policy.get("minimum_tokens") != 2 or
            selection_policy.get("contexts") != [40, 128] or
            SALT != "e6a-v2-panels" or
            selection_policy.get("normalization") !=
            "Unicode NFC; CRLF to LF; strip outer whitespace; empty text rejected" or
            selection_policy.get("tokenization") !=
            "pinned GPT-2 tokenizer encode(add_special_tokens=False)" or
            selection_policy.get("deduplication") !=
            "one candidate per document_sha256; first source occurrence retained for external selection provenance" or
            selection_policy.get("optional_contexts") != {"512": "disabled", "1024": "disabled"} or
            selection_policy.get("answers_completions_test_code_in_model_inputs") is not False or
            selection_policy.get("pairing") !=
            "identical 100 document hashes in the same order for both contexts" or
            selection_policy.get("padding") !=
            "right pad with GPT-2 EOS token; attention mask 1 for real tokens and 0 for pad" or
            selection_policy.get("ranking") !=
            'ascending tuple (SHA-256("e6a-v2-panels:<domain>:<document_sha256>"), document_sha256)'):
        raise ValueError("D25 S6 selection/context/no-leakage policy differs from implementation")
    metadata = validate_s6_source_metadata(lock["domains"])

    artifact_lock, _ = read_envelope(repo_root / "protocols/artifact.lock.json")
    tokenizer_lock = artifact_lock["tokenizer"]
    actual_tokenizer_sha = tokenizer_files_hash(tokenizer_dir)
    expected_tokenizer = lock["tokenizer"]
    if (tokenizer_lock.get("id") != expected_tokenizer["id"] or
            tokenizer_lock.get("revision") != expected_tokenizer["revision"] or
            tokenizer_lock.get("files_sha256") != expected_tokenizer["files_sha256"] or
            actual_tokenizer_sha != expected_tokenizer["files_sha256"]):
        raise ValueError("local tokenizer identity/revision/file hash differs from D25 and S1 lock")
    tokenizer = local_tokenizer(tokenizer_dir)
    if tokenizer.eos_token_id != expected_tokenizer["eos_pad_token_id"]:
        raise ValueError("local tokenizer EOS/padding token differs from D25")

    sources = {}
    selected = {}
    source_receipts = {}
    for domain, source in metadata.items():
        source_dir = source_root / domain
        parquet_path = source_dir / Path(source["file_path"]).name
        card_path = source_dir / "README.md"
        if (sha256_file(parquet_path) != source["file_sha256"] or
                parquet_path.stat().st_size != source["file_bytes"]):
            raise ValueError(f"{domain} pinned source parquet hash or size mismatch")
        license_record = source["license"]
        if sha256_file(card_path) != license_record["evidence_sha256"]:
            raise ValueError(f"{domain} pinned license evidence hash mismatch")

        parquet_file = parquet.ParquetFile(parquet_path)
        schema_columns = parquet_file.schema_arrow.names
        if (schema_columns != source["columns"] or
                parquet_file.metadata.num_rows != source["row_count"]):
            raise ValueError(f"{domain} parquet schema or row count differs from pinned source")
        excluded_columns = EXCLUDED_SOURCE_FIELDS.intersection(schema_columns)
        if excluded_columns.intersection(source["parquet_columns_read"]):
            raise ValueError(f"{domain} answer/completion/test-code column selected for reading")
        table = parquet.read_table(parquet_path, columns=source["parquet_columns_read"])
        records = table.to_pylist()
        if len(records) != source["row_count"]:
            raise ValueError(f"{domain} projected source row count changed")
        field = source["field"]
        id_field = source["selection_identifier_field"]
        rows = []
        for row_index, record in enumerate(records):
            selected_row = {"split": source["split"], field: record[field]}
            if id_field != "split_row_index":
                selected_row[id_field] = record[id_field]
            rows.append(selected_row)
        selection, unique_eligible = _selection(rows, domain=domain, field=field,
            id_field=id_field, tokenizer=tokenizer)
        selected[domain] = {"eligible_unique_document_count": unique_eligible,
                            "selected_items": selection}
        sources[domain] = rows
        source_receipts[domain] = {
            "repository": source["repository"], "revision": source["revision"],
            "config": source["config"], "split": source["split"], "field": field,
            "file_path": source["file_path"], "file_sha256": sha256_file(parquet_path),
            "file_bytes": parquet_path.stat().st_size,
            "card_sha256": sha256_file(card_path),
            "rows": len(records), "columns": schema_columns,
            "columns_read": source["parquet_columns_read"],
            "excluded_answer_completion_test_fields": sorted(excluded_columns),
            "license_status": license_record["status"],
        }

    panel_document = prepare_s6_domains(sources, tokenizer,
        tokenizer_id=expected_tokenizer["id"],
        tokenizer_revision=expected_tokenizer["revision"],
        tokenizer_sha256=actual_tokenizer_sha,
        source_metadata=metadata, decision_sha256=decision_sha,
        source_lock_sha256=source_lock_sha)
    payload, panel_envelope_sha = validate_s6_domains(
        panel_document, tokenizer_sha256=actual_tokenizer_sha)

    context_panel_hashes = {}
    for domain in sources:
        expected_ids = [x["document_sha256"] for x in selected[domain]["selected_items"]]
        actual_ids = [x["document_sha256"] for x in
                      payload["base_panel"]["domains"][domain]["items"]]
        if actual_ids != expected_ids or len(actual_ids) != 100:
            raise ValueError(f"{domain} deterministic selection differs from S6 panel recipe")
        for context in (40, 128):
            items, panel_hash = render_domain_items(panel_document,
                tokenizer_sha256=actual_tokenizer_sha, domain=domain, context=context)
            if len(items) != 100 or any(item["source_field"] != metadata[domain]["field"] or
                    item["source_split"] != metadata[domain]["split"] or
                    item["valid_target_count"] < 1 for item in items):
                raise ValueError(f"{domain} max{context} panel failed item/padding verification")
            context_panel_hashes[f"{domain}_max{context}"] = panel_hash

    selection_document = seal_payload({
        "kind": "s6-d25-source-selection-v1", "decision_sha256": decision_sha,
        "source_lock_sha256": source_lock_sha, "tokenizer_sha256": actual_tokenizer_sha,
        "selection": selected,
    })
    panel_bytes = canonical_json_bytes(panel_document) + b"\n"
    selection_bytes = canonical_json_bytes(selection_document) + b"\n"
    out_files = {
        "S6_D25_DOMAIN_PANELS.json": panel_bytes,
        "S6_D25_SOURCE_SELECTION.json": selection_bytes,
    }
    output_dir.mkdir(parents=True, exist_ok=False)
    for filename, content in out_files.items():
        (output_dir / filename).write_bytes(content)
    panel_file_sha = sha256_file(output_dir / "S6_D25_DOMAIN_PANELS.json")
    selection_file_sha = sha256_file(output_dir / "S6_D25_SOURCE_SELECTION.json")
    prep_manifest = seal_payload({
        "kind": "s6-d25-preparation-manifest-v1", "decision_sha256": decision_sha,
        "source_lock_sha256": source_lock_sha,
        "tokenizer": {"id": expected_tokenizer["id"],
                      "revision": expected_tokenizer["revision"],
                      "files_sha256": actual_tokenizer_sha,
                      "eos_pad_token_id": tokenizer.eos_token_id},
        "source_receipts": source_receipts,
        "panel_manifest_sha256": panel_envelope_sha,
        "panel_file_sha256": panel_file_sha,
        "source_selection_envelope_sha256": selection_document["sha256"],
        "source_selection_file_sha256": selection_file_sha,
        "paired_context_panel_sha256": context_panel_hashes,
        "optional_contexts": {"512": "disabled", "1024": "disabled"},
        "scientific_evaluation_launched": False,
    })
    manifest_bytes = canonical_json_bytes(prep_manifest) + b"\n"
    manifest_path = output_dir / "S6_D25_PREPARATION_MANIFEST.json"
    manifest_path.write_bytes(manifest_bytes)
    manifest_file_sha = sha256_file(manifest_path)
    (output_dir / "S6_D25_PREPARATION_MANIFEST.sha256").write_text(
        f"{manifest_file_sha}  {manifest_path.name}\n", encoding="ascii")
    return {"output_dir": str(output_dir), "decision_sha256": decision_sha,
            "source_lock_sha256": source_lock_sha,
            "panel_manifest_sha256": panel_envelope_sha,
            "panel_file_sha256": panel_file_sha,
            "source_selection_sha256": selection_document["sha256"],
            "source_selection_file_sha256": selection_file_sha,
            "preparation_manifest_sha256": prep_manifest["sha256"],
            "preparation_manifest_file_sha256": manifest_file_sha,
            "paired_context_panel_sha256": context_panel_hashes,
            "source_receipts": source_receipts,
            "selected_items_per_domain": 100,
            "optional_contexts": {"512": "disabled", "1024": "disabled"}}


def verify_frozen_panels(*, repo_root: Path, source_root: Path, tokenizer_dir: Path,
                         output_dir: Path) -> dict:
    """Read-only verification of the sealed D25 sources, tokenizer, panels, and receipts."""
    repo_root = Path(repo_root).resolve()
    source_root = Path(source_root).resolve()
    tokenizer_dir = Path(tokenizer_dir).resolve()
    output_dir = Path(output_dir).resolve()
    if (repo_root in source_root.parents or repo_root in output_dir.parents or
            not output_dir.is_dir()):
        raise ValueError("D25 sources and frozen outputs must exist outside Git")

    lock, source_lock_sha = read_envelope(
        repo_root / "protocols/s6_external_dataset_sources_d25.json")
    decision, decision_sha = read_envelope(
        repo_root / "protocols/s1_researcher_amendment_d25_s6_external_datasets_20261004.json")
    if (decision.get("source_lock_sha256") != source_lock_sha or
            decision.get("decision_id") != "D25"):
        raise ValueError("D25/source-lock cross-reference mismatch")
    metadata = validate_s6_source_metadata(lock["domains"])
    artifact_lock, _ = read_envelope(repo_root / "protocols/artifact.lock.json")
    tokenizer_lock = lock["tokenizer"]
    actual_tokenizer_sha = tokenizer_files_hash(tokenizer_dir)
    if (actual_tokenizer_sha != tokenizer_lock["files_sha256"] or
            artifact_lock["tokenizer"].get("id") != tokenizer_lock["id"] or
            artifact_lock["tokenizer"].get("revision") != tokenizer_lock["revision"] or
            artifact_lock["tokenizer"].get("files_sha256") != actual_tokenizer_sha):
        raise ValueError("D25 tokenizer differs from the sealed S1 tokenizer identity")

    expected_files = {
        "S6_D25_DOMAIN_PANELS.json", "S6_D25_SOURCE_SELECTION.json",
        "S6_D25_PREPARATION_MANIFEST.json", "S6_D25_PREPARATION_MANIFEST.sha256",
    }
    if {path.name for path in output_dir.iterdir()} != expected_files:
        raise ValueError("unexpected or missing frozen S6 output file")
    panel_path = output_dir / "S6_D25_DOMAIN_PANELS.json"
    selection_path = output_dir / "S6_D25_SOURCE_SELECTION.json"
    manifest_path = output_dir / "S6_D25_PREPARATION_MANIFEST.json"
    panel_document = json.loads(panel_path.read_text(encoding="utf-8"))
    selection_document = json.loads(selection_path.read_text(encoding="utf-8"))
    manifest, manifest_envelope_sha = read_envelope(manifest_path)
    selection, selection_envelope_sha = verify_envelope(selection_document)
    payload, panel_envelope_sha = validate_s6_domains(
        panel_document, tokenizer_sha256=actual_tokenizer_sha)
    manifest_file_sha = sha256_file(manifest_path)
    sidecar = (output_dir / "S6_D25_PREPARATION_MANIFEST.sha256").read_text(
        encoding="ascii").strip().split()
    if (len(sidecar) != 2 or sidecar[0] != manifest_file_sha or
            sidecar[1] != manifest_path.name or
            manifest.get("decision_sha256") != decision_sha or
            manifest.get("source_lock_sha256") != source_lock_sha or
            manifest.get("tokenizer", {}).get("files_sha256") != actual_tokenizer_sha or
            manifest.get("panel_manifest_sha256") != panel_envelope_sha or
            manifest.get("panel_file_sha256") != sha256_file(panel_path) or
            manifest.get("source_selection_envelope_sha256") != selection_envelope_sha or
            manifest.get("source_selection_file_sha256") != sha256_file(selection_path) or
            selection.get("decision_sha256") != decision_sha or
            selection.get("source_lock_sha256") != source_lock_sha or
            selection.get("tokenizer_sha256") != actual_tokenizer_sha or
            payload.get("decision_sha256") != decision_sha or
            payload.get("source_lock_sha256") != source_lock_sha or
            payload.get("source_metadata") != metadata):
        raise ValueError("frozen S6 panel/selection manifest integrity mismatch")

    for domain, source in metadata.items():
        source_dir = source_root / domain
        file_path = source_dir / Path(source["file_path"]).name
        card_path = source_dir / "README.md"
        if (sha256_file(file_path) != source["file_sha256"] or
                file_path.stat().st_size != source["file_bytes"] or
                sha256_file(card_path) != source["license"]["evidence_sha256"]):
            raise ValueError(f"{domain} pinned source/evidence file changed")
        receipt = manifest["source_receipts"].get(domain, {})
        if (receipt.get("file_sha256") != source["file_sha256"] or
                receipt.get("card_sha256") != source["license"]["evidence_sha256"] or
                receipt.get("license_status") != source["license"]["status"]):
            raise ValueError(f"{domain} source receipt differs from sealed source lock")
        selected = selection["selection"].get(domain, {})
        selected_items = selected.get("selected_items", [])
        panel_items = payload["base_panel"]["domains"][domain]["items"]
        expected_ids = [item["document_sha256"] for item in selected_items]
        if (selected.get("eligible_unique_document_count", 0) < 100 or
                len(selected_items) != 100 or len(set(expected_ids)) != 100 or
                expected_ids != [item["document_sha256"] for item in panel_items]):
            raise ValueError(f"{domain} selected identities/order differs from frozen panel")
        for item in panel_items:
            if set(item) != {"document_sha256", "source_field", "source_split", "renderings"}:
                raise ValueError(f"{domain} panel contains an unapproved input/provenance field")
        for context in (40, 128):
            rendered, panel_sha = render_domain_items(panel_document,
                tokenizer_sha256=actual_tokenizer_sha, domain=domain, context=context)
            if (len(rendered) != 100 or any(row["valid_target_count"] < 1 for row in rendered) or
                    manifest["paired_context_panel_sha256"].get(f"{domain}_max{context}") != panel_sha):
                raise ValueError(f"{domain} max{context} panel hash or item check failed")

    if (manifest.get("optional_contexts") != {"512": "disabled", "1024": "disabled"} or
            manifest.get("scientific_evaluation_launched") is not False):
        raise ValueError("D25 output count, optional context, or execution boundary mismatch")
    return {"status": "verified", "decision_sha256": decision_sha,
        "source_lock_sha256": source_lock_sha, "tokenizer_sha256": actual_tokenizer_sha,
        "panel_manifest_sha256": panel_envelope_sha,
        "panel_file_sha256": sha256_file(panel_path),
        "source_selection_sha256": selection_envelope_sha,
        "preparation_manifest_sha256": manifest_envelope_sha,
        "preparation_manifest_file_sha256": manifest_file_sha,
        "paired_context_panel_sha256": manifest["paired_context_panel_sha256"],
        "selected_items_per_domain": 100,
        "optional_contexts": manifest["optional_contexts"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True,
                        help="external directory with verified per-domain Parquet and README snapshots")
    parser.add_argument("--tokenizer-dir", type=Path, required=True,
                        help="the exact locally locked S1 GPT-2 tokenizer directory")
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="new output directory outside Git; existing output is never overwritten")
    parser.add_argument("--verify-only", action="store_true",
                        help="verify an existing frozen output without writing or evaluating")
    args = parser.parse_args()
    operation = verify_frozen_panels if args.verify_only else build_panels
    print(json.dumps(operation(repo_root=args.repo_root, source_root=args.source_root,
        tokenizer_dir=args.tokenizer_dir, output_dir=args.output_dir), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
