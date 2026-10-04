"""S6 frozen domain and context evaluation; no task scoring or training."""

from __future__ import annotations

from .followup_policy import admit_s1_followup

from typing import Iterable

from .evaluate import RecordStore, _key, evaluate_panel
from .metrics import aggregate_behavior
from .panels import DOMAIN_FIELDS, prepare_domain_panels, validate_domain_panels
from .provenance import COMMIT_PATTERN, SHA256_PATTERN, payload_digest, seal_payload, verify_envelope


class S6Error(ValueError):
    pass


DOMAIN_SOURCE_CONTRACTS = {
    "sst2": {
        "repository": "nyu-mll/glue", "config": "sst2", "split": "validation",
        "field": "sentence", "file_path": "sst2/validation-00000-of-00001.parquet",
        "columns": ["sentence", "label", "idx"],
        "parquet_columns_read": ["sentence", "idx"],
        "selection_identifier_field": "idx",
    },
    "gsm8k": {
        "repository": "openai/gsm8k", "config": "main", "split": "test",
        "field": "question", "file_path": "main/test-00000-of-00001.parquet",
        "columns": ["question", "answer"],
        "parquet_columns_read": ["question"],
        "selection_identifier_field": "split_row_index",
    },
    "humaneval": {
        "repository": "openai/openai_humaneval", "config": "openai_humaneval",
        "split": "test", "field": "prompt",
        "file_path": "openai_humaneval/test-00000-of-00001.parquet",
        "columns": ["task_id", "prompt", "canonical_solution", "test", "entry_point"],
        "parquet_columns_read": ["task_id", "prompt"],
        "selection_identifier_field": "task_id",
    },
}
LICENSE_STATUSES = {"upstream_stated", "upstream_unspecified", "upstream_ambiguous"}
SOURCE_METADATA_FIELDS = {
    "repository", "revision", "config", "split", "field", "file_path", "file_sha256",
    "file_bytes", "row_count", "columns", "parquet_columns_read", "model_input_fields",
    "selection_identifier_field", "source_uri", "license",
}
LICENSE_METADATA_FIELDS = {
    "status", "upstream_value", "evidence_uri", "evidence_sha256", "evidence_text", "status_basis",
}


def validate_s6_source_metadata(source_metadata: dict) -> dict:
    """Validate pinned source identity and documented license status without requiring a license claim."""
    if not isinstance(source_metadata, dict) or set(source_metadata) != set(DOMAIN_SOURCE_CONTRACTS):
        raise S6Error("complete SST-2, GSM8K, and HumanEval source metadata required")
    normalized = {}
    for domain, contract in DOMAIN_SOURCE_CONTRACTS.items():
        metadata = source_metadata[domain]
        if not isinstance(metadata, dict) or set(metadata) != SOURCE_METADATA_FIELDS:
            raise S6Error(f"{domain} source metadata is incomplete")
        for field, expected in contract.items():
            if metadata[field] != expected:
                raise S6Error(f"{domain} source {field} differs from the approved S6 identity")
        if (not isinstance(metadata["revision"], str) or
                not COMMIT_PATTERN.fullmatch(metadata["revision"])):
            raise S6Error(f"{domain} source revision must be an immutable commit SHA")
        if (not isinstance(metadata["file_sha256"], str) or
                not SHA256_PATTERN.fullmatch(metadata["file_sha256"])):
            raise S6Error(f"{domain} source file SHA-256 is required")
        if (type(metadata["file_bytes"]) is not int or metadata["file_bytes"] <= 0 or
                type(metadata["row_count"]) is not int or metadata["row_count"] <= 0):
            raise S6Error(f"{domain} source size and row count must be positive integers")
        if metadata["model_input_fields"] != [contract["field"]]:
            raise S6Error(f"{domain} model input must use only the registered source field")
        expected_uri = (f"https://huggingface.co/datasets/{contract['repository']}"
                        f"/resolve/{metadata['revision']}/{contract['file_path']}")
        if metadata["source_uri"] != expected_uri:
            raise S6Error(f"{domain} source URI must bind the immutable revision and file")

        license_metadata = metadata["license"]
        if (not isinstance(license_metadata, dict) or
                set(license_metadata) != LICENSE_METADATA_FIELDS):
            raise S6Error(f"{domain} complete license metadata/status/evidence required")
        status = license_metadata["status"]
        value = license_metadata["upstream_value"]
        if not isinstance(status, str) or status not in LICENSE_STATUSES:
            raise S6Error(f"{domain} license status is not recognized")
        if status == "upstream_stated":
            if not isinstance(value, str) or not value.strip():
                raise S6Error(f"{domain} upstream-stated license value must be preserved exactly")
        elif status == "upstream_unspecified" and value is not None:
            raise S6Error(f"{domain} unspecified upstream license value must remain null")
        elif value is not None and not isinstance(value, (str, list)):
            raise S6Error(f"{domain} uncertain upstream license value must be recorded verbatim or null")
        evidence_uri = (f"https://huggingface.co/datasets/{contract['repository']}"
                        f"/blob/{metadata['revision']}/README.md")
        if license_metadata["evidence_uri"] != evidence_uri:
            raise S6Error(f"{domain} license evidence must reference the pinned upstream card")
        if (not isinstance(license_metadata["evidence_sha256"], str) or
                not SHA256_PATTERN.fullmatch(license_metadata["evidence_sha256"]) or
                not isinstance(license_metadata["evidence_text"], list) or
                not license_metadata["evidence_text"] or
                any(not isinstance(line, str) or not line.strip()
                    for line in license_metadata["evidence_text"]) or
                not isinstance(license_metadata["status_basis"], str) or
                not license_metadata["status_basis"].strip()):
            raise S6Error(f"{domain} license status requires hashed source evidence and explanation")
        normalized[domain] = metadata
    return normalized


def prepare_s6_domains(sources: dict[str, Iterable[dict]], tokenizer, *,
                       tokenizer_id: str, tokenizer_revision: str,
                       tokenizer_sha256: str, source_metadata: dict,
                       decision_sha256: str, source_lock_sha256: str) -> dict:
    if (not tokenizer_id or not COMMIT_PATTERN.fullmatch(tokenizer_revision) or
            not SHA256_PATTERN.fullmatch(decision_sha256) or
            not SHA256_PATTERN.fullmatch(source_lock_sha256)):
        raise S6Error("pinned tokenizer, D25 decision, and source lock required")
    source_metadata = validate_s6_source_metadata(source_metadata)
    base, _ = verify_envelope(prepare_domain_panels(sources, tokenizer,
        tokenizer_sha256=tokenizer_sha256,
        revisions={name: metadata["revision"] for name, metadata in source_metadata.items()}))
    return seal_payload({"kind": "s6-domains-v1", "base_panel": base,
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision,
                      "files_sha256": tokenizer_sha256},
        "decision_sha256": decision_sha256,
        "source_lock_sha256": source_lock_sha256,
        "source_metadata": source_metadata,
        "metric_label": "causal_language_modeling_not_domain_task_accuracy"})


def validate_s6_domains(document: dict, *, tokenizer_sha256: str) -> tuple[dict, str]:
    payload, digest = verify_envelope(document)
    if payload.get("kind") != "s6-domains-v1" or payload["tokenizer"]["files_sha256"] != tokenizer_sha256:
        raise S6Error("S6 tokenizer provenance mismatch")
    if payload.get("metric_label") != "causal_language_modeling_not_domain_task_accuracy":
        raise S6Error("invalid S6 metric label")
    if (not payload["tokenizer"].get("id") or
            not COMMIT_PATTERN.fullmatch(payload["tokenizer"].get("revision", ""))):
        raise S6Error("unpinned S6 tokenizer")
    if (not SHA256_PATTERN.fullmatch(payload.get("decision_sha256", "")) or
            not SHA256_PATTERN.fullmatch(payload.get("source_lock_sha256", ""))):
        raise S6Error("D25 decision or S6 source lock is not pinned")
    source_metadata = validate_s6_source_metadata(payload.get("source_metadata"))
    validate_domain_panels(seal_payload(payload["base_panel"]),
                           tokenizer_sha256=tokenizer_sha256)
    for domain in DOMAIN_FIELDS:
        if (payload["base_panel"]["domains"][domain]["revision"] !=
                source_metadata[domain]["revision"]):
            raise S6Error(f"{domain} panel source revision differs from source metadata")
    return payload, digest


def render_domain_items(document: dict, *, tokenizer_sha256: str,
                        domain: str, context: int) -> tuple[list[dict], str]:
    payload, digest = validate_s6_domains(document, tokenizer_sha256=tokenizer_sha256)
    if domain not in DOMAIN_FIELDS or context not in {40, 128}:
        raise S6Error("domain/context must be one registered source and 40 or 128 tokens")
    source = payload["base_panel"]["domains"][domain]
    metadata = payload["source_metadata"][domain]
    license_metadata = metadata["license"]
    items = []
    for row in source["items"]:
        rendering = row["renderings"][str(context)]
        ids = rendering["input_ids"]
        mask = rendering["attention_mask"]
        items.append({"id": row["document_sha256"],
                      "document_sha256": row["document_sha256"],
                      "source_field": row["source_field"],
                      "source_split": row["source_split"],
                      "source_revision": source["revision"],
                      "source_file_sha256": metadata["file_sha256"],
                      "source_license_status": license_metadata["status"],
                      "source_upstream_license_value": license_metadata["upstream_value"],
                      "source_license_evidence_sha256": license_metadata["evidence_sha256"],
                      "input_ids": ids, "attention_mask": mask,
                      "labels": [token if valid else -100 for token, valid in zip(ids, mask)],
                      "valid_target_count": rendering["real_token_count"] - 1,
                      "metric_label": "causal_language_modeling_not_domain_task_accuracy"})
    panel_hash = payload_digest({"s6_manifest_sha256": digest, "domain": domain,
        "context": context, "items": items})
    return items, panel_hash


def equal_item_behavior(values: list[dict]) -> dict:
    """Each document contributes one equally weighted per-target item mean."""
    if not values or any(v.get("valid_targets", 0) <= 0 for v in values):
        raise S6Error("nonempty valid item observations required")
    count = len(values)
    def item_mean(field):
        return sum(v[field] / v["valid_targets"] for v in values) / count
    return {"item_count": count, "pooling": "equal_item",
            "clean_ce_nats": item_mean("clean_nll_sum_nats"),
            "edited_ce_nats": item_mean("edited_nll_sum_nats"),
            "delta_ce_nats": item_mean("edited_nll_sum_nats") - item_mean("clean_nll_sum_nats"),
            "self_kl_nats": item_mean("self_kl_sum_nats"),
            "absolute_target_logprob_change_nats": item_mean("absolute_target_logprob_change_sum_nats"),
            "prediction_flip_fraction": item_mean("flip_count"),
            "clean_next_token_accuracy_fraction": item_mean("clean_correct_count"),
            "edited_next_token_accuracy_fraction": item_mean("edited_correct_count"),
            "total_valid_targets": sum(v["valid_targets"] for v in values),
            "metric_label": "causal_language_modeling_not_domain_task_accuracy"}


def evaluate_s6_domains(*, adapter, document: dict, tokenizer_sha256: str,
                        checkpoint_sha256: str, run_id: str, step: int,
                        store: RecordStore, run_identity: dict,
                        denominator_floor: float, precision: str,
                        contexts: tuple[int, ...] = (40, 128),
                        teacher_adapter=None, teacher_map=None) -> dict:
    """Explicit domain battery; matched IDs, separate context masks and cache keys."""
    followup_policy = (admit_s1_followup(run_identity, study="S6", step=step)
                       if run_identity.get("study") == "S1" else None)
    if contexts != (40, 128):
        raise S6Error("S6 primary domain battery requires paired 40/128 renderings")
    if step not in {0, 500, 2000, 10000}:
        raise S6Error("S6 requires one of the fixed retained checkpoints")
    if not isinstance(checkpoint_sha256, str) or not SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise S6Error("immutable checkpoint SHA-256 required")
    if precision not in {"fp32", "bf16"}:
        raise S6Error("explicit numerical precision required")
    payload, manifest_sha = validate_s6_domains(document, tokenizer_sha256=tokenizer_sha256)
    results = {}
    pooled = {}
    observations = {}
    for context in contexts:
        for domain in DOMAIN_FIELDS:
            items, panel_hash = render_domain_items(document, tokenizer_sha256=tokenizer_sha256,
                                                   domain=domain, context=context)
            panel_name = f"s6_{domain}_{context}"
            result = evaluate_panel(adapter=adapter, items=items, panel=panel_name,
                panel_hash=panel_hash, checkpoint_hash=checkpoint_sha256,
                run_id=run_id, step=step, store=store, teacher_adapter=teacher_adapter,
                teacher_map=teacher_map, operations=("clean", "delete", "relocate"),
                precision=precision,
                denominator_floor=denominator_floor, run_identity=run_identity,
                terminal=False, followup_study="S6")
            results[f"{domain}_{context}"] = result
            scope = result["key"]["scope"]
            for op in ("clean", "delete", "relocate"):
                observations[domain, context, op] = {}
                for item in items:
                    key = _key(run_id=run_id, step=step, panel=panel_name,
                        panel_hash=panel_hash, checkpoint_hash=checkpoint_sha256,
                        item_id=item["id"], scope=scope, operation=op,
                        strength=0. if op == "clean" else 1., precision=precision,
                        model_role="student", evaluation_mode="full", run_identity=run_identity,
                        denominator_floor=denominator_floor, followup_policy=followup_policy)
                    row = store.read(key)
                    if row is None or row["status"] != "complete":
                        raise S6Error(f"missing domain observation: {domain}/{context}/{op}/{item['id']}")
                    observations[domain, context, op][item["id"]] = row["value"]["behavior"]
    for context in contexts:
        for op in ("clean", "delete", "relocate"):
            all_values = [value for domain in DOMAIN_FIELDS
                          for value in observations[domain, context, op].values()]
            weighted = aggregate_behavior(all_values)
            weighted["metric_label"] = "causal_language_modeling_not_domain_task_accuracy"
            weighted["accuracy_definition"] = "next_token_argmax_fraction"
            pooled[f"{context}_{op}"] = {"equal_item": equal_item_behavior(all_values),
                                         "token_weighted": weighted,
                                         "domain_count": 3}
    paired = {}
    for domain in DOMAIN_FIELDS:
        for op in ("clean", "delete", "relocate"):
            short, full = observations[domain, 40, op], observations[domain, 128, op]
            if set(short) != set(full):
                raise S6Error("40/128 document membership differs")
            paired[f"{domain}_{op}"] = [{"document_sha256": ident,
                "targets_40": short[ident]["valid_targets"],
                "targets_128": full[ident]["valid_targets"],
                "delta_ce_128_minus_40":
                    (full[ident]["edited_nll_sum_nats"] - full[ident]["clean_nll_sum_nats"]) / full[ident]["valid_targets"] -
                    (short[ident]["edited_nll_sum_nats"] - short[ident]["clean_nll_sum_nats"]) / short[ident]["valid_targets"]}
                for ident in short]
    return {"status": "complete", "metric_label": "causal_language_modeling_not_domain_task_accuracy",
            "decision_sha256": payload["decision_sha256"],
            "source_lock_sha256": payload["source_lock_sha256"],
            "s6_manifest_sha256": manifest_sha, "source_revisions": {d: payload["base_panel"]["domains"][d]["revision"] for d in DOMAIN_FIELDS},
            "source_license_statuses": {d: payload["source_metadata"][d]["license"]["status"] for d in DOMAIN_FIELDS},
            "results": results, "pooled": pooled, "paired": paired}


def validate_optional_long_context(*, model, context: int, approval: dict | None,
                                   memory_validation: dict | None) -> dict:
    """Admission check only; never truncates or starts a long evaluation."""
    if context not in {512, 1024}:
        raise S6Error("optional context must be exactly 512 or 1024")
    if not approval or approval.get("status") != "approved" or approval.get("context") != context:
        raise S6Error("optional long-context scope is not approved")
    if (not memory_validation or memory_validation.get("context") != context or
            memory_validation.get("batch_size") != 1 or
            type(memory_validation.get("peak_bytes")) is not int or
            type(memory_validation.get("limit_bytes")) is not int or
            memory_validation["peak_bytes"] <= 0 or
            memory_validation["peak_bytes"] >= memory_validation["limit_bytes"]):
        raise S6Error("separate one-item long-context memory validation required")
    limit = getattr(model.config, "n_positions", None) or getattr(model.config, "max_position_embeddings", None)
    if type(limit) is not int or context > limit:
        raise S6Error("context exceeds native positional limit; no truncation allowed")
    return {"context": context, "batch_size": 1, "status": "admissible_only",
            "approved_scope": approval, "memory_validation": memory_validation}


def evaluate_optional_long_context(*, adapter, items: list[dict], context: int,
                                   approval: dict, memory_validation: dict,
                                   panel_sha256: str, checkpoint_sha256: str,
                                   run_id: str, step: int, store: RecordStore,
                                   run_identity: dict, denominator_floor: float,
                                   precision: str, teacher_adapter=None,
                                   teacher_map=None) -> dict:
    """An explicit admitted long-context call, with no clipping or batch adaptation."""
    if run_identity.get("study") == "S1":
        admit_s1_followup(run_identity, study="S6", step=step)
    gate = validate_optional_long_context(model=adapter.model, context=context,
        approval=approval, memory_validation=memory_validation)
    if teacher_adapter is not None:
        validate_optional_long_context(model=teacher_adapter.model, context=context,
            approval=approval, memory_validation=memory_validation)
    if not items or len({item["id"] for item in items}) != len(items):
        raise S6Error("unique frozen long-context windows required")
    for item in items:
        if (len(item["input_ids"]) != context or len(item["attention_mask"]) != context or
                item["attention_mask"] != [1] * context):
            raise S6Error("long-context window must have exact real length; no truncation")
    if not SHA256_PATTERN.fullmatch(panel_sha256) or not SHA256_PATTERN.fullmatch(checkpoint_sha256):
        raise S6Error("pinned long-panel and checkpoint hashes required")
    result = evaluate_panel(adapter=adapter, items=items, panel=f"owt_long{context}",
        panel_hash=panel_sha256, checkpoint_hash=checkpoint_sha256,
        run_id=run_id, step=step, store=store, teacher_adapter=teacher_adapter,
        teacher_map=teacher_map, operations=("clean", "delete", "relocate"),
        precision=precision, denominator_floor=denominator_floor,
        run_identity=run_identity, terminal=False, followup_study="S6")
    return {"gate": gate, "evaluation": result,
            "metric_label": "fixed_window_language_modeling_not_task_accuracy"}
