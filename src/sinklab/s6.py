"""S6 frozen domain and context evaluation; no task scoring or training."""

from __future__ import annotations

from typing import Iterable

from .evaluate import RecordStore, _key, evaluate_panel
from .metrics import aggregate_behavior
from .panels import DOMAIN_FIELDS, prepare_domain_panels, validate_domain_panels
from .provenance import COMMIT_PATTERN, SHA256_PATTERN, payload_digest, seal_payload, verify_envelope


class S6Error(ValueError):
    pass


def prepare_s6_domains(sources: dict[str, Iterable[dict]], tokenizer, *,
                       tokenizer_id: str, tokenizer_revision: str,
                       tokenizer_sha256: str, revisions: dict[str, str],
                       licenses: dict[str, str]) -> dict:
    if (not tokenizer_id or not COMMIT_PATTERN.fullmatch(tokenizer_revision) or
            set(licenses) != set(DOMAIN_FIELDS) or
            any(not isinstance(value, str) or not value.strip() for value in licenses.values())):
        raise S6Error("pinned tokenizer revision and all source licenses required")
    base, _ = verify_envelope(prepare_domain_panels(sources, tokenizer,
        tokenizer_sha256=tokenizer_sha256, revisions=revisions))
    return seal_payload({"kind": "s6-domains-v1", "base_panel": base,
        "tokenizer": {"id": tokenizer_id, "revision": tokenizer_revision,
                      "files_sha256": tokenizer_sha256},
        "source_licenses": licenses,
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
    validate_domain_panels(seal_payload(payload["base_panel"]),
                           tokenizer_sha256=tokenizer_sha256)
    if (set(payload["source_licenses"]) != set(DOMAIN_FIELDS) or
            any(not isinstance(v, str) or not v.strip() for v in payload["source_licenses"].values())):
        raise S6Error("missing domain license provenance")
    return payload, digest


def render_domain_items(document: dict, *, tokenizer_sha256: str,
                        domain: str, context: int) -> tuple[list[dict], str]:
    payload, digest = validate_s6_domains(document, tokenizer_sha256=tokenizer_sha256)
    if domain not in DOMAIN_FIELDS or context not in {40, 128}:
        raise S6Error("domain/context must be one registered source and 40 or 128 tokens")
    source = payload["base_panel"]["domains"][domain]
    items = []
    for row in source["items"]:
        rendering = row["renderings"][str(context)]
        items.append({"id": row["document_sha256"],
                      "input_ids": rendering["input_ids"],
                      "attention_mask": rendering["attention_mask"]})
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
            "clean_accuracy_fraction": item_mean("clean_correct_count"),
            "edited_accuracy_fraction": item_mean("edited_correct_count"),
            "total_valid_targets": sum(v["valid_targets"] for v in values),
            "metric_label": "causal_language_modeling_not_domain_task_accuracy"}


def evaluate_s6_domains(*, adapter, document: dict, tokenizer_sha256: str,
                        checkpoint_sha256: str, run_id: str, step: int,
                        store: RecordStore, run_identity: dict,
                        denominator_floor: float, precision: str,
                        contexts: tuple[int, ...] = (40, 128),
                        teacher_adapter=None, teacher_map=None) -> dict:
    """Explicit domain battery; matched IDs, separate context masks and cache keys."""
    if contexts != (40, 128):
        raise S6Error("S6 primary domain battery requires paired 40/128 renderings")
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
                terminal=False)
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
                        denominator_floor=denominator_floor)
                    row = store.read(key)
                    if row is None or row["status"] != "complete":
                        raise S6Error(f"missing domain observation: {domain}/{context}/{op}/{item['id']}")
                    observations[domain, context, op][item["id"]] = row["value"]["behavior"]
    for context in contexts:
        for op in ("clean", "delete", "relocate"):
            all_values = [value for domain in DOMAIN_FIELDS
                          for value in observations[domain, context, op].values()]
            pooled[f"{context}_{op}"] = {"equal_item": equal_item_behavior(all_values),
                                         "token_weighted": aggregate_behavior(all_values),
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
            "s6_manifest_sha256": manifest_sha, "source_revisions": {d: payload["base_panel"]["domains"][d]["revision"] for d in DOMAIN_FIELDS},
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
        run_identity=run_identity, terminal=False)
    return {"gate": gate, "evaluation": result,
            "metric_label": "fixed_window_language_modeling_not_task_accuracy"}
