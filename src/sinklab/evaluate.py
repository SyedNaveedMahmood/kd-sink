"""RNG-neutral causal evaluation and idempotent versioned item records."""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator, Sequence

import numpy as np
import torch
from tqdm import tqdm

from .interventions import AttentionIntervention, normalize_layer_scope
from .metrics import (METRIC_VERSION, aggregate_behavior, attention_similarity,
                      behavioral_item, depth_support, fingerprint,
                      guarded_spearman, sink_profile, weighted_depth_wasserstein)
from .provenance import _no_duplicate_keys, canonical_json_bytes, payload_digest, seal_payload, verify_envelope


class EvaluationError(ValueError):
    pass


RETAINED_FULL = {0, 100, 250, 500, 1000, 2000, 5000, 7500, 10000}


def cadence(step: int) -> tuple[str, ...]:
    if type(step) is not int or not 0 <= step <= 10000:
        raise EvaluationError("evaluation clock must be an integer in [0,10000]")
    panels = []
    if step % 100 == 0:
        panels.append("owt_dense64")
    if step in RETAINED_FULL:
        panels.append("owt_full300")
    if step in {0, 10000}:
        panels.append("owt_lm2000")
    return tuple(panels)


@contextmanager
def rng_neutral(*models: torch.nn.Module) -> Iterator[None]:
    """Restore global RNG and every supplied mode, even after an exception."""
    modes = [m.training for m in models]
    py = random.getstate()
    np_state = np.random.get_state()
    cpu = torch.get_rng_state()
    cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
    try:
        for model in models:
            model.eval()
        with torch.inference_mode():
            yield
    finally:
        for model, mode in zip(models, modes):
            model.train(mode)
        random.setstate(py)
        np.random.set_state(np_state)
        torch.set_rng_state(cpu)
        if cuda:
            torch.cuda.set_rng_state_all(cuda)


class RecordStore:
    def __init__(self, root: Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: dict) -> Path:
        return self.root / f"{payload_digest(key)}.json"

    def read(self, key: dict) -> dict | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            payload, _ = verify_envelope(json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys))
        except (OSError, ValueError) as exc:
            raise EvaluationError(f"corrupt metric cache {path.name}") from exc
        if payload.get("key") != key:
            raise EvaluationError("metric cache key collision or stale version")
        return payload

    def write(self, key: dict, *, status: str, value: dict | None,
              error: str | None = None, retry_failed: bool = False) -> Path:
        if status not in {"complete", "failed"}:
            raise EvaluationError("invalid observation status")
        prior = self.read(key)
        if prior is not None and prior["status"] == "complete":
            if status == "complete" and prior["value"] == value:
                return self._path(key)
            raise EvaluationError("completed observation is immutable")
        if prior is not None and not retry_failed:
            raise EvaluationError("failed observation requires explicit retry")
        payload = {"key": key, "status": status, "value": value,
                   "error": error, "schema_version": METRIC_VERSION}
        path = self._path(key)
        temporary = path.with_name(f".{path.stem}.{uuid.uuid4().hex}.tmp")
        try:
            temporary.write_bytes(canonical_json_bytes(seal_payload(payload)) + b"\n")
            with temporary.open("rb+") as stream:
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)
        return path


def _key(*, run_id: str, step: int, panel: str, panel_hash: str,
         checkpoint_hash: str, item_id: str, scope: Sequence[int],
         operation: str, strength: float, precision: str, model_role: str,
         evaluation_mode: str, run_identity: dict, denominator_floor: float) -> dict:
    return {"run_id": run_id, "step": step, "panel": panel, "panel_hash": panel_hash,
            "checkpoint_hash": checkpoint_hash, "item_id": item_id,
            "scope": list(scope), "operation": operation, "strength": strength,
            "precision": precision, "model_role": model_role,
            "evaluation_mode": evaluation_mode,
            "run_identity": run_identity,
            "fingerprint_denominator_floor": denominator_floor,
            "metric_version": METRIC_VERSION, "intervention_version": "prevalue-v1"}


def evaluate_panel(*, adapter, items: Sequence[dict], panel: str, panel_hash: str,
                   checkpoint_hash: str, run_id: str, step: int, store: RecordStore,
                   layer_scope: Sequence[int] | None = None,
                   teacher_adapter=None, teacher_map: Sequence[int] | None = None,
                   operations: Sequence[str] = ("clean", "delete", "relocate"),
                   precision: str = "fp32", model_role: str = "student",
                   retry_failed: bool = False, terminal: bool | None = None,
                   denominator_floor: float | None = None, behavior_only: bool = False,
                   run_identity: dict | None = None) -> dict:
    if not items or len({item["id"] for item in items}) != len(items):
        raise EvaluationError("nonempty unique frozen panel items required")
    if len(set(operations)) != len(operations) or any(op not in {"clean", "none", "delete", "relocate"} for op in operations):
        raise EvaluationError("explicit unique clean/no-op/delete/relocate operations required")
    if behavior_only and operations != ("clean",):
        raise EvaluationError("behavior-only endpoint is clean-only")
    scope = normalize_layer_scope(layer_scope, adapter.layer_count)
    if teacher_adapter is not None and (teacher_map is None or len(teacher_map) != adapter.layer_count):
        raise EvaluationError("teacher layer map must cover student native layers")
    if run_identity is None:
        run_identity = {"engineering_fixture": True}
        if denominator_floor is None:
            denominator_floor = 1e-8
    elif set(run_identity) != {"study", "condition", "seed", "model_sha256",
                                "corpus_sha256", "protocol_sha256"}:
        raise EvaluationError("complete immutable run provenance required")
    elif denominator_floor is None:
        raise EvaluationError("approved fingerprint denominator floor required")
    if terminal is None:
        terminal = sys.stderr.isatty()
    keys = {item["id"]: {op: _key(run_id=run_id, step=step, panel=panel,
                                  panel_hash=panel_hash, checkpoint_hash=checkpoint_hash,
                                  item_id=item["id"], scope=scope, operation=op,
                                  strength=0. if op in {"clean", "none"} else 1.,
                                  precision=precision, model_role=model_role,
                                  evaluation_mode="nll_only" if behavior_only else "full",
                                  run_identity=run_identity, denominator_floor=denominator_floor)
                          for op in operations} for item in items}
    progress = tqdm(total=len(items) * len(operations), disable=not terminal,
                    desc=f"eval {panel} step{step} {model_role}", unit="item-op")
    if not terminal:
        print(json.dumps({"event": "evaluation_start", "run_id": run_id, "step": step,
                          "panel": panel, "items": len(items), "operations": list(operations)}, sort_keys=True), flush=True)
    started = time.perf_counter()
    def log_progress(done: int) -> None:
        if not terminal and (done % 20 == 0 or done == len(items)):
            elapsed = time.perf_counter() - started
            print(json.dumps({"event": "evaluation_progress", "run_id": run_id,
                              "step": step, "panel": panel, "completed_items": done,
                              "total_items": len(items), "elapsed_seconds": elapsed,
                              "eta_seconds": (len(items) - done) * elapsed / done},
                             sort_keys=True), flush=True)
    models = [adapter.model] + ([teacher_adapter.model] if teacher_adapter is not None else [])
    try:
        model_device = next(adapter.model.parameters()).device
        with rng_neutral(*models), torch.autocast(device_type=model_device.type, dtype=torch.bfloat16,
                                                  enabled=precision == "bf16" and model_device.type == "cuda"):
            for item_index, item in enumerate(items, start=1):
                pending = [op for op in operations if (store.read(keys[item["id"]][op]) is None or
                           retry_failed and store.read(keys[item["id"]][op])["status"] == "failed")]
                if not pending:
                    progress.update(len(operations))
                    log_progress(item_index)
                    continue
                ids = torch.tensor([item["input_ids"]], dtype=torch.long,
                                   device=next(adapter.model.parameters()).device)
                mask = torch.tensor([item["attention_mask"]], dtype=torch.bool, device=ids.device)
                try:
                    clean = (adapter.forward(input_ids=ids, attention_mask=mask) if behavior_only else
                             adapter.forward_with_features(input_ids=ids, attention_mask=mask))
                    teacher = ((teacher_adapter.forward(input_ids=ids, attention_mask=mask) if behavior_only else
                                teacher_adapter.forward_with_features(input_ids=ids, attention_mask=mask))
                               if teacher_adapter else None)
                    clean_logits = clean.logits if behavior_only else clean.outputs.logits
                    teacher_logits = (teacher.logits if behavior_only else teacher.outputs.logits) if teacher else None
                    clean_structure = None if behavior_only else sink_profile([f.probabilities for f in clean.attention], mask)
                    similarities = None
                    topology = None
                    if teacher is not None and not behavior_only:
                        similarities = []
                        for student_layer in scope:
                            teacher_layer = teacher_map[student_layer]
                            similarities.append({"student_layer": student_layer,
                                "teacher_layer": teacher_layer,
                                "full": attention_similarity(teacher.attention[teacher_layer].probabilities,
                                                             clean.attention[student_layer].probabilities, mask),
                                "sink_excluded": attention_similarity(teacher.attention[teacher_layer].probabilities,
                                                                       clean.attention[student_layer].probabilities,
                                                                       mask, exclude_sink=True)})
                        student_amplitude = [clean_structure["layer_amplitude"][s] for s in scope]
                        teacher_structure = sink_profile(
                            [teacher.attention[teacher_map[s]].probabilities for s in scope], mask)
                        teacher_amplitude = teacher_structure["layer_amplitude"]
                        student_depth = depth_support(student_amplitude)
                        teacher_depth = depth_support(teacher_amplitude)
                        topology = {"student_amplitude": student_amplitude,
                                    "teacher_amplitude": teacher_amplitude,
                                    "student_normalized_depth_16": student_depth,
                                    "teacher_normalized_depth_16": teacher_depth,
                                    "weighted_wasserstein_depth": weighted_depth_wasserstein(
                                        teacher_amplitude, student_amplitude),
                                    "spearman_depth": guarded_spearman(teacher_depth, student_depth)}
                    for op in pending:
                        key = keys[item["id"]][op]
                        try:
                            if op == "clean":
                                logits = clean_logits
                                probed_structure = clean_structure
                            else:
                                edited = adapter.forward(input_ids=ids, attention_mask=mask,
                                                         output_attentions=True,
                                                         intervention=AttentionIntervention(
                                                             "none" if op == "none" else op),
                                                         layer_scope=scope)
                                logits = edited.logits
                                probed_structure = (sink_profile(edited.attentions, mask)
                                                    if edited.attentions is not None else None)
                            behavior = behavioral_item(clean_logits, logits, ids, mask,
                                                       teacher=teacher_logits)
                            fp = (fingerprint(clean_structure["native_layer_mean"],
                                              probed_structure["native_layer_mean"],
                                              denominator_floor=denominator_floor)
                                  if probed_structure is not None and clean_structure is not None else None)
                            value = ({"input_token_count": int(mask.sum()),
                                      "endpoint_nll_only": {k: behavior[k] for k in
                                      ("valid_targets", "clean_nll_sum_nats", "clean_correct_count")}}
                                     if behavior_only else
                                     {"input_token_count": int(mask.sum()),
                                      "behavior": behavior, "clean_structure": clean_structure,
                                      "probed_structure": probed_structure,
                                      "fingerprint": fp, "topology": topology,
                                      "mapped_attention_similarity": similarities})
                            store.write(key, status="complete", value=value,
                                        retry_failed=retry_failed)
                        except Exception as exc:
                            store.write(key, status="failed", value=None,
                                        error=f"{type(exc).__name__}: {exc}", retry_failed=retry_failed)
                        progress.update(1)
                except Exception as exc:
                    for op in pending:
                        key = keys[item["id"]][op]
                        if store.read(key) is None:
                            store.write(key, status="failed", value=None,
                                        error=f"{type(exc).__name__}: {exc}", retry_failed=retry_failed)
                        progress.update(1)
                log_progress(item_index)
    finally:
        progress.close()
    aggregates = {}
    for op in operations:
        records = [store.read(keys[item["id"]][op]) for item in items]
        complete = [r for r in records if r is not None and r["status"] == "complete"]
        failed = [r["key"]["item_id"] for r in records if r is not None and r["status"] == "failed"]
        missing = [item["id"] for item, r in zip(items, records) if r is None]
        if behavior_only and complete:
            count = sum(r["value"]["endpoint_nll_only"]["valid_targets"] for r in complete)
            ce = sum(r["value"]["endpoint_nll_only"]["clean_nll_sum_nats"] for r in complete) / count
            metrics = {"schema_version": METRIC_VERSION, "valid_targets": count,
                       "clean_ce_nats": ce, "clean_log_ppl": ce,
                       "clean_ppl": math.exp(ce) if ce <= math.log(sys.float_info.max) else None,
                       "ppl_unavailable_reason": None if ce <= math.log(sys.float_info.max) else "exponent_overflow",
                       "clean_accuracy_fraction": sum(r["value"]["endpoint_nll_only"]["clean_correct_count"] for r in complete) / count,
                       "units": {"ce": "nats_per_target", "ppl": "ratio", "accuracy": "fraction"}}
        else:
            metrics = aggregate_behavior([r["value"]["behavior"] for r in complete]) if complete else None
        aggregates[op] = {"status": "complete" if len(complete) == len(items) else "incomplete",
                          "complete_item_ids": [r["key"]["item_id"] for r in complete],
                          "failed_item_ids": failed, "missing_item_ids": missing,
                          "metrics": metrics}
    aggregate_key = {"run_id": run_id, "step": step, "panel": panel, "panel_hash": panel_hash,
                     "checkpoint_hash": checkpoint_hash, "scope": list(scope),
                     "precision": precision, "model_role": model_role,
                     "run_identity": run_identity,
                     "fingerprint_denominator_floor": denominator_floor,
                     "operations": list(operations),
                     "evaluation_mode": "nll_only" if behavior_only else "full",
                     "metric_version": METRIC_VERSION, "kind": "aggregate"}
    # Aggregate can gain items after an explicit failed-item retry; atomic replace is safe.
    aggregate_path = store.root / f"aggregate-{payload_digest(aggregate_key)}.json"
    temporary = aggregate_path.with_name(f".{aggregate_path.stem}.{uuid.uuid4().hex}.tmp")
    document = seal_payload({"key": aggregate_key, "operations": aggregates})
    if aggregate_path.exists():
        try:
            old = json.loads(aggregate_path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
            verify_envelope(old)
        except (OSError, ValueError) as exc:
            raise EvaluationError("corrupt aggregate record") from exc
        if old == document:
            return {"key": aggregate_key, "operations": aggregates, "record_path": str(aggregate_path)}
    try:
        temporary.write_bytes(canonical_json_bytes(document) + b"\n")
        with temporary.open("rb+") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, aggregate_path)
    finally:
        temporary.unlink(missing_ok=True)
    if not terminal:
        print(json.dumps({"event": "evaluation_complete", "run_id": run_id,
                          "step": step, "panel": panel,
                          "status": "complete" if all(x["status"] == "complete" for x in aggregates.values()) else "incomplete"},
                         sort_keys=True), flush=True)
    return {"key": aggregate_key, "operations": aggregates, "record_path": str(aggregate_path)}


def precision_diagnostics(adapter, ids: torch.Tensor, mask: torch.Tensor) -> dict:
    """Stage06 harness: compare one pinned FP32 forward with BF16 autocast."""
    from .metrics import behavioral_item

    if ids.device != next(adapter.model.parameters()).device or mask.shape != ids.shape:
        raise EvaluationError("diagnostic inputs must match adapter device and shape")
    with rng_neutral(adapter.model):
        with torch.autocast(device_type=ids.device.type, enabled=False):
            fp32 = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
            noop = adapter.forward(input_ids=ids, attention_mask=mask,
                                   intervention=AttentionIntervention("none"), output_attentions=True)
        with torch.autocast(device_type=ids.device.type, dtype=torch.bfloat16):
            bf16 = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
        f_logits, b_logits = fp32.outputs.logits.float(), bf16.outputs.logits.float()
        f = behavioral_item(f_logits, f_logits, ids, mask)
        b = behavioral_item(b_logits, b_logits, ids, mask)
        row_errors = []
        for feature in bf16.attention:
            rows = feature.valid_edges.any(-1).expand_as(feature.probabilities.sum(-1))
            row_errors.append(float((feature.probabilities.sum(-1)[rows] - 1).abs().max()))
        return {"schema_version": METRIC_VERSION,
                "fp32_vs_bf16_logit_max_abs": float((f_logits - b_logits).abs().max()),
                "fp32_vs_bf16_ce_abs_nats": abs(f["clean_nll_sum_nats"] - b["clean_nll_sum_nats"]) / f["valid_targets"],
                "fp32_noop_logit_max_abs": float((f_logits - noop.logits.float()).abs().max()),
                "bf16_probability_row_sum_max_abs": max(row_errors),
                "device": str(ids.device), "status": "diagnostic_only"}
