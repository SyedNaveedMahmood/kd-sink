"""S4 evaluation-only positional and parameter-route probes."""

from __future__ import annotations

import hashlib
import random
from contextlib import contextmanager
from typing import Iterator

import torch
from transformers import GPT2LMHeadModel, GPTNeoXForCausalLM

from .evaluate import RecordStore, rng_neutral
from .metrics import aggregate_behavior, behavioral_item, fingerprint, sink_profile
from .models import GPT2Adapter
from .provenance import SHA256_PATTERN, canonical_json_bytes


class ProbeError(ValueError):
    pass


PROBE_VERSION = "s4-gpt2-route-v1"
PROBE_IDS = ("position0_to1", "remove_absolute_positions", "q_bias_all",
             "epe_transport_layer0", "k_top3_all", *(f"k_random{i}_all" for i in range(5)))


def _tensor_hash(tensor: torch.Tensor) -> str:
    raw = tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def epe_directions(model: GPT2LMHeadModel) -> dict:
    """Legacy diagnostic E_j=p_j+MLP_0(p_j), deliberately without LayerNorm."""
    if not isinstance(model, GPT2LMHeadModel):
        raise ProbeError("absolute-position EPE is applicable only to GPT-2")
    if model.transformer.wpe.weight.shape[0] < 2:
        raise ProbeError("EPE requires positions 0 and 1")
    with rng_neutral(model):
        p = model.transformer.wpe.weight[:2].detach()
        e = p + model.transformer.h[0].mlp(p)
        norms = torch.linalg.vector_norm(e.float(), dim=-1)
        if not torch.isfinite(norms).all() or (norms <= 0).any():
            raise ProbeError("zero or nonfinite EPE direction")
        unit = e.float() / norms[:, None]
    return {"e": e.detach().float(), "u": unit.detach(),
            "norms": norms.detach().cpu().tolist(), "definition": "p_j+MLP_0(p_j);no_LayerNorm"}


def transport_epe_batch(output: torch.Tensor, u0: torch.Tensor,
                        u1: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Exact per-example transport, conserving m0+m1 for each batch row."""
    if (output.ndim != 3 or output.shape[1] < 2 or mask.shape != output.shape[:2] or
            not mask[:, :2].bool().all() or u0.shape != output.shape[-1:] or
            u1.shape != output.shape[-1:]):
        raise ProbeError("EPE transport needs two real positions and native-width directions")
    if not torch.isfinite(output).all() or not torch.isfinite(u0).all() or not torch.isfinite(u1).all():
        raise ProbeError("nonfinite EPE input")
    delta = (u1 - u0).to(device=output.device, dtype=output.dtype)
    a = (output[:, 0, :] * u0.to(output.device, output.dtype)).sum(-1)
    changed = output.clone()
    changed[:, 0, :] = output[:, 0, :] + a[:, None] * delta
    changed[:, 1, :] = output[:, 1, :] - a[:, None] * delta
    return changed


def coordinate_controls(model: GPT2LMHeadModel, *, control_seed: int) -> dict:
    """Select top |E0| and five distinct, model-local random K-input sets."""
    if type(control_seed) is not int or control_seed < 0:
        raise ProbeError("explicit nonnegative control RNG seed required")
    e0 = epe_directions(model)["e"][0].abs().cpu().tolist()
    if len(e0) < 8:
        raise ProbeError("width is too small for five distinct controls")
    top = sorted(range(len(e0)), key=lambda i: (-e0[i], i))[:3]
    pool = [i for i in range(len(e0)) if i not in top]
    rng = random.Random(control_seed)
    controls = []
    seen = set()
    while len(controls) < 5:
        picked = tuple(sorted(rng.sample(pool, 3)))
        if picked not in seen:
            seen.add(picked)
            controls.append(list(picked))
    return {"top3": top, "top3_abs_e0": [e0[i] for i in top],
            "random_sets": controls, "control_seed": control_seed,
            "model_width": len(e0), "coordinate_source": "model_local_abs_E0"}


def probe_plan(model, *, control_seed: int) -> dict:
    if isinstance(model, GPTNeoXForCausalLM):
        return {name: {"status": "not_applicable",
                       "reason": "GPT-NeoX rotary architecture has no identical GPT-2 absolute-position/Conv1D route"}
                for name in PROBE_IDS}
    if not isinstance(model, GPT2LMHeadModel):
        raise ProbeError("unsupported S4 probe architecture")
    controls = coordinate_controls(model, control_seed=control_seed)
    layers = list(range(len(model.transformer.h)))
    plan = {
        "position0_to1": {"status": "applicable", "locus": "wpe.position0", "layers": []},
        "remove_absolute_positions": {"status": "applicable", "locus": "wpe.all_positions", "layers": []},
        "q_bias_all": {"status": "applicable", "locus": "c_attn.Q_bias", "layers": layers},
        "epe_transport_layer0": {"status": "applicable", "locus": "layer0.MLP_output.positions0_1", "layers": [0],
                                 "epe_definition": "p_j+MLP_0(p_j);no_LayerNorm"},
        "k_top3_all": {"status": "applicable", "locus": "c_attn.K_input_rows", "layers": layers,
                       "coordinates": controls["top3"], "magnitudes": controls["top3_abs_e0"]},
    }
    for index, coords in enumerate(controls["random_sets"]):
        plan[f"k_random{index}_all"] = {"status": "applicable",
            "locus": "c_attn.K_input_rows", "layers": layers,
            "coordinates": coords, "control_seed": control_seed}
    return plan


@contextmanager
def _temporary_tensors(tensors: list[torch.Tensor]) -> Iterator[None]:
    originals = [t.detach().clone() for t in tensors]
    digests = [_tensor_hash(t) for t in tensors]
    try:
        yield
    finally:
        with torch.no_grad():
            for tensor, original in zip(tensors, originals, strict=True):
                tensor.copy_(original)
        if [_tensor_hash(t) for t in tensors] != digests:
            raise ProbeError("parameter probe failed exact restoration")


@contextmanager
def apply_probe(model: GPT2LMHeadModel, probe_id: str, *,
                plan: dict, mask: torch.Tensor) -> Iterator[None]:
    """Transactional parameter or hook edit; restores on exceptions."""
    if not isinstance(model, GPT2LMHeadModel) or probe_id not in plan or plan[probe_id]["status"] != "applicable":
        raise ProbeError("probe is not applicable to this model")
    width = model.config.n_embd
    if mask.ndim != 2 or mask.shape[1] < 2 or not mask[:, :2].bool().all():
        raise ProbeError("probe needs two real tokens and a valid mask")
    if probe_id == "epe_transport_layer0":
        directions = epe_directions(model)
        def hook(_module, _inputs, output):
            return transport_epe_batch(output, directions["u"][0], directions["u"][1], mask)
        handle = model.transformer.h[0].mlp.register_forward_hook(hook)
        try:
            yield
        finally:
            handle.remove()
        return
    if probe_id in {"position0_to1", "remove_absolute_positions"}:
        tensors = [model.transformer.wpe.weight]
    elif probe_id == "q_bias_all":
        tensors = [model.transformer.h[i].attn.c_attn.bias for i in plan[probe_id]["layers"]]
    else:
        coords = plan[probe_id]["coordinates"]
        if len(coords) != 3 or len(set(coords)) != 3 or any(type(i) is not int or i < 0 or i >= width for i in coords):
            raise ProbeError("K-input coordinates must be three model-local indices")
        tensors = [model.transformer.h[i].attn.c_attn.weight for i in plan[probe_id]["layers"]]
    with _temporary_tensors(tensors):
        with torch.no_grad():
            if probe_id == "position0_to1":
                tensors[0][0].copy_(tensors[0][1])
            elif probe_id == "remove_absolute_positions":
                tensors[0].zero_()
            elif probe_id == "q_bias_all":
                for bias in tensors:
                    bias[:width].zero_()
            else:
                for weight in tensors:
                    weight[coords, width:2 * width] = 0
        yield


def evaluate_probe_battery(*, adapter, items: list[dict], store: RecordStore,
                           checkpoint_sha256: str, panel_sha256: str,
                           run_id: str, model_role: str, control_seed: int,
                           denominator_floor: float, responsiveness_floor: float,
                           provenance: dict, retry_failed: bool = False) -> dict:
    """Freeze one battery definition; report every applicable or inapplicable probe."""
    if any(not isinstance(x, str) or not SHA256_PATTERN.fullmatch(x) for x in (checkpoint_sha256, panel_sha256)):
        raise ProbeError("checkpoint and panel SHA-256 required")
    if not items or len({i["id"] for i in items}) != len(items):
        raise ProbeError("unique frozen panel items required")
    if denominator_floor <= 0 or responsiveness_floor < 0:
        raise ProbeError("explicit nonnegative numerical guards required")
    if not provenance or not isinstance(provenance, dict):
        raise ProbeError("immutable run provenance required")
    plan = probe_plan(adapter.model, control_seed=control_seed)
    if isinstance(adapter.model, GPTNeoXForCausalLM):
        return {"status": "not_applicable", "probes": plan,
                "checkpoint_sha256": checkpoint_sha256, "panel_sha256": panel_sha256}
    if not isinstance(adapter, GPT2Adapter):
        raise ProbeError("GPT-2 adapter required for applicable S4 probes")
    results = {name: [] for name in PROBE_IDS}
    with rng_neutral(adapter.model):
        for item in items:
            device = next(adapter.model.parameters()).device
            ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
            mask = torch.tensor([item["attention_mask"]], dtype=torch.bool, device=device)
            clean = None
            clean_structure = None
            for name in PROBE_IDS:
                key = {"study": "S4", "run_id": run_id, "model_role": model_role,
                    "checkpoint_sha256": checkpoint_sha256, "panel_sha256": panel_sha256,
                    "item_id": item["id"], "probe_id": name, "probe": plan[name],
                    "provenance": provenance, "probe_version": PROBE_VERSION,
                    "denominator_floor": denominator_floor,
                    "responsiveness_floor": responsiveness_floor}
                prior = store.read(key)
                if prior is not None and prior["status"] == "complete":
                    results[name].append(prior)
                    continue
                if prior is not None and not retry_failed:
                    results[name].append(prior)
                    continue
                try:
                    if clean is None:
                        clean = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
                        clean_structure = sink_profile([f.probabilities for f in clean.attention], mask)
                    with apply_probe(adapter.model, name, plan=plan, mask=mask):
                        edited = adapter.forward_with_features(input_ids=ids, attention_mask=mask)
                    edited_structure = sink_profile([f.probabilities for f in edited.attention], mask)
                    behavior = behavioral_item(clean.outputs.logits, edited.outputs.logits, ids, mask)
                    value = {"behavior": behavior, "clean_structure": clean_structure,
                        "probed_structure": edited_structure, "fingerprint": fingerprint(
                            clean_structure["native_layer_mean"], edited_structure["native_layer_mean"],
                            denominator_floor=denominator_floor), "input_token_count": int(mask.sum())}
                    store.write(key, status="complete", value=value, retry_failed=retry_failed)
                except Exception as exc:
                    store.write(key, status="failed", value=None,
                                error=f"{type(exc).__name__}: {exc}", retry_failed=retry_failed)
                results[name].append(store.read(key))
    summaries = {}
    for name, records in results.items():
        complete = [r["value"] for r in records if r["status"] == "complete"]
        failures = [r["key"]["item_id"] for r in records if r["status"] == "failed"]
        if complete:
            clean_s = sum(v["clean_structure"]["native_layer_mean"] for v in complete) / len(complete)
            edited_s = sum(v["probed_structure"]["native_layer_mean"] for v in complete) / len(complete)
            behavior = aggregate_behavior([v["behavior"] for v in complete])
            fp = fingerprint(clean_s, edited_s, denominator_floor=denominator_floor)
            response = "responsive" if (abs(fp["absolute_delta_sink"]) > responsiveness_floor or
                                        behavior["self_kl_nats"] > responsiveness_floor) else "nonresponsive"
        else:
            behavior = fp = response = None
        summaries[name] = {"status": "complete" if len(complete) == len(items) else "incomplete",
            "probe": plan[name], "complete_item_count": len(complete), "failed_item_ids": failures,
            "fingerprint": fp, "behavior": behavior, "responsiveness": response,
            "responsiveness_floor": responsiveness_floor}
    return {"status": "complete" if all(v["status"] == "complete" for v in summaries.values()) else "incomplete",
            "probes": summaries, "checkpoint_sha256": checkpoint_sha256,
            "panel_sha256": panel_sha256, "provenance": provenance,
            "model_role": model_role, "probe_version": PROBE_VERSION}
