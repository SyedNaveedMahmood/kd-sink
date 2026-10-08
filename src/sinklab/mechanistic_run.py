"""Single-state E1/E2/E3 evaluator and immutable, auditable artifact bundles.

This is inference only. Scientific admission lives in mechanistic_admission;
engineering fixtures never share a scientific identity or source checkpoint.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path

import torch
from tqdm.auto import tqdm

from .calibrated_probes import (ALPHAS, RouteProbe, activation_effect, apply_route,
    observed_forward, require, route_controls)
from .mechanism_trace import MechanisticGPT2Adapter, OutputEdit, ParityTolerance, factor_summary, isolated_parity
from .mechanism_injection import evaluation_mode, equal_norm_injections, loss_geometry, rescue_for_layer, telescoping
from .metrics import aggregate_behavior, behavioral_item, fingerprint, sink_profile
from .provenance import canonical_json_bytes, payload_digest, _no_duplicate_keys

VERSION = "mechanistic-followup-records-v1"


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError(f"nonfinite JSON: {value}")))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, data):
    # Exclusive creation forbids overwriting a successful or failed attempt.
    with Path(path).open("xb") as stream:
        stream.write(canonical_json_bytes(data) + b"\n")


def validate_items(items, *, length, vocab_size):
    require(type(length) is int and length >= 2 and items and
            len({i["id"] for i in items}) == len(items), "unique nonempty item coverage required")
    for item in items:
        ids, mask = item["input_ids"], item["attention_mask"]
        require(isinstance(item["id"], str) and item["id"] and len(ids) == len(mask) == length,
                "item ID or fixed token length mismatch")
        require(all(type(i) is int and 0 <= i < vocab_size for i in ids), "invalid token ID")
        require(all(type(m) is bool for m in mask) and sum(mask) >= 2 and
                mask == [True]*sum(mask) + [False]*(length-sum(mask)), "boolean right padding required")


def validate_settings(phase, settings, layers):
    require(phase in {"E1", "E2", "E3"}, "explicit E1/E2/E3 phase required")
    tolerance = ParityTolerance(settings["atol"], settings["rtol"])
    require(type(settings["token_chunk"]) is int and settings["token_chunk"] > 0,
            "explicit positive token chunk required")
    require(type(settings["denominator_floor"]) in (int,float) and
            math.isfinite(settings["denominator_floor"]) and settings["denominator_floor"] > 0,
            "explicit positive denominator floor required")
    if phase == "E1":
        require(tuple(settings["alphas"]) == ALPHAS, "E1 uses declared five-alpha grid")
        require(type(settings["control_seed"]) is int and 0 <= settings["control_seed"] < 2**63,
                "explicit coordinate-control seed required")
        scopes = settings["scopes"]
        require(scopes and len({s["name"] for s in scopes}) == len(scopes), "unique explicit scopes required")
        for scope in scopes:
            selected = scope["layers"]
            require(scope["name"] and selected and len(set(selected)) == len(selected) and
                    all(type(i) is int and 0 <= i < layers for i in selected), "invalid route scope")
    else:
        selected = settings["layers"]
        require(selected and len(set(selected)) == len(selected) and
                all(type(i) is int and 0 <= i < layers for i in selected), "preselected native layers required")
    if phase == "E3":
        etas = settings["etas"]
        require(etas and len(set(etas)) == len(etas) and etas[0] == 0 and
                all(type(e) in (int,float) and math.isfinite(e) and e >= 0 for e in etas) and
                all(a < b for a,b in zip(etas,etas[1:])), "explicit increasing eta grid including zero required")
        require(settings["reference"] == "clean_residual_input_before_ln_1", "researcher-selected E3 reference required")
        require(type(settings["norm_floor"]) in (int,float) and math.isfinite(settings["norm_floor"]) and
                settings["norm_floor"] > 0, "explicit positive normalization floor required")
        require(type(settings["control_seed"]) is int and 0 <= settings["control_seed"] < 2**63,
                "explicit direction-control seed required")
        keys = settings["nonsink_keys"]
        require(len(keys) == 2 and len(set(keys)) == 2 and all(type(k) is int and 0 < k < 128 for k in keys) and
                type(settings["query_min"]) is int and max(keys) <= settings["query_min"] < 128,
                "explicit causal non-sink keys/query support required")
        require(settings["orders"] and all(len(o) == layers and set(o) == set(range(layers)) and
                all(type(i) is int for i in o) for o in settings["orders"]), "explicit complete deletion orders required")
        require(len({tuple(o) for o in settings["orders"]}) == len(settings["orders"]), "duplicate deletion orders")
    return tolerance


def _route_plan(model, settings):
    controls = route_controls(model, settings["control_seed"])
    coordinates = {"top3": controls["top3"]}
    coordinates.update({f"random{i}": row for i,row in enumerate(controls["random_sets"])})
    plan = [("none", RouteProbe("none", 0., ()))]
    for scope in settings["scopes"]:
        name, layers = scope["name"], tuple(scope["layers"])
        plan.append((f"reapply_q/{name}", RouteProbe("reapply_q", 1., layers)))
        for alpha in settings["alphas"]:
            plan.append((f"q_bias/{name}/{alpha:g}", RouteProbe("q_bias", alpha, layers)))
            for group, coords in coordinates.items():
                plan.append((f"k_{group}/{name}/{alpha:g}", RouteProbe("k_input", alpha, layers, tuple(coords))))
    for route, layers in (("epe_transport", (0,)), ("position0_to1", ())):
        for alpha in settings["alphas"]:
            plan.append((f"{route}/{alpha:g}", RouteProbe(route, alpha, layers)))
    return plan, controls


def operation_ids(phase, settings):
    if phase == "E1":
        result = ["none"]
        for scope in settings["scopes"]:
            name = scope["name"]
            result.append(f"reapply_q/{name}")
            for alpha in settings["alphas"]:
                result.append(f"q_bias/{name}/{alpha:g}")
                result.extend(f"k_{group}/{name}/{alpha:g}" for group in
                              ("top3", *(f"random{i}" for i in range(5))))
        result.extend(f"{route}/{alpha:g}" for route in ("epe_transport", "position0_to1") for alpha in settings["alphas"])
        return result
    if phase == "E2":
        return [f"delete/layer{layer}" for layer in settings["layers"]]
    result = []
    for layer in settings["layers"]:
        result.extend(f"{name}/layer{layer}" for name in ("single_delete", "single_rescue", "all_delete", "conditional_rescue"))
        result.extend(f"injection/{direction}/layer{layer}/eta{eta:g}" for eta in settings["etas"]
                      for direction in ("sink", "random", "orthogonal", "non_sink"))
    return result


def _effect(clean, edited, ids, mask, settings, tolerance, teacher=None):
    return {"status": "measured", "behavior": behavioral_item(clean, edited, ids, mask,
            teacher=teacher, token_chunk=settings["token_chunk"]),
        "teacher_reference_status": "available" if teacher is not None else "not_supplied",
        "geometry": loss_geometry(clean, edited, ids, mask, tolerance=tolerance, token_chunk=settings["token_chunk"])}


def evaluate_item(adapter, item, phase, settings, *, teacher_logits=None):
    """One item; live forwards only, no layer/head/query/key matrices serialized."""
    require(adapter.model.config._attn_implementation == "eager" and
            all(p.dtype == torch.float32 for p in adapter.model.parameters()), "all phases require FP32 eager models")
    tolerance = validate_settings(phase, settings, adapter.layer_count)
    device = next(adapter.model.parameters()).device
    require(not torch.is_autocast_enabled(device.type), "all phases forbid autocast")
    ids = torch.tensor([item["input_ids"]], dtype=torch.long, device=device)
    mask = torch.tensor([item["attention_mask"]], dtype=torch.bool, device=device)
    rows, diagnostics = [], {}
    with evaluation_mode(adapter.model):
        if phase == "E1":
            clean, baseline = observed_forward(adapter, ids, mask)
            clean_profile = sink_profile(clean.attentions, mask)
            plan, controls = _route_plan(adapter.model, settings)
            diagnostics["coordinate_controls"] = controls
            for label, probe in plan:
                with apply_route(adapter.model, probe, mask) as dose:
                    edited, observed = observed_forward(adapter, ids, mask)
                if probe.alpha == 0 or probe.route in {"none", "reapply_q"}:
                    tolerance.check(edited.logits, clean.logits, "route_noop_logits")
                profile = sink_profile(edited.attentions, mask)
                # Scope-matched sink dose for Q/K, global native profile separately retained.
                scope = probe.layers if probe.route in {"q_bias", "k_input", "reapply_q"} else tuple(range(adapter.layer_count))
                before = sum(clean_profile["layer_amplitude"][i] for i in scope)/len(scope)
                after = sum(profile["layer_amplitude"][i] for i in scope)/len(scope)
                fp = fingerprint(before, after, denominator_floor=settings["denominator_floor"])
                activation = activation_effect(baseline, observed, mask)
                activation["scope_attention_output_delta_rms"] = math.sqrt(math.fsum(
                    activation["layers"][i]["attention_output_delta_rms"]**2 for i in scope)/len(scope))
                row = {"operation": label, **_effect(clean.logits, edited.logits, ids, mask, settings, tolerance, teacher_logits),
                    "probe": dose, "clean_sink": clean_profile, "edited_sink": profile,
                    "sink_fingerprint": fp, "sink_removed_fraction": None if fp["ratio"] is None else 1-fp["ratio"],
                    "activation_dose": activation}
                rows.append(row)
        else:
            for layer in settings["layers"]:
                clean, parity = isolated_parity(adapter, ids, mask, layer, tolerance)
                factors = factor_summary(clean.traces[layer], mask, denominator_floor=settings["denominator_floor"])
                if phase == "E2":
                    rows.append({"operation": f"delete/layer{layer}", "layer": layer, "factors": factors,
                        "parity_checks": parity["checks"], **_effect(clean.outputs.logits, parity["direct"].outputs.logits,
                        ids, mask, settings, tolerance, teacher_logits)})
                    continue
                # Parity must pass before constructing any equal-norm direction.
                rescues = rescue_for_layer(adapter, ids, mask, layer, tolerance=tolerance)
                for name, key in (("single_delete", "single_deleted"), ("single_rescue", "single_rescued"),
                                  ("all_delete", "all_deleted"), ("conditional_rescue", "conditional_rescue")):
                    rows.append({"operation": f"{name}/layer{layer}", "layer": layer,
                        "parity_checks": parity["checks"]+[rescues["parity"]], "factors": factors,
                        "interpretation": rescues["interpretation"], **_effect(clean.outputs.logits,
                        rescues[key].outputs.logits, ids, mask, settings, tolerance, teacher_logits)})
                for eta in settings["etas"]:
                    edits, metadata = equal_norm_injections(clean.traces[layer], mask, eta=eta,
                        norm_floor=settings["norm_floor"], control_seed=settings["control_seed"],
                        query_min=settings["query_min"], nonsink_keys=tuple(settings["nonsink_keys"]))
                    for direction, delta in edits.items():
                        effect = {"status": "unavailable", "behavior": None, "geometry": None,
                                  "reason": "no_common_normalizable_direction_support"}
                        if metadata["common_available_positions"] or eta == 0:
                            edited = adapter.traced_forward(input_ids=ids, attention_mask=mask, trace_layers=(layer,),
                                                           output_edits={layer: OutputEdit("add", delta)})
                            effect = _effect(clean.outputs.logits, edited.outputs.logits, ids, mask, settings, tolerance, teacher_logits)
                            if eta == 0:
                                tolerance.check(edited.outputs.logits, clean.outputs.logits, "zero_eta_logits")
                        rows.append({"operation": f"injection/{direction}/layer{layer}/eta{eta:g}",
                            "layer": layer, "direction": direction, "injection": metadata, **effect})
            if phase == "E3":
                diagnostics["telescopes"] = [telescoping(adapter, ids, mask, order=tuple(order), tolerance=tolerance)
                                             for order in settings["orders"]]
    require([r["operation"] for r in rows] == operation_ids(phase, settings), "operation coverage mismatch")
    return {"item_id": item["id"], "operations": rows, "diagnostics": diagnostics}


def aggregate_records(records, identity, settings):
    require(records and len({r["item_id"] for r in records}) == len(records), "duplicate/missing item records")
    expected = operation_ids(identity["phase"], settings)
    for record in records:
        require(record["identity"] == identity and record["schema_version"] == VERSION and
                [r["operation"] for r in record["operations"]] == expected, "record identity/coverage mismatch")
    aggregate = {}
    for index, operation in enumerate(expected):
        rows = [r["operations"][index] for r in records]
        measured = [r for r in rows if r["status"] == "measured"]
        require(all(r["status"] in {"measured", "unavailable"} for r in rows), "unknown measurement status")
        stats = {"item_count": len(rows), "measured_items": len(measured), "unavailable_items": len(rows)-len(measured),
                 "behavior": aggregate_behavior([r["behavior"] for r in measured]) if measured else None}
        # Item-weighted structure/dose versus token-weighted behavioral/geometry totals.
        if identity["phase"] == "E1":
            stats["sink_fingerprint"] = {k: math.fsum(r["sink_fingerprint"][k] for r in rows)/len(rows)
                                         for k in ("baseline_sink", "probed_sink", "absolute_delta_sink")}
            stats["attention_output_delta_rms_item_mean"] = math.fsum(r["activation_dose"]["attention_output_delta_rms"] for r in rows)/len(rows)
            stats["scope_attention_output_delta_rms_item_mean"] = math.fsum(r["activation_dose"]["scope_attention_output_delta_rms"] for r in rows)/len(rows)
            ratios = [r["sink_removed_fraction"] for r in rows if r["sink_removed_fraction"] is not None]
            stats["sink_removed_fraction_item_mean"] = math.fsum(ratios)/len(ratios) if ratios else None
            stats["sink_fraction_available_items"] = len(ratios)
        stats["geometry"] = {k: math.fsum(r["geometry"][k] for r in measured)
            for k in ("delta_nll_sum_nats", "positive_delta_nll_sum_nats", "negative_delta_nll_sum_nats", "centered_logit_displacement_sum")}
        stats["geometry"]["valid_targets"] = sum(r["geometry"]["valid_targets"] for r in measured)
        stats["geometry"]["identity_max_error"] = max((r["geometry"]["identity_max_error"] for r in measured), default=None)
        aggregate[operation] = stats
    diagnostics = {}
    if identity["phase"] in {"E2", "E3"}:
        diagnostics["factor_summary"] = {}
        for layer in settings["layers"]:
            factors = [next(r["factors"] for r in item["operations"] if r.get("layer") == layer and "factors" in r) for item in records]
            layer_result = {}
            for support in ("all_q_ge1", "second_half_queries"):
                entries = [f[support] for f in factors]
                n = sum(e["real_query_count"] for e in entries)
                pooled = {"real_query_count": n}
                for field in ("head_sink_mass_mean", "projected_delta_norm_mean", "relative_projected_delta_mean"):
                    pooled[field] = math.fsum(e[field]*e["real_query_count"] for e in entries)/n
                for field in ("per_head_conditional_value_norm_mean", "per_head_value_contrast_norm_mean", "per_head_local_delta_norm_mean"):
                    pooled[field] = [math.fsum(e[field][h]*e["real_query_count"] for e in entries)/n for h in range(len(entries[0][field]))]
                pooled["per_head_sink_value_norm_item_mean"] = [math.fsum(e["per_head_sink_value_norm"][h] for e in entries)/len(entries)
                                                                  for h in range(len(entries[0]["per_head_sink_value_norm"]))]
                defined = sum(e["cancellation_ratio_defined_queries"] for e in entries)
                pooled["cancellation_ratio_defined_queries"] = defined
                pooled["cancellation_ratio_mean"] = math.fsum(e["cancellation_ratio_mean"]*e["cancellation_ratio_defined_queries"]
                    for e in entries if e["cancellation_ratio_defined_queries"])/defined if defined else None
                for field in ("zero_projected_head_sum_positions", "clean_output_below_floor_positions"):
                    pooled[field] = sum(e[field] for e in entries)
                layer_result[support] = pooled
            diagnostics["factor_summary"][str(layer)] = layer_result
    if identity["phase"] == "E3":
        diagnostics["telescopes"] = []
        for index, order in enumerate(settings["orders"]):
            entries = [r["diagnostics"]["telescopes"][index] for r in records]
            counts = [r["operations"][0]["behavior"]["valid_targets"] for r in records]
            total = sum(counts)
            diagnostics["telescopes"].append({"order":order,"valid_targets":total,
                "increments_ce_nats":[math.fsum(e["increments"][k]["increment_ce_nats"]*n for e,n in zip(entries,counts,strict=True))/total for k in range(len(order))],
                "all_layer_delta_ce_nats":math.fsum(e["all_layer_delta_ce_nats"]*n for e,n in zip(entries,counts,strict=True))/total,
                "interpretation":"token-weighted, explicitly order-dependent increments"})
    return {"schema_version": VERSION, "identity": identity, "settings": settings,
        "items": [r["item_id"] for r in records], "operations": aggregate, "diagnostics":diagnostics,
        "interpretation": "designated-seed0 descriptive inference; no across-seed replication, equivalence or mediation claim"}


def run_state(adapter, items, *, phase, settings, identity, output: Path, teacher_provider=None):
    """One explicit state; preserve partial evidence on failures, never reuse it."""
    output = Path(output)
    require(identity["phase"] == phase and identity["schema_version"] == VERSION and identity["seed"] == 0,
            "explicit versioned phase/seed0 identity required")
    validate_settings(phase, settings, adapter.layer_count)
    validate_items(items, length=identity["context_length"], vocab_size=adapter.model.config.vocab_size)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output/"invocation.json", {"identity": identity, "settings": settings, "items": [i["id"] for i in items]})
    records, files, start = [], {}, time.monotonic()
    try:
        progress = tqdm(items, desc=f"{phase} {identity['state']} seed0 {identity['run_id']}", unit="item")
        with (output/"events.jsonl").open("x", encoding="utf-8") as events:
            for ordinal, item in enumerate(progress):
                teacher = teacher_provider(item) if teacher_provider else None
                record = {"schema_version": VERSION, "identity": identity,
                          **evaluate_item(adapter, item, phase, settings, teacher_logits=teacher)}
                path = output/f"item-{ordinal:05d}.json"
                write_json(path, record); files[path.name] = sha256_file(path); records.append(record)
                elapsed = time.monotonic()-start
                event = {"phase": phase, "state": identity["state"], "run_id": identity["run_id"], "seed": 0,
                    "device": str(next(adapter.model.parameters()).device), "completed_items": len(records),
                    "total_items": len(items), "elapsed_seconds": elapsed,
                    "eta_seconds": elapsed/len(records)*(len(items)-len(records)),
                    "input_tokens": sum(sum(i["attention_mask"]) for i in items[:ordinal+1]),
                    "cuda_allocated_bytes": torch.cuda.memory_allocated(next(adapter.model.parameters()).device) if next(adapter.model.parameters()).is_cuda else 0,
                    "cuda_reserved_bytes": torch.cuda.memory_reserved(next(adapter.model.parameters()).device) if next(adapter.model.parameters()).is_cuda else 0}
                events.write(json.dumps(event, allow_nan=False)+"\n"); events.flush()
        summary = aggregate_records(records, identity, settings)
        write_json(output/"summary.json", summary)
        files.update({name: sha256_file(output/name) for name in ("summary.json", "invocation.json", "events.jsonl")})
        write_json(output/"manifest.json", {"schema_version": VERSION, "identity": identity, "files": files})
        with (output/"COMPLETE").open("x", encoding="ascii") as marker:
            marker.write(sha256_file(output/"manifest.json")+"\n")
        verify_bundle(output)
        return summary
    except BaseException as exc:
        write_json(output/"FAILED.json", {"schema_version": VERSION, "identity": identity,
                   "completed_items": len(records), "error_type": type(exc).__name__, "error": str(exc)})
        raise


def verify_bundle(output):
    output = Path(output)
    require(not (output/"FAILED.json").exists(), "failed bundle cannot be reused")
    manifest = read_json(output/"manifest.json")
    require(manifest["schema_version"] == VERSION and
            (output/"COMPLETE").read_text(encoding="ascii").strip() == sha256_file(output/"manifest.json"), "incomplete bundle")
    names = set(manifest["files"])
    require(all(Path(name).name == name and not (output/name).is_symlink() for name in names), "unsafe manifested path")
    require({p.name for p in output.iterdir()} == names | {"manifest.json", "COMPLETE"}, "unmanifested/missing files")
    for name, digest in manifest["files"].items():
        require(sha256_file(output/name) == digest, f"artifact hash mismatch: {name}")
    invocation = read_json(output/"invocation.json")
    records = [read_json(output/name) for name in sorted(names) if name.startswith("item-")]
    require(len(records) == len(invocation["items"]) and [r["item_id"] for r in records] == invocation["items"] and
            manifest["identity"] == invocation["identity"], "incomplete item coverage/identity")
    recomputed = aggregate_records(records, invocation["identity"], invocation["settings"])
    require(recomputed == read_json(output/"summary.json"), "summary differs from item reaggregation")
    return recomputed
