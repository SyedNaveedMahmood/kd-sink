"""E0: read-only S4 audit and descriptive route trajectories; no model imports."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
from pathlib import Path, PurePosixPath
from typing import Callable

from .provenance import canonical_json_bytes, payload_digest, seal_payload, verify_envelope, _no_duplicate_keys
from .followup_policy import D24_SHA256, admit_s1_followup, original_protocol_sha256

CONDITIONS = tuple(f"C{i}" for i in range(7))
STEPS = (0, 100, 500, 2000, 10000)
PROBES = ("position0_to1", "remove_absolute_positions", "q_bias_all",
          "epe_transport_layer0", "k_top3_all", *(f"k_random{i}_all" for i in range(5)))
CHANNELS = {"delta_sink": "attention_mass_fraction", "delta_ce_nats": "nats_per_target",
            "self_kl_nats": "nats_per_target", "absolute_target_logprob_change_nats": "nats_per_target",
            "prediction_flip_fraction": "fraction"}
VERSION = "mechanistic-e0-v1"
TRAJECTORY_FIELDS = ["model_role", "condition", "step", "probe_id", "baseline_sink", "probed_sink", "delta_sink",
    "relative_delta_sink_fraction", "ratio_unavailable_reason", "clean_ce_nats", "edited_ce_nats",
    "delta_ce_nats", "self_kl_nats", "absolute_target_logprob_change_nats", "prediction_flip_fraction",
    "item_count", "valid_targets", "scope", "probe_plan", "source_protocol_sha256", "checkpoint_sha256",
    "panel_sha256", "summary_file_sha256", "summary_relative_path"]
OUTPUT_FILES = {"E0_PROVENANCE.json", "E0_ANALYSIS.json", "E0_S4_ROUTE_TRAJECTORIES.csv",
    "E0_TEACHER_STUDENT_FINGERPRINTS.csv", "E0_PLOT_DATA.json", "E0_ROUTE_TRAJECTORIES.svg",
    "E0_ROUTE_TRAJECTORIES.png", "SHA256SUMS.txt"}


class E0Error(ValueError):
    """An incomplete, incompatible, or corrupt source/output cannot support E0."""


def require(condition: bool, message: str) -> None:
    if not condition:
        raise E0Error(message)


def digest_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sealed(path: Path) -> tuple[dict, str]:
    data = Path(path).read_bytes()
    try:
        payload, _ = verify_envelope(json.loads(data, object_pairs_hook=_no_duplicate_keys))
    except (ValueError, TypeError) as exc:
        raise E0Error(f"invalid sealed JSON: {path}") from exc
    return payload, hashlib.sha256(data).hexdigest()


def number(value, label: str) -> float:
    require(type(value) in (int, float) and math.isfinite(value), f"nonfinite/invalid {label}")
    return float(value)


def close(actual, expected, label: str, tolerance: float = 1e-8) -> None:
    require(math.isclose(number(actual, label), number(expected, label), rel_tol=0, abs_tol=tolerance),
            f"arithmetic mismatch: {label}")


def fingerprint_recipe(document: dict | None) -> tuple[list[dict], str | None]:
    if document is None:
        return [], None
    payload, digest = verify_envelope(document)
    require(set(payload) == {"kind", "status", "rationale", "discovery_description", "components"} and
            payload["kind"] == "e0-fingerprint-recipe-v1" and payload["status"] == "exploratory",
            "explicit exploratory fingerprint recipe required")
    require(all(isinstance(payload[k], str) and payload[k].strip()
                for k in ("rationale", "discovery_description")), "recipe rationale/discovery required")
    components, seen = payload["components"], set()
    require(isinstance(components, list) and bool(components), "nonempty fingerprint components required")
    for component in components:
        require(isinstance(component, dict) and set(component) == {"probe", "metric", "scale"}, "recipe fields")
        pair = (component["probe"], component["metric"])
        require(pair[0] in PROBES and pair[1] in CHANNELS and pair not in seen, "unknown/duplicate component")
        require(number(component["scale"], "component scale") > 0, "positive component scale required")
        seen.add(pair)
    return components, digest


def trajectory_direction(values: list[float]) -> str:
    changes = [b - a for a, b in zip(values, values[1:])]
    signs = {1 if x > 0 else -1 for x in changes if x != 0}
    return "nonmonotonic" if len(signs) > 1 else ("increasing" if signs == {1} else
            "decreasing" if signs == {-1} else "unchanged")


def analyze(rows: list[dict], recipe: dict | None = None) -> dict:
    """No inference, fitted scales, outcome selection, significance or imputation."""
    required = {(c, s, p) for c in CONDITIONS for s in STEPS for p in PROBES}
    student = {(r["condition"], r["step"], r["probe_id"]): r for r in rows if r["model_role"] == "student"}
    teacher = {r["probe_id"]: r for r in rows if r["model_role"] == "teacher"}
    require(len(rows) == 360 and set(student) == required and set(teacher) == set(PROBES), "complete unique E0 grid required")
    components, recipe_hash = fingerprint_recipe(recipe)
    comparisons, trends = [], []
    for c in CONDITIONS:
        for p in PROBES:
            for metric, units in CHANNELS.items():
                gaps = []
                for s in STEPS:
                    value = number(student[c, s, p][metric], metric)
                    reference = number(teacher[p][metric], metric)
                    gap = abs(value - reference)
                    gaps.append(gap)
                    comparisons.append({"condition": c, "step": s, "probe_id": p, "metric": metric,
                        "units": units, "student": value, "teacher": reference,
                        "signed_difference": value - reference, "absolute_gap": gap,
                        "student_scope": student[c, s, p]["scope"], "teacher_scope": teacher[p]["scope"]})
                delta = gaps[-1] - gaps[2]
                trends.append({"condition": c, "probe_id": p, "metric": metric, "units": units,
                    "gap_500": gaps[2], "gap_10000": gaps[-1], "gap_change_500_to_10000": delta,
                    "endpoint_direction": "toward_teacher" if delta < 0 else "away_from_teacher" if delta > 0 else "unchanged",
                    "five_point_direction": trajectory_direction(gaps),
                    "post500_direction": trajectory_direction(gaps[2:]), "random_initialization_step": 0})
    distances, conclusions = [], []
    if components:
        for c in CONDITIONS:
            values = []
            for s in STEPS:
                distance = math.hypot(*((student[c, s, k["probe"]][k["metric"]] -
                    teacher[k["probe"]][k["metric"]]) / k["scale"] for k in components)) / math.sqrt(len(components))
                require(math.isfinite(distance), "nonfinite scaled distance")
                values.append(distance)
                distances.append({"condition": c, "step": s, "exploratory_rms_distance": distance})
            delta = values[-1] - values[2]
            conclusions.append({"condition": c, "endpoint": "converge" if delta < 0 else "diverge" if delta > 0 else "unchanged",
                "trajectory": trajectory_direction(values[2:]), "endpoint_change": delta})
    return {"version": VERSION, "status": "retrospective_descriptive",
        "comparisons": comparisons, "component_trends": trends, "distances": distances,
        "fingerprint_recipe_sha256": recipe_hash, "fingerprint_components": components,
        "composite_status": "exploratory_explicit_recipe" if components else "insufficient",
        "composite_unavailable_reason": None if components else "No independently specified scaling recipe supplied",
        "condition_conclusions": conclusions, "training_seed_count": 1,
        "limits": ["native24 student versus historical native36 teacher; scopes and effective doses unmatched",
                   "Retrospective exploratory analysis; no new inference or preregistration",
                   "Component gap trends do not establish circuit convergence/divergence",
                   "Step0 is random initialization; model-local coordinates may change across checkpoints",
                   "No equivalence margin, significance test, mediation or across-seed claim"]}


def _summary_rows(payload: dict, label: str, manifest: dict, expected_items: int) -> list[dict]:
    result = payload["result"]
    require(payload["kind"] == "s4-probe-battery-summary-v1" and payload["precision"] == "fp32" and
            payload["panel_sha256"] == manifest["panel"]["panel_manifest_sha256"], f"summary panel/precision: {label}")
    require(result["status"] == "complete" and result["probe_version"] == "s4-gpt2-route-v1" and
            set(result["probes"]) == set(PROBES) and result["panel_sha256"] == payload["panel_sha256"] and
            result["model_role"] == payload["model_role"] and result["step"] == payload["step"], f"summary identity: {label}")
    provenance = result["provenance"]
    require(provenance["control_seed"] == manifest["control_seed"] and provenance["precision"] == "fp32" and
            provenance["denominator_floor"] == manifest["denominator_floor"] and
            provenance["responsiveness_floor"] == manifest["responsiveness_floor"], f"provenance drift: {label}")
    role = payload["model_role"]
    if role == "student":
        identity = result["source_identity"]
        followup = admit_s1_followup(identity, study="S4", step=payload["step"])
        require(payload["training_seed"] == 0 and payload["condition"] in CONDITIONS and
                identity["condition"] == payload["condition"] and identity["run_id"] == payload["run_id"] and
                original_protocol_sha256(identity) == payload["original_protocol_root_sha256"] and
                result["followup_policy"] == followup and payload["D24_sha256"] == D24_SHA256 and
                result["checkpoint_sha256"] == payload["checkpoint_model_sha256"] and
                provenance["source_checkpoint_manifest_sha256"] == payload["checkpoint_manifest_sha256"],
                f"source checkpoint identity: {label}")
        require(identity == manifest["source_inventory"]["sources"][payload["condition"]]["identity"], f"source manifest binding: {label}")
        device = provenance["inference_device"]
    else:
        require(role == "teacher" and payload["step"] is None and result["source_identity"] is None and
                result["followup_policy"] is None and result["checkpoint_sha256"] == payload["checkpoint_sha256"] and
                provenance["teacher_weights_sha256"] == payload["checkpoint_sha256"], "teacher identity")
        device = provenance["device"]
    require(device == manifest["inference_device"], f"device drift: {label}")
    rows = []
    for probe_id in PROBES:
        probe = result["probes"][probe_id]
        require(probe["status"] == "complete" and probe["complete_item_count"] == expected_items and
                probe["failed_item_ids"] == [] and probe["probe"]["status"] == "applicable", f"probe coverage: {label}/{probe_id}")
        behavior, fp = probe["behavior"], probe["fingerprint"]
        require(behavior["schema_version"] == "e6a-v2-metrics-1" and behavior["item_count"] == expected_items and
                behavior["valid_targets"] == expected_items * 127 and fp["denominator_floor"] == manifest["denominator_floor"],
                f"denominators: {label}/{probe_id}")
        baseline, edited = number(fp["baseline_sink"], "baseline"), number(fp["probed_sink"], "edited")
        require(0 <= baseline <= 1 and 0 <= edited <= 1, "sink range")
        close(fp["absolute_delta_sink"], edited - baseline, "delta sink")
        close(behavior["delta_ce_nats"], behavior["edited_ce_nats"] - behavior["clean_ce_nats"], "delta CE")
        ratio = None if abs(baseline) < fp["denominator_floor"] else edited / baseline
        require(fp["ratio"] is None if ratio is None else fp["ratio"] is not None, "ratio availability")
        if ratio is not None:
            close(fp["ratio"], ratio, "sink ratio")
        expected_scope = ([] if probe_id in ("position0_to1", "remove_absolute_positions") else
                          [0] if probe_id == "epe_transport_layer0" else list(range(24 if role == "student" else 36)))
        require(probe["probe"]["layers"] == expected_scope, "historical native probe scope mismatch")
        row = {"model_role": role, "condition": payload.get("condition"), "step": payload["step"],
            "probe_id": probe_id, "scope": expected_scope, "probe_plan": probe["probe"],
            "baseline_sink": baseline, "probed_sink": edited, "delta_sink": fp["absolute_delta_sink"],
            "relative_delta_sink_fraction": None if ratio is None else ratio - 1,
            "ratio_unavailable_reason": fp["ratio_unavailable_reason"], "denominator_floor": fp["denominator_floor"],
            "valid_targets": behavior["valid_targets"], "item_count": expected_items,
            "clean_ce_nats": number(behavior["clean_ce_nats"], "clean CE"),
            "edited_ce_nats": number(behavior["edited_ce_nats"], "edited CE"),
            "delta_ce_nats": number(behavior["delta_ce_nats"], "delta CE"),
            **{metric: number(behavior[metric], metric) for metric in CHANNELS if metric != "delta_sink"},
            "source_protocol_sha256": payload.get("original_protocol_root_sha256"),
            "checkpoint_sha256": result["checkpoint_sha256"], "panel_sha256": payload["panel_sha256"],
            "source_identity": result["source_identity"], "provenance": provenance}
        require(row["self_kl_nats"] >= -2e-5 and 0 <= row["prediction_flip_fraction"] <= 1 and
                row["absolute_target_logprob_change_nats"] >= 0, "behavior range")
        rows.append(row)
    require(max(r["baseline_sink"] for r in rows) - min(r["baseline_sink"] for r in rows) < 1e-9 and
            max(r["clean_ce_nats"] for r in rows) - min(r["clean_ce_nats"] for r in rows) < 1e-8, "probe baseline drift")
    return rows


def audit_source(root: Path, audit_path: Path, expected_audit_sha256: str, *,
                 engineering_fixture: bool = False, progress: Callable[[dict], None] | None = None) -> tuple[list[dict], dict]:
    """Rehash all records and independently recompute every required aggregate."""
    root, audit_path = Path(root).resolve(strict=True), Path(audit_path).resolve(strict=True)
    require(digest_file(audit_path) == expected_audit_sha256, "independent audit file hash mismatch")
    audit, audit_hash = read_sealed(audit_path)
    require(audit["study"] == "S4" and audit["status"] == "COMPLETE_INDEPENDENTLY_VERIFIED" and
            Path(audit["result_directory"]).resolve() == root, "independent audit identity/root")
    manifest, manifest_hash = read_sealed(root / "S4_RUN_MANIFEST.json")
    runner, runner_hash = read_sealed(root / "S4_FINAL_AUDIT.json")
    require(runner["status"] == "COMPLETE" and runner["run_manifest_sha256"] == payload_digest(manifest) and
            audit["run_manifest_envelope_sha256"] == payload_digest(manifest) and
            audit["audit_receipt_file_sha256"] == runner_hash and
            manifest["D24_sha256"] == D24_SHA256 and manifest["conditions"] == list(CONDITIONS) and
            manifest["checkpoint_count"] == 35 and manifest["probe_ids"] == list(PROBES), "audit/run manifest binding")
    count = manifest["panel"]["panel_item_count"]
    require(type(count) is int and count > 0 and (engineering_fixture or count == 300), "production requires 300 items")
    require(manifest["panel"]["sequence_length"] == 128 and manifest["precision"] == "fp32" and
            manifest["denominator_floor"] > 0 and manifest["responsiveness_floor"] >= 0, "historical numerical policy")
    expected_records = 36 * 10 * count
    require(runner["verified_record_count"] == expected_records and runner["manifested_file_count"] == expected_records + 37 and
            runner["training"] is False and runner["source_runs_modified"] is False and
            audit["training"] is False and audit["source_runs_modified"] is False, "runner coverage/preservation")
    require(set(audit["S1_original_protocol_roots_preserved"]) == set(CONDITIONS) and
            all(original_protocol_sha256(manifest["source_inventory"]["sources"][c]["identity"]) ==
                audit["S1_original_protocol_roots_preserved"][c] for c in CONDITIONS), "audited source roots")
    checks = audit["independent_checks"]
    require(checks["student_batteries"] == 35 and checks["student_records"] == 35 * 10 * count and
            checks["teacher_records"] == 10 * count and checks["manifested_file_count"] == expected_records + 37 and
            checks["sha256s_entries_verified"] == expected_records + 37 and
            all(checks[k] == "PASS" for k in ("record_envelopes_and_file_hashes", "unique_record_coverage",
                "run_and_checkpoint_identities", "panel_file_and_item_identity")), "independent coverage evidence")
    index_path = root / "SHA256SUMS.txt"
    index_hash = digest_file(index_path)
    require(index_hash == runner["SHA256SUMS_file_sha256"], "checksum inventory hash mismatch")
    files = {}
    for line in index_path.read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        path = PurePosixPath(name)
        require(len(digest) == 64 and all(c in "0123456789abcdef" for c in digest) and
                not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name and
                path.as_posix() == name and name not in files, "unsafe/duplicate checksum entry")
        target = root / name
        require(target.resolve().is_relative_to(root) and not target.is_symlink(), "source path escapes root")
        files[name] = digest
    expected_summaries = {f"summaries/{c}/step-{s:05d}.json" for c in CONDITIONS for s in STEPS} | {"summaries/teacher_full300.json"}
    record_names = {name for name in files if name.startswith("records/")}
    require(len(files) == expected_records + 37 and len(record_names) == expected_records and
            set(files) - record_names == expected_summaries | {"S4_RUN_MANIFEST.json"}, "inventory coverage")
    require(files["S4_RUN_MANIFEST.json"] == manifest_hash, "manifest inventory binding")
    actual = {p.relative_to(root).as_posix() for folder in ("records", "summaries") for p in (root / folder).rglob("*") if p.is_file()}
    require(actual == record_names | expected_summaries, "unmanifested/missing record or summary")
    snapshots, summaries, rows = {}, {}, []
    for name in sorted(expected_summaries):
        payload, digest = read_sealed(root / name)
        require(digest == files[name], "summary file hash mismatch")
        if name != "summaries/teacher_full300.json":
            c, filename = PurePosixPath(name).parts[1:]
            require(payload["condition"] == c and payload["step"] == int(filename[5:-5]), "summary path/identity mismatch")
        group = (payload["model_role"], payload.get("condition"), payload["step"])
        require(group not in summaries, "duplicate summary identity")
        summaries[group] = payload
        extracted = _summary_rows(payload, name, manifest, count)
        for row in extracted:
            row.update(summary_file_sha256=digest, summary_relative_path=name)
        rows.extend(extracted)
        stat = (root / name).stat()
        snapshots[name] = (stat.st_size, stat.st_mtime_ns)
    groups = {(r["model_role"], r["condition"], r["step"], r["probe_id"]): [] for r in rows}
    item_sets = {key: set() for key in groups}
    for index, name in enumerate(sorted(record_names), 1):
        path = root / name
        record, digest = read_sealed(path)
        require(digest == files[name] and record["status"] == "complete" and record["value"] is not None, "record hash/status")
        key, value = record["key"], record["value"]
        require(Path(name).stem == payload_digest(key) and key["study"] == "S4", "record key/filename")
        group = (key["model_role"], (key.get("source_identity") or {}).get("condition"), key.get("step"), key["probe_id"])
        require(group in groups, "unknown record group")
        summary = summaries[group[:3]]["result"]
        require(key["probe"] == summary["probes"][group[3]]["probe"] and key["provenance"] == summary["provenance"] and
                key["checkpoint_sha256"] == summary["checkpoint_sha256"] and key["panel_sha256"] == summary["panel_sha256"] and
                key["probe_version"] == summary["probe_version"] and key["denominator_floor"] == manifest["denominator_floor"] and
                key["responsiveness_floor"] == manifest["responsiveness_floor"], "record/summary identity mismatch")
        if group[0] == "student":
            require(key["source_identity"] == summary["source_identity"] and key["followup_policy"] == summary["followup_policy"], "record source identity mismatch")
        item = key["item_id"]
        require(isinstance(item, str) and item and item not in item_sets[group], "duplicate/invalid item")
        item_sets[group].add(item)
        require(value["attention_mask"] == [[1] * 128] and value["input_token_count"] == 128 and
                value["behavior"]["valid_targets"] == 127, "item target mask/count")
        behavior = value["behavior"]
        for field in ("clean_nll_sum_nats", "edited_nll_sum_nats", "self_kl_sum_nats", "absolute_target_logprob_change_sum_nats"):
            number(behavior[field], field)
        require(behavior["clean_nll_sum_nats"] >= 0 and behavior["edited_nll_sum_nats"] >= 0 and
                behavior["absolute_target_logprob_change_sum_nats"] >= 0 and behavior["self_kl_sum_nats"] >= -2e-5 * 127,
                "item loss/KL range")
        for field in ("flip_count", "clean_correct_count", "edited_correct_count"):
            require(type(behavior[field]) is int and 0 <= behavior[field] <= 127, "item count range")
        clean_sink = number(value["clean_structure"]["native_layer_mean"], "clean sink")
        edited_sink = number(value["probed_structure"]["native_layer_mean"], "edited sink")
        require(0 <= clean_sink <= 1 and 0 <= edited_sink <= 1, "item sink range")
        close(value["fingerprint"]["baseline_sink"], clean_sink, "item clean sink")
        close(value["fingerprint"]["probed_sink"], edited_sink, "item edited sink")
        close(value["fingerprint"]["absolute_delta_sink"], edited_sink - clean_sink, "item sink delta")
        groups[group].append((behavior, clean_sink, edited_sink))
        stat = path.stat()
        snapshots[name] = (stat.st_size, stat.st_mtime_ns)
        if progress and (index % 1000 == 0 or index == len(record_names)):
            progress({"event": "e0_source_audit", "completed_records": index, "total_records": len(record_names)})
    canonical_ids = None
    for group, values in groups.items():
        ids = item_sets[group]
        require(len(ids) == count, "item coverage")
        canonical_ids = ids if canonical_ids is None else canonical_ids
        require(ids == canonical_ids, "panel item mismatch")
        probe = summaries[group[:3]]["result"]["probes"][group[3]]
        behavior = probe["behavior"]
        n = count * 127
        for output, source in (("clean_ce_nats", "clean_nll_sum_nats"), ("edited_ce_nats", "edited_nll_sum_nats"),
                ("self_kl_nats", "self_kl_sum_nats"), ("absolute_target_logprob_change_nats", "absolute_target_logprob_change_sum_nats"),
                ("prediction_flip_fraction", "flip_count"), ("clean_accuracy_fraction", "clean_correct_count"), ("edited_accuracy_fraction", "edited_correct_count")):
            close(behavior[output], math.fsum(v[0][source] for v in values) / n, f"aggregate {group}/{output}")
        close(probe["fingerprint"]["baseline_sink"], math.fsum(v[1] for v in values) / count, "aggregate clean sink")
        close(probe["fingerprint"]["probed_sink"], math.fsum(v[2] for v in values) / count, "aggregate edited sink")
    for name, stat in snapshots.items():
        current = (root / name).stat()
        require((current.st_size, current.st_mtime_ns) == stat, "source changed during audit")
    require(digest_file(root / "S4_RUN_MANIFEST.json") == manifest_hash and digest_file(index_path) == index_hash and
            digest_file(root / "S4_FINAL_AUDIT.json") == runner_hash and digest_file(audit_path) == audit_hash,
            "source receipt changed during audit")
    rows.sort(key=lambda r: (r["model_role"], r["condition"] or "", r["step"] or 0, PROBES.index(r["probe_id"])))
    return rows, {"source_directory": str(root), "independent_audit_file_sha256": audit_hash,
        "independent_audit": audit, "runner_audit_file_sha256": runner_hash, "run_manifest_file_sha256": manifest_hash,
        "checksum_inventory_file_sha256": index_hash, "run_manifest": manifest,
        "fresh_records_verified": expected_records, "fresh_summaries_verified": 36,
        "source_files_unchanged": True, "engineering_only": engineering_fixture,
        "training": False, "model_loaded": False, "new_inference": False}


def csv_bytes(rows: list[dict], fields: list[str]) -> bytes:
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        writer.writerow({k: json.dumps(v, sort_keys=True, separators=(",", ":")) if isinstance(v, (dict, list)) else v for k, v in row.items()})
    return stream.getvalue().encode("utf-8")


def write_new(path: Path, data: bytes) -> None:
    with path.open("xb") as stream:
        stream.write(data)
        stream.flush()
        os.fsync(stream.fileno())


def validate_bundle(root: Path, receipt: dict, *, completion_present: bool) -> dict:
    root = Path(root)
    require(receipt["version"] == VERSION and receipt["status"] == "COMPLETE" and
            receipt["training"] is False and receipt["new_inference"] is False, "invalid E0 completion")
    expected = receipt["files"]
    require(set(expected) == OUTPUT_FILES and set(p.name for p in root.iterdir()) ==
            set(expected) | ({"COMPLETE.json"} if completion_present else set()), "output file coverage")
    for name, digest in expected.items():
        require(Path(name).name == name and not (Path(root) / name).is_symlink() and
                digest_file(Path(root) / name) == digest, "output artifact hash mismatch")
    data, _ = read_sealed(root / "E0_ANALYSIS.json")
    analysis = analyze(data["rows"], data["recipe"])
    require(analysis == data["analysis"], "output analysis disagrees with recorded rows/recipe")
    require((root / "E0_S4_ROUTE_TRAJECTORIES.csv").read_bytes() == csv_bytes(data["rows"], TRAJECTORY_FIELDS) and
            (root / "E0_TEACHER_STUDENT_FINGERPRINTS.csv").read_bytes() ==
            csv_bytes(analysis["comparisons"], list(analysis["comparisons"][0])), "CSV semantic mismatch")
    provenance, _ = read_sealed(root / "E0_PROVENANCE.json")
    require(provenance["engineering_only"] == receipt["engineering_only"] and
            receipt["scientific_record_reanalysis"] is (not receipt["engineering_only"]) and
            provenance["model_loaded"] is False and provenance["training"] is False and
            provenance["new_inference"] is False and provenance["source_files_unchanged"] is True and
            receipt["trajectory_rows"] == len(data["rows"]) and receipt["component_comparison_rows"] == len(analysis["comparisons"]) and
            receipt["composite_status"] == analysis["composite_status"], "completion claims mismatch")
    return receipt


def verify_bundle(root: Path) -> dict:
    receipt, _ = read_sealed(Path(root) / "COMPLETE.json")
    return validate_bundle(root, receipt, completion_present=True)
