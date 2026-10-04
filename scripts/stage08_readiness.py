"""Read-only Stage08 artifact checks. No model loading or scientific execution."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

from sinklab.provenance import _no_duplicate_keys, payload_digest, verify_envelope


EXCEPTION_PATH = "protocols/s1_c3_seed0_transferred_log_exception.json"
EXCEPTION_SHA256 = "f021ac0441f775d45d19a8109d47c0f45a0b0bbfc025bbd2df35be688e291cd2"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def accepted_c3_log_exception(*, identity: dict, log_sha256: str, document: dict) -> dict:
    """Only exact approved log bytes AND exact original checkpoint identity qualify."""
    payload, digest = verify_envelope(document)
    if digest != EXCEPTION_SHA256:
        raise ValueError("unapproved historical log exception")
    if (identity != payload["identity"] or payload_digest(identity) != payload["identity_sha256"]
            or log_sha256 != payload["train_jsonl_sha256"]):
        raise ValueError("outside exact C3 seed0 transferred-log exception")
    return payload


def verify_training_log(run: Path, identity: dict, *, exception_document: dict) -> dict:
    raw = (run / "train.jsonl").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    rows = [json.loads(line, object_pairs_hook=_no_duplicate_keys)
            for line in raw.splitlines() if line.strip()]
    if not rows or any(row.get("event") not in {"start", "update"} for row in rows):
        raise ValueError("missing or unknown training events")
    starts = [row for row in rows if row["event"] == "start"]
    updates = [row for row in rows if row["event"] == "update"]
    steps = [row["step"] for row in updates]
    exception = None
    if steps != list(range(1, 10001)):
        exception = accepted_c3_log_exception(identity=identity, log_sha256=digest,
                                              document=exception_document)
        if (steps != list(range(2501, 10001)) or len(starts) != 1 or
                starts[0].get("resumed_step") != 2500 or len(raw) != exception["train_jsonl_bytes"]):
            raise ValueError("C3 discrepancy beyond approved missing prefix")
        if sha256(run / "checkpoints/final-010000/manifest.json") != exception["final_manifest_sha256"]:
            raise ValueError("C3 final manifest differs from exception binding")
        if sha256(run / "TRANSFER_MANIFEST.json") != exception["transfer_manifest_sha256"]:
            raise ValueError("C3 transfer provenance differs from exception binding")
    if not starts or rows[0]["event"] != "start":
        raise ValueError("training start provenance missing")
    last_step = 2500 if exception else 0
    for row in rows:
        if row["event"] == "start":
            for field in ("study", "condition", "seed", "run_id", "protocol_hash", "gpu_uuid",
                          "microbatch", "precision"):
                if row.get(field) != identity.get(field):
                    raise ValueError(f"training start identity differs: {field}")
            if row.get("resumed_step") != last_step:
                raise ValueError("unexplained training resume boundary")
        else:
            last_step = row["step"]
            if type(last_step) is not int or row.get("input_tokens") != last_step * 8192 or row.get("target_tokens") != last_step * 8128:
                raise ValueError("training token counters differ")
            active = {"loss", "ce", "lr", "grad_norm"}
            condition = identity["condition"]
            if condition != "C0":
                active.add("kd")
            if condition in {"C2", "C3", "C5", "C6"}:
                active.add("attention")
            if condition == "C4":
                active.add("relation")
            for field in active:
                if type(row.get(field)) not in (int, float) or not math.isfinite(row[field]):
                    raise ValueError(f"nonfinite/missing training metric: {field}")
            if any(row.get(field) is not None for field in {"kd", "attention", "relation"} - active):
                raise ValueError("inactive objective is not null")
    return {"path": str(run / "train.jsonl"), "sha256": digest,
            "update_count": len(updates), "first_update": steps[0], "last_update": steps[-1],
            "start_event_count": len(starts), "final_input_tokens": updates[-1]["input_tokens"],
            "final_shifted_targets": updates[-1]["target_tokens"],
            "status": "accepted_historical_prefix_gap" if exception else "complete",
            "exception_sha256": EXCEPTION_SHA256 if exception else None}


def verify_required_checkpoint(path: Path, *, identity: dict, step: int) -> dict:
    """Independently verify every byte and marker; no log exception enters this path."""
    from sinklab.checkpoint import verify_checkpoint
    before = (path / "manifest.json").read_bytes()
    # Strict JSON precheck supplements the existing payload/COMPLETE verifier.
    header = json.loads(before, object_pairs_hook=_no_duplicate_keys)
    if (header.get("schema_version") != 1 or type(header.get("step")) is not int
            or header["step"] != step or header.get("identity") != identity):
        raise ValueError("required checkpoint step/schema/identity differs")
    manifest = verify_checkpoint(path, identity=identity)
    if (path / "manifest.json").read_bytes() != before:
        raise ValueError("checkpoint manifest changed during verification")
    return {"path": str(path), "step": step, "kind": manifest["kind"],
            "manifest_sha256": hashlib.sha256(before).hexdigest(),
            "identity_sha256": manifest["identity_sha256"], "identity": manifest["identity"],
            "payload_sha256": manifest["files"], "status": "verified"}


def verify_evaluation_records(run: Path, identity: dict) -> dict:
    """Validate sealed existing observations and coverage without any model inference."""
    from sinklab.evaluate import _key, RETAINED_FULL
    from sinklab.metrics import aggregate_behavior
    expected = {(s, "owt_dense64", role): 64 for s in range(0, 10001, 100)
                for role in ("student", "teacher")}
    expected.update({(s, "owt_full300", role): 300 for s in RETAINED_FULL
                     for role in ("student", "teacher")})
    expected.update({(s, "owt_lm2000", "student"): 2000 for s in (0, 10000)})
    root = run / "evaluation"
    files = {p.name for p in root.iterdir() if p.is_file()}
    seen_files, seen_groups, panels = set(), set(), {}
    for path in sorted(root.glob("aggregate-*.json")):
        aggregate, _ = verify_envelope(read_json(path))
        key, operations = aggregate["key"], aggregate["operations"]
        if path.name != f"aggregate-{payload_digest(key)}.json":
            raise ValueError(f"aggregate filename/key mismatch: {path}")
        source = key["run_identity"]
        if (key["run_id"] != identity["run_id"] or key["precision"] != identity["precision"]
                or any(source.get(k) != v for k, v in {
                    "study": "S1", "condition": identity["condition"], "seed": identity["seed"],
                    "protocol_sha256": identity["protocol_hash"],
                    "corpus_sha256": identity["data_hash"]}.items())):
            raise ValueError(f"evaluation/source identity mismatch: {path}")
        if key["model_role"] == "student" and source["model_sha256"] != identity["model_hash"]:
            raise ValueError(f"student model identity mismatch: {path}")
        group = (key["step"], key["panel"], key["model_role"])
        if group not in expected or group in seen_groups:
            raise ValueError(f"unexpected/duplicate aggregate: {path}")
        wanted_ops = {"clean"} if key["panel"] == "owt_lm2000" else {"clean", "delete", "relocate"}
        if set(operations) != wanted_ops or set(key["operations"]) != wanted_ops:
            raise ValueError(f"evaluation operations mismatch: {path}")
        ids = operations["clean"]["complete_item_ids"]
        if len(ids) != expected[group] or len(set(ids)) != len(ids):
            raise ValueError(f"panel item count/duplicates differ: {path}")
        panel_identity = {"panel_hash": key["panel_hash"], "item_ids": ids}
        prior = panels.setdefault(key["panel"], panel_identity)
        if prior != panel_identity:
            raise ValueError(f"panel membership changed across roles/steps: {path}")
        for operation, summary in operations.items():
            if (summary["status"] != "complete" or summary["failed_item_ids"]
                    or summary["missing_item_ids"] or summary["complete_item_ids"] != ids):
                raise ValueError(f"incomplete aggregate: {path}")
            values = []
            for item_id in ids:
                item_key = _key(run_id=key["run_id"], step=key["step"], panel=key["panel"],
                    panel_hash=key["panel_hash"], checkpoint_hash=key["checkpoint_hash"],
                    item_id=item_id, scope=key["scope"], operation=operation,
                    strength=0. if operation == "clean" else 1., precision=key["precision"],
                    model_role=key["model_role"], evaluation_mode=key["evaluation_mode"],
                    run_identity=source, denominator_floor=key["fingerprint_denominator_floor"])
                item_path = root / f"{payload_digest(item_key)}.json"
                item, _ = verify_envelope(read_json(item_path))
                if (item["key"] != item_key or item["status"] != "complete"
                        or item.get("error") is not None or not item.get("value")):
                    raise ValueError(f"invalid/failed scientific item: {item_path}")
                if item_path.name in seen_files:
                    raise ValueError(f"duplicate item reference: {item_path}")
                seen_files.add(item_path.name)
                values.append(item["value"])
            if key["panel"] != "owt_lm2000":
                if aggregate_behavior([v["behavior"] for v in values]) != summary["metrics"]:
                    raise ValueError(f"aggregate metrics disagree with items: {path}")
            else:
                endpoints = [v["endpoint_nll_only"] for v in values]
                count = sum(v["valid_targets"] for v in endpoints)
                ce = sum(v["clean_nll_sum_nats"] for v in endpoints) / count
                if (summary["metrics"]["valid_targets"] != count
                        or summary["metrics"]["clean_ce_nats"] != ce):
                    raise ValueError(f"endpoint aggregate differs from items: {path}")
        seen_files.add(path.name)
        seen_groups.add(group)
    if seen_groups != set(expected) or seen_files != files:
        raise ValueError("missing evaluation coverage or unexpected/unreferenced files")
    return {"path": str(root), "status": "verified", "aggregate_count": len(seen_groups),
            "item_record_count": len(seen_files) - len(seen_groups),
            "file_count": len(files), "panels": {k: {"panel_hash": v["panel_hash"],
                "item_ids_sha256": payload_digest(v["item_ids"]), "item_count": len(v["item_ids"])}
                for k, v in panels.items()}}
