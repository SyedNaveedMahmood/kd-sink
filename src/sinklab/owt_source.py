"""Explicit pinned OpenWebText source extraction for S1 preparation."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

from .provenance import seal_payload


DATASET_ID = "Skylion007/openwebtext"
DATASET_REVISION = "79d93d786212f7344586290adb811d4ae6a1762c"
DOCUMENT_LIMIT = 416_000


def extract_source(directory: Path) -> dict:
    """Write exactly the frozen source prefix; refuse mutable/partial output."""
    from datasets import load_dataset
    from huggingface_hub import HfApi

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / "openwebtext-train-first416000.jsonl"
    partial = directory / "openwebtext-train-first416000.jsonl.partial"
    record = directory / "openwebtext-source.json"
    if any(path.exists() for path in (target, partial, record)):
        raise FileExistsError("source extraction refuses an existing final or partial artifact")
    info = HfApi().dataset_info(DATASET_ID, revision=DATASET_REVISION)
    if info.sha != DATASET_REVISION:
        raise ValueError("OpenWebText revision did not resolve to the pinned commit")
    licenses = info.cardData.get("license") if info.cardData else None
    if licenses != ["cc0-1.0"]:
        raise ValueError("OpenWebText card license differs from the reviewed source")
    stream = load_dataset(DATASET_ID, "plain_text", split="train", streaming=True,
                          revision=DATASET_REVISION)
    digest = hashlib.sha256()
    count = 0
    with partial.open("wb") as sink:
        for row in itertools.islice(stream, DOCUMENT_LIMIT):
            if not isinstance(row, dict) or not isinstance(row.get("text"), str):
                raise ValueError(f"source row {count} has no text string")
            raw = (json.dumps({"text": row["text"]}, ensure_ascii=False,
                              separators=(",", ":")) + "\n").encode("utf-8")
            sink.write(raw)
            digest.update(raw)
            count += 1
            if count % 10_000 == 0:
                print(json.dumps({"event": "owt_source_progress", "rows": count}), flush=True)
        sink.flush()
    if count != DOCUMENT_LIMIT:
        raise ValueError(f"source ended after {count} rows; {DOCUMENT_LIMIT} required")
    partial.replace(target)
    result = seal_payload({"kind": "pinned-owt-source-prefix-v1", "dataset_id": DATASET_ID,
        "revision": DATASET_REVISION, "configuration": "plain_text", "split": "train",
        "row_count": count, "windows": {"training": [0, 400_000],
            "evaluation": [400_000, 408_000], "calibration": [408_000, 416_000]},
        "license_card": licenses, "source_jsonl_sha256": digest.hexdigest(),
        "source_jsonl_bytes": target.stat().st_size})
    record.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return {"source": str(target), "record": str(record), "sha256": result["sha256"]}


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch only the pinned S1 OWT source prefix")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(extract_source(args.out_dir), sort_keys=True))


if __name__ == "__main__":
    main()
