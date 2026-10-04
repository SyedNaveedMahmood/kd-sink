"""Read-only D24 checkpoint inventory check. Never loads model/state tensors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sinklab.checkpoint import verify_checkpoint
from sinklab.followup_policy import (D24_PATH, D24_SHA256, CONDITIONS, FOLLOWUP_STEPS,
    checkpoint_inventory_coverage, validate_followup_amendment)
from sinklab.provenance import _no_duplicate_keys


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--study", choices=("S4", "S6"), required=True)
    parser.add_argument("--checkpoint", type=Path, action="append", default=[],
                        help="Explicit original checkpoint directory; repeat as needed")
    args = parser.parse_args(argv)
    repo = Path(__file__).resolve().parents[1]
    validate_followup_amendment(read_json(repo / D24_PATH))
    plan = read_json(repo / "configs/s1_checkpoint_followups_seed0.json")
    if (plan["amendment_sha256"] != D24_SHA256 or plan["source_study"] != "S1"
            or plan["training_seeds"] != [0] or plan["conditions"] != list(CONDITIONS)
            or plan["checkpoints"] != {k: list(v) for k, v in FOLLOWUP_STEPS.items()}
            or plan["execution_authorized"] is not False):
        raise ValueError("follow-up requirements disagree with D24")
    manifests = []
    paths = []
    excluded = []
    for path in args.checkpoint:
        header = read_json(path / "manifest.json")
        identity = header["identity"]
        if (identity.get("study") != "S1" or identity.get("seed") != 0
                or header["step"] not in FOLLOWUP_STEPS[args.study]):
            excluded.append(str(path))
            continue
        manifest = verify_checkpoint(path)
        if manifest != header:
            raise ValueError("checkpoint manifest changed during verification")
        manifests.append(manifest)
        paths.append(str(path.resolve()))
    report = checkpoint_inventory_coverage(manifests, study=args.study)
    report.update(verified_checkpoint_paths=paths, excluded_from_requirements=excluded,
                  payload_verification="SHA-256 only; no tensor deserialization",
                  remaining_gates="explicit run/device selection, panels, runtime and execution approval")
    print(json.dumps(report, indent=2))
    return 0 if report["status"] == "complete_inventory" else 1


if __name__ == "__main__":
    raise SystemExit(main())
