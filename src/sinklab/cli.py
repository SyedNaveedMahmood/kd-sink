"""Stage 00 validation entry point. It never starts an experiment."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .config import ConfigError, resolve_config
from .provenance import LockError, _no_duplicate_keys


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="sinklab")
    command = parser.add_subparsers(dest="command", required=True)
    validate = command.add_parser("validate", help="validate one explicit run; no training")
    validate.add_argument("--config", type=Path, required=True)
    validate.add_argument("--seed", type=int, required=True)
    validate.add_argument("--protocol-lock", type=Path)
    validate.add_argument("--production", action="store_true")
    args = parser.parse_args(argv)
    try:
        raw = json.loads(args.config.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        lock = None
        if args.protocol_lock is not None:
            lock = json.loads(args.protocol_lock.read_text(encoding="utf-8"), object_pairs_hook=_no_duplicate_keys)
        spec = resolve_config(raw, seed=args.seed, protocol_lock=lock, production=args.production)
    except (OSError, UnicodeError, json.JSONDecodeError, ConfigError, LockError) as exc:
        print(f"sinklab: {exc}", file=sys.stderr)
        return 2
    print(json.dumps({
        "study": spec.study, "condition": spec.condition, "variant": spec.variant,
        "seed": spec.seed, "device_role": spec.device_role,
        "protocol_digest": spec.protocol_digest,
        "production_requested": args.production, "action": "validated_only",
    }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
