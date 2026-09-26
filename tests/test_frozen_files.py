"""Mechanical enforcement of the frozen-baseline rule (WP0).

The twelve files listed in ``BASELINE_HASHES.json`` reproduce the published E1-E5
numbers and are read-only for the life of the project (see CLAUDE.md rule 1,
``new_design_plans/01_REFACTOR_SPEC_existing_code.md`` section 1). This test recomputes
each file's sha256 and asserts it matches the recorded baseline, naming exactly which
file drifted on failure.

Runs two ways:
  * under pytest:  ``pytest tests/test_frozen_files.py``
  * standalone:    ``python tests/test_frozen_files.py``  (no third-party deps required)

Stdlib only, so it is runnable even on a workstation without the model runtime.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE_PATH = REPO_ROOT / "BASELINE_HASHES.json"


def _load_baseline() -> dict:
    with BASELINE_PATH.open(encoding="utf-8") as fh:
        return json.load(fh)


def _normalise_eol(data: bytes) -> bytes:
    """Canonicalise line endings to LF before hashing.

    The recorded hashes must identify a frozen file's *content*, not the line endings the
    local git checkout happened to produce. ``core.autocrlf=true`` on Windows writes CRLF;
    a Linux or ``autocrlf=false`` checkout of the same commit writes LF. Hashing raw bytes
    makes the two disagree, so the test would report all twelve files as DRIFTED on a
    machine that has changed nothing -- and that reading is indistinguishable from the one
    case this test exists to catch, someone editing a frozen file.

    The trade-off is deliberate and stated: an EOL-only change to a frozen file is no
    longer detected. That is correct, because git itself creates those changes, and no
    E1-E5 number depends on a line terminator. Every other byte is still covered.
    """
    return data.replace(b"\r\n", b"\n")


def _sha256(path: Path) -> str:
    return hashlib.sha256(_normalise_eol(path.read_bytes())).hexdigest()


def check_frozen_files() -> list[str]:
    """Return a list of human-readable drift messages (empty if all match)."""
    baseline = _load_baseline()
    frozen = baseline["frozen_files"]
    problems: list[str] = []
    for rel_path, expected in frozen.items():
        abs_path = REPO_ROOT / rel_path
        if not abs_path.is_file():
            problems.append(f"MISSING: {rel_path} (expected sha256 {expected})")
            continue
        actual = _sha256(abs_path)
        if actual != expected:
            problems.append(
                f"DRIFTED: {rel_path}\n"
                f"           expected {expected}\n"
                f"           actual   {actual}"
            )
    return problems


def test_frozen_files_unchanged() -> None:
    """Every frozen file's sha256 still matches BASELINE_HASHES.json."""
    assert BASELINE_PATH.is_file(), f"baseline manifest not found: {BASELINE_PATH}"
    problems = check_frozen_files()
    assert not problems, (
        "Frozen file(s) changed since frozen-e1-e5 -- this is forbidden "
        "(CLAUDE.md rule 1). Offending file(s):\n" + "\n".join(problems)
    )


def test_hashing_is_line_ending_independent() -> None:
    """The same content in CRLF and LF must hash identically.

    This is the property that makes ``BASELINE_HASHES.json`` portable across the Windows
    workstation and any Linux compute box. Asserted on synthetic bytes rather than on a
    frozen file, so it cannot be satisfied by accident of what is currently on disk.
    """
    lf = b"import torch\ndef f():\n    return 1\n"
    crlf = b"import torch\r\ndef f():\r\n    return 1\r\n"
    assert lf != crlf, "the fixture must actually differ in raw bytes"
    assert (hashlib.sha256(_normalise_eol(lf)).hexdigest()
            == hashlib.sha256(_normalise_eol(crlf)).hexdigest())
    # ... and a real content change must still be caught.
    changed = lf.replace(b"return 1", b"return 2")
    assert (hashlib.sha256(_normalise_eol(changed)).hexdigest()
            != hashlib.sha256(_normalise_eol(lf)).hexdigest())


def test_recorded_hashes_are_the_normalised_ones() -> None:
    """Guard against a future regeneration that records raw CRLF hashes again.

    ``BASELINE_HASHES.json`` records which convention it used. If that field says ``lf``
    but a hash was captured raw on Windows, the manifest is internally inconsistent and
    every non-Windows checkout fails -- so the claim is asserted rather than trusted.
    """
    baseline = _load_baseline()
    assert baseline.get("hash_convention", {}).get("line_endings") == "lf", (
        "BASELINE_HASHES.json must declare hash_convention.line_endings == 'lf'; "
        "see _normalise_eol for why the hashes are EOL-canonical.")
    for rel_path, expected in baseline["frozen_files"].items():
        raw = (REPO_ROOT / rel_path).read_bytes()
        assert hashlib.sha256(_normalise_eol(raw)).hexdigest() == expected, rel_path


if __name__ == "__main__":
    if not BASELINE_PATH.is_file():
        raise SystemExit(f"baseline manifest not found: {BASELINE_PATH}")
    drift = check_frozen_files()
    if drift:
        print("FAIL: frozen file(s) changed:\n" + "\n".join(drift))
        raise SystemExit(1)
    n = len(_load_baseline()["frozen_files"])
    print(f"OK: all {n} frozen files match BASELINE_HASHES.json")
