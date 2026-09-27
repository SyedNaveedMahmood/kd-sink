"""WP1: corpus manifest determinism (06 test_corpus_determinism.py).

Each provider must yield an identical ``manifest_sha256`` twice in-process and once in a
fresh subprocess — the property that lets two runs prove they saw byte-identical inputs.
Uses the fully offline ``synthetic_corpus`` (no dataset download) so the gate runs on this
smoke-only box. The subprocess build rebuilds the tokenizer and corpus from scratch, so a
match rules out any in-process caching or RNG-state leakage.
"""

from __future__ import annotations

import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import corpus_providers as cp  # noqa: E402
from nnsight_smoke_utils import _tiny_tokenizer  # noqa: E402

VOCAB = 64
KIND = "random_zipf"
N = 8
SEED = 5
CUT = 16

_CHILD = """
import sys, tempfile
from pathlib import Path
sys.path.insert(0, r"{repo}")
sys.path.insert(0, r"{repo}/common")
sys.path.insert(0, r"{tests}")
import corpus_providers as cp
from nnsight_smoke_utils import _tiny_tokenizer
with tempfile.TemporaryDirectory() as t:
    tok = _tiny_tokenizer(Path(t)/"tok", {vocab})
    c = cp.synthetic_corpus(tok, "{kind}", {n}, seed={seed}, cut_length={cut})
    print(c.manifest_sha256)
"""


def _in_process_hash() -> str:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        tok = _tiny_tokenizer(Path(t) / "tok", VOCAB)
        return cp.synthetic_corpus(tok, KIND, N, seed=SEED, cut_length=CUT).manifest_sha256


def test_manifest_deterministic_in_process() -> None:
    assert _in_process_hash() == _in_process_hash()


def test_manifest_deterministic_across_subprocess() -> None:
    ref = _in_process_hash()
    child = _CHILD.format(repo=str(REPO).replace("\\", "/"),
                          tests=str(Path(__file__).resolve().parent).replace("\\", "/"),
                          vocab=VOCAB, kind=KIND, n=N, seed=SEED, cut=CUT)
    proc = subprocess.run([sys.executable, "-c", child], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    assert proc.stdout.strip().splitlines()[-1] == ref, (proc.stdout, ref)


def test_different_seed_changes_hash() -> None:
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as t:
        tok = _tiny_tokenizer(Path(t) / "tok", VOCAB)
        a = cp.synthetic_corpus(tok, KIND, N, seed=SEED, cut_length=CUT).manifest_sha256
        b = cp.synthetic_corpus(tok, KIND, N, seed=SEED + 1, cut_length=CUT).manifest_sha256
    assert a != b
