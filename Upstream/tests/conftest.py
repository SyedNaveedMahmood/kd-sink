"""Pytest session setup for the sink-inheritance smoke suite.

Pins the Hugging Face cache to an on-repo (X-drive) location so no model or dataset
bytes land on the C drive, and silences the cosmetic Windows symlink warning that HF
emits when Developer Mode is off. Both are only set when the environment does not
already define them, so an explicit override (e.g. pointing at a shared cache on the
compute PC) still wins.

This runs at collection time, before any test imports ``transformers``/``datasets``,
so the cache location is in effect for the whole session.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

# X-drive HF cache: keeps downloads off C:. Override by exporting HF_HOME yourself.
_DEFAULT_HF_HOME = REPO / ".hf_cache"

os.environ.setdefault("HF_HOME", str(_DEFAULT_HF_HOME))
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
