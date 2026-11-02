"""Pytest setup: put the repo root on sys.path and stub out heavy ML dependencies."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
for path in (ROOT, ROOT / "tests"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import stubs  # noqa: E402

stubs.install()
