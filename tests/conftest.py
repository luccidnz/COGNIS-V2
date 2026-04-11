from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import pytest


_TEST_TEMP_ROOT = Path(os.environ.get("COGNIS_TEST_TEMP_ROOT", Path(__file__).resolve().parents[1] / ".tmp" / "pytest-temp-active"))
_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

for key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[key] = str(_TEST_TEMP_ROOT)

tempfile.tempdir = None


@pytest.fixture
def tmp_path(request):
    safe_name = "".join(ch if ch.isalnum() else "_" for ch in request.node.name)[:80]
    path = _TEST_TEMP_ROOT / "manual" / f"{safe_name}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=True, exist_ok=False)
    return path
