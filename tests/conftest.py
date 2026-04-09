from __future__ import annotations

import os
import tempfile
from pathlib import Path


_TEST_TEMP_ROOT = Path(__file__).resolve().parents[1] / ".tmp" / "pytest-temp"
_TEST_TEMP_ROOT.mkdir(parents=True, exist_ok=True)

for key in ("TMP", "TEMP", "TMPDIR"):
    os.environ[key] = str(_TEST_TEMP_ROOT)

tempfile.tempdir = None
