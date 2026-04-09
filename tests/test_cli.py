from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

from cognis.cli import main as cli_main


def _write_audio(path: Path, audio: np.ndarray, sr: int = 48000) -> None:
    sf.write(path, audio.T, sr)


def _demo_audio() -> np.ndarray:
    t = np.linspace(0, 0.25, 12000, endpoint=False)
    left = 0.35 * np.sin(2 * np.pi * 110 * t) + 0.12 * np.sin(2 * np.pi * 2200 * t)
    right = 0.32 * np.sin(2 * np.pi * 140 * t + 0.2) + 0.10 * np.sin(2 * np.pi * 6100 * t)
    return np.vstack((left, right))


def test_cli_reference_run_writes_reference_artifacts(monkeypatch):
    tmp_path = Path(".tmp") / "test-cli-reference-run"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)

    input_path = tmp_path / "input.wav"
    reference_path = tmp_path / "reference.wav"
    output_path = tmp_path / "master.wav"
    artifacts_dir = tmp_path / "artifacts"

    input_audio = _demo_audio()
    reference_audio = input_audio * 0.9
    _write_audio(input_path, input_audio)
    _write_audio(reference_path, reference_audio)

    argv = [
        "cognis",
        str(input_path),
        str(output_path),
        "--reference",
        str(reference_path),
        "--mode",
        "REFERENCE_MATCH",
        "--artifacts-dir",
        str(artifacts_dir),
        "--write-markdown-report",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli_main()

    expected_files = {
        artifacts_dir / "master.recipe.json",
        artifacts_dir / "master.analysis.input.json",
        artifacts_dir / "master.analysis.output.json",
        artifacts_dir / "master.analysis.reference.json",
        artifacts_dir / "master.report.json",
        artifacts_dir / "master.report.md",
    }

    for path in expected_files:
        assert path.exists()

    report_payload = json.loads((artifacts_dir / "master.report.json").read_text(encoding="utf-8"))
    assert report_payload["schema_version"] == "report_schema_v2"
    assert report_payload["reference_assessment"]["outcome"] in {"constrained", "partial", "matched", "deviated"}

    reference_payload = json.loads((artifacts_dir / "master.analysis.reference.json").read_text(encoding="utf-8"))
    assert reference_payload["identity"]["role"] == "reference"
    assert reference_payload["identity"]["source_path"] == str(reference_path)

    markdown = (artifacts_dir / "master.report.md").read_text(encoding="utf-8")
    assert "## Reference" in markdown
    assert "## QC Findings" in markdown
