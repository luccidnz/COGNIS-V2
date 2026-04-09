import shutil
from pathlib import Path

import numpy as np

from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.engine import Engine
from cognis.serialization.artifacts import serialize_analysis, serialize_report, write_render_artifacts
from cognis.serialization.recipe import serialize_recipe


def _make_config(backend: str = "AUTO") -> MasteringConfig:
    return MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=1,
        bass_preservation=0.9,
        stereo_width=1.0,
        dynamics_preservation=0.5,
        brightness=0.1,
        fir_backend=backend,
    )


def _make_audio() -> np.ndarray:
    t = np.linspace(0, 0.5, 24000, endpoint=False)
    left = 0.45 * np.sin(2 * np.pi * 90 * t) + 0.2 * np.sin(2 * np.pi * 2200 * t)
    right = 0.4 * np.sin(2 * np.pi * 140 * t + 0.2) + 0.15 * np.sin(2 * np.pi * 7000 * t)
    return np.vstack((left, right))


def test_engine_render_emits_first_class_artifacts():
    engine = Engine()
    result = engine.render(_make_audio(), 48000, _make_config())

    assert result.audio.shape == _make_audio().shape
    assert result.recipe["schema_version"] == "recipe_v2"
    assert result.input_analysis.schema_version == "analysis_schema_v2"
    assert result.output_analysis.schema_version == "analysis_schema_v2"
    assert result.reference_analysis is None
    assert result.targets.reference_targeting is None
    assert result.report.schema_version == "report_schema_v2"
    assert result.report.findings


def test_engine_process_remains_compatible():
    engine = Engine()
    mastered, report, recipe = engine.process(_make_audio(), 48000, _make_config())

    assert mastered.shape == _make_audio().shape
    assert report.schema_version == "report_schema_v2"
    assert recipe["schema_version"] == "recipe_v2"


def test_engine_render_is_deterministic_for_same_input():
    engine = Engine()
    audio = _make_audio()
    config = _make_config()

    first = engine.render(audio, 48000, config)
    second = engine.render(audio, 48000, config)

    assert np.allclose(first.audio, second.audio, atol=1e-10)
    assert serialize_recipe(first.recipe) == serialize_recipe(second.recipe)
    assert serialize_analysis(first.output_analysis) == serialize_analysis(second.output_analysis)
    assert serialize_report(first.report) == serialize_report(second.report)


def test_engine_process_with_different_backends():
    engine = Engine()
    audio = _make_audio()

    def run_with_backend(backend: str):
        return engine.render(audio, 48000, _make_config(backend)).audio

    out_auto = run_with_backend("AUTO")
    out_fft = run_with_backend("FFT")
    out_part = run_with_backend("PARTITIONED")

    assert np.allclose(out_auto, out_part, atol=1e-10)
    assert np.allclose(out_fft, out_part, atol=1e-10)


def test_artifact_writer_writes_expected_files():
    engine = Engine()
    result = engine.render(_make_audio(), 48000, _make_config())
    artifact_root = Path(".tmp") / "test-artifact-writer"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    written = write_render_artifacts(
        result,
        str(artifact_root / "master.wav"),
        write_recipe=True,
        write_analysis=True,
        write_report=True,
        write_markdown_report=True,
    )

    assert set(written) == {"recipe", "analysis_input", "analysis_output", "report", "report_markdown"}
    for path in written.values():
        assert Path(path).exists()


def test_engine_render_with_reference_emits_reference_artifacts():
    engine = Engine()
    audio = _make_audio()
    reference_audio = audio * 0.9
    config = _make_config()
    config.mode = MasteringMode.REFERENCE_MATCH
    config.reference_path = "reference.wav"

    result = engine.render(audio, 48000, config, reference_audio=reference_audio, reference_sr=48000)

    assert result.reference_analysis is not None
    assert result.reference_analysis.identity.role == "reference"
    assert result.targets.reference_targeting is not None
    assert result.report.reference_status in {"constrained", "partial", "matched", "deviated"}
    assert result.report.reference_assessment is not None
    assert result.report.reference_assessment.reference_analysis_schema_version == "analysis_schema_v2"

    artifact_root = Path(".tmp") / "test-reference-artifact-writer"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    written = write_render_artifacts(
        result,
        str(artifact_root / "master.wav"),
        write_recipe=True,
        write_analysis=True,
        reference_analysis=result.reference_analysis,
        write_report=True,
        write_markdown_report=True,
    )

    assert "analysis_reference" in written
    assert Path(written["analysis_reference"]).exists()
