import shutil
from types import SimpleNamespace
from pathlib import Path
import json

from cognis.analysis.features import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYZER_VERSION,
    AnalysisIdentity,
    AnalysisNotes,
    AnalysisResult,
    LoudnessSummary,
    RiskSummary,
    StereoSummary,
    TonalSummary,
)
from cognis.serialization.analysis_artifacts import (
    analysis_artifact_filename,
    analysis_artifact_path,
    build_analysis_artifact,
)
from cognis.serialization.artifacts import serialize_analysis_artifact, write_render_artifacts


FIXTURES = Path(__file__).parent / "fixtures"


def _analysis() -> AnalysisResult:
    return AnalysisResult(
        identity=AnalysisIdentity(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            analyzer_version=ANALYZER_VERSION,
            sample_rate_hz=48000,
            channels=2,
            samples=240000,
            duration_s=5.0,
        ),
        loudness=LoudnessSummary(
            integrated_lufs=-14.2,
            short_term_max_lufs=-12.1,
            short_term_mean_lufs=-13.7,
            short_term_min_lufs=-15.7,
            short_term_range_lu=2.0,
            momentary_max_lufs=-9.0,
            momentary_mean_lufs=-13.2,
            momentary_min_lufs=-16.2,
            loudness_range_lu=3.0,
            sample_peak_dbfs=-1.6,
            true_peak_dbfs=-1.3,
            peak_to_loudness_ratio_lu=12.9,
            crest_factor_db=7.0,
        ),
        tonal=TonalSummary(
            spectral_tilt_db_per_decade=0.8,
            low_mid_balance_db=1.2,
            high_mid_balance_db=-1.5,
            sub_energy_ratio=0.12,
            bass_energy_ratio=0.19,
            low_energy_ratio=0.31,
            high_energy_ratio=0.08,
            low_band_centroid_hz=82.0,
        ),
        stereo=StereoSummary(
            phase_correlation=0.88,
            low_band_width=0.05,
            mid_band_width=0.28,
            high_band_width=0.42,
            side_energy_ratio=0.12,
            mono_null_ratio_db=-10.5,
            left_right_balance_db=0.2,
        ),
        risks=RiskSummary(
            limiter_stress_estimate=0.4,
            codec_risk_estimate=0.3,
            clipping_risk_estimate=0.1,
            delivery_safety_estimate=0.86,
            hot_sample_ratio=0.0,
            near_full_scale_ratio=0.0,
            clipped_sample_count=0,
            clipped_sample_ratio=0.0,
            intersample_peak_excess_db=0.3,
            codec_headroom_margin_db=0.3,
        ),
        notes=AnalysisNotes(
            momentary_available=True,
            loudness_range_available=True,
            codec_risk_is_proxy=True,
            limiter_stress_is_proxy=True,
        ),
    )


def test_analysis_artifact_naming_is_stable() -> None:
    assert analysis_artifact_filename("master", "reference") == "master.analysis.reference.json"
    assert analysis_artifact_path("artifacts", "master", "input").as_posix().endswith("master.analysis.input.json")


def test_build_analysis_artifact_serializes_reference_metadata() -> None:
    artifact = build_analysis_artifact(
        _analysis(),
        role="reference",
        artifact_stem="master",
        source_label="reference",
        source_path="references/reference.wav",
    )

    assert json.loads(serialize_analysis_artifact(artifact)) == json.loads(
        (FIXTURES / "analysis_reference_artifact.json").read_text(encoding="utf-8")
    )


def test_write_render_artifacts_emits_reference_sibling_when_available() -> None:
    analysis = _analysis()
    artifact_root = Path(".tmp") / "analysis-artifacts-reference"
    if artifact_root.exists():
        shutil.rmtree(artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)

    render_result = SimpleNamespace(
        recipe={"schema_version": "recipe_v1", "config": {"reference_path": "references/reference.wav"}},
        input_analysis=analysis,
        output_analysis=analysis,
        reference_analysis=analysis,
        report=SimpleNamespace(),
    )

    written = write_render_artifacts(
        render_result,
        str(artifact_root / "master.wav"),
        write_recipe=False,
        write_analysis=True,
        write_report=False,
        write_markdown_report=False,
    )

    assert set(written) == {"analysis_input", "analysis_output", "analysis_reference"}
    reference_payload = json.loads(Path(written["analysis_reference"]).read_text(encoding="utf-8"))
    assert reference_payload == json.loads((FIXTURES / "analysis_reference_artifact.json").read_text(encoding="utf-8"))
