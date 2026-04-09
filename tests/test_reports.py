import json
from pathlib import Path

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
from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.optimizer.targets import build_targets
from cognis.reports.qc import build_report, render_report_markdown
from cognis.serialization.artifacts import serialize_analysis, serialize_report


FIXTURES = Path(__file__).parent / "fixtures"


def _analysis(**overrides) -> AnalysisResult:
    data = {
        "integrated_lufs": -15.0,
        "short_term_max_lufs": -12.5,
        "momentary_max_lufs": -10.8,
        "sample_peak_dbfs": -3.0,
        "true_peak_dbfs": -2.5,
        "crest_factor_db": 10.0,
        "spectral_tilt_db_per_decade": 0.0,
        "low_mid_balance_db": 0.0,
        "high_mid_balance_db": 0.0,
        "sub_energy_ratio": 0.10,
        "bass_energy_ratio": 0.12,
        "low_energy_ratio": 0.24,
        "high_energy_ratio": 0.18,
        "low_band_centroid_hz": 88.0,
        "phase_correlation": 0.7,
        "low_band_width": 0.16,
        "mid_band_width": 0.40,
        "high_band_width": 0.48,
        "side_energy_ratio": 0.18,
        "mono_null_ratio_db": -9.0,
        "left_right_balance_db": 0.0,
        "limiter_stress_estimate": 0.25,
        "codec_risk_estimate": 0.20,
        "clipping_risk_estimate": 0.0,
        "delivery_safety_estimate": 0.85,
        "hot_sample_ratio": 0.001,
        "near_full_scale_ratio": 0.0,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "intersample_peak_excess_db": 0.2,
        "codec_headroom_margin_db": 1.5,
    }
    data.update(overrides)

    return AnalysisResult(
        identity=AnalysisIdentity(
            schema_version=ANALYSIS_SCHEMA_VERSION,
            analyzer_version=ANALYZER_VERSION,
            sample_rate_hz=48000,
            channels=2,
            samples=240000,
            duration_s=5.0,
            role="analysis",
            source_path=None,
            source_name=None,
        ),
        loudness=LoudnessSummary(
            integrated_lufs=data["integrated_lufs"],
            short_term_max_lufs=data["short_term_max_lufs"],
            short_term_mean_lufs=data["integrated_lufs"] + 0.5,
            short_term_min_lufs=data["integrated_lufs"] - 1.5,
            short_term_range_lu=2.0,
            momentary_max_lufs=data["momentary_max_lufs"],
            momentary_mean_lufs=data["integrated_lufs"] + 1.0,
            momentary_min_lufs=data["integrated_lufs"] - 2.0,
            loudness_range_lu=3.0,
            sample_peak_dbfs=data["sample_peak_dbfs"],
            true_peak_dbfs=data["true_peak_dbfs"],
            peak_to_loudness_ratio_lu=data["true_peak_dbfs"] - data["integrated_lufs"],
            crest_factor_db=data["crest_factor_db"],
        ),
        tonal=TonalSummary(
            spectral_tilt_db_per_decade=data["spectral_tilt_db_per_decade"],
            low_mid_balance_db=data["low_mid_balance_db"],
            high_mid_balance_db=data["high_mid_balance_db"],
            sub_energy_ratio=data["sub_energy_ratio"],
            bass_energy_ratio=data["bass_energy_ratio"],
            low_energy_ratio=data["low_energy_ratio"],
            high_energy_ratio=data["high_energy_ratio"],
            low_band_centroid_hz=data["low_band_centroid_hz"],
        ),
        stereo=StereoSummary(
            phase_correlation=data["phase_correlation"],
            low_band_width=data["low_band_width"],
            mid_band_width=data["mid_band_width"],
            high_band_width=data["high_band_width"],
            side_energy_ratio=data["side_energy_ratio"],
            mono_null_ratio_db=data["mono_null_ratio_db"],
            left_right_balance_db=data["left_right_balance_db"],
        ),
        risks=RiskSummary(
            limiter_stress_estimate=data["limiter_stress_estimate"],
            codec_risk_estimate=data["codec_risk_estimate"],
            clipping_risk_estimate=data["clipping_risk_estimate"],
            delivery_safety_estimate=data["delivery_safety_estimate"],
            hot_sample_ratio=data["hot_sample_ratio"],
            near_full_scale_ratio=data["near_full_scale_ratio"],
            clipped_sample_count=data["clipped_sample_count"],
            clipped_sample_ratio=data["clipped_sample_ratio"],
            intersample_peak_excess_db=data["intersample_peak_excess_db"],
            codec_headroom_margin_db=data["codec_headroom_margin_db"],
        ),
        notes=AnalysisNotes(
            momentary_available=True,
            loudness_range_available=True,
            codec_risk_is_proxy=True,
            limiter_stress_is_proxy=True,
        ),
    )


def _config() -> MasteringConfig:
    return MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=4,
        bass_preservation=0.9,
        stereo_width=1.0,
        dynamics_preservation=0.5,
        brightness=0.1,
        fir_backend="AUTO",
    )


def _reference_case():
    input_analysis = _analysis(
        integrated_lufs=-18.4,
        true_peak_dbfs=-3.8,
        crest_factor_db=11.2,
        spectral_tilt_db_per_decade=1.1,
        low_mid_balance_db=0.8,
        high_mid_balance_db=0.3,
        sub_energy_ratio=0.14,
        bass_energy_ratio=0.22,
        phase_correlation=0.92,
        low_band_width=0.18,
    )
    reference_analysis = _analysis(
        integrated_lufs=-12.0,
        true_peak_dbfs=-1.1,
        crest_factor_db=7.0,
        spectral_tilt_db_per_decade=0.4,
        low_mid_balance_db=0.2,
        high_mid_balance_db=-0.9,
        sub_energy_ratio=0.12,
        bass_energy_ratio=0.18,
        phase_correlation=0.84,
        low_band_width=0.11,
    )
    output_analysis = _analysis(
        integrated_lufs=-14.3,
        true_peak_dbfs=-1.12,
        crest_factor_db=8.7,
        spectral_tilt_db_per_decade=0.6,
        low_mid_balance_db=0.4,
        high_mid_balance_db=-0.2,
        sub_energy_ratio=0.13,
        bass_energy_ratio=0.19,
        phase_correlation=0.88,
        low_band_width=0.13,
    )
    return input_analysis, reference_analysis, output_analysis


def _attribution_case_exact():
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        true_peak_dbfs=-3.0,
        crest_factor_db=10.0,
        spectral_tilt_db_per_decade=0.0,
        low_mid_balance_db=0.1,
        high_mid_balance_db=0.0,
        low_band_width=0.10,
        mid_band_width=0.30,
        phase_correlation=0.82,
    )
    reference_analysis = _analysis(
        integrated_lufs=-12.0,
        true_peak_dbfs=-1.2,
        crest_factor_db=7.0,
        spectral_tilt_db_per_decade=1.25,
        low_mid_balance_db=-0.2,
        high_mid_balance_db=-0.8,
        low_band_width=0.125,
        mid_band_width=0.28,
        phase_correlation=0.74,
    )
    output_analysis = _analysis(
        integrated_lufs=-13.5,
        true_peak_dbfs=-1.25,
        crest_factor_db=8.5,
        spectral_tilt_db_per_decade=0.25,
        low_mid_balance_db=0.0,
        high_mid_balance_db=-0.3,
        low_band_width=0.03125,
        mid_band_width=0.29,
        phase_correlation=0.79,
    )
    return input_analysis, reference_analysis, output_analysis


def _attribution_case_heuristic():
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        true_peak_dbfs=-3.0,
        crest_factor_db=9.5,
        spectral_tilt_db_per_decade=-0.2,
        low_mid_balance_db=0.0,
        high_mid_balance_db=0.2,
        low_band_width=0.11,
        mid_band_width=0.30,
        phase_correlation=0.81,
    )
    reference_analysis = _analysis(
        integrated_lufs=-12.5,
        true_peak_dbfs=-1.4,
        crest_factor_db=7.2,
        spectral_tilt_db_per_decade=0.10,
        low_mid_balance_db=-0.1,
        high_mid_balance_db=0.6,
        low_band_width=0.10,
        mid_band_width=0.27,
        phase_correlation=0.75,
    )
    output_analysis = _analysis(
        integrated_lufs=-13.4,
        true_peak_dbfs=-1.25,
        crest_factor_db=8.0,
        spectral_tilt_db_per_decade=0.95,
        low_mid_balance_db=-0.1,
        high_mid_balance_db=0.3,
        low_band_width=0.09,
        mid_band_width=0.28,
        phase_correlation=0.77,
    )
    return input_analysis, reference_analysis, output_analysis


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_analysis_serialization_matches_golden_fixture():
    analysis = _analysis(
        integrated_lufs=-14.2,
        short_term_max_lufs=-12.1,
        momentary_max_lufs=-9.0,
        sample_peak_dbfs=-1.6,
        true_peak_dbfs=-1.3,
        crest_factor_db=7.0,
        spectral_tilt_db_per_decade=0.8,
        low_mid_balance_db=1.2,
        high_mid_balance_db=-1.5,
        sub_energy_ratio=0.12,
        bass_energy_ratio=0.19,
        low_energy_ratio=0.31,
        high_energy_ratio=0.08,
        low_band_centroid_hz=82.0,
        phase_correlation=0.88,
        low_band_width=0.05,
        mid_band_width=0.28,
        high_band_width=0.42,
        side_energy_ratio=0.12,
        mono_null_ratio_db=-10.5,
        left_right_balance_db=0.2,
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
    )

    assert json.loads(serialize_analysis(analysis)) == _load_fixture("analysis_output_safe.json")


def test_pass_report_serialization_matches_golden_fixture():
    config = _config()
    targets = build_targets(config)
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        short_term_max_lufs=-16.0,
        momentary_max_lufs=-14.0,
        sample_peak_dbfs=-4.0,
        true_peak_dbfs=-3.5,
        crest_factor_db=10.0,
        spectral_tilt_db_per_decade=1.0,
        low_mid_balance_db=1.5,
        high_mid_balance_db=0.0,
        sub_energy_ratio=0.13,
        bass_energy_ratio=0.21,
        low_energy_ratio=0.31,
        high_energy_ratio=0.08,
        low_band_centroid_hz=82.0,
        phase_correlation=0.95,
        low_band_width=0.15,
        mid_band_width=0.3,
        high_band_width=0.4,
        side_energy_ratio=0.12,
        mono_null_ratio_db=-10.5,
        left_right_balance_db=0.2,
        limiter_stress_estimate=0.2,
        codec_risk_estimate=0.15,
        clipping_risk_estimate=0.05,
        delivery_safety_estimate=0.95,
        hot_sample_ratio=0.0,
        near_full_scale_ratio=0.0,
        clipped_sample_count=0,
        clipped_sample_ratio=0.0,
        intersample_peak_excess_db=0.3,
        codec_headroom_margin_db=2.5,
    )
    output_analysis = _analysis(
        integrated_lufs=-14.2,
        short_term_max_lufs=-12.1,
        momentary_max_lufs=-9.0,
        sample_peak_dbfs=-1.6,
        true_peak_dbfs=-1.3,
        crest_factor_db=7.0,
        spectral_tilt_db_per_decade=0.8,
        low_mid_balance_db=1.2,
        high_mid_balance_db=-1.5,
        sub_energy_ratio=0.12,
        bass_energy_ratio=0.19,
        low_energy_ratio=0.31,
        high_energy_ratio=0.08,
        low_band_centroid_hz=82.0,
        phase_correlation=0.88,
        low_band_width=0.05,
        mid_band_width=0.28,
        high_band_width=0.42,
        side_energy_ratio=0.12,
        mono_null_ratio_db=-10.5,
        left_right_balance_db=0.2,
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
    )

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis)
    assert json.loads(serialize_report(report)) == _load_fixture("report_pass.json")


def test_fail_report_generates_expected_reason_codes():
    config = _config()
    targets = build_targets(config)
    input_analysis = _analysis(integrated_lufs=-18.0)
    output_analysis = _analysis(
        integrated_lufs=-11.5,
        short_term_max_lufs=-8.4,
        momentary_max_lufs=-2.0,
        sample_peak_dbfs=0.1,
        true_peak_dbfs=-0.6,
        crest_factor_db=4.5,
        spectral_tilt_db_per_decade=3.5,
        low_mid_balance_db=13.0,
        high_mid_balance_db=12.5,
        low_energy_ratio=0.42,
        high_energy_ratio=0.19,
        low_band_centroid_hz=70.0,
        phase_correlation=-0.2,
        low_band_width=0.5,
        mid_band_width=0.52,
        high_band_width=0.75,
        side_energy_ratio=0.55,
        mono_null_ratio_db=-1.5,
        left_right_balance_db=0.8,
        limiter_stress_estimate=0.92,
        codec_risk_estimate=0.8,
        clipping_risk_estimate=0.9,
        delivery_safety_estimate=0.12,
        hot_sample_ratio=0.04,
        near_full_scale_ratio=0.02,
        clipped_sample_count=25,
        clipped_sample_ratio=0.0001,
        intersample_peak_excess_db=0.7,
        codec_headroom_margin_db=-0.4,
    )

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis)
    codes = [finding.code for finding in report.findings]

    assert report.overall_status == "fail"
    assert "LOUDNESS_OUTSIDE_HARD_BAND" in codes
    assert "TRUE_PEAK_ABOVE_CEILING" in codes
    assert "SAMPLE_PEAK_ABOVE_0_DBFS" in codes
    assert "PHASE_CORRELATION_CRITICAL" in codes
    assert "LOW_BAND_WIDTH_CRITICAL" in codes
    assert "CLIPPING_RISK_CRITICAL" in codes
    assert "LIMITER_STRESS_CRITICAL" in codes
    assert "CODEC_RISK_HIGH" in codes


def test_fail_report_summary_and_markdown_do_not_claim_safety():
    config = _config()
    targets = build_targets(config)
    input_analysis = _analysis(integrated_lufs=-18.0)
    output_analysis = _analysis(
        integrated_lufs=-11.5,
        short_term_max_lufs=-8.4,
        momentary_max_lufs=-2.0,
        sample_peak_dbfs=0.1,
        true_peak_dbfs=-0.6,
        crest_factor_db=4.5,
        phase_correlation=-0.2,
        low_band_width=0.5,
        limiter_stress_estimate=0.92,
        clipping_risk_estimate=0.9,
        codec_risk_estimate=0.8,
        codec_headroom_margin_db=-0.4,
    )

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis)
    summary_text = " ".join(bullet.message for bullet in report.summary)
    markdown = render_report_markdown(report)

    assert report.overall_status == "fail"
    assert "release-safety constraints" not in summary_text
    assert "not release-ready" in summary_text
    assert "release-safety constraints" not in markdown
    assert "not release-ready" in markdown


def test_dynamics_risk_summary_does_not_claim_punch_preservation():
    config = _config()
    targets = build_targets(config)
    input_analysis = _analysis(integrated_lufs=-18.0, crest_factor_db=10.0)
    output_analysis = _analysis(
        integrated_lufs=-14.2,
        true_peak_dbfs=-1.3,
        sample_peak_dbfs=-1.6,
        crest_factor_db=6.0,
    )

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis)
    summary_text = " ".join(bullet.message for bullet in report.summary)

    assert any(finding.code == "DYNAMICS_COLLAPSE_RISK" for finding in report.findings)
    assert "Preserved punch within the current crest-factor tolerance." not in summary_text
    assert "fell below the requested dynamics-preservation tolerance" in summary_text


def test_markdown_report_has_single_title_and_populated_requested_section():
    config = _config()
    targets = build_targets(config)
    report = build_report(config, "recipe_v2", targets, _analysis(), _analysis(integrated_lufs=-14.2, true_peak_dbfs=-1.3))
    markdown = render_report_markdown(report)

    assert markdown.count("# COGNIS Render Report") == 1
    assert "# COGNIS QC Report" not in markdown
    assert "## Requested" in markdown
    assert "- Mode: `STREAMING_SAFE`" in markdown
    assert "- Ceiling: `-1.00 dBFS` (`TRUE_PEAK`)" in markdown


def test_reference_report_serialization_includes_reference_assessment():
    config = _config()
    targets = build_targets(config)
    input_analysis, reference_analysis, output_analysis = _reference_case()

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    payload = json.loads(serialize_report(report))

    assert report.schema_version == "report_schema_v3"
    assert report.reference_status in {"constrained", "partial", "matched", "deviated"}
    assert payload["schema_version"] == "report_schema_v3"
    assert payload["reference_assessment"]["schema_version"] == "reference_assessment_schema_v2"
    assert payload["reference_assessment"]["reference_analysis_schema_version"] == "analysis_schema_v2"
    assert payload["reference_assessment"]["outcome"] == report.reference_status
    assert any(item["metric"] == "integrated_lufs" for item in payload["reference_assessment"]["comparisons"])
    assert all(not finding["code"].startswith("REFERENCE_") for finding in payload["findings"])


def test_reference_report_markdown_separates_reference_and_safety_language():
    config = _config()
    targets = build_targets(config)
    input_analysis, reference_analysis, output_analysis = _reference_case()

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    markdown = render_report_markdown(report)

    assert "## Reference" in markdown
    assert "## Reference Comparison" in markdown
    assert "## Reference Summary" in markdown
    assert "## Reference Attribution" in markdown
    assert "## Reference Findings" in markdown
    assert "limited by a" in markdown or "consistent with the dynamics-preservation target" in markdown
    assert "## QC Findings" in markdown
    assert "REFERENCE_" in markdown


def test_reference_report_includes_constraint_attribution_with_stable_schema():
    config = _config()
    input_analysis, reference_analysis, output_analysis = _attribution_case_exact()
    targets = build_targets(config, input_analysis, reference_analysis)

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    payload = json.loads(serialize_report(report))

    attribution = payload["reference_assessment"]["attribution"]
    assert attribution["schema_version"] == "reference_attribution_schema_v1"
    assert attribution["available"] is True
    levels = [entry["attribution_level"] for entry in attribution["entries"]]
    assert levels[:2] == ["exact", "exact"]
    assert set(levels[2:]).issubset({"exact", "inferred"})
    assert {entry["category"] for entry in attribution["entries"]} == {
        "loudness_ceiling_constraint",
        "mono_low_band_width_safety",
        "dynamics_preservation_constraint",
        "tonal_correction_limit",
    }
    assert all(entry["explanation"] for entry in attribution["entries"])


def test_reference_report_markdown_reflects_attribution_payload():
    config = _config()
    input_analysis, reference_analysis, output_analysis = _attribution_case_exact()
    targets = build_targets(config, input_analysis, reference_analysis)

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    payload = json.loads(serialize_report(report))
    markdown = render_report_markdown(report)

    attribution = payload["reference_assessment"]["attribution"]
    for entry in attribution["entries"]:
        assert entry["category"] in markdown
        assert entry["explanation"] in markdown
        assert f"`{entry['attribution_level']}`" in markdown


def test_reference_report_honestly_states_when_attribution_is_unavailable():
    config = _config()
    targets = build_targets(config)
    input_analysis, reference_analysis, output_analysis = _reference_case()

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    payload = json.loads(serialize_report(report))
    markdown = render_report_markdown(report)

    attribution = payload["reference_assessment"]["attribution"]
    assert attribution["available"] is False
    assert attribution["entries"] == []
    assert "Attribution unavailable" in markdown
    assert "no causal attribution was generated" in markdown


def test_reference_report_uses_unavailable_label_when_no_supported_tradeoff_exists():
    config = _config()
    input_analysis, reference_analysis, output_analysis = _attribution_case_heuristic()
    targets = build_targets(config, input_analysis, reference_analysis)

    report = build_report(config, "recipe_v2", targets, input_analysis, output_analysis, reference_analysis)
    payload = json.loads(serialize_report(report))

    labels = [entry["attribution_level"] for entry in payload["reference_assessment"]["attribution"]["entries"]]
    assert "exact" in labels
    assert "unavailable" in labels
