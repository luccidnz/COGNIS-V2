import json

import pytest

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
from cognis.optimizer.reference_deltas import build_reference_deltas
from cognis.optimizer.reference_targets import build_reference_aware_targets
from cognis.optimizer.targets import build_targets


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
        mode=MasteringMode.REFERENCE_MATCH,
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


def test_reference_target_plan_is_deterministic_and_safe():
    config = _config()
    baseline_targets = build_targets(config)
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        true_peak_dbfs=-3.2,
        crest_factor_db=11.0,
        spectral_tilt_db_per_decade=-0.2,
        low_mid_balance_db=0.8,
        high_mid_balance_db=0.4,
        low_band_centroid_hz=92.0,
        phase_correlation=0.82,
        low_band_width=0.14,
        mid_band_width=0.42,
        high_band_width=0.49,
    )
    reference_analysis = _analysis(
        integrated_lufs=-10.0,
        true_peak_dbfs=-1.4,
        crest_factor_db=6.5,
        spectral_tilt_db_per_decade=1.8,
        low_mid_balance_db=-1.6,
        high_mid_balance_db=2.1,
        low_band_centroid_hz=68.0,
        phase_correlation=0.31,
        low_band_width=0.46,
        mid_band_width=0.57,
        high_band_width=0.63,
    )

    first = build_reference_aware_targets(config, input_analysis, reference_analysis)
    second = build_reference_aware_targets(config, input_analysis, reference_analysis)

    assert first.to_dict() == second.to_dict()
    assert first.reference_available is True
    assert first.baseline_targets == baseline_targets
    assert first.loudness.target_value == -14.0
    assert first.loudness.safety_limited is True
    assert first.spectral_tilt.target_value == baseline_targets.target_tilt
    assert first.spectral_tilt.safety_limited is True
    assert first.low_mid_balance.target_value == -1.6
    assert first.high_mid_balance.target_value == 2.1
    assert first.low_band_centroid.target_value == 68.0
    assert first.mid_band_width.target_value == 0.57
    assert first.low_band_width.target_value == baseline_targets.target_low_band_width
    assert first.phase_correlation.target_value == 0.82
    assert first.crest_factor.target_value == 11.0
    assert first.crest_factor.safety_limited is True


def test_reference_target_plan_without_reference_uses_config_and_input_baselines():
    config = _config()
    baseline_targets = build_targets(config)
    input_analysis = _analysis(
        integrated_lufs=-17.5,
        spectral_tilt_db_per_decade=-0.4,
        low_mid_balance_db=1.1,
        high_mid_balance_db=-0.9,
        low_band_centroid_hz=84.0,
        phase_correlation=0.74,
        low_band_width=0.18,
        mid_band_width=0.39,
        crest_factor_db=9.6,
    )

    plan = build_reference_aware_targets(config, input_analysis)

    assert plan.reference_available is False
    assert plan.reference_identity is None
    assert plan.loudness.target_value == baseline_targets.target_loudness
    assert plan.spectral_tilt.target_value == baseline_targets.target_tilt
    assert plan.low_mid_balance.target_value == 1.1
    assert plan.high_mid_balance.target_value == -0.9
    assert plan.low_band_centroid.target_value == 84.0
    assert plan.mid_band_width.target_value == baseline_targets.target_width
    assert plan.low_band_width.target_value == baseline_targets.target_low_band_width
    assert plan.phase_correlation.target_value == 0.74
    assert plan.crest_factor.target_value == baseline_targets.target_crest_factor


def test_build_targets_populates_reference_targeting_payload():
    config = _config()
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        low_mid_balance_db=0.6,
        high_mid_balance_db=-0.2,
        low_energy_ratio=0.21,
        sub_energy_ratio=0.12,
        phase_correlation=0.78,
    )
    reference_analysis = _analysis(
        integrated_lufs=-11.0,
        low_mid_balance_db=-1.4,
        high_mid_balance_db=1.7,
        low_energy_ratio=0.33,
        sub_energy_ratio=0.18,
        side_energy_ratio=0.29,
        phase_correlation=0.41,
    )

    targets = build_targets(config, input_analysis=input_analysis, reference_analysis=reference_analysis)

    assert targets.reference_targeting is not None
    assert targets.target_low_mid_balance == reference_analysis.tonal.low_mid_balance_db
    assert targets.target_high_mid_balance == reference_analysis.tonal.high_mid_balance_db
    assert targets.target_sub_energy_ratio == reference_analysis.tonal.sub_energy_ratio
    assert targets.target_low_energy_ratio == reference_analysis.tonal.low_energy_ratio
    assert targets.target_side_energy_ratio == reference_analysis.stereo.side_energy_ratio
    assert targets.reference_targeting.reference_targets["integrated_lufs"] == reference_analysis.loudness.integrated_lufs
    assert targets.reference_targeting.input_vs_reference["phase_correlation_delta"] == pytest.approx(-0.37)
    assert targets.reference_targeting.safety_constraints["low_band_width_cap"] == pytest.approx(
        targets.target_low_band_width + 0.1
    )


def test_reference_delta_bundle_reports_all_pairwise_deltas():
    input_analysis = _analysis(
        integrated_lufs=-18.0,
        true_peak_dbfs=-3.2,
        crest_factor_db=11.0,
        spectral_tilt_db_per_decade=-0.2,
        low_mid_balance_db=0.8,
        high_mid_balance_db=0.4,
        low_band_centroid_hz=92.0,
        phase_correlation=0.82,
        low_band_width=0.14,
        mid_band_width=0.42,
        high_band_width=0.49,
        side_energy_ratio=0.15,
        mono_null_ratio_db=-8.0,
        left_right_balance_db=0.2,
    )
    reference_analysis = _analysis(
        integrated_lufs=-10.0,
        true_peak_dbfs=-1.4,
        crest_factor_db=6.5,
        spectral_tilt_db_per_decade=1.8,
        low_mid_balance_db=-1.6,
        high_mid_balance_db=2.1,
        low_band_centroid_hz=68.0,
        phase_correlation=0.31,
        low_band_width=0.46,
        mid_band_width=0.57,
        high_band_width=0.63,
        side_energy_ratio=0.41,
        mono_null_ratio_db=-2.5,
        left_right_balance_db=0.6,
    )
    output_analysis = _analysis(
        integrated_lufs=-12.5,
        true_peak_dbfs=-1.9,
        crest_factor_db=7.7,
        spectral_tilt_db_per_decade=0.9,
        low_mid_balance_db=-0.3,
        high_mid_balance_db=1.0,
        low_band_centroid_hz=74.0,
        phase_correlation=0.61,
        low_band_width=0.22,
        mid_band_width=0.51,
        high_band_width=0.56,
        side_energy_ratio=0.28,
        mono_null_ratio_db=-4.1,
        left_right_balance_db=0.4,
    )

    deltas = build_reference_deltas(input_analysis, reference_analysis, output_analysis)
    repeat = build_reference_deltas(input_analysis, reference_analysis, output_analysis)

    assert deltas.to_dict() == repeat.to_dict()
    assert deltas.reference_available is True
    assert deltas.loudness.integrated_lufs.input_vs_reference == -8.0
    assert deltas.loudness.integrated_lufs.output_vs_reference == -2.5
    assert deltas.loudness.integrated_lufs.input_vs_output == -5.5
    assert deltas.loudness.true_peak_dbfs.input_vs_reference == pytest.approx(-1.8)
    assert deltas.tonal.spectral_tilt_db_per_decade.input_vs_reference == pytest.approx(-2.0)
    assert deltas.stereo.low_band_width.output_vs_reference == pytest.approx(-0.24)
    assert deltas.stereo.phase_correlation.input_vs_output == pytest.approx(0.21)


def test_reference_delta_bundle_handles_missing_reference():
    input_analysis = _analysis()
    output_analysis = _analysis(integrated_lufs=-14.0, true_peak_dbfs=-1.5)

    deltas = build_reference_deltas(input_analysis, None, output_analysis)

    assert deltas.reference_available is False
    assert deltas.loudness.integrated_lufs.input_vs_reference is None
    assert deltas.loudness.integrated_lufs.output_vs_reference is None
    assert deltas.loudness.integrated_lufs.input_vs_output == -1.0
    assert json.loads(json.dumps(deltas.to_dict(), sort_keys=True))
