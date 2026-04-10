import numpy as np
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
from cognis.analysis.analyzer import Analyzer
from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.optimizer.objective import build_objective_attribution, compute_objective
from cognis.optimizer.search import grid_search, grid_search_with_trace
from cognis.optimizer.targets import TargetValues, build_targets


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


def test_objective_attribution_is_exact_and_honest_without_reference():
    targets = TargetValues(
        target_loudness=-14.0,
        ceiling_db=-1.0,
        target_tilt=0.0,
        target_width=1.0,
        target_crest_factor=9.0,
        target_low_band_width=0.2,
    )
    analysis = _analysis(
        integrated_lufs=-12.7,
        true_peak_dbfs=-0.4,
        phase_correlation=-0.2,
        low_band_width=0.35,
        mid_band_width=1.2,
        crest_factor_db=5.5,
    )

    attribution = build_objective_attribution(analysis, targets)

    assert attribution.schema_version == "objective_attribution_v1"
    assert attribution.score_basis == "exact"
    assert attribution.reference_basis == "unavailable"
    assert attribution.reference_constraints == ()
    assert attribution.total_score == pytest.approx(compute_objective(analysis, targets))
    assert attribution.dominant_term == "true_peak_ceiling"
    assert attribution.dominant_penalty == pytest.approx(6000.0)

    terms = {term.name: term for term in attribution.terms}
    assert terms["true_peak_ceiling"].basis == "exact"
    assert terms["true_peak_ceiling"].penalty == pytest.approx(6000.0)
    assert terms["phase_correlation_floor"].active is True
    assert terms["phase_correlation_floor"].penalty == pytest.approx(1000.0)
    assert terms["crest_factor_floor"].active is True


def test_reference_constraint_attribution_is_structured_and_non_fabricated():
    config = _config()
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

    targets = build_targets(config, input_analysis=input_analysis, reference_analysis=reference_analysis)
    attribution = build_objective_attribution(_analysis(), targets)

    assert attribution.reference_basis == "derived"
    assert attribution.reference_targeting_schema_version == "reference_targeting_v1"
    assert len(attribution.reference_constraints) == 8

    constraints = {item.metric: item for item in attribution.reference_constraints}
    assert constraints["low_band_width"].category == "mono_safety"
    assert constraints["low_band_width"].basis == "derived"
    assert constraints["low_band_width"].safety_limited is True
    assert "mono-safety baseline" in constraints["low_band_width"].note
    assert constraints["crest_factor_db"].category == "dynamics_preservation"
    assert constraints["crest_factor_db"].basis == "derived"
    assert constraints["crest_factor_db"].safety_limited is True


def test_grid_search_with_trace_matches_scalar_search_and_reports_margin():
    analyzer = Analyzer()
    targets = TargetValues(
        target_loudness=-14.0,
        ceiling_db=-1.0,
        target_tilt=0.0,
        target_width=1.0,
        target_crest_factor=9.0,
        target_low_band_width=0.2,
    )

    rng = np.random.default_rng(0)
    audio = rng.normal(size=(2, 4800)).astype(np.float32) * 0.1

    def dummy_render(aud, params):
        gain = 1.0 + params.get("brightness", 0.0)
        return aud * gain

    trace = grid_search_with_trace(audio, 48000, targets, dummy_render, analyzer)
    best_params = grid_search(audio, 48000, targets, dummy_render, analyzer)

    assert trace.schema_version == "objective_search_trace_v2"
    assert trace.selection_basis == "exact_bounded_grid_search"
    assert trace.candidate_count == 36
    assert trace.ranking_rule == "sort_by_score_then_index"
    assert trace.ranked_candidate_indexes[0] == trace.best_index
    assert trace.best_params == best_params
    assert trace.best_score == min(item.score for item in trace.evaluations)
    assert trace.tie_count_at_best_score >= 1
    assert trace.score_margin_to_next is None or trace.score_margin_to_next >= 0.0
    assert trace.evaluations[trace.best_index].params == trace.best_params
