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
from cognis.optimizer.decision_history import (
    build_decision_history_artifact,
    unavailable_decision_history_summary,
)
from cognis.optimizer.objective import ObjectiveAttribution, ObjectiveTermAttribution
from cognis.optimizer.search import SearchCandidateEvaluation, SearchTrace
from cognis.optimizer.targets import ReferenceTargeting, TargetValues
from cognis.reports.qc import build_report, render_report_markdown
from cognis.serialization.artifacts import serialize_decision_history


FIXTURES = Path(__file__).parent / "fixtures"


def _analysis(**overrides) -> AnalysisResult:
    data = {
        "integrated_lufs": -14.2,
        "short_term_max_lufs": -12.1,
        "momentary_max_lufs": -9.0,
        "sample_peak_dbfs": -1.6,
        "true_peak_dbfs": -1.3,
        "crest_factor_db": 8.0,
        "spectral_tilt_db_per_decade": 0.5,
        "low_mid_balance_db": 0.0,
        "high_mid_balance_db": 0.0,
        "sub_energy_ratio": 0.12,
        "bass_energy_ratio": 0.19,
        "low_energy_ratio": 0.31,
        "high_energy_ratio": 0.08,
        "low_band_centroid_hz": 82.0,
        "phase_correlation": 0.88,
        "low_band_width": 0.05,
        "mid_band_width": 0.28,
        "high_band_width": 0.42,
        "side_energy_ratio": 0.12,
        "mono_null_ratio_db": -10.5,
        "left_right_balance_db": 0.2,
        "limiter_stress_estimate": 0.4,
        "codec_risk_estimate": 0.3,
        "clipping_risk_estimate": 0.1,
        "delivery_safety_estimate": 0.86,
        "hot_sample_ratio": 0.0,
        "near_full_scale_ratio": 0.0,
        "clipped_sample_count": 0,
        "clipped_sample_ratio": 0.0,
        "intersample_peak_excess_db": 0.3,
        "codec_headroom_margin_db": 0.3,
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


def _term(name: str, category: str, difference: float | None, penalty: float) -> ObjectiveTermAttribution:
    return ObjectiveTermAttribution(
        name=name,
        category=category,
        basis="exact",
        value=0.0,
        target_value=0.0,
        difference=difference,
        weight=1.0,
        penalty=penalty,
        active=True,
        note="test term",
    )


def _reference_targeting() -> ReferenceTargeting:
    return ReferenceTargeting(
        schema_version="reference_targeting_v1",
        reference_analysis_schema_version="analysis_schema_v2",
        reference_identity={},
        input_vs_reference={},
        reference_targets={},
        safety_constraints={},
        guidance=("guided",),
        alignment_index=1.0,
    )


def _attr(total: float, dominant: str, loud_diff: float, tp_penalty: float, loud_penalty: float) -> ObjectiveAttribution:
    return ObjectiveAttribution(
        schema_version="objective_attribution_v1",
        score_basis="exact",
        reference_basis="derived",
        reference_targeting_schema_version="reference_targeting_v1",
        total_score=total,
        terms=(
            _term("true_peak_ceiling", "hard_constraint", 0.0, tp_penalty),
            _term("reference_integrated_lufs", "reference_alignment", loud_diff, loud_penalty),
        ),
        reference_constraints=(),
        dominant_term=dominant,
        dominant_penalty=max(tp_penalty, loud_penalty),
    )


def _fixture_trace() -> SearchTrace:
    return SearchTrace(
        schema_version="objective_search_trace_v2",
        selection_basis="exact_bounded_grid_search",
        parameter_axes={
            "brightness": (-0.2, 0.0, 0.2),
            "width": (0.9, 1.0, 1.1),
            "bass_preservation": (0.8, 1.0),
            "dynamics_preservation": (0.8, 1.0),
        },
        candidate_count=3,
        ranking_rule="sort_by_score_then_index",
        ranked_candidate_indexes=(0, 1, 2),
        best_index=0,
        best_params={"bass_preservation": 1.0, "brightness": 0.0, "dynamics_preservation": 1.0, "width": 1.0},
        best_score=2.0,
        runner_up_index=1,
        runner_up_score=3.0,
        winner_score_margin_to_runner_up=1.0,
        tie_count_at_best_score=1,
        score_margin_to_next=1.0,
        evaluations=(
            SearchCandidateEvaluation(
                index=0,
                params={"bass_preservation": 1.0, "brightness": 0.0, "dynamics_preservation": 1.0, "width": 1.0},
                score=2.0,
                attribution=_attr(2.0, "reference_integrated_lufs", -2.0, 0.0, 2.0),
            ),
            SearchCandidateEvaluation(
                index=1,
                params={"bass_preservation": 1.0, "brightness": 0.2, "dynamics_preservation": 1.0, "width": 1.0},
                score=3.0,
                attribution=_attr(3.0, "true_peak_ceiling", -1.5, 2.0, 1.0),
            ),
            SearchCandidateEvaluation(
                index=2,
                params={"bass_preservation": 0.8, "brightness": 0.2, "dynamics_preservation": 0.8, "width": 1.1},
                score=4.0,
                attribution=_attr(4.0, "true_peak_ceiling", -0.5, 3.5, 0.5),
            ),
        ),
    )


def _fixture_targets() -> TargetValues:
    return TargetValues(
        target_loudness=-14.0,
        ceiling_db=-1.0,
        target_tilt=0.0,
        target_width=1.0,
        target_crest_factor=9.0,
        target_low_band_width=0.2,
        reference_targeting=_reference_targeting(),
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


def test_decision_history_serialization_matches_golden_fixture():
    artifact = build_decision_history_artifact(_fixture_trace(), _fixture_targets())
    payload = json.loads(serialize_decision_history(artifact))
    expected = json.loads((FIXTURES / "decision_history_tradeoff_loudness_vs_ceiling.json").read_text(encoding="utf-8"))
    assert payload == expected


def test_decision_history_ordering_is_deterministic_for_winner_and_runner_up():
    artifact = build_decision_history_artifact(_fixture_trace(), _fixture_targets())
    assert artifact.selection.winner_candidate_index == 0
    assert artifact.selection.runner_up_candidate_index == 1
    assert [candidate.index for candidate in artifact.evaluated_candidates] == [0, 1, 2]
    assert [candidate.rank for candidate in artifact.evaluated_candidates] == [1, 2, 3]


def test_decision_history_labels_exact_and_inferred_and_unavailable_honestly():
    artifact = build_decision_history_artifact(_fixture_trace(), _fixture_targets())
    assert artifact.selection_tradeoffs[0].separation_terms[0].evidence_level == "exact"
    assert artifact.selection_tradeoffs[0].summary_level == "inferred"
    unavailable = [item for item in artifact.reference_metric_tradeoffs if item.status == "unavailable"]
    assert unavailable
    assert all(item.summary_level == "unavailable" for item in unavailable)


def test_decision_history_honestly_reports_unavailable_when_not_emitted():
    summary = unavailable_decision_history_summary(
        "Decision-history artifacts are only emitted for reference-aware runs in this milestone."
    )
    report = build_report(
        _config(),
        "recipe_v2",
        TargetValues(
            target_loudness=-14.0,
            ceiling_db=-1.0,
            target_tilt=0.0,
            target_width=1.0,
            target_crest_factor=9.0,
            target_low_band_width=0.2,
        ),
        _analysis(integrated_lufs=-18.0),
        _analysis(),
        decision_history_summary=summary,
    )
    markdown = render_report_markdown(report)
    assert report.decision_history_summary is not None
    assert report.decision_history_summary.available is False
    assert "Decision history unavailable" in markdown
    assert "reference-aware runs" in markdown


def test_decision_history_does_not_claim_continuous_or_global_search():
    artifact = build_decision_history_artifact(_fixture_trace(), _fixture_targets())
    messages = [item.message for item in artifact.limitations]
    assert any("bounded deterministic grid" in message for message in messages)
    assert any("global optimum" in message for message in messages)
