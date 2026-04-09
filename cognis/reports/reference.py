from __future__ import annotations

from dataclasses import dataclass

from cognis.analysis.features import AnalysisIdentity, AnalysisResult
from cognis.config import MasteringConfig
from cognis.optimizer.targets import TargetValues


REFERENCE_ASSESSMENT_SCHEMA_VERSION = "reference_assessment_schema_v1"

REFERENCE_TOLERANCES: dict[str, float] = {
    "integrated_lufs": 0.5,
    "true_peak_dbfs": 0.3,
    "spectral_tilt_db_per_decade": 0.75,
    "low_mid_balance_db": 1.0,
    "high_mid_balance_db": 1.0,
    "sub_energy_ratio": 0.05,
    "bass_energy_ratio": 0.05,
    "low_band_width": 0.05,
    "mid_band_width": 0.05,
    "phase_correlation": 0.05,
    "crest_factor_db": 1.0,
}


@dataclass(frozen=True)
class ReferenceEvidence:
    metric: str
    value: float
    unit: str
    threshold: float | None = None
    comparator: str | None = None


@dataclass(frozen=True)
class ReferenceFinding:
    code: str
    severity: str
    message: str
    evidence: tuple[ReferenceEvidence, ...]


@dataclass(frozen=True)
class ReferenceSummaryBullet:
    category: str
    message: str


@dataclass(frozen=True)
class ReferenceMetricComparison:
    metric: str
    label: str
    unit: str
    tolerance: float
    input_value: float
    reference_value: float
    output_value: float
    input_delta_to_reference: float
    output_delta_to_reference: float
    movement_toward_reference: float
    status: str
    explanation: str


@dataclass(frozen=True)
class ReferenceAssessment:
    schema_version: str
    reference_path: str | None
    reference_analysis_schema_version: str
    reference_analysis_identity: AnalysisIdentity
    comparisons: tuple[ReferenceMetricComparison, ...]
    findings: tuple[ReferenceFinding, ...]
    summary: tuple[ReferenceSummaryBullet, ...]
    outcome: str


def _format_value(value: float, unit: str) -> str:
    precision = 2 if unit not in {"ratio", "score"} else 3
    if unit == "ratio":
        precision = 2
    return f"{value:.{precision}f} {unit}"


def _evidence(metric: str, value: float, unit: str, threshold: float | None = None, comparator: str | None = None) -> ReferenceEvidence:
    return ReferenceEvidence(metric=metric, value=float(value), unit=unit, threshold=threshold, comparator=comparator)


def _comparison_label(metric: str) -> tuple[str, str]:
    labels: dict[str, tuple[str, str]] = {
        "integrated_lufs": ("Integrated loudness", "LUFS"),
        "true_peak_dbfs": ("True peak", "dBFS"),
        "spectral_tilt_db_per_decade": ("Spectral tilt", "dB/decade"),
        "low_mid_balance_db": ("Low-mid balance", "dB"),
        "high_mid_balance_db": ("High-mid balance", "dB"),
        "sub_energy_ratio": ("Sub energy ratio", "ratio"),
        "bass_energy_ratio": ("Bass energy ratio", "ratio"),
        "low_band_width": ("Low-band width", "ratio"),
        "mid_band_width": ("Mid-band width", "ratio"),
        "phase_correlation": ("Phase correlation", "ratio"),
        "crest_factor_db": ("Crest factor", "dB"),
    }
    return labels[metric]


def _metric_value(analysis: AnalysisResult, metric: str) -> float:
    lookup: dict[str, float] = {
        "integrated_lufs": analysis.loudness.integrated_lufs,
        "true_peak_dbfs": analysis.loudness.true_peak_dbfs,
        "spectral_tilt_db_per_decade": analysis.tonal.spectral_tilt_db_per_decade,
        "low_mid_balance_db": analysis.tonal.low_mid_balance_db,
        "high_mid_balance_db": analysis.tonal.high_mid_balance_db,
        "sub_energy_ratio": analysis.tonal.sub_energy_ratio,
        "bass_energy_ratio": analysis.tonal.bass_energy_ratio,
        "low_band_width": analysis.stereo.low_band_width,
        "mid_band_width": analysis.stereo.mid_band_width,
        "phase_correlation": analysis.stereo.phase_correlation,
        "crest_factor_db": analysis.loudness.crest_factor_db,
    }
    return float(lookup[metric])


def _comparison_status(input_gap: float, output_gap: float, tolerance: float) -> tuple[str, str]:
    if abs(output_gap) <= tolerance:
        return "matched", "output landed within the reference tolerance"
    if abs(output_gap) < abs(input_gap):
        return "improved", "output moved closer to the reference"
    if abs(output_gap) > abs(input_gap):
        return "moved_away", "output moved farther from the reference"
    return "unchanged", "output stayed at the same distance from the reference"


def _build_comparison(metric: str, input_analysis: AnalysisResult, reference_analysis: AnalysisResult, output_analysis: AnalysisResult) -> ReferenceMetricComparison:
    label, unit = _comparison_label(metric)
    tolerance = REFERENCE_TOLERANCES[metric]
    input_value = _metric_value(input_analysis, metric)
    reference_value = _metric_value(reference_analysis, metric)
    output_value = _metric_value(output_analysis, metric)
    input_delta = input_value - reference_value
    output_delta = output_value - reference_value
    movement = abs(input_delta) - abs(output_delta)
    status, explanation = _comparison_status(input_delta, output_delta, tolerance)
    return ReferenceMetricComparison(
        metric=metric,
        label=label,
        unit=unit,
        tolerance=tolerance,
        input_value=input_value,
        reference_value=reference_value,
        output_value=output_value,
        input_delta_to_reference=input_delta,
        output_delta_to_reference=output_delta,
        movement_toward_reference=movement,
        status=status,
        explanation=explanation,
    )


def _metric_lookup(comparisons: tuple[ReferenceMetricComparison, ...]) -> dict[str, ReferenceMetricComparison]:
    return {comparison.metric: comparison for comparison in comparisons}


def _format_gap_message(comparison: ReferenceMetricComparison) -> str:
    return f"{comparison.label} moved {abs(comparison.movement_toward_reference):.2f} {comparison.unit} toward the reference; {abs(comparison.output_delta_to_reference):.2f} {comparison.unit} remain."


def _summary_for_loudness(
    comparison: ReferenceMetricComparison,
    output_analysis: AnalysisResult,
    config: MasteringConfig,
) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    true_peak_margin_db = config.ceiling_db - output_analysis.loudness.true_peak_dbfs
    if comparison.status == "matched":
        message = f"{comparison.label} matched the reference within {comparison.tolerance:.2f} {comparison.unit}."
        finding = ReferenceFinding(
            code="REFERENCE_LOUDNESS_MATCHED",
            severity="informational",
            message=message,
            evidence=(
                _evidence("integrated_lufs", comparison.output_value, "LUFS", comparison.reference_value, "=="),
                _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, "LUFS", comparison.tolerance, "<="),
            ),
        )
    elif true_peak_margin_db < 0.3:
        message = (
            f"{comparison.label} moved {abs(comparison.movement_toward_reference):.2f} {comparison.unit} toward the reference; "
            f"{abs(comparison.output_delta_to_reference):.2f} {comparison.unit} remain, limited by a {true_peak_margin_db:.2f} dB true-peak margin."
        )
        finding = ReferenceFinding(
            code="REFERENCE_LOUDNESS_CONSTRAINED_BY_CEILING",
            severity="warning",
            message=message,
            evidence=(
                _evidence("true_peak_margin_db", true_peak_margin_db, "dB", 0.3, "<"),
                _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, "LUFS", comparison.tolerance, ">"),
            ),
        )
    elif comparison.status == "improved":
        message = _format_gap_message(comparison)
        finding = ReferenceFinding(
            code="REFERENCE_LOUDNESS_CLOSER",
            severity="informational",
            message=message,
            evidence=(
                _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit, comparison.tolerance, ">"),
            ),
        )
    elif comparison.status == "moved_away":
        message = (
            f"{comparison.label} moved away from the reference by {abs(comparison.movement_toward_reference):.2f} {comparison.unit}; "
            f"{abs(comparison.output_delta_to_reference):.2f} {comparison.unit} remain."
        )
        finding = ReferenceFinding(
            code="REFERENCE_LOUDNESS_MOVED_AWAY",
            severity="warning",
            message=message,
            evidence=(
                _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit, comparison.tolerance, ">"),
            ),
        )
    else:
        message = f"{comparison.label} stayed at the same distance from the reference."
        finding = ReferenceFinding(
            code="REFERENCE_LOUDNESS_UNCHANGED",
            severity="informational",
            message=message,
            evidence=(
                _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit),
            ),
        )
    return ReferenceSummaryBullet("loudness", message), finding


def _summary_for_tonal(
    spectral_tilt: ReferenceMetricComparison,
    low_mid: ReferenceMetricComparison,
    high_mid: ReferenceMetricComparison,
) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    primary = spectral_tilt if abs(spectral_tilt.output_delta_to_reference) >= abs(high_mid.output_delta_to_reference) else high_mid
    if spectral_tilt.status == "matched" and low_mid.status == "matched" and high_mid.status == "matched":
        message = "Tonal balance matched the reference within tolerance."
        finding = ReferenceFinding(
            code="REFERENCE_TONAL_MATCHED",
            severity="informational",
            message=message,
            evidence=(
                _evidence("spectral_tilt_db_per_decade", spectral_tilt.output_delta_to_reference, "dB/decade", spectral_tilt.tolerance, "<="),
                _evidence("high_mid_balance_db", high_mid.output_delta_to_reference, "dB", high_mid.tolerance, "<="),
            ),
        )
    else:
        message = (
            f"{spectral_tilt.label} moved {abs(spectral_tilt.movement_toward_reference):.2f} {spectral_tilt.unit} toward the reference; "
            f"{primary.label.lower()} remains {abs(primary.output_delta_to_reference):.2f} {primary.unit} from it."
        )
        finding = ReferenceFinding(
            code="REFERENCE_TONAL_DEVIATION",
            severity="warning",
            message=message,
            evidence=(
                _evidence(spectral_tilt.metric, spectral_tilt.output_delta_to_reference, spectral_tilt.unit, spectral_tilt.tolerance, ">"),
                _evidence(high_mid.metric, high_mid.output_delta_to_reference, high_mid.unit, high_mid.tolerance, ">"),
            ),
        )
    return ReferenceSummaryBullet("tonal", message), finding


def _summary_for_low_end(
    sub_energy: ReferenceMetricComparison,
    bass_energy: ReferenceMetricComparison,
    low_band_width: ReferenceMetricComparison,
    targets: TargetValues,
) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    if low_band_width.status == "matched":
        message = (
            f"{low_band_width.label} matched the reference within {low_band_width.tolerance:.2f} {low_band_width.unit}; "
            f"sub and bass energy ratios also remained close to the reference."
        )
        finding = ReferenceFinding(
            code="REFERENCE_LOW_END_MATCHED",
            severity="informational",
            message=message,
            evidence=(
                _evidence(low_band_width.metric, low_band_width.output_delta_to_reference, low_band_width.unit, low_band_width.tolerance, "<="),
            ),
        )
    elif low_band_width.output_value <= targets.target_low_band_width + 0.1 and low_band_width.output_value <= low_band_width.reference_value:
        message = (
            f"{low_band_width.label} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} narrower than the reference to preserve mono safety."
        )
        finding = ReferenceFinding(
            code="REFERENCE_LOW_END_CONSTRAINED_BY_MONO_SAFETY",
            severity="informational",
            message=message,
            evidence=(
                _evidence(low_band_width.metric, low_band_width.output_value, low_band_width.unit, targets.target_low_band_width + 0.1, "<="),
                _evidence("bass_preservation_target_low_band_width", targets.target_low_band_width, "ratio"),
            ),
        )
    else:
        message = (
            f"{low_band_width.label} moved {abs(low_band_width.movement_toward_reference):.2f} {low_band_width.unit} toward the reference; "
            f"sub energy and bass energy remain {abs(sub_energy.output_delta_to_reference):.2f} and {abs(bass_energy.output_delta_to_reference):.2f} {sub_energy.unit} from it."
        )
        finding = ReferenceFinding(
            code="REFERENCE_LOW_END_DEVIATION",
            severity="warning",
            message=message,
            evidence=(
                _evidence(sub_energy.metric, sub_energy.output_delta_to_reference, sub_energy.unit, sub_energy.tolerance, ">"),
                _evidence(bass_energy.metric, bass_energy.output_delta_to_reference, bass_energy.unit, bass_energy.tolerance, ">"),
            ),
        )
    return ReferenceSummaryBullet("low_end", message), finding


def _summary_for_stereo(
    mid_band_width: ReferenceMetricComparison,
    phase_correlation: ReferenceMetricComparison,
    low_band_width: ReferenceMetricComparison,
    targets: TargetValues,
) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    if mid_band_width.status == "matched" and phase_correlation.status == "matched" and low_band_width.status == "matched":
        message = "Stereo width and balance matched the reference within tolerance."
        finding = ReferenceFinding(
            code="REFERENCE_STEREO_MATCHED",
            severity="informational",
            message=message,
            evidence=(
                _evidence(mid_band_width.metric, mid_band_width.output_delta_to_reference, mid_band_width.unit, mid_band_width.tolerance, "<="),
                _evidence(phase_correlation.metric, phase_correlation.output_delta_to_reference, phase_correlation.unit, phase_correlation.tolerance, "<="),
            ),
        )
    elif mid_band_width.status != "matched" and abs(mid_band_width.output_delta_to_reference) > 0.05:
        message = (
            f"{mid_band_width.label} moved {abs(mid_band_width.movement_toward_reference):.2f} {mid_band_width.unit} toward the reference; "
            f"residual gap is {abs(mid_band_width.output_delta_to_reference):.2f} {mid_band_width.unit}."
        )
        finding = ReferenceFinding(
            code="REFERENCE_STEREO_WIDTH_DEVIATION",
            severity="warning",
            message=message,
            evidence=(
                _evidence(mid_band_width.metric, mid_band_width.output_delta_to_reference, mid_band_width.unit, mid_band_width.tolerance, ">"),
            ),
        )
    elif low_band_width.output_value <= targets.target_low_band_width + 0.1 and low_band_width.output_value <= low_band_width.reference_value:
        message = (
            f"{low_band_width.label} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} narrower than the reference to preserve mono safety."
        )
        finding = ReferenceFinding(
            code="REFERENCE_STEREO_CONSTRAINED_BY_MONO_SAFETY",
            severity="informational",
            message=message,
            evidence=(
                _evidence(low_band_width.metric, low_band_width.output_value, low_band_width.unit, targets.target_low_band_width + 0.1, "<="),
            ),
        )
    else:
        message = (
            f"{phase_correlation.label} moved {abs(phase_correlation.movement_toward_reference):.2f} {phase_correlation.unit} toward the reference; "
            f"{low_band_width.label.lower()} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} from it."
        )
        finding = ReferenceFinding(
            code="REFERENCE_STEREO_DEVIATION",
            severity="warning",
            message=message,
            evidence=(
                _evidence(phase_correlation.metric, phase_correlation.output_delta_to_reference, phase_correlation.unit, phase_correlation.tolerance, ">"),
                _evidence(low_band_width.metric, low_band_width.output_delta_to_reference, low_band_width.unit, low_band_width.tolerance, ">"),
            ),
        )
    return ReferenceSummaryBullet("stereo", message), finding


def _summary_for_dynamics(
    crest_factor: ReferenceMetricComparison,
    output_analysis: AnalysisResult,
    targets: TargetValues,
) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    target_floor = targets.target_crest_factor - 1.0
    if crest_factor.status == "matched":
        message = f"{crest_factor.label} matched the reference within tolerance."
        finding = ReferenceFinding(
            code="REFERENCE_DYNAMICS_MATCHED",
            severity="informational",
            message=message,
            evidence=(
                _evidence(crest_factor.metric, crest_factor.output_delta_to_reference, crest_factor.unit, crest_factor.tolerance, "<="),
            ),
        )
    elif output_analysis.loudness.crest_factor_db >= target_floor and crest_factor.output_value > crest_factor.reference_value:
        message = (
            f"{crest_factor.label} remains {abs(crest_factor.output_delta_to_reference):.2f} {crest_factor.unit} above the reference, consistent with the dynamics-preservation target."
        )
        finding = ReferenceFinding(
            code="REFERENCE_DYNAMICS_RETAINED",
            severity="informational",
            message=message,
            evidence=(
                _evidence(crest_factor.metric, crest_factor.output_value, crest_factor.unit, target_floor, ">="),
            ),
        )
    else:
        message = (
            f"{crest_factor.label} moved {abs(crest_factor.movement_toward_reference):.2f} {crest_factor.unit} toward the reference; "
            f"{abs(crest_factor.output_delta_to_reference):.2f} {crest_factor.unit} remain."
        )
        finding = ReferenceFinding(
            code="REFERENCE_DYNAMICS_DEVIATION",
            severity="warning",
            message=message,
            evidence=(
                _evidence(crest_factor.metric, crest_factor.output_delta_to_reference, crest_factor.unit, crest_factor.tolerance, ">"),
            ),
        )
    return ReferenceSummaryBullet("dynamics", message), finding


def build_reference_assessment(
    config: MasteringConfig,
    targets: TargetValues,
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
) -> ReferenceAssessment:
    comparisons = tuple(
        _build_comparison(metric, input_analysis, reference_analysis, output_analysis)
        for metric in (
            "integrated_lufs",
            "true_peak_dbfs",
            "spectral_tilt_db_per_decade",
            "low_mid_balance_db",
            "high_mid_balance_db",
            "sub_energy_ratio",
            "bass_energy_ratio",
            "low_band_width",
            "mid_band_width",
            "phase_correlation",
            "crest_factor_db",
        )
    )
    comparison_map = _metric_lookup(comparisons)

    loudness_summary, loudness_finding = _summary_for_loudness(comparison_map["integrated_lufs"], output_analysis, config)
    tonal_summary, tonal_finding = _summary_for_tonal(
        comparison_map["spectral_tilt_db_per_decade"],
        comparison_map["low_mid_balance_db"],
        comparison_map["high_mid_balance_db"],
    )
    low_end_summary, low_end_finding = _summary_for_low_end(
        comparison_map["sub_energy_ratio"],
        comparison_map["bass_energy_ratio"],
        comparison_map["low_band_width"],
        targets,
    )
    stereo_summary, stereo_finding = _summary_for_stereo(
        comparison_map["mid_band_width"],
        comparison_map["phase_correlation"],
        comparison_map["low_band_width"],
        targets,
    )
    dynamics_summary, dynamics_finding = _summary_for_dynamics(comparison_map["crest_factor_db"], output_analysis, targets)

    findings = (loudness_finding, tonal_finding, low_end_finding, stereo_finding, dynamics_finding)
    summary = (loudness_summary, tonal_summary, low_end_summary, stereo_summary, dynamics_summary)

    if any("CONSTRAINED" in finding.code for finding in findings):
        outcome = "constrained"
    elif any(comparison.status == "moved_away" for comparison in comparisons):
        outcome = "deviated"
    elif all(comparison.status == "matched" for comparison in comparisons):
        outcome = "matched"
    else:
        outcome = "partial"

    return ReferenceAssessment(
        schema_version=REFERENCE_ASSESSMENT_SCHEMA_VERSION,
        reference_path=config.reference_path,
        reference_analysis_schema_version=reference_analysis.schema_version,
        reference_analysis_identity=reference_analysis.identity,
        comparisons=comparisons,
        findings=findings,
        summary=summary,
        outcome=outcome,
    )


def render_reference_markdown_section(assessment: ReferenceAssessment) -> list[str]:
    lines = [
        "## Reference",
        "",
        f"- Outcome: `{assessment.outcome}`",
        f"- Reference path: `{assessment.reference_path or 'not provided'}`",
        f"- Reference analysis schema: `{assessment.reference_analysis_schema_version}`",
        (
            "- Reference identity: "
            f"`{assessment.reference_analysis_identity.analyzer_version}` / "
            f"{assessment.reference_analysis_identity.sample_rate_hz} Hz / "
            f"{assessment.reference_analysis_identity.channels} ch / "
            f"{assessment.reference_analysis_identity.samples} samples / "
            f"{assessment.reference_analysis_identity.duration_s:.2f} s / "
            f"{assessment.reference_analysis_identity.role} / "
            f"{assessment.reference_analysis_identity.source_path or 'no source path'}"
        ),
        "",
        "## Reference Comparison",
        "",
    ]

    for comparison in assessment.comparisons:
        lines.append(
            f"- {comparison.label}: input `{comparison.input_value:.2f} {comparison.unit}`, reference `{comparison.reference_value:.2f} {comparison.unit}`, output `{comparison.output_value:.2f} {comparison.unit}`, "
            f"output moved `{abs(comparison.movement_toward_reference):.2f} {comparison.unit}` toward the reference, residual gap `{abs(comparison.output_delta_to_reference):.2f} {comparison.unit}`."
        )

    lines.extend(["", "## Reference Summary", ""])
    for bullet in assessment.summary:
        lines.append(f"- {bullet.message}")

    lines.extend(["", "## Reference Findings", ""])
    for finding in assessment.findings:
        lines.append(f"- `{finding.severity}` `{finding.code}`: {finding.message}")

    lines.append("")
    return lines
