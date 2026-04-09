from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognis.analysis.features import AnalysisResult
from cognis.config import CeilingMode, MasteringConfig
from cognis.optimizer.targets import TargetValues
from cognis.reports.reference import ReferenceAssessment, build_reference_assessment, render_reference_markdown_section


REPORT_SCHEMA_VERSION = "report_schema_v3"

SEVERITY_PASS = "pass"
SEVERITY_INFO = "informational"
SEVERITY_WARNING = "warning"
SEVERITY_FAIL = "fail"

STATUS_ORDER = {
    SEVERITY_FAIL: 0,
    SEVERITY_WARNING: 1,
    SEVERITY_INFO: 2,
    SEVERITY_PASS: 3,
}


@dataclass(frozen=True)
class RequestedTarget:
    mode: str
    target_loudness_lufs: float
    ceiling_mode: str
    ceiling_dbfs: float
    codec_safe_requested: bool
    reference_path: str | None
    bass_preservation: float
    stereo_width: float
    dynamics_preservation: float
    brightness: float
    fir_backend: str


@dataclass(frozen=True)
class AchievedTarget:
    integrated_lufs: float
    short_term_max_lufs: float
    momentary_max_lufs: float
    sample_peak_dbfs: float
    true_peak_dbfs: float
    phase_correlation: float
    mono_compatibility_score: float
    low_band_width: float
    spectral_tilt_db_per_decade: float
    crest_factor_db: float
    limiter_stress_estimate: float
    codec_risk_estimate: float
    clipping_risk_estimate: float


@dataclass(frozen=True)
class TargetDelta:
    loudness_delta_lu: float
    true_peak_margin_db: float
    sample_peak_margin_db: float
    codec_safety_margin_db: float
    phase_correlation_delta: float
    low_band_width_delta: float
    spectral_tilt_delta: float
    crest_factor_delta_db: float


@dataclass(frozen=True)
class QCEvidence:
    metric: str
    value: float
    unit: str
    threshold: float | None = None
    comparator: str | None = None


@dataclass(frozen=True)
class QCFinding:
    code: str
    severity: str
    message: str
    evidence: tuple[QCEvidence, ...]


@dataclass(frozen=True)
class ChangeBullet:
    category: str
    message: str


@dataclass(frozen=True)
class ReportResult:
    schema_version: str
    analysis_schema_version: str
    recipe_schema_version: str
    requested: RequestedTarget
    achieved: AchievedTarget
    delta: TargetDelta
    findings: tuple[QCFinding, ...]
    summary: tuple[ChangeBullet, ...]
    overall_status: str
    reference_assessment: ReferenceAssessment | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def status(self) -> str:
        return self.overall_status

    @property
    def reference_status(self) -> str | None:
        if self.reference_assessment is None:
            return None
        return self.reference_assessment.outcome

    @property
    def integrated_loudness(self) -> float:
        return self.achieved.integrated_lufs

    @property
    def short_term_loudness(self) -> float:
        return self.achieved.short_term_max_lufs

    @property
    def sample_peak(self) -> float:
        return self.achieved.sample_peak_dbfs

    @property
    def true_peak(self) -> float:
        return self.achieved.true_peak_dbfs

    @property
    def spectral_tilt(self) -> float:
        return self.achieved.spectral_tilt_db_per_decade

    @property
    def phase_correlation(self) -> float:
        return self.achieved.phase_correlation

    @property
    def target_deltas(self) -> tuple[TargetDelta, ...]:
        return (self.delta,)


def _finding(code: str, severity: str, message: str, *evidence: QCEvidence) -> QCFinding:
    return QCFinding(code=code, severity=severity, message=message, evidence=tuple(evidence))


def _evidence(metric: str, value: float, unit: str, threshold: float | None = None, comparator: str | None = None) -> QCEvidence:
    return QCEvidence(metric=metric, value=float(value), unit=unit, threshold=threshold, comparator=comparator)


def _build_requested(config: MasteringConfig) -> RequestedTarget:
    return RequestedTarget(
        mode=config.mode.value,
        target_loudness_lufs=config.target_loudness,
        ceiling_mode=config.ceiling_mode.value,
        ceiling_dbfs=config.ceiling_db,
        codec_safe_requested=config.ceiling_mode == CeilingMode.CODEC_SAFE,
        reference_path=config.reference_path,
        bass_preservation=config.bass_preservation,
        stereo_width=config.stereo_width,
        dynamics_preservation=config.dynamics_preservation,
        brightness=config.brightness,
        fir_backend=config.fir_backend,
    )


def _build_achieved(analysis: AnalysisResult) -> AchievedTarget:
    return AchievedTarget(
        integrated_lufs=analysis.loudness.integrated_lufs,
        short_term_max_lufs=analysis.loudness.short_term_max_lufs,
        momentary_max_lufs=analysis.loudness.momentary_max_lufs,
        sample_peak_dbfs=analysis.loudness.sample_peak_dbfs,
        true_peak_dbfs=analysis.loudness.true_peak_dbfs,
        phase_correlation=analysis.stereo.phase_correlation,
        mono_compatibility_score=analysis.stereo.mono_compatibility_score,
        low_band_width=analysis.stereo.low_band_width,
        spectral_tilt_db_per_decade=analysis.tonal.spectral_tilt_db_per_decade,
        crest_factor_db=analysis.loudness.crest_factor_db,
        limiter_stress_estimate=analysis.risks.limiter_stress_estimate,
        codec_risk_estimate=analysis.risks.codec_risk_estimate,
        clipping_risk_estimate=analysis.risks.clipping_risk_estimate,
    )


def _build_delta(
    config: MasteringConfig,
    targets: TargetValues,
    input_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
) -> TargetDelta:
    codec_reference_ceiling = min(config.ceiling_db, -1.0) if config.ceiling_mode == CeilingMode.CODEC_SAFE else -1.0
    return TargetDelta(
        loudness_delta_lu=output_analysis.loudness.integrated_lufs - targets.target_loudness,
        true_peak_margin_db=config.ceiling_db - output_analysis.loudness.true_peak_dbfs,
        sample_peak_margin_db=0.0 - output_analysis.loudness.sample_peak_dbfs,
        codec_safety_margin_db=codec_reference_ceiling - output_analysis.loudness.true_peak_dbfs,
        phase_correlation_delta=output_analysis.stereo.phase_correlation - input_analysis.stereo.phase_correlation,
        low_band_width_delta=output_analysis.stereo.low_band_width - targets.target_low_band_width,
        spectral_tilt_delta=output_analysis.tonal.spectral_tilt_db_per_decade - targets.target_tilt,
        crest_factor_delta_db=output_analysis.loudness.crest_factor_db - targets.target_crest_factor,
    )


def _evaluate_findings(
    config: MasteringConfig,
    targets: TargetValues,
    input_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
    delta: TargetDelta,
) -> tuple[QCFinding, ...]:
    findings: list[QCFinding] = []

    abs_loudness_delta = abs(delta.loudness_delta_lu)
    if abs_loudness_delta > 1.0:
        findings.append(
            _finding(
                "LOUDNESS_OUTSIDE_HARD_BAND",
                SEVERITY_FAIL,
                "Integrated loudness misses the requested target by more than 1 LU.",
                _evidence("loudness_delta_lu", delta.loudness_delta_lu, "LU", 1.0, "<="),
                _evidence("integrated_lufs", output_analysis.loudness.integrated_lufs, "LUFS", config.target_loudness, "=="),
            )
        )
    elif abs_loudness_delta > 0.5:
        findings.append(
            _finding(
                "LOUDNESS_OUTSIDE_TARGET_TOLERANCE",
                SEVERITY_WARNING,
                "Integrated loudness is outside the preferred target tolerance.",
                _evidence("loudness_delta_lu", delta.loudness_delta_lu, "LU", 0.5, "<="),
            )
        )
    else:
        findings.append(
            _finding(
                "TARGET_LOUDNESS_HIT",
                SEVERITY_INFO,
                "Integrated loudness landed within the requested tolerance.",
                _evidence("loudness_delta_lu", delta.loudness_delta_lu, "LU", 0.5, "<="),
            )
        )

    if delta.true_peak_margin_db < 0.0:
        findings.append(
            _finding(
                "TRUE_PEAK_ABOVE_CEILING",
                SEVERITY_FAIL,
                "True peak exceeds the requested ceiling.",
                _evidence("true_peak_dbfs", output_analysis.loudness.true_peak_dbfs, "dBFS", config.ceiling_db, "<="),
                _evidence("true_peak_margin_db", delta.true_peak_margin_db, "dB", 0.0, ">="),
            )
        )
    elif delta.true_peak_margin_db < 0.3:
        findings.append(
            _finding(
                "TRUE_PEAK_TOO_CLOSE_TO_CEILING",
                SEVERITY_WARNING,
                "True peak is technically under ceiling but with very little safety margin.",
                _evidence("true_peak_margin_db", delta.true_peak_margin_db, "dB", 0.3, ">="),
            )
        )
    else:
        findings.append(
            _finding(
                "TRUE_PEAK_WITHIN_MARGIN",
                SEVERITY_INFO,
                "True peak stays within the requested ceiling margin.",
                _evidence("true_peak_margin_db", delta.true_peak_margin_db, "dB", 0.3, ">="),
            )
        )

    if output_analysis.loudness.sample_peak_dbfs > 0.0:
        findings.append(
            _finding(
                "SAMPLE_PEAK_ABOVE_0_DBFS",
                SEVERITY_FAIL,
                "Sample peak exceeds digital full scale.",
                _evidence("sample_peak_dbfs", output_analysis.loudness.sample_peak_dbfs, "dBFS", 0.0, "<="),
            )
        )
    elif output_analysis.loudness.sample_peak_dbfs > -0.2:
        findings.append(
            _finding(
                "SAMPLE_PEAK_TOO_CLOSE_TO_0_DBFS",
                SEVERITY_WARNING,
                "Sample peak is very close to digital full scale.",
                _evidence("sample_peak_dbfs", output_analysis.loudness.sample_peak_dbfs, "dBFS", -0.2, "<="),
            )
        )

    if output_analysis.stereo.phase_correlation < 0.0:
        findings.append(
            _finding(
                "PHASE_CORRELATION_CRITICAL",
                SEVERITY_FAIL,
                "Phase correlation is negative and indicates serious mono-compatibility risk.",
                _evidence("phase_correlation", output_analysis.stereo.phase_correlation, "ratio", 0.0, ">="),
            )
        )
    elif output_analysis.stereo.phase_correlation < 0.2:
        findings.append(
            _finding(
                "PHASE_CORRELATION_LOW",
                SEVERITY_WARNING,
                "Phase correlation is low and may reduce mono compatibility.",
                _evidence("phase_correlation", output_analysis.stereo.phase_correlation, "ratio", 0.2, ">="),
            )
        )
    else:
        findings.append(
            _finding(
                "MONO_COMPATIBILITY_OK",
                SEVERITY_INFO,
                "Phase correlation remains within the acceptable mono-compatibility range.",
                _evidence("phase_correlation", output_analysis.stereo.phase_correlation, "ratio", 0.2, ">="),
            )
        )

    if output_analysis.stereo.low_band_width > targets.target_low_band_width + 0.2:
        findings.append(
            _finding(
                "LOW_BAND_WIDTH_CRITICAL",
                SEVERITY_FAIL,
                "Low-band stereo width materially exceeds the requested mono-safety target.",
                _evidence("low_band_width", output_analysis.stereo.low_band_width, "ratio", targets.target_low_band_width + 0.2, "<="),
            )
        )
    elif output_analysis.stereo.low_band_width > targets.target_low_band_width + 0.1:
        findings.append(
            _finding(
                "LOW_BAND_WIDTH_HIGH",
                SEVERITY_WARNING,
                "Low-band stereo width is wider than requested and may reduce mono safety.",
                _evidence("low_band_width", output_analysis.stereo.low_band_width, "ratio", targets.target_low_band_width + 0.1, "<="),
            )
        )
    else:
        findings.append(
            _finding(
                "SUB_WIDTH_CONTROLLED",
                SEVERITY_INFO,
                "Low-band stereo width stays within the requested tolerance.",
                _evidence("low_band_width", output_analysis.stereo.low_band_width, "ratio", targets.target_low_band_width + 0.1, "<="),
            )
        )

    if output_analysis.risks.clipping_risk_estimate >= 0.85:
        findings.append(
            _finding(
                "CLIPPING_RISK_CRITICAL",
                SEVERITY_FAIL,
                "Measured peak behavior suggests a critical clipping risk.",
                _evidence("clipping_risk_estimate", output_analysis.risks.clipping_risk_estimate, "score", 0.85, "<"),
            )
        )

    if output_analysis.risks.limiter_stress_estimate >= 0.90:
        findings.append(
            _finding(
                "LIMITER_STRESS_CRITICAL",
                SEVERITY_FAIL,
                "Limiter stress proxy is critically high and likely to sound strained.",
                _evidence("limiter_stress_estimate", output_analysis.risks.limiter_stress_estimate, "score", 0.90, "<"),
                _evidence("crest_factor_db", output_analysis.loudness.crest_factor_db, "dB"),
            )
        )
    elif output_analysis.risks.limiter_stress_estimate >= 0.75:
        findings.append(
            _finding(
                "LIMITER_STRESS_HIGH",
                SEVERITY_WARNING,
                "Limiter stress proxy is elevated.",
                _evidence("limiter_stress_estimate", output_analysis.risks.limiter_stress_estimate, "score", 0.75, "<"),
            )
        )

    if output_analysis.risks.codec_risk_estimate >= 0.75:
        findings.append(
            _finding(
                "CODEC_RISK_HIGH",
                SEVERITY_WARNING,
                "Codec-risk proxy is elevated based on peak margin and top-end energy.",
                _evidence("codec_risk_estimate", output_analysis.risks.codec_risk_estimate, "score", 0.75, "<"),
                _evidence("codec_safety_margin_db", delta.codec_safety_margin_db, "dB", 0.0, ">="),
            )
        )
    else:
        findings.append(
            _finding(
                "CODEC_SAFE_MARGIN_OK",
                SEVERITY_INFO,
                "Codec-safety proxy remains within the preferred range.",
                _evidence("codec_risk_estimate", output_analysis.risks.codec_risk_estimate, "score", 0.75, "<"),
            )
        )

    if abs(output_analysis.tonal.low_mid_balance_db) > 12.0 or abs(output_analysis.tonal.high_mid_balance_db) > 12.0:
        findings.append(
            _finding(
                "TONAL_BALANCE_EXTREME",
                SEVERITY_WARNING,
                "Tonal balance landed in an extreme region and should be reviewed.",
                _evidence("low_mid_balance_db", output_analysis.tonal.low_mid_balance_db, "dB", 12.0, "<="),
                _evidence("high_mid_balance_db", output_analysis.tonal.high_mid_balance_db, "dB", 12.0, "<="),
            )
        )
    else:
        findings.append(
            _finding(
                "TONAL_BALANCE_WITHIN_EXPECTED_RANGE",
                SEVERITY_INFO,
                "Measured tonal balance stays within the expected operating range.",
                _evidence("low_mid_balance_db", output_analysis.tonal.low_mid_balance_db, "dB", 12.0, "<="),
                _evidence("high_mid_balance_db", output_analysis.tonal.high_mid_balance_db, "dB", 12.0, "<="),
            )
        )

    if output_analysis.loudness.momentary_max_lufs - output_analysis.loudness.integrated_lufs > 8.0:
        findings.append(
            _finding(
                "MOMENTARY_LOUDNESS_SPIKE",
                SEVERITY_WARNING,
                "Momentary loudness rises sharply above the integrated program level.",
                _evidence(
                    "momentary_minus_integrated_lu",
                    output_analysis.loudness.momentary_max_lufs - output_analysis.loudness.integrated_lufs,
                    "LU",
                    8.0,
                    "<=",
                ),
            )
        )

    if output_analysis.loudness.crest_factor_db < targets.target_crest_factor - 1.0:
        findings.append(
            _finding(
                "DYNAMICS_COLLAPSE_RISK",
                SEVERITY_WARNING,
                "Crest factor fell materially below the requested dynamics-preservation target.",
                _evidence("crest_factor_db", output_analysis.loudness.crest_factor_db, "dB", targets.target_crest_factor - 1.0, ">="),
            )
        )
    else:
        findings.append(
            _finding(
                "DYNAMICS_PRESERVED_WITHIN_TOLERANCE",
                SEVERITY_INFO,
                "Crest factor remains within the requested dynamics tolerance.",
                _evidence("crest_factor_db", output_analysis.loudness.crest_factor_db, "dB", targets.target_crest_factor - 1.0, ">="),
            )
        )

    findings.sort(key=lambda item: (STATUS_ORDER[item.severity], item.code))
    return tuple(findings)


def _format_change(value: float, unit: str, precision: int = 2) -> str:
    return f"{abs(value):.{precision}f} {unit}"


def _build_summary(
    input_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
    delta: TargetDelta,
    overall_status: str,
    findings: tuple[QCFinding, ...],
    reference_assessment: ReferenceAssessment | None = None,
) -> tuple[ChangeBullet, ...]:
    bullets: list[ChangeBullet] = []
    finding_codes = {finding.code for finding in findings}
    has_fail = any(finding.severity == SEVERITY_FAIL for finding in findings)
    has_dynamics_risk = "DYNAMICS_COLLAPSE_RISK" in finding_codes

    loudness_change = output_analysis.loudness.integrated_lufs - input_analysis.loudness.integrated_lufs
    if abs(loudness_change) >= 0.5:
        direction = "Increased" if loudness_change > 0.0 else "Reduced"
        bullets.append(ChangeBullet("loudness", f"{direction} integrated loudness by {_format_change(loudness_change, 'LUFS')}."))

    high_mid_change = output_analysis.tonal.high_mid_balance_db - input_analysis.tonal.high_mid_balance_db
    if high_mid_change <= -1.0:
        bullets.append(ChangeBullet("tonal", f"Reduced upper-mid / treble balance by {_format_change(high_mid_change, 'dB')}."))
    elif high_mid_change >= 1.0:
        bullets.append(ChangeBullet("tonal", f"Increased upper-mid / treble balance by {_format_change(high_mid_change, 'dB')}."))

    low_width_change = output_analysis.stereo.low_band_width - input_analysis.stereo.low_band_width
    if low_width_change <= -0.05:
        bullets.append(ChangeBullet("stereo", f"Narrowed sub / low-band stereo width by {_format_change(low_width_change, 'ratio')}."))
    elif low_width_change >= 0.05:
        bullets.append(ChangeBullet("stereo", f"Widened sub / low-band stereo width by {_format_change(low_width_change, 'ratio')}."))

    crest_change = output_analysis.loudness.crest_factor_db - input_analysis.loudness.crest_factor_db
    if has_dynamics_risk:
        bullets.append(ChangeBullet("dynamics", "Measured crest factor fell below the requested dynamics-preservation tolerance."))
    elif crest_change <= -1.0:
        bullets.append(ChangeBullet("dynamics", f"Reduced crest factor by {_format_change(crest_change, 'dB')}, increasing density."))
    elif abs(crest_change) < 1.0:
        bullets.append(ChangeBullet("dynamics", "Preserved punch within the current crest-factor tolerance."))

    if has_fail:
        bullets.append(ChangeBullet("safety", "Render failed QC and is not release-ready under the current measured constraints."))
    elif delta.true_peak_margin_db < 0.0:
        bullets.append(ChangeBullet("safety", f"Failed true-peak safety by {_format_change(delta.true_peak_margin_db, 'dB')}."))
    elif overall_status == SEVERITY_WARNING:
        bullets.append(ChangeBullet("safety", "Render passed with warnings that should be reviewed before release."))
    else:
        bullets.append(ChangeBullet("safety", "Render stayed within the measured release-safety constraints."))

    if reference_assessment is not None:
        progress = reference_assessment.summary
        for bullet in progress:
            bullets.append(ChangeBullet(f"reference:{bullet.category}", bullet.message))

    return tuple(bullets)


def _overall_status(findings: tuple[QCFinding, ...]) -> str:
    if any(f.severity == SEVERITY_FAIL for f in findings):
        return SEVERITY_FAIL
    if any(f.severity == SEVERITY_WARNING for f in findings):
        return SEVERITY_WARNING
    return SEVERITY_PASS


def build_report(
    config: MasteringConfig,
    recipe_schema_version: str,
    targets: TargetValues,
    input_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None = None,
    optimizer_trace: Any | None = None,
) -> ReportResult:
    requested = _build_requested(config)
    achieved = _build_achieved(output_analysis)
    delta = _build_delta(config, targets, input_analysis, output_analysis)
    findings = _evaluate_findings(config, targets, input_analysis, output_analysis, delta)
    overall_status = _overall_status(findings)
    reference_assessment = None
    if reference_analysis is not None:
        reference_assessment = build_reference_assessment(
            config,
            targets,
            input_analysis,
            reference_analysis,
            output_analysis,
            optimizer_trace=optimizer_trace,
        )
    summary = _build_summary(input_analysis, output_analysis, delta, overall_status, findings, reference_assessment)

    return ReportResult(
        schema_version=REPORT_SCHEMA_VERSION,
        analysis_schema_version=output_analysis.schema_version,
        recipe_schema_version=recipe_schema_version,
        requested=requested,
        achieved=achieved,
        delta=delta,
        findings=findings,
        summary=summary,
        overall_status=overall_status,
        reference_assessment=reference_assessment,
    )


def render_report_markdown(report: ReportResult) -> str:
    lines = [
        "# COGNIS Render Report",
        "",
        "## Requested",
        "",
        f"- Mode: `{report.requested.mode}`",
        f"- Overall status: `{report.overall_status}`",
        f"- Target loudness: `{report.requested.target_loudness_lufs:.2f} LUFS`",
        f"- Ceiling: `{report.requested.ceiling_dbfs:.2f} dBFS` (`{report.requested.ceiling_mode}`)",
        f"- Codec-safe requested: `{str(report.requested.codec_safe_requested).lower()}`",
        f"- Stereo width target: `{report.requested.stereo_width:.2f}`",
        f"- Dynamics preservation target: `{report.requested.dynamics_preservation:.2f}`",
        f"- Brightness target: `{report.requested.brightness:.2f}`",
        "",
        "## Achieved",
        "",
        f"- Achieved loudness: `{report.achieved.integrated_lufs:.2f} LUFS`",
        f"- True peak: `{report.achieved.true_peak_dbfs:.2f} dBFS`",
        f"- Ceiling margin: `{report.delta.true_peak_margin_db:.2f} dB`",
        f"- Low-band width: `{report.achieved.low_band_width:.2f}`",
        f"- Phase correlation: `{report.achieved.phase_correlation:.2f}`",
        "",
        "## What Changed",
        "",
    ]
    for bullet in report.summary:
        lines.append(f"- {bullet.message}")

    if report.reference_assessment is not None:
        lines.extend([""] + render_reference_markdown_section(report.reference_assessment))

    lines.extend(["", "## QC Findings", ""])
    for finding in report.findings:
        lines.append(f"- `{finding.severity}` `{finding.code}`: {finding.message}")

    lines.append("")
    return "\n".join(lines)


QCReport = ReportResult


def generate_qc_report(
    input_analysis: AnalysisResult,
    output_analysis: AnalysisResult,
    config: MasteringConfig,
    targets: TargetValues,
    recipe_schema_version: str = "recipe_v2",
    reference_analysis: AnalysisResult | None = None,
    optimizer_trace: Any | None = None,
) -> QCReport:
    return build_report(
        config,
        recipe_schema_version,
        targets,
        input_analysis,
        output_analysis,
        reference_analysis,
        optimizer_trace=optimizer_trace,
    )


def format_report_markdown(report: QCReport) -> str:
    return render_report_markdown(report)
