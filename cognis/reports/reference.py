from __future__ import annotations

from dataclasses import dataclass

from cognis.analysis.features import AnalysisIdentity, AnalysisResult
from cognis.config import MasteringConfig
from cognis.optimizer.reference_targets import ReferenceAwareTargetPlan, build_reference_aware_targets
from cognis.optimizer.search import SearchTrace
from cognis.optimizer.targets import TargetValues


REFERENCE_ASSESSMENT_SCHEMA_VERSION = "reference_assessment_schema_v2"
REFERENCE_ATTRIBUTION_SCHEMA_VERSION = "reference_attribution_schema_v1"

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

PLAN_METRIC_UNITS: dict[str, str] = {
    "integrated_lufs": "LUFS",
    "spectral_tilt_db_per_decade": "dB/decade",
    "low_mid_balance_db": "dB",
    "high_mid_balance_db": "dB",
    "low_band_centroid_hz": "Hz",
    "mid_band_width": "ratio",
    "low_band_width": "ratio",
    "phase_correlation": "ratio",
    "crest_factor_db": "dB",
}

REFERENCE_TERM_CODES: dict[str, str] = {
    "integrated_lufs": "reference_integrated_lufs",
    "spectral_tilt_db_per_decade": "reference_spectral_tilt_db_per_decade",
    "low_band_width": "reference_low_band_width",
    "crest_factor_db": "reference_crest_factor_db",
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
class ReferenceAttributionEntry:
    category: str
    metric: str
    attribution_level: str
    decision: str
    tradeoff: str
    required_change: str
    explanation: str
    evidence: tuple[ReferenceEvidence, ...]


@dataclass(frozen=True)
class ReferenceAttribution:
    schema_version: str
    available: bool
    availability_reason: str | None
    entries: tuple[ReferenceAttributionEntry, ...]


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
    attribution: ReferenceAttribution | None
    outcome: str


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


def _selected_evaluation(optimizer_trace: SearchTrace | None):
    if optimizer_trace is None:
        return None
    for evaluation in optimizer_trace.evaluations:
        if evaluation.index == optimizer_trace.best_index:
            return evaluation
    return None


def _term_lookup(evaluation) -> dict[str, object]:
    if evaluation is None:
        return {}
    return {term.name: term for term in evaluation.attribution.terms}


def _rule_details(term_name: str | None) -> tuple[str, str, str]:
    mapping = {
        "true_peak_ceiling": ("true_peak_safety", "True-peak safety over a closer reference match.", "Raise `ceiling_db` or lower the requested loudness target."),
        "low_band_width_cap": ("low_band_mono_safety", "Low-band mono safety over a wider low-end match.", "Relax bass preservation or allow a wider low-band cap."),
        "crest_factor_floor": ("dynamics_preservation_floor", "Dynamics preservation over a denser reference match.", "Reduce `dynamics_preservation` or accept a lower crest-factor floor."),
        "integrated_lufs_target": ("configured_loudness_target", "Configured loudness intent over a closer reference loudness match.", "Raise `target_loudness` or rebalance loudness weighting."),
        "spectral_tilt_target": ("brightness_budget", "Conservative tonal correction over a closer reference tilt match.", "Increase `brightness` or relax tonal weighting."),
    }
    return mapping.get(term_name, ("objective_tradeoff", "Lower total objective score over a closer single-metric match.", "Relax the governing target or expand the optimizer search space."))


def _search_tradeoff(optimizer_trace: SearchTrace | None, metric: str) -> dict[str, object] | None:
    selected = _selected_evaluation(optimizer_trace)
    if selected is None:
        return None
    selected_terms = _term_lookup(selected)
    reference_term_name = REFERENCE_TERM_CODES.get(metric)
    if reference_term_name is None:
        return None
    selected_reference_term = selected_terms.get(reference_term_name)
    if selected_reference_term is None or selected_reference_term.difference is None:
        return None

    better_candidates: list[dict[str, object]] = []
    for evaluation in optimizer_trace.evaluations:
        if evaluation.index == selected.index:
            continue
        candidate_terms = _term_lookup(evaluation)
        candidate_reference_term = candidate_terms.get(reference_term_name)
        if candidate_reference_term is None or candidate_reference_term.difference is None:
            continue
        if abs(candidate_reference_term.difference) + 1e-9 >= abs(selected_reference_term.difference):
            continue

        dominant_name = None
        dominant_diff = 0.0
        for name, candidate_term in candidate_terms.items():
            if name == reference_term_name:
                continue
            baseline_penalty = selected_terms[name].penalty if name in selected_terms else 0.0
            penalty_diff = candidate_term.penalty - baseline_penalty
            if penalty_diff > dominant_diff:
                dominant_diff = penalty_diff
                dominant_name = name

        better_candidates.append(
            {
                "evaluation": evaluation,
                "reference_term": candidate_reference_term,
                "gap": abs(candidate_reference_term.difference),
                "dominant_name": dominant_name,
            }
        )

    if not better_candidates:
        return {"type": "no_measured_justification", "selected_term": selected_reference_term}

    chosen = min(better_candidates, key=lambda item: (item["gap"], item["evaluation"].score))
    rule_name, tradeoff, required_change = _rule_details(chosen["dominant_name"])
    return {
        "type": "tradeoff",
        "selected_term": selected_reference_term,
        "alternative_term": chosen["reference_term"],
        "alternative_score": chosen["evaluation"].score,
        "blocking_rule": rule_name,
        "tradeoff": tradeoff,
        "required_change": required_change,
    }


def _entry(
    *,
    category: str,
    metric: str,
    attribution_level: str,
    decision: str,
    tradeoff: str,
    required_change: str,
    explanation: str,
    evidence: tuple[ReferenceEvidence, ...],
) -> ReferenceAttributionEntry:
    return ReferenceAttributionEntry(
        category=category,
        metric=metric,
        attribution_level=attribution_level,
        decision=decision,
        tradeoff=tradeoff,
        required_change=required_change,
        explanation=explanation,
        evidence=evidence,
    )


def _unavailable_entry(category: str, metric: str, comparison: ReferenceMetricComparison) -> ReferenceAttributionEntry:
    return _entry(
        category=category,
        metric=metric,
        attribution_level="unavailable",
        decision="no_supported_attribution",
        tradeoff="No supported causal claim is available.",
        required_change="Capture optimizer trace to attribute this residual honestly.",
        explanation=f"{comparison.label} still differs from the reference, but this run did not record enough optimizer evidence to attribute why.",
        evidence=(_evidence(f"{metric}_delta_to_reference", comparison.output_delta_to_reference, comparison.unit, comparison.tolerance, ">"),),
    )


def _tradeoff_entry(category: str, metric: str, comparison: ReferenceMetricComparison, tradeoff: dict[str, object] | None) -> ReferenceAttributionEntry:
    if tradeoff is None:
        return _unavailable_entry(category, metric, comparison)
    if tradeoff["type"] == "tradeoff":
        return _entry(
            category=category,
            metric=metric,
            attribution_level="exact",
            decision="lower_score_tradeoff_selected",
            tradeoff=str(tradeoff["tradeoff"]),
            required_change=str(tradeoff["required_change"]),
            explanation=f"A candidate closer to the reference {comparison.label.lower()} existed in the bounded grid, but it scored worse on the full objective so the lower-score tradeoff was selected.",
            evidence=(
                _evidence(f"selected_{metric}_delta_to_reference", tradeoff["selected_term"].difference, comparison.unit),
                _evidence(f"alternative_{metric}_delta_to_reference", tradeoff["alternative_term"].difference, comparison.unit),
                _evidence("alternative_total_score", tradeoff["alternative_score"], "score"),
            ),
        )
    return _entry(
        category="no_measured_justification",
        metric=metric,
        attribution_level="exact",
        decision="no_better_evaluated_candidate",
        tradeoff="No evaluated candidate improved this metric without increasing total objective score.",
        required_change="Expand the optimizer search space or rebalance the objective if a closer match is worth the cost.",
        explanation=f"Within the bounded deterministic grid search, no evaluated candidate landed closer to the reference {comparison.label.lower()} than the selected result.",
        evidence=(_evidence(f"selected_{metric}_delta_to_reference", tradeoff["selected_term"].difference, comparison.unit),),
    )


def _build_reference_attribution(
    config: MasteringConfig,
    comparisons: tuple[ReferenceMetricComparison, ...],
    output_analysis: AnalysisResult,
    reference_target_plan: ReferenceAwareTargetPlan | None,
    optimizer_trace: SearchTrace | None,
) -> ReferenceAttribution:
    if reference_target_plan is None or not reference_target_plan.reference_available:
        return ReferenceAttribution(
            schema_version=REFERENCE_ATTRIBUTION_SCHEMA_VERSION,
            available=False,
            availability_reason="No reference-aware target plan was available, so no causal attribution was generated.",
            entries=(),
        )

    comparison_map = _metric_lookup(comparisons)
    plan = reference_target_plan
    entries: list[ReferenceAttributionEntry] = []

    loudness = comparison_map["integrated_lufs"]
    true_peak_margin_db = config.ceiling_db - output_analysis.loudness.true_peak_dbfs
    if loudness.status == "matched":
        entries.append(_entry(category="loudness_ceiling_constraint", metric="integrated_lufs", attribution_level="exact", decision="matched_to_reference", tradeoff="No tradeoff was required.", required_change="No change required.", explanation="Integrated loudness matched the reference within tolerance.", evidence=(_evidence("integrated_lufs_delta_to_reference", loudness.output_delta_to_reference, loudness.unit, loudness.tolerance, "<="),)))
    elif plan.loudness.safety_limited:
        entries.append(_entry(category="loudness_ceiling_constraint", metric="integrated_lufs", attribution_level="exact", decision="capped_by_loudness_baseline", tradeoff="Configured loudness target over a closer reference match.", required_change="Raise `target_loudness` or loosen the loudness baseline policy.", explanation=f"Integrated loudness remained conservative because the reference-aware target was capped at {plan.loudness.target_value:.2f} LUFS instead of chasing the {plan.loudness.reference_value:.2f} LUFS reference.", evidence=(_evidence("reference_loudness", plan.loudness.reference_value or 0.0, "LUFS"), _evidence("reference_target", plan.loudness.target_value, "LUFS"), _evidence("loudness_baseline", plan.loudness.baseline_target_value, "LUFS"))))
    elif true_peak_margin_db < 0.3:
        entries.append(_entry(category="loudness_ceiling_constraint", metric="integrated_lufs", attribution_level="inferred", decision="blocked_by_true_peak_margin", tradeoff="Safety over a closer loudness match.", required_change="Raise `ceiling_db` or accept less true-peak margin.", explanation=f"Integrated loudness stayed {abs(loudness.output_delta_to_reference):.2f} {loudness.unit} from the reference because only {true_peak_margin_db:.2f} dB of true-peak headroom remained.", evidence=(_evidence("true_peak_margin_db", true_peak_margin_db, "dB", 0.3, "<"), _evidence("integrated_lufs_delta_to_reference", loudness.output_delta_to_reference, loudness.unit, loudness.tolerance, ">"))))
    else:
        entries.append(_tradeoff_entry("loudness_ceiling_constraint", "integrated_lufs", loudness, _search_tradeoff(optimizer_trace, "integrated_lufs")))

    low_band = comparison_map["low_band_width"]
    if low_band.status == "matched":
        entries.append(_entry(category="mono_low_band_width_safety", metric="low_band_width", attribution_level="exact", decision="matched_to_reference", tradeoff="No tradeoff was required.", required_change="No change required.", explanation="Low-band width matched the reference within tolerance.", evidence=(_evidence("low_band_width_delta_to_reference", low_band.output_delta_to_reference, low_band.unit, low_band.tolerance, "<="),)))
    elif plan.low_band_width.safety_limited:
        entries.append(_entry(category="mono_low_band_width_safety", metric="low_band_width", attribution_level="exact", decision="held_by_mono_safety", tradeoff="Mono safety over a wider low-band match.", required_change="Relax bass preservation or allow a wider low-band cap.", explanation=f"Low-band width remained {abs(low_band.output_delta_to_reference):.2f} {low_band.unit} narrower than the reference because the mono-safety target was capped at {plan.low_band_width.target_value:.2f}.", evidence=(_evidence("reference_low_band_width", plan.low_band_width.reference_value or 0.0, low_band.unit), _evidence("reference_target_low_band_width", plan.low_band_width.target_value, low_band.unit), _evidence("low_band_width_baseline", plan.low_band_width.baseline_target_value, low_band.unit))))
    elif optimizer_trace is None and low_band.output_value <= low_band.reference_value:
        entries.append(_entry(category="mono_low_band_width_safety", metric="low_band_width", attribution_level="inferred", decision="likely_held_by_mono_safety", tradeoff="Likely mono safety over a wider low-band match.", required_change="Capture optimizer trace to confirm whether mono safety or another conservative rule dominated.", explanation=f"Low-band width remained {abs(low_band.output_delta_to_reference):.2f} {low_band.unit} narrower than the reference, which is consistent with the mono-safety baseline, but this run did not record optimizer trace evidence to prove the exact blocker.", evidence=(_evidence("low_band_width", low_band.output_value, low_band.unit, plan.low_band_width.baseline_target_value, "<="), _evidence("low_band_width_delta_to_reference", low_band.output_delta_to_reference, low_band.unit, low_band.tolerance, ">"),)))
    else:
        entries.append(_tradeoff_entry("mono_low_band_width_safety", "low_band_width", low_band, _search_tradeoff(optimizer_trace, "low_band_width")))

    crest = comparison_map["crest_factor_db"]
    if crest.status == "matched":
        entries.append(_entry(category="dynamics_preservation_constraint", metric="crest_factor_db", attribution_level="exact", decision="matched_to_reference", tradeoff="No tradeoff was required.", required_change="No change required.", explanation="Crest factor matched the reference within tolerance.", evidence=(_evidence("crest_factor_delta_to_reference", crest.output_delta_to_reference, crest.unit, crest.tolerance, "<="),)))
    elif plan.crest_factor.safety_limited:
        entries.append(_entry(category="dynamics_preservation_constraint", metric="crest_factor_db", attribution_level="exact", decision="held_by_dynamics_floor", tradeoff="Dynamics preservation over a denser reference match.", required_change="Reduce `dynamics_preservation` or accept a lower crest-factor floor.", explanation=f"Crest factor remained {abs(crest.output_delta_to_reference):.2f} {crest.unit} above the reference because the preservation target stayed at {plan.crest_factor.target_value:.2f} {crest.unit}.", evidence=(_evidence("reference_crest_factor", plan.crest_factor.reference_value or 0.0, crest.unit), _evidence("reference_target_crest_factor", plan.crest_factor.target_value, crest.unit), _evidence("crest_factor_baseline", plan.crest_factor.baseline_target_value, crest.unit))))
    elif output_analysis.loudness.crest_factor_db >= plan.crest_factor.baseline_target_value:
        entries.append(_entry(category="dynamics_preservation_constraint", metric="crest_factor_db", attribution_level="inferred", decision="held_above_preservation_floor", tradeoff="Dynamics preservation over a denser reference match.", required_change="Reduce `dynamics_preservation` or accept stronger compression.", explanation=f"Crest factor remained {abs(crest.output_delta_to_reference):.2f} {crest.unit} above the reference while staying above the preservation floor.", evidence=(_evidence("crest_factor_db", output_analysis.loudness.crest_factor_db, crest.unit, plan.crest_factor.baseline_target_value, ">="), _evidence("crest_factor_floor", plan.crest_factor.baseline_target_value, crest.unit))))
    else:
        entries.append(_tradeoff_entry("dynamics_preservation_constraint", "crest_factor_db", crest, _search_tradeoff(optimizer_trace, "crest_factor_db")))

    tilt = comparison_map["spectral_tilt_db_per_decade"]
    if tilt.status == "matched":
        entries.append(_entry(category="tonal_correction_limit", metric="spectral_tilt_db_per_decade", attribution_level="exact", decision="matched_to_reference", tradeoff="No tradeoff was required.", required_change="No change required.", explanation="Spectral tilt matched the reference within tolerance.", evidence=(_evidence("spectral_tilt_delta_to_reference", tilt.output_delta_to_reference, tilt.unit, tilt.tolerance, "<="),)))
    elif plan.spectral_tilt.safety_limited:
        unit = PLAN_METRIC_UNITS[plan.spectral_tilt.metric]
        entries.append(_entry(category="tonal_correction_limit", metric="spectral_tilt_db_per_decade", attribution_level="exact", decision="capped_by_brightness_budget", tradeoff="Conservative tonal correction over a closer reference tilt match.", required_change="Increase `brightness` or relax the tonal correction cap.", explanation=f"Tonal correction stayed conservative because the spectral-tilt target was capped at {plan.spectral_tilt.target_value:.2f} {unit} instead of the {plan.spectral_tilt.reference_value:.2f} {unit} reference.", evidence=(_evidence("reference_spectral_tilt", plan.spectral_tilt.reference_value or 0.0, unit), _evidence("reference_target_spectral_tilt", plan.spectral_tilt.target_value, unit), _evidence("spectral_tilt_baseline", plan.spectral_tilt.baseline_target_value, unit))))
    elif output_analysis.tonal.spectral_tilt_db_per_decade <= plan.spectral_tilt.baseline_target_value:
        unit = PLAN_METRIC_UNITS[plan.spectral_tilt.metric]
        entries.append(_entry(category="tonal_correction_limit", metric="spectral_tilt_db_per_decade", attribution_level="inferred", decision="capped_by_brightness_budget", tradeoff="Conservative tonal correction over a closer reference tilt match.", required_change="Increase `brightness` or relax the tonal correction cap.", explanation=f"Tonal correction stayed conservative because the spectral-tilt move remained within the brightness budget.", evidence=(_evidence("spectral_tilt_db_per_decade", output_analysis.tonal.spectral_tilt_db_per_decade, unit, plan.spectral_tilt.baseline_target_value, "<="), _evidence("spectral_tilt_baseline", plan.spectral_tilt.baseline_target_value, unit))))
    else:
        entries.append(_tradeoff_entry("tonal_correction_limit", "spectral_tilt_db_per_decade", tilt, _search_tradeoff(optimizer_trace, "spectral_tilt_db_per_decade")))

    return ReferenceAttribution(schema_version=REFERENCE_ATTRIBUTION_SCHEMA_VERSION, available=True, availability_reason=None, entries=tuple(entries))


def _summary_for_loudness(comparison: ReferenceMetricComparison, output_analysis: AnalysisResult, config: MasteringConfig) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    true_peak_margin_db = config.ceiling_db - output_analysis.loudness.true_peak_dbfs
    if comparison.status == "matched":
        message = f"{comparison.label} matched the reference within {comparison.tolerance:.2f} {comparison.unit}."
        finding = ReferenceFinding("REFERENCE_LOUDNESS_MATCHED", "informational", message, (_evidence("integrated_lufs", comparison.output_value, "LUFS", comparison.reference_value, "=="), _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, "LUFS", comparison.tolerance, "<=")))
    elif true_peak_margin_db < 0.3:
        message = f"{comparison.label} moved {abs(comparison.movement_toward_reference):.2f} {comparison.unit} toward the reference; {abs(comparison.output_delta_to_reference):.2f} {comparison.unit} remain, limited by a {true_peak_margin_db:.2f} dB true-peak margin."
        finding = ReferenceFinding("REFERENCE_LOUDNESS_CONSTRAINED_BY_CEILING", "warning", message, (_evidence("true_peak_margin_db", true_peak_margin_db, "dB", 0.3, "<"), _evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, "LUFS", comparison.tolerance, ">")))
    elif comparison.status == "improved":
        message = _format_gap_message(comparison)
        finding = ReferenceFinding("REFERENCE_LOUDNESS_CLOSER", "informational", message, (_evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit, comparison.tolerance, ">"),))
    elif comparison.status == "moved_away":
        message = f"{comparison.label} moved away from the reference by {abs(comparison.movement_toward_reference):.2f} {comparison.unit}; {abs(comparison.output_delta_to_reference):.2f} {comparison.unit} remain."
        finding = ReferenceFinding("REFERENCE_LOUDNESS_MOVED_AWAY", "warning", message, (_evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit, comparison.tolerance, ">"),))
    else:
        message = f"{comparison.label} stayed at the same distance from the reference."
        finding = ReferenceFinding("REFERENCE_LOUDNESS_UNCHANGED", "informational", message, (_evidence("loudness_delta_to_reference", comparison.output_delta_to_reference, comparison.unit),))
    return ReferenceSummaryBullet("loudness", message), finding


def _summary_for_tonal(spectral_tilt: ReferenceMetricComparison, low_mid: ReferenceMetricComparison, high_mid: ReferenceMetricComparison) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    primary = spectral_tilt if abs(spectral_tilt.output_delta_to_reference) >= abs(high_mid.output_delta_to_reference) else high_mid
    if spectral_tilt.status == "matched" and low_mid.status == "matched" and high_mid.status == "matched":
        message = "Tonal balance matched the reference within tolerance."
        finding = ReferenceFinding("REFERENCE_TONAL_MATCHED", "informational", message, (_evidence("spectral_tilt_db_per_decade", spectral_tilt.output_delta_to_reference, "dB/decade", spectral_tilt.tolerance, "<="), _evidence("high_mid_balance_db", high_mid.output_delta_to_reference, "dB", high_mid.tolerance, "<=")))
    else:
        message = f"{spectral_tilt.label} moved {abs(spectral_tilt.movement_toward_reference):.2f} {spectral_tilt.unit} toward the reference; {primary.label.lower()} remains {abs(primary.output_delta_to_reference):.2f} {primary.unit} from it."
        finding = ReferenceFinding("REFERENCE_TONAL_DEVIATION", "warning", message, (_evidence(spectral_tilt.metric, spectral_tilt.output_delta_to_reference, spectral_tilt.unit, spectral_tilt.tolerance, ">"), _evidence(high_mid.metric, high_mid.output_delta_to_reference, high_mid.unit, high_mid.tolerance, ">")))
    return ReferenceSummaryBullet("tonal", message), finding


def _summary_for_low_end(sub_energy: ReferenceMetricComparison, bass_energy: ReferenceMetricComparison, low_band_width: ReferenceMetricComparison, targets: TargetValues) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    if low_band_width.status == "matched":
        message = f"{low_band_width.label} matched the reference within {low_band_width.tolerance:.2f} {low_band_width.unit}; sub and bass energy ratios also remained close to the reference."
        finding = ReferenceFinding("REFERENCE_LOW_END_MATCHED", "informational", message, (_evidence(low_band_width.metric, low_band_width.output_delta_to_reference, low_band_width.unit, low_band_width.tolerance, "<="),))
    elif low_band_width.output_value <= targets.target_low_band_width + 0.1 and low_band_width.output_value <= low_band_width.reference_value:
        message = f"{low_band_width.label} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} narrower than the reference to preserve mono safety."
        finding = ReferenceFinding("REFERENCE_LOW_END_CONSTRAINED_BY_MONO_SAFETY", "informational", message, (_evidence(low_band_width.metric, low_band_width.output_value, low_band_width.unit, targets.target_low_band_width + 0.1, "<="), _evidence("bass_preservation_target_low_band_width", targets.target_low_band_width, "ratio")))
    else:
        message = f"{low_band_width.label} moved {abs(low_band_width.movement_toward_reference):.2f} {low_band_width.unit} toward the reference; sub energy and bass energy remain {abs(sub_energy.output_delta_to_reference):.2f} and {abs(bass_energy.output_delta_to_reference):.2f} {sub_energy.unit} from it."
        finding = ReferenceFinding("REFERENCE_LOW_END_DEVIATION", "warning", message, (_evidence(sub_energy.metric, sub_energy.output_delta_to_reference, sub_energy.unit, sub_energy.tolerance, ">"), _evidence(bass_energy.metric, bass_energy.output_delta_to_reference, bass_energy.unit, bass_energy.tolerance, ">")))
    return ReferenceSummaryBullet("low_end", message), finding


def _summary_for_stereo(mid_band_width: ReferenceMetricComparison, phase_correlation: ReferenceMetricComparison, low_band_width: ReferenceMetricComparison, targets: TargetValues) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    if mid_band_width.status == "matched" and phase_correlation.status == "matched" and low_band_width.status == "matched":
        message = "Stereo width and balance matched the reference within tolerance."
        finding = ReferenceFinding("REFERENCE_STEREO_MATCHED", "informational", message, (_evidence(mid_band_width.metric, mid_band_width.output_delta_to_reference, mid_band_width.unit, mid_band_width.tolerance, "<="), _evidence(phase_correlation.metric, phase_correlation.output_delta_to_reference, phase_correlation.unit, phase_correlation.tolerance, "<=")))
    elif mid_band_width.status != "matched" and abs(mid_band_width.output_delta_to_reference) > 0.05:
        message = f"{mid_band_width.label} moved {abs(mid_band_width.movement_toward_reference):.2f} {mid_band_width.unit} toward the reference; residual gap is {abs(mid_band_width.output_delta_to_reference):.2f} {mid_band_width.unit}."
        finding = ReferenceFinding("REFERENCE_STEREO_WIDTH_DEVIATION", "warning", message, (_evidence(mid_band_width.metric, mid_band_width.output_delta_to_reference, mid_band_width.unit, mid_band_width.tolerance, ">"),))
    elif low_band_width.output_value <= targets.target_low_band_width + 0.1 and low_band_width.output_value <= low_band_width.reference_value:
        message = f"{low_band_width.label} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} narrower than the reference to preserve mono safety."
        finding = ReferenceFinding("REFERENCE_STEREO_CONSTRAINED_BY_MONO_SAFETY", "informational", message, (_evidence(low_band_width.metric, low_band_width.output_value, low_band_width.unit, targets.target_low_band_width + 0.1, "<="),))
    else:
        message = f"{phase_correlation.label} moved {abs(phase_correlation.movement_toward_reference):.2f} {phase_correlation.unit} toward the reference; {low_band_width.label.lower()} remains {abs(low_band_width.output_delta_to_reference):.2f} {low_band_width.unit} from it."
        finding = ReferenceFinding("REFERENCE_STEREO_DEVIATION", "warning", message, (_evidence(phase_correlation.metric, phase_correlation.output_delta_to_reference, phase_correlation.unit, phase_correlation.tolerance, ">"), _evidence(low_band_width.metric, low_band_width.output_delta_to_reference, low_band_width.unit, low_band_width.tolerance, ">")))
    return ReferenceSummaryBullet("stereo", message), finding


def _summary_for_dynamics(crest_factor: ReferenceMetricComparison, output_analysis: AnalysisResult, targets: TargetValues) -> tuple[ReferenceSummaryBullet, ReferenceFinding]:
    target_floor = targets.target_crest_factor - 1.0
    if crest_factor.status == "matched":
        message = f"{crest_factor.label} matched the reference within tolerance."
        finding = ReferenceFinding("REFERENCE_DYNAMICS_MATCHED", "informational", message, (_evidence(crest_factor.metric, crest_factor.output_delta_to_reference, crest_factor.unit, crest_factor.tolerance, "<="),))
    elif output_analysis.loudness.crest_factor_db >= target_floor and crest_factor.output_value > crest_factor.reference_value:
        message = f"{crest_factor.label} remains {abs(crest_factor.output_delta_to_reference):.2f} {crest_factor.unit} above the reference, consistent with the dynamics-preservation target."
        finding = ReferenceFinding("REFERENCE_DYNAMICS_RETAINED", "informational", message, (_evidence(crest_factor.metric, crest_factor.output_value, crest_factor.unit, target_floor, ">="),))
    else:
        message = f"{crest_factor.label} moved {abs(crest_factor.movement_toward_reference):.2f} {crest_factor.unit} toward the reference; {abs(crest_factor.output_delta_to_reference):.2f} {crest_factor.unit} remain."
        finding = ReferenceFinding("REFERENCE_DYNAMICS_DEVIATION", "warning", message, (_evidence(crest_factor.metric, crest_factor.output_delta_to_reference, crest_factor.unit, crest_factor.tolerance, ">"),))
    return ReferenceSummaryBullet("dynamics", message), finding


def build_reference_assessment(config: MasteringConfig, targets: TargetValues, input_analysis: AnalysisResult, reference_analysis: AnalysisResult, output_analysis: AnalysisResult, optimizer_trace: SearchTrace | None = None) -> ReferenceAssessment:
    reference_target_plan = build_reference_aware_targets(config, input_analysis, reference_analysis) if targets.reference_targeting is not None else None
    comparisons = tuple(_build_comparison(metric, input_analysis, reference_analysis, output_analysis) for metric in ("integrated_lufs", "true_peak_dbfs", "spectral_tilt_db_per_decade", "low_mid_balance_db", "high_mid_balance_db", "sub_energy_ratio", "bass_energy_ratio", "low_band_width", "mid_band_width", "phase_correlation", "crest_factor_db"))
    comparison_map = _metric_lookup(comparisons)
    loudness_summary, loudness_finding = _summary_for_loudness(comparison_map["integrated_lufs"], output_analysis, config)
    tonal_summary, tonal_finding = _summary_for_tonal(comparison_map["spectral_tilt_db_per_decade"], comparison_map["low_mid_balance_db"], comparison_map["high_mid_balance_db"])
    low_end_summary, low_end_finding = _summary_for_low_end(comparison_map["sub_energy_ratio"], comparison_map["bass_energy_ratio"], comparison_map["low_band_width"], targets)
    stereo_summary, stereo_finding = _summary_for_stereo(comparison_map["mid_band_width"], comparison_map["phase_correlation"], comparison_map["low_band_width"], targets)
    dynamics_summary, dynamics_finding = _summary_for_dynamics(comparison_map["crest_factor_db"], output_analysis, targets)
    findings = (loudness_finding, tonal_finding, low_end_finding, stereo_finding, dynamics_finding)
    summary = (loudness_summary, tonal_summary, low_end_summary, stereo_summary, dynamics_summary)
    attribution = _build_reference_attribution(config, comparisons, output_analysis, reference_target_plan, optimizer_trace)
    outcome = "constrained" if any("CONSTRAINED" in finding.code for finding in findings) else "deviated" if any(comparison.status == "moved_away" for comparison in comparisons) else "matched" if all(comparison.status == "matched" for comparison in comparisons) else "partial"
    return ReferenceAssessment(REFERENCE_ASSESSMENT_SCHEMA_VERSION, config.reference_path, reference_analysis.schema_version, reference_analysis.identity, comparisons, findings, summary, attribution, outcome)


def render_reference_markdown_section(assessment: ReferenceAssessment) -> list[str]:
    lines = ["## Reference", "", f"- Outcome: `{assessment.outcome}`", f"- Reference path: `{assessment.reference_path or 'not provided'}`", f"- Reference analysis schema: `{assessment.reference_analysis_schema_version}`", ("- Reference identity: " f"`{assessment.reference_analysis_identity.analyzer_version}` / " f"{assessment.reference_analysis_identity.sample_rate_hz} Hz / " f"{assessment.reference_analysis_identity.channels} ch / " f"{assessment.reference_analysis_identity.samples} samples / " f"{assessment.reference_analysis_identity.duration_s:.2f} s / " f"{assessment.reference_analysis_identity.role} / " f"{assessment.reference_analysis_identity.source_path or 'no source path'}"), "", "## Reference Comparison", ""]
    for comparison in assessment.comparisons:
        lines.append(f"- {comparison.label}: input `{comparison.input_value:.2f} {comparison.unit}`, reference `{comparison.reference_value:.2f} {comparison.unit}`, output `{comparison.output_value:.2f} {comparison.unit}`, output moved `{abs(comparison.movement_toward_reference):.2f} {comparison.unit}` toward the reference, residual gap `{abs(comparison.output_delta_to_reference):.2f} {comparison.unit}`.")
    lines.extend(["", "## Reference Summary", ""])
    for bullet in assessment.summary:
        lines.append(f"- {bullet.message}")
    lines.extend(["", "## Reference Attribution", ""])
    if assessment.attribution is None:
        lines.append("- Attribution unavailable.")
    elif not assessment.attribution.available:
        lines.append(f"- Attribution unavailable: {assessment.attribution.availability_reason or 'no causal explanation was generated.'}")
    else:
        for entry in assessment.attribution.entries:
            lines.append(f"- `{entry.attribution_level}` `{entry.category}`: {entry.explanation} Tradeoff: {entry.tradeoff} Required change: {entry.required_change}")
    lines.extend(["", "## Reference Findings", ""])
    for finding in assessment.findings:
        lines.append(f"- `{finding.severity}` `{finding.code}`: {finding.message}")
    lines.append("")
    return lines
