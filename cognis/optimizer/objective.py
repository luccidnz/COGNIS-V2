from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognis.analysis.features import AnalysisResult
from cognis.optimizer.targets import TargetValues


OBJECTIVE_ATTRIBUTION_SCHEMA_VERSION = "objective_attribution_v1"


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_builtin(item) for item in value)
    if hasattr(value, "item"):
        return value.item()
    return value


@dataclass(frozen=True)
class ObjectiveTermAttribution:
    name: str
    category: str
    basis: str
    value: float
    target_value: float | None
    difference: float | None
    weight: float
    penalty: float
    active: bool
    note: str

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass(frozen=True)
class ReferenceConstraintAttribution:
    metric: str
    category: str
    basis: str
    input_value: float
    reference_value: float
    baseline_target_value: float | None
    target_value: float
    input_vs_reference: float
    target_vs_input: float
    target_vs_reference: float
    safety_limited: bool
    policy: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass(frozen=True)
class ObjectiveAttribution:
    schema_version: str
    score_basis: str
    reference_basis: str
    reference_targeting_schema_version: str | None
    total_score: float
    terms: tuple[ObjectiveTermAttribution, ...]
    reference_constraints: tuple[ReferenceConstraintAttribution, ...]
    dominant_term: str | None
    dominant_penalty: float

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


def _term(
    name: str,
    category: str,
    *,
    value: float,
    target_value: float | None,
    difference: float | None,
    weight: float,
    penalty: float,
    active: bool,
    note: str,
    basis: str = "exact",
) -> ObjectiveTermAttribution:
    return ObjectiveTermAttribution(
        name=name,
        category=category,
        basis=basis,
        value=float(value),
        target_value=None if target_value is None else float(target_value),
        difference=None if difference is None else float(difference),
        weight=float(weight),
        penalty=float(penalty),
        active=bool(active),
        note=note,
    )


def _reference_constraint_category(metric: str) -> str:
    return {
        "integrated_lufs": "loudness_cap",
        "spectral_tilt_db_per_decade": "brightness_budget",
        "low_mid_balance_db": "tonal_balance_guidance",
        "high_mid_balance_db": "tonal_balance_guidance",
        "low_band_centroid_hz": "low_end_balance_guidance",
        "mid_band_width": "stereo_shape_guidance",
        "low_band_width": "mono_safety",
        "phase_correlation": "mono_compatibility",
        "crest_factor_db": "dynamics_preservation",
    }[metric]


def _reference_constraints(targets: TargetValues) -> tuple[ReferenceConstraintAttribution, ...]:
    reference_targeting = targets.reference_targeting
    if reference_targeting is None:
        return ()
    reference_metrics = reference_targeting.reference_targets
    input_vs_reference = reference_targeting.input_vs_reference
    safety_constraints = reference_targeting.safety_constraints

    specs = (
        (
            "integrated_lufs",
            targets.target_loudness,
            "loudness_cap",
            "derived",
            safety_constraints.get("reference_loudness_limit_db"),
            "Reference loudness is capped by the configured loudness baseline.",
            True,
        ),
        (
            "spectral_tilt_db_per_decade",
            targets.target_tilt,
            "brightness_budget",
            "derived",
            None,
            "Reference tonal tilt is capped by the configured brightness budget.",
            True,
        ),
        (
            "low_mid_balance_db",
            targets.target_low_mid_balance,
            "tonal_balance_guidance",
            "exact",
            None,
            "Reference low/mid balance is used directly.",
            False,
        ),
        (
            "high_mid_balance_db",
            targets.target_high_mid_balance,
            "tonal_balance_guidance",
            "exact",
            None,
            "Reference high/mid balance is used directly.",
            False,
        ),
        (
            "mid_band_width",
            targets.target_width,
            "stereo_shape_guidance",
            "exact",
            None,
            "Reference mid-band width is used directly.",
            False,
        ),
        (
            "low_band_width",
            targets.target_low_band_width,
            "mono_safety",
            "derived",
            safety_constraints.get("low_band_width_cap"),
            "Reference low-band width is capped by the mono-safety baseline.",
            True,
        ),
        (
            "phase_correlation",
            None,
            "mono_compatibility",
            "derived",
            safety_constraints.get("phase_correlation_floor"),
            "Reference phase correlation is not allowed to reduce mono compatibility below the input.",
            True,
        ),
        (
            "crest_factor_db",
            targets.target_crest_factor,
            "dynamics_preservation",
            "derived",
            safety_constraints.get("crest_factor_floor_db"),
            "Reference crest factor is capped by the dynamics-preservation baseline.",
            True,
        ),
    )

    entries: list[ReferenceConstraintAttribution] = []
    for metric, target_value, category, basis, baseline_target_value, note, safety_limited in specs:
        reference_value = float(reference_metrics[metric])
        delta_key = f"{metric}_delta"
        input_delta = float(input_vs_reference[delta_key])
        input_value = reference_value - input_delta
        effective_target_value = reference_value if target_value is None else float(target_value)
        entries.append(
            ReferenceConstraintAttribution(
                metric=metric,
                category=category,
                basis=basis,
                input_value=input_value,
                reference_value=reference_value,
                baseline_target_value=None if baseline_target_value is None else float(baseline_target_value),
                target_value=effective_target_value,
                input_vs_reference=input_delta,
                target_vs_input=float(effective_target_value - input_value),
                target_vs_reference=float(effective_target_value - reference_value),
                safety_limited=safety_limited,
                policy="reference_guided_conservative",
                note=note,
            )
        )

    return tuple(entries)


def _objective_breakdown(analysis: AnalysisResult, targets: TargetValues) -> tuple[tuple[ObjectiveTermAttribution, ...], float]:
    terms: list[ObjectiveTermAttribution] = []
    score = 0.0

    tp_violation = max(0.0, analysis.loudness.true_peak_dbfs - targets.ceiling_db)
    penalty = tp_violation * 10000.0
    score += penalty
    terms.append(
        _term(
            "true_peak_ceiling",
            "hard_constraint",
            value=analysis.loudness.true_peak_dbfs,
            target_value=targets.ceiling_db,
            difference=analysis.loudness.true_peak_dbfs - targets.ceiling_db,
            weight=10000.0,
            penalty=penalty,
            active=tp_violation > 0.0,
            note=(
                "True peak exceeded the ceiling."
                if tp_violation > 0.0
                else "True peak stayed within the ceiling."
            ),
        )
    )

    phase_violation = analysis.stereo.phase_correlation < 0.0
    penalty = abs(analysis.stereo.phase_correlation) * 5000.0 if phase_violation else 0.0
    score += penalty
    terms.append(
        _term(
            "phase_correlation_floor",
            "hard_constraint",
            value=analysis.stereo.phase_correlation,
            target_value=0.0,
            difference=analysis.stereo.phase_correlation,
            weight=5000.0,
            penalty=penalty,
            active=phase_violation,
            note=(
                "Phase correlation dropped below zero."
                if phase_violation
                else "Phase correlation stayed above the hard floor."
            ),
        )
    )

    lb_width_cap = targets.target_low_band_width + 0.1
    lb_width_violation = max(0.0, analysis.stereo.low_band_width - lb_width_cap)
    penalty = lb_width_violation * 5000.0
    score += penalty
    terms.append(
        _term(
            "low_band_width_cap",
            "hard_constraint",
            value=analysis.stereo.low_band_width,
            target_value=lb_width_cap,
            difference=analysis.stereo.low_band_width - lb_width_cap,
            weight=5000.0,
            penalty=penalty,
            active=lb_width_violation > 0.0,
            note=(
                "Low-band width exceeded the mono-safety cap."
                if lb_width_violation > 0.0
                else "Low-band width stayed within the mono-safety cap."
            ),
        )
    )

    loudness_diff = abs(analysis.loudness.integrated_lufs - targets.target_loudness)
    penalty = loudness_diff * 20.0
    score += penalty
    terms.append(
        _term(
            "integrated_lufs_target",
            "soft_target",
            value=analysis.loudness.integrated_lufs,
            target_value=targets.target_loudness,
            difference=analysis.loudness.integrated_lufs - targets.target_loudness,
            weight=20.0,
            penalty=penalty,
            active=True,
            note="Integrated loudness was scored against the configured target.",
        )
    )

    tilt_diff = abs(analysis.tonal.spectral_tilt_db_per_decade - targets.target_tilt)
    penalty = tilt_diff * 10.0
    score += penalty
    terms.append(
        _term(
            "spectral_tilt_target",
            "soft_target",
            value=analysis.tonal.spectral_tilt_db_per_decade,
            target_value=targets.target_tilt,
            difference=analysis.tonal.spectral_tilt_db_per_decade - targets.target_tilt,
            weight=10.0,
            penalty=penalty,
            active=True,
            note="Spectral tilt was scored against the configured tonal target.",
        )
    )

    low_mid_diff = abs(analysis.tonal.low_mid_balance_db - targets.target_low_mid_balance)
    penalty = low_mid_diff * 5.0
    score += penalty
    terms.append(
        _term(
            "low_mid_balance_target",
            "soft_target",
            value=analysis.tonal.low_mid_balance_db,
            target_value=targets.target_low_mid_balance,
            difference=analysis.tonal.low_mid_balance_db - targets.target_low_mid_balance,
            weight=5.0,
            penalty=penalty,
            active=True,
            note="Low/mid balance was scored against the configured tonal target.",
        )
    )

    high_mid_diff = abs(analysis.tonal.high_mid_balance_db - targets.target_high_mid_balance)
    penalty = high_mid_diff * 5.0
    score += penalty
    terms.append(
        _term(
            "high_mid_balance_target",
            "soft_target",
            value=analysis.tonal.high_mid_balance_db,
            target_value=targets.target_high_mid_balance,
            difference=analysis.tonal.high_mid_balance_db - targets.target_high_mid_balance,
            weight=5.0,
            penalty=penalty,
            active=True,
            note="High/mid balance was scored against the configured tonal target.",
        )
    )

    crest_active = analysis.loudness.crest_factor_db < targets.target_crest_factor
    crest_diff = targets.target_crest_factor - analysis.loudness.crest_factor_db if crest_active else 0.0
    penalty = crest_diff * 15.0
    score += penalty
    terms.append(
        _term(
            "crest_factor_floor",
            "soft_target",
            value=analysis.loudness.crest_factor_db,
            target_value=targets.target_crest_factor,
            difference=analysis.loudness.crest_factor_db - targets.target_crest_factor,
            weight=15.0,
            penalty=penalty,
            active=crest_active,
            note=(
                "Crest factor fell below the dynamics-preservation target."
                if crest_active
                else "Crest factor stayed at or above the dynamics-preservation target."
            ),
        )
    )

    width_diff = abs(analysis.stereo.mid_band_width - targets.target_width)
    penalty = width_diff * 10.0
    score += penalty
    terms.append(
        _term(
            "mid_band_width_target",
            "soft_target",
            value=analysis.stereo.mid_band_width,
            target_value=targets.target_width,
            difference=analysis.stereo.mid_band_width - targets.target_width,
            weight=10.0,
            penalty=penalty,
            active=True,
            note="Mid-band stereo width was scored against the configured target.",
        )
    )

    if targets.target_sub_energy_ratio > 0.0:
        sub_energy_diff = abs(analysis.tonal.sub_energy_ratio - targets.target_sub_energy_ratio)
        penalty = sub_energy_diff * 50.0
        score += penalty
        terms.append(
            _term(
                "sub_energy_ratio_reference",
                "reference_alignment",
                value=analysis.tonal.sub_energy_ratio,
                target_value=targets.target_sub_energy_ratio,
                difference=analysis.tonal.sub_energy_ratio - targets.target_sub_energy_ratio,
                weight=50.0,
                penalty=penalty,
                active=True,
                note="Sub energy was scored against the reference-aligned target.",
            )
        )

    if targets.target_low_energy_ratio > 0.0:
        low_energy_diff = abs(analysis.tonal.low_energy_ratio - targets.target_low_energy_ratio)
        penalty = low_energy_diff * 40.0
        score += penalty
        terms.append(
            _term(
                "low_energy_ratio_reference",
                "reference_alignment",
                value=analysis.tonal.low_energy_ratio,
                target_value=targets.target_low_energy_ratio,
                difference=analysis.tonal.low_energy_ratio - targets.target_low_energy_ratio,
                weight=40.0,
                penalty=penalty,
                active=True,
                note="Low energy was scored against the reference-aligned target.",
            )
        )

    if targets.target_side_energy_ratio > 0.0:
        side_energy_diff = abs(analysis.stereo.side_energy_ratio - targets.target_side_energy_ratio)
        penalty = side_energy_diff * 25.0
        score += penalty
        terms.append(
            _term(
                "side_energy_ratio_reference",
                "reference_alignment",
                value=analysis.stereo.side_energy_ratio,
                target_value=targets.target_side_energy_ratio,
                difference=analysis.stereo.side_energy_ratio - targets.target_side_energy_ratio,
                weight=25.0,
                penalty=penalty,
                active=True,
                note="Side energy was scored against the reference-aligned target.",
            )
        )

    mono_penalty = 1.0 - analysis.stereo.phase_correlation
    penalty = mono_penalty * 5.0
    score += penalty
    terms.append(
        _term(
            "phase_correlation_soft",
            "soft_target",
            value=analysis.stereo.phase_correlation,
            target_value=1.0,
            difference=analysis.stereo.phase_correlation - 1.0,
            weight=5.0,
            penalty=penalty,
            active=True,
            note="Phase correlation was softly encouraged toward full mono compatibility.",
        )
    )

    reference_targeting = targets.reference_targeting
    if reference_targeting is not None:
        reference_metrics = reference_targeting.reference_targets
        reference_pairs = (
            ("integrated_lufs", analysis.loudness.integrated_lufs, 12.0),
            ("spectral_tilt_db_per_decade", analysis.tonal.spectral_tilt_db_per_decade, 8.0),
            ("mid_band_width", analysis.stereo.mid_band_width, 10.0),
            ("low_band_width", analysis.stereo.low_band_width, 10.0),
            ("crest_factor_db", analysis.loudness.crest_factor_db, 8.0),
            ("phase_correlation", analysis.stereo.phase_correlation, 5.0),
            ("low_energy_ratio", analysis.tonal.low_energy_ratio, 6.0),
            ("sub_energy_ratio", analysis.tonal.sub_energy_ratio, 6.0),
            ("bass_energy_ratio", analysis.tonal.bass_energy_ratio, 6.0),
        )
        for metric, value, weight in reference_pairs:
            reference_value = reference_metrics[metric]
            diff = abs(value - reference_value)
            penalty = diff * weight
            score += penalty
            terms.append(
                _term(
                    f"reference_{metric}",
                    "reference_alignment",
                    value=value,
                    target_value=reference_value,
                    difference=value - reference_value,
                    weight=weight,
                    penalty=penalty,
                    active=True,
                    note="Reference-aligned guidance was scored directly against the measured reference.",
                )
            )

    return tuple(terms), float(score)


def build_objective_attribution(analysis: AnalysisResult, targets: TargetValues) -> ObjectiveAttribution:
    terms, score = _objective_breakdown(analysis, targets)
    dominant_term = max(terms, key=lambda term: term.penalty, default=None)
    reference_targeting = targets.reference_targeting
    reference_basis = "derived" if reference_targeting is not None else "unavailable"
    return ObjectiveAttribution(
        schema_version=OBJECTIVE_ATTRIBUTION_SCHEMA_VERSION,
        score_basis="exact",
        reference_basis=reference_basis,
        reference_targeting_schema_version=(
            None if reference_targeting is None else reference_targeting.schema_version
        ),
        total_score=score,
        terms=terms,
        reference_constraints=_reference_constraints(targets),
        dominant_term=None if dominant_term is None else dominant_term.name,
        dominant_penalty=0.0 if dominant_term is None else dominant_term.penalty,
    )


def compute_objective(analysis: AnalysisResult, targets: TargetValues) -> float:
    """
    Compute penalty score for a rendered candidate.
    Lower is better.
    """

    _, score = _objective_breakdown(analysis, targets)
    return score
