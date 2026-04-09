from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognis.analysis.features import AnalysisIdentity, AnalysisResult
from cognis.config import MasteringConfig, MasteringMode
from cognis.optimizer.targets import TargetValues, build_targets


REFERENCE_TARGET_SCHEMA_VERSION = "reference_target_schema_v1"


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
class MetricTargetGuidance:
    metric: str
    input_value: float
    reference_value: float | None
    baseline_target_value: float
    target_value: float
    input_vs_reference: float | None
    target_vs_input: float
    target_vs_reference: float | None
    safety_limited: bool
    policy: str
    note: str

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass(frozen=True)
class ReferenceAwareTargetPlan:
    schema_version: str
    mode: str
    ceiling_mode: str
    ceiling_db: float
    reference_available: bool
    input_identity: AnalysisIdentity
    reference_identity: AnalysisIdentity | None
    baseline_targets: TargetValues
    loudness: MetricTargetGuidance
    spectral_tilt: MetricTargetGuidance
    low_mid_balance: MetricTargetGuidance
    high_mid_balance: MetricTargetGuidance
    low_band_centroid: MetricTargetGuidance
    mid_band_width: MetricTargetGuidance
    low_band_width: MetricTargetGuidance
    phase_correlation: MetricTargetGuidance
    crest_factor: MetricTargetGuidance

    @property
    def target_values(self) -> TargetValues:
        return self.baseline_targets

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


def _metric_guidance(
    metric: str,
    input_value: float,
    reference_value: float | None,
    baseline_target_value: float,
    target_value: float,
    *,
    safety_limited: bool,
    policy: str,
    note: str,
) -> MetricTargetGuidance:
    input_vs_reference = None if reference_value is None else float(input_value - reference_value)
    target_vs_reference = None if reference_value is None else float(target_value - reference_value)
    return MetricTargetGuidance(
        metric=metric,
        input_value=float(input_value),
        reference_value=None if reference_value is None else float(reference_value),
        baseline_target_value=float(baseline_target_value),
        target_value=float(target_value),
        input_vs_reference=input_vs_reference,
        target_vs_input=float(target_value - input_value),
        target_vs_reference=target_vs_reference,
        safety_limited=bool(safety_limited),
        policy=policy,
        note=note,
    )


def _reference_mode(config: MasteringConfig, reference_available: bool) -> str:
    if not reference_available:
        return "config_only"
    if config.mode == MasteringMode.REFERENCE_MATCH:
        return "reference_match_conservative"
    return "reference_guided_conservative"


def build_reference_aware_targets(
    config: MasteringConfig,
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None = None,
) -> ReferenceAwareTargetPlan:
    """
    Build a deterministic reference-aware target plan.

    The plan keeps the current config-derived baseline intact and adds
    reference guidance in a safety-aware way:
    - louder references are not blindly chased past the configured loudness baseline
    - bass width stays at or below the safer of the reference and config baseline
    - dynamics are not compressed below the safer of the reference, input, and baseline
    - tonal balance guidance remains reference-led when a reference exists
    """

    baseline_targets = build_targets(config)
    reference_available = reference_analysis is not None
    policy = _reference_mode(config, reference_available)
    reference_identity = None if reference_analysis is None else reference_analysis.identity

    reference_loudness = None if reference_analysis is None else reference_analysis.loudness.integrated_lufs
    reference_tilt = None if reference_analysis is None else reference_analysis.tonal.spectral_tilt_db_per_decade
    reference_low_mid = None if reference_analysis is None else reference_analysis.tonal.low_mid_balance_db
    reference_high_mid = None if reference_analysis is None else reference_analysis.tonal.high_mid_balance_db
    reference_low_centroid = None if reference_analysis is None else reference_analysis.tonal.low_band_centroid_hz
    reference_mid_width = None if reference_analysis is None else reference_analysis.stereo.mid_band_width
    reference_low_width = None if reference_analysis is None else reference_analysis.stereo.low_band_width
    reference_phase = None if reference_analysis is None else reference_analysis.stereo.phase_correlation
    reference_crest = None if reference_analysis is None else reference_analysis.loudness.crest_factor_db

    loudness_target = (
        baseline_targets.target_loudness
        if reference_loudness is None
        else min(reference_loudness, baseline_targets.target_loudness)
    )
    spectral_tilt_target = (
        baseline_targets.target_tilt
        if reference_tilt is None
        else min(reference_tilt, baseline_targets.target_tilt)
    )
    low_mid_balance_target = (
        input_analysis.tonal.low_mid_balance_db
        if reference_low_mid is None
        else reference_low_mid
    )
    high_mid_balance_target = (
        input_analysis.tonal.high_mid_balance_db
        if reference_high_mid is None
        else reference_high_mid
    )
    low_band_centroid_target = (
        input_analysis.tonal.low_band_centroid_hz
        if reference_low_centroid is None
        else reference_low_centroid
    )
    mid_band_width_target = (
        baseline_targets.target_width
        if reference_mid_width is None
        else reference_mid_width
    )
    low_band_width_target = (
        baseline_targets.target_low_band_width
        if reference_low_width is None
        else min(reference_low_width, baseline_targets.target_low_band_width)
    )
    phase_correlation_target = (
        input_analysis.stereo.phase_correlation
        if reference_phase is None
        else max(reference_phase, input_analysis.stereo.phase_correlation, 0.0)
    )
    crest_factor_target = (
        baseline_targets.target_crest_factor
        if reference_crest is None
        else max(reference_crest, baseline_targets.target_crest_factor, input_analysis.loudness.crest_factor_db)
    )

    return ReferenceAwareTargetPlan(
        schema_version=REFERENCE_TARGET_SCHEMA_VERSION,
        mode=config.mode.value,
        ceiling_mode=config.ceiling_mode.value,
        ceiling_db=float(config.ceiling_db),
        reference_available=reference_available,
        input_identity=input_analysis.identity,
        reference_identity=reference_identity,
        baseline_targets=baseline_targets,
        loudness=_metric_guidance(
            "integrated_lufs",
            input_analysis.loudness.integrated_lufs,
            reference_loudness,
            baseline_targets.target_loudness,
            loudness_target,
            safety_limited=reference_loudness is not None and loudness_target != reference_loudness,
            policy=policy,
            note=(
                "Reference loudness is capped by the configured loudness baseline."
                if reference_loudness is not None and loudness_target != reference_loudness
                else (
                    "Reference loudness is used directly."
                    if reference_loudness is not None
                    else "No reference supplied; using the configured loudness baseline."
                )
            ),
        ),
        spectral_tilt=_metric_guidance(
            "spectral_tilt_db_per_decade",
            input_analysis.tonal.spectral_tilt_db_per_decade,
            reference_tilt,
            baseline_targets.target_tilt,
            spectral_tilt_target,
            safety_limited=reference_tilt is not None and spectral_tilt_target != reference_tilt,
            policy=policy,
            note=(
                "Reference tonal tilt is capped by the configured brightness budget."
                if reference_tilt is not None and spectral_tilt_target != reference_tilt
                else (
                    "Reference tonal tilt is used directly."
                    if reference_tilt is not None
                    else "No reference supplied; using the configured brightness baseline."
                )
            ),
        ),
        low_mid_balance=_metric_guidance(
            "low_mid_balance_db",
            input_analysis.tonal.low_mid_balance_db,
            reference_low_mid,
            input_analysis.tonal.low_mid_balance_db,
            low_mid_balance_target,
            safety_limited=False,
            policy=policy,
            note=(
                "Reference low/mid balance is used as advisory tonal guidance."
                if reference_low_mid is not None
                else "No reference supplied; preserving the current low/mid balance."
            ),
        ),
        high_mid_balance=_metric_guidance(
            "high_mid_balance_db",
            input_analysis.tonal.high_mid_balance_db,
            reference_high_mid,
            input_analysis.tonal.high_mid_balance_db,
            high_mid_balance_target,
            safety_limited=False,
            policy=policy,
            note=(
                "Reference high/mid balance is used as advisory tonal guidance."
                if reference_high_mid is not None
                else "No reference supplied; preserving the current high/mid balance."
            ),
        ),
        low_band_centroid=_metric_guidance(
            "low_band_centroid_hz",
            input_analysis.tonal.low_band_centroid_hz,
            reference_low_centroid,
            input_analysis.tonal.low_band_centroid_hz,
            low_band_centroid_target,
            safety_limited=False,
            policy=policy,
            note=(
                "Reference low-end centroid is used as advisory balance guidance."
                if reference_low_centroid is not None
                else "No reference supplied; preserving the current low-end centroid."
            ),
        ),
        mid_band_width=_metric_guidance(
            "mid_band_width",
            input_analysis.stereo.mid_band_width,
            reference_mid_width,
            baseline_targets.target_width,
            mid_band_width_target,
            safety_limited=False,
            policy=policy,
            note=(
                "Reference mid-band width is used as stylistic guidance."
                if reference_mid_width is not None
                else "No reference supplied; using the configured stereo width baseline."
            ),
        ),
        low_band_width=_metric_guidance(
            "low_band_width",
            input_analysis.stereo.low_band_width,
            reference_low_width,
            baseline_targets.target_low_band_width,
            low_band_width_target,
            safety_limited=reference_low_width is not None and low_band_width_target != reference_low_width,
            policy=policy,
            note=(
                "Reference low-band width is capped by the mono-safety baseline."
                if reference_low_width is not None and low_band_width_target != reference_low_width
                else (
                    "Reference low-band width is used directly."
                    if reference_low_width is not None
                    else "No reference supplied; using the configured low-band width baseline."
                )
            ),
        ),
        phase_correlation=_metric_guidance(
            "phase_correlation",
            input_analysis.stereo.phase_correlation,
            reference_phase,
            input_analysis.stereo.phase_correlation,
            phase_correlation_target,
            safety_limited=reference_phase is not None and phase_correlation_target != reference_phase,
            policy=policy,
            note=(
                "Reference phase correlation is not allowed to reduce mono compatibility below the input."
                if reference_phase is not None and phase_correlation_target != reference_phase
                else (
                    "Reference phase correlation is used directly."
                    if reference_phase is not None
                    else "No reference supplied; preserving the current phase correlation."
                )
            ),
        ),
        crest_factor=_metric_guidance(
            "crest_factor_db",
            input_analysis.loudness.crest_factor_db,
            reference_crest,
            baseline_targets.target_crest_factor,
            crest_factor_target,
            safety_limited=reference_crest is not None and crest_factor_target != reference_crest,
            policy=policy,
            note=(
                "Reference crest factor is capped by the dynamics-preservation baseline."
                if reference_crest is not None and crest_factor_target != reference_crest
                else (
                    "Reference crest factor is used directly."
                    if reference_crest is not None
                    else "No reference supplied; using the configured dynamics baseline."
                )
            ),
        ),
    )
