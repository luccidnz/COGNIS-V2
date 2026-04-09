from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cognis.analysis.features import AnalysisResult
from cognis.config import CeilingMode, MasteringConfig


REFERENCE_TARGETING_SCHEMA_VERSION = "reference_targeting_v1"


@dataclass(frozen=True)
class ReferenceTargeting:
    schema_version: str
    reference_analysis_schema_version: str
    reference_identity: dict[str, Any]
    input_vs_reference: dict[str, float]
    reference_targets: dict[str, float]
    safety_constraints: dict[str, float | bool]
    guidance: tuple[str, ...]
    alignment_index: float


ReferenceTargetSummary = ReferenceTargeting


@dataclass(frozen=True)
class TargetValues:
    target_loudness: float
    ceiling_db: float
    target_tilt: float
    target_width: float
    target_crest_factor: float
    target_low_band_width: float
    target_low_mid_balance: float = 0.0
    target_high_mid_balance: float = 0.0
    target_sub_energy_ratio: float = 0.0
    target_low_energy_ratio: float = 0.0
    target_side_energy_ratio: float = 0.0
    reference_targeting: ReferenceTargeting | None = None

    @property
    def reference(self) -> ReferenceTargeting | None:
        return self.reference_targeting


def _reference_alignment_index(input_analysis: AnalysisResult, reference_analysis: AnalysisResult) -> float:
    loudness_gap = abs(reference_analysis.loudness.integrated_lufs - input_analysis.loudness.integrated_lufs) / 12.0
    tilt_gap = abs(reference_analysis.tonal.spectral_tilt_db_per_decade - input_analysis.tonal.spectral_tilt_db_per_decade) / 6.0
    width_gap = abs(reference_analysis.stereo.mid_band_width - input_analysis.stereo.mid_band_width) / 0.5
    low_band_gap = abs(reference_analysis.stereo.low_band_width - input_analysis.stereo.low_band_width) / 0.5
    crest_gap = abs(reference_analysis.loudness.crest_factor_db - input_analysis.loudness.crest_factor_db) / 6.0
    stereo_gap = abs(reference_analysis.stereo.phase_correlation - input_analysis.stereo.phase_correlation) / 1.0
    gap = loudness_gap + tilt_gap + width_gap + low_band_gap + crest_gap + stereo_gap
    return float(np.clip(gap, 0.0, 10.0))


def _reference_targeting(
    config: MasteringConfig,
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult,
    target_width: float,
    target_crest_factor: float,
    target_low_band_width: float,
) -> ReferenceTargeting:
    input_vs_reference = {
        "integrated_lufs_delta": float(reference_analysis.loudness.integrated_lufs - input_analysis.loudness.integrated_lufs),
        "spectral_tilt_db_per_decade_delta": float(
            reference_analysis.tonal.spectral_tilt_db_per_decade - input_analysis.tonal.spectral_tilt_db_per_decade
        ),
        "low_mid_balance_db_delta": float(reference_analysis.tonal.low_mid_balance_db - input_analysis.tonal.low_mid_balance_db),
        "high_mid_balance_db_delta": float(reference_analysis.tonal.high_mid_balance_db - input_analysis.tonal.high_mid_balance_db),
        "mid_band_width_delta": float(reference_analysis.stereo.mid_band_width - input_analysis.stereo.mid_band_width),
        "low_band_width_delta": float(reference_analysis.stereo.low_band_width - input_analysis.stereo.low_band_width),
        "phase_correlation_delta": float(reference_analysis.stereo.phase_correlation - input_analysis.stereo.phase_correlation),
        "crest_factor_db_delta": float(reference_analysis.loudness.crest_factor_db - input_analysis.loudness.crest_factor_db),
        "low_energy_ratio_delta": float(reference_analysis.tonal.low_energy_ratio - input_analysis.tonal.low_energy_ratio),
        "sub_energy_ratio_delta": float(reference_analysis.tonal.sub_energy_ratio - input_analysis.tonal.sub_energy_ratio),
        "bass_energy_ratio_delta": float(reference_analysis.tonal.bass_energy_ratio - input_analysis.tonal.bass_energy_ratio),
    }

    reference_targets = {
        "integrated_lufs": float(reference_analysis.loudness.integrated_lufs),
        "spectral_tilt_db_per_decade": float(reference_analysis.tonal.spectral_tilt_db_per_decade),
        "low_mid_balance_db": float(reference_analysis.tonal.low_mid_balance_db),
        "high_mid_balance_db": float(reference_analysis.tonal.high_mid_balance_db),
        "mid_band_width": float(reference_analysis.stereo.mid_band_width),
        "low_band_width": float(reference_analysis.stereo.low_band_width),
        "phase_correlation": float(reference_analysis.stereo.phase_correlation),
        "crest_factor_db": float(reference_analysis.loudness.crest_factor_db),
        "low_energy_ratio": float(reference_analysis.tonal.low_energy_ratio),
        "sub_energy_ratio": float(reference_analysis.tonal.sub_energy_ratio),
        "bass_energy_ratio": float(reference_analysis.tonal.bass_energy_ratio),
    }

    safety_constraints = {
        "ceiling_db": float(config.ceiling_db),
        "codec_safe_requested": config.ceiling_mode == CeilingMode.CODEC_SAFE,
        "true_peak_margin_floor_db": 0.0,
        "phase_correlation_floor": 0.2,
        "low_band_width_cap": float(target_low_band_width + 0.1),
        "crest_factor_floor_db": float(target_crest_factor - 1.0),
        "reference_loudness_limit_db": float(config.target_loudness),
    }

    guidance = (
        "Use the reference as a measured tonal and dynamics guide, not a blind copy.",
        "Keep true-peak and mono-safety limits ahead of reference matching.",
        "Treat the reference loudness and stereo shape as goals only when they do not violate safety constraints.",
    )

    return ReferenceTargeting(
        schema_version=REFERENCE_TARGETING_SCHEMA_VERSION,
        reference_analysis_schema_version=reference_analysis.schema_version,
        reference_identity=asdict(reference_analysis.identity),
        input_vs_reference=input_vs_reference,
        reference_targets=reference_targets,
        safety_constraints=safety_constraints,
        guidance=guidance,
        alignment_index=_reference_alignment_index(input_analysis, reference_analysis),
    )


def build_targets(
    config: MasteringConfig,
    input_analysis: AnalysisResult | None = None,
    reference_analysis: AnalysisResult | None = None,
) -> TargetValues:
    """
    Map config to explicit target values.

    If a reference analysis is supplied, the returned target values include a
    versioned reference-targeting payload that guides the optimizer toward the
    measured reference while preserving safety constraints.
    """

    target_loudness = config.target_loudness
    ceiling_db = config.ceiling_db
    target_tilt = config.brightness * 3.0
    target_width = config.stereo_width
    target_crest_factor = 6.0 + (config.dynamics_preservation * 3.0)
    target_low_band_width = target_width * (1.0 - config.bass_preservation)
    target_low_mid_balance = 0.0
    target_high_mid_balance = 0.0
    target_sub_energy_ratio = 0.0
    target_low_energy_ratio = 0.0
    target_side_energy_ratio = 0.0

    reference_targeting = None
    if input_analysis is not None and reference_analysis is not None:
        target_loudness = float(min(config.target_loudness, reference_analysis.loudness.integrated_lufs))
        target_tilt = float(min(reference_analysis.tonal.spectral_tilt_db_per_decade, target_tilt))
        target_width = float(reference_analysis.stereo.mid_band_width)
        target_crest_factor = float(
            max(reference_analysis.loudness.crest_factor_db, target_crest_factor, input_analysis.loudness.crest_factor_db)
        )
        target_low_band_width = float(min(reference_analysis.stereo.low_band_width, target_low_band_width))
        target_low_mid_balance = float(reference_analysis.tonal.low_mid_balance_db)
        target_high_mid_balance = float(reference_analysis.tonal.high_mid_balance_db)
        target_sub_energy_ratio = float(reference_analysis.tonal.sub_energy_ratio)
        target_low_energy_ratio = float(reference_analysis.tonal.low_energy_ratio)
        target_side_energy_ratio = float(reference_analysis.stereo.side_energy_ratio)
        reference_targeting = _reference_targeting(
            config=config,
            input_analysis=input_analysis,
            reference_analysis=reference_analysis,
            target_width=target_width,
            target_crest_factor=target_crest_factor,
            target_low_band_width=target_low_band_width,
        )

    return TargetValues(
        target_loudness=target_loudness,
        ceiling_db=ceiling_db,
        target_tilt=target_tilt,
        target_width=target_width,
        target_crest_factor=target_crest_factor,
        target_low_band_width=target_low_band_width,
        target_low_mid_balance=target_low_mid_balance,
        target_high_mid_balance=target_high_mid_balance,
        target_sub_energy_ratio=target_sub_energy_ratio,
        target_low_energy_ratio=target_low_energy_ratio,
        target_side_energy_ratio=target_side_energy_ratio,
        reference_targeting=reference_targeting,
    )
