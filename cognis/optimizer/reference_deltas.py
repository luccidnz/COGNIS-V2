from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from cognis.analysis.features import AnalysisResult


REFERENCE_DELTA_SCHEMA_VERSION = "reference_delta_schema_v1"


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


def _delta(current: float, other: float | None) -> float | None:
    if other is None:
        return None
    return float(current - other)


@dataclass(frozen=True)
class MetricDelta:
    metric: str
    input_value: float
    reference_value: float | None
    output_value: float | None
    input_vs_reference: float | None
    output_vs_reference: float | None
    input_vs_output: float | None

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


@dataclass(frozen=True)
class LoudnessDeltas:
    integrated_lufs: MetricDelta
    sample_peak_dbfs: MetricDelta
    true_peak_dbfs: MetricDelta
    crest_factor_db: MetricDelta


@dataclass(frozen=True)
class TonalDeltas:
    spectral_tilt_db_per_decade: MetricDelta
    low_mid_balance_db: MetricDelta
    high_mid_balance_db: MetricDelta
    sub_energy_ratio: MetricDelta
    bass_energy_ratio: MetricDelta
    low_energy_ratio: MetricDelta
    high_energy_ratio: MetricDelta
    low_band_centroid_hz: MetricDelta


@dataclass(frozen=True)
class StereoDeltas:
    phase_correlation: MetricDelta
    low_band_width: MetricDelta
    mid_band_width: MetricDelta
    high_band_width: MetricDelta
    side_energy_ratio: MetricDelta
    mono_null_ratio_db: MetricDelta
    left_right_balance_db: MetricDelta


@dataclass(frozen=True)
class ReferenceDeltaBundle:
    schema_version: str
    reference_available: bool
    loudness: LoudnessDeltas
    tonal: TonalDeltas
    stereo: StereoDeltas

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


def _metric_delta(
    metric: str,
    input_value: float,
    reference_value: float | None,
    output_value: float | None,
) -> MetricDelta:
    return MetricDelta(
        metric=metric,
        input_value=float(input_value),
        reference_value=None if reference_value is None else float(reference_value),
        output_value=None if output_value is None else float(output_value),
        input_vs_reference=_delta(input_value, reference_value),
        output_vs_reference=_delta(output_value, reference_value) if output_value is not None else None,
        input_vs_output=_delta(input_value, output_value) if output_value is not None else None,
    )


def _loudness_deltas(
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None,
    output_analysis: AnalysisResult | None,
) -> LoudnessDeltas:
    reference_loudness = None if reference_analysis is None else reference_analysis.loudness
    output_loudness = None if output_analysis is None else output_analysis.loudness
    return LoudnessDeltas(
        integrated_lufs=_metric_delta(
            "integrated_lufs",
            input_analysis.loudness.integrated_lufs,
            None if reference_loudness is None else reference_loudness.integrated_lufs,
            None if output_loudness is None else output_loudness.integrated_lufs,
        ),
        sample_peak_dbfs=_metric_delta(
            "sample_peak_dbfs",
            input_analysis.loudness.sample_peak_dbfs,
            None if reference_loudness is None else reference_loudness.sample_peak_dbfs,
            None if output_loudness is None else output_loudness.sample_peak_dbfs,
        ),
        true_peak_dbfs=_metric_delta(
            "true_peak_dbfs",
            input_analysis.loudness.true_peak_dbfs,
            None if reference_loudness is None else reference_loudness.true_peak_dbfs,
            None if output_loudness is None else output_loudness.true_peak_dbfs,
        ),
        crest_factor_db=_metric_delta(
            "crest_factor_db",
            input_analysis.loudness.crest_factor_db,
            None if reference_loudness is None else reference_loudness.crest_factor_db,
            None if output_loudness is None else output_loudness.crest_factor_db,
        ),
    )


def _tonal_deltas(
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None,
    output_analysis: AnalysisResult | None,
) -> TonalDeltas:
    reference_tonal = None if reference_analysis is None else reference_analysis.tonal
    output_tonal = None if output_analysis is None else output_analysis.tonal
    return TonalDeltas(
        spectral_tilt_db_per_decade=_metric_delta(
            "spectral_tilt_db_per_decade",
            input_analysis.tonal.spectral_tilt_db_per_decade,
            None if reference_tonal is None else reference_tonal.spectral_tilt_db_per_decade,
            None if output_tonal is None else output_tonal.spectral_tilt_db_per_decade,
        ),
        low_mid_balance_db=_metric_delta(
            "low_mid_balance_db",
            input_analysis.tonal.low_mid_balance_db,
            None if reference_tonal is None else reference_tonal.low_mid_balance_db,
            None if output_tonal is None else output_tonal.low_mid_balance_db,
        ),
        high_mid_balance_db=_metric_delta(
            "high_mid_balance_db",
            input_analysis.tonal.high_mid_balance_db,
            None if reference_tonal is None else reference_tonal.high_mid_balance_db,
            None if output_tonal is None else output_tonal.high_mid_balance_db,
        ),
        sub_energy_ratio=_metric_delta(
            "sub_energy_ratio",
            input_analysis.tonal.sub_energy_ratio,
            None if reference_tonal is None else reference_tonal.sub_energy_ratio,
            None if output_tonal is None else output_tonal.sub_energy_ratio,
        ),
        bass_energy_ratio=_metric_delta(
            "bass_energy_ratio",
            input_analysis.tonal.bass_energy_ratio,
            None if reference_tonal is None else reference_tonal.bass_energy_ratio,
            None if output_tonal is None else output_tonal.bass_energy_ratio,
        ),
        low_energy_ratio=_metric_delta(
            "low_energy_ratio",
            input_analysis.tonal.low_energy_ratio,
            None if reference_tonal is None else reference_tonal.low_energy_ratio,
            None if output_tonal is None else output_tonal.low_energy_ratio,
        ),
        high_energy_ratio=_metric_delta(
            "high_energy_ratio",
            input_analysis.tonal.high_energy_ratio,
            None if reference_tonal is None else reference_tonal.high_energy_ratio,
            None if output_tonal is None else output_tonal.high_energy_ratio,
        ),
        low_band_centroid_hz=_metric_delta(
            "low_band_centroid_hz",
            input_analysis.tonal.low_band_centroid_hz,
            None if reference_tonal is None else reference_tonal.low_band_centroid_hz,
            None if output_tonal is None else output_tonal.low_band_centroid_hz,
        ),
    )


def _stereo_deltas(
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None,
    output_analysis: AnalysisResult | None,
) -> StereoDeltas:
    reference_stereo = None if reference_analysis is None else reference_analysis.stereo
    output_stereo = None if output_analysis is None else output_analysis.stereo
    return StereoDeltas(
        phase_correlation=_metric_delta(
            "phase_correlation",
            input_analysis.stereo.phase_correlation,
            None if reference_stereo is None else reference_stereo.phase_correlation,
            None if output_stereo is None else output_stereo.phase_correlation,
        ),
        low_band_width=_metric_delta(
            "low_band_width",
            input_analysis.stereo.low_band_width,
            None if reference_stereo is None else reference_stereo.low_band_width,
            None if output_stereo is None else output_stereo.low_band_width,
        ),
        mid_band_width=_metric_delta(
            "mid_band_width",
            input_analysis.stereo.mid_band_width,
            None if reference_stereo is None else reference_stereo.mid_band_width,
            None if output_stereo is None else output_stereo.mid_band_width,
        ),
        high_band_width=_metric_delta(
            "high_band_width",
            input_analysis.stereo.high_band_width,
            None if reference_stereo is None else reference_stereo.high_band_width,
            None if output_stereo is None else output_stereo.high_band_width,
        ),
        side_energy_ratio=_metric_delta(
            "side_energy_ratio",
            input_analysis.stereo.side_energy_ratio,
            None if reference_stereo is None else reference_stereo.side_energy_ratio,
            None if output_stereo is None else output_stereo.side_energy_ratio,
        ),
        mono_null_ratio_db=_metric_delta(
            "mono_null_ratio_db",
            input_analysis.stereo.mono_null_ratio_db,
            None if reference_stereo is None else reference_stereo.mono_null_ratio_db,
            None if output_stereo is None else output_stereo.mono_null_ratio_db,
        ),
        left_right_balance_db=_metric_delta(
            "left_right_balance_db",
            input_analysis.stereo.left_right_balance_db,
            None if reference_stereo is None else reference_stereo.left_right_balance_db,
            None if output_stereo is None else output_stereo.left_right_balance_db,
        ),
    )


def build_reference_deltas(
    input_analysis: AnalysisResult,
    reference_analysis: AnalysisResult | None,
    output_analysis: AnalysisResult | None = None,
) -> ReferenceDeltaBundle:
    """
    Build deterministic pairwise deltas for reference-aware workflows.

    Sign convention:
    - input_vs_reference = input - reference
    - output_vs_reference = output - reference
    - input_vs_output = input - output
    """

    return ReferenceDeltaBundle(
        schema_version=REFERENCE_DELTA_SCHEMA_VERSION,
        reference_available=reference_analysis is not None,
        loudness=_loudness_deltas(input_analysis, reference_analysis, output_analysis),
        tonal=_tonal_deltas(input_analysis, reference_analysis, output_analysis),
        stereo=_stereo_deltas(input_analysis, reference_analysis, output_analysis),
    )
