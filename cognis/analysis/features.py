from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


ANALYSIS_SCHEMA_VERSION = "analysis_schema_v2"
ANALYZER_VERSION = "cognis_analyzer_v2"


@dataclass(frozen=True)
class AnalysisIdentity:
    schema_version: str
    analyzer_version: str
    sample_rate_hz: int
    channels: int
    samples: int
    duration_s: float
    role: str = "analysis"
    source_path: str | None = None
    source_name: str | None = None


@dataclass(frozen=True)
class LoudnessSummary:
    integrated_lufs: float
    short_term_max_lufs: float
    short_term_mean_lufs: float
    short_term_min_lufs: float
    short_term_range_lu: float
    momentary_max_lufs: float
    momentary_mean_lufs: float
    momentary_min_lufs: float
    loudness_range_lu: float
    sample_peak_dbfs: float
    true_peak_dbfs: float
    peak_to_loudness_ratio_lu: float
    crest_factor_db: float

    @property
    def integrated_loudness(self) -> float:
        return self.integrated_lufs

    @property
    def short_term_loudness(self) -> float:
        return self.short_term_max_lufs

    @property
    def momentary_loudness(self) -> float:
        return self.momentary_max_lufs

    @property
    def sample_peak(self) -> float:
        return self.sample_peak_dbfs

    @property
    def true_peak(self) -> float:
        return self.true_peak_dbfs

    @property
    def crest_factor(self) -> float:
        return self.crest_factor_db


@dataclass(frozen=True)
class TonalSummary:
    spectral_tilt_db_per_decade: float
    low_mid_balance_db: float
    high_mid_balance_db: float
    sub_energy_ratio: float
    bass_energy_ratio: float
    low_energy_ratio: float
    high_energy_ratio: float
    low_band_centroid_hz: float

    @property
    def spectral_tilt(self) -> float:
        return self.spectral_tilt_db_per_decade

    @property
    def low_mid_balance(self) -> float:
        return self.low_mid_balance_db

    @property
    def high_mid_balance(self) -> float:
        return self.high_mid_balance_db


@dataclass(frozen=True)
class StereoSummary:
    phase_correlation: float
    low_band_width: float
    mid_band_width: float
    high_band_width: float
    side_energy_ratio: float
    mono_null_ratio_db: float
    left_right_balance_db: float

    @property
    def mono_compatibility_score(self) -> float:
        return max(0.0, min(1.0, (self.phase_correlation + 1.0) / 2.0))


@dataclass(frozen=True)
class RiskSummary:
    limiter_stress_estimate: float
    codec_risk_estimate: float
    clipping_risk_estimate: float
    delivery_safety_estimate: float
    hot_sample_ratio: float
    near_full_scale_ratio: float
    clipped_sample_count: int
    clipped_sample_ratio: float
    intersample_peak_excess_db: float
    codec_headroom_margin_db: float


@dataclass(frozen=True)
class AnalysisNotes:
    momentary_available: bool
    loudness_range_available: bool
    codec_risk_is_proxy: bool
    limiter_stress_is_proxy: bool


@dataclass(frozen=True)
class AnalysisResult:
    identity: AnalysisIdentity
    loudness: LoudnessSummary
    tonal: TonalSummary
    stereo: StereoSummary
    risks: RiskSummary
    notes: AnalysisNotes

    @property
    def schema_version(self) -> str:
        return self.identity.schema_version

    @property
    def duration(self) -> float:
        return self.identity.duration_s

    @property
    def sample_rate(self) -> int:
        return self.identity.sample_rate_hz

    @property
    def channels(self) -> int:
        return self.identity.channels

    @property
    def samples(self) -> int:
        return self.identity.samples

    @property
    def spectrum(self) -> TonalSummary:
        return self.tonal

    def to_dict(self) -> dict[str, Any]:
        return _to_builtin(asdict(self))


def _to_builtin(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_builtin(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value
