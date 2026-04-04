from dataclasses import dataclass

@dataclass
class LoudnessMeasures:
    integrated_loudness: float
    short_term_loudness: float
    momentary_loudness: float
    true_peak: float
    sample_peak: float
    crest_factor: float

@dataclass
class SpectrumMeasures:
    spectral_tilt: float
    low_mid_balance: float
    high_mid_balance: float

@dataclass
class StereoMeasures:
    low_band_width: float
    mid_band_width: float
    high_band_width: float
    phase_correlation: float

@dataclass
class AnalysisResult:
    duration: float
    sample_rate: int
    channels: int
    schema_version: str
    loudness: LoudnessMeasures
    spectrum: SpectrumMeasures
    stereo: StereoMeasures
