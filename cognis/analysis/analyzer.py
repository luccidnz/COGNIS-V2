import numpy as np
from cognis.analysis.features import AnalysisResult, LoudnessMeasures, SpectrumMeasures, StereoMeasures
from cognis.analysis.preflight import validate_audio
from cognis.analysis.loudness import compute_loudness
from cognis.analysis.spectrum import compute_spectrum_features
from cognis.analysis.stereo import compute_stereo_features

class Analyzer:
    def __init__(self):
        self.schema_version = "analysis_schema_v1"

    def analyze(self, audio: np.ndarray, sr: int) -> AnalysisResult:
        """
        Analyze audio and return structured features.
        Audio should be shape (channels, samples).
        """
        validate_audio(audio)
        
        duration = audio.shape[1] / sr
        channels = audio.shape[0]
        
        # Loudness
        integrated, short_term, momentary, true_peak, sample_peak = compute_loudness(audio, sr)
        
        # Crest factor
        rms = np.sqrt(np.mean(audio**2))
        rms_dbfs = -70.0 if rms < 1e-10 else 20 * np.log10(rms)
        crest_factor = max(0.0, sample_peak - rms_dbfs)
        
        loudness_measures = LoudnessMeasures(
            integrated_loudness=integrated,
            short_term_loudness=short_term,
            momentary_loudness=momentary,
            true_peak=true_peak,
            sample_peak=sample_peak,
            crest_factor=crest_factor
        )
        
        # Spectrum
        tilt, low_mid, high_mid = compute_spectrum_features(audio, sr)
        spectrum_measures = SpectrumMeasures(
            spectral_tilt=tilt,
            low_mid_balance=low_mid,
            high_mid_balance=high_mid
        )
        
        # Stereo
        low_w, mid_w, high_w, phase_corr = compute_stereo_features(audio, sr)
        stereo_measures = StereoMeasures(
            low_band_width=low_w,
            mid_band_width=mid_w,
            high_band_width=high_w,
            phase_correlation=phase_corr
        )
        
        return AnalysisResult(
            duration=duration,
            sample_rate=sr,
            channels=channels,
            schema_version=self.schema_version,
            loudness=loudness_measures,
            spectrum=spectrum_measures,
            stereo=stereo_measures
        )
