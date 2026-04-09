from __future__ import annotations

import numpy as np

from cognis.analysis.features import (
    ANALYSIS_SCHEMA_VERSION,
    ANALYZER_VERSION,
    AnalysisIdentity,
    AnalysisNotes,
    AnalysisResult,
    LoudnessSummary,
    RiskSummary,
    StereoSummary,
    TonalSummary,
)
from cognis.analysis.loudness import compute_loudness
from cognis.analysis.preflight import validate_audio
from cognis.analysis.spectrum import compute_spectrum_features
from cognis.analysis.stereo import compute_stereo_features


def _compute_risk_summary(audio: np.ndarray, loudness: LoudnessSummary) -> RiskSummary:
    abs_audio = np.abs(audio)
    total_samples = int(abs_audio.size)
    clipped_sample_count = int(np.count_nonzero(abs_audio >= 0.9999))
    hot_threshold = 10 ** (-1.0 / 20.0)
    near_full_scale_threshold = 10 ** (-0.1 / 20.0)
    hot_sample_count = int(np.count_nonzero(abs_audio >= hot_threshold))
    near_full_scale_count = int(np.count_nonzero(abs_audio >= near_full_scale_threshold))

    hot_sample_ratio = hot_sample_count / total_samples
    near_full_scale_ratio = near_full_scale_count / total_samples
    clipped_sample_ratio = clipped_sample_count / total_samples
    intersample_peak_excess_db = max(0.0, loudness.true_peak_dbfs - loudness.sample_peak_dbfs)
    codec_headroom_margin_db = -1.0 - loudness.true_peak_dbfs

    limiter_stress_estimate = float(
        np.clip(
            (hot_sample_ratio * 40.0)
            + (near_full_scale_ratio * 80.0)
            + max(0.0, 7.0 - loudness.crest_factor_db) / 7.0,
            0.0,
            1.0,
        )
    )
    codec_risk_estimate = float(
        np.clip(
            max(0.0, loudness.true_peak_dbfs + 1.0) / 2.0
            + (near_full_scale_ratio * 50.0)
            + (intersample_peak_excess_db / 2.0),
            0.0,
            1.0,
        )
    )
    clipping_risk_estimate = float(
        np.clip((clipped_sample_ratio * 1000.0) + max(0.0, loudness.true_peak_dbfs) / 2.0, 0.0, 1.0)
    )
    delivery_safety_estimate = float(
        np.clip(
            1.0
            - (
                0.35 * limiter_stress_estimate
                + 0.45 * codec_risk_estimate
                + 0.20 * clipping_risk_estimate
            ),
            0.0,
            1.0,
        )
    )

    return RiskSummary(
        limiter_stress_estimate=limiter_stress_estimate,
        codec_risk_estimate=codec_risk_estimate,
        clipping_risk_estimate=clipping_risk_estimate,
        delivery_safety_estimate=delivery_safety_estimate,
        hot_sample_ratio=float(hot_sample_ratio),
        near_full_scale_ratio=float(near_full_scale_ratio),
        clipped_sample_count=clipped_sample_count,
        clipped_sample_ratio=float(clipped_sample_ratio),
        intersample_peak_excess_db=float(intersample_peak_excess_db),
        codec_headroom_margin_db=float(codec_headroom_margin_db),
    )


class Analyzer:
    def __init__(self):
        self.schema_version = ANALYSIS_SCHEMA_VERSION
        self.analyzer_version = ANALYZER_VERSION

    def analyze(self, audio: np.ndarray, sr: int) -> AnalysisResult:
        """
        Analyze audio and return structured, deterministic features.
        Audio must be shaped (channels, samples).
        """
        validate_audio(audio)

        loudness_data = compute_loudness(audio, sr)
        sample_peak = loudness_data["sample_peak_dbfs"]
        integrated = loudness_data["integrated_lufs"]

        rms = np.sqrt(np.mean(audio**2))
        rms_dbfs = -70.0 if rms < 1e-10 else 20 * np.log10(rms)
        crest_factor_db = max(0.0, sample_peak - rms_dbfs)
        peak_to_loudness_ratio = loudness_data["true_peak_dbfs"] - integrated

        loudness = LoudnessSummary(
            integrated_lufs=float(integrated),
            short_term_max_lufs=float(loudness_data["short_term_max_lufs"]),
            short_term_mean_lufs=float(loudness_data["short_term_mean_lufs"]),
            short_term_min_lufs=float(loudness_data["short_term_min_lufs"]),
            short_term_range_lu=float(loudness_data["short_term_range_lu"]),
            momentary_max_lufs=float(loudness_data["momentary_max_lufs"]),
            momentary_mean_lufs=float(loudness_data["momentary_mean_lufs"]),
            momentary_min_lufs=float(loudness_data["momentary_min_lufs"]),
            loudness_range_lu=float(loudness_data["loudness_range_lu"]),
            sample_peak_dbfs=float(sample_peak),
            true_peak_dbfs=float(loudness_data["true_peak_dbfs"]),
            peak_to_loudness_ratio_lu=float(peak_to_loudness_ratio),
            crest_factor_db=float(crest_factor_db),
        )

        spectrum_data = compute_spectrum_features(audio, sr)
        tonal = TonalSummary(
            spectral_tilt_db_per_decade=float(spectrum_data["spectral_tilt_db_per_decade"]),
            low_mid_balance_db=float(spectrum_data["low_mid_balance_db"]),
            high_mid_balance_db=float(spectrum_data["high_mid_balance_db"]),
            sub_energy_ratio=float(spectrum_data["sub_energy_ratio"]),
            bass_energy_ratio=float(spectrum_data["bass_energy_ratio"]),
            low_energy_ratio=float(spectrum_data["low_energy_ratio"]),
            high_energy_ratio=float(spectrum_data["high_energy_ratio"]),
            low_band_centroid_hz=float(spectrum_data["low_band_centroid_hz"]),
        )

        stereo_data = compute_stereo_features(audio, sr)
        stereo = StereoSummary(
            phase_correlation=float(stereo_data["phase_correlation"]),
            low_band_width=float(stereo_data["low_band_width"]),
            mid_band_width=float(stereo_data["mid_band_width"]),
            high_band_width=float(stereo_data["high_band_width"]),
            side_energy_ratio=float(stereo_data["side_energy_ratio"]),
            mono_null_ratio_db=float(stereo_data["mono_null_ratio_db"]),
            left_right_balance_db=float(stereo_data["left_right_balance_db"]),
        )

        identity = AnalysisIdentity(
            schema_version=self.schema_version,
            analyzer_version=self.analyzer_version,
            sample_rate_hz=int(sr),
            channels=int(audio.shape[0]),
            samples=int(audio.shape[1]),
            duration_s=float(audio.shape[1] / sr),
        )

        return AnalysisResult(
            identity=identity,
            loudness=loudness,
            tonal=tonal,
            stereo=stereo,
            risks=_compute_risk_summary(audio, loudness),
            notes=AnalysisNotes(
                momentary_available=True,
                loudness_range_available=audio.shape[1] >= int(3.0 * sr),
                codec_risk_is_proxy=True,
                limiter_stress_is_proxy=True,
            ),
        )
