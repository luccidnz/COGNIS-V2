from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter


def compute_stereo_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute stereo width and mono-compatibility indicators."""
    if audio.shape[0] == 1:
        return {
            "phase_correlation": 1.0,
            "low_band_width": 0.0,
            "mid_band_width": 0.0,
            "high_band_width": 0.0,
            "side_energy_ratio": 0.0,
            "mono_null_ratio_db": -120.0,
            "left_right_balance_db": 0.0,
        }

    left = audio[0]
    right = audio[1]

    numerator = np.sum(left * right)
    denominator = np.sqrt(np.sum(left**2) * np.sum(right**2)) + 1e-10
    phase_correlation = float(numerator / denominator)

    def get_width(left_band: np.ndarray, right_band: np.ndarray) -> float:
        mid = (left_band + right_band) / 2
        side = (left_band - right_band) / 2
        mid_energy = np.sum(mid**2)
        side_energy = np.sum(side**2)
        if mid_energy + side_energy < 1e-10:
            return 0.0
        return float(side_energy / (mid_energy + side_energy))

    b_low, a_low = butter(2, 250 / (sr / 2), btype="lowpass")
    b_mid, a_mid = butter(2, [250 / (sr / 2), 4000 / (sr / 2)], btype="bandpass")
    b_high, a_high = butter(2, 4000 / (sr / 2), btype="highpass")

    left_low = lfilter(b_low, a_low, left)
    right_low = lfilter(b_low, a_low, right)
    left_mid = lfilter(b_mid, a_mid, left)
    right_mid = lfilter(b_mid, a_mid, right)
    left_high = lfilter(b_high, a_high, left)
    right_high = lfilter(b_high, a_high, right)

    mid = (left + right) / 2.0
    side = (left - right) / 2.0
    mid_rms = float(np.sqrt(np.mean(mid**2)) + 1e-10)
    side_rms = float(np.sqrt(np.mean(side**2)) + 1e-10)
    side_energy_ratio = float((side_rms**2) / ((mid_rms**2) + (side_rms**2)))
    mono_null_ratio_db = float(20 * np.log10(side_rms / mid_rms))

    left_rms = float(np.sqrt(np.mean(left**2)) + 1e-10)
    right_rms = float(np.sqrt(np.mean(right**2)) + 1e-10)
    left_right_balance_db = float(20 * np.log10(left_rms / right_rms))

    return {
        "phase_correlation": phase_correlation,
        "low_band_width": get_width(left_low, right_low),
        "mid_band_width": get_width(left_mid, right_mid),
        "high_band_width": get_width(left_high, right_high),
        "side_energy_ratio": side_energy_ratio,
        "mono_null_ratio_db": mono_null_ratio_db,
        "left_right_balance_db": left_right_balance_db,
    }
