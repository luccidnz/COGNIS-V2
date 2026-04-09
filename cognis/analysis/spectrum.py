from __future__ import annotations

import numpy as np
from scipy.signal import welch


def _band_energy_ratio(pxx: np.ndarray, mask: np.ndarray, total_energy: float) -> float:
    return float(np.sum(pxx[mask]) / total_energy) if np.any(mask) else 0.0


def compute_spectrum_features(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute tonal balance and low-end distribution features."""
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]

    if len(mono) < 256:
        return {
            "spectral_tilt_db_per_decade": 0.0,
            "low_mid_balance_db": 0.0,
            "high_mid_balance_db": 0.0,
            "sub_energy_ratio": 0.0,
            "bass_energy_ratio": 0.0,
            "low_energy_ratio": 0.0,
            "high_energy_ratio": 0.0,
            "low_band_centroid_hz": 0.0,
        }

    freq, pxx = welch(mono, sr, nperseg=min(4096, len(mono)))
    pxx = np.maximum(pxx, 1e-10)

    sub_mask = (freq >= 20) & (freq < 60)
    bass_mask = (freq >= 60) & (freq < 120)
    low_mask = (freq >= 20) & (freq < 250)
    mid_mask = (freq >= 250) & (freq < 4000)
    high_mask = (freq >= 4000) & (freq < 20000)

    low_energy = float(np.sum(pxx[low_mask]))
    mid_energy = float(np.sum(pxx[mid_mask]))
    high_energy = float(np.sum(pxx[high_mask]))
    total_energy = low_energy + mid_energy + high_energy + 1e-10

    valid = (freq > 50) & (freq < 16000)
    if np.sum(valid) > 2:
        log_freq = np.log10(freq[valid])
        log_power = 10 * np.log10(pxx[valid])
        tilt, _ = np.polyfit(log_freq, log_power, 1)
    else:
        tilt = 0.0

    if np.any(low_mask):
        low_band_energy = np.sum(pxx[low_mask]) + 1e-10
        low_band_centroid_hz = float(np.sum(freq[low_mask] * pxx[low_mask]) / low_band_energy)
    else:
        low_band_centroid_hz = 0.0

    return {
        "spectral_tilt_db_per_decade": float(tilt),
        "low_mid_balance_db": float(10 * np.log10((low_energy + 1e-10) / (mid_energy + 1e-10))),
        "high_mid_balance_db": float(10 * np.log10((high_energy + 1e-10) / (mid_energy + 1e-10))),
        "sub_energy_ratio": _band_energy_ratio(pxx, sub_mask, total_energy),
        "bass_energy_ratio": _band_energy_ratio(pxx, bass_mask, total_energy),
        "low_energy_ratio": float(low_energy / total_energy),
        "high_energy_ratio": float(high_energy / total_energy),
        "low_band_centroid_hz": low_band_centroid_hz,
    }
