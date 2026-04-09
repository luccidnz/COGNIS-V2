from __future__ import annotations

import numpy as np
from scipy.signal import butter, lfilter, resample_poly


def k_weighting_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Approximate K-weighting filter for BS.1770."""
    if sr == 48000:
        b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
        a1 = [1.0, -1.69065929318241, 0.73248077421585]
        b2 = [1.0, -2.0, 1.0]
        a2 = [1.0, -1.99004745483398, 0.99007225036621]

        filtered = lfilter(b1, a1, audio, axis=-1)
        filtered = lfilter(b2, a2, filtered, axis=-1)
        return filtered

    b, a = butter(2, 100 / (sr / 2), btype="highpass")
    return lfilter(b, a, audio, axis=-1)


def _to_lufs(energy: np.ndarray) -> np.ndarray:
    return -0.691 + 10 * np.log10(np.maximum(energy, 1e-10))


def _compute_loudness_range(short_term_lufs: np.ndarray) -> float:
    gated = short_term_lufs[short_term_lufs > -70.0]
    if gated.size < 2:
        return 0.0
    return float(np.percentile(gated, 95) - np.percentile(gated, 10))


def compute_loudness(audio: np.ndarray, sr: int) -> dict[str, float]:
    """Compute integrated, short-term, momentary loudness, and peak features."""
    weighted = k_weighting_filter(audio, sr)

    block_size = int(0.4 * sr)
    overlap = int(0.3 * sr)
    step = block_size - overlap

    if audio.shape[1] < block_size:
        rms = np.sqrt(np.mean(weighted**2))
        lufs = -70.0 if rms < 1e-10 else 20 * np.log10(rms)
        sample_peak_linear = float(np.max(np.abs(audio)))
        true_peak_linear = sample_peak_linear
        sample_peak_dbfs = 20 * np.log10(sample_peak_linear + 1e-10)
        true_peak_dbfs = 20 * np.log10(true_peak_linear + 1e-10)
        return {
            "integrated_lufs": float(lufs),
            "short_term_max_lufs": float(lufs),
            "short_term_mean_lufs": float(lufs),
            "short_term_min_lufs": float(lufs),
            "short_term_range_lu": 0.0,
            "momentary_max_lufs": float(lufs),
            "momentary_mean_lufs": float(lufs),
            "momentary_min_lufs": float(lufs),
            "loudness_range_lu": 0.0,
            "sample_peak_dbfs": float(sample_peak_dbfs),
            "true_peak_dbfs": float(true_peak_dbfs),
        }

    num_blocks = (weighted.shape[1] - block_size) // step + 1
    energies = np.zeros((weighted.shape[0], num_blocks))

    for index in range(num_blocks):
        start = index * step
        end = start + block_size
        energies[:, index] = np.mean(weighted[:, start:end] ** 2, axis=1)

    block_energies = np.sum(energies, axis=0)
    abs_gate_energy = 10 ** (-70.0 / 10.0)
    gated_blocks = block_energies[block_energies > abs_gate_energy]

    if gated_blocks.size == 0:
        integrated = -70.0
    else:
        rel_threshold = np.mean(gated_blocks) * (10 ** (-10.0 / 10.0))
        rel_gated_blocks = gated_blocks[gated_blocks > rel_threshold]
        if rel_gated_blocks.size == 0:
            integrated = -70.0
        else:
            integrated = float(_to_lufs(np.array([np.mean(rel_gated_blocks)]))[0])

    momentary_lufs = _to_lufs(block_energies)

    st_block_size = int(3.0 * sr)
    if audio.shape[1] >= st_block_size:
        st_step = int(1.0 * sr)
        st_num_blocks = (weighted.shape[1] - st_block_size) // st_step + 1
        st_energies = np.zeros(st_num_blocks)
        for index in range(st_num_blocks):
            start = index * st_step
            end = start + st_block_size
            st_energies[index] = np.sum(np.mean(weighted[:, start:end] ** 2, axis=1))
        short_term_lufs = _to_lufs(st_energies)
    else:
        short_term_lufs = np.array([integrated], dtype=float)

    sample_peak_linear = float(np.max(np.abs(audio)))

    if audio.shape[1] > 100:
        peak_idx = int(np.argmax(np.max(np.abs(audio), axis=0)))
        start = max(0, peak_idx - 50)
        end = min(audio.shape[1], peak_idx + 50)
        chunk = audio[:, start:end]
        os_chunk = resample_poly(chunk, 4, 1, axis=-1)
        true_peak_linear = float(np.max(np.abs(os_chunk)))
    else:
        true_peak_linear = sample_peak_linear

    true_peak_linear = max(true_peak_linear, sample_peak_linear)

    return {
        "integrated_lufs": float(integrated),
        "short_term_max_lufs": float(np.max(short_term_lufs)),
        "short_term_mean_lufs": float(np.mean(short_term_lufs)),
        "short_term_min_lufs": float(np.min(short_term_lufs)),
        "short_term_range_lu": float(np.max(short_term_lufs) - np.min(short_term_lufs)),
        "momentary_max_lufs": float(np.max(momentary_lufs)),
        "momentary_mean_lufs": float(np.mean(momentary_lufs)),
        "momentary_min_lufs": float(np.min(momentary_lufs)),
        "loudness_range_lu": _compute_loudness_range(short_term_lufs),
        "sample_peak_dbfs": float(20 * np.log10(sample_peak_linear + 1e-10)),
        "true_peak_dbfs": float(20 * np.log10(true_peak_linear + 1e-10)),
    }
