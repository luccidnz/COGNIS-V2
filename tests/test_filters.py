import numpy as np
from scipy.signal import fftconvolve

import pytest

from cognis.dsp.filters import (
    FirBackend,
    apply_fir,
    clear_fir_design_cache,
    get_fir_design_cache_info,
    get_linear_phase_three_band_splitter,
    split_linear_phase_three_band,
)


def _stereo_tone(freq_hz: float, sr: int, seconds: float, right_gain: float = 0.7) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    left = np.sin(2.0 * np.pi * freq_hz * t)
    right = right_gain * np.sin(2.0 * np.pi * freq_hz * t + 0.25)
    return np.vstack((left, right))


def _band_rms(split) -> tuple[float, float, float]:
    return tuple(float(np.sqrt(np.mean(band ** 2))) for band in (split.low, split.mid, split.high))


def test_three_band_split_reconstructs_stereo_noise():
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((2, 48000)) * 0.1

    bands = split_linear_phase_three_band(audio, 250.0, 4000.0, 48000)
    reconstructed = bands.low + bands.mid + bands.high

    assert reconstructed.shape == audio.shape
    assert np.allclose(reconstructed, audio, atol=1e-10)


def test_apply_fir_matches_reference_fft_convolution():
    rng = np.random.default_rng(5)
    audio = rng.standard_normal((2, 4096))
    taps = rng.standard_normal(257)

    filtered = apply_fir(audio, taps)
    reference = np.vstack([fftconvolve(channel, taps, mode="same") for channel in audio])

    assert np.allclose(filtered, reference, atol=1e-12)


def test_apply_fir_backends_are_equivalent():
    rng = np.random.default_rng(6)
    audio = rng.standard_normal((2, 4096))
    taps = rng.standard_normal(257)

    auto_out = apply_fir(audio, taps, backend=FirBackend.AUTO)
    direct_out = apply_fir(audio, taps, backend=FirBackend.DIRECT)
    fft_out = apply_fir(audio, taps, backend=FirBackend.FFT)

    assert np.allclose(auto_out, direct_out, atol=1e-12)
    assert np.allclose(auto_out, fft_out, atol=1e-12)


def test_apply_fir_backends_are_equivalent_long_signal():
    rng = np.random.default_rng(6)
    audio = rng.standard_normal((2, 48000))
    taps = rng.standard_normal(1025)

    auto_out = apply_fir(audio, taps, backend=FirBackend.AUTO)
    fft_out = apply_fir(audio, taps, backend=FirBackend.FFT)

    assert np.allclose(auto_out, fft_out, atol=1e-12)

def test_auto_backend_heuristic():
    from cognis.dsp.filters import _choose_backend_method
    # Signal with NaN should fallback to direct
    assert _choose_backend_method(48000, 1025, 2, has_nan_inf=True) == "direct"

    # Short signal, short kernel
    assert _choose_backend_method(512, 64, 2) == "direct"

    # Long signal, short kernel
    assert _choose_backend_method(4096, 64, 2) == "fft"

    # Short signal, long kernel
    assert _choose_backend_method(512, 256, 2) == "fft"

    # Long signal, long kernel -> goes to partitioned now
    assert _choose_backend_method(48000, 1025, 2) == "partitioned"

    # Long signal but short kernel
    assert _choose_backend_method(48000, 64, 2) == "fft"


def test_partitioned_backend_equivalent_to_fft_backend():
    rng = np.random.default_rng(8)
    # Long signal to fully test block-based overlap-save overlap processing
    audio = rng.standard_normal((2, 16000))
    taps = rng.standard_normal(1025)

    fft_out = apply_fir(audio, taps, backend=FirBackend.FFT)
    part_out = apply_fir(audio, taps, backend=FirBackend.PARTITIONED)

    assert part_out.shape == audio.shape
    assert np.allclose(fft_out, part_out, atol=1e-10)


def test_three_band_split_low_tone_lands_in_low_band():
    low_rms, mid_rms, high_rms = _band_rms(split_linear_phase_three_band(_stereo_tone(80.0, 48000, 0.5), 250.0, 4000.0, 48000))

    assert low_rms > mid_rms * 8.0
    assert low_rms > high_rms * 100.0


def test_three_band_split_mid_tone_lands_in_mid_band():
    low_rms, mid_rms, high_rms = _band_rms(split_linear_phase_three_band(_stereo_tone(1000.0, 48000, 0.5), 250.0, 4000.0, 48000))

    assert mid_rms > low_rms * 8.0
    assert mid_rms > high_rms * 8.0


def test_three_band_split_high_tone_lands_in_high_band():
    low_rms, mid_rms, high_rms = _band_rms(split_linear_phase_three_band(_stereo_tone(8000.0, 48000, 0.5), 250.0, 4000.0, 48000))

    assert high_rms > mid_rms * 4.0
    assert high_rms > low_rms * 100.0


def test_three_band_split_behaves_sensibly_near_crossovers():
    near_low = split_linear_phase_three_band(_stereo_tone(250.0, 48000, 0.5), 250.0, 4000.0, 48000)
    near_high = split_linear_phase_three_band(_stereo_tone(4000.0, 48000, 0.5), 250.0, 4000.0, 48000)

    low_rms, mid_rms, high_rms = _band_rms(near_low)
    assert low_rms > high_rms * 20.0
    assert low_rms > 0.05
    assert mid_rms > 0.05

    low_rms, mid_rms, high_rms = _band_rms(near_high)
    assert high_rms > low_rms * 20.0
    assert high_rms > 0.05
    assert mid_rms > 0.05


def test_three_band_split_broadband_signal_stays_finite():
    rng = np.random.default_rng(19)
    t = np.linspace(0, 0.5, 24000, endpoint=False)
    tonal = np.vstack(
        (
            0.3 * np.sin(2.0 * np.pi * 90.0 * t) + 0.2 * np.sin(2.0 * np.pi * 1600.0 * t),
            0.25 * np.sin(2.0 * np.pi * 7000.0 * t + 0.3),
        )
    )
    audio = tonal + rng.standard_normal((2, t.size)) * 0.02

    bands = split_linear_phase_three_band(audio, 250.0, 4000.0, 48000)
    rms = _band_rms(bands)

    assert np.isfinite(bands.low).all()
    assert np.isfinite(bands.mid).all()
    assert np.isfinite(bands.high).all()
    assert all(value > 1e-3 for value in rms)


def test_three_band_splitter_cache_reuses_designs():
    clear_fir_design_cache()

    splitter_a = get_linear_phase_three_band_splitter(48000, 250.0, 4000.0, low_taps=1537, high_taps=513)
    first_info = get_fir_design_cache_info()
    splitter_b = get_linear_phase_three_band_splitter(48000, 250.0, 4000.0, low_taps=1537, high_taps=513)
    second_info = get_fir_design_cache_info()

    assert splitter_a is splitter_b
    assert splitter_a.low_taps is splitter_b.low_taps
    assert splitter_a.high_taps is splitter_b.high_taps
    assert second_info["splitter"]["hits"] == first_info["splitter"]["hits"] + 1
    assert second_info["fir"]["currsize"] == first_info["fir"]["currsize"]
