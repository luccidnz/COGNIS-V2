import numpy as np

from cognis.dsp.filters import split_linear_phase_three_band


def _stereo_tone(freq_hz: float, sr: int, seconds: float, right_gain: float = 0.7) -> np.ndarray:
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    left = np.sin(2.0 * np.pi * freq_hz * t)
    right = right_gain * np.sin(2.0 * np.pi * freq_hz * t + 0.25)
    return np.vstack((left, right))


def test_three_band_split_reconstructs_stereo_noise():
    rng = np.random.default_rng(7)
    audio = rng.standard_normal((2, 48000)) * 0.1

    bands = split_linear_phase_three_band(audio, 250.0, 4000.0, 48000)
    reconstructed = bands.low + bands.mid + bands.high

    assert reconstructed.shape == audio.shape
    assert np.allclose(reconstructed, audio, atol=1e-10)


def test_three_band_split_low_tone_lands_in_low_band():
    audio = _stereo_tone(80.0, 48000, 0.5)
    bands = split_linear_phase_three_band(audio, 250.0, 4000.0, 48000)

    low_rms = np.sqrt(np.mean(bands.low ** 2))
    mid_rms = np.sqrt(np.mean(bands.mid ** 2))
    high_rms = np.sqrt(np.mean(bands.high ** 2))

    assert low_rms > mid_rms * 8.0
    assert low_rms > high_rms * 100.0


def test_three_band_split_high_tone_lands_in_high_band():
    audio = _stereo_tone(8000.0, 48000, 0.5)
    bands = split_linear_phase_three_band(audio, 250.0, 4000.0, 48000)

    low_rms = np.sqrt(np.mean(bands.low ** 2))
    mid_rms = np.sqrt(np.mean(bands.mid ** 2))
    high_rms = np.sqrt(np.mean(bands.high ** 2))

    assert high_rms > mid_rms * 4.0
    assert high_rms > low_rms * 100.0
