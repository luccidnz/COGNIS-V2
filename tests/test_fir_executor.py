import numpy as np
import pytest

from cognis.dsp.fir_executor import FirBackend, execute_fir_2d

def test_execute_fir_2d_enforces_dimensionality():
    audio_1d = np.zeros(10)
    taps = np.zeros(3)

    with pytest.raises(ValueError, match="audio_2d must be 2-dimensional"):
        execute_fir_2d(audio_1d, taps, FirBackend.AUTO)

    audio_2d = np.zeros((2, 10))
    taps_2d = np.zeros((2, 3))

    with pytest.raises(ValueError, match="taps must be 1-dimensional"):
        execute_fir_2d(audio_2d, taps_2d, FirBackend.AUTO)

def test_execute_fir_2d_preserves_shape_and_type():
    audio = np.random.randn(2, 512).astype(np.float32) # force wrong type
    taps = np.random.randn(33).astype(np.float32)

    # Transpose to break C-contiguity
    audio_non_contig = audio.T.copy().T
    assert not audio_non_contig.flags['C_CONTIGUOUS']

    out = execute_fir_2d(audio_non_contig, taps, FirBackend.AUTO)

    assert out.shape == audio.shape
    assert out.dtype == np.float64
    assert out.flags['C_CONTIGUOUS']

def test_execute_fir_2d_fallback_works_cleanly():
    # Even if native module is absent, it should correctly execute Python fallback
    audio = np.random.randn(2, 1024)
    taps = np.random.randn(129)

    out = execute_fir_2d(audio, taps, FirBackend.PARTITIONED)
    assert out.shape == audio.shape
