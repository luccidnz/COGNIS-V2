import numpy as np
import pytest

from cognis.dsp.fir_executor import FirBackend, execute_fir_2d, _NATIVE_FIR_AVAILABLE, execute_python_fir_2d, get_fir_execution_info
import cognis.dsp.fir_executor as fir_exec_mod

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

    info = get_fir_execution_info()
    assert info["used_native"] is False
    assert info["selected_method"] == "partitioned"

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_observability_and_native_execution():
    audio = np.random.randn(2, 1024).astype(np.float64)
    taps = np.random.randn(65).astype(np.float64)

    # Force native FFT
    out = execute_fir_2d(audio, taps, FirBackend.FFT)
    info = get_fir_execution_info()
    assert info["used_native"] is True
    assert info["fallback_triggered"] is False
    assert info["selected_method"] == "fft"

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_failure_semantics():
    # We can trigger a native failure by passing an unsupported method to the native layer.
    # Actually, the python layer protects against this.
    # Let's temporarily inject a bad method to test the exception throwing.
    audio = np.random.randn(2, 1024).astype(np.float64)
    taps = np.random.randn(65).astype(np.float64)

    # This shouldn't normally happen since the boundary filters it, but if Native raises an error:
    with pytest.raises(RuntimeError, match="Native FIR execution failed"):
        # To test the boundary logic (which wraps the native throw in a new RuntimeError),
        # we mock the native call to throw something we can catch.
        original_execute = fir_exec_mod._cognis_native.execute_native_fir_2d
        def mock_execute(*args, **kwargs):
            raise Exception("Simulated Native Crash")
        fir_exec_mod._cognis_native.execute_native_fir_2d = mock_execute
        try:
            fir_exec_mod.execute_fir_2d(audio, taps, FirBackend.FFT)
        finally:
            fir_exec_mod._cognis_native.execute_native_fir_2d = original_execute


@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_fft_matches_python_fft():
    np.random.seed(42)
    audio = np.random.randn(2, 1024).astype(np.float64)
    taps = np.random.randn(65).astype(np.float64)

    # Calculate native output
    out_native = execute_fir_2d(audio, taps, FirBackend.FFT)
    assert get_fir_execution_info()["used_native"] is True

    # Calculate python reference output
    out_python = execute_python_fir_2d(audio, taps, FirBackend.FFT)

    # The FFT approach might have minor floating point discrepancies,
    # but they should be effectively identical
    np.testing.assert_allclose(out_native, out_python, rtol=1e-5, atol=1e-7)

    # Assert same mode exactly matches input shape
    assert out_native.shape == audio.shape
