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

    # Use DIRECT since we don't have native DIRECT support
    out = execute_fir_2d(audio, taps, FirBackend.DIRECT)
    assert out.shape == audio.shape

    info = get_fir_execution_info()
    assert info["used_native"] is False
    assert info["fallback_triggered"] is False # It wasn't a native crash fallback, just purely unsupported/unavailable so it ran pure python
    assert info["selected_method"] == "direct"
    assert info["execution_state"] in {
        "python_reference_native_unavailable",
        "python_fallback_intentional_unsupported_mode",
    }


def test_execute_fir_2d_python_fallback_when_native_unavailable(monkeypatch):
    rng = np.random.default_rng(101)
    audio = rng.standard_normal((2, 2048)).astype(np.float64)
    taps = rng.standard_normal(65).astype(np.float64)

    monkeypatch.setattr(fir_exec_mod, "_NATIVE_FIR_AVAILABLE", False)
    monkeypatch.setattr(fir_exec_mod, "_cognis_native", None)

    out = execute_fir_2d(audio, taps, FirBackend.FFT)
    info = get_fir_execution_info()

    assert out.shape == audio.shape
    assert out.dtype == np.float64
    assert info["used_native"] is False
    assert info["fallback_triggered"] is False
    assert info["selected_method"] == "fft"
    assert info["execution_state"] == "python_reference_native_unavailable"
    np.testing.assert_allclose(out, execute_python_fir_2d(audio, taps, FirBackend.FFT), rtol=1e-12, atol=1e-12)


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
    assert info["execution_state"] == "native_imported_and_used"

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_failure_semantics():
    # We can trigger a native failure by simulating a crash in the native layer.
    audio = np.random.randn(2, 1024).astype(np.float64)
    taps = np.random.randn(65).astype(np.float64)

    # Ensure fallback is off for this test
    original_fallback = fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE
    fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE = False

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
            fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE = original_fallback

    # Prove that the state reflects no native usage and no fallback recovery
    info = get_fir_execution_info()
    assert info["used_native"] is False
    assert info["fallback_triggered"] is False
    assert info["selected_method"] == "fft"
    assert info["execution_state"] == "unexpected_native_failure"

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_failure_fallback_semantics():
    # Test that fallback works when _FALLBACK_ON_NATIVE_FAILURE is True
    audio = np.random.randn(2, 1024).astype(np.float64)
    taps = np.random.randn(65).astype(np.float64)

    original_execute = fir_exec_mod._cognis_native.execute_native_fir_2d
    original_fallback = fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE
    fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE = True

    def mock_execute(*args, **kwargs):
        raise Exception("Simulated Native Crash")
    fir_exec_mod._cognis_native.execute_native_fir_2d = mock_execute

    try:
        out = fir_exec_mod.execute_fir_2d(audio, taps, FirBackend.FFT)
        info = get_fir_execution_info()
        assert info["used_native"] is False
        assert info["fallback_triggered"] is True
        assert info["execution_state"] == "python_fallback_after_native_failure"
        assert out.shape == audio.shape
    finally:
        fir_exec_mod._cognis_native.execute_native_fir_2d = original_execute
        fir_exec_mod._FALLBACK_ON_NATIVE_FAILURE = original_fallback


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

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_partitioned_matches_python_partitioned():
    """Ensure that the native PARTITIONED backend exactly matches the Python reference implementation."""
    np.random.seed(43)
    # Use a long enough audio signal to cross block boundaries
    audio = np.random.randn(2, 8192).astype(np.float64)
    taps = np.random.randn(129).astype(np.float64)

    # Calculate native output
    out_native = execute_fir_2d(audio, taps, FirBackend.PARTITIONED)
    assert get_fir_execution_info()["used_native"] is True

    # Calculate python reference output
    out_python = execute_python_fir_2d(audio, taps, FirBackend.PARTITIONED)

    np.testing.assert_allclose(out_native, out_python, rtol=1e-5, atol=1e-7)
    assert out_native.shape == audio.shape

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_fft_multichannel_separation():
    """Ensure that the native FFT correctly processes multiple channels without cross-talk."""
    np.random.seed(42)
    # Channel 0 has signal, Channel 1 is completely silent
    audio = np.zeros((2, 1024), dtype=np.float64)
    audio[0, :] = np.random.randn(1024)

    taps = np.random.randn(65).astype(np.float64)

    out_native = execute_fir_2d(audio, taps, FirBackend.FFT)
    assert get_fir_execution_info()["used_native"] is True

    # Check that channel 1 remained completely silent
    np.testing.assert_allclose(out_native[1, :], 0.0, atol=1e-12)

    # Check that channel 0 was processed
    assert not np.allclose(out_native[0, :], 0.0)

    out_python = execute_python_fir_2d(audio, taps, FirBackend.FFT)
    np.testing.assert_allclose(out_native[0, :], out_python[0, :], rtol=1e-5, atol=1e-7)

@pytest.mark.skipif(not _NATIVE_FIR_AVAILABLE, reason="Native FIR extension is not available")
def test_execute_fir_2d_native_fft_alignment():
    """Ensure padding/shift alignment exactly matches Python's mode='same' semantics for symmetric kernels."""
    # An impulse should simply delay the signal according to mode="same"
    # Actually, if we apply a Dirac delta centered at tap N/2, it shouldn't delay at all under mode="same".
    audio = np.zeros((1, 512), dtype=np.float64)
    audio[0, 256] = 1.0 # Impulse in the middle

    taps = np.zeros(65, dtype=np.float64)
    taps[32] = 1.0 # Impulse at the center tap (assuming odd length)

    out_native = execute_fir_2d(audio, taps, FirBackend.FFT)
    out_python = execute_python_fir_2d(audio, taps, FirBackend.FFT)

    # Verify impulse hasn't shifted compared to Python
    assert np.argmax(out_native[0]) == 256
    np.testing.assert_allclose(out_native, out_python, rtol=1e-5, atol=1e-7)
