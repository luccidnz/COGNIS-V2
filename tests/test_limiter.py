import numpy as np
from cognis.dsp.limiter import Limiter

def test_limiter_respects_ceiling():
    limiter = Limiter(48000)
    # Audio exceeding ceiling
    audio = np.ones((2, 1000)) * 2.0
    ceiling_db = -1.0
    ceiling_linear = 10 ** (ceiling_db / 20.0)
    
    processed = limiter.process(audio, ceiling_db, "PEAK", oversampling=1)
    
    assert np.max(np.abs(processed)) <= ceiling_linear + 1e-5

def test_codec_safe_mode():
    limiter = Limiter(48000)
    audio = np.random.randn(2, 48000)
    
    try:
        processed = limiter.process(audio, -1.0, "CODEC_SAFE", oversampling=1)
        success = True
    except Exception:
        success = False
        
    assert success
    assert processed.shape == audio.shape

def test_oversampling_paths():
    limiter = Limiter(48000)
    audio = np.random.randn(2, 1000)
    
    processed_1x = limiter.process(audio, -1.0, "PEAK", oversampling=1)
    processed_4x = limiter.process(audio, -1.0, "PEAK", oversampling=4)
    
    assert processed_1x.shape == audio.shape
    assert processed_4x.shape == audio.shape

def test_native_vs_python_equivalence_limiter():
    import cognis.dsp.limiter as lim

    # Force python
    orig_available = lim.NATIVE_AVAILABLE
    lim.NATIVE_AVAILABLE = False

    limiter_py = Limiter(48000)
    rng = np.random.default_rng(42)
    # Give it some peaks that exceed ceiling to trigger gain reduction
    audio = rng.standard_normal((2, 48000)) * 2.0

    processed_py = limiter_py.process(audio, ceiling_db=-1.0, mode="PEAK")

    # Restore native
    lim.NATIVE_AVAILABLE = orig_available
    if not lim.NATIVE_AVAILABLE:
        return # Skip if native not available

    limiter_native = Limiter(48000)
    processed_nat = limiter_native.process(audio, ceiling_db=-1.0, mode="PEAK")

    assert limiter_native.last_execution_info["used_native"] is True

    # Tolerance relaxed due to FFT convolution vs spatial correlation drift
    np.testing.assert_allclose(processed_py, processed_nat, rtol=1e-5, atol=1e-5)

def test_native_limiter_strict_failure():
    import cognis.dsp.limiter as lim
    import pytest

    if not lim.NATIVE_AVAILABLE:
        return

    limiter = Limiter(48000)
    # Create invalid raw_gain shape to trigger native C++ exception
    audio = np.random.randn(2, 1000)

    # We will mock the C++ fused call to raise an error
    import cognis.dsp.cognis_native as native
    original_fused = native.compute_native_limiter_gain_fused

    def mock_fused(*args, **kwargs):
        raise RuntimeError("Simulated C++ Exception")

    native.compute_native_limiter_gain_fused = mock_fused

    try:
        # Should raise by default
        lim._FALLBACK_ON_NATIVE_FAILURE = False
        with pytest.raises(RuntimeError, match=r"Native limiter execution failed: Simulated C\+\+ Exception"):
            limiter.process(audio, ceiling_db=-1.0, mode="PEAK")

        assert limiter.last_execution_info["fallback_triggered"] is True

        # Test explicit fallback mode
        lim._FALLBACK_ON_NATIVE_FAILURE = True
        processed = limiter.process(audio, ceiling_db=-1.0, mode="PEAK")
        assert processed.shape == audio.shape
        assert limiter.last_execution_info["fallback_triggered"] is True
        assert limiter.last_execution_info["used_native"] is False
    finally:
        native.compute_native_limiter_gain_fused = original_fused
        lim._FALLBACK_ON_NATIVE_FAILURE = False

def test_native_limiter_edge_cases():
    import cognis.dsp.limiter as lim
    if not lim.NATIVE_AVAILABLE:
        return

    import cognis.dsp.cognis_native as native

    # Test hold=0, sigma=0
    raw_gain = np.ones(100)
    raw_gain[50] = 0.5

    out = native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), 0, 0.0)
    np.testing.assert_allclose(out, raw_gain)

    out2 = native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), 1, 0.0)
    np.testing.assert_allclose(out2, raw_gain)
