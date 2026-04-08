import numpy as np
import pytest

import cognis.dsp.limiter as limiter_mod
from cognis.dsp.limiter import Limiter


def test_limiter_respects_ceiling():
    limiter = Limiter(48000)
    audio = np.ones((2, 1000)) * 2.0
    ceiling_db = -1.0
    ceiling_linear = 10 ** (ceiling_db / 20.0)

    processed = limiter.process(audio, ceiling_db, "PEAK", oversampling=1)

    assert np.max(np.abs(processed)) <= ceiling_linear + 1e-5


def test_codec_safe_mode():
    limiter = Limiter(48000)
    audio = np.random.randn(2, 48000)

    processed = limiter.process(audio, -1.0, "CODEC_SAFE", oversampling=1)

    assert processed.shape == audio.shape


def test_oversampling_paths():
    limiter = Limiter(48000)
    audio = np.random.randn(2, 1000)

    processed_1x = limiter.process(audio, -1.0, "PEAK", oversampling=1)
    processed_4x = limiter.process(audio, -1.0, "PEAK", oversampling=4)

    assert processed_1x.shape == audio.shape
    assert processed_4x.shape == audio.shape


def test_limiter_python_only_when_native_unavailable(monkeypatch):
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((2, 48000)) * 2.0

    monkeypatch.setattr(limiter_mod, "NATIVE_AVAILABLE", False)

    limiter = Limiter(48000)
    processed = limiter.process(audio, ceiling_db=-1.0, mode="PEAK")

    assert processed.shape == audio.shape
    assert limiter.last_execution_info["used_native"] is False
    assert limiter.last_execution_info["fallback_triggered"] is False
    assert limiter.last_execution_info["execution_state"] == "python_reference_native_unavailable"
    assert limiter.last_execution_info["native_available"] is False
    assert limiter.last_execution_info["module_imported"] is True


@pytest.mark.skipif(not limiter_mod.NATIVE_AVAILABLE, reason="Native limiter helper is not available")
def test_native_vs_python_equivalence_limiter(monkeypatch):
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((2, 48000)) * 2.0

    monkeypatch.setattr(limiter_mod, "NATIVE_AVAILABLE", False)
    limiter_py = Limiter(48000)
    processed_py = limiter_py.process(audio, ceiling_db=-1.0, mode="PEAK")
    assert limiter_py.last_execution_info["used_native"] is False
    assert limiter_py.last_execution_info["fallback_triggered"] is False
    assert limiter_py.last_execution_info["execution_state"] == "python_reference_native_unavailable"

    monkeypatch.setattr(limiter_mod, "NATIVE_AVAILABLE", True)
    limiter_native = Limiter(48000)
    processed_nat = limiter_native.process(audio, ceiling_db=-1.0, mode="PEAK")

    assert limiter_native.last_execution_info["used_native"] is True
    assert limiter_native.last_execution_info["fallback_triggered"] is False
    assert limiter_native.last_execution_info["execution_state"] == "native_imported_and_used"

    np.testing.assert_allclose(processed_py, processed_nat, rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(not limiter_mod.NATIVE_AVAILABLE, reason="Native limiter helper is not available")
def test_native_limiter_strict_failure(monkeypatch):
    limiter = Limiter(48000)
    audio = np.random.randn(2, 1000)

    monkeypatch.setattr(limiter_mod, "NATIVE_AVAILABLE", False)
    processed_py = Limiter(48000).process(audio, ceiling_db=-1.0, mode="PEAK")

    monkeypatch.setattr(limiter_mod, "NATIVE_AVAILABLE", True)

    def mock_fused(*args, **kwargs):
        raise RuntimeError("Simulated C++ Exception")

    monkeypatch.setattr(limiter_mod.native, "compute_native_limiter_gain_fused", mock_fused)
    monkeypatch.setattr(limiter_mod, "_FALLBACK_ON_NATIVE_FAILURE", False)

    with pytest.raises(RuntimeError, match=r"Native limiter execution failed: Simulated C\+\+ Exception"):
        limiter.process(audio, ceiling_db=-1.0, mode="PEAK")

    assert limiter.last_execution_info["used_native"] is False
    assert limiter.last_execution_info["fallback_triggered"] is False
    assert limiter.last_execution_info["execution_state"] == "unexpected_native_failure"

    monkeypatch.setattr(limiter_mod, "_FALLBACK_ON_NATIVE_FAILURE", True)
    processed_fallback = limiter.process(audio, ceiling_db=-1.0, mode="PEAK")
    assert processed_fallback.shape == audio.shape
    assert limiter.last_execution_info["used_native"] is False
    assert limiter.last_execution_info["fallback_triggered"] is True
    assert limiter.last_execution_info["execution_state"] == "python_fallback_after_native_failure"
    np.testing.assert_allclose(processed_fallback, processed_py, rtol=1e-7, atol=1e-9)


@pytest.mark.skipif(not limiter_mod.NATIVE_AVAILABLE, reason="Native limiter helper is not available")
def test_native_limiter_edge_cases():
    raw_gain = np.ones(100)
    raw_gain[50] = 0.5

    out = limiter_mod.native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), 0, 0.0)
    np.testing.assert_allclose(out, raw_gain)

    out2 = limiter_mod.native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), 1, 0.0)
    np.testing.assert_allclose(out2, raw_gain)


def test_limiter_dtype_preservation():
    limiter = Limiter(48000)
    rng = np.random.default_rng(42)

    audio_f32 = rng.standard_normal((2, 1000)).astype(np.float32)
    processed_f32 = limiter.process(audio_f32, ceiling_db=-1.0, mode="PEAK")
    assert processed_f32.dtype == np.float32

    audio_f64 = rng.standard_normal((2, 1000)).astype(np.float64)
    processed_f64 = limiter.process(audio_f64, ceiling_db=-1.0, mode="PEAK")
    assert processed_f64.dtype == np.float64
