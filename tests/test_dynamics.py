import numpy as np
import pytest

import cognis.dsp.dynamics as dynamics_mod
from cognis.dsp.dynamics import MultibandDynamics


def test_multiband_dynamics_is_deterministic_and_finite():
    rng = np.random.default_rng(11)
    audio = rng.standard_normal((2, 24000)) * 0.15
    dynamics = MultibandDynamics(48000)

    first = dynamics.process(audio, dynamics_preservation=0.4)
    second = dynamics.process(audio, dynamics_preservation=0.4)

    assert first.shape == audio.shape
    assert np.isfinite(first).all()
    assert np.allclose(first, second, atol=1e-12)


def test_multiband_dynamics_reuses_cached_splitter_configuration():
    first = MultibandDynamics(48000)
    second = MultibandDynamics(48000)

    assert first._splitter is second._splitter


def test_multiband_dynamics_python_only_when_native_unavailable(monkeypatch):
    rng = np.random.default_rng(12)
    audio = rng.standard_normal((2, 24000)) * 0.15
    md = MultibandDynamics(48000)

    monkeypatch.setattr(dynamics_mod, "_NATIVE_AVAILABLE", False)

    first = md.process(audio, dynamics_preservation=0.4)
    second = md.process(audio, dynamics_preservation=0.4)

    assert md.last_execution_info["used_native"] is False
    assert md.last_execution_info["fallback_triggered"] is False
    assert md.last_execution_info["execution_state"] == "python_reference_native_unavailable"
    assert md.last_execution_info["native_available"] is False
    assert md.last_execution_info["module_imported"] is True
    assert np.allclose(first, second, atol=1e-12)
    assert np.isfinite(first).all()


@pytest.mark.skipif(not dynamics_mod._NATIVE_AVAILABLE, reason="Native dynamics helper is not available")
def test_multiband_dynamics_native_matches_python(monkeypatch):
    rng = np.random.default_rng(13)
    audio = rng.standard_normal((2, 24000)) * 0.15
    md = MultibandDynamics(48000)

    out_native = md.process(audio, dynamics_preservation=0.4)
    assert md.last_execution_info["used_native"] is True
    assert md.last_execution_info["fallback_triggered"] is False
    assert md.last_execution_info["execution_state"] == "native_imported_and_used"

    monkeypatch.setattr(dynamics_mod, "_NATIVE_AVAILABLE", False)
    out_python = md.process(audio, dynamics_preservation=0.4)
    assert md.last_execution_info["used_native"] is False
    assert md.last_execution_info["fallback_triggered"] is False
    assert md.last_execution_info["execution_state"] == "python_reference_native_unavailable"

    np.testing.assert_allclose(out_native, out_python, atol=1e-12)


@pytest.mark.skipif(not dynamics_mod._NATIVE_AVAILABLE, reason="Native dynamics helper is not available")
def test_multiband_dynamics_native_failure_semantics(monkeypatch):
    rng = np.random.default_rng(14)
    audio = rng.standard_normal((2, 24000)) * 0.15
    md = MultibandDynamics(48000)

    monkeypatch.setattr(dynamics_mod, "_NATIVE_AVAILABLE", False)
    out_python = md.process(audio, dynamics_preservation=0.4)
    assert md.last_execution_info["used_native"] is False
    assert md.last_execution_info["fallback_triggered"] is False

    monkeypatch.setattr(dynamics_mod, "_NATIVE_AVAILABLE", True)

    def mock_compute_gain(*args, **kwargs):
        raise RuntimeError("Mock Native Failure")

    monkeypatch.setattr(dynamics_mod.cognis_native, "compute_native_compressor_gain", mock_compute_gain)

    with pytest.raises(RuntimeError, match="Native dynamics execution failed: Mock Native Failure"):
        md.process(audio, dynamics_preservation=0.4)

    assert md.last_execution_info["used_native"] is False
    assert md.last_execution_info["fallback_triggered"] is False

    monkeypatch.setattr(dynamics_mod, "_FALLBACK_ON_NATIVE_FAILURE", True)
    out_fallback = md.process(audio, dynamics_preservation=0.4)
    assert md.last_execution_info["used_native"] is False
    assert md.last_execution_info["fallback_triggered"] is True
    assert md.last_execution_info["execution_state"] == "python_fallback_after_native_failure"
    np.testing.assert_allclose(out_fallback, out_python, atol=1e-12)
