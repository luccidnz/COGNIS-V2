import numpy as np

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


def test_multiband_dynamics_native_fallback_and_equivalence(monkeypatch):
    from cognis.dsp import dynamics

    rng = np.random.default_rng(12)
    audio = rng.standard_normal((2, 24000)) * 0.15
    md = MultibandDynamics(48000)

    # If native is available, let's run it once natively and once forcing python
    if dynamics._NATIVE_AVAILABLE:
        # Run Native
        out_native = md.process(audio, dynamics_preservation=0.4)
        assert md.last_execution_info["used_native"] is True
        assert md.last_execution_info["fallback_triggered"] is False

        # Run Python
        monkeypatch.setattr(dynamics, "_NATIVE_AVAILABLE", False)
        out_python = md.process(audio, dynamics_preservation=0.4)
        assert md.last_execution_info["used_native"] is False
        assert md.last_execution_info["fallback_triggered"] is False

        # Output must be numerically equivalent
        assert np.allclose(out_native, out_python, atol=1e-12)

        # Test explicit fallback handling when native throws
        monkeypatch.setattr(dynamics, "_NATIVE_AVAILABLE", True)

        def mock_compute_gain(*args, **kwargs):
            raise RuntimeError("Mock Native Failure")

        monkeypatch.setattr(dynamics.cognis_native, "compute_native_compressor_gain", mock_compute_gain)

        out_fallback = md.process(audio, dynamics_preservation=0.4)
        assert md.last_execution_info["used_native"] is False
        assert md.last_execution_info["fallback_triggered"] is True
        assert np.allclose(out_fallback, out_python, atol=1e-12)
