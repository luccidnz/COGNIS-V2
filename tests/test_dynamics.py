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
