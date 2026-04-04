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
