import numpy as np
from cognis.engine import Engine
from cognis.config import MasteringConfig, MasteringMode, CeilingMode

def test_engine_process():
    engine = Engine()
    config = MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=1, # Keep fast for tests
        bass_preservation=1.0,
        stereo_width=1.0,
        dynamics_preservation=1.0,
        brightness=0.0
    )
    
    # Small synthetic stereo signal
    t = np.linspace(0, 0.5, 24000, endpoint=False)
    audio = np.vstack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t))) * 0.5
    
    mastered, report, recipe = engine.process(audio, 48000, config)
    
    assert mastered.shape == audio.shape
    assert report is not None
    assert hasattr(report, 'integrated_loudness')
    assert recipe is not None
    assert "params" in recipe


def test_engine_process_with_multiband_dynamics_enabled():
    engine = Engine()
    config = MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=1,
        bass_preservation=0.9,
        stereo_width=1.1,
        dynamics_preservation=0.45,
        brightness=0.1
    )

    t = np.linspace(0, 0.5, 24000, endpoint=False)
    left = 0.45 * np.sin(2 * np.pi * 90 * t) + 0.2 * np.sin(2 * np.pi * 2200 * t)
    right = 0.4 * np.sin(2 * np.pi * 140 * t + 0.2) + 0.15 * np.sin(2 * np.pi * 7000 * t)
    audio = np.vstack((left, right))

    mastered, report, recipe = engine.process(audio, 48000, config)

    assert mastered.shape == audio.shape
    assert np.isfinite(mastered).all()
    assert report is not None
    assert recipe is not None


def test_engine_process_with_different_backends():
    engine = Engine()

    t = np.linspace(0, 0.5, 24000, endpoint=False)
    audio = np.vstack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t))) * 0.5

    def run_with_backend(backend: str):
        config = MasteringConfig(
            mode=MasteringMode.STREAMING_SAFE,
            target_loudness=-14.0,
            ceiling_mode=CeilingMode.TRUE_PEAK,
            ceiling_db=-1.0,
            oversampling=1,
            bass_preservation=0.9,
            stereo_width=1.0,
            dynamics_preservation=0.5,
            brightness=0.1,
            fir_backend=backend
        )
        # Using a fixed single render instead of full optimization loop to save time,
        # but since Engine.process does optimization, we'll let it run on a very small space
        mastered, _, _ = engine.process(audio, 48000, config)
        return mastered

    out_auto = run_with_backend("AUTO")
    out_fft = run_with_backend("FFT")

    # We won't test DIRECT on full engine run as it's very slow for 24000 samples and 1537 taps.
    assert np.allclose(out_auto, out_fft, atol=1e-10)
