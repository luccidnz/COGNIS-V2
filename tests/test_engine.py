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
