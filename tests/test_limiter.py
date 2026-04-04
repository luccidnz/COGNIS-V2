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
