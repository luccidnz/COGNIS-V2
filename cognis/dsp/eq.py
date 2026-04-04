import numpy as np
from cognis.dsp.filters import apply_lowpass, apply_highpass

class EQ:
    def __init__(self, sr: int):
        self.sr = sr
        
    def process(self, audio: np.ndarray, brightness: float) -> np.ndarray:
        """
        Apply simple corrective EQ.
        brightness: -1.0 to 1.0. Positive means brighter, negative means darker.
        For MVP, we use a simple tilt-like approach by blending HPF and LPF.
        """
        if brightness == 0.0:
            return audio
            
        # Simple tilt approximation
        lows = apply_lowpass(audio, 1000.0, self.sr, order=1)
        highs = audio - lows  # Complementary
        
        # Scale
        if brightness > 0:
            high_gain = 1.0 + brightness * 0.5  # Max +3.5dB approx
            low_gain = 1.0 - brightness * 0.2
        else:
            high_gain = 1.0 + brightness * 0.5
            low_gain = 1.0 - brightness * 0.2
            
        return lows * low_gain + highs * high_gain
