import numpy as np
from cognis.dsp.filters import apply_lowpass

class StereoControl:
    def __init__(self, sr: int):
        self.sr = sr
        
    def process(self, audio: np.ndarray, width: float, bass_preservation: float) -> np.ndarray:
        """
        Apply stereo width control and bass monoing.
        width: 0.0 (mono) to 2.0 (extra wide). 1.0 is unchanged.
        bass_preservation: 0.0 (mono bass) to 1.0 (unchanged bass).
        """
        if audio.shape[0] == 1:
            return audio # Mono input, can't do much
            
        L = audio[0]
        R = audio[1]
        
        mid = (L + R) / 2.0
        side = (L - R) / 2.0
        
        # Bass monoing
        if bass_preservation < 1.0:
            # Extract low side
            side_low = apply_lowpass(side, 150.0, self.sr, order=2)
            # Reduce low side
            side = side - side_low * (1.0 - bass_preservation)
            
        # Apply overall width to remaining side
        side = side * width
        
        out_L = mid + side
        out_R = mid - side
        
        return np.vstack((out_L, out_R))
