import numpy as np
from scipy.signal import resample_poly
from cognis.dsp.filters import apply_lowpass

class Limiter:
    def __init__(self, sr: int):
        self.sr = sr
        
    def process(self, audio: np.ndarray, ceiling_db: float, mode: str, oversampling: int = 1) -> np.ndarray:
        """
        Apply peak limiting.
        For MVP, this is a static waveshaper/clipper with optional oversampling and codec-safe LPF.
        TODO: Implement true lookahead envelope-based limiter for Phase 2.
        """
        ceiling_linear = 10 ** (ceiling_db / 20.0)
        
        if mode == "CODEC_SAFE":
            # Gentle LPF at 18kHz to prevent codec ringing
            audio = apply_lowpass(audio, 18000.0, self.sr, order=2)
            
        if oversampling > 1:
            # Oversample
            os_audio = resample_poly(audio, oversampling, 1, axis=-1)
            
            # Soft clip / limit
            # Simple tanh waveshaper for MVP
            gain = ceiling_linear / np.max(np.abs(os_audio) + 1e-10)
            if gain < 1.0:
                os_audio = np.tanh(os_audio * (1.0 / ceiling_linear)) * ceiling_linear
                
            # Downsample
            audio = resample_poly(os_audio, 1, oversampling, axis=-1)
        else:
            # Simple clip
            gain = ceiling_linear / np.max(np.abs(audio) + 1e-10)
            if gain < 1.0:
                audio = np.tanh(audio * (1.0 / ceiling_linear)) * ceiling_linear
                
        # Hard ceiling enforcement just in case
        audio = np.clip(audio, -ceiling_linear, ceiling_linear)
        
        return audio
