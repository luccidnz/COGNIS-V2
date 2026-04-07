import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from cognis.dsp.filters import apply_lowpass

class Limiter:
    def __init__(self, sr: int):
        self.sr = sr
        
    def process(self, audio: np.ndarray, ceiling_db: float, mode: str, oversampling: int = 1) -> np.ndarray:
        """
        Apply peak limiting.
        This is an envelope-aware quasi-lookahead limiter.
        
        TODO: Implement a more sophisticated true lookahead limiter with 
        smarter multi-stage release handling for Phase 2 (C++ migration).
        """
        ceiling_linear = 10 ** (ceiling_db / 20.0)
        
        if mode == "CODEC_SAFE":
            # Gentle LPF at 18kHz to prevent codec ringing
            audio = apply_lowpass(audio, 18000.0, self.sr, order=2)
            
        if oversampling > 1:
            # Oversample
            audio = resample_poly(audio, oversampling, 1, axis=-1)
            fs = self.sr * oversampling
        else:
            fs = self.sr
            
        # --- Envelope-aware Limiting ---
        if audio.ndim > 1:
            abs_signal = np.max(np.abs(audio), axis=0)
        else:
            abs_signal = np.abs(audio)
            
        # 1. Compute raw gain reduction needed to hit ceiling
        raw_gain = np.ones_like(abs_signal)
        mask = abs_signal > ceiling_linear
        raw_gain[mask] = ceiling_linear / abs_signal[mask]
        
        # 2. Hold the gain reduction (sustain)
        # Widens the dips so smoothing doesn't under-reduce at the exact peak
        hold_ms = 1.5
        hold_samples = max(1, int((hold_ms / 1000.0) * fs))
        held_gain = minimum_filter1d(raw_gain, size=hold_samples)
        
        # 3. Smooth the gain reduction envelope (attack / release)
        # Gaussian filter provides a smooth, symmetric transition (quasi-lookahead)
        # TODO(optimization): This gaussian_filter1d call is the secondary bottleneck
        # in the DSP chain. Evaluate whether it can be replaced by cascading simpler
        # native IIR passes or accelerated in C++ during a future phase.
        release_ms = 10.0
        sigma_samples = (release_ms / 1000.0) * fs
        smooth_gain = gaussian_filter1d(held_gain, sigma=sigma_samples)
        
        # 4. Guarantee ceiling is met before hard clip
        # Ensures we never overshoot due to smoothing averaging
        final_gain = np.minimum(raw_gain, smooth_gain)
        
        # Apply gain reduction
        audio = audio * final_gain
        
        if oversampling > 1:
            # Downsample
            audio = resample_poly(audio, 1, oversampling, axis=-1)
            
        # Hard ceiling enforcement just in case (catches inter-sample peaks or overshoot)
        audio = np.clip(audio, -ceiling_linear, ceiling_linear)
        
        return audio
