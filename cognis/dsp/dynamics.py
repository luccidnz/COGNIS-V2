import numpy as np
from cognis.dsp.filters import apply_lowpass, apply_highpass

class MultibandDynamics:
    def __init__(self, sr: int):
        self.sr = sr
        
    def _compress_band(self, band: np.ndarray, threshold_db: float, ratio: float, attack_ms: float, release_ms: float) -> np.ndarray:
        """Simple feed-forward compressor."""
        # Envelope follower
        attack_coef = np.exp(-1.0 / (self.sr * attack_ms / 1000.0))
        release_coef = np.exp(-1.0 / (self.sr * release_ms / 1000.0))
        
        env = np.zeros_like(band)
        gain = np.ones_like(band)
        
        # Mix to mono for sidechain
        sc = np.abs(np.mean(band, axis=0)) if band.shape[0] > 1 else np.abs(band[0])
        
        # Process envelope
        curr_env = 0.0
        for i in range(len(sc)):
            if sc[i] > curr_env:
                curr_env = attack_coef * curr_env + (1 - attack_coef) * sc[i]
            else:
                curr_env = release_coef * curr_env + (1 - release_coef) * sc[i]
            
            env_db = 20 * np.log10(curr_env + 1e-10)
            
            if env_db > threshold_db:
                overshoot = env_db - threshold_db
                reduction = overshoot * (1.0 - 1.0 / ratio)
                gain_linear = 10 ** (-reduction / 20.0)
            else:
                gain_linear = 1.0
                
            # Apply to all channels
            gain[:, i] = gain_linear
            
        return band * gain

    def process(self, audio: np.ndarray, dynamics_preservation: float) -> np.ndarray:
        """
        Apply multiband dynamics.
        dynamics_preservation: 0.0 (heavy compression) to 1.0 (no compression).
        """
        if dynamics_preservation >= 0.99:
            return audio
            
        # Split into 3 bands
        lows = apply_lowpass(audio, 250.0, self.sr, order=2)
        highs = apply_highpass(audio, 4000.0, self.sr, order=2)
        mids = audio - lows - highs
        
        # Map preservation to ratio and threshold
        # 0.0 -> ratio 4.0, thresh -20
        # 1.0 -> ratio 1.0, thresh 0
        ratio = 1.0 + 3.0 * (1.0 - dynamics_preservation)
        threshold_db = -20.0 * (1.0 - dynamics_preservation)
        
        comp_lows = self._compress_band(lows, threshold_db, ratio, 30.0, 150.0)
        comp_mids = self._compress_band(mids, threshold_db, ratio, 10.0, 100.0)
        comp_highs = self._compress_band(highs, threshold_db, ratio, 5.0, 50.0)
        
        return comp_lows + comp_mids + comp_highs
