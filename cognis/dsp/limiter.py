import logging
import numpy as np
from scipy.signal import resample_poly
from scipy.ndimage import gaussian_filter1d, minimum_filter1d
from cognis.dsp.filters import apply_lowpass

logger = logging.getLogger(__name__)

# Native helpers are optional but must be strict if present.
_FALLBACK_ON_NATIVE_FAILURE = False

try:
    import cognis.dsp.cognis_native as native
    NATIVE_AVAILABLE = True
except ImportError:
    NATIVE_AVAILABLE = False


class Limiter:
    def __init__(self, sr: int):
        self.sr = sr
        self.last_execution_info = {}
        
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
        
        hold_ms = 1.5
        hold_samples = max(1, int((hold_ms / 1000.0) * fs))
        release_ms = 10.0
        sigma_samples = (release_ms / 1000.0) * fs

        self.last_execution_info = {
            "used_native": False,
            "fallback_triggered": False
        }

        smooth_gain = None

        if NATIVE_AVAILABLE:
            try:
                # 2 & 3. Fused hold and smoothing natively
                smooth_gain = native.compute_native_limiter_gain_fused(
                    np.ascontiguousarray(raw_gain),
                    int(hold_samples),
                    float(sigma_samples)
                )
                self.last_execution_info["used_native"] = True
            except Exception as e:
                self.last_execution_info["fallback_triggered"] = True
                if not _FALLBACK_ON_NATIVE_FAILURE:
                    raise RuntimeError(f"Native limiter execution failed: {e}") from e
                logger.warning(f"Native limiter execution failed, falling back to Python. Error: {e}")

        if smooth_gain is None:
            # 2. Hold the gain reduction (sustain)
            # Widens the dips so smoothing doesn't under-reduce at the exact peak
            held_gain = minimum_filter1d(raw_gain, size=hold_samples)

            # 3. Smooth the gain reduction envelope (attack / release)
            # Gaussian filter provides a smooth, symmetric transition (quasi-lookahead)
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
