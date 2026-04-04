import numpy as np
from scipy.signal import butter, lfilter

def compute_stereo_features(audio: np.ndarray, sr: int):
    """Compute stereo width per band and overall phase correlation."""
    if audio.shape[0] == 1:
        return 0.0, 0.0, 0.0, 1.0
        
    L = audio[0]
    R = audio[1]
    
    # Phase correlation
    num = np.sum(L * R)
    den = np.sqrt(np.sum(L**2) * np.sum(R**2)) + 1e-10
    phase_correlation = num / den
    
    # Band widths
    def get_width(l_band, r_band):
        mid = (l_band + r_band) / 2
        side = (l_band - r_band) / 2
        mid_energy = np.sum(mid**2)
        side_energy = np.sum(side**2)
        if mid_energy + side_energy < 1e-10:
            return 0.0
        return side_energy / (mid_energy + side_energy)
        
    # Simple crossovers
    b_low, a_low = butter(2, 250 / (sr / 2), btype='lowpass')
    b_mid, a_mid = butter(2, [250 / (sr / 2), 4000 / (sr / 2)], btype='bandpass')
    b_high, a_high = butter(2, 4000 / (sr / 2), btype='highpass')
    
    L_low = lfilter(b_low, a_low, L)
    R_low = lfilter(b_low, a_low, R)
    
    L_mid = lfilter(b_mid, a_mid, L)
    R_mid = lfilter(b_mid, a_mid, R)
    
    L_high = lfilter(b_high, a_high, L)
    R_high = lfilter(b_high, a_high, R)
    
    low_width = get_width(L_low, R_low)
    mid_width = get_width(L_mid, R_mid)
    high_width = get_width(L_high, R_high)
    
    return low_width, mid_width, high_width, phase_correlation
