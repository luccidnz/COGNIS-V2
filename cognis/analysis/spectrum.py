import numpy as np
from scipy.signal import welch

def compute_spectrum_features(audio: np.ndarray, sr: int):
    """Compute spectral tilt and band balances."""
    # Mix to mono for spectral analysis
    mono = np.mean(audio, axis=0) if audio.shape[0] > 1 else audio[0]
    
    if len(mono) < 256:
        return 0.0, 0.0, 0.0
        
    f, pxx = welch(mono, sr, nperseg=min(4096, len(mono)))
    
    # Avoid log of zero
    pxx = np.maximum(pxx, 1e-10)
    
    # Define bands
    low_mask = (f >= 20) & (f < 250)
    mid_mask = (f >= 250) & (f < 4000)
    high_mask = (f >= 4000) & (f < 20000)
    
    low_energy = np.sum(pxx[low_mask])
    mid_energy = np.sum(pxx[mid_mask])
    high_energy = np.sum(pxx[high_mask])
    
    total_energy = low_energy + mid_energy + high_energy + 1e-10
    
    low_mid_balance = 10 * np.log10((low_energy + 1e-10) / (mid_energy + 1e-10))
    high_mid_balance = 10 * np.log10((high_energy + 1e-10) / (mid_energy + 1e-10))
    
    # Spectral tilt (simple linear regression on log-log scale)
    valid = (f > 50) & (f < 16000)
    if np.sum(valid) > 2:
        log_f = np.log10(f[valid])
        log_p = 10 * np.log10(pxx[valid])
        tilt, _ = np.polyfit(log_f, log_p, 1)
    else:
        tilt = 0.0
        
    return tilt, low_mid_balance, high_mid_balance
