import numpy as np
from scipy.signal import lfilter, butter, resample_poly

def k_weighting_filter(audio: np.ndarray, sr: int) -> np.ndarray:
    """Approximate K-weighting filter for BS.1770."""
    # Pre-filter 1: High shelf
    # Pre-filter 2: High pass
    # For MVP, we use a simple high-pass and high-shelf approximation
    # Real BS.1770 uses specific biquad coefficients for 48kHz
    
    # Simple approximation for 48kHz
    if sr == 48000:
        b1 = [1.53512485958697, -2.69169618940638, 1.19839281085285]
        a1 = [1.0, -1.69065929318241, 0.73248077421585]
        b2 = [1.0, -2.0, 1.0]
        a2 = [1.0, -1.99004745483398, 0.99007225036621]
        
        filtered = lfilter(b1, a1, audio, axis=-1)
        filtered = lfilter(b2, a2, filtered, axis=-1)
        return filtered
    else:
        # Fallback generic HPF for other sample rates
        b, a = butter(2, 100 / (sr / 2), btype='highpass')
        return lfilter(b, a, audio, axis=-1)

def compute_loudness(audio: np.ndarray, sr: int):
    """Compute integrated, short-term, momentary loudness, and peaks."""
    # K-weighting
    weighted = k_weighting_filter(audio, sr)
    
    # Mean square energy per channel
    # 400ms blocks
    block_size = int(0.4 * sr)
    overlap = int(0.3 * sr)
    step = block_size - overlap
    
    if audio.shape[1] < block_size:
        # Too short, just return simple RMS
        rms = np.sqrt(np.mean(weighted**2))
        lufs = -70.0 if rms < 1e-10 else 20 * np.log10(rms)
        return lufs, lufs, lufs, np.max(np.abs(audio)), np.max(np.abs(audio))
        
    num_blocks = (weighted.shape[1] - block_size) // step + 1
    energies = np.zeros((weighted.shape[0], num_blocks))
    
    for i in range(num_blocks):
        start = i * step
        end = start + block_size
        energies[:, i] = np.mean(weighted[:, start:end]**2, axis=1)
        
    # Sum channels (with 1.5dB boost for surround channels, but we assume stereo)
    block_energies = np.sum(energies, axis=0)
    
    # Absolute gate at -70 LUFS
    abs_gate_energy = 10**(-70.0 / 10.0)
    gated_blocks = block_energies[block_energies > abs_gate_energy]
    
    if len(gated_blocks) == 0:
        integrated = -70.0
    else:
        # Relative gate at -10 LU from absolute gated loudness
        rel_threshold = np.mean(gated_blocks) * (10**(-10.0 / 10.0))
        rel_gated_blocks = gated_blocks[gated_blocks > rel_threshold]
        
        if len(rel_gated_blocks) == 0:
            integrated = -70.0
        else:
            integrated = -0.691 + 10 * np.log10(np.mean(rel_gated_blocks))
            
    # Short-term (3s blocks)
    st_block_size = int(3.0 * sr)
    if audio.shape[1] >= st_block_size:
        st_step = int(1.0 * sr)
        st_num_blocks = (weighted.shape[1] - st_block_size) // st_step + 1
        st_energies = np.zeros(st_num_blocks)
        for i in range(st_num_blocks):
            start = i * st_step
            end = start + st_block_size
            st_energies[i] = np.sum(np.mean(weighted[:, start:end]**2, axis=1))
        short_term = -0.691 + 10 * np.log10(np.max(st_energies) + 1e-10)
    else:
        short_term = integrated
        
    momentary = -0.691 + 10 * np.log10(np.max(block_energies) + 1e-10)
    
    sample_peak = np.max(np.abs(audio))
    
    # True peak (4x oversampling approximation)
    if audio.shape[1] > 100:
        # Just oversample a small chunk around the peak for speed in MVP
        peak_idx = np.argmax(np.abs(audio[0]))
        start = max(0, peak_idx - 50)
        end = min(audio.shape[1], peak_idx + 50)
        chunk = audio[:, start:end]
        os_chunk = resample_poly(chunk, 4, 1, axis=-1)
        true_peak = np.max(np.abs(os_chunk))
    else:
        true_peak = sample_peak
        
    # Ensure true peak is at least sample peak
    true_peak = max(true_peak, sample_peak)
    
    return integrated, short_term, momentary, true_peak, sample_peak
