import numpy as np
from scipy.signal import butter, lfilter

def apply_lowpass(audio: np.ndarray, cutoff: float, sr: int, order: int = 1) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, audio, axis=-1)

def apply_highpass(audio: np.ndarray, cutoff: float, sr: int, order: int = 1) -> np.ndarray:
    nyq = 0.5 * sr
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='high', analog=False)
    return lfilter(b, a, audio, axis=-1)

def apply_bandpass(audio: np.ndarray, lowcut: float, highcut: float, sr: int, order: int = 2) -> np.ndarray:
    nyq = 0.5 * sr
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return lfilter(b, a, audio, axis=-1)
