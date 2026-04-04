import numpy as np
import soundfile as sf
from typing import Tuple

def load_audio(path: str) -> Tuple[np.ndarray, int]:
    """Load audio file. Returns shape (channels, samples) and sample rate."""
    data, sr = sf.read(path, always_2d=True)
    return data.T, sr

def save_audio(path: str, data: np.ndarray, sr: int):
    """Save audio file. Expects shape (channels, samples)."""
    sf.write(path, data.T, sr)
