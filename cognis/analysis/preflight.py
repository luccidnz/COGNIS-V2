import numpy as np

def validate_audio(audio: np.ndarray) -> None:
    if not isinstance(audio, np.ndarray):
        raise TypeError("Audio must be a numpy array")
    if audio.ndim != 2:
        raise ValueError(f"Audio must be 2D (channels, samples), got shape {audio.shape}")
    if audio.shape[0] not in (1, 2):
        raise ValueError(f"Audio must be mono or stereo, got {audio.shape[0]} channels")
    if audio.shape[1] == 0:
        raise ValueError("Audio is empty")
    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains NaN or Inf")
