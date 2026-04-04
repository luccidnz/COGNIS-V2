from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, fftconvolve, firwin, sosfilt


@dataclass(frozen=True)
class ThreeBandSplit:
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray


def _as_audio_2d(audio: np.ndarray) -> tuple[np.ndarray, bool]:
    arr = np.asarray(audio, dtype=np.float64)
    if arr.ndim == 1:
        return arr[np.newaxis, :], True
    if arr.ndim == 2:
        return arr, False
    raise ValueError("audio must be a 1D mono or 2D channel-first array")


def _restore_audio_shape(audio: np.ndarray, squeeze: bool) -> np.ndarray:
    return audio[0] if squeeze else audio


def _validate_cutoff(cutoff: float, sr: int) -> None:
    if cutoff <= 0.0:
        raise ValueError("cutoff must be positive")
    nyquist = 0.5 * sr
    if cutoff >= nyquist:
        raise ValueError("cutoff must be below Nyquist")


def _design_iir_sos(cutoff: float, sr: int, order: int, btype: str) -> np.ndarray:
    _validate_cutoff(cutoff, sr)
    return butter(order, cutoff, btype=btype, fs=sr, output="sos")


def apply_lowpass(audio: np.ndarray, cutoff: float, sr: int, order: int = 1) -> np.ndarray:
    audio_2d, squeeze = _as_audio_2d(audio)
    filtered = sosfilt(_design_iir_sos(cutoff, sr, order, "lowpass"), audio_2d, axis=-1)
    return _restore_audio_shape(filtered, squeeze)


def apply_highpass(audio: np.ndarray, cutoff: float, sr: int, order: int = 1) -> np.ndarray:
    audio_2d, squeeze = _as_audio_2d(audio)
    filtered = sosfilt(_design_iir_sos(cutoff, sr, order, "highpass"), audio_2d, axis=-1)
    return _restore_audio_shape(filtered, squeeze)


def apply_bandpass(audio: np.ndarray, lowcut: float, highcut: float, sr: int, order: int = 2) -> np.ndarray:
    if lowcut >= highcut:
        raise ValueError("lowcut must be below highcut")
    _validate_cutoff(lowcut, sr)
    _validate_cutoff(highcut, sr)

    audio_2d, squeeze = _as_audio_2d(audio)
    sos = butter(order, [lowcut, highcut], btype="bandpass", fs=sr, output="sos")
    filtered = sosfilt(sos, audio_2d, axis=-1)
    return _restore_audio_shape(filtered, squeeze)


def estimate_fir_taps(
    sr: int,
    cutoff: float,
    *,
    transition_hz: float | None = None,
    attenuation_db: float = 80.0,
    minimum: int = 257,
    maximum: int = 2049,
) -> int:
    """
    Estimate an odd FIR length for an offline mastering crossover.

    The default heuristic targets a narrow-enough transition band to keep a
    low crossover such as 250 Hz credible, while still capping kernel sizes so
    the Python MVP stays practical during optimization passes.
    """
    _validate_cutoff(cutoff, sr)
    if transition_hz is None:
        transition_hz = max(100.0, cutoff * 0.5)

    transition_hz = float(np.clip(transition_hz, 1.0, 0.5 * sr - cutoff))
    taps = int(np.ceil(5.0 * sr * max(attenuation_db - 8.0, 1.0) / (72.0 * transition_hz)))
    taps = int(np.clip(taps, minimum, maximum))
    if taps % 2 == 0:
        taps += 1
    return taps


def design_linear_phase_lowpass(
    cutoff: float,
    sr: int,
    *,
    numtaps: int | None = None,
    window: tuple[str, float] = ("kaiser", 8.6),
) -> np.ndarray:
    if numtaps is None:
        numtaps = estimate_fir_taps(sr, cutoff)
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for a symmetric linear-phase FIR")

    _validate_cutoff(cutoff, sr)
    return firwin(numtaps, cutoff, fs=sr, window=window, pass_zero="lowpass")


def design_linear_phase_highpass(
    cutoff: float,
    sr: int,
    *,
    numtaps: int | None = None,
    window: tuple[str, float] = ("kaiser", 8.6),
) -> np.ndarray:
    if numtaps is None:
        numtaps = estimate_fir_taps(sr, cutoff, maximum=1025)
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for a symmetric linear-phase FIR")

    _validate_cutoff(cutoff, sr)
    return firwin(numtaps, cutoff, fs=sr, window=window, pass_zero="highpass")


def apply_fir(audio: np.ndarray, taps: np.ndarray) -> np.ndarray:
    audio_2d, squeeze = _as_audio_2d(audio)
    taps = np.asarray(taps, dtype=np.float64)
    filtered = np.vstack([fftconvolve(channel, taps, mode="same") for channel in audio_2d])
    return _restore_audio_shape(filtered, squeeze)


def split_linear_phase_three_band(
    audio: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sr: int,
    *,
    low_taps: int | None = None,
    high_taps: int | None = None,
    window: tuple[str, float] = ("kaiser", 8.6),
) -> ThreeBandSplit:
    """
    Split audio into low, mid, and high bands for offline mastering work.

    This is intentionally an offline FIR splitter rather than a real-time
    crossover. It trades latency for better transparency and exact residual
    reconstruction inside this Python MVP.
    """
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be below high_cutoff")

    low = apply_fir(audio, design_linear_phase_lowpass(low_cutoff, sr, numtaps=low_taps, window=window))
    high = apply_fir(audio, design_linear_phase_highpass(high_cutoff, sr, numtaps=high_taps, window=window))
    mid = np.asarray(audio, dtype=np.float64) - low - high
    return ThreeBandSplit(low=low, mid=mid, high=high)
