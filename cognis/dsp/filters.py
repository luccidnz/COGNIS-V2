from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.signal import butter, convolve, firwin, sosfilt


class FirBackend(Enum):
    """
    Execution backend for FIR crossover application.

    AUTO: Uses scipy.signal.convolve(method="auto") for fastest general path.
    DIRECT: Forces direct convolution (method="direct").
    FFT: Forces FFT-based convolution (method="fft").
    PARTITIONED: Placeholder for future real-time/compiled partitioned convolution.
    """
    AUTO = "auto"
    DIRECT = "direct"
    FFT = "fft"
    PARTITIONED = "partitioned"


WindowSpec = str | tuple[str, float]


@dataclass(frozen=True)
class ThreeBandSplit:
    low: np.ndarray
    mid: np.ndarray
    high: np.ndarray


@dataclass(frozen=True)
class LinearPhaseThreeBandSplitter:
    """
    Reusable offline FIR splitter for repeated mastering renders.

    The splitter itself is immutable and safe to cache. The tap arrays are
    marked read-only so cached instances cannot be mutated accidentally.
    """

    sr: int
    low_cutoff: float
    high_cutoff: float
    low_taps: np.ndarray = field(repr=False)
    high_taps: np.ndarray = field(repr=False)

    def split(self, audio: np.ndarray, backend: FirBackend = FirBackend.AUTO) -> ThreeBandSplit:
        audio_2d, squeeze = _as_audio_2d(audio)
        low = _apply_fir_2d(audio_2d, self.low_taps, backend)
        high = _apply_fir_2d(audio_2d, self.high_taps, backend)
        mid = audio_2d - low - high
        return ThreeBandSplit(
            low=_restore_audio_shape(low, squeeze),
            mid=_restore_audio_shape(mid, squeeze),
            high=_restore_audio_shape(high, squeeze),
        )


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


def _normalize_window(window: WindowSpec) -> tuple[object, ...]:
    if isinstance(window, str):
        return (window,)
    return tuple(window)


def _restore_window(window_key: tuple[object, ...]) -> WindowSpec:
    if len(window_key) == 1 and isinstance(window_key[0], str):
        return window_key[0]
    return window_key  # type: ignore[return-value]


def _cache_info_dict(cache_info) -> dict[str, int]:
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
    }


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

    The default heuristic keeps low crossover points such as 250 Hz credible
    without letting the Python MVP spend unlimited time building huge kernels.
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


@lru_cache(maxsize=64)
def _design_linear_phase_fir_cached(
    filter_kind: str,
    cutoff: float,
    sr: int,
    numtaps: int,
    window_key: tuple[object, ...],
) -> np.ndarray:
    pass_zero = "lowpass" if filter_kind == "lowpass" else "highpass"
    taps = firwin(numtaps, cutoff, fs=sr, window=_restore_window(window_key), pass_zero=pass_zero)
    taps.setflags(write=False)
    return taps


def design_linear_phase_lowpass(
    cutoff: float,
    sr: int,
    *,
    numtaps: int | None = None,
    window: WindowSpec = ("kaiser", 8.6),
) -> np.ndarray:
    if numtaps is None:
        numtaps = estimate_fir_taps(sr, cutoff)
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for a symmetric linear-phase FIR")

    _validate_cutoff(cutoff, sr)
    return _design_linear_phase_fir_cached("lowpass", cutoff, sr, numtaps, _normalize_window(window))


def design_linear_phase_highpass(
    cutoff: float,
    sr: int,
    *,
    numtaps: int | None = None,
    window: WindowSpec = ("kaiser", 8.6),
) -> np.ndarray:
    if numtaps is None:
        numtaps = estimate_fir_taps(sr, cutoff, maximum=1025)
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for a symmetric linear-phase FIR")

    _validate_cutoff(cutoff, sr)
    return _design_linear_phase_fir_cached("highpass", cutoff, sr, numtaps, _normalize_window(window))


def apply_fir(audio: np.ndarray, taps: np.ndarray, backend: FirBackend = FirBackend.AUTO) -> np.ndarray:
    audio_2d, squeeze = _as_audio_2d(audio)
    filtered = _apply_fir_2d(audio_2d, taps, backend)
    return _restore_audio_shape(filtered, squeeze)


def _apply_fir_2d(audio_2d: np.ndarray, taps: np.ndarray, backend: FirBackend) -> np.ndarray:
    """
    Apply one FIR kernel across channel-first audio using the selected backend.

    `scipy.signal.convolve(..., method="auto")` lets SciPy choose direct or
    FFT convolution per shape, which is faster here than our previous explicit
    per-channel `fftconvolve` loop while preserving deterministic output.
    """
    if backend == FirBackend.PARTITIONED:
        # TODO: Implement partitioned convolution backend.
        # This is a hook point for future compiled/real-time block processing
        # where we might want zero latency or strictly bounded block sizes.
        raise NotImplementedError("Partitioned convolution backend not yet implemented.")

    kernel = np.asarray(taps, dtype=np.float64)
    return convolve(audio_2d, kernel[np.newaxis, :], mode="same", method=backend.value)


@lru_cache(maxsize=32)
def _get_linear_phase_three_band_splitter_cached(
    sr: int,
    low_cutoff: float,
    high_cutoff: float,
    low_taps: int,
    high_taps: int,
    window_key: tuple[object, ...],
) -> LinearPhaseThreeBandSplitter:
    return LinearPhaseThreeBandSplitter(
        sr=sr,
        low_cutoff=low_cutoff,
        high_cutoff=high_cutoff,
        low_taps=design_linear_phase_lowpass(low_cutoff, sr, numtaps=low_taps, window=_restore_window(window_key)),
        high_taps=design_linear_phase_highpass(high_cutoff, sr, numtaps=high_taps, window=_restore_window(window_key)),
    )


def get_linear_phase_three_band_splitter(
    sr: int,
    low_cutoff: float,
    high_cutoff: float,
    *,
    low_taps: int | None = None,
    high_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.6),
) -> LinearPhaseThreeBandSplitter:
    """
    Return a cached immutable splitter for repeated candidate renders.

    The cache is keyed by the design parameters that materially affect the FIR
    kernels, so identical optimizer renders can reuse setup deterministically.
    """
    if low_cutoff >= high_cutoff:
        raise ValueError("low_cutoff must be below high_cutoff")

    if low_taps is None:
        low_taps = estimate_fir_taps(sr, low_cutoff)
    if high_taps is None:
        high_taps = estimate_fir_taps(sr, high_cutoff, maximum=1025)

    return _get_linear_phase_three_band_splitter_cached(
        sr,
        low_cutoff,
        high_cutoff,
        low_taps,
        high_taps,
        _normalize_window(window),
    )


def split_linear_phase_three_band(
    audio: np.ndarray,
    low_cutoff: float,
    high_cutoff: float,
    sr: int,
    *,
    low_taps: int | None = None,
    high_taps: int | None = None,
    window: WindowSpec = ("kaiser", 8.6),
    backend: FirBackend = FirBackend.AUTO,
) -> ThreeBandSplit:
    """
    Split audio into low, mid, and high bands for offline mastering work.

    This remains an offline FIR path rather than a real-time crossover. The
    main hardening here is reuse: taps and splitter setup are cached so
    repeated optimizer renders avoid redesigning identical filters.
    """
    splitter = get_linear_phase_three_band_splitter(
        sr,
        low_cutoff,
        high_cutoff,
        low_taps=low_taps,
        high_taps=high_taps,
        window=window,
    )
    return splitter.split(audio, backend)


def clear_fir_design_cache() -> None:
    _get_linear_phase_three_band_splitter_cached.cache_clear()
    _design_linear_phase_fir_cached.cache_clear()


def get_fir_design_cache_info() -> dict[str, dict[str, int]]:
    return {
        "splitter": _cache_info_dict(_get_linear_phase_three_band_splitter_cached.cache_info()),
        "fir": _cache_info_dict(_design_linear_phase_fir_cached.cache_info()),
    }
