from enum import Enum
from functools import lru_cache

import numpy as np
from scipy.signal import convolve
from scipy.fft import next_fast_len, rfft, irfft


class FirBackend(Enum):
    """
    Execution backend for FIR crossover application.

    AUTO: Uses heuristics to choose fastest path. Dispatches to native if available and optimal.
    DIRECT: Forces direct convolution (method="direct").
    FFT: Forces FFT-based convolution (method="fft").
    PARTITIONED: Overlap-save partitioned convolution.
    """
    AUTO = "auto"
    DIRECT = "direct"
    FFT = "fft"
    PARTITIONED = "partitioned"


# --------------------------------------------------------------------------------
# PYTHON REFERENCE FIR IMPLEMENTATION
# --------------------------------------------------------------------------------

@lru_cache(maxsize=128)
def _get_rfft_kernel_cached(taps_bytes: bytes, N: int, dtype: np.dtype) -> np.ndarray:
    taps = np.frombuffer(taps_bytes, dtype=dtype)
    return rfft(taps, n=N)

def _get_rfft_kernel(taps: np.ndarray, N: int) -> np.ndarray:
    return _get_rfft_kernel_cached(taps.tobytes(), N, taps.dtype)

def _apply_python_partitioned_fir_2d(audio_2d: np.ndarray, taps: np.ndarray) -> np.ndarray:
    """
    Uniform overlap-save partitioned convolution.
    Serves as the explicit Python reference for future compiled C++ implementations.
    """
    channels, signal_len = audio_2d.shape
    kernel_len = taps.shape[0]

    # 1. Choose sensible fixed partition size
    partition_size = 4096

    # 2. Determine optimal FFT size (N >= partition_size + kernel_len - 1)
    N = next_fast_len(partition_size + kernel_len - 1)

    # Block step size
    step = N - kernel_len + 1

    # 3. Calculate padding to preserve exact mode="same" semantics
    shift = (kernel_len - 1) // 2
    pad_start = shift

    num_blocks = (signal_len + step - 1) // step
    pad_end = num_blocks * step + kernel_len - 1 - signal_len - pad_start

    # 4. Pad input signal
    padded_audio = np.pad(audio_2d, ((0, 0), (pad_start, pad_end)), mode="constant")

    # 5. Output buffer
    out = np.zeros((channels, num_blocks * step), dtype=np.float64)

    # 6. Retrieve/Compute cached kernel FFT
    H = _get_rfft_kernel(taps, N)

    # 7. Overlap-save block processing loop
    for b in range(num_blocks):
        start_idx = b * step
        end_idx = start_idx + N

        block = padded_audio[:, start_idx:end_idx]
        block_fft = rfft(block, n=N, axis=-1)

        out_fft = block_fft * H
        out_block = irfft(out_fft, n=N, axis=-1)

        # Discard the first (kernel_len - 1) circular convolution wrap-around samples
        valid_part = out_block[:, kernel_len - 1:]
        out[:, b * step : (b + 1) * step] = valid_part

    # 8. Truncate exactly to original signal length
    return out[:, :signal_len]

def _choose_backend_method(signal_len: int, kernel_len: int, channels: int, has_nan_inf: bool = False) -> str:
    """
    Explicit, benchmark-backed heuristic for choosing convolution method if backend is AUTO.
    """
    if has_nan_inf:
        # FFT paths spread NaNs/Infs across the entire block. Direct convolution localizes them.
        return "direct"

    # Direct convolution is only faster for extremely short signals and short kernels.
    if signal_len < 1024 and kernel_len < 128:
        return "direct"

    # For long signals and substantial kernels, PARTITIONED is memory-efficient,
    # benefits from kernel FFT caching, and serves as our offline reference.
    if signal_len > 16384 and kernel_len >= 256:
        return "partitioned"

    # For intermediate lengths, a single monolithic FFT is generally fastest.
    return "fft"

def execute_python_fir_2d(audio_2d: np.ndarray, taps: np.ndarray, backend: FirBackend) -> np.ndarray:
    """
    Reference Python FIR execution path.
    """
    kernel = np.asarray(taps, dtype=np.float64)

    if backend == FirBackend.AUTO:
        has_nan_inf = not np.isfinite(audio_2d).all()
        method = _choose_backend_method(audio_2d.shape[-1], kernel.shape[-1], audio_2d.shape[0], has_nan_inf)
    else:
        method = backend.value

    if method == "partitioned":
        return _apply_python_partitioned_fir_2d(audio_2d, kernel)

    return convolve(audio_2d, kernel[np.newaxis, :], mode="same", method=method)


# --------------------------------------------------------------------------------
# NATIVE BACKEND DISPATCH BOUNDARY
# --------------------------------------------------------------------------------

_NATIVE_FIR_AVAILABLE = False

try:
    # Attempt to import the optional compiled C++ backend.
    # Currently expected to be missing as this is the preparation pass.
    import cognis_native
    if hasattr(cognis_native, "execute_native_fir_2d"):
        _NATIVE_FIR_AVAILABLE = True
except ImportError:
    pass


def execute_fir_2d(audio_2d: np.ndarray, taps: np.ndarray, backend: FirBackend) -> np.ndarray:
    """
    Apply one FIR kernel across channel-first audio using the selected backend.

    This acts as the explicit memory and dispatch boundary layer:
    - Normalizes array shapes and memory layouts for strict C++ safety.
    - Dispatches to the optional native backend if available.
    - Gracefully falls back to the Python reference path otherwise.

    Data Contract Expectations:
    - `audio_2d`: Must be 2-dimensional `[channels, samples]`.
    - `taps`: Must be 1-dimensional `[samples]`.
    - Both must be `float64` (`np.float64`).
    - Both will be forced to C-contiguous memory layout before dispatch to ensure
      deterministic, crash-free zero-copy behavior in native extensions.
    - `mode="same"` strict behavior is guaranteed: the output array length must
      exactly match the input `samples` length.
    """
    # 1. Enforce memory layout and type contract for native safety
    audio_2d = np.ascontiguousarray(audio_2d, dtype=np.float64)
    taps = np.ascontiguousarray(taps, dtype=np.float64)

    if audio_2d.ndim != 2:
        raise ValueError(f"audio_2d must be 2-dimensional [channels, samples], got shape {audio_2d.shape}")
    if taps.ndim != 1:
        raise ValueError(f"taps must be 1-dimensional, got shape {taps.shape}")

    # 2. Dispatch to Native if available and explicitly requested (or AUTO decides to use it)
    if _NATIVE_FIR_AVAILABLE:
        # Native backend integration goes here in the future
        # e.g., return cognis_native.execute_native_fir_2d(audio_2d, taps, backend.name)
        pass

    # 3. Fallback to Python reference behavior
    return execute_python_fir_2d(audio_2d, taps, backend)

def get_fir_executor_cache_info() -> dict[str, int]:
    cache_info = _get_rfft_kernel_cached.cache_info()
    return {
        "hits": cache_info.hits,
        "misses": cache_info.misses,
        "maxsize": cache_info.maxsize,
        "currsize": cache_info.currsize,
    }

def clear_fir_executor_cache() -> None:
    _get_rfft_kernel_cached.cache_clear()
