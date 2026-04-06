import time

import numpy as np
from scipy.signal import fftconvolve

from cognis.dsp.filters import (
    FirBackend,
    apply_fir,
    clear_fir_design_cache,
    get_fir_design_cache_info,
    get_linear_phase_three_band_splitter,
    split_linear_phase_three_band,
)


def _benchmark(label: str, fn, iterations: int) -> float:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    avg_ms = (elapsed / iterations) * 1000.0
    print(f"{label}: total={elapsed:.4f}s avg={avg_ms:.3f}ms over {iterations} iterations")
    return elapsed


def main() -> None:
    sr = 48000
    rng = np.random.default_rng(29)
    audio = rng.standard_normal((2, sr)) * 0.1
    low_cutoff = 250.0
    high_cutoff = 4000.0
    low_taps = 1537
    high_taps = 513

    clear_fir_design_cache()
    cold_start = time.perf_counter()
    splitter = get_linear_phase_three_band_splitter(
        sr,
        low_cutoff,
        high_cutoff,
        low_taps=low_taps,
        high_taps=high_taps,
    )
    cold_elapsed = time.perf_counter() - cold_start
    print(f"cold splitter build: {cold_elapsed * 1000.0:.3f}ms")

    warm_lookup = _benchmark(
        "cached splitter lookup",
        lambda: get_linear_phase_three_band_splitter(
            sr,
            low_cutoff,
            high_cutoff,
            low_taps=low_taps,
            high_taps=high_taps,
        ),
        iterations=200,
    )
    reference_low = _benchmark(
        "reference per-channel fftconvolve(low band)",
        lambda: np.vstack([fftconvolve(channel, splitter.low_taps, mode="same") for channel in audio]),
        iterations=25,
    )
    optimized_low = _benchmark("apply_fir(low band)", lambda: apply_fir(audio, splitter.low_taps), iterations=25)
    repeated_split_auto = _benchmark("splitter.split(audio, backend=AUTO)", lambda: splitter.split(audio, backend=FirBackend.AUTO), iterations=25)
    repeated_split_direct = _benchmark("splitter.split(audio, backend=DIRECT)", lambda: splitter.split(audio, backend=FirBackend.DIRECT), iterations=25)
    repeated_split_fft = _benchmark("splitter.split(audio, backend=FFT)", lambda: splitter.split(audio, backend=FirBackend.FFT), iterations=25)

    wrapper_split = _benchmark(
        "split_linear_phase_three_band(audio, ...)",
        lambda: split_linear_phase_three_band(
            audio,
            low_cutoff,
            high_cutoff,
            sr,
            low_taps=low_taps,
            high_taps=high_taps,
        ),
        iterations=25,
    )

    cache_info = get_fir_design_cache_info()
    print(f"cache info: {cache_info}")
    print(f"apply_fir vs reference fft ratio: {optimized_low / reference_low:.4f}")
    print(f"cached lookup vs AUTO split ratio: {warm_lookup / repeated_split_auto:.4f}")
    print(f"wrapper split vs AUTO split ratio: {wrapper_split / repeated_split_auto:.4f}")
    print(f"FFT split vs DIRECT split ratio: {repeated_split_fft / repeated_split_direct:.4f}")
    print(f"AUTO split vs FFT split ratio: {repeated_split_auto / repeated_split_fft:.4f}")


if __name__ == "__main__":
    main()
