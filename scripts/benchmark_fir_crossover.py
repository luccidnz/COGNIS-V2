import time

import numpy as np

from cognis.dsp.filters import (
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
    repeated_split = _benchmark("splitter.split(audio)", lambda: splitter.split(audio), iterations=25)
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
    print(f"cached lookup vs direct split ratio: {warm_lookup / repeated_split:.4f}")
    print(f"wrapper split vs direct split ratio: {wrapper_split / repeated_split:.4f}")


if __name__ == "__main__":
    main()
