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


def run_benchmark_for_signal(label: str, audio: np.ndarray, splitter, is_short: bool = False):
    print(f"\n--- {label} (shape: {audio.shape}) ---")

    reference_low = _benchmark(
        "reference per-channel fftconvolve(low band)",
        lambda: np.vstack([fftconvolve(channel, splitter.low_taps, mode="same") for channel in audio]),
        iterations=25,
    )
    from cognis.dsp.fir_executor import _NATIVE_FIR_AVAILABLE, get_fir_execution_info

    def _apply_fir_and_report_auto():
        apply_fir(audio, splitter.low_taps, backend=FirBackend.AUTO)

    optimized_low = _benchmark("apply_fir(low band, backend=AUTO)", _apply_fir_and_report_auto, iterations=25)
    auto_info = get_fir_execution_info()
    print(f"  -> AUTO decided on method: '{auto_info['selected_method']}'")
    print(f"  -> AUTO execution path:    {'NATIVE' if auto_info['used_native'] else 'PYTHON (Fallback)' if auto_info['fallback_triggered'] else 'PYTHON'}")


    repeated_split_auto = _benchmark("splitter.split(audio, backend=AUTO)", lambda: splitter.split(audio, backend=FirBackend.AUTO), iterations=25)

    def _run_fft_and_check():
        splitter.split(audio, backend=FirBackend.FFT)
        info = get_fir_execution_info()
        return info

    repeated_split_fft = _benchmark("splitter.split(audio, backend=FFT)", _run_fft_and_check, iterations=25)
    fft_info = _run_fft_and_check()
    if fft_info['used_native']:
        print(f"  -> Proof: FFT executed natively.")
    elif fft_info['fallback_triggered']:
        print(f"  -> Proof: FFT execution triggered fallback (Python used).")
    else:
        print(f"  -> Proof: FFT executed in Python (Native unavailable or not used).")

    def _run_partitioned_and_check():
        splitter.split(audio, backend=FirBackend.PARTITIONED)
        return get_fir_execution_info()

    repeated_split_partitioned = _benchmark("splitter.split(audio, backend=PARTITIONED)", _run_partitioned_and_check, iterations=25)
    part_info = _run_partitioned_and_check()
    if part_info['used_native']:
        print(f"  -> Proof: PARTITIONED executed natively.")
    elif part_info['fallback_triggered']:
        print(f"  -> Proof: PARTITIONED execution triggered fallback (Python used).")
    else:
        print(f"  -> Proof: PARTITIONED executed in Python (Native unavailable or not used).")

    # Direct is extremely slow for long signals and long taps. Only run if it's short, or a very small number of iterations.
    if is_short:
        repeated_split_direct = _benchmark("splitter.split(audio, backend=DIRECT)", lambda: splitter.split(audio, backend=FirBackend.DIRECT), iterations=25)
    else:
        repeated_split_direct = _benchmark("splitter.split(audio, backend=DIRECT)", lambda: splitter.split(audio, backend=FirBackend.DIRECT), iterations=2)
        # Normalize to 25 iterations for ratio comparison
        repeated_split_direct = (repeated_split_direct / 2) * 25


    print(f"apply_fir vs reference fft ratio: {optimized_low / reference_low:.4f}")
    print(f"PARTITIONED split vs FFT split ratio: {repeated_split_partitioned / repeated_split_fft:.4f}")
    print(f"FFT split vs DIRECT split ratio: {repeated_split_fft / repeated_split_direct:.4f}")
    print(f"AUTO split vs reference best possible ratio: {repeated_split_auto / min(repeated_split_partitioned, repeated_split_fft):.4f}")


def profile_render_loop(audio: np.ndarray, splitter):
    """
    Simulates a repeated render loop to represent the optimizer's workload.
    """
    print("\n--- Profiling Repeated Render Loop (Simulating Optimizer) ---")
    import cProfile
    import pstats

    def render_loop():
        for _ in range(15):
            # simulate 15 iterations of optimizer tweaks resulting in FIR splits
            splitter.split(audio, backend=FirBackend.AUTO)

    profiler = cProfile.Profile()
    profiler.enable()
    render_loop()
    profiler.disable()

    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(15)

def main() -> None:
    sr = 48000
    rng = np.random.default_rng(29)
    audio_long = rng.standard_normal((2, sr * 5)) * 0.1  # 5 seconds
    audio_short = rng.standard_normal((2, 512)) * 0.1    # Short block

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

    run_benchmark_for_signal("Short Signal Scenario", audio_short, splitter, is_short=True)
    run_benchmark_for_signal("Long Signal Scenario (5s)", audio_long, splitter, is_short=False)

    cache_info = get_fir_design_cache_info()
    print(f"\ncache info: {cache_info}")

    # 5s signal loop profile
    profile_render_loop(audio_long, splitter)


if __name__ == "__main__":
    main()
