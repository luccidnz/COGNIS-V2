from __future__ import annotations

import argparse
import platform
import time
from typing import Any

import numpy as np
from scipy.signal import fftconvolve

from cognis.dsp.filters import (
    FirBackend,
    apply_fir,
    clear_fir_design_cache,
    get_fir_design_cache_info,
    get_linear_phase_three_band_splitter,
)
from cognis.dsp.fir_executor import _NATIVE_FIR_AVAILABLE, _cognis_native, get_fir_execution_info

from scripts._benchmark_common import build_module_state, dumps_json, to_jsonable


def _benchmark(label: str, fn, iterations: int) -> dict[str, Any]:
    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "iterations": iterations,
        "total_seconds": elapsed,
        "avg_ms": (elapsed / iterations) * 1000.0,
    }


def _backend_result(label: str, backend: FirBackend, audio: np.ndarray, splitter, iterations: int) -> dict[str, Any]:
    def _call() -> None:
        splitter.split(audio, backend=backend)

    timing = _benchmark(label, _call, iterations)
    execution = build_module_state(
        available=_NATIVE_FIR_AVAILABLE,
        imported_module=_cognis_native,
        execution_info=get_fir_execution_info(),
    )
    timing["execution"] = execution
    return timing


def _apply_fir_result(label: str, audio: np.ndarray, taps: np.ndarray, iterations: int) -> dict[str, Any]:
    def _call() -> None:
        apply_fir(audio, taps, backend=FirBackend.AUTO)

    timing = _benchmark(label, _call, iterations)
    timing["execution"] = build_module_state(
        available=_NATIVE_FIR_AVAILABLE,
        imported_module=_cognis_native,
        execution_info=get_fir_execution_info(),
    )
    return timing


def _signal_benchmark(label: str, audio: np.ndarray, splitter, *, is_short: bool) -> dict[str, Any]:
    print(f"\n--- {label} (shape: {audio.shape}) ---")

    reference = _benchmark(
        "reference per-channel fftconvolve(low band)",
        lambda: np.vstack([fftconvolve(channel, splitter.low_taps, mode="same") for channel in audio]),
        iterations=25,
    )

    auto_apply = _apply_fir_result("apply_fir(low band, backend=AUTO)", audio, splitter.low_taps, iterations=25)
    auto_split = _backend_result("splitter.split(audio, backend=AUTO)", FirBackend.AUTO, audio, splitter, iterations=25)
    fft_split = _backend_result("splitter.split(audio, backend=FFT)", FirBackend.FFT, audio, splitter, iterations=25)
    part_split = _backend_result("splitter.split(audio, backend=PARTITIONED)", FirBackend.PARTITIONED, audio, splitter, iterations=25)

    direct_iterations = 25 if is_short else 2
    direct_split = _backend_result("splitter.split(audio, backend=DIRECT)", FirBackend.DIRECT, audio, splitter, iterations=direct_iterations)
    if not is_short:
        direct_split["normalized_to_iterations"] = 25
        direct_split["normalized_total_seconds"] = (direct_split["total_seconds"] / 2.0) * 25.0
        direct_split["normalized_avg_ms"] = (direct_split["normalized_total_seconds"] / 25.0) * 1000.0

    return {
        "label": label,
        "shape": list(audio.shape),
        "reference_low_band": reference,
        "apply_fir_auto": auto_apply,
        "split_auto": auto_split,
        "split_fft": fft_split,
        "split_partitioned": part_split,
        "split_direct": direct_split,
    }


def run_benchmark(*, sample_rate: int, short_seconds: float, long_seconds: float) -> dict[str, Any]:
    rng = np.random.default_rng(29)
    audio_long = rng.standard_normal((2, int(sample_rate * long_seconds))) * 0.1
    audio_short = rng.standard_normal((2, int(short_seconds * sample_rate))) * 0.1

    low_cutoff = 250.0
    high_cutoff = 4000.0
    low_taps = 1537
    high_taps = 513

    clear_fir_design_cache()
    cold_start = time.perf_counter()
    splitter = get_linear_phase_three_band_splitter(
        sample_rate,
        low_cutoff,
        high_cutoff,
        low_taps=low_taps,
        high_taps=high_taps,
    )
    cold_elapsed = time.perf_counter() - cold_start

    warm_lookup = _benchmark(
        "cached splitter lookup",
        lambda: get_linear_phase_three_band_splitter(
            sample_rate,
            low_cutoff,
            high_cutoff,
            low_taps=low_taps,
            high_taps=high_taps,
        ),
        iterations=200,
    )

    short_summary = _signal_benchmark("Short Signal Scenario", audio_short, splitter, is_short=True)
    long_summary = _signal_benchmark("Long Signal Scenario", audio_long, splitter, is_short=False)

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "native_state": build_module_state(
            available=_NATIVE_FIR_AVAILABLE,
            imported_module=_cognis_native,
            execution_info={},
        ),
        "design_cache": {
            "cold_splitter_build_ms": cold_elapsed * 1000.0,
            "cached_lookup": warm_lookup,
            "cache_info": get_fir_design_cache_info(),
        },
        "signals": {
            "short": short_summary,
            "long": long_summary,
        },
    }


def _print_human_summary(summary: dict[str, Any]) -> None:
    print("=== COGNIS FIR crossover benchmark ===")
    print(f"Environment: Python {summary['environment']['python']} on {summary['environment']['platform']}")
    native = summary["native_state"]
    execution = native.get("execution_info") or {}
    print(
        f"Native state: {native['state']} available={native['available']} imported={native['imported']}"
        f" (selected={execution.get('selected_method')}, used_native={execution.get('used_native')}, fallback={execution.get('fallback_triggered')})"
    )
    print(f"Cold splitter build: {summary['design_cache']['cold_splitter_build_ms']:.3f} ms")
    print(
        f"Cached splitter lookup: {summary['design_cache']['cached_lookup']['avg_ms']:.3f} ms/iter over "
        f"{summary['design_cache']['cached_lookup']['iterations']} iterations"
    )

    for signal_key in ("short", "long"):
        signal = summary["signals"][signal_key]
        print(f"\n{signal['label']} {tuple(signal['shape'])}")
        print(
            f"  reference low band: {signal['reference_low_band']['avg_ms']:.3f} ms/iter"
        )
        for backend_key in ("apply_fir_auto", "split_auto", "split_fft", "split_partitioned", "split_direct"):
            result = signal[backend_key]
            execution = result.get("execution") or {}
            line = (
                f"  {backend_key:<18} {result['avg_ms']:.3f} ms/iter "
                f"(state={execution.get('state')}, selected={execution.get('execution_info', {}).get('selected_method')})"
            )
            if "normalized_avg_ms" in result:
                line += f", normalized={result['normalized_avg_ms']:.3f} ms/iter"
            print(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark FIR crossover paths with native-state observability.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Sample rate used to synthesize benchmark audio.")
    parser.add_argument("--short-seconds", type=float, default=512 / 48_000, help="Duration for the short-signal scenario.")
    parser.add_argument("--long-seconds", type=float, default=5.0, help="Duration for the long-signal scenario.")
    args = parser.parse_args()

    summary = run_benchmark(
        sample_rate=args.sample_rate,
        short_seconds=args.short_seconds,
        long_seconds=args.long_seconds,
    )

    if args.json:
        print(dumps_json(to_jsonable(summary)))
    else:
        _print_human_summary(summary)


if __name__ == "__main__":
    main()
