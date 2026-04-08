from __future__ import annotations

import argparse
import platform
import time
from typing import Any

import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

from cognis.dsp import limiter as limiter_module
from cognis.dsp.limiter import Limiter, NATIVE_AVAILABLE

native = getattr(limiter_module, "native", None)

from scripts._benchmark_common import build_module_state, dumps_json, to_jsonable


def _benchmark(label: str, fn, iterations: int) -> dict[str, Any]:
    start = time.perf_counter()
    result = None
    for _ in range(iterations):
        result = fn()
    elapsed = time.perf_counter() - start
    return {
        "label": label,
        "iterations": iterations,
        "total_seconds": elapsed,
        "avg_ms": (elapsed / iterations) * 1000.0,
        "result_shape": tuple(result.shape) if hasattr(result, "shape") else None,
    }


def run_benchmark(*, sample_rate: int, iterations: int) -> dict[str, Any]:
    print("--- Benchmarking limiter smoothing variants ---")

    hold_ms = 1.5
    release_ms = 10.0
    hold_samples = max(1, int((hold_ms / 1000.0) * sample_rate))
    sigma_samples = (release_ms / 1000.0) * sample_rate

    rng = np.random.default_rng(42)
    raw_gain = rng.uniform(0.5, 1.0, size=(sample_rate * 5,))
    limiter = Limiter(sample_rate)

    def _python_path() -> np.ndarray:
        held_gain = minimum_filter1d(raw_gain, size=hold_samples)
        return gaussian_filter1d(held_gain, sigma=sigma_samples)

    python_result = _benchmark("python_scipy_fused", _python_path, iterations)
    smooth_gain_py = _python_path()

    native_state = build_module_state(
        available=NATIVE_AVAILABLE,
        imported_module=native,
        execution_info=limiter.last_execution_info or {},
    )

    if NATIVE_AVAILABLE and native is not None:
        def _native_gaussian_only() -> np.ndarray:
            held_gain = minimum_filter1d(raw_gain, size=hold_samples)
            return native.compute_native_limiter_gain_gaussian_only(np.ascontiguousarray(held_gain), sigma_samples)

        def _native_fused() -> np.ndarray:
            return native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), hold_samples, sigma_samples)

        gaussian_only = _benchmark("native_gaussian_only", _native_gaussian_only, iterations)
        fused = _benchmark("native_fused", _native_fused, iterations)
        smooth_gain_g_only = _native_gaussian_only()
        smooth_gain_fused = _native_fused()

        gaussian_only_pass = bool(np.allclose(smooth_gain_py, smooth_gain_g_only, rtol=1e-5, atol=1e-5))
        fused_pass = bool(np.allclose(smooth_gain_py, smooth_gain_fused, rtol=1e-5, atol=1e-5))
    else:
        gaussian_only = {"label": "native_gaussian_only", "available": False, "skipped": True}
        fused = {"label": "native_fused", "available": False, "skipped": True}
        gaussian_only_pass = None
        fused_pass = None

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "native_state": native_state,
        "sample_rate": sample_rate,
        "benchmarks": {
            "python_scipy_fused": python_result,
            "native_gaussian_only": gaussian_only,
            "native_fused": fused,
        },
        "equivalence": {
            "gaussian_only_matches_python": gaussian_only_pass,
            "fused_matches_python": fused_pass,
        },
    }


def _print_human_summary(summary: dict[str, Any]) -> None:
    print("=== COGNIS limiter helper benchmark ===")
    print(f"Environment: Python {summary['environment']['python']} on {summary['environment']['platform']}")
    native_state = summary["native_state"]
    execution = native_state.get("execution_info") or {}
    print(
        f"Native state: {native_state['state']} available={native_state['available']} imported={native_state['imported']}"
        f" (used_native={execution.get('used_native')}, fallback={execution.get('fallback_triggered')})"
    )
    for key, benchmark in summary["benchmarks"].items():
        if benchmark.get("skipped"):
            print(f"{key}: skipped (native helper unavailable)")
        else:
            print(f"{key}: {benchmark['total_seconds']:.4f}s total, {benchmark['avg_ms']:.2f} ms/iter")
    if summary["equivalence"]["gaussian_only_matches_python"] is not None:
        print(f"Gaussian-only equivalence: {'PASS' if summary['equivalence']['gaussian_only_matches_python'] else 'FAIL'}")
        print(f"Fused equivalence: {'PASS' if summary['equivalence']['fused_matches_python'] else 'FAIL'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark limiter smoothing helpers with native-state observability.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--sample-rate", type=int, default=48_000, help="Sample rate used to synthesize benchmark audio.")
    parser.add_argument("--iterations", type=int, default=50, help="Number of timing iterations for each path.")
    args = parser.parse_args()

    summary = run_benchmark(sample_rate=args.sample_rate, iterations=args.iterations)
    if args.json:
        print(dumps_json(to_jsonable(summary)))
    else:
        _print_human_summary(summary)


if __name__ == "__main__":
    main()
