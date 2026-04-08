import time
import numpy as np
from scipy.ndimage import gaussian_filter1d, minimum_filter1d

try:
    import cognis.dsp.cognis_native as native
except ImportError:
    native = None

def benchmark_limiter_helpers():
    print("--- Benchmarking Limiter Smoothing Variants ---")

    sr = 48000
    hold_ms = 1.5
    release_ms = 10.0

    hold_samples = max(1, int((hold_ms / 1000.0) * sr))
    sigma_samples = (release_ms / 1000.0) * sr

    # Generate 5 seconds of mock raw_gain
    rng = np.random.default_rng(42)
    raw_gain = rng.uniform(0.5, 1.0, size=(sr * 5,))

    iterations = 50

    # 1. Pure Python (SciPy)
    start_time = time.perf_counter()
    for _ in range(iterations):
        held_gain = minimum_filter1d(raw_gain, size=hold_samples)
        smooth_gain_py = gaussian_filter1d(held_gain, sigma=sigma_samples)
    py_time = time.perf_counter() - start_time
    print(f"Python (SciPy) Fused Path: {py_time:.4f}s total ({py_time/iterations*1000:.2f}ms/iter)")

    # 2. Native Gaussian-only
    start_time = time.perf_counter()
    for _ in range(iterations):
        held_gain = minimum_filter1d(raw_gain, size=hold_samples)
        smooth_gain_g_only = native.compute_native_limiter_gain_gaussian_only(np.ascontiguousarray(held_gain), sigma_samples)
    g_only_time = time.perf_counter() - start_time
    print(f"Native Gaussian-Only Path: {g_only_time:.4f}s total ({g_only_time/iterations*1000:.2f}ms/iter)")

    # 3. Native Fused (Hold + Gaussian)
    start_time = time.perf_counter()
    for _ in range(iterations):
        smooth_gain_fused = native.compute_native_limiter_gain_fused(np.ascontiguousarray(raw_gain), hold_samples, sigma_samples)
    fused_time = time.perf_counter() - start_time
    print(f"Native Fused Path:         {fused_time:.4f}s total ({fused_time/iterations*1000:.2f}ms/iter)")

    # Verify correctness
    # We use rtol=1e-5 because FFT convolution has a tiny bit more floating point drift than direct spatial correlation.
    rtol = 1e-5
    atol = 1e-5

    np.testing.assert_allclose(smooth_gain_py, smooth_gain_g_only, rtol=rtol, atol=atol)
    print("Gaussian-only equivalence: PASS")

    np.testing.assert_allclose(smooth_gain_py, smooth_gain_fused, rtol=rtol, atol=atol)
    print("Fused equivalence: PASS")

if __name__ == "__main__":
    benchmark_limiter_helpers()
