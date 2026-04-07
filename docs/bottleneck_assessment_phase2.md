# Post-FIR Bottleneck Assessment (Phase 2)

## 1. Profiling Summary

A detailed profiling run simulating the optimizer's repeated render loop (5 passes of a 5-second 48kHz stereo signal) demonstrates clear performance bottlenecks remaining after FIR crossover acceleration.

The profiling was conducted via the new `scripts/benchmark_render_loop.py` script.

**Full Engine Render Timing (5 iterations):**
*   Total DSP Chain execution: ~9.0 seconds
*   **MultibandDynamics**: ~4.9 seconds (~54% of render time)
    *   `MultibandDynamics._compress_band`: ~4.6 seconds (spent almost entirely in the per-sample Python `for` loop)
*   **Limiter**: ~3.7 seconds (~41% of render time)
    *   Spent almost entirely inside `scipy.ndimage.gaussian_filter1d` for lookahead gain smoothing.

## 2. Hotspot Ranking

1.  **Primary Hotspot**: The explicit per-sample envelope calculation loop inside `cognis/dsp/dynamics.py` (`_compress_band`). This loop tracks attack/release and computes recursive state, which is fundamentally unvectorized in standard NumPy.
2.  **Secondary Hotspot**: The quasi-lookahead `gaussian_filter1d` step inside `cognis/dsp/limiter.py` (`process`). The cost of convolving a large Gaussian kernel dynamically with standard SciPy tools creates a noticeable drag on the loop.

## 3. Is `dynamics.py` the True Next Bottleneck?

**Yes.** The profiling proves unequivocally that `dynamics.py` is the main drag on the optimizer loop, precisely because of the sample-by-sample Python envelope execution.

## 4. Is the Limiter Smoothing the Secondary Bottleneck?

**Yes.** The `gaussian_filter1d` calls add up very quickly during repeated renders. Since the limiter runs after dynamics, accelerating the dynamics envelope first is the logical next step, but the limiter will immediately become the new primary constraint once dynamics are solved.

## 5. Next Engineering Step Recommendation

**Primary Next Step:** Introduce a narrow native/C++ acceleration boundary for the stateful envelope and gain smoothing loop inside `dynamics.py`.

*   **Why a C++ helper?** The envelope loop is fundamentally recursive (`y[n] = alpha * y[n-1] + beta * x[n]`). This is extremely difficult to accelerate in pure Python/NumPy without breaking exact mathematical/behavioral equivalency. A C++ extension (similar to `apply_fir`) is the safest, most performant way to preserve exact deterministic behavior while drastically cutting loop times.
*   We strongly advise against rushing a pure Python approximation (like `lfilter` hacks), as attack/release times are state-dependent. Python remains the reference spec, so a compiled C++ pathway maintains the strict testing contract established by the FIR optimizations.

**Secondary Next Step:** Once dynamics are accelerated, perform an assessment to restructure or compile the limiter smoothing step (`gaussian_filter1d`) to alleviate the remaining overhead.
