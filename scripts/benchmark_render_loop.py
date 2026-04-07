import cProfile
import pstats
import time
import numpy as np

from cognis.engine import Engine
from cognis.config import MasteringConfig, MasteringMode, CeilingMode
from cognis.dsp.dynamics import MultibandDynamics
from cognis.dsp.limiter import Limiter

def benchmark_full_render_loop():
    engine = Engine()
    config = MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=1,
        bass_preservation=0.9,
        stereo_width=1.0,
        dynamics_preservation=0.5,
        brightness=0.1
    )

    rng = np.random.default_rng(42)
    audio = rng.standard_normal((2, 48000 * 5)) * 0.1

    print("--- Profiling Engine._render_chain (5 iterations) ---")
    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(5):
        engine._render_chain(
            audio, 48000,
            {"brightness": 0.1, "dynamics_preservation": 0.5, "width": 1.0, "bass_preservation": 0.9},
            config, trim_gain_db=-3.0, makeup_gain_db=2.0
        )

    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)

def benchmark_components():
    sr = 48000
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((2, sr * 5)) * 0.1

    dynamics = MultibandDynamics(sr)
    limiter = Limiter(sr)

    print("--- Component Benchmarks (5 iterations, 5s stereo signal) ---")

    # Benchmarking Dynamics
    start_dyn = time.perf_counter()
    for _ in range(5):
        dynamics.process(audio, 0.5)
    dyn_time = time.perf_counter() - start_dyn

    dyn_info = dynamics.last_execution_info or {}
    dyn_native = dyn_info.get("used_native", False)
    dyn_fallback = dyn_info.get("fallback_triggered", False)

    start_lim = time.perf_counter()
    for _ in range(5):
        limiter.process(audio, -1.0, "TRUE_PEAK", 1)
    lim_time = time.perf_counter() - start_lim

    print(f"MultibandDynamics.process: {dyn_time:.3f}s total (Native Used: {dyn_native}, Fallback: {dyn_fallback})")
    print(f"Limiter.process: {lim_time:.3f}s total")

if __name__ == "__main__":
    benchmark_full_render_loop()
    benchmark_components()
