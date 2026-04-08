from __future__ import annotations

import argparse
import platform
import time
from typing import Any

import numpy as np

from cognis.config import CeilingMode, MasteringConfig, MasteringMode
from cognis.dsp import limiter as limiter_module
from cognis.dsp.dynamics import _NATIVE_AVAILABLE as DYNAMICS_NATIVE_AVAILABLE, cognis_native as dynamics_native
from cognis.dsp.dynamics import MultibandDynamics
from cognis.dsp.eq import EQ
from cognis.dsp.fir_executor import _NATIVE_FIR_AVAILABLE, _cognis_native, get_fir_execution_info
from cognis.dsp.limiter import Limiter, NATIVE_AVAILABLE as LIMITER_NATIVE_AVAILABLE
from cognis.dsp.stereo import StereoControl
from cognis.engine import Engine

from scripts._benchmark_common import build_module_state, dumps_json, to_jsonable

limiter_native = getattr(limiter_module, "native", None)


def _time_call(fn, iterations: int) -> tuple[float, Any]:
    result = None
    start = time.perf_counter()
    for _ in range(iterations):
        result = fn()
    elapsed = time.perf_counter() - start
    return elapsed, result


def _render_loop_summary(engine: Engine, audio: np.ndarray, config: MasteringConfig, params: dict[str, float], *, iterations: int) -> dict[str, Any]:
    total, _ = _time_call(
        lambda: engine._render_chain(
            audio,
            48000,
            params,
            config,
            trim_gain_db=-3.0,
            makeup_gain_db=2.0,
        ),
        iterations,
    )

    return {
        "iterations": iterations,
        "total_seconds": total,
        "avg_ms": (total / iterations) * 1000.0,
    }


def _component_breakdown(audio: np.ndarray, config: MasteringConfig, params: dict[str, float], *, iterations: int) -> dict[str, Any]:
    sr = 48000
    eq = EQ(sr)
    dynamics = MultibandDynamics(sr, backend=config.fir_backend)
    stereo = StereoControl(sr)
    limiter = Limiter(sr)

    timing = {"eq": 0.0, "dynamics": 0.0, "stereo": 0.0, "limiter": 0.0}
    last_output = audio

    for _ in range(iterations):
        stage = audio

        start = time.perf_counter()
        stage = eq.process(stage, params["brightness"])
        timing["eq"] += time.perf_counter() - start

        start = time.perf_counter()
        stage = dynamics.process(stage, params["dynamics_preservation"])
        timing["dynamics"] += time.perf_counter() - start

        start = time.perf_counter()
        stage = stereo.process(stage, params["width"], params["bass_preservation"])
        timing["stereo"] += time.perf_counter() - start

        start = time.perf_counter()
        last_output = limiter.process(
            stage,
            ceiling_db=config.ceiling_db,
            mode=config.ceiling_mode.value,
            oversampling=config.oversampling,
        )
        timing["limiter"] += time.perf_counter() - start

    return {
        "iterations": iterations,
        "avg_ms": {name: (seconds / iterations) * 1000.0 for name, seconds in timing.items()},
        "total_seconds": timing,
        "last_output_shape": tuple(last_output.shape),
        "fir": build_module_state(
            available=_NATIVE_FIR_AVAILABLE,
            imported_module=_cognis_native,
            execution_info=get_fir_execution_info(),
        ),
        "dynamics": build_module_state(
            available=DYNAMICS_NATIVE_AVAILABLE,
            imported_module=dynamics_native,
            execution_info=dynamics.last_execution_info or {},
        ),
        "limiter": build_module_state(
            available=LIMITER_NATIVE_AVAILABLE,
            imported_module=limiter_native,
            execution_info=limiter.last_execution_info or {},
        ),
    }


def run_benchmark(*, render_iterations: int, component_iterations: int) -> dict[str, Any]:
    sr = 48000
    rng = np.random.default_rng(42)
    audio = rng.standard_normal((2, sr * 5)) * 0.1

    config = MasteringConfig(
        mode=MasteringMode.STREAMING_SAFE,
        target_loudness=-14.0,
        ceiling_mode=CeilingMode.TRUE_PEAK,
        ceiling_db=-1.0,
        oversampling=1,
        bass_preservation=0.9,
        stereo_width=1.0,
        dynamics_preservation=0.5,
        brightness=0.1,
    )
    params = {
        "brightness": 0.1,
        "dynamics_preservation": 0.5,
        "width": 1.0,
        "bass_preservation": 0.9,
    }

    engine = Engine()
    component_breakdown = _component_breakdown(audio, config, params, iterations=component_iterations)

    return {
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "signal": {
            "sample_rate": sr,
            "channels": int(audio.shape[0]),
            "seconds": 5.0,
        },
        "native_state": {
            "fir": component_breakdown["fir"],
            "dynamics": component_breakdown["dynamics"],
            "limiter": component_breakdown["limiter"],
        },
        "component_breakdown": component_breakdown,
        "render_loop": _render_loop_summary(engine, audio, config, params, iterations=render_iterations),
    }


def _print_human_summary(summary: dict[str, Any]) -> None:
    print("=== COGNIS render loop benchmark ===")
    print(f"Environment: Python {summary['environment']['python']} on {summary['environment']['platform']}")
    print(f"Signal: {summary['signal']['channels']}ch @ {summary['signal']['sample_rate']} Hz for {summary['signal']['seconds']}s")

    for name, state in summary["native_state"].items():
        execution = state.get("execution_info") or {}
        extra = []
        if execution.get("selected_method"):
            extra.append(f"selected={execution['selected_method']}")
        if execution.get("used_native") is not None:
            extra.append(f"used_native={execution.get('used_native')}")
        if execution.get("fallback_triggered") is not None:
            extra.append(f"fallback={execution.get('fallback_triggered')}")
        print(
            f"{name.capitalize():<9} native: state={state['state']} available={state['available']} imported={state['imported']}"
            + (f" ({', '.join(extra)})" if extra else "")
        )

    breakdown = summary["component_breakdown"]
    print(f"Component breakdown over {breakdown['iterations']} iterations:")
    for name, avg_ms in breakdown["avg_ms"].items():
        print(f"  {name:<8} {avg_ms:.3f} ms/iter")

    print(
        f"Render loop over {summary['render_loop']['iterations']} iterations: "
        f"{summary['render_loop']['total_seconds']:.4f}s total, {summary['render_loop']['avg_ms']:.3f} ms/iter"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the COGNIS render loop with native-state observability.")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON only.")
    parser.add_argument("--render-iterations", type=int, default=8, help="Number of full render-loop iterations.")
    parser.add_argument("--component-iterations", type=int, default=8, help="Number of stage-level component iterations.")
    args = parser.parse_args()

    summary = run_benchmark(render_iterations=args.render_iterations, component_iterations=args.component_iterations)

    if args.json:
        print(dumps_json(to_jsonable(summary)))
    else:
        _print_human_summary(summary)


if __name__ == "__main__":
    main()
