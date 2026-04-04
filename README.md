# COGNIS-V2

COGNIS-V2 is a deterministic, modular, white-box automated mastering engine. It is designed to be a serious Phase-1 mastering engine backbone, prioritizing hard constraints, BS.1770-style loudness, and cloud/API-first architecture.

## Architecture

COGNIS-V2 consists of three main layers:
1. **Analyzer**: Measures the premaster deeply and reliably.
2. **DSP Mastering Chain**: A white-box DSP chain including EQ, dynamics, stereo control, loudness staging, and true-peak limiting.
3. **Optimization Brain**: Chooses and refines parameters against hard constraints and artistic targets.

## Current MVP Scope

The current implementation is a Phase-1 Python backbone that provides:
- A structured configuration and schema layer.
- An initial analyzer for loudness (approximate BS.1770), spectrum, and stereo features. Peak values are reported in dBFS.
- A basic white-box DSP chain (EQ, dynamics, stereo control, limiter).
- The multiband dynamics stage now uses a hardened offline linear-phase FIR band split with cached kernel reuse, adaptive convolution execution, exact residual reconstruction, and validation around isolation near and away from the crossover points.
- An optimization layer that performs bounded searches over DSP parameters (`brightness`, `stereo_width`, `bass_preservation`, `dynamics_preservation`).
- A CLI for running the engine on audio files.
- Basic reporting and QC.

## How to Install

Install the package and development dependencies:
```bash
pip install -e .[dev]
```

## How to Run Tests

Run the test suite from the repository root:
```bash
pytest -q
```

*Note: The `pyproject.toml` includes a `pythonpath = ["."]` configuration. This is intentional and allows you to run `pytest` directly from a fresh checkout without needing to run `pip install -e .` first.*

For a lightweight FIR crossover validation/benchmark, run:
```bash
python scripts/benchmark_fir_crossover.py
```

The benchmark reports cold splitter build time, cached lookup overhead, the optimized FIR execution path, a reference per-channel FFT path, and repeated split timings.

## How to Run the CLI

Process an audio file using the CLI:
```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14.0 --ceiling_db -1.0
```

## Known Limitations
- The BS.1770 loudness measurement is an approximation and not yet fully certification-grade.
- The limiter is an envelope-aware quasi-lookahead limiter. While better than a static waveshaper, it is not yet a multi-stage true lookahead limiter.
- The dynamics crossover is now cached, uses a faster adaptive SciPy convolution path, and is better validated, but it is still an offline Python FIR implementation with finite-kernel edge effects and no compiled or partitioned-convolution backend yet.
- Other DSP blocks still use simple first/second-order Butterworth filters where phase shift is acceptable for the current MVP.
- The optimizer uses a small, bounded grid search for deterministic and fast MVP execution.

## Roadmap
- Refine the Limiter and Dynamics modules (e.g., implement a true lookahead envelope-based limiter with smarter release handling).
- Migrate the offline FIR crossover and dynamics block to a higher-performance C++ DSP core or partitioned-convolution path.
- Refine BS.1770 loudness measurement to full compliance.
- Integrate C++20 DSP core via pybind11.
- Develop ML models for style encoding and preference ranking.
