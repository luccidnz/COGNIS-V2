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
python -m scripts.benchmark_fir_crossover
```

The benchmark reports cold splitter build time, cached lookup overhead, the optimized FIR execution path, a reference per-channel FFT path, and repeated split timings across short and long signals.

## FIR Backend Options

The `fir_backend` configuration in `MasteringConfig` selects how FIR crossovers execute:
- **`AUTO` (default)**: Explicitly selects the best execution strategy based on benchmark-backed heuristics. Falls back to `DIRECT` if NaNs are present to avoid block-wide spreading. Chooses `DIRECT` for exceptionally short workloads, `FFT` for intermediate lengths, and `PARTITIONED` for typical long audio where memory limits and repeated kernel evaluations matter.
- **`PARTITIONED`**: Explicit overlap-save block convolution. For long mastering signals it divides the processing into fixed blocks and caches the real FFT of the kernel. This significantly speeds up optimizer sweeps and serves as the explicit blueprint for future compiled implementations.
- **`FFT`**: Forces `method="fft"`, using monolithic fast convolution.
- **`DIRECT`**: Forces `method="direct"`. Very slow on long kernels; kept for verification.

## How to Run the CLI

Process an audio file using the CLI:
```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14.0 --ceiling_db -1.0
```

## Known Limitations
- The BS.1770 loudness measurement is an approximation and not yet fully certification-grade.
- The limiter is an envelope-aware quasi-lookahead limiter. While better than a static waveshaper, it is not yet a multi-stage true lookahead limiter.
- The dynamics crossover uses a `PARTITIONED` backend that is a robust Python overlap-save implementation. While efficient, it is still bounded by Python execution speed and represents a blueprint for a future C++ migration.
- Other DSP blocks still use simple first/second-order Butterworth filters where phase shift is acceptable for the current MVP.
- The optimizer uses a small, bounded grid search for deterministic and fast MVP execution.

## Native Backend Support
The repository includes an optional high-performance C++ DSP core via `pybind11` (`cpp/` directory).
- **Optionality:** Normal pure-Python installation and workflows will not require compilation.
- **Capabilities:** The native backend implements highly-optimized paths for `FFT` and `PARTITIONED` FIR convolution, as well as a dedicated helper for the recursive envelope tracking and gain calculation loop inside Multiband Dynamics. The C++ helpers remove significant Python-side loop overhead and are recommended for standard repeated rendering scenarios.
- **Reference Spec:** The Python implementations (e.g., in `fir_executor.py` and `dynamics.py`) remain the absolute behavioral reference. Native implementations must prove equivalence down to floating point margins.
- **Build/Discovery:** The build process heavily prefers finding standard python development tools (`find_package(Python COMPONENTS Interpreter Development REQUIRED)`) and a standard `pybind11` install (`find_package(pybind11 CONFIG)`) to prevent network brittleness, keeping `FetchContent` as a strictly documented last resort.
- **Fallback Behavior:** If the native module is absent, the execution safely falls back to Python. If the native module is present but an unsupported backend mode is selected (e.g. `DIRECT`), it explicitly falls back to Python. Fallback on explicit runtime failure is strictly controlled by `_FALLBACK_ON_NATIVE_FAILURE` and is disabled by default to maintain deterministic trust.
- **Error Handling:** If the native runtime fails unexpectedly, it will throw an explicit `RuntimeError` by default. Silent native failure swallowing is disabled.
- **Compile Flags:** The native module strictly avoids aggressive non-deterministic compiler optimizations (like `-ffast-math`) by default to ensure DSP correctness.
- **Validation:** You can run `./scripts/validate_native.sh` to build, test, and benchmark the native integration pipeline explicitly. This script checks explicit prerequisites (Python, CMake, pybind11), refuses to build if prerequisites are missing, clearly proves whether the native module built and loaded successfully, and explicitly distinguishes native path executions from fallback triggers.

### Native Validation Prerequisites & Environment Warning
Successful native validation expects the following to be installed and available:
- Python development headers and tools (e.g., `python3-dev`)
- `cmake`
- `pybind11` **installed directly in the active Python environment** (`pip install pybind11`)

**Important Environment Note:** If `pybind11` is missing from the active Python environment, the native proof script can hang, time out, or appear to fail even if the underlying native DSP algorithmic logic (like FFT or PARTITIONED) is completely correct. Always ensure the native module is built and importable in the *same* Python environment used for validation to avoid misinterpreting environment failures as algorithmic ones.

### Validation Path & Interpretation Guidance
Run the validation script using:
```bash
./scripts/validate_native.sh
```
When reviewing validation and benchmark outputs, clearly distinguish between:
- **Native unavailable:** The C++ module wasn't built or imported successfully (e.g. environment issue).
- **Native built but not imported:** The `.so` was compiled but isn't located where the python module expects it.
- **Native imported and used:** The benchmark explicitly states `NATIVE` or `Proof: [Backend] executed natively`.
- **Python fallback triggered:** The native runtime was present but explicitly handed execution back to Python due to an error or an unsupported mode.

To build the optional native module manually, ensure you have the prerequisites installed, then run:
```bash
# Provide CMake with a hint to the python environment's pybind11
CMAKE_ARGS="-Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
mkdir -p cpp/build
cd cpp/build
cmake .. $CMAKE_ARGS
make
cp cognis_native*.so ../../cognis/dsp/
```

## Roadmap
- Refine the Limiter and Dynamics modules (e.g., implement a true lookahead envelope-based limiter with smarter release handling).
- Profile other Python DSP orchestration hot-spots (e.g. envelope follower/gain smoothing inside the Dynamics stage).
- Refine BS.1770 loudness measurement to full compliance.
- Develop ML models for style encoding and preference ranking.
