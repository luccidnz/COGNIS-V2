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

That editable install is the canonical bootstrap path. It pulls in the Python test tooling and the native build helper dependency (`pybind11`) so the validation script does not require a separate manual `pybind11` install.

## How to Run Tests

Run the test suite from the repository root:
```bash
pytest -q
```

*Note: The `pyproject.toml` includes a `pythonpath = ["."]` configuration. This is intentional and allows you to run `pytest` directly from a fresh checkout without needing to run `pip install -e .` first.*

For an editable install sanity check, run:
```bash
python -m pip install -e ".[dev]"
python -m pip check
```

For a lightweight FIR crossover validation/benchmark, run:
```bash
python -m scripts.benchmark_fir_crossover
```

The benchmark now reports native availability/import/use state, cold splitter build time, cached lookup overhead, the optimized FIR execution path, a reference per-channel FFT path, and repeated split timings across short and long signals. Add `--json` for machine-readable output.

For a broader render-loop profiling pass, run:
```bash
python -m scripts.benchmark_render_loop
```

That command reports the FIR, dynamics, and limiter native-state summary alongside component timings and the overall render-loop timing. It also accepts `--json`.

For the limiter helper micro-benchmark, run:
```bash
python -m scripts.benchmark_limiter_helpers
```

It reports Python-vs-native smoothing timings, skips native-only paths cleanly when the extension is unavailable, and accepts `--json`.

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

To also emit a human-readable markdown report:
```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14.0 --ceiling_db -1.0 --write-markdown-report
```

To write artifacts into a dedicated directory:
```bash
python -m cognis.cli input.wav output.wav --artifacts-dir build-artifacts
```

To render audio only:
```bash
python -m cognis.cli input.wav output.wav --no-artifacts
```

The canonical CLI render path now writes deterministic artifacts alongside the mastered audio:
- `output.recipe.json`
- `output.analysis.input.json`
- `output.analysis.output.json`
- `output.report.json`
- `output.report.md` when markdown output is requested

## Analyzer / QC / Report Artifacts

The render pipeline now exposes three versioned artifacts:
- `analysis_schema_v1`: structured post-render analysis including loudness, tonal balance, stereo, and delivery-risk proxies.
- `report_schema_v1`: requested-vs-achieved deltas, QC findings, and concise "what changed" bullets.
- `recipe_v1`: chosen DSP parameters plus the derived target values and gain staging context used for the render.

QC findings are intentionally severity-based rather than boolean-only:
- `pass`: no blocking issue detected under the current checks
- `informational`: context only
- `warning`: review recommended before release
- `fail`: not release-ready under the measured constraints

See [docs/analyzer_qc_reporting.md](docs/analyzer_qc_reporting.md) for schema and usage details.

## Known Limitations
- The BS.1770 loudness measurement is an approximation and not yet fully certification-grade.
- The limiter is an envelope-aware quasi-lookahead limiter. While better than a static waveshaper, it is not yet a multi-stage true lookahead limiter.
- The dynamics crossover uses a `PARTITIONED` backend that is a robust Python overlap-save implementation. While efficient, it is still bounded by Python execution speed and represents a blueprint for a future C++ migration.
- Other DSP blocks still use simple first/second-order Butterworth filters where phase shift is acceptable for the current MVP.
- The optimizer uses a small, bounded grid search for deterministic and fast MVP execution.

## Native Backend Support
The repository includes an optional high-performance C++ DSP core via `pybind11` (`cpp/` directory).
- **Optionality:** Native support is strictly optional. Normal Python workflows will not require compilation, and missing native support is gracefully handled.
- **Capabilities:** The native backend implements highly-optimized paths for `FFT` and `PARTITIONED` FIR convolution, a dedicated helper for recursive envelope tracking inside Multiband Dynamics, and a fused hold + Gaussian smoothing path for the Limiter. The C++ helpers remove significant Python-side loop overhead and are recommended for standard repeated rendering scenarios.
- **Reference Spec:** The Python implementations (e.g., in `fir_executor.py` and `dynamics.py`) remain the absolute behavioral reference. Native implementations must prove equivalence down to floating point margins.
- **Build/Discovery:** The build process heavily prefers finding standard python development tools (`find_package(Python COMPONENTS Interpreter Development REQUIRED)`) and a standard `pybind11` install (`find_package(pybind11 CONFIG)`) to prevent network brittleness, keeping `FetchContent` as a strictly documented last resort. The editable development install (`pip install -e .[dev]`) is the supported bootstrap path for the native validation flow.
- **Fallback Behavior:** If the native module is absent, the execution safely falls back to Python. If the native module is present but an unsupported backend mode is selected (e.g. `DIRECT`), it explicitly falls back to Python. Fallback on explicit runtime failure is strictly controlled by `_FALLBACK_ON_NATIVE_FAILURE` and is disabled by default to maintain deterministic trust.
- **Error Handling:** If the native runtime fails unexpectedly, it will throw an explicit `RuntimeError` by default. Silent native failure swallowing is disabled.
- **Compile Flags:** The native module strictly avoids aggressive non-deterministic compiler optimizations (like `-ffast-math`) by default to ensure DSP correctness.
- **Validation:** You can run `./scripts/validate_native.sh` to build, test, and benchmark the native integration pipeline explicitly. This script checks explicit prerequisites (Python, CMake, pybind11), refuses to build if prerequisites are missing, clearly proves whether the native module built and loaded successfully, and explicitly distinguishes native path executions from fallback triggers.

### Native Validation Prerequisites & Environment Warning
Successful native validation expects the following to be installed and available:
- Python development headers and tools (e.g., `python3-dev`)
- `cmake`
- `pybind11` available after the editable development install (`pip install -e .[dev]`)

**Important Environment Note:** Missing `pybind11` or other build tools is an **environment/setup issue**, not a **DSP algorithm failure**. If `pybind11` is missing after the editable development install, the native validation flow will fail before CMake configuration. Always ensure the native module is built and importable in the *same* Python environment used for validation to avoid misinterpreting environment failures as algorithmic ones.

### Validation Path & Interpretation Guidance
Run the validation script from a Unix-like shell such as WSL or Git Bash:
```bash
./scripts/validate_native.sh
```
When reviewing validation and benchmark outputs, future automated agents and human developers must clearly distinguish between the following states:
- **Native unavailable:** The C++ module wasn't built (e.g., due to a missing `pybind11` build dependency) or could not be loaded. This is an environment issue.
- **Native built but not imported:** The `.so` was compiled but isn't located where the Python module expects it (e.g., a pathing issue).
- **Native imported and used:** The native module was found, and the benchmark explicitly states `NATIVE` or `Proof: [Backend] executed natively`.
- **Python fallback triggered intentionally:** The native runtime was present but explicitly handed execution back to Python due to an intentionally unsupported mode (like `DIRECT`).
- **Unexpected Python fallback:** The native runtime failed unexpectedly and triggered the fail-loud exception (or explicit fallback if enabled).

GitHub Actions includes two separate native-truth paths:
- `CI` is the merge-gating workflow. It checks editable install sanity and runs the test suite.
- `Native Truth` is a manual/scheduled workflow. It performs the native validation smoke check and uploads benchmark logs without gating merges on noisy performance thresholds.

To build the optional native module manually, ensure you have the prerequisites installed, then run:
```bash
# Provide CMake with a hint to the python environment's pybind11
CMAKE_ARGS="-Dpybind11_DIR=$(python -c 'import pybind11; print(pybind11.get_cmake_dir())')"
mkdir -p cpp/build
cd cpp/build
cmake .. $CMAKE_ARGS
cmake --build .
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
```

## Roadmap
- Refine the Limiter and Dynamics modules (e.g., implement a true lookahead envelope-based limiter with smarter release handling).
- Refine BS.1770 loudness measurement to full compliance.
- Expand the report layer with reference-track deltas and richer delivery-profile presets built on the current analyzer/QC artifacts.
- Develop ML models for style encoding and preference ranking.
