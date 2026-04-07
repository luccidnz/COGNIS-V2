# COGNIS-V2 AI Instructions

This is the `AGENTS.md` file for COGNIS-V2. It contains core guidelines for any automated agent or human contributor working on the project.

## Core Identity
COGNIS-V2 is a deterministic, modular, white-box automated mastering engine. It strictly prioritizes hard constraints, predictable behavior, BS.1770-style loudness compliance, and a cloud/API-first architecture.

## Project Architecture
The project is built around three core layers:
1. **Analyzer**: Measures premaster characteristics deeply and reliably.
2. **DSP Mastering Chain**: A deterministic, white-box DSP chain, including parametric EQs, dynamics control, stereo tools, loudness staging, and true-peak limiting.
3. **Optimization Brain**: Evaluates the output and drives DSP parameter tuning against hard constraints.

## Critical DSP Guidelines
- **Deterministic DSP is paramount.** Outputs for the same audio with the same settings must be numerically reproducible. Do not introduce probabilistic steps in the DSP chain.
- **The FIR crossover is explicitly test-backed.** Its behavior (isolation, reconstruction, finite output properties) is verified by tests. **Do not casually break or bypass the offline FIR setup.**
- **Backend Swappability**: Future optimization work must maintain deterministic execution. When adding a compiled backend or updating the execution path, the existing validation (tests, API contracts) must remain intact. Any backend must support testing against the reference implementation to ensure phase and magnitude equivalency. The Python `execute_python_fir_2d` implementation remains the absolute specification. The native execution paths must mirror the Python-equivalent behavior perfectly. Native compilation is strictly optional.
- **Native Failure Semantics**: Do not swallow exceptions on native runtimes silently. Python fallback must be intentional (e.g. native unavailable, unsupported path explicitly requested). If the native engine actually errors during a supported run, raise the exception explicitly by default to maintain determinism and catch bugs fast. Silent native failure swallowing is disabled.
- **FIR Memory Contract**: C++ extensions must respect the explicit `float64`, channel-first `[channels, samples]`, C-contiguous boundary established in `fir_executor.py`. `mode="same"` alignment behavior is non-negotiable and must match the Python reference exactly. Do not change the FIR contract casually.
- **Performance vs Default Options**: Benchmark regressions must be treated as failures. Performance claims must be benchmark-backed. If a proposed optimization or backend benchmarks worse, do not make it the default path just because the code looks cleaner. `AUTO` should base its heuristic on benchmarks, and must prefer practical efficiency for offline mastering workloads. Benchmark results must clearly log `used_native` and `fallback_triggered` attributes to distinguish true native performance from silent python execution.
- **Native Contract Discipline**: When mapping between Python and C++, use clear enumerations instead of string-typing wherever possible for backend dispatches to ensure robust error checking and performance.
- **Compile Configurations**: Determinism takes precedence over risky optimization shortcuts. Compile flags like `-ffast-math` must be gated off by default.
- **Build Reproducibility**: Dependency fetching during native builds must heavily prioritize standard configuration mechanisms (`find_package(Python ... REQUIRED)`, `find_package(pybind11 CONFIG)`) and local environments to ensure deterministic reproducibility, keeping network fetching strictly as a documented fallback. Do not hide missing native prerequisites behind magical fallback behavior.
- **Native Verification Observability**: Validation of native functionality must not confuse native-unavailable with native-used. If native behavior was requested but python fallback triggered, verification output must make this completely explicit.

## General Engineering Principles
- Maintain test coverage. If you introduce a new execution mode, test it, including testing for equivalent behaviors against established reference backends.
- Keep the codebase focused. Do not add bloat or introduce unnecessary frameworks.
- Leave clear, honest comments/docstrings explaining *why* an approach is taken (e.g. explaining the logic behind `AUTO` backend heuristics).
- Always ensure the baseline test suite (`pytest -q`) passes cleanly.