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
- **Backend Swappability**: Future optimization work must maintain deterministic execution. When adding a compiled backend or a partitioned-convolution execution path, the existing validation (tests, API contracts) must remain intact. Any backend must support testing against the reference implementation to ensure phase and magnitude equivalency. The `PARTITIONED` backend is currently a placeholder for a future zero-latency/compiled execution path.
- **Performance vs Default Options**: Benchmark regressions must be treated as failures. If a proposed optimization or backend benchmarks worse, do not make it the default path just because the code looks cleaner. `AUTO` should base its heuristic on benchmarks, and must prefer practical efficiency for offline mastering workloads.

## General Engineering Principles
- Maintain test coverage. If you introduce a new execution mode, test it, including testing for equivalent behaviors against established reference backends.
- Keep the codebase focused. Do not add bloat or introduce unnecessary frameworks.
- Leave clear, honest comments/docstrings explaining *why* an approach is taken (e.g. explaining the logic behind `AUTO` backend heuristics).
- Always ensure the baseline test suite (`pytest -q`) passes cleanly.