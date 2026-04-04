# COGNIS

COGNIS is a deterministic, modular, white-box automated mastering engine.

## Architecture

COGNIS consists of three main layers:
1. **Analyzer**: Measures the premaster deeply and reliably.
2. **DSP Mastering Chain**: A white-box DSP chain including EQ, dynamics, stereo control, loudness staging, and true-peak limiting.
3. **Optimization Brain**: Chooses and refines parameters against hard constraints and artistic targets.

## Current MVP Scope

The current implementation is a Phase-1 Python backbone that provides:
- A structured configuration and schema layer.
- An initial analyzer for loudness (approximate BS.1770), spectrum, and stereo features.
- A basic white-box DSP chain (EQ, dynamics, stereo control, limiter).
- An optimization layer that performs bounded searches over DSP parameters.
- A CLI for running the engine on audio files.
- Basic reporting and QC.

## How to Run

Install dependencies:
```bash
pip install -e .[dev]
```

Run the CLI:
```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE
```

## Roadmap
- Refine BS.1770 loudness measurement to full compliance.
- Expand DSP modules with more sophisticated algorithms.
- Implement true lookahead in the limiter.
- Integrate C++20 DSP core via pybind11.
- Develop ML models for style encoding and preference ranking.
