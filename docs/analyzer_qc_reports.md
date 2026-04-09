# Analyzer / QC / Report Artifacts

COGNIS now emits first-class deterministic artifacts for post-render inspection.

## Schemas

- `analysis_schema_v2`
  - Produced by `cognis.analysis.Analyzer`.
  - Contains versioned identity metadata plus loudness, tonal, stereo, and safety/risk summaries.
  - Identity includes role/source metadata so input, output, and reference artifacts remain distinct.

- `report_schema_v2`
  - Produced by `cognis.reports.qc.build_report()` / `generate_qc_report()`.
  - Captures requested settings, achieved outcome, target deltas, QC findings, a nested reference assessment, and concise human-readable change bullets.

- `recipe_v2`
  - Captures render configuration, chosen DSP parameters, and the derived target values / gain staging context used for the render.
  - When a reference is supplied, the recipe also records the reference-aware target payload.

- `reference_assessment_schema_v1`
  - Captures input-vs-reference and output-vs-reference comparison metrics plus reference-aware findings and summary bullets.

## Canonical render flow

Use the engine-level render path when you need all artifacts in-memory:

```python
from cognis.engine import Engine

engine = Engine()
result = engine.render(audio, sample_rate, config)

result.audio
result.recipe
result.input_analysis
result.output_analysis
result.reference_analysis
result.report
```

Use the CLI when you want audio plus serialized artifacts on disk:

```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14 --ceiling_db -1 --write-markdown-report
python -m cognis.cli input.wav output.wav --reference reference.wav --mode REFERENCE_MATCH --write-markdown-report
```

That writes:

- `output.wav`
- `output.wav.recipe.json`
- `output.analysis.input.json`
- `output.analysis.output.json`
- `output.analysis.reference.json` when a reference is supplied
- `output.wav.report.json`
- `output.wav.report.md` when `--write-markdown-report` is supplied

If you pass `--artifacts-dir`, those JSON / markdown files are written there instead of beside the mastered audio.

## QC severity semantics

- `pass`
  - No configured safety or target rule was violated.
- `informational`
  - Context only. Useful for inspection, not a release block.
- `warning`
  - The render is usable, but there is a measured concern worth review before shipping.
- `fail`
  - A hard safety or strong target miss was detected. Treat as not release-ready until resolved.

Reference-aware runs keep safety and reference comparison separate:

- top-level QC findings still describe release safety and target misses
- the nested reference assessment explains how the render differs from the reference and where safety constraints stayed active
- reference mismatch is not automatically a failure

## Current rule coverage

Implemented now:

- loudness target hit/miss
- true-peak ceiling margin
- sample-peak full-scale safety
- phase / mono-compatibility risk
- low-band width vs bass-preservation target
- limiter stress estimate
- codec-risk estimate
- tonal extremity guardrail
- crest-factor / dynamics-preservation drift
- reference-vs-input and reference-vs-output deltas when a reference is supplied

Still intentionally proxy-based in `v1`:

- limiter stress
- codec risk
- delivery safety

Those fields are explicit heuristics derived from measured evidence, not hidden model scores.

## Versioning expectations

- `analysis_schema_v2`, `report_schema_v2`, `recipe_v2`, and `reference_assessment_schema_v1` are explicit compatibility markers.
- Add a new schema version when field names, payload meaning, or artifact structure changes incompatibly.
- Additive fields that preserve existing semantics can stay within the current version, but the tests and golden fixtures should be updated in the same change.
- The current contract is artifact-first and engine-first; future UI layers should consume these artifacts rather than inventing a separate payload.

## Testing and fixtures

The trust layer is covered by:

- schema/determinism tests in `tests/test_analyzer.py`
- end-to-end engine artifact tests in `tests/test_engine.py`
- reference-aware CLI artifact tests in `tests/test_cli.py`
- golden report cases in `tests/test_reports.py`
- reference target and delta tests in `tests/test_reference_targets.py`

Golden fixture coverage currently includes:

- safe streaming render
- loud render near ceiling
- obvious fail case
- mono safety concern
- limiter-stress-heavy case
- reference-aware render with nested reference assessment

## Current gaps intentionally left for later

- codec safety remains a measurement-backed proxy, not an encoder simulation
- limiter stress remains a measurement-backed proxy, not a distortion model
- delivery-profile presets beyond the current config/targets mapping are not yet implemented
- the reference comparison remains measurement-backed rather than perceptual/similarity-model backed
