# Analyzer / QC / Reporting

COGNIS now emits first-class deterministic render artifacts alongside mastered audio.

## Artifact Set

For an output like `master.wav`, the canonical artifact writer produces:

- `master.recipe.json`
- `master.analysis.input.json`
- `master.analysis.output.json`
- `master.report.json`
- optional: `master.report.md`

These filenames are deterministic and contain no timestamps.

## Schema Versions

- Analysis artifact: `analysis_schema_v1`
- Render report artifact: `report_schema_v1`
- Recipe artifact: `recipe_v1`

Schema versions are explicit from day one so later optimizer, API, and UI work can reference them safely.

## Analysis Schema v1

`analysis_schema_v1` is measurement-first and contains:

- identity: analyzer version, sample rate, channels, samples, duration
- loudness: integrated / short-term / momentary stats, loudness range, sample peak, true peak, PLR, crest factor
- tonal: spectral tilt, low-vs-mid / high-vs-mid balances, low-end energy distribution, low-band centroid
- stereo: phase correlation, low/mid/high width, side energy, mono-null ratio, left/right balance
- risks: deterministic proxy estimates for limiter stress, codec risk, clipping risk, delivery safety, plus measured peak-density evidence
- notes: which fields are approximations or availability flags

The analysis artifact is intended to be stable enough to reuse in report generation and later target-builder work.

## Report Schema v1

`report_schema_v1` is the derived render judgement layer. It contains:

- requested: requested mode, loudness, ceiling, codec-safe request state, and core config intent
- achieved: the subset of measured output metrics needed for target checks and release review
- delta: requested-vs-achieved loudness, peak margin, codec-safety margin, tonal/stereo/dynamics deltas
- findings: machine-readable QC findings with reason codes, severity, explanation, and measured evidence
- summary: deterministic factual bullets describing what changed
- overall_status: `pass`, `warning`, or `fail`

The report does not duplicate the full input/output analysis payloads. Those live in the analysis artifacts.

## QC Severity

- `fail`: hard safety or target miss that should block release review until fixed
- `warning`: technically renderable, but notable risk or target miss should be reviewed
- `informational`: measured confirmation / context
- `pass`: used for overall status only; individual findings are currently `informational`, `warning`, or `fail`

Current v1 reason codes cover loudness misses, true-peak safety, digital overs, mono/stereo safety, low-end width drift, limiter stress, codec risk, tonal extremes, momentary spikes, and dynamics collapse risk.

## CLI Usage

Canonical artifact-first run:

```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14 --ceiling_db -1 --write-markdown-report
```

Write artifacts to a dedicated directory:

```bash
python -m cognis.cli input.wav output.wav --artifacts-dir build-artifacts
```

Audio-only run:

```bash
python -m cognis.cli input.wav output.wav --no-artifacts
```

## Current Limits

- Loudness is still an approximate BS.1770 implementation, not certification-grade.
- Codec-risk and limiter-stress values are deterministic evidence-backed proxies, not black-box perceptual predictions.
- Reference-track deltas are not implemented yet even though `reference_path` remains part of the request surface.

## Intended Next Extension Points

- richer delivery profiles with target presets and per-platform QC bands
- reference-track comparison deltas
- render-batch aggregation / fleet-level QC summaries
