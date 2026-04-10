# Analyzer / QC / Reporting

COGNIS now emits first-class deterministic render artifacts alongside mastered audio.

## Artifact Set

For an output like `master.wav`, the canonical artifact writer produces:

- `master.recipe.json`
- `master.analysis.input.json`
- `master.analysis.output.json`
- `master.analysis.reference.json` when `--reference` is supplied
- `master.decision_history.json` for reference-aware runs
- `master.report.json`
- optional: `master.report.md`

These filenames are deterministic and contain no timestamps.

## Schema Versions

- Analysis artifact: `analysis_schema_v2`
- Render report artifact: `report_schema_v4`
- Recipe artifact: `recipe_v2`
- Reference assessment artifact: `reference_assessment_schema_v2`
- Decision-history artifact: `decision_history_schema_v1`

The reference assessment also carries a nested attribution payload for constraint-aware explanations, versioned as `reference_attribution_schema_v1`.

Schema versions are explicit from day one so later optimizer, API, and UI work can reference them safely.

## Analysis Schema v2

`analysis_schema_v2` is measurement-first and contains:

- identity: analyzer version, sample rate, channels, samples, duration, role, source path/name metadata
- loudness: integrated / short-term / momentary stats, loudness range, sample peak, true peak, PLR, crest factor
- tonal: spectral tilt, low-vs-mid / high-vs-mid balances, low-end energy distribution, low-band centroid
- stereo: phase correlation, low/mid/high width, side energy, mono-null ratio, left/right balance
- risks: deterministic proxy estimates for limiter stress, codec risk, clipping risk, delivery safety, plus measured peak-density evidence
- notes: which fields are approximations or availability flags

The analysis artifact is intended to be stable enough to reuse in report generation and later target-builder work.

## Report Schema v4

`report_schema_v4` is the derived render judgement layer. It contains:

- requested: requested mode, loudness, ceiling, codec-safe request state, and core config intent
- achieved: the subset of measured output metrics needed for target checks and release review
- delta: requested-vs-achieved loudness, peak margin, codec-safety margin, tonal/stereo/dynamics deltas
- findings: machine-readable QC findings with reason codes, severity, explanation, and measured evidence
- summary: deterministic factual bullets describing what changed
- reference assessment: nested reference-vs-input/output comparisons, reference-aware summary bullets, and constraint-aware attribution when a reference is supplied
- decision history summary: a compact bounded-grid optimizer summary for markdown/report convenience when a decision-history artifact exists
- overall_status: `pass`, `warning`, or `fail`

The report does not duplicate the full input/output analysis payloads. Those live in the analysis artifacts.

## QC Severity

- `fail`: hard safety or target miss that should block release review until fixed
- `warning`: technically renderable, but notable risk or target miss should be reviewed
- `informational`: measured confirmation / context
- `pass`: used for overall status only; individual findings are currently `informational`, `warning`, or `fail`

Current top-level reason codes cover loudness misses, true-peak safety, digital overs, mono/stereo safety, low-end width drift, limiter stress, codec risk, tonal extremes, momentary spikes, and dynamics collapse risk.
Reference runs add a nested assessment rather than replacing those top-level safety findings.

The reference assessment and decision-history layer keep explanation labels honest:

- `exact` means the report can point to a direct measured blocker or threshold
- `inferred` means the report can connect the result to recorded target-plan or winner-vs-alternative evidence, but the prose is still an interpretation of exact numbers
- `unavailable` means the report could not attribute the residual gap without inventing causality

## Decision History Schema v1

`decision_history_schema_v1` is a first-class sibling artifact for reference-aware runs. It contains:

- search metadata: bounded-grid identity, parameter axes, ranking rule, and explicit limitations
- selection metadata: winner, runner-up, score margin, and tie count
- evaluated candidates: deterministic candidate ordering with exact per-candidate objective attribution
- selection tradeoffs: winner-vs-runner-up separation terms, with exact penalty deltas and inferred summary prose
- reference metric tradeoffs: exact statements when a closer evaluated reference match existed, plus unavailable markers when the trace cannot support more

This artifact is bounded on purpose:

- it covers the exact deterministic candidates that were evaluated
- it does not claim a continuous optimizer path
- it does not claim a global optimum outside the evaluated grid

## CLI Usage

Canonical artifact-first run:

```bash
python -m cognis.cli input.wav output.wav --mode STREAMING_SAFE --target_loudness -14 --ceiling_db -1 --write-markdown-report
python -m cognis.cli input.wav output.wav --reference reference.wav --mode REFERENCE_MATCH --write-markdown-report
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
- Reference-track deltas are deterministic, measurement-backed comparisons, not similarity-model scores.

## Intended Next Extension Points

- richer delivery profiles with target presets and per-platform QC bands
- render-batch aggregation / fleet-level QC summaries
