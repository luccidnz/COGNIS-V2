# COGNIS Module Structure

## `cognis.config`
Defines the configuration schema, mastering modes, and ceiling modes.

## `cognis.analysis`
Responsible for measuring the audio.
- `analyzer.py`: Orchestrates the analysis.
- `features.py`: Dataclasses for analysis results.
- `loudness.py`: Loudness measurement (BS.1770 style).
- `spectrum.py`: Spectral analysis.
- `stereo.py`: Stereo width and correlation.
- `preflight.py`: Input validation.

## `cognis.dsp`
White-box DSP processing modules.
- `eq.py`: Corrective and tonal EQ.
- `dynamics.py`: Multiband dynamics.
- `limiter.py`: True-peak limiter.
- `stereo.py`: Stereo field control.
- `filters.py`: Reusable filter utilities.
- `utils.py`: General DSP utilities.

## `cognis.optimizer`
Optimization brain for choosing DSP parameters.
- `objective.py`: Scoring function for candidate parameters.
- `search.py`: Search algorithms over parameter space.
- `targets.py`: Target building from configuration.

## `cognis.models`
Stubs for future ML models.

## `cognis.reports`
Quality control and reporting.
- `qc.py`: versioned target-vs-result reporting, QC findings, and markdown summaries.

## `cognis.serialization`
Saving and loading recipes and configurations.
- `recipe.py`: stable JSON serialization helpers for configs and recipes.
- `artifacts.py`: writes recipe, analysis, and report artifacts to disk.

## `cognis.io`
Audio loading and saving.
