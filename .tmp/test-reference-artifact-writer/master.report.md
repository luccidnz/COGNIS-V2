# COGNIS Render Report

## Requested

- Mode: `REFERENCE_MATCH`
- Overall status: `fail`
- Target loudness: `-14.00 LUFS`
- Ceiling: `-1.00 dBFS` (`TRUE_PEAK`)
- Codec-safe requested: `false`
- Stereo width target: `1.00`
- Dynamics preservation target: `0.50`
- Brightness target: `0.10`

## Achieved

- Achieved loudness: `-14.57 LUFS`
- True peak: `-10.59 dBFS`
- Ceiling margin: `9.59 dB`
- Low-band width: `0.42`
- Phase correlation: `0.15`

## What Changed

- Reduced integrated loudness by 7.19 LUFS.
- Narrowed sub / low-band stereo width by 0.08 ratio.
- Measured crest factor fell below the requested dynamics-preservation tolerance.
- Render failed QC and is not release-ready under the current measured constraints.
- Integrated loudness moved away from the reference by 5.36 LUFS; 6.28 LUFS remain.
- Tonal balance matched the reference within tolerance.
- Low-band width moved 0.08 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- Mid-band width moved 0.05 ratio toward the reference; residual gap is 0.05 ratio.
- Crest factor matched the reference within tolerance.

## Reference

- Outcome: `deviated`
- Reference path: `reference.wav`
- Reference analysis schema: `analysis_schema_v2`
- Reference identity: `cognis_analyzer_v2` / 48000 Hz / 2 ch / 24000 samples / 0.50 s / reference / reference.wav

## Reference Comparison

- Integrated loudness: input `-7.37 LUFS`, reference `-8.29 LUFS`, output `-14.57 LUFS`, output moved `5.36 LUFS` toward the reference, residual gap `6.28 LUFS`.
- True peak: input `-3.74 dBFS`, reference `-4.66 dBFS`, output `-10.59 dBFS`, output moved `5.02 dBFS` toward the reference, residual gap `5.94 dBFS`.
- Spectral tilt: input `-5.08 dB/decade`, reference `-4.97 dB/decade`, output `-4.30 dB/decade`, output moved `0.55 dB/decade` toward the reference, residual gap `0.67 dB/decade`.
- Low-mid balance: input `9.57 dB`, reference `9.57 dB`, output `9.57 dB`, output moved `0.00 dB` toward the reference, residual gap `0.00 dB`.
- High-mid balance: input `-2.50 dB`, reference `-2.50 dB`, output `-2.49 dB`, output moved `0.01 dB` toward the reference, residual gap `0.01 dB`.
- Sub energy ratio: input `0.00 ratio`, reference `0.00 ratio`, output `0.00 ratio`, output moved `0.00 ratio` toward the reference, residual gap `0.00 ratio`.
- Bass energy ratio: input `0.48 ratio`, reference `0.48 ratio`, output `0.48 ratio`, output moved `0.00 ratio` toward the reference, residual gap `0.00 ratio`.
- Low-band width: input `0.50 ratio`, reference `0.50 ratio`, output `0.42 ratio`, output moved `0.08 ratio` toward the reference, residual gap `0.08 ratio`.
- Mid-band width: input `0.50 ratio`, reference `0.50 ratio`, output `0.45 ratio`, output moved `0.05 ratio` toward the reference, residual gap `0.05 ratio`.
- Phase correlation: input `0.00 ratio`, reference `0.00 ratio`, output `0.15 ratio`, output moved `0.15 ratio` toward the reference, residual gap `0.15 ratio`.
- Crest factor: input `5.99 dB`, reference `5.99 dB`, output `6.38 dB`, output moved `0.39 dB` toward the reference, residual gap `0.39 dB`.

## Reference Summary

- Integrated loudness moved away from the reference by 5.36 LUFS; 6.28 LUFS remain.
- Tonal balance matched the reference within tolerance.
- Low-band width moved 0.08 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- Mid-band width moved 0.05 ratio toward the reference; residual gap is 0.05 ratio.
- Crest factor matched the reference within tolerance.

## Reference Attribution

- `exact` `loudness_ceiling_constraint`: Integrated loudness remained conservative because the reference-aware target was capped at -14.00 LUFS instead of chasing the -8.29 LUFS reference. Tradeoff: Configured loudness target over a closer reference match. Required change: Raise `target_loudness` or loosen the loudness baseline policy.
- `exact` `mono_low_band_width_safety`: Low-band width remained 0.08 ratio narrower than the reference because the mono-safety target was capped at 0.10. Tradeoff: Mono safety over a wider low-band match. Required change: Relax bass preservation or allow a wider low-band cap.
- `exact` `dynamics_preservation_constraint`: Crest factor matched the reference within tolerance. Tradeoff: No tradeoff was required. Required change: No change required.
- `exact` `tonal_correction_limit`: Spectral tilt matched the reference within tolerance. Tradeoff: No tradeoff was required. Required change: No change required.

## Reference Findings

- `warning` `REFERENCE_LOUDNESS_MOVED_AWAY`: Integrated loudness moved away from the reference by 5.36 LUFS; 6.28 LUFS remain.
- `informational` `REFERENCE_TONAL_MATCHED`: Tonal balance matched the reference within tolerance.
- `warning` `REFERENCE_LOW_END_DEVIATION`: Low-band width moved 0.08 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- `warning` `REFERENCE_STEREO_WIDTH_DEVIATION`: Mid-band width moved 0.05 ratio toward the reference; residual gap is 0.05 ratio.
- `informational` `REFERENCE_DYNAMICS_MATCHED`: Crest factor matched the reference within tolerance.


## Optimizer Decision History

- Selection basis: `exact_bounded_grid_search`
- Evaluated candidates: `36`
- Winner candidate index: `13`
- Runner-up candidate index: `12`
- Winner score margin to runner-up: `0.0000`
- Within the evaluated bounded grid, the winner beat the runner-up on total objective score, with the largest separating penalties in low_band_width_cap.
- Limitation: This artifact covers all evaluated candidates in the bounded deterministic grid, not the full continuous parameter space.
- Limitation: The optimizer does not record evidence for a global optimum outside the evaluated bounded grid.

## QC Findings

- `fail` `LOW_BAND_WIDTH_CRITICAL`: Low-band stereo width materially exceeds the requested mono-safety target.
- `warning` `DYNAMICS_COLLAPSE_RISK`: Crest factor fell materially below the requested dynamics-preservation target.
- `warning` `LOUDNESS_OUTSIDE_TARGET_TOLERANCE`: Integrated loudness is outside the preferred target tolerance.
- `warning` `PHASE_CORRELATION_LOW`: Phase correlation is low and may reduce mono compatibility.
- `informational` `CODEC_SAFE_MARGIN_OK`: Codec-safety proxy remains within the preferred range.
- `informational` `TONAL_BALANCE_WITHIN_EXPECTED_RANGE`: Measured tonal balance stays within the expected operating range.
- `informational` `TRUE_PEAK_WITHIN_MARGIN`: True peak stays within the requested ceiling margin.
