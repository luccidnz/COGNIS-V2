# COGNIS Render Report

## Requested

- Mode: `REFERENCE_MATCH`
- Overall status: `fail`
- Target loudness: `-14.00 LUFS`
- Ceiling: `-1.00 dBFS` (`TRUE_PEAK`)
- Codec-safe requested: `false`
- Stereo width target: `1.00`
- Dynamics preservation target: `1.00`
- Brightness target: `0.00`

## Achieved

- Achieved loudness: `-14.53 LUFS`
- True peak: `-8.52 dBFS`
- Ceiling margin: `7.52 dB`
- Low-band width: `0.43`
- Phase correlation: `0.13`

## What Changed

- Reduced integrated loudness by 2.36 LUFS.
- Narrowed sub / low-band stereo width by 0.07 ratio.
- Measured crest factor fell below the requested dynamics-preservation tolerance.
- Render failed QC and is not release-ready under the current measured constraints.
- Integrated loudness moved away from the reference by 0.53 LUFS; 1.45 LUFS remain.
- Tonal balance matched the reference within tolerance.
- Low-band width moved 0.07 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- Mid-band width moved 0.06 ratio toward the reference; residual gap is 0.06 ratio.
- Crest factor matched the reference within tolerance.

## Reference

- Outcome: `deviated`
- Reference path: `.tmp/test-cli-reference-run/reference.wav`
- Reference analysis schema: `analysis_schema_v2`
- Reference identity: `cognis_analyzer_v2` / 48000 Hz / 2 ch / 12000 samples / 0.25 s / reference / .tmp/test-cli-reference-run/reference.wav

## Reference Comparison

- Integrated loudness: input `-12.16 LUFS`, reference `-13.08 LUFS`, output `-14.53 LUFS`, output moved `0.53 LUFS` toward the reference, residual gap `1.45 LUFS`.
- True peak: input `-6.58 dBFS`, reference `-7.49 dBFS`, output `-8.52 dBFS`, output moved `0.11 dBFS` toward the reference, residual gap `1.03 dBFS`.
- Spectral tilt: input `-4.63 dB/decade`, reference `-4.51 dB/decade`, output `-4.39 dB/decade`, output moved `0.00 dB/decade` toward the reference, residual gap `0.12 dB/decade`.
- Low-mid balance: input `11.94 dB`, reference `11.94 dB`, output `11.92 dB`, output moved `0.02 dB` toward the reference, residual gap `0.02 dB`.
- High-mid balance: input `-1.58 dB`, reference `-1.58 dB`, output `-1.58 dB`, output moved `0.00 dB` toward the reference, residual gap `0.01 dB`.
- Sub energy ratio: input `0.00 ratio`, reference `0.00 ratio`, output `0.00 ratio`, output moved `0.00 ratio` toward the reference, residual gap `0.00 ratio`.
- Bass energy ratio: input `0.49 ratio`, reference `0.49 ratio`, output `0.49 ratio`, output moved `0.00 ratio` toward the reference, residual gap `0.00 ratio`.
- Low-band width: input `0.50 ratio`, reference `0.50 ratio`, output `0.43 ratio`, output moved `0.07 ratio` toward the reference, residual gap `0.07 ratio`.
- Mid-band width: input `0.50 ratio`, reference `0.50 ratio`, output `0.44 ratio`, output moved `0.06 ratio` toward the reference, residual gap `0.06 ratio`.
- Phase correlation: input `-0.01 ratio`, reference `-0.01 ratio`, output `0.13 ratio`, output moved `0.14 ratio` toward the reference, residual gap `0.14 ratio`.
- Crest factor: input `5.48 dB`, reference `5.48 dB`, output `5.92 dB`, output moved `0.45 dB` toward the reference, residual gap `0.45 dB`.

## Reference Summary

- Integrated loudness moved away from the reference by 0.53 LUFS; 1.45 LUFS remain.
- Tonal balance matched the reference within tolerance.
- Low-band width moved 0.07 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- Mid-band width moved 0.06 ratio toward the reference; residual gap is 0.06 ratio.
- Crest factor matched the reference within tolerance.

## Reference Attribution

- `exact` `loudness_ceiling_constraint`: Integrated loudness remained conservative because the reference-aware target was capped at -14.00 LUFS instead of chasing the -13.08 LUFS reference. Tradeoff: Configured loudness target over a closer reference match. Required change: Raise `target_loudness` or loosen the loudness baseline policy.
- `exact` `mono_low_band_width_safety`: Low-band width remained 0.07 ratio narrower than the reference because the mono-safety target was capped at 0.00. Tradeoff: Mono safety over a wider low-band match. Required change: Relax bass preservation or allow a wider low-band cap.
- `exact` `dynamics_preservation_constraint`: Crest factor matched the reference within tolerance. Tradeoff: No tradeoff was required. Required change: No change required.
- `exact` `tonal_correction_limit`: Spectral tilt matched the reference within tolerance. Tradeoff: No tradeoff was required. Required change: No change required.

## Reference Findings

- `warning` `REFERENCE_LOUDNESS_MOVED_AWAY`: Integrated loudness moved away from the reference by 0.53 LUFS; 1.45 LUFS remain.
- `informational` `REFERENCE_TONAL_MATCHED`: Tonal balance matched the reference within tolerance.
- `warning` `REFERENCE_LOW_END_DEVIATION`: Low-band width moved 0.07 ratio toward the reference; sub energy and bass energy remain 0.00 and 0.00 ratio from it.
- `warning` `REFERENCE_STEREO_WIDTH_DEVIATION`: Mid-band width moved 0.06 ratio toward the reference; residual gap is 0.06 ratio.
- `informational` `REFERENCE_DYNAMICS_MATCHED`: Crest factor matched the reference within tolerance.


## Optimizer Decision History

- Selection basis: `exact_bounded_grid_search`
- Evaluated candidates: `36`
- Winner candidate index: `13`
- Multiple candidates tied at the best evaluated score; winner selected deterministically by candidate index.
- Runner-up candidate index: `12`
- Winner score margin to runner-up: `0.0000`
- Within the evaluated bounded grid, the winner beat the runner-up on total objective score, but no positive per-term penalty separation was recorded.
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
