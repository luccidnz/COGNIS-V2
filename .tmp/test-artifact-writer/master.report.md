# COGNIS Render Report

## Requested

- Mode: `STREAMING_SAFE`
- Overall status: `fail`
- Target loudness: `-14.00 LUFS`
- Ceiling: `-1.00 dBFS` (`TRUE_PEAK`)
- Codec-safe requested: `false`
- Stereo width target: `1.00`
- Dynamics preservation target: `0.50`
- Brightness target: `0.10`

## Achieved

- Achieved loudness: `-14.51 LUFS`
- True peak: `-10.61 dBFS`
- Ceiling margin: `9.61 dB`
- Low-band width: `0.42`
- Phase correlation: `0.15`

## What Changed

- Reduced integrated loudness by 7.14 LUFS.
- Narrowed sub / low-band stereo width by 0.08 ratio.
- Preserved punch within the current crest-factor tolerance.
- Render failed QC and is not release-ready under the current measured constraints.

## QC Findings

- `fail` `LOW_BAND_WIDTH_CRITICAL`: Low-band stereo width materially exceeds the requested mono-safety target.
- `warning` `LOUDNESS_OUTSIDE_TARGET_TOLERANCE`: Integrated loudness is outside the preferred target tolerance.
- `warning` `PHASE_CORRELATION_LOW`: Phase correlation is low and may reduce mono compatibility.
- `informational` `CODEC_SAFE_MARGIN_OK`: Codec-safety proxy remains within the preferred range.
- `informational` `DYNAMICS_PRESERVED_WITHIN_TOLERANCE`: Crest factor remains within the requested dynamics tolerance.
- `informational` `TONAL_BALANCE_WITHIN_EXPECTED_RANGE`: Measured tonal balance stays within the expected operating range.
- `informational` `TRUE_PEAK_WITHIN_MARGIN`: True peak stays within the requested ceiling margin.
