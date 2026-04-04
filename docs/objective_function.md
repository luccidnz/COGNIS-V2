# Objective Function

The objective function evaluates a candidate set of DSP parameters by rendering the audio and analyzing the result.

## Hard Constraints
- True Peak <= Ceiling
- Integrated Loudness ~ Target Loudness

## Soft Penalties
- Loudness penalty: Distance from target loudness.
- True-peak penalty: Distance from ceiling (if exceeding).
- Spectral tilt penalty: Deviation from target tonal balance.
- Dynamics penalty: Excessive reduction in crest factor.
- Stereo penalty: Excessive width or negative correlation.
