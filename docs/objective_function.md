# Objective Function

The objective function evaluates a candidate set of DSP parameters by rendering the audio and analyzing the result. It computes a penalty score where lower is better.

The function explicitly separates hard constraints (which are heavily penalized to ensure they are avoided) from soft penalties (which guide the artistic and technical targets).

## Hard Constraints
- **True Peak**: Must not exceed the specified ceiling.
- **Phase Correlation**: Must not go catastrophically negative (below 0.0).
- **Low-band Width**: Bass should remain relatively mono (width < 0.3).

## Soft Penalties
- **Loudness Penalty**: Distance from the target integrated loudness.
- **Spectral Tilt Penalty**: Deviation from the target overall tonal balance.
- **Low/Mid Tonal Penalty**: Excessive bass buildup or loss relative to the mid-band.
- **High/Mid Tonal Penalty**: Excessive harshness or dullness relative to the mid-band.
- **Crest-Factor / Dynamics Penalty**: Penalizes over-compression if the crest factor drops below a safe target (e.g., 9.0 dB).
- **Width Penalty**: Deviation from the target mid-band stereo width.
- **Phase / Mono Compatibility Penalty**: Soft penalty for phase correlation dropping below 1.0, encouraging better mono compatibility.
