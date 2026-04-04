# Parameter Schema

## MasteringConfig
- `mode`: MasteringMode (STREAMING_SAFE, CLUB_LOUD, etc.)
- `target_loudness`: float (LUFS)
- `ceiling_mode`: CeilingMode (PEAK, TRUE_PEAK, CODEC_SAFE)
- `ceiling_db`: float (dBFS, must be negative)
- `oversampling`: int (1, 2, 4, 8)
- `bass_preservation`: float (0.0 to 1.0)
- `stereo_width`: float (0.0 to 2.0)
- `dynamics_preservation`: float (0.0 to 1.0)
- `brightness`: float (-1.0 to 1.0)
- `reference_path`: Optional[str]
