import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional

class MasteringMode(Enum):
    STREAMING_SAFE = "STREAMING_SAFE"
    CLUB_LOUD = "CLUB_LOUD"
    REFERENCE_MATCH = "REFERENCE_MATCH"
    PRESERVE_DYNAMICS = "PRESERVE_DYNAMICS"
    YOUTUBE_SAFE = "YOUTUBE_SAFE"

class CeilingMode(Enum):
    PEAK = "PEAK"
    TRUE_PEAK = "TRUE_PEAK"
    CODEC_SAFE = "CODEC_SAFE"

@dataclass
class MasteringConfig:
    mode: MasteringMode
    target_loudness: float
    ceiling_mode: CeilingMode
    ceiling_db: float
    oversampling: int
    bass_preservation: float
    stereo_width: float
    dynamics_preservation: float
    brightness: float
    reference_path: Optional[str] = None
    fir_backend: str = "AUTO"

    def __post_init__(self):
        if self.fir_backend not in ("AUTO", "DIRECT", "FFT", "PARTITIONED"):
            raise ValueError("fir_backend must be AUTO, DIRECT, FFT, or PARTITIONED")
        if self.oversampling not in (1, 2, 4, 8):
            raise ValueError("oversampling must be 1, 2, 4, or 8")
        if self.ceiling_db >= 0:
            raise ValueError("ceiling_db must be negative")
        self.bass_preservation = max(0.0, min(1.0, self.bass_preservation))
        self.stereo_width = max(0.0, min(2.0, self.stereo_width))
        self.dynamics_preservation = max(0.0, min(1.0, self.dynamics_preservation))
        self.brightness = max(-1.0, min(1.0, self.brightness))
