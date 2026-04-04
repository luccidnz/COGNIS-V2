from dataclasses import dataclass
from cognis.config import MasteringConfig

@dataclass
class TargetValues:
    target_loudness: float
    ceiling_db: float
    target_tilt: float
    target_width: float

def build_targets(config: MasteringConfig) -> TargetValues:
    """Map config to explicit target values."""
    
    # Base targets
    target_loudness = config.target_loudness
    ceiling_db = config.ceiling_db
    
    # Tonal intent (brightness maps to tilt)
    # Brightness 0.0 -> tilt 0.0
    # Brightness 1.0 -> tilt +3.0 dB/octave approx
    target_tilt = config.brightness * 3.0
    
    # Width constraints
    target_width = config.stereo_width
    
    return TargetValues(
        target_loudness=target_loudness,
        ceiling_db=ceiling_db,
        target_tilt=target_tilt,
        target_width=target_width
    )
