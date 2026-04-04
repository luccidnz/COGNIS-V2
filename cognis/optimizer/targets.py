from dataclasses import dataclass
from cognis.config import MasteringConfig

@dataclass
class TargetValues:
    target_loudness: float
    ceiling_db: float
    target_tilt: float
    target_width: float
    target_crest_factor: float
    target_low_band_width: float

def build_targets(config: MasteringConfig) -> TargetValues:
    """
    Map config to explicit target values.
    
    Assumptions:
    - Brightness maps linearly to spectral tilt (0.0 -> flat, 1.0 -> +3dB/octave).
    - Dynamics preservation scales the target crest factor. A value of 1.0 aims for a 
      healthy master crest factor (e.g. 9.0 dB), while lower values allow more compression.
    - Bass preservation forces the low band to be more mono. A value of 1.0 forces 
      low_band_width close to 0.0, while 0.0 allows it to match the overall width.
    """
    
    # Base targets
    target_loudness = config.target_loudness
    ceiling_db = config.ceiling_db
    
    # Tonal intent (brightness maps to tilt)
    target_tilt = config.brightness * 3.0
    
    # Width constraints
    target_width = config.stereo_width
    
    # Dynamics preservation -> target crest factor
    # 1.0 -> ~9.0 dB (dynamic master)
    # 0.0 -> ~6.0 dB (squashed master)
    target_crest_factor = 6.0 + (config.dynamics_preservation * 3.0)
    
    # Bass preservation -> low band width constraint
    # 1.0 -> 0.0 (mono bass)
    # 0.0 -> target_width (unconstrained bass)
    target_low_band_width = target_width * (1.0 - config.bass_preservation)
    
    return TargetValues(
        target_loudness=target_loudness,
        ceiling_db=ceiling_db,
        target_tilt=target_tilt,
        target_width=target_width,
        target_crest_factor=target_crest_factor,
        target_low_band_width=target_low_band_width
    )
