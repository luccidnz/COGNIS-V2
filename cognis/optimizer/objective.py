from cognis.analysis.features import AnalysisResult
from cognis.optimizer.targets import TargetValues

def compute_objective(analysis: AnalysisResult, targets: TargetValues) -> float:
    """
    Compute penalty score for a rendered candidate.
    Lower is better.
    """
    score = 0.0
    
    # Hard constraints (heavily penalized if violated)
    # True peak
    tp_violation = max(0.0, analysis.loudness.true_peak - (10 ** (targets.ceiling_db / 20.0)))
    score += tp_violation * 1000.0
    
    # Soft penalties
    # Loudness
    loudness_diff = abs(analysis.loudness.integrated_loudness - targets.target_loudness)
    score += loudness_diff * 10.0
    
    # Spectral tilt
    tilt_diff = abs(analysis.spectrum.spectral_tilt - targets.target_tilt)
    score += tilt_diff * 5.0
    
    # Stereo width
    width_diff = abs(analysis.stereo.mid_band_width - targets.target_width)
    score += width_diff * 2.0
    
    # Phase correlation (penalize negative correlation)
    if analysis.stereo.phase_correlation < 0.0:
        score += abs(analysis.stereo.phase_correlation) * 50.0
        
    return score
