from cognis.analysis.features import AnalysisResult
from cognis.optimizer.targets import TargetValues

def compute_objective(analysis: AnalysisResult, targets: TargetValues) -> float:
    """
    Compute penalty score for a rendered candidate.
    Lower is better.
    """
    score = 0.0
    
    # ==========================================
    # HARD CONSTRAINTS
    # Heavily penalized if violated to ensure they are avoided
    # ==========================================
    
    # 1. True Peak Constraint
    # Must not exceed ceiling
    tp_violation = max(0.0, analysis.loudness.true_peak_dbfs - targets.ceiling_db)
    score += tp_violation * 10000.0
    
    # 2. Phase Correlation Constraint
    # Must not go catastrophically negative (e.g., below 0.0)
    if analysis.stereo.phase_correlation < 0.0:
        score += abs(analysis.stereo.phase_correlation) * 5000.0
        
    # 3. Low-band Width Constraint
    # Bass should remain relatively mono, guided by target_low_band_width
    # Add a small tolerance (e.g., 0.1) to avoid over-penalizing slight widening
    lb_width_violation = max(0.0, analysis.stereo.low_band_width - (targets.target_low_band_width + 0.1))
    score += lb_width_violation * 5000.0

    # ==========================================
    # SOFT PENALTIES
    # Artistic and technical targets
    # ==========================================
    
    # 1. Loudness Penalty
    # Distance from target integrated loudness
    loudness_diff = abs(analysis.loudness.integrated_lufs - targets.target_loudness)
    score += loudness_diff * 20.0
    
    # 2. Spectral Tilt Penalty
    # Deviation from target overall tonal balance
    tilt_diff = abs(analysis.tonal.spectral_tilt_db_per_decade - targets.target_tilt)
    score += tilt_diff * 10.0
    
    # 3. Low/Mid Tonal Penalty
    low_mid_diff = abs(analysis.tonal.low_mid_balance_db - targets.target_low_mid_balance)
    score += low_mid_diff * 5.0
    
    # 4. High/Mid Tonal Penalty
    high_mid_diff = abs(analysis.tonal.high_mid_balance_db - targets.target_high_mid_balance)
    score += high_mid_diff * 5.0
    
    # 5. Crest-Factor / Dynamics Penalty
    # Penalize over-compression (crest factor getting too small)
    if analysis.loudness.crest_factor_db < targets.target_crest_factor:
        crest_diff = targets.target_crest_factor - analysis.loudness.crest_factor_db
        score += crest_diff * 15.0
    
    # 6. Width Penalty
    # Deviation from target mid-band width
    width_diff = abs(analysis.stereo.mid_band_width - targets.target_width)
    score += width_diff * 10.0

    # 6b. Reference-aware low-end / side-energy guidance when available.
    if targets.target_sub_energy_ratio > 0.0:
        sub_energy_diff = abs(analysis.tonal.sub_energy_ratio - targets.target_sub_energy_ratio)
        score += sub_energy_diff * 50.0
    if targets.target_low_energy_ratio > 0.0:
        low_energy_diff = abs(analysis.tonal.low_energy_ratio - targets.target_low_energy_ratio)
        score += low_energy_diff * 40.0
    if targets.target_side_energy_ratio > 0.0:
        side_energy_diff = abs(analysis.stereo.side_energy_ratio - targets.target_side_energy_ratio)
        score += side_energy_diff * 25.0
    
    # 7. Phase / Mono Compatibility Penalty (Soft)
    # Even if positive, we prefer higher correlation (closer to 1.0)
    mono_penalty = 1.0 - analysis.stereo.phase_correlation
    score += mono_penalty * 5.0

    reference_targeting = targets.reference_targeting
    if reference_targeting is not None:
        reference_metrics = reference_targeting.reference_targets
        score += abs(analysis.loudness.integrated_lufs - reference_metrics["integrated_lufs"]) * 12.0
        score += abs(analysis.tonal.spectral_tilt_db_per_decade - reference_metrics["spectral_tilt_db_per_decade"]) * 8.0
        score += abs(analysis.stereo.mid_band_width - reference_metrics["mid_band_width"]) * 10.0
        score += abs(analysis.stereo.low_band_width - reference_metrics["low_band_width"]) * 10.0
        score += abs(analysis.loudness.crest_factor_db - reference_metrics["crest_factor_db"]) * 8.0
        score += abs(analysis.stereo.phase_correlation - reference_metrics["phase_correlation"]) * 5.0
        score += abs(analysis.tonal.low_energy_ratio - reference_metrics["low_energy_ratio"]) * 6.0
        score += abs(analysis.tonal.sub_energy_ratio - reference_metrics["sub_energy_ratio"]) * 6.0
        score += abs(analysis.tonal.bass_energy_ratio - reference_metrics["bass_energy_ratio"]) * 6.0
        
    return score
