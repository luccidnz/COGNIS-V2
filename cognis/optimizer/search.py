import numpy as np
from typing import Dict, Any, Callable
from cognis.optimizer.targets import TargetValues
from cognis.optimizer.objective import compute_objective
from cognis.analysis.analyzer import Analyzer

def grid_search(
    audio: np.ndarray, 
    sr: int, 
    targets: TargetValues, 
    render_fn: Callable[[np.ndarray, Dict[str, float]], np.ndarray],
    analyzer: Analyzer
) -> Dict[str, float]:
    """
    Bounded first-pass search over a small space.
    render_fn takes (audio, params) and returns rendered audio.
    """
    
    # Define search space
    # For MVP, we search over a small grid of brightness and width
    brightness_grid = [-0.5, 0.0, 0.5]
    width_grid = [0.8, 1.0, 1.2]
    
    best_score = float('inf')
    best_params = {"brightness": 0.0, "width": 1.0, "bass_preservation": 1.0, "dynamics_preservation": 1.0}
    
    for b in brightness_grid:
        for w in width_grid:
            params = {
                "brightness": b,
                "width": w,
                "bass_preservation": 1.0, # Fixed for MVP search
                "dynamics_preservation": 1.0 # Fixed for MVP search
            }
            
            # Render candidate
            rendered = render_fn(audio, params)
            
            # Analyze
            analysis = analyzer.analyze(rendered, sr)
            
            # Score
            score = compute_objective(analysis, targets)
            
            if score < best_score:
                best_score = score
                best_params = params
                
    return best_params
