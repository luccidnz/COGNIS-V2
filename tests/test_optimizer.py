import numpy as np
from cognis.optimizer.search import grid_search
from cognis.optimizer.targets import TargetValues
from cognis.analysis.analyzer import Analyzer

def test_optimizer_returns_params():
    analyzer = Analyzer()
    targets = TargetValues(
        target_loudness=-14.0,
        ceiling_db=-1.0,
        target_tilt=0.0,
        target_width=1.0
    )
    
    audio = np.random.randn(2, 4800).astype(np.float32) * 0.1
    
    def dummy_render(aud, params):
        # Just apply a gain based on brightness to simulate change
        gain = 1.0 + params.get("brightness", 0.0)
        return aud * gain
        
    best_params = grid_search(audio, 48000, targets, dummy_render, analyzer)
    
    assert isinstance(best_params, dict)
    assert "brightness" in best_params
    assert "width" in best_params
    assert "bass_preservation" in best_params
    assert "dynamics_preservation" in best_params
