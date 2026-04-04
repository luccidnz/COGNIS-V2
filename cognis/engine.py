import numpy as np
from typing import Tuple, Dict, Any
from cognis.config import MasteringConfig
from cognis.analysis.analyzer import Analyzer
from cognis.optimizer.targets import build_targets
from cognis.optimizer.search import grid_search
from cognis.dsp.eq import EQ
from cognis.dsp.dynamics import MultibandDynamics
from cognis.dsp.stereo import StereoControl
from cognis.dsp.limiter import Limiter
from cognis.reports.qc import generate_qc_report, QCReport

class Engine:
    def __init__(self):
        self.analyzer = Analyzer()
        
    def _render_chain(self, audio: np.ndarray, sr: int, params: Dict[str, float], config: MasteringConfig) -> np.ndarray:
        """Execute the DSP chain."""
        # Instantiate DSP modules
        eq = EQ(sr)
        dynamics = MultibandDynamics(sr)
        stereo = StereoControl(sr)
        limiter = Limiter(sr)
        
        # 1. Input trim (simple normalization for headroom)
        peak = np.max(np.abs(audio))
        if peak > 0:
            audio = audio * (0.5 / peak) # -6dBFS headroom
            
        # 2. Cleanup (skip for MVP)
        
        # 3. Corrective EQ
        # 4. Bass protection / mono sub handling
        # 5. Broad tonal shaping
        audio = eq.process(audio, params.get("brightness", 0.0))
        
        # 6. Dynamics
        audio = dynamics.process(audio, params.get("dynamics_preservation", 1.0))
        
        # 7. Stereo field control
        audio = stereo.process(audio, params.get("width", 1.0), params.get("bass_preservation", 1.0))
        
        # 8. Loudness staging (simple gain to hit target before limiter)
        # For MVP, we just apply a static gain based on config
        # A real implementation would measure here and apply makeup gain
        makeup_gain_db = 6.0 # Arbitrary makeup for MVP
        audio = audio * (10 ** (makeup_gain_db / 20.0))
        
        # 9. True-peak limiter
        audio = limiter.process(
            audio, 
            ceiling_db=config.ceiling_db, 
            mode=config.ceiling_mode.value, 
            oversampling=config.oversampling
        )
        
        return audio

    def process(self, audio: np.ndarray, sr: int, config: MasteringConfig) -> Tuple[np.ndarray, QCReport, Dict[str, Any]]:
        """
        Orchestrate the mastering process.
        Returns (mastered_audio, qc_report, recipe)
        """
        # 1. Analyze input
        input_analysis = self.analyzer.analyze(audio, sr)
        
        # 2. Build targets
        targets = build_targets(config)
        
        # 3. Optimize parameters
        # Define a render function for the optimizer
        def render_fn(aud, params):
            return self._render_chain(aud, sr, params, config)
            
        best_params = grid_search(audio, sr, targets, render_fn, self.analyzer)
        
        # 4. Render final
        mastered_audio = self._render_chain(audio, sr, best_params, config)
        
        # 5. QC re-measure
        output_analysis = self.analyzer.analyze(mastered_audio, sr)
        qc_report = generate_qc_report(output_analysis)
        
        # 6. Recipe
        recipe = {
            "config": config.__dict__,
            "params": best_params,
            "schema_version": "recipe_v1"
        }
        
        return mastered_audio, qc_report, recipe
