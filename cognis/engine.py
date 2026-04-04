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
        
    def _render_chain(self, audio: np.ndarray, sr: int, params: Dict[str, float], config: MasteringConfig, trim_gain_db: float, makeup_gain_db: float) -> np.ndarray:
        """Execute the DSP chain."""
        # Instantiate DSP modules
        eq = EQ(sr)
        dynamics = MultibandDynamics(sr)
        stereo = StereoControl(sr)
        limiter = Limiter(sr)
        
        # 1. Input trim (normalization for headroom)
        audio = audio * (10 ** (trim_gain_db / 20.0))
        
        # 2. Cleanup (skip for MVP)
        
        # 3. Corrective EQ
        # 4. Bass protection / mono sub handling
        # 5. Broad tonal shaping
        audio = eq.process(audio, params.get("brightness", 0.0))
        
        # 6. Dynamics
        audio = dynamics.process(audio, params.get("dynamics_preservation", 1.0))
        
        # 7. Stereo field control
        audio = stereo.process(audio, params.get("width", 1.0), params.get("bass_preservation", 1.0))
        
        # 8. Loudness staging
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
        
        # Calculate required gains
        # We want to trim the input to -6dBFS true peak to give headroom for EQ/Dynamics
        input_tp = input_analysis.loudness.true_peak
        input_lufs = input_analysis.loudness.integrated_loudness
        
        # If input is digital silence, avoid math errors
        if input_lufs < -69.0:
            trim_gain_db = 0.0
            makeup_gain_db = 0.0
        else:
            trim_gain_db = -6.0 - input_tp
            trimmed_lufs = input_lufs + trim_gain_db
            
            # Assumption: EQ and Dynamics will not drastically alter the overall LUFS.
            # We compute a static makeup gain here to hit the target loudness before the limiter.
            # In Phase 2, this could be a dynamic parameter searched by the optimizer, 
            # or computed dynamically after the EQ/Dynamics stages.
            makeup_gain_db = config.target_loudness - trimmed_lufs
            
        # 2. Build targets
        targets = build_targets(config)
        
        # 3. Optimize parameters
        # Define a render function for the optimizer
        def render_fn(aud, params):
            return self._render_chain(aud, sr, params, config, trim_gain_db, makeup_gain_db)
            
        best_params = grid_search(audio, sr, targets, render_fn, self.analyzer)
        
        # 4. Render final
        mastered_audio = self._render_chain(audio, sr, best_params, config, trim_gain_db, makeup_gain_db)
        
        # 5. QC re-measure
        output_analysis = self.analyzer.analyze(mastered_audio, sr)
        qc_report = generate_qc_report(output_analysis)
        
        # 6. Recipe
        from dataclasses import asdict
        config_dict = asdict(config)
        config_dict['mode'] = config_dict['mode'].value
        config_dict['ceiling_mode'] = config_dict['ceiling_mode'].value
        
        recipe = {
            "config": config_dict,
            "params": best_params,
            "schema_version": "recipe_v1"
        }
        
        return mastered_audio, qc_report, recipe
