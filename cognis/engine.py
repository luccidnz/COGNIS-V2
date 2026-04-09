from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from cognis.analysis.analyzer import Analyzer
from cognis.analysis.features import AnalysisResult
from cognis.config import MasteringConfig
from cognis.dsp.dynamics import MultibandDynamics
from cognis.dsp.eq import EQ
from cognis.dsp.limiter import Limiter
from cognis.dsp.stereo import StereoControl
from cognis.optimizer.search import grid_search
from cognis.optimizer.targets import TargetValues, build_targets
from cognis.reports.qc import ReportResult, build_report


RECIPE_SCHEMA_VERSION = "recipe_v2"


@dataclass(frozen=True)
class RenderResult:
    audio: np.ndarray
    recipe: dict[str, Any]
    input_analysis: AnalysisResult
    output_analysis: AnalysisResult
    reference_analysis: AnalysisResult | None
    targets: TargetValues
    report: ReportResult

    @property
    def mastered_audio(self) -> np.ndarray:
        return self.audio


class Engine:
    def __init__(self):
        self.analyzer = Analyzer()

    def _render_chain(
        self,
        audio: np.ndarray,
        sr: int,
        params: dict[str, float],
        config: MasteringConfig,
        trim_gain_db: float,
        makeup_gain_db: float,
    ) -> np.ndarray:
        """Execute the DSP chain."""
        eq = EQ(sr)
        dynamics = MultibandDynamics(sr, backend=config.fir_backend)
        stereo = StereoControl(sr)
        limiter = Limiter(sr)

        audio = audio * (10 ** (trim_gain_db / 20.0))
        audio = eq.process(audio, params.get("brightness", 0.0))
        audio = dynamics.process(audio, params.get("dynamics_preservation", 1.0))
        audio = stereo.process(audio, params.get("width", 1.0), params.get("bass_preservation", 1.0))
        audio = audio * (10 ** (makeup_gain_db / 20.0))
        audio = limiter.process(
            audio,
            ceiling_db=config.ceiling_db,
            mode=config.ceiling_mode.value,
            oversampling=config.oversampling,
        )
        return audio

    def _compute_gain_staging(self, input_analysis: AnalysisResult, target_loudness: float) -> tuple[float, float]:
        input_tp = input_analysis.loudness.true_peak_dbfs
        input_lufs = input_analysis.loudness.integrated_lufs

        if input_lufs < -69.0:
            return 0.0, 0.0

        trim_gain_db = -6.0 - input_tp
        trimmed_lufs = input_lufs + trim_gain_db
        raw_makeup_gain = target_loudness - trimmed_lufs
        makeup_gain_db = float(np.clip(raw_makeup_gain, -24.0, 24.0))
        return trim_gain_db, makeup_gain_db

    def _build_recipe(
        self,
        config: MasteringConfig,
        best_params: dict[str, float],
        targets: TargetValues,
        trim_gain_db: float,
        makeup_gain_db: float,
    ) -> dict[str, Any]:
        config_dict = asdict(config)
        config_dict["mode"] = config.mode.value
        config_dict["ceiling_mode"] = config.ceiling_mode.value

        return {
            "schema_version": RECIPE_SCHEMA_VERSION,
            "config": config_dict,
            "params": dict(sorted(best_params.items())),
            "derived_targets": {
                "target_loudness": targets.target_loudness,
                "ceiling_db": targets.ceiling_db,
                "target_tilt": targets.target_tilt,
                "target_width": targets.target_width,
                "target_crest_factor": targets.target_crest_factor,
                "target_low_band_width": targets.target_low_band_width,
                "target_low_mid_balance": targets.target_low_mid_balance,
                "target_high_mid_balance": targets.target_high_mid_balance,
                "target_sub_energy_ratio": targets.target_sub_energy_ratio,
                "target_low_energy_ratio": targets.target_low_energy_ratio,
                "target_side_energy_ratio": targets.target_side_energy_ratio,
                "reference_targeting": asdict(targets.reference_targeting) if targets.reference_targeting else None,
            },
            "render_context": {
                "trim_gain_db": trim_gain_db,
                "makeup_gain_db": makeup_gain_db,
                "targets": {
                    "target_loudness": targets.target_loudness,
                    "ceiling_db": targets.ceiling_db,
                    "target_tilt": targets.target_tilt,
                    "target_width": targets.target_width,
                    "target_crest_factor": targets.target_crest_factor,
                    "target_low_band_width": targets.target_low_band_width,
                    "target_low_mid_balance": targets.target_low_mid_balance,
                    "target_high_mid_balance": targets.target_high_mid_balance,
                    "target_sub_energy_ratio": targets.target_sub_energy_ratio,
                    "target_low_energy_ratio": targets.target_low_energy_ratio,
                    "target_side_energy_ratio": targets.target_side_energy_ratio,
                    "reference_targeting": asdict(targets.reference_targeting) if targets.reference_targeting else None,
                },
            },
        }

    def render(
        self,
        audio: np.ndarray,
        sr: int,
        config: MasteringConfig,
        *,
        reference_audio: np.ndarray | None = None,
        reference_sr: int | None = None,
    ) -> RenderResult:
        """Run the canonical mastering flow and return all deterministic artifacts."""
        input_analysis = self.analyzer.analyze(audio, sr, role="input")
        reference_analysis = None
        if reference_audio is not None:
            if reference_sr is None:
                raise ValueError("reference_sr must be supplied when reference_audio is provided")
            reference_analysis = self.analyzer.analyze(
                reference_audio,
                reference_sr,
                role="reference",
                source_path=config.reference_path,
            )
        targets = build_targets(config, input_analysis=input_analysis, reference_analysis=reference_analysis)
        trim_gain_db, makeup_gain_db = self._compute_gain_staging(input_analysis, targets.target_loudness)

        def render_fn(aud: np.ndarray, params: dict[str, float]) -> np.ndarray:
            return self._render_chain(aud, sr, params, config, trim_gain_db, makeup_gain_db)

        best_params = grid_search(audio, sr, targets, render_fn, self.analyzer)
        mastered_audio = self._render_chain(audio, sr, best_params, config, trim_gain_db, makeup_gain_db)
        output_analysis = self.analyzer.analyze(mastered_audio, sr, role="output")
        recipe = self._build_recipe(config, best_params, targets, trim_gain_db, makeup_gain_db)
        report = build_report(
            config,
            recipe["schema_version"],
            targets,
            input_analysis,
            output_analysis,
            reference_analysis=reference_analysis,
        )

        return RenderResult(
            audio=mastered_audio,
            recipe=recipe,
            input_analysis=input_analysis,
            output_analysis=output_analysis,
            reference_analysis=reference_analysis,
            targets=targets,
            report=report,
        )

    def process(self, audio: np.ndarray, sr: int, config: MasteringConfig) -> tuple[np.ndarray, ReportResult, dict[str, Any]]:
        """Compatibility shim around :meth:`render` for existing callers."""
        result = self.render(audio, sr, config)
        return result.audio, result.report, result.recipe


RenderArtifacts = RenderResult
