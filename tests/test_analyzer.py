import numpy as np

from cognis.analysis.analyzer import Analyzer


def _stereo_tone(duration_s: float = 1.0, sr: int = 48000) -> np.ndarray:
    t = np.linspace(0, duration_s, int(duration_s * sr), endpoint=False)
    left = 0.5 * np.sin(2 * np.pi * 55 * t) + 0.2 * np.sin(2 * np.pi * 3200 * t)
    right = 0.45 * np.sin(2 * np.pi * 55 * t + 0.1) + 0.15 * np.sin(2 * np.pi * 4800 * t)
    return np.vstack((left, right)).astype(np.float64)


def test_analyzer_accepts_stereo_float() -> None:
    analyzer = Analyzer()
    result = analyzer.analyze(_stereo_tone().astype(np.float32), 48000)
    assert result.channels == 2
    assert result.sample_rate == 48000


def test_analyzer_surfaces_v1_schema_metrics() -> None:
    analyzer = Analyzer()
    result = analyzer.analyze(_stereo_tone(duration_s=4.0), 48000)

    assert result.schema_version == "analysis_schema_v1"
    assert np.isfinite(result.loudness.integrated_loudness)
    assert np.isfinite(result.loudness.loudness_range_lu)
    assert np.isfinite(result.tonal.sub_energy_ratio)
    assert np.isfinite(result.tonal.low_band_centroid_hz)
    assert np.isfinite(result.stereo.side_energy_ratio)
    assert np.isfinite(result.stereo.mono_null_ratio_db)
    assert np.isfinite(result.risks.limiter_stress_estimate)
    assert np.isfinite(result.risks.codec_risk_estimate)


def test_analysis_json_is_deterministic_for_same_input() -> None:
    analyzer = Analyzer()
    audio = _stereo_tone(duration_s=3.5)
    first = analyzer.analyze(audio, 48000).to_dict()
    second = analyzer.analyze(audio, 48000).to_dict()
    assert first == second
