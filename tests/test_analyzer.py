import numpy as np
from cognis.analysis.analyzer import Analyzer

def test_analyzer_accepts_stereo_float():
    analyzer = Analyzer()
    audio = np.random.randn(2, 48000).astype(np.float32)
    result = analyzer.analyze(audio, 48000)
    assert result.channels == 2
    assert result.sample_rate == 48000

def test_integrated_loudness_finite():
    analyzer = Analyzer()
    # 1 second sine wave
    t = np.linspace(0, 1, 48000, endpoint=False)
    audio = np.vstack((np.sin(2 * np.pi * 440 * t), np.sin(2 * np.pi * 440 * t))) * 0.5
    result = analyzer.analyze(audio, 48000)
    assert np.isfinite(result.loudness.integrated_loudness)

def test_true_peak_finite():
    analyzer = Analyzer()
    audio = np.random.randn(2, 48000) * 0.1
    result = analyzer.analyze(audio, 48000)
    assert np.isfinite(result.loudness.true_peak)

def test_schema_version_exists():
    analyzer = Analyzer()
    audio = np.zeros((2, 1000))
    result = analyzer.analyze(audio, 48000)
    assert hasattr(result, 'schema_version')
    assert isinstance(result.schema_version, str)
