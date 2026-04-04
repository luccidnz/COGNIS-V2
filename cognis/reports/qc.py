from dataclasses import dataclass
from cognis.analysis.features import AnalysisResult

@dataclass
class QCReport:
    integrated_loudness: float
    short_term_loudness: float
    sample_peak: float
    true_peak: float
    spectral_tilt: float
    phase_correlation: float
    
def generate_qc_report(analysis: AnalysisResult) -> QCReport:
    """Generate a QC summary from analysis results."""
    return QCReport(
        integrated_loudness=analysis.loudness.integrated_loudness,
        short_term_loudness=analysis.loudness.short_term_loudness,
        sample_peak=analysis.loudness.sample_peak,
        true_peak=analysis.loudness.true_peak,
        spectral_tilt=analysis.spectrum.spectral_tilt,
        phase_correlation=analysis.stereo.phase_correlation
    )
