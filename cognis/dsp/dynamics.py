import numpy as np

from cognis.dsp.filters import FirBackend, get_linear_phase_three_band_splitter


class MultibandDynamics:
    LOW_CROSSOVER_HZ = 250.0
    HIGH_CROSSOVER_HZ = 4000.0

    # These FIR lengths are long enough to keep the low crossover credible,
    # but still bounded so repeated optimizer renders stay practical in Python.
    LOW_CROSSOVER_TAPS = 1537
    HIGH_CROSSOVER_TAPS = 513

    def __init__(self, sr: int, backend: str = "AUTO"):
        self.sr = sr
        self.backend = FirBackend(backend.lower())
        self._splitter = get_linear_phase_three_band_splitter(
            sr,
            self.LOW_CROSSOVER_HZ,
            self.HIGH_CROSSOVER_HZ,
            low_taps=self.LOW_CROSSOVER_TAPS,
            high_taps=self.HIGH_CROSSOVER_TAPS,
        )

    def _compress_band(
        self,
        band: np.ndarray,
        threshold_db: float,
        ratio: float,
        attack_ms: float,
        release_ms: float,
    ) -> np.ndarray:
        """
        Apply a simple feed-forward compressor driven by a mono-linked sidechain.

        Attack and release are intentionally conservative for mastering. Low
        band timing stays slower to avoid bass pumping, while the high band is
        allowed to react faster to transient detail.
        """
        if ratio <= 1.0:
            return band

        attack_seconds = max(attack_ms, 0.1) / 1000.0
        release_seconds = max(release_ms, 0.1) / 1000.0
        attack_coef = np.exp(-1.0 / (self.sr * attack_seconds))
        release_coef = np.exp(-1.0 / (self.sr * release_seconds))

        gain = np.ones_like(band, dtype=np.float64)
        sidechain = np.abs(np.mean(band, axis=0)) if band.ndim > 1 else np.abs(band)

        curr_env = 0.0
        # TODO(optimization): This explicit sample-by-sample Python loop is the primary
        # bottleneck in the mastering render loop. It should be moved to a C++ extension
        # (similar to the FIR path) to drastically speed up recursive envelope tracking.
        for sample_index, sample in enumerate(sidechain):
            if sample > curr_env:
                curr_env = attack_coef * curr_env + (1.0 - attack_coef) * sample
            else:
                curr_env = release_coef * curr_env + (1.0 - release_coef) * sample

            env_db = 20.0 * np.log10(curr_env + 1e-10)
            if env_db > threshold_db:
                overshoot = env_db - threshold_db
                reduction_db = overshoot * (1.0 - 1.0 / ratio)
                gain[:, sample_index] = 10.0 ** (-reduction_db / 20.0)

        return band * gain

    def process(self, audio: np.ndarray, dynamics_preservation: float) -> np.ndarray:
        """
        Apply conservative multiband compression for offline mastering.

        The FIR crossover remains offline-oriented: it prioritizes transparent
        band separation and deterministic reuse over low-latency operation.
        `dynamics_preservation` maps from 0.0 (stronger control) to 1.0
        (effectively bypass), and the threshold curve stays gentle because
        this block feeds the final limiter rather than replacing it.
        """
        if dynamics_preservation >= 0.99:
            return audio

        bands = self._splitter.split(audio, backend=self.backend)

        compression_amount = 1.0 - float(np.clip(dynamics_preservation, 0.0, 1.0))
        ratio = 1.0 + 2.5 * compression_amount
        threshold_db = -18.0 * compression_amount

        comp_lows = self._compress_band(bands.low, threshold_db, ratio, 40.0, 200.0)
        comp_mids = self._compress_band(bands.mid, threshold_db, ratio, 15.0, 120.0)
        comp_highs = self._compress_band(bands.high, threshold_db, ratio, 6.0, 60.0)

        return comp_lows + comp_mids + comp_highs
