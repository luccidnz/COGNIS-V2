import numpy as np

from cognis.dsp.filters import FirBackend, get_linear_phase_three_band_splitter

try:
    import cognis_native
    _NATIVE_AVAILABLE = hasattr(cognis_native, "compute_native_compressor_gain")
except ImportError:
    try:
        from cognis.dsp import cognis_native
        _NATIVE_AVAILABLE = hasattr(cognis_native, "compute_native_compressor_gain")
    except ImportError:
        cognis_native = None
        _NATIVE_AVAILABLE = False

_FALLBACK_ON_NATIVE_FAILURE = False


def get_dynamics_execution_info() -> dict:
    return {
        "native_available": _NATIVE_AVAILABLE,
        "module_imported": cognis_native is not None,
        "fallback_on_native_failure": _FALLBACK_ON_NATIVE_FAILURE,
    }


def _compute_gain_python(
    sidechain: np.ndarray,
    attack_coef: float,
    release_coef: float,
    threshold_db: float,
    ratio: float,
    initial_env: float,
) -> tuple[np.ndarray, float]:
    """
    Compute the per-sample gain and updated envelope state using pure Python.
    This explicit sample-by-sample loop is the primary bottleneck in the dynamics path.
    """
    gain = np.ones_like(sidechain, dtype=np.float64)
    curr_env = initial_env

    for sample_index, sample in enumerate(sidechain):
        if sample > curr_env:
            curr_env = attack_coef * curr_env + (1.0 - attack_coef) * sample
        else:
            curr_env = release_coef * curr_env + (1.0 - release_coef) * sample

        env_db = 20.0 * np.log10(curr_env + 1e-10)
        if env_db > threshold_db:
            overshoot = env_db - threshold_db
            reduction_db = overshoot * (1.0 - 1.0 / ratio)
            gain[sample_index] = 10.0 ** (-reduction_db / 20.0)

    return gain, curr_env


class MultibandDynamics:
    LOW_CROSSOVER_HZ = 250.0
    HIGH_CROSSOVER_HZ = 4000.0

    # Test observability attribute
    last_execution_info = None

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
    ) -> tuple[np.ndarray, dict[str, object]]:
        """
        Apply a simple feed-forward compressor driven by a mono-linked sidechain.

        Attack and release are intentionally conservative for mastering. Low
        band timing stays slower to avoid bass pumping, while the high band is
        allowed to react faster to transient detail.
        """
        if ratio <= 1.0:
            return band, {
                "used_native": False,
                "fallback_triggered": False,
                "execution_state": "python_reference_bypass",
            }

        attack_seconds = max(attack_ms, 0.1) / 1000.0
        release_seconds = max(release_ms, 0.1) / 1000.0
        attack_coef = np.exp(-1.0 / (self.sr * attack_seconds))
        release_coef = np.exp(-1.0 / (self.sr * release_seconds))

        sidechain = np.abs(np.mean(band, axis=0)) if band.ndim > 1 else np.abs(band)

        used_native = False
        fallback_triggered = False

        if _NATIVE_AVAILABLE:
            try:
                # C-contiguous enforce boundary
                sidechain_c = np.ascontiguousarray(sidechain, dtype=np.float64)

                # We could support initial_env passing from state in the future,
                # currently keeping it at 0.0 to match original python implementation
                gain_1d, final_env = cognis_native.compute_native_compressor_gain(
                    sidechain_c, float(attack_coef), float(release_coef), float(threshold_db), float(ratio), 0.0
                )
                used_native = True
            except Exception as e:
                if _FALLBACK_ON_NATIVE_FAILURE:
                    fallback_triggered = True
                    gain_1d, _ = _compute_gain_python(
                        sidechain, attack_coef, release_coef, threshold_db, ratio, 0.0
                    )
                else:
                    raise RuntimeError(f"Native dynamics execution failed: {e}") from e
        else:
            gain_1d, _ = _compute_gain_python(
                sidechain, attack_coef, release_coef, threshold_db, ratio, 0.0
            )

        execution_info = {
            "native_available": _NATIVE_AVAILABLE,
            "module_imported": cognis_native is not None,
            "used_native": used_native,
            "fallback_triggered": fallback_triggered,
            "execution_state": (
                "native_imported_and_used"
                if used_native
                else "python_fallback_after_native_failure"
                if fallback_triggered
                else "python_reference_native_unavailable"
            ),
        }

        return band * gain_1d, execution_info

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

        comp_lows, low_info = self._compress_band(bands.low, threshold_db, ratio, 40.0, 200.0)
        comp_mids, mid_info = self._compress_band(bands.mid, threshold_db, ratio, 15.0, 120.0)
        comp_highs, high_info = self._compress_band(bands.high, threshold_db, ratio, 6.0, 60.0)

        band_infos = [low_info, mid_info, high_info]
        self.last_execution_info = {
            "native_available": _NATIVE_AVAILABLE,
            "module_imported": cognis_native is not None,
            "used_native": any(bool(info["used_native"]) for info in band_infos),
            "fallback_triggered": any(bool(info["fallback_triggered"]) for info in band_infos),
            "execution_state": (
                "native_imported_and_used"
                if any(bool(info["used_native"]) for info in band_infos)
                else "python_fallback_after_native_failure"
                if any(bool(info["fallback_triggered"]) for info in band_infos)
                else "python_reference_native_unavailable"
            ),
            "bands": band_infos,
        }

        return comp_lows + comp_mids + comp_highs
