#include "cognis_dsp/fir_executor.hpp"
#include <stdexcept>

namespace cognis_dsp {

std::vector<double> FirExecutor::execute(const FirExecutionRequest& request) {
    // TODO: Implement the native FIR execution algorithms here.
    // - "partitioned": Overlap-save block convolution using a native FFT library (e.g. FFTW or PFFFT).
    // - "direct": Standard time-domain convolution for very small kernels.
    // - "fft": Single monolithic block convolution.

    throw std::runtime_error("Native FIR execution is not yet implemented.");
}

} // namespace cognis_dsp
