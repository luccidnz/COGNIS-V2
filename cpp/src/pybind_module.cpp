#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <stdexcept>
#include "cognis_dsp/fir_executor.hpp"

namespace py = pybind11;

py::array_t<double> execute_native_fir_2d(py::array_t<double> audio, py::array_t<double> taps, int backend_id) {
    cognis_dsp::FirBackendMode mode = cognis_dsp::FirBackendMode::UNKNOWN;
    if (backend_id == 1) mode = cognis_dsp::FirBackendMode::DIRECT;
    else if (backend_id == 2) mode = cognis_dsp::FirBackendMode::FFT;
    else if (backend_id == 3) mode = cognis_dsp::FirBackendMode::PARTITIONED;

    if (mode != cognis_dsp::FirBackendMode::FFT && mode != cognis_dsp::FirBackendMode::PARTITIONED) {
        throw std::runtime_error("Native FIR execution currently only supports 'FFT' and 'PARTITIONED' backends.");
    }

    // Validate type and C-contiguous layout
    py::buffer_info audio_info = audio.request();
    py::buffer_info taps_info = taps.request();

    if (audio_info.format != py::format_descriptor<double>::format() ||
        taps_info.format != py::format_descriptor<double>::format()) {
        throw std::invalid_argument("Inputs must be float64");
    }

    if (audio_info.ndim != 2) {
        throw std::invalid_argument("audio must be 2-dimensional");
    }

    if (taps_info.ndim != 1) {
        throw std::invalid_argument("taps must be 1-dimensional");
    }

    // Pybind11's array_t with default flags will not enforce contiguous if we just take it as py::array_t<double>.
    // It's safer to explicitly check the flags.
    // However, the Python boundary in fir_executor.py enforces np.ascontiguousarray.
    // Let's do a fast check:
    if (!(audio.flags() & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_) ||
        !(taps.flags() & pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
        throw std::invalid_argument("Inputs must be C-contiguous");
    }

    size_t channels = audio_info.shape[0];
    size_t samples = audio_info.shape[1];
    size_t num_taps = taps_info.shape[0];

    cognis_dsp::FirExecutionRequest req;
    req.audio_data = static_cast<double*>(audio_info.ptr);
    req.channels = channels;
    req.samples = samples;
    req.taps_data = static_cast<double*>(taps_info.ptr);
    req.num_taps = num_taps;
    req.backend_mode = mode;

    cognis_dsp::FirExecutor executor;
    std::vector<double> output_data = executor.execute(req);

    // Create a new NumPy array to hold the result
    auto result = py::array_t<double>({channels, samples});
    py::buffer_info result_info = result.request();
    double* result_ptr = static_cast<double*>(result_info.ptr);

    // Copy data from the vector to the numpy array
    // (We could avoid the copy by allocating the numpy array first and passing its pointer to executor,
    // but the vector is fine for the MVP Native pass)
    std::copy(output_data.begin(), output_data.end(), result_ptr);

    return result;
}

PYBIND11_MODULE(cognis_native, m) {
    m.doc() = "COGNIS Native DSP Module";
    m.def("execute_native_fir_2d", &execute_native_fir_2d, "Apply native FIR filter to audio");
}