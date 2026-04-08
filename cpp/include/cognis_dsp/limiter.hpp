#pragma once

#include <vector>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <complex>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include "pocketfft/pocketfft_hdronly.h"

namespace cognis_dsp {

// Next power of 2 or smooth number for fast FFT
inline size_t get_fast_fft_size(size_t n) {
    // A simple next power of 2 for maximum speed, though pocketfft is fine with smooth numbers.
    size_t pow2 = 1;
    while (pow2 < n) pow2 *= 2;
    return pow2;
}


namespace py = pybind11;

inline std::vector<double> get_gaussian_weights(double sigma, double truncate = 4.0) {
    int lw = static_cast<int>(truncate * sigma + 0.5);
    std::vector<double> weights(2 * lw + 1);
    double sum = 0.0;
    for (int i = -lw; i <= lw; ++i) {
        double val = std::exp(-0.5 * (i / sigma) * (i / sigma));
        weights[i + lw] = val;
        sum += val;
    }
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] /= sum;
    }
    return weights;
}

// Scipy reflect boundary indexing helper
// Handles boundary conditions for reflecting: abc|cba (index -1 -> 0, -2 -> 1, N -> N-1, N+1 -> N-2)
inline int reflect_index(int index, int size) {
    if (size == 1) return 0;
    if (index < 0) {
        int rem = (-index) % (2 * size);
        if (rem == 0) return 0;
        if (rem <= size) return rem - 1;
        return 2 * size - rem;
    } else if (index >= size) {
        int rem = index % (2 * size);
        if (rem < size) return rem;
        return 2 * size - 1 - rem;
    }
    return index;
}

void apply_gaussian_filter1d(const double* input, double* output, int size, double sigma) {
    if (sigma <= 0.0) {
        for (int i = 0; i < size; ++i) {
            output[i] = input[i];
        }
        return;
    }

    std::vector<double> weights = get_gaussian_weights(sigma);
    int num_weights = weights.size();
    int lw = num_weights / 2;

    // We pad the input array with `lw` on both sides using reflect mode.
    // Padded size: size + 2 * lw
    size_t padded_size = size + 2 * lw;

    // Convolution size: padded_size + num_weights - 1
    size_t conv_size = padded_size + num_weights - 1;
    size_t n_fft = get_fast_fft_size(conv_size);

    std::vector<double> in_padded(n_fft, 0.0);
    for (int i = 0; i < padded_size; ++i) {
        int original_idx = i - lw;
        int r_idx = reflect_index(original_idx, size);
        in_padded[i] = input[r_idx];
    }

    std::vector<double> weights_padded(n_fft, 0.0);
    for (int i = 0; i < num_weights; ++i) {
        weights_padded[i] = weights[i];
    }

    // FFT
    pocketfft::shape_t shape_fft = {n_fft};
    pocketfft::stride_t stride_in = {sizeof(double)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<double>)};
    size_t axis = 0;

    std::vector<std::complex<double>> in_c(n_fft / 2 + 1);
    std::vector<std::complex<double>> weights_c(n_fft / 2 + 1);

    pocketfft::r2c(shape_fft, stride_in, stride_out, axis, pocketfft::FORWARD,
                   in_padded.data(), in_c.data(), 1.0);

    pocketfft::r2c(shape_fft, stride_in, stride_out, axis, pocketfft::FORWARD,
                   weights_padded.data(), weights_c.data(), 1.0);

    // Multiply
    for (size_t i = 0; i < in_c.size(); ++i) {
        in_c[i] *= weights_c[i];
    }

    // IFFT
    std::vector<double> result_padded(n_fft);
    pocketfft::c2r(shape_fft, stride_out, stride_in, axis, pocketfft::BACKWARD,
                   in_c.data(), result_padded.data(), 1.0 / n_fft);

    // Extract valid region.
    // The valid region of a full convolution between length N and length M is M-1 to N-1.
    // Our padded input is `padded_size` (N), and weights is `num_weights` (M).
    // The valid region starts at `num_weights - 1`.
    for (int i = 0; i < size; ++i) {
        output[i] = result_padded[i + num_weights - 1];
    }
}

void apply_minimum_filter1d(const double* input, double* output, int size, int hold_samples) {
    if (hold_samples <= 1) {
        for (int i = 0; i < size; ++i) {
            output[i] = input[i];
        }
        return;
    }

    int offset = hold_samples / 2;
    int right_lw = hold_samples - 1 - offset;
    int left_lw = offset;

    // Boundaries (left)
    for (int i = 0; i < std::min(left_lw, size); ++i) {
        double min_val = std::numeric_limits<double>::max();
        for (int j = 0; j < hold_samples; ++j) {
            int idx = reflect_index(i + j - offset, size);
            if (input[idx] < min_val) {
                min_val = input[idx];
            }
        }
        output[i] = min_val;
    }

    // Main loop (no boundary checks)
    for (int i = left_lw; i < size - right_lw; ++i) {
        double min_val = input[i - offset];
        for (int j = 1; j < hold_samples; ++j) {
            if (input[i + j - offset] < min_val) {
                min_val = input[i + j - offset];
            }
        }
        output[i] = min_val;
    }

    // Boundaries (right)
    for (int i = std::max(left_lw, size - right_lw); i < size; ++i) {
        double min_val = std::numeric_limits<double>::max();
        for (int j = 0; j < hold_samples; ++j) {
            int idx = reflect_index(i + j - offset, size);
            if (input[idx] < min_val) {
                min_val = input[idx];
            }
        }
        output[i] = min_val;
    }
}

// compute_native_limiter_gain_gaussian_only
py::array_t<double> compute_native_limiter_gain_gaussian_only(
    py::array_t<double, py::array::c_style | py::array::forcecast> raw_gain,
    double sigma_samples
) {
    py::buffer_info buf = raw_gain.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("raw_gain must be a 1D array");
    }

    int size = buf.shape[0];
    auto result = py::array_t<double>(size);

    const double* input_ptr = static_cast<const double*>(buf.ptr);
    double* output_ptr = static_cast<double*>(result.request().ptr);

    apply_gaussian_filter1d(input_ptr, output_ptr, size, sigma_samples);

    return result;
}

// compute_native_limiter_gain_fused
py::array_t<double> compute_native_limiter_gain_fused(
    py::array_t<double, py::array::c_style | py::array::forcecast> raw_gain,
    int hold_samples,
    double sigma_samples
) {
    py::buffer_info buf = raw_gain.request();
    if (buf.ndim != 1) {
        throw std::runtime_error("raw_gain must be a 1D array");
    }

    int size = buf.shape[0];
    auto result = py::array_t<double>(size);

    const double* input_ptr = static_cast<const double*>(buf.ptr);
    double* output_ptr = static_cast<double*>(result.request().ptr);

    if (hold_samples > 1) {
        std::vector<double> held_gain(size);
        apply_minimum_filter1d(input_ptr, held_gain.data(), size, hold_samples);
        apply_gaussian_filter1d(held_gain.data(), output_ptr, size, sigma_samples);
    } else {
        apply_gaussian_filter1d(input_ptr, output_ptr, size, sigma_samples);
    }

    return result;
}

} // namespace cognis_dsp
