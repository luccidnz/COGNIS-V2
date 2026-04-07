#include "cognis_dsp/fir_executor.hpp"
#include <stdexcept>
#include <vector>
#include <complex>
#include <cmath>
#include "pocketfft/pocketfft_hdronly.h"

namespace cognis_dsp {

// Utility function to get the next fast length for FFT
size_t next_fast_len(size_t target) {
    // PocketFFT is very flexible but prefers sizes with small prime factors.
    // However, it can handle any size efficiently. We can just use the target length.
    // For even better performance, we could pad to a 5-smooth number, but
    // simply returning target or padding slightly is fine for now as a baseline.
    // To match scipy.fft.next_fast_len perfectly, we'd need a 5-smooth algorithm.
    // PocketFFT's internal blue-stein handles non-smooth sizes, but let's pad to a
    // highly composite number if needed.

    // A simple 5-smooth generator is overkill here, pocketfft is fast for most sizes.
    // We'll implement a simple one to be safe:
    size_t n = target;
    while (true) {
        size_t m = n;
        while (m % 2 == 0) m /= 2;
        while (m % 3 == 0) m /= 3;
        while (m % 5 == 0) m /= 5;
        if (m == 1) break;
        n++;
    }
    return n;
}

std::vector<double> execute_fft(const FirExecutionRequest& request);
std::vector<double> execute_partitioned(const FirExecutionRequest& request);

std::vector<double> FirExecutor::execute(const FirExecutionRequest& request) {
    if (request.backend_mode == FirBackendMode::FFT) {
        return execute_fft(request);
    } else if (request.backend_mode == FirBackendMode::PARTITIONED) {
        return execute_partitioned(request);
    } else {
        throw std::runtime_error("Native FIR execution currently only supports 'FFT' and 'PARTITIONED' backends.");
    }
}

std::vector<double> execute_fft(const FirExecutionRequest& request) {

    size_t signal_len = request.samples;
    size_t kernel_len = request.num_taps;
    size_t channels = request.channels;

    if (signal_len == 0 || kernel_len == 0 || channels == 0) {
        return std::vector<double>(channels * signal_len, 0.0);
    }

    // mode="same" alignment

    size_t full_len = signal_len + kernel_len - 1;
    size_t shift = (kernel_len - 1) / 2;

    size_t n_fft = next_fast_len(full_len);

    std::vector<double> output(channels * signal_len, 0.0);

    // Prepare kernel FFT
    std::vector<double> kernel_padded(n_fft, 0.0);
    for (size_t i = 0; i < kernel_len; ++i) {
        kernel_padded[i] = request.taps_data[i];
    }

    std::vector<std::complex<double>> kernel_fft(n_fft / 2 + 1);
    pocketfft::shape_t shape_fft = {n_fft};
    pocketfft::stride_t stride_kernel_in = {sizeof(double)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<double>)};
    size_t axis = 0;

    pocketfft::r2c(shape_fft, stride_kernel_in, stride_out, axis, pocketfft::FORWARD,
                   kernel_padded.data(), kernel_fft.data(), 1.0, 0);

    // Process each channel
    std::vector<double> channel_padded(n_fft, 0.0);
    std::vector<std::complex<double>> channel_fft(n_fft / 2 + 1);
    std::vector<double> channel_out(n_fft, 0.0);

    for (size_t c = 0; c < channels; ++c) {
        // Zero pad the input channel
        std::fill(channel_padded.begin(), channel_padded.end(), 0.0);
        const double* channel_data = request.audio_data + c * signal_len;
        for (size_t i = 0; i < signal_len; ++i) {
            channel_padded[i] = channel_data[i];
        }

        // FFT of the channel
        pocketfft::r2c(shape_fft, stride_kernel_in, stride_out, axis, pocketfft::FORWARD,
                       channel_padded.data(), channel_fft.data(), 1.0, 0);

        // Multiply in frequency domain
        for (size_t i = 0; i < kernel_fft.size(); ++i) {
            channel_fft[i] *= kernel_fft[i];
        }

        // IFFT back to time domain
        // Normalize by 1/N for IFFT
        pocketfft::c2r(shape_fft, stride_out, stride_kernel_in, axis, pocketfft::BACKWARD,
                       channel_fft.data(), channel_out.data(), 1.0 / static_cast<double>(n_fft), 0);

        // Extract "same" mode segment
        double* out_ptr = output.data() + c * signal_len;
        for (size_t i = 0; i < signal_len; ++i) {
            out_ptr[i] = channel_out[i + shift];
        }
    }

    return output;
}

std::vector<double> execute_partitioned(const FirExecutionRequest& request) {
    size_t signal_len = request.samples;
    size_t kernel_len = request.num_taps;
    size_t channels = request.channels;

    if (signal_len == 0 || kernel_len == 0 || channels == 0) {
        return std::vector<double>(channels * signal_len, 0.0);
    }

    size_t partition_size = 4096;
    size_t n_fft = next_fast_len(partition_size + kernel_len - 1);
    size_t step = n_fft - kernel_len + 1;

    size_t shift = (kernel_len - 1) / 2;
    size_t pad_start = shift;
    size_t num_blocks = (signal_len + step - 1) / step;
    // Note: ensure we don't underflow. Python does:
    // pad_end = num_blocks * step + kernel_len - 1 - signal_len - pad_start
    size_t pad_end = num_blocks * step + kernel_len - 1 - signal_len - pad_start;

    size_t padded_signal_len = signal_len + pad_start + pad_end;
    std::vector<double> output(channels * signal_len, 0.0);

    std::vector<double> kernel_padded(n_fft, 0.0);
    for (size_t i = 0; i < kernel_len; ++i) {
        kernel_padded[i] = request.taps_data[i];
    }

    std::vector<std::complex<double>> kernel_fft(n_fft / 2 + 1);
    pocketfft::shape_t shape_fft = {n_fft};
    pocketfft::stride_t stride_in = {sizeof(double)};
    pocketfft::stride_t stride_out = {sizeof(std::complex<double>)};
    size_t axis = 0;

    pocketfft::r2c(shape_fft, stride_in, stride_out, axis, pocketfft::FORWARD,
                   kernel_padded.data(), kernel_fft.data(), 1.0, 0);

    std::vector<double> block_padded(n_fft, 0.0);
    std::vector<std::complex<double>> block_fft(n_fft / 2 + 1);
    std::vector<double> block_out(n_fft, 0.0);

    for (size_t c = 0; c < channels; ++c) {
        const double* channel_data = request.audio_data + c * signal_len;
        double* out_ptr = output.data() + c * signal_len;

        for (size_t b = 0; b < num_blocks; ++b) {
            size_t start_idx = b * step;

            std::fill(block_padded.begin(), block_padded.end(), 0.0);
            for (size_t i = 0; i < n_fft; ++i) {
                // Determine actual signal index by subtracting pad_start
                long long true_idx = static_cast<long long>(start_idx + i) - static_cast<long long>(pad_start);
                if (true_idx >= 0 && true_idx < static_cast<long long>(signal_len)) {
                    block_padded[i] = channel_data[true_idx];
                }
            }

            pocketfft::r2c(shape_fft, stride_in, stride_out, axis, pocketfft::FORWARD,
                           block_padded.data(), block_fft.data(), 1.0, 0);

            for (size_t i = 0; i < kernel_fft.size(); ++i) {
                block_fft[i] *= kernel_fft[i];
            }

            pocketfft::c2r(shape_fft, stride_out, stride_in, axis, pocketfft::BACKWARD,
                           block_fft.data(), block_out.data(), 1.0 / static_cast<double>(n_fft), 0);

            // Copy valid part (discard first kernel_len - 1 samples)
            size_t valid_start = kernel_len - 1;
            for (size_t i = 0; i < step; ++i) {
                size_t out_idx = b * step + i;
                if (out_idx < signal_len) {
                    out_ptr[out_idx] = block_out[valid_start + i];
                }
            }
        }
    }

    return output;
}

} // namespace cognis_dsp