#pragma once

#include <vector>
#include <string>
#include <cstddef>

namespace cognis_dsp {

// Explicit execution boundary for the FIR executor.
// Mirrors the contract in Python: execute_fir_2d(audio_2d, taps, backend)
//
// Expectations:
// - audio is assumed to be channel-first [channels, samples]
// - contiguous memory layout in row-major (C) order is expected
// - padding/overlap logic must preserve strict mode="same" alignment
// - return value must exactly match Python's scipy.signal.convolve(mode="same") behavior

struct FirExecutionRequest {
    const double* audio_data;
    size_t channels;
    size_t samples;

    const double* taps_data;
    size_t num_taps;

    std::string backend_mode; // e.g. "auto", "partitioned", "fft", "direct"
};

class FirExecutor {
public:
    // Future: implement dispatch to partitioned convolution or other algorithms.
    // Returns flattened channel-first output array.
    std::vector<double> execute(const FirExecutionRequest& request);
};

} // namespace cognis_dsp
