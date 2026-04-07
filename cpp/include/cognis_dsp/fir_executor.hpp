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

enum class FirBackendMode {
    UNKNOWN = 0,
    DIRECT = 1,
    FFT = 2,
    PARTITIONED = 3
};

struct FirExecutionRequest {
    const double* audio_data;
    size_t channels;
    size_t samples;

    const double* taps_data;
    size_t num_taps;

    FirBackendMode backend_mode; // e.g. 1=DIRECT, 2=FFT, 3=PARTITIONED
};

class FirExecutor {
public:
    // Future: implement dispatch to partitioned convolution or other algorithms.
    // Returns flattened channel-first output array.
    std::vector<double> execute(const FirExecutionRequest& request);
};

} // namespace cognis_dsp
