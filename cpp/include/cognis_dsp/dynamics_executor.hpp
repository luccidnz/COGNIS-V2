#pragma once

#include <vector>
#include <cstddef>
#include <tuple>

namespace cognis_dsp {

struct DynamicsExecutionRequest {
    const double* sidechain_data;
    size_t samples;

    double attack_coef;
    double release_coef;
    double threshold_db;
    double ratio;
    double initial_env;
};

class DynamicsExecutor {
public:
    // Returns a tuple of the 1D gain array and the final envelope state.
    std::tuple<std::vector<double>, double> compute_gain(const DynamicsExecutionRequest& request);
};

} // namespace cognis_dsp
