#include "cognis_dsp/dynamics_executor.hpp"
#include <cmath>

namespace cognis_dsp {

std::tuple<std::vector<double>, double> DynamicsExecutor::compute_gain(const DynamicsExecutionRequest& request) {
    std::vector<double> gain(request.samples, 1.0);
    double curr_env = request.initial_env;

    for (size_t i = 0; i < request.samples; ++i) {
        double sample = request.sidechain_data[i];

        if (sample > curr_env) {
            curr_env = request.attack_coef * curr_env + (1.0 - request.attack_coef) * sample;
        } else {
            curr_env = request.release_coef * curr_env + (1.0 - request.release_coef) * sample;
        }

        double env_db = 20.0 * std::log10(curr_env + 1e-10);
        if (env_db > request.threshold_db) {
            double overshoot = env_db - request.threshold_db;
            double reduction_db = overshoot * (1.0 - 1.0 / request.ratio);
            gain[i] = std::pow(10.0, -reduction_db / 20.0);
        }
    }

    return std::make_tuple(gain, curr_env);
}

} // namespace cognis_dsp
