#pragma once

#include "aerial_manipulator/integrator.hpp"

namespace aerial_manipulator {

/**
 * @brief Runge-Kutta-Fehlberg 4(5) adaptive step-size integrator.
 *
 * Uses embedded 4th and 5th order solutions to estimate local error
 * and adapt the step size. Efficient for problems with varying dynamics.
 */
class RKF45Integrator : public IntegratorBase {
public:
    struct Config {
        double atol = 1e-8;     // absolute tolerance
        double rtol = 1e-6;     // relative tolerance
        double dt_min = 1e-6;   // minimum step size
        double dt_max = 0.01;   // maximum step size
        double safety = 0.9;    // safety factor for step adjustment
    };

    RKF45Integrator() = default;
    explicit RKF45Integrator(const Config& config) : config_(config) {}

    StateVector step(const DerivativeFunc& f,
                     double t, const StateVector& x, double dt) override;

    const char* name() const override { return "RKF45"; }

    /// Last adaptive step size used (for diagnostics)
    double last_step_size() const { return last_dt_; }

    Config& config() { return config_; }

private:
    Config config_;
    double last_dt_ = 0.0;
};

}  // namespace aerial_manipulator
