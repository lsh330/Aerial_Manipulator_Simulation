#include "aerial_manipulator/rkf45_integrator.hpp"
#include <cmath>
#include <algorithm>

namespace aerial_manipulator {

StateVector RKF45Integrator::step(const DerivativeFunc& f,
                                   double t, const StateVector& x, double dt) {
    // Runge-Kutta-Fehlberg coefficients (Butcher tableau)
    constexpr double a2 = 1.0/4.0;
    constexpr double a3 = 3.0/8.0;
    constexpr double a4 = 12.0/13.0;
    constexpr double a5 = 1.0;
    constexpr double a6 = 1.0/2.0;

    constexpr double b21 = 1.0/4.0;
    constexpr double b31 = 3.0/32.0,   b32 = 9.0/32.0;
    constexpr double b41 = 1932.0/2197.0, b42 = -7200.0/2197.0, b43 = 7296.0/2197.0;
    constexpr double b51 = 439.0/216.0, b52 = -8.0, b53 = 3680.0/513.0, b54 = -845.0/4104.0;
    constexpr double b61 = -8.0/27.0, b62 = 2.0, b63 = -3544.0/2565.0, b64 = 1859.0/4104.0, b65 = -11.0/40.0;

    // 4th order weights
    constexpr double c1 = 25.0/216.0, c3 = 1408.0/2565.0, c4 = 2197.0/4104.0, c5 = -1.0/5.0;

    // 5th order weights
    constexpr double d1 = 16.0/135.0, d3 = 6656.0/12825.0, d4 = 28561.0/56430.0, d5 = -9.0/50.0, d6 = 2.0/55.0;

    double h = dt;
    StateVector x_current = x;
    double t_current = t;
    double t_end = t + dt;

    while (t_current < t_end - 1e-14) {
        h = std::min(h, t_end - t_current);
        h = std::clamp(h, config_.dt_min, config_.dt_max);

        StateVector k1 = f(t_current, x_current);
        StateVector k2 = f(t_current + a2*h, x_current + h*(b21*k1));
        StateVector k3 = f(t_current + a3*h, x_current + h*(b31*k1 + b32*k2));
        StateVector k4 = f(t_current + a4*h, x_current + h*(b41*k1 + b42*k2 + b43*k3));
        StateVector k5 = f(t_current + a5*h, x_current + h*(b51*k1 + b52*k2 + b53*k3 + b54*k4));
        StateVector k6 = f(t_current + a6*h, x_current + h*(b61*k1 + b62*k2 + b63*k3 + b64*k4 + b65*k5));

        // 4th order solution
        StateVector x4 = x_current + h * (c1*k1 + c3*k3 + c4*k4 + c5*k5);
        // 5th order solution
        StateVector x5 = x_current + h * (d1*k1 + d3*k3 + d4*k4 + d5*k5 + d6*k6);

        // Error estimate
        StateVector err = x5 - x4;
        double err_norm = 0.0;
        for (int i = 0; i < STATE_DIM; ++i) {
            double scale = config_.atol + config_.rtol * std::max(std::abs(x_current(i)), std::abs(x5(i)));
            err_norm = std::max(err_norm, std::abs(err(i)) / scale);
        }

        if (err_norm <= 1.0) {
            // Accept step
            x_current = x5;
            t_current += h;
            last_dt_ = h;

            // Increase step size
            if (err_norm > 1e-15) {
                h *= config_.safety * std::pow(1.0 / err_norm, 0.2);
            } else {
                h *= 2.0;
            }
        } else {
            // Reject step, reduce step size
            h *= config_.safety * std::pow(1.0 / err_norm, 0.25);
            h = std::max(h, config_.dt_min);
        }
    }

    return x_current;
}

}  // namespace aerial_manipulator
