#include "aerial_manipulator/rk4_integrator.hpp"

namespace aerial_manipulator {

StateVector RK4Integrator::step(const DerivativeFunc& f,
                                 double t, const StateVector& x, double dt) {
    StateVector k1 = f(t, x);
    StateVector k2 = f(t + 0.5 * dt, x + 0.5 * dt * k1);
    StateVector k3 = f(t + 0.5 * dt, x + 0.5 * dt * k2);
    StateVector k4 = f(t + dt, x + dt * k3);

    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
}

}  // namespace aerial_manipulator
