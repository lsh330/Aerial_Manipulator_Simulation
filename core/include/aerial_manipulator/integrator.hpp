#pragma once

#include "aerial_manipulator/types.hpp"
#include <functional>

namespace aerial_manipulator {

/**
 * @brief Abstract interface for ODE integrators (Strategy pattern).
 *
 * Integrates dx/dt = f(t, x) from (t, x) to (t + dt, x_next).
 * Subclasses implement specific numerical methods.
 */
class IntegratorBase {
public:
    using DerivativeFunc = std::function<StateVector(double, const StateVector&)>;

    virtual ~IntegratorBase() = default;

    /// Advance state by one step. Returns new state.
    virtual StateVector step(const DerivativeFunc& f,
                             double t, const StateVector& x, double dt) = 0;

    /// Name of the integration method
    virtual const char* name() const = 0;
};

}  // namespace aerial_manipulator
