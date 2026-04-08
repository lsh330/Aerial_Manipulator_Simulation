#pragma once

#include "aerial_manipulator/integrator.hpp"

namespace aerial_manipulator {

/**
 * @brief Classic 4th-order Runge-Kutta integrator.
 *
 * Fixed step size. Order 4 accuracy: local error O(dt^5).
 */
class RK4Integrator : public IntegratorBase {
public:
    StateVector step(const DerivativeFunc& f,
                     double t, const StateVector& x, double dt) override;

    const char* name() const override { return "RK4"; }
};

}  // namespace aerial_manipulator
