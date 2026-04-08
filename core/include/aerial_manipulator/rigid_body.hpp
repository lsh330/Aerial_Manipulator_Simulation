#pragma once

#include "aerial_manipulator/types.hpp"

namespace aerial_manipulator {

/**
 * @brief Single rigid body state and physical properties.
 *
 * Pure data holder — no dynamics computation.
 * Stores mass, inertia tensor, and kinematic state.
 */
class RigidBody {
public:
    RigidBody() = default;
    RigidBody(double mass, const Mat3& inertia,
              double length = 0.0, double com_distance = 0.0);

    // ── Accessors ──
    double mass() const { return mass_; }
    const Mat3& inertia() const { return inertia_; }
    double length() const { return length_; }
    double com_distance() const { return com_distance_; }

    void set_mass(double m) { mass_ = m; }
    void set_inertia(const Mat3& I) { inertia_ = I; }

private:
    double mass_ = 0.0;
    Mat3 inertia_ = Mat3::Zero();
    double length_ = 0.0;         // total link length
    double com_distance_ = 0.0;   // distance from joint to COM
};

}  // namespace aerial_manipulator
