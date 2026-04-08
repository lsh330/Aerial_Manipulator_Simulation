#include "aerial_manipulator/rigid_body.hpp"

namespace aerial_manipulator {

RigidBody::RigidBody(double mass, const Mat3& inertia,
                     double length, double com_distance)
    : mass_(mass), inertia_(inertia),
      length_(length), com_distance_(com_distance) {}

}  // namespace aerial_manipulator
