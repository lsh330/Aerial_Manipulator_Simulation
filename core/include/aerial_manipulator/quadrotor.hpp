#pragma once

#include "aerial_manipulator/types.hpp"
#include "aerial_manipulator/rigid_body.hpp"

namespace aerial_manipulator {

/**
 * @brief Quadrotor-specific dynamics: thrust model, torque generation, drag.
 *
 * Computes body-frame forces and torques from 4 rotor speeds.
 * Motor layout (X-configuration, looking from above):
 *
 *     1(CW)  front  2(CCW)
 *          \  |  /
 *           [COM]
 *          /  |  \
 *     4(CCW) back  3(CW)
 */
class Quadrotor {
public:
    Quadrotor() = default;
    explicit Quadrotor(const QuadrotorParams& params);

    /// Total thrust force in body z-axis [N] from 4 motor thrusts
    Vec3 compute_forces(const Vec4& motor_thrusts) const;

    /// Body-frame torques [N·m] from 4 motor thrusts
    Vec3 compute_torques(const Vec4& motor_thrusts) const;

    /// Translational drag force in body frame (opposes velocity)
    Vec3 compute_drag(const Vec3& body_velocity) const;

    /// First-order motor dynamics: d(omega)/dt = (cmd - omega) / tau
    Vec4 motor_dynamics(const Vec4& motor_cmd, const Vec4& motor_current) const;

    /// Mixing matrix: [F; tau_x; tau_y; tau_z] = A_mix * [f1; f2; f3; f4]
    const Eigen::Matrix4d& mixing_matrix() const { return mixing_matrix_; }

    /// Inverse mixing matrix for control allocation
    const Eigen::Matrix4d& mixing_matrix_inv() const { return mixing_matrix_inv_; }

    const RigidBody& body() const { return body_; }
    const QuadrotorParams& params() const { return params_; }

    /// Total thrust for hover: (m_quad + m_payload) * g
    double hover_thrust(double total_mass, double gravity) const;

private:
    void build_mixing_matrix();

    RigidBody body_;
    QuadrotorParams params_{};
    Eigen::Matrix4d mixing_matrix_ = Eigen::Matrix4d::Zero();
    Eigen::Matrix4d mixing_matrix_inv_ = Eigen::Matrix4d::Zero();
};

}  // namespace aerial_manipulator
