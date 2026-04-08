#pragma once

#include "aerial_manipulator/types.hpp"
#include "aerial_manipulator/rigid_body.hpp"
#include <vector>

namespace aerial_manipulator {

/**
 * @brief 2-DOF 3D manipulator kinematics and dynamics.
 *
 * Joint configuration:
 *   Joint 1 (azimuth):   rotation about body z-axis (yaw of arm)
 *   Joint 2 (elevation): rotation about the resulting y-axis (pitch of arm)
 *
 * This gives a hemispherical workspace below the quadrotor.
 * The arm is attached at attachment_offset in body frame.
 *
 * Coordinate convention:
 *   - q1 = 0, q2 = 0 → arm pointing straight down (-z_body)
 *   - q1 rotates the arm azimuthally (about z_body)
 *   - q2 tilts the arm from vertical (about rotated y-axis)
 */
class Manipulator {
public:
    Manipulator() = default;
    explicit Manipulator(const ManipulatorParams& params);

    // ── Kinematics ──

    /// Forward kinematics: joint angles → end-effector position in body frame
    Vec3 forward_kinematics(const Vec2& q) const;

    /// Link 1 COM position in body frame
    Vec3 link1_com_position(const Vec2& q) const;

    /// Link 2 COM position in body frame
    Vec3 link2_com_position(const Vec2& q) const;

    /// Geometric Jacobian of end-effector w.r.t. joint angles (body frame)
    Eigen::Matrix<double, 3, 2> jacobian(const Vec2& q) const;

    /// Rotation matrix from joint1 frame: R_z(q1)
    Mat3 joint1_rotation(double q1) const;

    /// Rotation matrix for link orientation in body frame
    Mat3 link1_orientation(const Vec2& q) const;
    Mat3 link2_orientation(const Vec2& q) const;

    // ── Dynamics matrices (manipulator subsystem only) ──

    /// 2×2 mass/inertia matrix of manipulator
    Mat2 mass_matrix(const Vec2& q) const;

    /// 2×2 Coriolis/centrifugal matrix
    Mat2 coriolis_matrix(const Vec2& q, const Vec2& q_dot) const;

    /// 2×1 gravity vector (depends on body orientation via R_body)
    Vec2 gravity_vector(const Vec2& q, const Mat3& R_body, double gravity) const;

    // ── Accessors ──
    const std::vector<RigidBody>& links() const { return links_; }
    const ManipulatorParams& params() const { return params_; }
    const Vec3& attachment_offset() const { return params_.attachment_offset; }

    /// Reaction wrench on quadrotor body from manipulator motion (Newton-Euler)
    /// Returns [force(3), torque(3)] in body frame
    Eigen::Matrix<double, 6, 1> joint_torque_to_body_wrench(
        const Vec2& q, const Vec2& q_dot, const Vec2& q_ddot,
        const Vec3& body_ang_vel, double gravity, const Mat3& R_body) const;

    /// COM Jacobians: d(r_ci)/d(q) for each link in body frame (3×2)
    Eigen::Matrix<double, 3, 2> link1_com_jacobian(const Vec2& q) const;
    Eigen::Matrix<double, 3, 2> link2_com_jacobian(const Vec2& q) const;

private:
    /// Skew-symmetric matrix [v]× for cross product
    static Mat3 skew(const Vec3& v);

    ManipulatorParams params_{};
    std::vector<RigidBody> links_;
};

}  // namespace aerial_manipulator
