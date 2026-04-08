#pragma once

#include <Eigen/Dense>
#include <cstdint>

namespace aerial_manipulator {

// ── Eigen type aliases ──
using Vec2 = Eigen::Vector2d;
using Vec3 = Eigen::Vector3d;
using Vec4 = Eigen::Vector4d;
using Mat2 = Eigen::Matrix2d;
using Mat3 = Eigen::Matrix3d;
using VecX = Eigen::VectorXd;
using MatX = Eigen::MatrixXd;
using Quat = Eigen::Quaterniond;

// ── System dimensions ──
// State: [pos(3), vel(3), quat(4), ang_vel(3), joint_pos(2), joint_vel(2)] = 17
// Input: [motor_thrusts(4), joint_torques(2)] = 6
constexpr int STATE_DIM = 17;
constexpr int INPUT_DIM = 6;
constexpr int NUM_ROTORS = 4;
constexpr int NUM_JOINTS = 2;
constexpr int DOF_QUADROTOR = 6;   // translation(3) + rotation(3)
constexpr int DOF_TOTAL = 8;       // quadrotor(6) + joints(2)

using StateVector = Eigen::Matrix<double, STATE_DIM, 1>;
using InputVector = Eigen::Matrix<double, INPUT_DIM, 1>;

// ── State index mapping ──
namespace idx {
    constexpr int POS    = 0;   // x, y, z
    constexpr int VEL    = 3;   // vx, vy, vz
    constexpr int QUAT   = 6;   // qw, qx, qy, qz
    constexpr int ANG_VEL = 10; // wx, wy, wz
    constexpr int JOINT_POS = 13; // q1, q2
    constexpr int JOINT_VEL = 15; // q1_dot, q2_dot
}

// ── Physical parameters ──

struct QuadrotorParams {
    double mass;                 ///< [kg] quadrotor body mass (excluding manipulator)
    Mat3 inertia;                ///< [kg·m²] body-frame inertia tensor (3×3 diagonal)
    double arm_length;           ///< [m] motor-to-center distance
    double thrust_coeff;         ///< [N/(rad/s)²] rotor thrust coefficient k_f: f_i = k_f * ω_i²
    double torque_coeff;         ///< [N·m/(rad/s)²] rotor torque coefficient k_τ: τ_i = k_τ * ω_i²
    double drag_coeff;           ///< [N·s/m] isotropic translational drag coefficient
    double motor_time_constant;  ///< [s] first-order motor response time constant τ_m
    double max_motor_speed;      ///< [rad/s] motor angular speed saturation ω_max
};

struct LinkParams {
    double mass;          ///< [kg] link mass
    double length;        ///< [m] total link length
    double com_distance;  ///< [m] distance from proximal joint to center of mass
    Mat3 inertia;         ///< [kg·m²] link-frame inertia tensor (3×3 diagonal, principal axes)
};

struct ManipulatorParams {
    Vec3 attachment_offset;    ///< [m] joint1 position in body frame relative to quadrotor COM
    LinkParams link1;          ///< First link (azimuth-elevation gimbal)
    LinkParams link2;          ///< Second link (extends along link1 direction, no elbow)
    Vec2 joint_lower_limit;    ///< [rad] lower joint limits [q1_min, q2_min]
    Vec2 joint_upper_limit;    ///< [rad] upper joint limits [q1_max, q2_max]
    double max_joint_torque;   ///< [N·m] maximum torque per joint actuator
};

struct EnvironmentParams {
    double gravity = 9.81;      ///< [m/s²] gravitational acceleration magnitude
    double air_density = 1.225; ///< [kg/m³] air density at sea level
    Vec3 wind_velocity = Vec3::Zero();  ///< [m/s] constant wind velocity in world frame (NED→ENU)
};

}  // namespace aerial_manipulator
