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
    double mass;
    Mat3 inertia;
    double arm_length;
    double thrust_coeff;
    double torque_coeff;
    double drag_coeff;
    double motor_time_constant;
    double max_motor_speed;
};

struct LinkParams {
    double mass;
    double length;
    double com_distance;
    Mat3 inertia;
};

struct ManipulatorParams {
    Vec3 attachment_offset;
    LinkParams link1;
    LinkParams link2;
    Vec2 joint_lower_limit;
    Vec2 joint_upper_limit;
    double max_joint_torque;
};

struct EnvironmentParams {
    double gravity = 9.81;
    double air_density = 1.225;
};

}  // namespace aerial_manipulator
