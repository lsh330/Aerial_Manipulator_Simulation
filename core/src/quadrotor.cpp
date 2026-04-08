#include "aerial_manipulator/quadrotor.hpp"
#include <cmath>

namespace aerial_manipulator {

Quadrotor::Quadrotor(const QuadrotorParams& params)
    : params_(params)
{
    body_ = RigidBody(params.mass, params.inertia);
    build_mixing_matrix();
}

void Quadrotor::build_mixing_matrix() {
    const double L = params_.arm_length;
    const double k = params_.torque_coeff / params_.thrust_coeff;

    // X-configuration mixing matrix
    // [F_total ]   [  1    1    1    1  ] [f1]
    // [tau_roll] = [  0   -L    0    L  ] [f2]
    // [tau_pitch]  [  L    0   -L    0  ] [f3]
    // [tau_yaw ]   [  k   -k    k   -k  ] [f4]
    mixing_matrix_ << 1.0,  1.0,  1.0,  1.0,
                      0.0, -L,    0.0,  L,
                      L,    0.0, -L,    0.0,
                      k,   -k,    k,   -k;

    mixing_matrix_inv_ = mixing_matrix_.inverse();
}

Vec3 Quadrotor::compute_forces(const Vec4& motor_thrusts) const {
    double total_thrust = motor_thrusts.sum();
    return Vec3(0.0, 0.0, total_thrust);  // body z-axis
}

Vec3 Quadrotor::compute_torques(const Vec4& motor_thrusts) const {
    Eigen::Vector4d wrench = mixing_matrix_ * motor_thrusts;
    return wrench.tail<3>();  // [tau_roll, tau_pitch, tau_yaw]
}

Vec3 Quadrotor::compute_drag(const Vec3& body_velocity) const {
    return -params_.drag_coeff * body_velocity;
}

Vec4 Quadrotor::motor_dynamics(const Vec4& motor_cmd, const Vec4& motor_current) const {
    // First-order lag: d(omega)/dt = (cmd - omega) / tau_m
    return (motor_cmd - motor_current) / params_.motor_time_constant;
}

double Quadrotor::hover_thrust(double total_mass, double gravity) const {
    return total_mass * gravity;
}

}  // namespace aerial_manipulator
