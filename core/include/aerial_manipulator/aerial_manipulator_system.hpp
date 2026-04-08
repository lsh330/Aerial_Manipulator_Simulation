#pragma once

#include "aerial_manipulator/types.hpp"
#include "aerial_manipulator/quadrotor.hpp"
#include "aerial_manipulator/manipulator.hpp"
#include "aerial_manipulator/integrator.hpp"
#include <memory>

namespace aerial_manipulator {

/**
 * @brief Coupled multibody dynamics: quadrotor + 2-DOF 3D manipulator.
 *
 * Implements the full coupled equation of motion:
 *   M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B * u
 *
 * where the generalized coordinates are split as:
 *   q = [position(3), orientation(quat→3 for EOM), joint_angles(2)]
 *
 * State vector uses quaternion for singularity-free attitude propagation:
 *   x = [pos(3), vel(3), quat(4), ang_vel(3), joint_pos(2), joint_vel(2)]
 */
class AerialManipulatorSystem {
public:
    AerialManipulatorSystem(const QuadrotorParams& quad_params,
                            const ManipulatorParams& manip_params,
                            const EnvironmentParams& env_params);

    // ── State derivative (core equation) ──

    /// Compute dx/dt = f(x, u). The main coupled dynamics equation.
    StateVector compute_state_derivative(const StateVector& state,
                                          const InputVector& input) const;

    // ── Dynamics matrices (8×8 generalized coordinates) ──

    /// Coupled mass matrix M(q) ∈ R^{8×8}
    Eigen::Matrix<double, DOF_TOTAL, DOF_TOTAL>
    compute_mass_matrix(const StateVector& state) const;

    /// Coriolis/centrifugal vector C(q, q_dot) * q_dot ∈ R^8
    Eigen::Matrix<double, DOF_TOTAL, 1>
    compute_coriolis_vector(const StateVector& state) const;

    /// Gravity vector G(q) ∈ R^8
    Eigen::Matrix<double, DOF_TOTAL, 1>
    compute_gravity_vector(const StateVector& state) const;

    /// Input mapping matrix B(q) ∈ R^{8×6}
    Eigen::Matrix<double, DOF_TOTAL, INPUT_DIM>
    compute_input_matrix(const StateVector& state) const;

    // ── Integration ──

    /// Advance state by dt using the current integrator
    StateVector step(double t, const StateVector& state,
                     const InputVector& input, double dt);

    /// Set the numerical integrator (Strategy pattern)
    void set_integrator(std::shared_ptr<IntegratorBase> integrator);

    // ── Accessors ──
    const Quadrotor& quadrotor() const { return quadrotor_; }
    const Manipulator& manipulator() const { return manipulator_; }
    const EnvironmentParams& environment() const { return env_params_; }

    /// Total system mass
    double total_mass() const;

private:
    /// Normalize quaternion in state vector
    static StateVector normalize_quaternion(const StateVector& state);

    /// Extract rotation matrix from quaternion in state
    static Mat3 quaternion_to_rotation(const StateVector& state);

    /// Angular velocity to quaternion derivative
    static Vec4 omega_to_quat_derivative(const Quat& q, const Vec3& omega);

    Quadrotor quadrotor_;
    Manipulator manipulator_;
    EnvironmentParams env_params_;
    std::shared_ptr<IntegratorBase> integrator_;

    // Cached input for use inside derivative function
    mutable InputVector cached_input_ = InputVector::Zero();
};

}  // namespace aerial_manipulator
