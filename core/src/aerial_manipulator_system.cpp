#include "aerial_manipulator/aerial_manipulator_system.hpp"
#include <cmath>
#include <stdexcept>

namespace aerial_manipulator {

// ── Constructor ──

AerialManipulatorSystem::AerialManipulatorSystem(
    const QuadrotorParams& quad_params,
    const ManipulatorParams& manip_params,
    const EnvironmentParams& env_params)
    : quadrotor_(quad_params),
      manipulator_(manip_params),
      env_params_(env_params)
{}

double AerialManipulatorSystem::total_mass() const {
    return quadrotor_.body().mass()
         + manipulator_.links()[0].mass()
         + manipulator_.links()[1].mass();
}

// ── Quaternion utilities ──

StateVector AerialManipulatorSystem::normalize_quaternion(const StateVector& state) {
    StateVector s = state;
    Eigen::Map<Vec4> q(s.data() + idx::QUAT);
    double norm = q.norm();
    if (norm > 1e-10) {
        q /= norm;
    }
    return s;
}

Mat3 AerialManipulatorSystem::quaternion_to_rotation(const StateVector& state) {
    Quat q(state(idx::QUAT), state(idx::QUAT+1),
           state(idx::QUAT+2), state(idx::QUAT+3));
    q.normalize();
    return q.toRotationMatrix();
}

Vec4 AerialManipulatorSystem::omega_to_quat_derivative(const Quat& q, const Vec3& omega) {
    // q_dot = 0.5 * q ⊗ [0, omega]
    // In component form with q = [w, x, y, z]:
    double w = q.w(), x = q.x(), y = q.y(), z = q.z();
    double ox = omega(0), oy = omega(1), oz = omega(2);

    Vec4 q_dot;
    q_dot(0) = 0.5 * (-x*ox - y*oy - z*oz);  // dw/dt
    q_dot(1) = 0.5 * ( w*ox + y*oz - z*oy);   // dx/dt
    q_dot(2) = 0.5 * ( w*oy + z*ox - x*oz);   // dy/dt
    q_dot(3) = 0.5 * ( w*oz + x*oy - y*ox);   // dz/dt
    return q_dot;
}

// ── Coupled mass matrix M(q) ∈ R^{8×8} ──

Eigen::Matrix<double, DOF_TOTAL, DOF_TOTAL>
AerialManipulatorSystem::compute_mass_matrix(const StateVector& state) const {
    using Mat8 = Eigen::Matrix<double, DOF_TOTAL, DOF_TOTAL>;

    const Mat3 R = quaternion_to_rotation(state);
    const Vec2 q_joint(state(idx::JOINT_POS), state(idx::JOINT_POS+1));

    const double m0 = quadrotor_.body().mass();
    const double m1 = manipulator_.links()[0].mass();
    const double m2 = manipulator_.links()[1].mass();
    const double m_total = m0 + m1 + m2;
    const Mat3 J0 = quadrotor_.params().inertia;

    // Link COM positions in body frame
    Vec3 r_c1 = manipulator_.link1_com_position(q_joint);
    Vec3 r_c2 = manipulator_.link2_com_position(q_joint);

    // COM Jacobians w.r.t. joint angles (body frame)
    auto Jv1 = manipulator_.link1_com_jacobian(q_joint);
    auto Jv2 = manipulator_.link2_com_jacobian(q_joint);

    // Skew-symmetric matrices
    auto skew = [](const Vec3& v) -> Mat3 {
        Mat3 S;
        S <<     0, -v(2),  v(1),
              v(2),     0, -v(0),
             -v(1),  v(0),     0;
        return S;
    };

    Mat3 r_c1_x = skew(r_c1);
    Mat3 r_c2_x = skew(r_c2);

    // Link inertias in body frame
    Mat3 R_link = manipulator_.link1_orientation(q_joint);
    Mat3 I1_body = R_link * manipulator_.links()[0].inertia() * R_link.transpose();
    Mat3 I2_body = R_link * manipulator_.links()[1].inertia() * R_link.transpose();

    Mat8 M = Mat8::Zero();

    // (a) Translation-Translation block: M_tt = m_total * I_3
    M.block<3,3>(0,0) = m_total * Mat3::Identity();

    // (b) Rotation-Rotation block: M_rr = J0 + Σ(m_i * [r_ci]×^T [r_ci]× + I_i_body)
    M.block<3,3>(3,3) = J0
        + m1 * (r_c1_x.transpose() * r_c1_x) + I1_body
        + m2 * (r_c2_x.transpose() * r_c2_x) + I2_body;

    // (c) Manipulator-Manipulator block: M_mm (2×2)
    M.block<2,2>(6,6) = manipulator_.mass_matrix(q_joint);

    // (d) Translation-Rotation coupling: M_tr = -R * Σ m_i [r_ci]×
    Mat3 M_tr = -R * (m1 * r_c1_x + m2 * r_c2_x);
    M.block<3,3>(0,3) = M_tr;
    M.block<3,3>(3,0) = M_tr.transpose();

    // (e) Translation-Manipulator coupling: M_tm = R * Σ m_i * Jv_i
    Eigen::Matrix<double, 3, 2> M_tm = R * (m1 * Jv1 + m2 * Jv2);
    M.block<3,2>(0,6) = M_tm;
    M.block<2,3>(6,0) = M_tm.transpose();

    // (f) Rotation-Manipulator coupling: M_rm
    // M_rm = Σ(m_i * [r_ci]× * Jv_i + I_i * J_omega_i)
    // Joint rotation axes in body frame
    Vec3 axis1(0, 0, 1);  // azimuth
    Mat3 R1 = manipulator_.joint1_rotation(q_joint(0));
    Vec3 axis2 = R1 * Vec3(0, 1, 0);  // elevation

    Eigen::Matrix<double, 3, 2> J_omega;
    J_omega.col(0) = axis1;
    J_omega.col(1) = axis2;

    Eigen::Matrix<double, 3, 2> M_rm =
        m1 * r_c1_x * Jv1 + I1_body * J_omega
      + m2 * r_c2_x * Jv2 + I2_body * J_omega;

    M.block<3,2>(3,6) = M_rm;
    M.block<2,3>(6,3) = M_rm.transpose();

    return M;
}

// ── Coriolis/centrifugal vector C(q,q̇)·q̇ ∈ R^8 ──

Eigen::Matrix<double, DOF_TOTAL, 1>
AerialManipulatorSystem::compute_coriolis_vector(const StateVector& state) const {
    using Vec8 = Eigen::Matrix<double, DOF_TOTAL, 1>;

    const Mat3 R = quaternion_to_rotation(state);
    const Vec3 vel(state(idx::VEL), state(idx::VEL+1), state(idx::VEL+2));
    const Vec3 omega(state(idx::ANG_VEL), state(idx::ANG_VEL+1), state(idx::ANG_VEL+2));
    const Vec2 q_joint(state(idx::JOINT_POS), state(idx::JOINT_POS+1));
    const Vec2 q_dot(state(idx::JOINT_VEL), state(idx::JOINT_VEL+1));

    const double m1 = manipulator_.links()[0].mass();
    const double m2 = manipulator_.links()[1].mass();

    // Link COM positions and velocities in body frame
    Vec3 r_c1 = manipulator_.link1_com_position(q_joint);
    Vec3 r_c2 = manipulator_.link2_com_position(q_joint);
    auto Jv1 = manipulator_.link1_com_jacobian(q_joint);
    auto Jv2 = manipulator_.link2_com_jacobian(q_joint);
    Vec3 v_c1_rel = Jv1 * q_dot;
    Vec3 v_c2_rel = Jv2 * q_dot;

    auto skew = [](const Vec3& v) -> Mat3 {
        Mat3 S;
        S <<     0, -v(2),  v(1),
              v(2),     0, -v(0),
             -v(1),  v(0),     0;
        return S;
    };

    Vec8 coriolis = Vec8::Zero();

    // Translation Coriolis: Σ m_i * (ω × (ω × R*r_ci) + 2ω × R*v_ci_rel)
    // In world frame, for translational DOF
    Vec3 c_trans = Vec3::Zero();
    for (int i = 0; i < 2; ++i) {
        Vec3 r_ci = (i == 0) ? r_c1 : r_c2;
        Vec3 v_ci = (i == 0) ? v_c1_rel : v_c2_rel;
        double mi = (i == 0) ? m1 : m2;

        Vec3 r_world = R * r_ci;
        Vec3 v_world = R * v_ci;
        // Note: ω here is in body frame; transform contributions to world frame
        Vec3 omega_world = R * omega;
        c_trans += mi * (omega_world.cross(omega_world.cross(r_world))
                       + 2.0 * omega_world.cross(v_world));
    }
    coriolis.head<3>() = c_trans;

    // Rotation Coriolis: ω × (J_eff * ω) + contributions from arm
    Mat3 J0 = quadrotor_.params().inertia;
    Mat3 R_link = manipulator_.link1_orientation(q_joint);
    Mat3 I1_body = R_link * manipulator_.links()[0].inertia() * R_link.transpose();
    Mat3 I2_body = R_link * manipulator_.links()[1].inertia() * R_link.transpose();

    Mat3 J_eff = J0
        + m1 * (skew(r_c1).transpose() * skew(r_c1)) + I1_body
        + m2 * (skew(r_c2).transpose() * skew(r_c2)) + I2_body;

    Vec3 c_rot = omega.cross(J_eff * omega);

    // Add Coriolis from moving links
    for (int i = 0; i < 2; ++i) {
        Vec3 r_ci = (i == 0) ? r_c1 : r_c2;
        Vec3 v_ci = (i == 0) ? v_c1_rel : v_c2_rel;
        double mi = (i == 0) ? m1 : m2;
        c_rot += mi * skew(r_ci) * (2.0 * omega.cross(v_ci));
    }
    coriolis.segment<3>(3) = c_rot;

    // Manipulator Coriolis: C_mm(q, q_dot) * q_dot
    Mat2 C_mm = manipulator_.coriolis_matrix(q_joint, q_dot);
    coriolis.tail<2>() = C_mm * q_dot;

    return coriolis;
}

// ── Gravity vector G(q) ∈ R^8 ──

Eigen::Matrix<double, DOF_TOTAL, 1>
AerialManipulatorSystem::compute_gravity_vector(const StateVector& state) const {
    using Vec8 = Eigen::Matrix<double, DOF_TOTAL, 1>;

    const Mat3 R = quaternion_to_rotation(state);
    const Vec2 q_joint(state(idx::JOINT_POS), state(idx::JOINT_POS+1));
    const double g = env_params_.gravity;

    Vec8 G = Vec8::Zero();

    // Translational gravity: -m_total * g * e3 (world frame)
    G(2) = -total_mass() * g;

    // Rotational gravity: Σ m_i * [r_ci]× * R^T * g_world
    // (torque due to gravity acting on off-center masses)
    Vec3 g_body = R.transpose() * Vec3(0, 0, -g);
    Vec3 r_c1 = manipulator_.link1_com_position(q_joint);
    Vec3 r_c2 = manipulator_.link2_com_position(q_joint);
    double m1 = manipulator_.links()[0].mass();
    double m2 = manipulator_.links()[1].mass();

    Vec3 g_torque = m1 * r_c1.cross(g_body) + m2 * r_c2.cross(g_body);
    G.segment<3>(3) = g_torque;

    // Joint gravity
    G.tail<2>() = manipulator_.gravity_vector(q_joint, R, g);

    return G;
}

// ── Input mapping B(q) ∈ R^{8×6} ──

Eigen::Matrix<double, DOF_TOTAL, INPUT_DIM>
AerialManipulatorSystem::compute_input_matrix(const StateVector& state) const {
    using MatB = Eigen::Matrix<double, DOF_TOTAL, INPUT_DIM>;

    const Mat3 R = quaternion_to_rotation(state);
    MatB B = MatB::Zero();

    // Motor thrusts → translational force (world frame)
    // Total thrust along body z-axis, rotated to world
    Vec3 z_body = R.col(2);  // body z in world frame
    for (int i = 0; i < NUM_ROTORS; ++i) {
        B.block<3,1>(0, i) = z_body;  // each motor contributes along body z
    }

    // Motor thrusts → body torques via mixing matrix
    const auto& A_mix = quadrotor_.mixing_matrix();
    // A_mix maps [f1,f2,f3,f4] → [F_total, τ_roll, τ_pitch, τ_yaw]
    // We need rows 1,2,3 (torques) of A_mix
    B.block<3,4>(3, 0) = A_mix.bottomRows<3>();

    // Joint torques → joint acceleration (direct)
    B(6, 4) = 1.0;  // tau_q1
    B(7, 5) = 1.0;  // tau_q2

    return B;
}

// ── State derivative: dx/dt = f(x, u) ──

StateVector AerialManipulatorSystem::compute_state_derivative(
    const StateVector& state, const InputVector& input) const
{
    StateVector x_dot = StateVector::Zero();

    // Extract state components
    const Vec3 vel(state(idx::VEL), state(idx::VEL+1), state(idx::VEL+2));
    const Quat quat(state(idx::QUAT), state(idx::QUAT+1),
                    state(idx::QUAT+2), state(idx::QUAT+3));
    const Vec3 omega(state(idx::ANG_VEL), state(idx::ANG_VEL+1), state(idx::ANG_VEL+2));
    const Vec2 q_joint(state(idx::JOINT_POS), state(idx::JOINT_POS+1));
    const Vec2 q_dot_joint(state(idx::JOINT_VEL), state(idx::JOINT_VEL+1));

    // 1. Position derivative = velocity
    x_dot(idx::POS)   = vel(0);
    x_dot(idx::POS+1) = vel(1);
    x_dot(idx::POS+2) = vel(2);

    // 2. Quaternion derivative from angular velocity
    Vec4 q_dot_quat = omega_to_quat_derivative(quat, omega);
    x_dot(idx::QUAT)   = q_dot_quat(0);
    x_dot(idx::QUAT+1) = q_dot_quat(1);
    x_dot(idx::QUAT+2) = q_dot_quat(2);
    x_dot(idx::QUAT+3) = q_dot_quat(3);

    // 3. Joint position derivative = joint velocity
    x_dot(idx::JOINT_POS)   = q_dot_joint(0);
    x_dot(idx::JOINT_POS+1) = q_dot_joint(1);

    // 4. Solve M(q) * q_ddot = B*u - C(q,q_dot)*q_dot - G(q)
    //    for q_ddot = [v_dot(3), omega_dot(3), q_ddot_joint(2)]
    auto M = compute_mass_matrix(state);
    auto C_qdot = compute_coriolis_vector(state);
    auto G = compute_gravity_vector(state);
    auto B = compute_input_matrix(state);

    Eigen::Matrix<double, DOF_TOTAL, 1> rhs = B * input - C_qdot - G;

    // Solve: M * q_ddot = rhs
    Eigen::Matrix<double, DOF_TOTAL, 1> q_ddot = M.ldlt().solve(rhs);

    // 5. Write accelerations into state derivative
    // Linear acceleration (world frame)
    x_dot(idx::VEL)   = q_ddot(0);
    x_dot(idx::VEL+1) = q_ddot(1);
    x_dot(idx::VEL+2) = q_ddot(2);

    // Angular acceleration (body frame)
    x_dot(idx::ANG_VEL)   = q_ddot(3);
    x_dot(idx::ANG_VEL+1) = q_ddot(4);
    x_dot(idx::ANG_VEL+2) = q_ddot(5);

    // Joint accelerations
    x_dot(idx::JOINT_VEL)   = q_ddot(6);
    x_dot(idx::JOINT_VEL+1) = q_ddot(7);

    return x_dot;
}

// ── Integration ──

void AerialManipulatorSystem::set_integrator(std::shared_ptr<IntegratorBase> integrator) {
    integrator_ = std::move(integrator);
}

StateVector AerialManipulatorSystem::step(
    double t, const StateVector& state,
    const InputVector& input, double dt)
{
    if (!integrator_) {
        throw std::runtime_error("No integrator set. Call set_integrator() first.");
    }

    cached_input_ = input;

    auto derivative_func = [this](double t_inner, const StateVector& x) -> StateVector {
        return compute_state_derivative(x, cached_input_);
    };

    StateVector new_state = integrator_->step(derivative_func, t, state, dt);

    // Normalize quaternion to prevent drift
    return normalize_quaternion(new_state);
}

}  // namespace aerial_manipulator
