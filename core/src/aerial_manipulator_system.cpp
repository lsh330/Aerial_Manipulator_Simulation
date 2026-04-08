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
    // If norm < 1e-10 the quaternion is effectively zero; leave it as-is so
    // the caller can detect the degenerate condition rather than silently
    // introducing an arbitrary rotation.
    return s;
}

Mat3 AerialManipulatorSystem::quaternion_to_rotation(const StateVector& state) {
    Quat q(state(idx::QUAT), state(idx::QUAT+1),
           state(idx::QUAT+2), state(idx::QUAT+3));
    double norm = q.norm();
    if (norm < 1e-10) {
        // Zero quaternion: return identity rotation instead of producing NaN.
        return Mat3::Identity();
    }
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
//
// Computed via Christoffel symbols of the first kind:
//   [C(q,q̇)·q̇]_i = Σ_j,k c_{ijk} q̇_j q̇_k
//   c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)
//
// Hybrid approach:
//   - dM/dq_k for euler angles (k=3,4,5): numerical central differences on quaternion
//   - dM/dq_k for joint angles (k=6,7): analytical partial derivatives
// This reduces mass matrix evaluations from 10 to 6 (only euler perturbations).
// Guarantees M_dot - 2C is skew-symmetric → energy conservation.

Eigen::Matrix<double, DOF_TOTAL, 1>
AerialManipulatorSystem::compute_coriolis_vector(const StateVector& state) const {
    using Vec8 = Eigen::Matrix<double, DOF_TOTAL, 1>;
    using Mat8 = Eigen::Matrix<double, DOF_TOTAL, DOF_TOTAL>;

    // Generalized velocity vector: [v(3), omega(3), q_dot_joint(2)]
    Vec8 q_dot;
    q_dot.head<3>() = state.segment<3>(idx::VEL);
    q_dot.segment<3>(3) = state.segment<3>(idx::ANG_VEL);
    q_dot.tail<2>() = state.segment<2>(idx::JOINT_VEL);

    // Store dM/dq_k for each generalized coordinate
    Mat8 dM[DOF_TOTAL];
    for (int i = 0; i < DOF_TOTAL; ++i) {
        dM[i] = Mat8::Zero();
    }

    // dM/d(translation) = 0 → skip indices 0,1,2

    // ────────────────────────────────────────────────────────────────
    // dM/d(euler_angles) via quaternion perturbation (numerical, 6 M evals)
    // ────────────────────────────────────────────────────────────────
    constexpr double eps = 1e-7;
    for (int ax = 0; ax < 3; ++ax) {
        Vec3 axis = Vec3::Zero();
        axis(ax) = 1.0;

        for (int sign = 0; sign < 2; ++sign) {
            double angle = (sign == 0) ? eps : -eps;

            Quat q_orig(state(idx::QUAT), state(idx::QUAT+1),
                        state(idx::QUAT+2), state(idx::QUAT+3));
            q_orig.normalize();
            Quat dq(Eigen::AngleAxisd(angle, axis));
            Quat q_pert = q_orig * dq;
            q_pert.normalize();

            StateVector s_pert = state;
            s_pert(idx::QUAT)   = q_pert.w();
            s_pert(idx::QUAT+1) = q_pert.x();
            s_pert(idx::QUAT+2) = q_pert.y();
            s_pert(idx::QUAT+3) = q_pert.z();

            Mat8 M_pert = compute_mass_matrix(s_pert);

            if (sign == 0) {
                dM[3 + ax] += M_pert;
            } else {
                dM[3 + ax] -= M_pert;
            }
        }
        dM[3 + ax] /= (2.0 * eps);
    }

    // ────────────────────────────────────────────────────────────────
    // dM/d(q1) and dM/d(q2) — ANALYTICAL (0 extra M evals)
    // ────────────────────────────────────────────────────────────────
    // Extract all needed quantities at the current configuration
    const Mat3 R = quaternion_to_rotation(state);
    const Vec2 q_joint(state(idx::JOINT_POS), state(idx::JOINT_POS+1));
    const double q1 = q_joint(0), q2 = q_joint(1);
    const double c1 = std::cos(q1), s1 = std::sin(q1);
    const double c2 = std::cos(q2), s2 = std::sin(q2);

    const double m1 = manipulator_.links()[0].mass();
    const double m2 = manipulator_.links()[1].mass();
    const double lc1 = manipulator_.links()[0].com_distance();
    const double l1  = manipulator_.links()[0].length();
    const double lc2 = manipulator_.links()[1].com_distance();
    const double D   = l1 + lc2;  // total reach for link2 COM

    auto skew = [](const Vec3& v) -> Mat3 {
        Mat3 S;
        S <<     0, -v(2),  v(1),
              v(2),     0, -v(0),
             -v(1),  v(0),     0;
        return S;
    };

    // Current COM positions in body frame
    const Vec3 att = manipulator_.attachment_offset();
    Vec3 r_c1 = att + Vec3(lc1*c1*s2, lc1*s1*s2, -lc1*c2);
    Vec3 r_c2 = att + Vec3(D*c1*s2,   D*s1*s2,   -D*c2);

    // d(r_c1)/dq1, d(r_c1)/dq2, d(r_c2)/dq1, d(r_c2)/dq2
    Vec3 dr_c1_dq1(-lc1*s1*s2,  lc1*c1*s2,  0.0);
    Vec3 dr_c1_dq2( lc1*c1*c2,  lc1*s1*c2,  lc1*s2);
    Vec3 dr_c2_dq1(-D*s1*s2,    D*c1*s2,    0.0);
    Vec3 dr_c2_dq2( D*c1*c2,    D*s1*c2,    D*s2);

    // COM Jacobians Jv (3×2) and their derivatives
    // Jv1 = [dr_c1_dq1, dr_c1_dq2]
    Eigen::Matrix<double, 3, 2> Jv1, Jv2;
    Jv1.col(0) = dr_c1_dq1;  Jv1.col(1) = dr_c1_dq2;
    Jv2.col(0) = dr_c2_dq1;  Jv2.col(1) = dr_c2_dq2;

    // d(Jv1)/dq1, d(Jv1)/dq2, d(Jv2)/dq1, d(Jv2)/dq2
    Eigen::Matrix<double, 3, 2> dJv1_dq1, dJv1_dq2, dJv2_dq1, dJv2_dq2;
    // d^2(r_c1)/(dq1 dq1), d^2(r_c1)/(dq2 dq1)
    dJv1_dq1.col(0) = Vec3(-lc1*c1*s2, -lc1*s1*s2, 0.0);    // d²r_c1/dq1²
    dJv1_dq1.col(1) = Vec3(-lc1*s1*c2,  lc1*c1*c2, 0.0);    // d²r_c1/(dq2 dq1)
    dJv1_dq2.col(0) = Vec3(-lc1*s1*c2,  lc1*c1*c2, 0.0);    // d²r_c1/(dq1 dq2)
    dJv1_dq2.col(1) = Vec3(-lc1*c1*s2, -lc1*s1*s2, lc1*c2); // d²r_c1/dq2²

    dJv2_dq1.col(0) = Vec3(-D*c1*s2, -D*s1*s2, 0.0);
    dJv2_dq1.col(1) = Vec3(-D*s1*c2,  D*c1*c2, 0.0);
    dJv2_dq2.col(0) = Vec3(-D*s1*c2,  D*c1*c2, 0.0);
    dJv2_dq2.col(1) = Vec3(-D*c1*s2, -D*s1*s2, D*c2);

    // Link orientation and its derivatives
    // R_link = Rz(q1) * Ry(q2) = [c1*c2, -s1, c1*s2;  s1*c2, c1, s1*s2;  -s2, 0, c2]
    Mat3 R_link = manipulator_.link1_orientation(q_joint);

    // dR_link/dq1 = dRz/dq1 * Ry(q2)
    Mat3 dR_link_dq1;
    dR_link_dq1 << -s1*c2, -c1, -s1*s2,
                    c1*c2, -s1,  c1*s2,
                        0,   0,      0;

    // dR_link/dq2 = Rz(q1) * dRy/dq2
    Mat3 dR_link_dq2;
    dR_link_dq2 << -c1*s2, 0, c1*c2,
                   -s1*s2, 0, s1*c2,
                      -c2, 0,   -s2;

    const Mat3& I1_local = manipulator_.links()[0].inertia();
    const Mat3& I2_local = manipulator_.links()[1].inertia();
    Mat3 I1_body = R_link * I1_local * R_link.transpose();
    Mat3 I2_body = R_link * I2_local * R_link.transpose();

    // d(I_body)/dq = dR/dq * I_local * R^T + R * I_local * dR^T/dq
    auto compute_dI_body = [](const Mat3& dR, const Mat3& Rl, const Mat3& I_loc) -> Mat3 {
        return dR * I_loc * Rl.transpose() + Rl * I_loc * dR.transpose();
    };
    Mat3 dI1_dq1 = compute_dI_body(dR_link_dq1, R_link, I1_local);
    Mat3 dI1_dq2 = compute_dI_body(dR_link_dq2, R_link, I1_local);
    Mat3 dI2_dq1 = compute_dI_body(dR_link_dq1, R_link, I2_local);
    Mat3 dI2_dq2 = compute_dI_body(dR_link_dq2, R_link, I2_local);

    // Joint rotation axes
    Vec3 axis1(0, 0, 1);
    Mat3 R1 = manipulator_.joint1_rotation(q1);
    Vec3 axis2 = R1 * Vec3(0, 1, 0);
    // d(axis2)/dq1 = dR1/dq1 * [0,1,0]
    // dRz/dq1 = [-s1,-c1,0; c1,-s1,0; 0,0,0]
    Vec3 daxis2_dq1(-c1, -s1, 0.0);
    // d(axis2)/dq2 = 0 (axis2 does not depend on q2)

    // Angular velocity Jacobian J_omega (3×2)
    Eigen::Matrix<double, 3, 2> J_omega;
    J_omega.col(0) = axis1;
    J_omega.col(1) = axis2;

    // d(J_omega)/dq1
    Eigen::Matrix<double, 3, 2> dJ_omega_dq1 = Eigen::Matrix<double, 3, 2>::Zero();
    dJ_omega_dq1.col(1) = daxis2_dq1;  // axis1 is constant, axis2 depends on q1
    // d(J_omega)/dq2 = 0 (neither axis depends on q2)
    Eigen::Matrix<double, 3, 2> dJ_omega_dq2 = Eigen::Matrix<double, 3, 2>::Zero();

    // Skew symmetric matrices and their derivatives
    Mat3 r_c1_x = skew(r_c1);
    Mat3 r_c2_x = skew(r_c2);
    Mat3 dr_c1_dq1_x = skew(dr_c1_dq1);
    Mat3 dr_c1_dq2_x = skew(dr_c1_dq2);
    Mat3 dr_c2_dq1_x = skew(dr_c2_dq1);
    Mat3 dr_c2_dq2_x = skew(dr_c2_dq2);

    // ── Now compute dM/dq1 and dM/dq2 for each block ──

    // Helper lambda to compute dM for a given joint index (jidx: 0=q1, 1=q2)
    for (int jidx = 0; jidx < 2; ++jidx) {
        Mat8& dMj = dM[6 + jidx];

        // Derivatives of building blocks w.r.t. current joint
        const Vec3& drc1 = (jidx == 0) ? dr_c1_dq1 : dr_c1_dq2;
        const Vec3& drc2 = (jidx == 0) ? dr_c2_dq1 : dr_c2_dq2;
        const Mat3& drc1_x = (jidx == 0) ? dr_c1_dq1_x : dr_c1_dq2_x;
        const Mat3& drc2_x = (jidx == 0) ? dr_c2_dq1_x : dr_c2_dq2_x;
        const Mat3& dI1 = (jidx == 0) ? dI1_dq1 : dI1_dq2;
        const Mat3& dI2 = (jidx == 0) ? dI2_dq1 : dI2_dq2;
        const Eigen::Matrix<double, 3, 2>& dJv1 = (jidx == 0) ? dJv1_dq1 : dJv1_dq2;
        const Eigen::Matrix<double, 3, 2>& dJv2 = (jidx == 0) ? dJv2_dq1 : dJv2_dq2;
        const Eigen::Matrix<double, 3, 2>& dJ_om = (jidx == 0) ? dJ_omega_dq1 : dJ_omega_dq2;

        // (a) Translation-Translation: d/dq(m_total * I_3) = 0

        // (b) Rotation-Rotation: d/dq[J0 + Σ(m_i * [r_ci]×^T [r_ci]× + I_i_body)]
        //   = Σ m_i * (d[r_ci]×^T/dq * [r_ci]× + [r_ci]×^T * d[r_ci]×/dq) + dI_i/dq
        Mat3 dMrr = m1 * (drc1_x.transpose() * r_c1_x + r_c1_x.transpose() * drc1_x) + dI1
                  + m2 * (drc2_x.transpose() * r_c2_x + r_c2_x.transpose() * drc2_x) + dI2;
        dMj.block<3,3>(3,3) = dMrr;

        // (c) Manipulator-Manipulator: d/dq of 2×2 mass matrix
        //   M11 = (m1*lc1^2 + m2*D^2)*s2^2 + I1 + I2
        //   M22 = m1*lc1^2 + m2*D^2 + I1 + I2
        //   M12 = 0
        Mat2 dMmm = Mat2::Zero();
        const double alpha = m1*lc1*lc1 + m2*D*D;
        if (jidx == 0) {
            // dM/dq1: M11 depends on s2 only → d/dq1 = 0; M22 const → 0
            // dMmm stays zero
        } else {
            // dM/dq2: dM11/dq2 = alpha * 2*s2*c2 = alpha*sin(2*q2)
            dMmm(0,0) = alpha * 2.0 * s2 * c2;
            // dM22/dq2 = 0, dM12/dq2 = 0
        }
        dMj.block<2,2>(6,6) = dMmm;

        // (d) Translation-Rotation: d/dq[-R * (m1*[r_c1]× + m2*[r_c2]×)]
        //   R does not depend on joint angles → d/dq = -R * (m1*d[r_c1]×/dq + m2*d[r_c2]×/dq)
        Mat3 dM_tr = -R * (m1 * drc1_x + m2 * drc2_x);
        dMj.block<3,3>(0,3) = dM_tr;
        dMj.block<3,3>(3,0) = dM_tr.transpose();

        // (e) Translation-Manipulator: d/dq[R * (m1*Jv1 + m2*Jv2)]
        //   = R * (m1*dJv1/dq + m2*dJv2/dq)
        Eigen::Matrix<double, 3, 2> dM_tm = R * (m1 * dJv1 + m2 * dJv2);
        dMj.block<3,2>(0,6) = dM_tm;
        dMj.block<2,3>(6,0) = dM_tm.transpose();

        // (f) Rotation-Manipulator: d/dq[Σ(m_i*[r_ci]×*Jv_i + I_i*J_omega)]
        //   = Σ m_i*(d[r_ci]×/dq * Jv_i + [r_ci]× * dJv_i/dq) + dI_i/dq * J_omega + I_i * dJ_omega/dq
        Eigen::Matrix<double, 3, 2> dM_rm =
              m1 * (drc1_x * Jv1 + r_c1_x * dJv1) + dI1 * J_omega + I1_body * dJ_om
            + m2 * (drc2_x * Jv2 + r_c2_x * dJv2) + dI2 * J_omega + I2_body * dJ_om;
        dMj.block<3,2>(3,6) = dM_rm;
        dMj.block<2,3>(6,3) = dM_rm.transpose();
    }

    // ── Compute C(q,q̇)·q̇ via Christoffel symbols ──
    Vec8 coriolis = Vec8::Zero();
    for (int i = 0; i < DOF_TOTAL; ++i) {
        double sum = 0.0;
        for (int j = 0; j < DOF_TOTAL; ++j) {
            for (int k = 0; k < DOF_TOTAL; ++k) {
                // c_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)
                double christoffel = 0.5 * (dM[k](i,j) + dM[j](i,k) - dM[i](j,k));
                sum += christoffel * q_dot(j) * q_dot(k);
            }
        }
        coriolis(i) = sum;
    }

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

    // Lagrangian form: G = dV/dq, V = m*g*z → G_z = +m*g
    // EOM: M*q_ddot = B*u - C*q_dot - G
    G(2) = total_mass() * g;

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
    // Ensure quaternion is normalised before computing derivatives.
    // This guards each RK sub-step against gradual norm drift without
    // requiring explicit normalisation inside the integrator.
    StateVector s = state;
    Eigen::Map<Vec4> q_map(s.data() + idx::QUAT);
    double qnorm = q_map.norm();
    if (qnorm > 1e-10) {
        q_map /= qnorm;
    }

    StateVector x_dot = StateVector::Zero();

    // Extract state components (use normalised s)
    const Vec3 vel(s(idx::VEL), s(idx::VEL+1), s(idx::VEL+2));
    const Quat quat(s(idx::QUAT), s(idx::QUAT+1),
                    s(idx::QUAT+2), s(idx::QUAT+3));
    const Vec3 omega(s(idx::ANG_VEL), s(idx::ANG_VEL+1), s(idx::ANG_VEL+2));
    const Vec2 q_joint(s(idx::JOINT_POS), s(idx::JOINT_POS+1));
    const Vec2 q_dot_joint(s(idx::JOINT_VEL), s(idx::JOINT_VEL+1));

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

    // 4. Solve M(q) * q_ddot = B*u - C(q,q_dot)*q_dot - G(q) + F_drag
    //    for q_ddot = [v_dot(3), omega_dot(3), q_ddot_joint(2)]
    auto M = compute_mass_matrix(s);
    auto C_qdot = compute_coriolis_vector(s);
    auto G = compute_gravity_vector(s);
    auto B = compute_input_matrix(s);

    // Translational aerodynamic drag (body frame → world frame)
    const Mat3 R = quaternion_to_rotation(s);
    Vec3 vel_body = R.transpose() * vel;
    Vec3 drag_body = quadrotor_.compute_drag(vel_body);
    Vec3 drag_world = R * drag_body;

    Eigen::Matrix<double, DOF_TOTAL, 1> rhs = B * input - C_qdot - G;
    // Add drag as external force on translational DOFs
    rhs.head<3>() += drag_world;

    // Solve: M * q_ddot = rhs
    // Primary: LDLT (fast, requires positive-definite M).
    // Fallback: column-pivoting QR (robust when M is near-singular,
    //           e.g. at kinematic singularities such as q2 near 0 or pi).
    Eigen::Matrix<double, DOF_TOTAL, 1> q_ddot;
    auto ldlt = M.ldlt();
    if (ldlt.info() == Eigen::Success) {
        q_ddot = ldlt.solve(rhs);
    } else {
        // M failed LDLT factorisation (non-positive-definite or numerically
        // singular). Fall back to the more robust QR decomposition.
        q_ddot = M.colPivHouseholderQr().solve(rhs);
    }

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

    // Capture input by value so that concurrent calls on different instances
    // are safe and no mutable member is needed.
    auto derivative_func = [this, input](double t_inner, const StateVector& x) -> StateVector {
        return compute_state_derivative(x, input);
    };

    StateVector new_state = integrator_->step(derivative_func, t, state, dt);

    // Normalize quaternion to prevent drift
    return normalize_quaternion(new_state);
}

}  // namespace aerial_manipulator
