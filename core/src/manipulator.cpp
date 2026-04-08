#include "aerial_manipulator/manipulator.hpp"
#include <cmath>

namespace aerial_manipulator {

Manipulator::Manipulator(const ManipulatorParams& params)
    : params_(params)
{
    links_.reserve(NUM_JOINTS);
    links_.emplace_back(params.link1.mass, params.link1.inertia,
                        params.link1.length, params.link1.com_distance);
    links_.emplace_back(params.link2.mass, params.link2.inertia,
                        params.link2.length, params.link2.com_distance);
}

// ── Helper: skew-symmetric matrix ──
Mat3 Manipulator::skew(const Vec3& v) {
    Mat3 S;
    S <<    0, -v(2),  v(1),
         v(2),     0, -v(0),
        -v(1),  v(0),     0;
    return S;
}

// ── Rotation matrices ──

Mat3 Manipulator::joint1_rotation(double q1) const {
    // Rotation about body z-axis (azimuth)
    const double c1 = std::cos(q1), s1 = std::sin(q1);
    Mat3 Rz;
    Rz << c1, -s1, 0,
          s1,  c1, 0,
           0,   0, 1;
    return Rz;
}

Mat3 Manipulator::link1_orientation(const Vec2& q) const {
    // R_z(q1) * R_y(q2)
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));
    Mat3 R;
    R << c1*c2, -s1, c1*s2,
         s1*c2,  c1, s1*s2,
           -s2,   0,    c2;
    return R;
}

Mat3 Manipulator::link2_orientation(const Vec2& q) const {
    // Same as link1 for 2-DOF (both links share azimuth-elevation)
    return link1_orientation(q);
}

// ── Forward Kinematics ──

Vec3 Manipulator::link1_com_position(const Vec2& q) const {
    const double lc1 = links_[0].com_distance();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    // COM of link1 in body frame
    return params_.attachment_offset + Vec3(
        lc1 * c1 * s2,
        lc1 * s1 * s2,
        -lc1 * c2
    );
}

Vec3 Manipulator::link2_com_position(const Vec2& q) const {
    const double l1 = links_[0].length();
    const double lc2 = links_[1].com_distance();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    // Joint2 is at end of link1, link2 extends in same direction (no elbow)
    // For 3D azimuth-elevation, link2 continues along link1's direction
    const double total = l1 + lc2;
    return params_.attachment_offset + Vec3(
        total * c1 * s2,
        total * s1 * s2,
        -total * c2
    );
}

Vec3 Manipulator::forward_kinematics(const Vec2& q) const {
    const double l1 = links_[0].length();
    const double l2 = links_[1].length();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    const double total = l1 + l2;
    return params_.attachment_offset + Vec3(
        total * c1 * s2,
        total * s1 * s2,
        -total * c2
    );
}

// ── Jacobian ──

Eigen::Matrix<double, 3, 2> Manipulator::jacobian(const Vec2& q) const {
    const double l1 = links_[0].length();
    const double l2 = links_[1].length();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));
    const double L = l1 + l2;

    Eigen::Matrix<double, 3, 2> J;
    // d(end_effector)/d(q1) — azimuth rotation
    J(0, 0) = -L * s1 * s2;
    J(1, 0) =  L * c1 * s2;
    J(2, 0) =  0.0;
    // d(end_effector)/d(q2) — elevation rotation
    J(0, 1) =  L * c1 * c2;
    J(1, 1) =  L * s1 * c2;
    J(2, 1) =  L * s2;

    return J;
}

// ── Dynamics: Mass matrix ──

Mat2 Manipulator::mass_matrix(const Vec2& q) const {
    const double m1 = links_[0].mass(), m2 = links_[1].mass();
    const double lc1 = links_[0].com_distance();
    const double l1 = links_[0].length();
    const double lc2 = links_[1].com_distance();
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    // Izz components from link inertia tensors
    const double I1 = links_[0].inertia()(2, 2);  // about link z-axis
    const double I2 = links_[1].inertia()(2, 2);

    // For azimuth-elevation 3D joint:
    // M(1,1): inertia about azimuth axis (z) — depends on q2
    const double M11 = (m1 * lc1 * lc1 + m2 * (l1 + lc2) * (l1 + lc2)) * s2 * s2
                       + I1 + I2;
    // M(2,2): inertia about elevation axis
    const double M22 = m1 * lc1 * lc1 + m2 * (l1 + lc2) * (l1 + lc2) + I1 + I2;
    // M(1,2) = M(2,1): coupling (zero for azimuth-elevation)
    const double M12 = 0.0;

    Mat2 M;
    M << M11, M12,
         M12, M22;
    return M;
}

// ── Dynamics: Coriolis matrix ──

Mat2 Manipulator::coriolis_matrix(const Vec2& q, const Vec2& q_dot) const {
    const double m1 = links_[0].mass(), m2 = links_[1].mass();
    const double lc1 = links_[0].com_distance();
    const double l1 = links_[0].length();
    const double lc2 = links_[1].com_distance();
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    // h = (m1*lc1^2 + m2*(l1+lc2)^2) * sin(q2)*cos(q2)
    const double h = (m1 * lc1 * lc1 + m2 * (l1 + lc2) * (l1 + lc2)) * s2 * c2;

    Mat2 C;
    C << h * q_dot(1),         h * q_dot(0),
        -h * q_dot(0),         0.0;
    return C;
}

// ── Dynamics: Gravity vector ──

Vec2 Manipulator::gravity_vector(const Vec2& q, const Mat3& R_body,
                                  double gravity) const {
    const double m1 = links_[0].mass(), m2 = links_[1].mass();
    const double lc1 = links_[0].com_distance();
    const double l1 = links_[0].length();
    const double lc2 = links_[1].com_distance();

    // Gravity in body frame: g_body = R_body^T * [0, 0, -g]
    Vec3 g_world(0.0, 0.0, -gravity);
    Vec3 g_body = R_body.transpose() * g_world;

    // Partial derivatives of potential energy w.r.t. q1, q2
    // V = -g_body^T * (m1 * r_c1 + m2 * r_c2)
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    // dV/dq1
    const double dV_dq1 = -(m1 * lc1 + m2 * (l1 + lc2)) * s2 *
                           (-g_body(0) * s1 + g_body(1) * c1);

    // dV/dq2
    const double dV_dq2 = -(m1 * lc1 + m2 * (l1 + lc2)) *
                           (g_body(0) * c1 * c2 + g_body(1) * s1 * c2 + g_body(2) * s2);

    return Vec2(dV_dq1, dV_dq2);
}

// ── COM Jacobians ──

Eigen::Matrix<double, 3, 2> Manipulator::link1_com_jacobian(const Vec2& q) const {
    const double lc1 = links_[0].com_distance();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));

    Eigen::Matrix<double, 3, 2> J;
    J(0, 0) = -lc1 * s1 * s2;  J(0, 1) =  lc1 * c1 * c2;
    J(1, 0) =  lc1 * c1 * s2;  J(1, 1) =  lc1 * s1 * c2;
    J(2, 0) =  0.0;            J(2, 1) =  lc1 * s2;
    return J;
}

Eigen::Matrix<double, 3, 2> Manipulator::link2_com_jacobian(const Vec2& q) const {
    const double l1 = links_[0].length();
    const double lc2 = links_[1].com_distance();
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));
    const double D = l1 + lc2;

    Eigen::Matrix<double, 3, 2> J;
    J(0, 0) = -D * s1 * s2;  J(0, 1) =  D * c1 * c2;
    J(1, 0) =  D * c1 * s2;  J(1, 1) =  D * s1 * c2;
    J(2, 0) =  0.0;          J(2, 1) =  D * s2;
    return J;
}

// ── Reaction wrench on quadrotor body (Newton-Euler backward recursion) ──

Eigen::Matrix<double, 6, 1> Manipulator::joint_torque_to_body_wrench(
    const Vec2& q, const Vec2& q_dot, const Vec2& q_ddot,
    const Vec3& body_ang_vel, double gravity, const Mat3& R_body) const
{
    const double m1 = links_[0].mass(), m2 = links_[1].mass();

    // Gravity in body frame
    Vec3 g_body = R_body.transpose() * Vec3(0.0, 0.0, -gravity);

    // COM positions in body frame
    Vec3 r_c1 = link1_com_position(q);
    Vec3 r_c2 = link2_com_position(q);

    // COM velocity Jacobians
    auto J_c1 = link1_com_jacobian(q);
    auto J_c2 = link2_com_jacobian(q);

    // COM velocities in body frame (relative to body, not including body motion)
    Vec3 v_c1_rel = J_c1 * q_dot;
    Vec3 v_c2_rel = J_c2 * q_dot;

    // COM accelerations in body frame (relative to body)
    // a_rel = J * q_ddot + J_dot * q_dot
    // Compute J_dot * q_dot numerically via finite difference of J*q_dot
    // Instead, use analytical: d/dt(J*q_dot) = J*q_ddot + dJ/dt*q_dot
    // For the reaction wrench, we need the total inertial acceleration of each link COM:
    // a_ci_world = a_body + alpha_body × r_ci + omega × (omega × r_ci)
    //            + 2*omega × v_ci_rel + a_ci_rel

    // Relative acceleration: a_ci_rel = J_ci * q_ddot + J_ci_dot * q_dot
    // We compute J_dot * q_dot analytically for each link
    const double c1 = std::cos(q(0)), s1 = std::sin(q(0));
    const double c2 = std::cos(q(1)), s2 = std::sin(q(1));
    const double q1d = q_dot(0), q2d = q_dot(1);

    // Link 1: J_dot * q_dot for COM
    const double lc1 = links_[0].com_distance();
    Vec3 Jdot_qd_1;
    Jdot_qd_1(0) = lc1 * (-c1*s2*q1d*q1d - s1*c2*q1d*q2d - s1*c2*q2d*q1d + c1*(-s2)*q2d*q2d);
    Jdot_qd_1(1) = lc1 * (-s1*s2*q1d*q1d + c1*c2*q1d*q2d + c1*c2*q2d*q1d + s1*(-s2)*q2d*q2d);
    Jdot_qd_1(2) = lc1 * (c2*q2d*q2d);

    // Link 2: J_dot * q_dot for COM
    const double D = links_[0].length() + links_[1].com_distance();
    Vec3 Jdot_qd_2;
    Jdot_qd_2(0) = D * (-c1*s2*q1d*q1d - s1*c2*q1d*q2d - s1*c2*q2d*q1d + c1*(-s2)*q2d*q2d);
    Jdot_qd_2(1) = D * (-s1*s2*q1d*q1d + c1*c2*q1d*q2d + c1*c2*q2d*q1d + s1*(-s2)*q2d*q2d);
    Jdot_qd_2(2) = D * (c2*q2d*q2d);

    Vec3 a_c1_rel = J_c1 * q_ddot + Jdot_qd_1;
    Vec3 a_c2_rel = J_c2 * q_ddot + Jdot_qd_2;

    // Coriolis acceleration from body rotation: 2 * omega × v_rel
    Vec3 coriolis_1 = 2.0 * body_ang_vel.cross(v_c1_rel);
    Vec3 coriolis_2 = 2.0 * body_ang_vel.cross(v_c2_rel);

    // Centripetal acceleration: omega × (omega × r_ci)
    Vec3 centripetal_1 = body_ang_vel.cross(body_ang_vel.cross(r_c1));
    Vec3 centripetal_2 = body_ang_vel.cross(body_ang_vel.cross(r_c2));

    // Total force each link exerts on body (reaction = negative of link's inertial force)
    // F_reaction = -sum_i m_i * (a_ci_rel + coriolis_i + centripetal_i - g_body)
    // Note: we exclude body linear/angular acceleration terms since those are
    // part of the coupled mass matrix, not the reaction wrench
    Vec3 f1 = m1 * (a_c1_rel + coriolis_1 + centripetal_1 - g_body);
    Vec3 f2 = m2 * (a_c2_rel + coriolis_2 + centripetal_2 - g_body);

    Vec3 reaction_force = -(f1 + f2);

    // Torque: reaction torque from joint axes plus moment of reaction forces
    // Joint rotation axes in body frame
    Vec3 axis1(0, 0, 1);  // azimuth: body z-axis
    Mat3 R1 = joint1_rotation(q(0));
    Vec3 axis2 = R1 * Vec3(0, 1, 0);  // elevation: rotated y-axis

    // Reaction torque = -J_omega^T * tau_manip_effective
    // Plus moments from reaction forces about body COM
    Vec3 tau_inertial_1 = r_c1.cross(f1);
    Vec3 tau_inertial_2 = r_c2.cross(f2);

    // Angular momentum rate of links about their own COM
    // H_dot_i = I_i * alpha_i + omega_i × (I_i * omega_i)
    Mat3 R_link = link1_orientation(q);
    Mat3 I1_body = R_link * links_[0].inertia() * R_link.transpose();
    Mat3 I2_body = R_link * links_[1].inertia() * R_link.transpose();

    Vec3 omega_link1 = body_ang_vel + q_dot(0) * axis1 + q_dot(1) * axis2;
    Vec3 omega_link2 = omega_link1;  // same for rigid 2-DOF azimuth-elevation

    Vec3 alpha_rel_1 = q_ddot(0) * axis1 + q_ddot(1) * axis2
                     + q_dot(0) * q_dot(1) * axis1.cross(axis2);
    Vec3 H_dot_1 = I1_body * alpha_rel_1 + omega_link1.cross(I1_body * omega_link1);
    Vec3 H_dot_2 = I2_body * alpha_rel_1 + omega_link2.cross(I2_body * omega_link2);

    Vec3 reaction_torque = -(tau_inertial_1 + tau_inertial_2 + H_dot_1 + H_dot_2);

    Eigen::Matrix<double, 6, 1> wrench;
    wrench.head<3>() = reaction_force;
    wrench.tail<3>() = reaction_torque;
    return wrench;
}

}  // namespace aerial_manipulator
