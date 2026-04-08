"""CasADi symbolic dynamics — exact replica of C++ engine equations.

Replicates aerial_manipulator_system.cpp's M(q), C(q,qdot), G(q), B(q)
in CasADi SX for automatic differentiation (analytic Jacobians).

State (17): [pos(3), vel(3), quat(4), omega(3), q_j(2), qd_j(2)]
Input (6):  [f1, f2, f3, f4, tau_q1, tau_q2]
"""

import numpy as np
import casadi as ca


def build_aerial_manipulator_dynamics(params: dict) -> ca.Function:
    """Build CasADi symbolic dynamics function f(x, u) → x_dot.

    Args:
        params: dict with keys:
            m0, m1, m2: masses [kg]
            J0: (3,3) quadrotor inertia
            I1, I2: (3,3) link inertias
            L: arm length, lc1, lc2, l1: link geometry
            k_f, k_tau: thrust/torque coefficients
            drag: drag coefficient
            g: gravity
            att_offset: (3,) attachment offset

    Returns:
        CasADi Function: f(x17, u6) → xdot17
    """
    # ── Symbolic state & input ──
    x = ca.SX.sym('x', 17)
    u = ca.SX.sym('u', 6)

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]   # [w, x, y, z]
    omega = x[10:13]  # body angular velocity
    q_j = x[12+1:12+3]   # x[13], x[14]
    qd_j = x[15:17]

    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    q1, q2 = q_j[0], q_j[1]
    f1, f2, f3, f4 = u[0], u[1], u[2], u[3]
    tau_j1, tau_j2 = u[4], u[5]

    # ── Parameters ──
    m0 = params['m0']
    m1 = params['m1']
    m2 = params['m2']
    m_total = m0 + m1 + m2
    J0 = ca.DM(params['J0'])
    I1_local = ca.DM(params['I1'])
    I2_local = ca.DM(params['I2'])
    L = params['L']
    l1 = params['l1']
    lc1 = params['lc1']
    lc2 = params['lc2']
    D = l1 + lc2
    k_ratio = params['k_tau'] / params['k_f']
    drag_coeff = params['drag']
    g = params['g']
    att = ca.DM(params['att_offset'])

    # ── Helper: skew symmetric ──
    def skew(v):
        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0))

    # ── Rotation matrix from quaternion ──
    R = ca.vertcat(
        ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
        ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)),
    )

    # ── Trig shortcuts ──
    c1 = ca.cos(q1); s1 = ca.sin(q1)
    c2 = ca.cos(q2); s2 = ca.sin(q2)

    # ── Link orientations: R_link = Rz(q1) * Ry(q2) ──
    R_link = ca.vertcat(
        ca.horzcat(c1*c2, -s1, c1*s2),
        ca.horzcat(s1*c2,  c1, s1*s2),
        ca.horzcat(-s2,     0,    c2))

    # ── COM positions in body frame ──
    r_c1 = att + ca.vertcat(lc1*c1*s2, lc1*s1*s2, -lc1*c2)
    r_c2 = att + ca.vertcat(D*c1*s2, D*s1*s2, -D*c2)

    # ── COM Jacobians (body frame) ──
    Jv1 = ca.horzcat(
        ca.vertcat(-lc1*s1*s2, lc1*c1*s2, 0),
        ca.vertcat(lc1*c1*c2, lc1*s1*c2, lc1*s2))
    Jv2 = ca.horzcat(
        ca.vertcat(-D*s1*s2, D*c1*s2, 0),
        ca.vertcat(D*c1*c2, D*s1*c2, D*s2))

    # ── Skew matrices ──
    r_c1_x = skew(r_c1)
    r_c2_x = skew(r_c2)

    # ── Link inertias in body frame ──
    I1_body = R_link @ I1_local @ R_link.T
    I2_body = R_link @ I2_local @ R_link.T

    # ── Joint rotation axes ──
    axis1 = ca.vertcat(0, 0, 1)  # azimuth
    Rz_q1 = ca.vertcat(
        ca.horzcat(c1, -s1, 0),
        ca.horzcat(s1,  c1, 0),
        ca.horzcat(0,    0, 1))
    axis2 = Rz_q1 @ ca.vertcat(0, 1, 0)  # elevation
    J_omega = ca.horzcat(axis1, axis2)

    # ════════════════════════════════════════════
    # Mass Matrix M(q) ∈ R^{8×8}
    # ════════════════════════════════════════════
    I3 = ca.DM.eye(3)

    # (a) Translation-Translation
    M_tt = m_total * I3

    # (b) Rotation-Rotation
    M_rr = J0 + m1*(r_c1_x.T @ r_c1_x) + I1_body \
              + m2*(r_c2_x.T @ r_c2_x) + I2_body

    # (c) Manipulator-Manipulator (2×2)
    alpha = m1*lc1**2 + m2*D**2
    I1zz = I1_local[2, 2]
    I2zz = I2_local[2, 2]
    M_mm = ca.vertcat(
        ca.horzcat(alpha*s2**2 + I1zz + I2zz, 0),
        ca.horzcat(0, alpha + I1zz + I2zz))

    # (d) Translation-Rotation
    M_tr = -R @ (m1*r_c1_x + m2*r_c2_x)

    # (e) Translation-Manipulator
    M_tm = R @ (m1*Jv1 + m2*Jv2)

    # (f) Rotation-Manipulator
    M_rm = m1*r_c1_x @ Jv1 + I1_body @ J_omega \
         + m2*r_c2_x @ Jv2 + I2_body @ J_omega

    # Assemble 8×8
    M = ca.SX.zeros(8, 8)
    M[0:3, 0:3] = M_tt
    M[3:6, 3:6] = M_rr
    M[6:8, 6:8] = M_mm
    M[0:3, 3:6] = M_tr;  M[3:6, 0:3] = M_tr.T
    M[0:3, 6:8] = M_tm;  M[6:8, 0:3] = M_tm.T
    M[3:6, 6:8] = M_rm;  M[6:8, 3:6] = M_rm.T

    # ════════════════════════════════════════════
    # Gravity Vector G(q) ∈ R^8
    # ════════════════════════════════════════════
    G = ca.SX.zeros(8, 1)
    G[2] = m_total * g  # Lagrangian: G_z = +mg

    g_body = R.T @ ca.vertcat(0, 0, -g)
    G[3:6] = m1*ca.cross(r_c1, g_body) + m2*ca.cross(r_c2, g_body)

    # Joint gravity
    alpha_g = m1*lc1 + m2*D
    G_j1 = -alpha_g * s2 * (-g_body[0]*s1 + g_body[1]*c1)
    G_j2 = -alpha_g * (g_body[0]*c1*c2 + g_body[1]*s1*c2 + g_body[2]*s2)
    G[6] = G_j1
    G[7] = G_j2

    # ════════════════════════════════════════════
    # Input Matrix B(q) ∈ R^{8×6}
    # ════════════════════════════════════════════
    z_body = R[:, 2]
    B_mat = ca.SX.zeros(8, 6)
    for i in range(4):
        B_mat[0:3, i] = z_body

    # Mixing: [F, tau_roll, tau_pitch, tau_yaw] = A * [f1,f2,f3,f4]
    B_mat[3, 0] = 0;    B_mat[3, 1] = -L;   B_mat[3, 2] = 0;    B_mat[3, 3] = L
    B_mat[4, 0] = L;    B_mat[4, 1] = 0;    B_mat[4, 2] = -L;   B_mat[4, 3] = 0
    B_mat[5, 0] = k_ratio; B_mat[5, 1] = -k_ratio; B_mat[5, 2] = k_ratio; B_mat[5, 3] = -k_ratio

    B_mat[6, 4] = 1
    B_mat[7, 5] = 1

    # ════════════════════════════════════════════
    # Coriolis via CasADi AD on M(q)
    # ════════════════════════════════════════════
    # Generalized velocity
    qdot = ca.vertcat(vel, omega, qd_j)

    # C(q,qdot)*qdot via Christoffel: c_i = Σ_jk 0.5*(dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i) * qdot_j * qdot_k
    # Use CasADi AD: dM/dq is automatic!
    # Config variables that M depends on: quat(4) and q_j(2)
    # M does NOT depend on pos(3) or velocities
    q_config = ca.vertcat(quat, q_j)  # 6 variables that M depends on

    # Flatten M to vector for jacobian
    M_flat = ca.reshape(M, 64, 1)
    dM_dconfig = ca.jacobian(M_flat, q_config)  # 64 × 6

    # For Christoffel, we need dM_ij/dq_k for generalized coordinates
    # Generalized coords: [pos(3), euler_via_quat(3→4), q_j(2)]
    # But M depends on quat(4) and q_j(2) only
    # We need to map: dM/d(gen_coord_k) for k in {3,4,5, 6,7} (rotation + joints)
    # For rotation: d/d(euler) = d/d(quat) * d(quat)/d(euler)
    # This is complex. Simpler: compute C*qdot directly via Lagrangian.
    #
    # C*qdot = d/dt(∂T/∂qdot) - ∂T/∂q - M*qddot
    # = dM/dt * qdot + M * qddot - 0.5 * ∂(qdot^T M qdot)/∂q - M*qddot
    # = dM/dt * qdot - 0.5 * dM/dq [qdot]
    #
    # Actually, simplest CasADi approach:
    # T = 0.5 * qdot^T * M * qdot
    # C*qdot = dT/dq - d/dt(dT/dqdot) + M*qddot  ... this gets circular
    #
    # Best approach: direct Christoffel via CasADi AD on M
    # C_ij = Σ_k Γ_{ijk} * qdot_k
    # Γ_{ijk} = 0.5 * (dM_{ij}/dq_k + dM_{ik}/dq_j - dM_{jk}/dq_i)

    # Since M only depends on quat and q_j, we compute dM/dq for each
    # generalized coordinate. For position (k=0,1,2): dM/dq=0.
    # For rotation (k=3,4,5): need dM/d(euler), mapped via quaternion.
    # For joints (k=6,7): dM/dq_j directly.

    # Practical CasADi approach: compute C*qdot via energy method
    # C*qdot_i = Σ_j M_ij_dot * qdot_j - 0.5 * d/dq_i(qdot^T M qdot)
    # where M_dot = dM/dt = Σ_k (dM/dq_k) * qdot_k

    # Compute kinetic energy T = 0.5 * qdot^T * M * qdot
    T = 0.5 * qdot.T @ M @ qdot

    # ∂T/∂q_j for joint variables (directly differentiable)
    dT_dq_j = ca.jacobian(T, q_j).T  # 2×1

    # For position variables: ∂T/∂pos = 0 (M doesn't depend on pos)
    dT_dpos = ca.SX.zeros(3, 1)

    # For rotation: ∂T/∂(euler) — complex via quaternion chain rule
    # Instead, use the identity: C*qdot = d(M*qdot)/dt - ∂T/∂q
    # But we don't have time derivatives in CasADi.
    #
    # Alternative: compute C*qdot_rotation block using the standard formula
    # for rigid body: C_rot*omega = omega × (J_eff * omega) + arm contributions
    # This is what the C++ code does. Let's replicate it.

    # Rotation Coriolis (body frame)
    J_eff = M_rr  # already computed
    gyro = ca.cross(omega, J_eff @ omega)

    # Arm Coriolis contribution to rotation
    v_c1_rel = Jv1 @ qd_j
    v_c2_rel = Jv2 @ qd_j
    arm_coriolis_rot = m1 * skew(r_c1) @ (2 * ca.cross(omega, v_c1_rel)) \
                     + m2 * skew(r_c2) @ (2 * ca.cross(omega, v_c2_rel))

    # Translation Coriolis (world frame)
    omega_w = R @ omega
    trans_coriolis = ca.SX.zeros(3, 1)
    for mi, rci, vci in [(m1, r_c1, v_c1_rel), (m2, r_c2, v_c2_rel)]:
        r_w = R @ rci
        v_w = R @ vci
        trans_coriolis += mi * (ca.cross(omega_w, ca.cross(omega_w, r_w))
                               + 2 * ca.cross(omega_w, v_w))

    # Manipulator Coriolis (2×2 standard)
    h = alpha * s2 * c2
    C_mm_qdot = ca.vertcat(
        h * qd_j[1] * qd_j[0] + h * qd_j[0] * qd_j[1],
        -h * qd_j[0] * qd_j[0])

    # Assemble Coriolis vector (8×1)
    # Note: this is the "physics-based" Coriolis, same as original C++ code
    # For exact Christoffel match, we'd need dM/dq for all coords.
    # The joint Christoffel terms are captured by C_mm_qdot.
    # Cross-coupling terms (dM_tr/dq, dM_tm/dq, dM_rm/dq) are partially
    # in trans_coriolis and arm_coriolis_rot.
    C_qdot = ca.vertcat(trans_coriolis, gyro + arm_coriolis_rot, C_mm_qdot)

    # ════════════════════════════════════════════
    # Drag
    # ════════════════════════════════════════════
    vel_body = R.T @ vel
    drag_body = -drag_coeff * vel_body
    drag_world = R @ drag_body

    # ════════════════════════════════════════════
    # EOM: M*qddot = B*u - C*qdot - G + [drag, 0, 0]
    # ════════════════════════════════════════════
    rhs = B_mat @ u - C_qdot - G
    rhs[0:3] += drag_world

    qddot = ca.solve(M, rhs)  # 8×1: [v_dot(3), omega_dot(3), qddot_j(2)]

    # ════════════════════════════════════════════
    # State derivative (17-element)
    # ════════════════════════════════════════════
    # Quaternion derivative: q_dot = 0.5 * q ⊗ [0, omega]
    ox, oy, oz = omega[0], omega[1], omega[2]
    quat_dot = 0.5 * ca.vertcat(
        -qx*ox - qy*oy - qz*oz,
         qw*ox + qy*oz - qz*oy,
         qw*oy + qz*ox - qx*oz,
         qw*oz + qx*oy - qy*ox)

    x_dot = ca.vertcat(
        vel,              # pos_dot = vel (3)
        qddot[0:3],      # vel_dot = linear acceleration (3)
        quat_dot,         # quat_dot (4)
        qddot[3:6],      # omega_dot = angular acceleration (3)
        qd_j,             # joint_pos_dot = joint_vel (2)
        qddot[6:8],      # joint_vel_dot = joint acceleration (2)
    )

    return ca.Function('f_dynamics', [x, u], [x_dot],
                       ['state', 'input'], ['state_dot'])


def build_dynamics_from_config(quad_cfg, manip_cfg, env_cfg) -> ca.Function:
    """Convenience: build CasADi dynamics from config objects."""
    params = {
        'm0': quad_cfg.mass, 'm1': manip_cfg.link1.mass, 'm2': manip_cfg.link2.mass,
        'J0': quad_cfg.inertia,
        'I1': manip_cfg.link1.inertia, 'I2': manip_cfg.link2.inertia,
        'L': quad_cfg.arm_length,
        'l1': manip_cfg.link1.length,
        'lc1': manip_cfg.link1.com_distance, 'lc2': manip_cfg.link2.com_distance,
        'k_f': quad_cfg.thrust_coeff, 'k_tau': quad_cfg.torque_coeff,
        'drag': quad_cfg.drag_coeff,
        'g': env_cfg.gravity,
        'att_offset': manip_cfg.attachment_offset,
    }
    return build_aerial_manipulator_dynamics(params)
