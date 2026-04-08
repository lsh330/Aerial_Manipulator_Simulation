"""CasADi symbolic dynamics -- exact replica of C++ engine equations.

Replicates aerial_manipulator_system.cpp M(q), C(q,qdot), G(q), B(q)
in CasADi SX for automatic differentiation (analytic Jacobians).

State (17): [pos(3), vel(3), quat(4), omega(3), q_j(2), qd_j(2)]
Input (6):  [f1, f2, f3, f4, tau_q1, tau_q2]

Coriolis is computed via the exact Christoffel-based energy method using
CasADi automatic differentiation on M(quat, q_j).
"""

import numpy as np
import casadi as ca


def build_aerial_manipulator_dynamics(params: dict) -> ca.Function:
    """Build CasADi symbolic dynamics function f(x, u) -> x_dot.

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
            wind_velocity: (3,) wind velocity in world frame (optional, default [0,0,0])
            contact_stiffness: spring stiffness [N/m] (optional, default 0)
            contact_damping: damping [N*s/m] (optional, default 0)
            contact_surface_z: z of contact surface [m] (optional, default 0)
            l2_length: length of link 2 (optional, used for contact EE position)

    Returns:
        CasADi Function: f(x17, u6) -> xdot17
    """
    # -- Symbolic state & input --
    x = ca.SX.sym('x', 17)
    u = ca.SX.sym('u', 6)

    pos = x[0:3]
    vel = x[3:6]
    quat = x[6:10]    # [w, x, y, z]
    omega = x[10:13]  # body angular velocity
    q_j = x[12+1:12+3]   # x[13], x[14]
    qd_j = x[15:17]

    qw, qx, qy, qz = quat[0], quat[1], quat[2], quat[3]
    q1, q2 = q_j[0], q_j[1]
    f1, f2, f3, f4 = u[0], u[1], u[2], u[3]
    tau_j1, tau_j2 = u[4], u[5]

    # -- Parameters --
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

    # -- Helper: skew symmetric --
    def skew(v):
        return ca.vertcat(
            ca.horzcat(0, -v[2], v[1]),
            ca.horzcat(v[2], 0, -v[0]),
            ca.horzcat(-v[1], v[0], 0))

    # -- Rotation matrix from quaternion --
    R = ca.vertcat(
        ca.horzcat(1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)),
        ca.horzcat(2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)),
        ca.horzcat(2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)),
    )

    # -- Trig shortcuts --
    c1 = ca.cos(q1); s1 = ca.sin(q1)
    c2 = ca.cos(q2); s2 = ca.sin(q2)

    # -- Link orientations: R_link = Rz(q1) * Ry(q2) --
    R_link = ca.vertcat(
        ca.horzcat(c1*c2, -s1, c1*s2),
        ca.horzcat(s1*c2,  c1, s1*s2),
        ca.horzcat(-s2,     0,    c2))

    # -- COM positions in body frame --
    r_c1 = att + ca.vertcat(lc1*c1*s2, lc1*s1*s2, -lc1*c2)
    r_c2 = att + ca.vertcat(D*c1*s2, D*s1*s2, -D*c2)

    # -- COM Jacobians (body frame, joint columns only) --
    Jv1 = ca.horzcat(
        ca.vertcat(-lc1*s1*s2, lc1*c1*s2, 0),
        ca.vertcat(lc1*c1*c2, lc1*s1*c2, lc1*s2))
    Jv2 = ca.horzcat(
        ca.vertcat(-D*s1*s2, D*c1*s2, 0),
        ca.vertcat(D*c1*c2, D*s1*c2, D*s2))

    # -- Skew matrices --
    r_c1_x = skew(r_c1)
    r_c2_x = skew(r_c2)

    # -- Link inertias in body frame --
    I1_body = R_link @ I1_local @ R_link.T
    I2_body = R_link @ I2_local @ R_link.T

    # -- Joint rotation axes --
    axis1 = ca.vertcat(0, 0, 1)  # azimuth
    Rz_q1 = ca.vertcat(
        ca.horzcat(c1, -s1, 0),
        ca.horzcat(s1,  c1, 0),
        ca.horzcat(0,    0, 1))
    axis2 = Rz_q1 @ ca.vertcat(0, 1, 0)  # elevation
    J_omega = ca.horzcat(axis1, axis2)
    # ============================================================
    # Mass Matrix M(q) in R^{8x8}
    # ============================================================
    I3 = ca.DM.eye(3)

    # (a) Translation-Translation
    M_tt = m_total * I3

    # (b) Rotation-Rotation
    M_rr = J0 + m1*(r_c1_x.T @ r_c1_x) + I1_body \
              + m2*(r_c2_x.T @ r_c2_x) + I2_body

    # (c) Manipulator-Manipulator (2x2)
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

    # Assemble 8x8
    M = ca.SX.zeros(8, 8)
    M[0:3, 0:3] = M_tt
    M[3:6, 3:6] = M_rr
    M[6:8, 6:8] = M_mm
    M[0:3, 3:6] = M_tr;  M[3:6, 0:3] = M_tr.T
    M[0:3, 6:8] = M_tm;  M[6:8, 0:3] = M_tm.T
    M[3:6, 6:8] = M_rm;  M[6:8, 3:6] = M_rm.T

    # ============================================================
    # Gravity Vector G(q) in R^8
    # ============================================================
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

    # ============================================================
    # Input Matrix B(q) in R^{8x6}
    # ============================================================
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

    # ============================================================
    # Coriolis C(q, qdot)*qdot via exact Christoffel-based energy method
    #
    # Identity:
    #   C(q,qdot)*qdot = dM/dt * qdot - dT/dq
    #   where  T = 0.5 * qdot^T * M(q) * qdot  (kinetic energy scalar)
    #   and    dM/dt = sum_k (dM/dconfig_k) * config_dot_k
    #
    # M depends only on q_config = [quat(4), q_j(2)], not on pos(3).
    #
    # config_dot:
    #   quat_dot = Q_mat(quat) @ omega,   Q_mat in R^{4x3}
    #   qd_j     = qd_j
    #
    # dT/dq for each generalized coordinate block:
    #   pos  (3): 0  (M is independent of pos)
    #   euler(3): dT/d_euler = dT/d_quat @ Q_mat  (quaternion chain rule)
    #   q_j  (2): ca.jacobian(T, q_j)  (direct CasADi AD)
    # ============================================================
    qdot = ca.vertcat(vel, omega, qd_j)  # 8x1 generalized velocity

    # Kinetic energy scalar
    T_kin = 0.5 * (qdot.T @ M @ qdot)

    # Quaternion kinematic map: quat_dot = Q_mat @ omega,  Q_mat in R^{4x3}
    # Derived from q_dot = 0.5 * q (x) [0, omega]
    Q_mat = 0.5 * ca.vertcat(
        ca.horzcat(-qx, -qy, -qz),
        ca.horzcat( qw, -qz,  qy),
        ca.horzcat( qz,  qw, -qx),
        ca.horzcat(-qy,  qx,  qw))  # 4x3

    # Config vector and its time derivative
    q_config = ca.vertcat(quat, q_j)              # 6x1
    config_dot = ca.vertcat(Q_mat @ omega, qd_j)  # 6x1: [quat_dot, qd_j]

    # dM/dt via chain rule using CasADi jtimes (forward-mode AD):
    #   (dM_flat/dt)_i = sum_k (dM_flat_i / dconfig_k) * config_dot_k
    M_flat = ca.reshape(M, 64, 1)
    M_dot_flat = ca.jtimes(M_flat, q_config, config_dot)  # 64x1
    M_dot = ca.reshape(M_dot_flat, 8, 8)                   # 8x8

    # dT/dq -- gradient of kinetic energy w.r.t. generalized coordinates
    # pos block (3x1): zero because M does not depend on pos
    dT_dpos = ca.SX.zeros(3, 1)

    # rotation block (3x1): dT/d(euler) = dT/d(quat) @ Q_mat, via chain rule
    # A virtual rotation delta(euler_k) moves quat by Q_mat[:,k]*delta(euler_k)
    dT_dquat = ca.jacobian(T_kin, quat)   # 1x4
    dT_d_euler = (dT_dquat @ Q_mat).T     # 3x1

    # joint block (2x1): direct differentiation
    dT_dq_j = ca.jacobian(T_kin, q_j).T  # 2x1

    # Full dT/dq in generalized coordinate order [pos, rotation, q_j]
    dT_dq = ca.vertcat(dT_dpos, dT_d_euler, dT_dq_j)  # 8x1

    # Exact Christoffel Coriolis vector: C(q,qdot)*qdot = dM/dt*qdot - dT/dq
    C_qdot = M_dot @ qdot - dT_dq  # 8x1

    # ============================================================
    # Drag (with optional wind velocity)
    # ============================================================
    wind = ca.DM(params.get('wind_velocity', [0.0, 0.0, 0.0]))
    vel_rel = vel - wind       # relative velocity w.r.t. wind [m/s]
    vel_body = R.T @ vel_rel
    drag_body = -drag_coeff * vel_body
    drag_world = R @ drag_body

    # ============================================================
    # EOM: M*qddot = B*u - C*qdot - G + [drag, 0, 0]
    # ============================================================
    rhs = B_mat @ u - C_qdot - G
    rhs[0:3] += drag_world

    # ============================================================
    # Contact dynamics (optional spring-damper at end-effector)
    #
    # Active when end-effector z < contact_surface_z.
    # Normal force: f_c = k * max(0, penetration) - b * z_ee_dot
    # Applied in world +z direction via full geometric Jacobian transpose
    # so that reaction torques are correctly distributed to all DOFs.
    # ============================================================
    contact_stiffness = params.get('contact_stiffness', 0.0)
    contact_damping   = params.get('contact_damping',   0.0)
    contact_surface_z = params.get('contact_surface_z', 0.0)

    if contact_stiffness > 0:
        l2_length = params.get('l2_length', lc2)

        # End-effector position in body frame (tip of link 2)
        r_ee_body = att + ca.vertcat(
            (l1 + l2_length) * c1 * s2,
            (l1 + l2_length) * s1 * s2,
            -(l1 + l2_length) * c2)

        # End-effector position in world frame
        r_ee_world = pos + R @ r_ee_body

        # Penetration depth (positive when end-effector is below surface)
        penetration = contact_surface_z - r_ee_world[2]
        contact_active = ca.fmax(0.0, penetration)  # one-sided constraint

        # End-effector velocity in world frame:
        # r_ee_world_dot = vel + R*(omega x r_ee_body) + R*J_ee_body*qd_j
        J_ee_body = ca.horzcat(
            ca.vertcat(-(l1 + l2_length)*s1*s2,
                        (l1 + l2_length)*c1*s2,
                        0.0),
            ca.vertcat( (l1 + l2_length)*c1*c2,
                        (l1 + l2_length)*s1*c2,
                        (l1 + l2_length)*s2))  # 3x2

        r_ee_dot_world = vel \
            + R @ ca.cross(omega, r_ee_body) \
            + R @ (J_ee_body @ qd_j)

        z_ee_dot = r_ee_dot_world[2]

        # Spring-damper contact force (world +z, compression only)
        # Damping gate uses sign of contact_active to be zero outside contact
        f_contact_z = contact_stiffness * contact_active \
                    - contact_damping * z_ee_dot * ca.sign(contact_active + 1e-9)

        # Geometric Jacobian of EE translational velocity w.r.t. qdot (3x8)
        # Columns: [I3 | -R*skew(r_ee_body) | R*J_ee_body]
        J_ee_geom = ca.horzcat(
            I3,
            -R @ skew(r_ee_body),
            R @ J_ee_body)  # 3x8

        # Contact force vector (world frame, z-only)
        f_contact_world = ca.vertcat(0.0, 0.0, f_contact_z)

        # Generalized forces from contact via Jacobian transpose: tau = J^T * f
        tau_contact = J_ee_geom.T @ f_contact_world  # 8x1
        rhs += tau_contact

    qddot = ca.solve(M, rhs)  # 8x1: [v_dot(3), omega_dot(3), qddot_j(2)]

    # ============================================================
    # State derivative (17-element)
    # ============================================================
    # Quaternion derivative: q_dot = 0.5 * q (x) [0, omega]
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
        # Environment extensions (all optional; fall back to neutral defaults)
        'wind_velocity':     getattr(env_cfg, 'wind_velocity',     [0.0, 0.0, 0.0]),
        'contact_stiffness': getattr(env_cfg, 'contact_stiffness', 0.0),
        'contact_damping':   getattr(env_cfg, 'contact_damping',   0.0),
        'contact_surface_z': getattr(env_cfg, 'contact_surface_z', 0.0),
        'l2_length':         manip_cfg.link2.length,
    }
    return build_aerial_manipulator_dynamics(params)
