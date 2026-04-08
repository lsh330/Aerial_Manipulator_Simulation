"""Hierarchical SDRE controller for the aerial manipulator.

Architecture:
    Outer loop: Position PID → F_des, R_des (unchanged, handles underactuation)
    Inner loop: SDRE on attitude(3) + joints(2) = 5-DOF fully-actuated subsystem
                Solves 10×10 CARE at each step with state-dependent M_rr, C, G

The inner SDRE automatically adapts gains to the current arm configuration,
accounting for coupled inertia M_rr(q_joint), cross-coupling M_rm, and
configuration-dependent Coriolis/gravity effects.

Reference:
    T. Cimen, "SDRE Control: A Survey," IFAC Proc., 2008.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from models.state import State, IDX


class SDREController:
    """Hierarchical Position-PID + Inner-SDRE controller.

    The outer position PID loop computes desired thrust and attitude R_des.
    The inner SDRE loop optimally tracks R_des and joint references by
    solving CARE on the coupled attitude+joint subsystem at each time step.
    """

    def __init__(self, system_wrapper,
                 Q_inner: np.ndarray = None, R_inner: np.ndarray = None,
                 pos_Kp=None, pos_Kd=None, pos_Ki=None):
        self._system = system_wrapper
        self._gravity = system_wrapper._env_cfg.gravity
        self._total_mass = system_wrapper.total_mass()

        # Position PID gains (outer loop, unchanged)
        self._pos_Kp = np.asarray(pos_Kp if pos_Kp is not None else [6.0, 6.0, 8.0])
        self._pos_Kd = np.asarray(pos_Kd if pos_Kd is not None else [4.5, 4.5, 5.6])
        self._pos_Ki = np.asarray(pos_Ki if pos_Ki is not None else [0.1, 0.1, 0.2])
        self._pos_integral = np.zeros(3)
        self._int_limit = 2.0

        # Inner SDRE: 5-DOF (roll, pitch, yaw, q1, q2), 5 inputs (τ_roll, τ_pitch, τ_yaw, τ_q1, τ_q2)
        # State: [e_att(3), e_joint(2), e_omega(3), e_joint_vel(2)] = 10
        if Q_inner is None:
            Q_inner = np.diag([
                200, 200, 67,      # attitude error (roll, pitch, yaw)
                3000, 3000,        # joint error — high for precise tracking
                2, 2, 1,           # angular velocity error
                5, 5,              # joint velocity error
            ])
        self._Q = Q_inner

        if R_inner is None:
            R_inner = np.diag([
                0.5, 0.5, 0.5,     # body torques
                0.05, 0.05,        # joint torques — low for aggressive tracking
            ])
        self._R = R_inner

        self._last_K = None
        self._mixing_inv = self._build_mixing_inv()

    def _build_mixing_inv(self):
        """Inverse mixing matrix for control allocation."""
        L = self._system._quad_cfg.arm_length
        k = self._system._quad_cfg.torque_coeff / self._system._quad_cfg.thrust_coeff
        mixing = np.array([
            [1, 1, 1, 1],
            [0, -L, 0, L],
            [L, 0, -L, 0],
            [k, -k, k, -k],
        ])
        return np.linalg.pinv(mixing)

    def compute_control(self, state: np.ndarray, reference: dict,
                        dt: float = 0.001) -> np.ndarray:
        s = State(state)

        # ═══ Outer Loop: Position PID → F_des, R_des ═══
        pos_ref = np.asarray(reference.get("position", s.position))
        vel_ref = np.asarray(reference.get("velocity", np.zeros(3)))
        acc_ff = np.asarray(reference.get("acceleration", np.zeros(3)))
        yaw_des = reference.get("yaw", 0.0)

        e_pos = pos_ref - s.position
        e_vel = vel_ref - s.velocity

        self._pos_integral += e_pos * dt
        self._pos_integral = np.clip(self._pos_integral, -self._int_limit, self._int_limit)

        a_des = (self._pos_Kp * e_pos + self._pos_Kd * e_vel
                 + self._pos_Ki * self._pos_integral + acc_ff)

        F_des = self._total_mass * (a_des + np.array([0, 0, self._gravity]))
        thrust_total = np.linalg.norm(F_des)
        if thrust_total < 1e-6:
            thrust_total = self._total_mass * self._gravity

        # Extract desired rotation from thrust direction
        z_des = F_des / np.linalg.norm(F_des)
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])
        y_des = np.cross(z_des, x_c)
        y_norm = np.linalg.norm(y_des)
        if y_norm < 1e-6:
            y_des = np.array([0, 1, 0])
        else:
            y_des /= y_norm
        x_des = np.cross(y_des, z_des)
        R_des = np.column_stack([x_des, y_des, z_des])

        # ═══ Inner Loop: SDRE on attitude + joints ═══
        joint_ref = np.asarray(reference.get("joint_positions", np.zeros(2)))
        joint_vel_ref = np.asarray(reference.get("joint_velocities", np.zeros(2)))

        # State error for inner loop (10-element)
        R = s.rotation_matrix()
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = np.array([e_R_mat[2,1], e_R_mat[0,2], e_R_mat[1,0]])  # vee map
        e_joint = s.joint_positions - joint_ref
        e_omega = s.angular_velocity  # desired omega ≈ 0
        e_joint_vel = s.joint_velocities - joint_vel_ref

        dx_inner = np.concatenate([e_R, e_joint, e_omega, e_joint_vel])

        # Build inner-loop SDC: A_inner(x), B_inner(x)
        A_inner, B_inner = self._build_inner_sdc(state)

        # Solve CARE
        try:
            P = solve_continuous_are(A_inner, B_inner, self._Q, self._R)
            K = np.linalg.solve(self._R, B_inner.T @ P)
            self._last_K = K
        except Exception:
            if self._last_K is not None:
                K = self._last_K
            else:
                K = np.zeros((5, 10))

        # Inner control: [τ_body(3), τ_joint(2)]
        tau_inner = -K @ dx_inner

        # Add gyroscopic compensation: ω × Jω
        J = self._system._quad_cfg.inertia
        gyro = np.cross(s.angular_velocity, J @ s.angular_velocity)
        tau_body = tau_inner[:3] + gyro
        tau_joint = tau_inner[3:]

        # ═══ Control Allocation ═══
        wrench = np.array([thrust_total, tau_body[0], tau_body[1], tau_body[2]])
        motor_thrusts = self._mixing_inv @ wrench
        motor_thrusts = np.clip(motor_thrusts, 0.0, 12.3)

        tau_joint = np.clip(tau_joint, -5.0, 5.0)

        return np.concatenate([motor_thrusts, tau_joint])

    def _build_inner_sdc(self, state: np.ndarray):
        """Build 10×10 A and 10×5 B for the attitude+joint subsystem.

        Inner state: [e_att(3), e_joint(2), e_omega(3), e_joint_vel(2)]
        Inner input: [τ_body(3), τ_joint(2)]

        From M_inner * q̈_inner + C_inner * q̇_inner + G_inner = τ_inner:
            A = [[0, I], [-M⁻¹*G_q, -M⁻¹*C]]
            B = [[0], [M⁻¹]]
        """
        sys_obj = self._system._system

        # Extract the 5×5 inner mass matrix from the full 8×8 M
        M_full = np.array(sys_obj.compute_mass_matrix(state))
        # Inner DOFs: indices 3,4,5 (attitude), 6,7 (joints) in the 8-DOF system
        inner_idx = [3, 4, 5, 6, 7]
        M_inner = M_full[np.ix_(inner_idx, inner_idx)]  # 5×5
        M_inner_inv = np.linalg.inv(M_inner)

        # Approximate inner Coriolis: extract from full coriolis vector
        # C_inner ≈ diagonal damping from the coriolis contribution
        q_dot_inner = np.concatenate([
            state[IDX.ANG_VEL], state[IDX.JOINT_VEL]
        ])

        # Column-by-column C matrix extraction for inner DOFs
        eps_c = 1e-5
        C_inner = np.zeros((5, 5))
        for k in range(5):
            state_pert = state.copy()
            # Zero all velocities first
            state_pert[IDX.VEL] = np.zeros(3)
            state_pert[IDX.ANG_VEL] = np.zeros(3)
            state_pert[IDX.JOINT_VEL] = np.zeros(2)
            # Set one inner velocity to eps
            if k < 3:
                state_pert[IDX.ANG_VEL.start + k] = eps_c
            else:
                state_pert[IDX.JOINT_VEL.start + (k-3)] = eps_c

            C_qdot_full = np.array(sys_obj.compute_coriolis_vector(state_pert))
            C_inner[:, k] = C_qdot_full[inner_idx] / eps_c

        # Gravity stiffness for inner DOFs (dG_inner/dq_inner)
        G_full_0 = np.array(sys_obj.compute_gravity_vector(state))
        G_inner_0 = G_full_0[inner_idx]

        eps_g = 1e-6
        G_q_inner = np.zeros((5, 5))
        for k in range(5):
            state_pert = state.copy()
            if k < 3:
                # Euler perturbation via quaternion
                ax = k
                axis = np.zeros(3); axis[ax] = 1.0
                from scipy.spatial.transform import Rotation
                q_orig = state[IDX.QUAT]
                R_orig = Rotation.from_quat([q_orig[1], q_orig[2], q_orig[3], q_orig[0]])
                R_pert = R_orig * Rotation.from_rotvec(eps_g * axis)
                q_p = R_pert.as_quat()
                state_pert[IDX.QUAT] = np.array([q_p[3], q_p[0], q_p[1], q_p[2]])
            else:
                state_pert[13 + (k-3)] += eps_g

            G_pert = np.array(sys_obj.compute_gravity_vector(state_pert))
            G_q_inner[:, k] = (G_pert[inner_idx] - G_inner_0) / eps_g

        # Build 10×10 A and 10×5 B
        n = 5
        A = np.zeros((2*n, 2*n))
        A[:n, n:] = np.eye(n)
        A[n:, :n] = -M_inner_inv @ G_q_inner
        A[n:, n:] = -M_inner_inv @ C_inner

        B_sdre = np.zeros((2*n, n))
        B_sdre[n:, :] = M_inner_inv

        return A, B_sdre

    @property
    def last_gain(self):
        return self._last_K
