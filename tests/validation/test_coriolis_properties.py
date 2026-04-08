"""Validate Coriolis matrix properties: M_dot - 2C must be skew-symmetric.

The skew-symmetry of (M_dot - 2C) is a fundamental property of Lagrangian
mechanics. It ensures energy conservation: q_dot^T * (M_dot - 2C) * q_dot = 0.
This test performs a numerical finite-difference verification.
"""

import numpy as np
import pytest


def _make_zero_drag_system():
    """Create a zero-drag system for conservative dynamics testing."""
    import os, sys
    os.add_dll_directory(r"C:\msys64\mingw64\bin")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    import _core

    qp = _core.QuadrotorParams()
    qp.mass = 1.5
    qp.inertia = np.diag([0.0347, 0.0458, 0.0977])
    qp.arm_length = 0.25
    qp.thrust_coeff = 8.54858e-6
    qp.torque_coeff = 1.36e-7
    qp.drag_coeff = 0.0   # conservative: no dissipation
    qp.motor_time_constant = 0.02
    qp.max_motor_speed = 1200.0

    mp = _core.ManipulatorParams()
    mp.attachment_offset = np.array([0, 0, -0.1])
    lp1 = _core.LinkParams()
    lp1.mass = 0.3; lp1.length = 0.3; lp1.com_distance = 0.15
    lp1.inertia = np.diag([0.001, 0.0027, 0.0027])
    lp2 = _core.LinkParams()
    lp2.mass = 0.2; lp2.length = 0.25; lp2.com_distance = 0.125
    lp2.inertia = np.diag([0.0005, 0.0012, 0.0012])
    mp.link1 = lp1; mp.link2 = lp2
    mp.joint_lower_limit = np.array([-3.14, -1.57])
    mp.joint_upper_limit = np.array([3.14, 2.36])
    mp.max_joint_torque = 5.0

    ep = _core.EnvironmentParams()
    ep.gravity = 9.81

    sys_obj = _core.AerialManipulatorSystem(qp, mp, ep)
    sys_obj.set_integrator(_core.RK4Integrator())
    return sys_obj


class TestCoriolisSkewSymmetry:
    """Verify that d/dt(M) - 2*C is skew-symmetric (energy conservation condition).

    For any generalized velocity q_dot, the relation
        q_dot^T * (M_dot - 2*C) * q_dot = 0
    must hold. This is equivalent to M_dot - 2*C being skew-symmetric.

    Method:
        - Numerically differentiate M using finite differences (dt = 1e-7 s).
        - Extract C*q_dot from sys_obj.compute_coriolis_vector(state).
        - Check the scalar residual: q_dot^T * M_dot * q_dot - 2 * q_dot^T * C * q_dot.
    """

    # Test configurations: (q1, q2, omega1, omega2)
    _CONFIGS = [
        (0.3,  0.5,  1.0, -0.5),
        (0.0,  0.0,  2.0,  1.0),
        (1.0,  0.8,  0.5,  0.3),
        (-0.5, 1.2, -1.0,  0.8),
        (0.0,  np.pi/4, 0.0, 2.0),
    ]

    def _build_state(self, q1, q2, w1, w2):
        """Construct a non-trivial state vector for the given joint angles and rates."""
        state = np.zeros(17)
        state[2] = 1.0    # z = 1 m (non-zero altitude)
        state[6] = 1.0    # identity quaternion (qw = 1)
        state[10] = 0.3   # body roll rate [rad/s]
        state[11] = 0.2   # body pitch rate [rad/s]
        state[12] = 0.1   # body yaw rate [rad/s]
        state[13] = q1    # joint 1 position [rad]
        state[14] = q2    # joint 2 position [rad]
        state[15] = w1    # joint 1 velocity [rad/s]
        state[16] = w2    # joint 2 velocity [rad/s]
        return state

    def test_mdot_minus_2c_skew_symmetric(self):
        """Numerical check: q_dot^T * (M_dot - 2C) * q_dot = 0 for all configs."""
        sys_obj = _make_zero_drag_system()
        u = np.zeros(6)
        dt = 1e-7   # small enough for accurate finite difference

        for q1, q2, w1, w2 in self._CONFIGS:
            state = self._build_state(q1, q2, w1, w2)

            # Mass matrix at current state
            M = np.array(sys_obj.compute_mass_matrix(state))

            # Step forward by dt to estimate M_dot
            state_next = np.array(sys_obj.step(0.0, state, u, dt))
            M_next = np.array(sys_obj.compute_mass_matrix(state_next))

            # Finite-difference approximation: M_dot ~ (M_next - M) / dt
            M_dot = (M_next - M) / dt

            # C * q_dot from the engine (Coriolis + centripetal vector)
            C_qdot = np.array(sys_obj.compute_coriolis_vector(state))

            # Generalized velocity: [v_body, omega_body, q_dot_joint]
            qdot = np.concatenate([state[3:6], state[10:13], state[15:17]])

            # Skew-symmetry check via the scalar identity:
            #   q_dot^T * M_dot * q_dot = 2 * q_dot^T * C * q_dot
            # Rearranged: residual = q_dot^T * M_dot * q_dot - 2 * q_dot^T * C * q_dot = 0
            qdot_M_dot_qdot = qdot @ M_dot @ qdot
            qdot_C_qdot = qdot @ C_qdot   # = q_dot^T * C(q,qdot) * q_dot

            residual = qdot_M_dot_qdot - 2.0 * qdot_C_qdot

            assert abs(residual) < 1e-3, (
                f"M_dot - 2C skew-symmetry violated at (q1={q1}, q2={q2}, "
                f"w1={w1}, w2={w2}): residual = {residual:.4e}")

    def test_coriolis_vector_zero_at_zero_velocity(self):
        """C(q, 0) * 0 = 0: Coriolis vector must vanish at zero velocity."""
        sys_obj = _make_zero_drag_system()

        for q1, q2 in [(0.0, 0.0), (0.5, 0.3), (1.0, 0.8)]:
            state = np.zeros(17)
            state[2] = 1.0; state[6] = 1.0
            state[13] = q1; state[14] = q2
            # All velocity components remain zero

            C_qdot = np.array(sys_obj.compute_coriolis_vector(state))
            np.testing.assert_allclose(
                C_qdot, 0.0, atol=1e-14,
                err_msg=f"C*qdot nonzero at zero velocity for q=[{q1},{q2}]")

    def test_coriolis_vector_quadratic_in_velocity(self):
        """C(q, alpha*qdot) * alpha*qdot = alpha^2 * C(q, qdot) * qdot.

        Coriolis/centripetal terms are quadratic in velocity.
        """
        sys_obj = _make_zero_drag_system()

        state_base = np.zeros(17)
        state_base[2] = 1.0; state_base[6] = 1.0
        state_base[13] = 0.5; state_base[14] = 0.4
        state_base[10] = 0.3; state_base[11] = 0.2; state_base[12] = 0.1
        state_base[15] = 1.0; state_base[16] = -0.5

        C_qdot_1 = np.array(sys_obj.compute_coriolis_vector(state_base))

        alpha = 2.0
        state_scaled = state_base.copy()
        # Scale all velocity components by alpha
        state_scaled[3:6]   *= alpha
        state_scaled[10:13] *= alpha
        state_scaled[15:17] *= alpha

        C_qdot_alpha = np.array(sys_obj.compute_coriolis_vector(state_scaled))

        np.testing.assert_allclose(
            C_qdot_alpha, alpha**2 * C_qdot_1, rtol=1e-10,
            err_msg="Coriolis vector is not quadratic in velocity (scaling failed)")
