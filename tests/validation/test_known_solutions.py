"""Validation test: comparison against known analytical solutions.

Tests the C++ dynamics engine against cases with known closed-form
solutions to verify numerical accuracy.
"""

import numpy as np
import pytest
from models.state import State, IDX


def _make_system():
    import os, sys
    os.add_dll_directory(r"C:\msys64\mingw64\bin")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    import _core

    qp = _core.QuadrotorParams()
    qp.mass = 1.5; qp.inertia = np.diag([0.0347, 0.0458, 0.0977])
    qp.arm_length = 0.25; qp.thrust_coeff = 8.54858e-6
    qp.torque_coeff = 1.36e-7; qp.drag_coeff = 0.016
    qp.motor_time_constant = 0.02; qp.max_motor_speed = 1200.0

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


class TestHoverEquilibrium:
    """Verify exact hover equilibrium: x_dot = 0 when Bu = G."""

    def test_hover_state_derivative_is_zero(self):
        sys_obj = _make_system()
        state = np.zeros(17)
        state[2] = 1.0; state[6] = 1.0  # z=1, qw=1

        m_total = sys_obj.total_mass()
        thrust_per_motor = m_total * 9.81 / 4.0
        u = np.array([thrust_per_motor] * 4 + [0, 0])

        xdot = np.array(sys_obj.compute_state_derivative(state, u))

        # Position derivative = velocity = 0
        np.testing.assert_allclose(xdot[IDX.VEL], 0, atol=1e-12,
                                   err_msg="Linear acceleration nonzero at hover")
        # Angular acceleration = 0
        np.testing.assert_allclose(xdot[IDX.ANG_VEL], 0, atol=1e-12,
                                   err_msg="Angular acceleration nonzero at hover")
        # Joint acceleration = 0
        np.testing.assert_allclose(xdot[IDX.JOINT_VEL], 0, atol=1e-12,
                                   err_msg="Joint acceleration nonzero at hover")

    def test_mass_matrix_symmetric(self):
        """M(q) must be symmetric at any configuration."""
        sys_obj = _make_system()
        for q1, q2 in [(0, 0), (0.5, 0.3), (1.0, 0.8), (-0.7, 1.2)]:
            state = np.zeros(17)
            state[2] = 1.0; state[6] = 1.0
            state[13] = q1; state[14] = q2
            M = np.array(sys_obj.compute_mass_matrix(state))
            np.testing.assert_allclose(
                M, M.T, atol=1e-12,
                err_msg=f"M not symmetric at q_joint=[{q1},{q2}]")

    def test_mass_matrix_positive_definite(self):
        """M(q) must be positive definite at any configuration."""
        sys_obj = _make_system()
        for q1, q2 in [(0, 0), (0.5, 0.3), (1.0, 0.8), (0, np.pi/2)]:
            state = np.zeros(17)
            state[2] = 1.0; state[6] = 1.0
            state[13] = q1; state[14] = q2
            M = np.array(sys_obj.compute_mass_matrix(state))
            eigvals = np.linalg.eigvalsh(M)
            assert np.all(eigvals > 0), (
                f"M not positive definite at q=[{q1},{q2}]: "
                f"min eigenvalue = {eigvals.min():.2e}")


class TestFreeFall:
    """Free fall: z(t) = z0 - 0.5*g*t² (analytical solution)."""

    def test_free_fall_trajectory(self):
        sys_obj = _make_system()
        state = np.zeros(17)
        state[2] = 10.0; state[6] = 1.0  # z=10m, level
        u = np.zeros(6)

        dt = 0.001
        T = 0.5  # 0.5s of free fall
        g = 9.81

        for i in range(int(T / dt)):
            state = np.array(sys_obj.step(i * dt, state, u, dt))

        z_analytical = 10.0 - 0.5 * g * T**2
        z_simulated = state[2]

        # Allow small error from drag (drag_coeff=0.016, very small at low speed)
        np.testing.assert_allclose(
            z_simulated, z_analytical, atol=0.01,
            err_msg=f"Free fall: simulated z={z_simulated:.4f} vs "
                    f"analytical z={z_analytical:.4f}")

    def test_free_fall_velocity(self):
        """v_z(t) = -g*t at t=0.5s."""
        sys_obj = _make_system()
        state = np.zeros(17)
        state[2] = 10.0; state[6] = 1.0
        u = np.zeros(6)

        dt = 0.001; T = 0.5
        for i in range(int(T / dt)):
            state = np.array(sys_obj.step(i * dt, state, u, dt))

        vz_analytical = -9.81 * T
        vz_simulated = state[5]

        np.testing.assert_allclose(
            vz_simulated, vz_analytical, atol=0.05,
            err_msg=f"Free fall velocity: sim={vz_simulated:.4f} vs "
                    f"analytical={vz_analytical:.4f}")


class TestQuaternionNormalization:
    """Quaternion must stay normalized through integration."""

    def test_quaternion_norm_preserved(self):
        sys_obj = _make_system()
        state = np.zeros(17)
        state[2] = 1.0; state[6] = 1.0
        state[10] = 2.0; state[11] = 1.0  # angular velocity

        u_hover = np.array([4.905] * 4 + [0, 0])
        for i in range(1000):
            state = np.array(sys_obj.step(i * 0.001, state, u_hover, 0.001))

        quat_norm = np.linalg.norm(state[IDX.QUAT])
        np.testing.assert_allclose(
            quat_norm, 1.0, atol=1e-10,
            err_msg=f"Quaternion norm drifted to {quat_norm}")
