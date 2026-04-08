"""Unit tests for quadrotor mixing matrix (thrust allocation).

The mixing matrix A_mix maps individual rotor thrusts [f1,f2,f3,f4] to
collective thrust and body torques [F_z, tau_x, tau_y, tau_z].

Properties verified:
    1. Full rank (invertible): rank(A_mix) = 4
    2. Left-inverse accuracy: A_mix^{-1} * A_mix = I_4
    3. Hover allocation: equal thrusts produce zero roll/pitch/yaw torques
    4. Torque sign convention: motor pairs produce correct torque directions
"""

import numpy as np
import pytest


def _make_system():
    """Build the full aerial manipulator C++ system."""
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
    qp.drag_coeff = 0.0
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


class TestMixingMatrix:
    """Structural and physical correctness tests for the quadrotor mixing matrix."""

    def _get_mixing_matrix(self, sys_obj):
        """Extract the 4x4 mixing matrix A from the quadrotor sub-system."""
        return np.array(sys_obj.quadrotor().mixing_matrix())

    def test_mixing_matrix_shape(self):
        """Mixing matrix must be 4x4 (maps 4 thrusts to [F, tau_x, tau_y, tau_z])."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)
        assert A.shape == (4, 4), f"Expected (4,4), got {A.shape}"

    def test_mixing_matrix_invertible(self):
        """A_mix must be full rank (invertible) to allow arbitrary wrench allocation."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)
        rank = np.linalg.matrix_rank(A)
        assert rank == 4, f"Mixing matrix rank = {rank}, expected 4 (singular matrix)"

    def test_mixing_matrix_inverse(self):
        """A_mix @ A_mix^{-1} = I_4 to numerical precision."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)
        A_inv = np.linalg.inv(A)
        np.testing.assert_allclose(
            A @ A_inv, np.eye(4), atol=1e-12,
            err_msg="A_mix @ inv(A_mix) != I_4")

    def test_hover_thrust_allocation(self):
        """Equal per-motor thrusts must produce zero roll, pitch, and yaw torques."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)

        # Hover: each motor contributes equal thrust
        m_total = sys_obj.total_mass()
        f_each = m_total * 9.81 / 4.0
        f_hover = np.array([f_each, f_each, f_each, f_each])

        result = A @ f_hover  # [F_z, tau_x, tau_y, tau_z]

        # Total thrust = sum of rotor thrusts
        np.testing.assert_allclose(
            result[0], 4.0 * f_each, rtol=1e-10,
            err_msg="Hover: total collective thrust incorrect")
        # Roll torque = 0
        assert abs(result[1]) < 1e-12, (
            f"Hover: roll torque nonzero = {result[1]:.4e}")
        # Pitch torque = 0
        assert abs(result[2]) < 1e-12, (
            f"Hover: pitch torque nonzero = {result[2]:.4e}")
        # Yaw torque = 0 for equal CW/CCW pairing
        assert abs(result[3]) < 1e-12, (
            f"Hover: yaw torque nonzero = {result[3]:.4e}")

    def test_collective_thrust_positive(self):
        """Collective thrust row (row 0) must have all positive coefficients."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)
        assert np.all(A[0, :] > 0), (
            f"Some rotor thrust coefficients are non-positive: {A[0, :]}")

    def test_roll_torque_antisymmetric(self):
        """Roll torque (row 1): left and right motors must have opposite signs."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)

        # Roll: tau_x row must sum to zero (equal + and - contributions)
        roll_row = A[1, :]
        np.testing.assert_allclose(
            np.sum(roll_row), 0.0, atol=1e-12,
            err_msg=f"Roll torque row does not sum to zero: {roll_row}")

    def test_pitch_torque_antisymmetric(self):
        """Pitch torque (row 2): front and rear motors must have opposite signs."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)

        pitch_row = A[2, :]
        np.testing.assert_allclose(
            np.sum(pitch_row), 0.0, atol=1e-12,
            err_msg=f"Pitch torque row does not sum to zero: {pitch_row}")

    def test_yaw_torque_antisymmetric(self):
        """Yaw torque (row 3): CW and CCW motor pairs must cancel."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)

        yaw_row = A[3, :]
        np.testing.assert_allclose(
            np.sum(yaw_row), 0.0, atol=1e-12,
            err_msg=f"Yaw torque row does not sum to zero: {yaw_row}")

    def test_inverse_thrust_allocation_recovery(self):
        """A_mix^{-1} @ [F, 0, 0, 0]^T must give equal thrusts for pure collective."""
        sys_obj = _make_system()
        A = self._get_mixing_matrix(sys_obj)
        A_inv = np.linalg.inv(A)

        F_total = 14.715   # [N]
        wrench = np.array([F_total, 0.0, 0.0, 0.0])
        f_motors = A_inv @ wrench

        # All motor thrusts must be equal and positive
        np.testing.assert_allclose(
            f_motors, np.full(4, F_total / 4.0), atol=1e-10,
            err_msg="Inverse allocation for pure collective thrust is not uniform")
