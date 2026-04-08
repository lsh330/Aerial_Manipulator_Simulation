"""Unit tests for manipulator dynamics components: M(q), C(q,qdot), G(q).

These tests validate individual dynamic matrices of the manipulator arm
against known analytical values and structural properties.
"""

import numpy as np
import pytest


# Physical parameters shared across tests (must match _core construction)
MASS_QUAD = 1.5    # [kg]
MASS_L1   = 0.3    # [kg]
MASS_L2   = 0.2    # [kg]
MASS_TOTAL = MASS_QUAD + MASS_L1 + MASS_L2   # [kg]
GRAVITY   = 9.81   # [m/s^2]
L1        = 0.3    # link 1 length [m]
L2        = 0.25   # link 2 length [m]
LC1       = 0.15   # link 1 COM distance [m]
LC2       = 0.125  # link 2 COM distance [m]
ATT_Z     = -0.1   # attachment offset z [m]


def _make_system():
    """Build the full aerial manipulator C++ system."""
    import os, sys
    os.add_dll_directory(r"C:\msys64\mingw64\bin")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    import _core

    qp = _core.QuadrotorParams()
    qp.mass = MASS_QUAD
    qp.inertia = np.diag([0.0347, 0.0458, 0.0977])
    qp.arm_length = 0.25
    qp.thrust_coeff = 8.54858e-6
    qp.torque_coeff = 1.36e-7
    qp.drag_coeff = 0.0
    qp.motor_time_constant = 0.02
    qp.max_motor_speed = 1200.0

    mp = _core.ManipulatorParams()
    mp.attachment_offset = np.array([0.0, 0.0, ATT_Z])
    lp1 = _core.LinkParams()
    lp1.mass = MASS_L1; lp1.length = L1; lp1.com_distance = LC1
    lp1.inertia = np.diag([0.001, 0.0027, 0.0027])
    lp2 = _core.LinkParams()
    lp2.mass = MASS_L2; lp2.length = L2; lp2.com_distance = LC2
    lp2.inertia = np.diag([0.0005, 0.0012, 0.0012])
    mp.link1 = lp1; mp.link2 = lp2
    mp.joint_lower_limit = np.array([-3.14, -1.57])
    mp.joint_upper_limit = np.array([3.14, 2.36])
    mp.max_joint_torque = 5.0

    ep = _core.EnvironmentParams()
    ep.gravity = GRAVITY

    sys_obj = _core.AerialManipulatorSystem(qp, mp, ep)
    sys_obj.set_integrator(_core.RK4Integrator())
    return sys_obj


def _hover_state(q1=0.0, q2=0.0):
    """Hover state: z=1m, identity quaternion, arm at given joint angles."""
    state = np.zeros(17)
    state[2] = 1.0    # z = 1 m
    state[6] = 1.0    # qw = 1 (identity quaternion)
    state[13] = q1
    state[14] = q2
    return state


# ---------------------------------------------------------------------------
# Mass matrix tests
# ---------------------------------------------------------------------------

class TestManipulatorMassMatrix:
    """Unit tests for the system mass matrix M(q)."""

    def test_mass_matrix_symmetry(self):
        """M(q) = M(q)^T for all tested configurations."""
        sys_obj = _make_system()
        configs = [(0, 0), (0.5, 0.3), (1.0, 0.8), (-0.7, 1.2), (0.0, np.pi/4)]

        for q1, q2 in configs:
            state = _hover_state(q1, q2)
            M = np.array(sys_obj.compute_mass_matrix(state))
            np.testing.assert_allclose(
                M, M.T, atol=1e-12,
                err_msg=f"M not symmetric at q=[{q1:.2f},{q2:.2f}]")

    def test_mass_matrix_positive_definite(self):
        """M(q) must have all positive eigenvalues (positive definite)."""
        sys_obj = _make_system()
        configs = [(0, 0), (0.5, 0.3), (1.0, 0.8), (0.0, np.pi/2)]

        for q1, q2 in configs:
            state = _hover_state(q1, q2)
            M = np.array(sys_obj.compute_mass_matrix(state))
            eigvals = np.linalg.eigvalsh(M)
            assert np.all(eigvals > 0), (
                f"M not positive definite at q=[{q1:.2f},{q2:.2f}]: "
                f"min eigenvalue = {eigvals.min():.4e}")

    def test_mass_matrix_diagonal_lower_bound(self):
        """Diagonal entries of M must be >= total mass for translational DOFs."""
        sys_obj = _make_system()
        state = _hover_state(0.0, 0.0)
        M = np.array(sys_obj.compute_mass_matrix(state))

        # Translational DOF (indices 0,1,2): diagonal element >= total_mass
        for i in range(3):
            assert M[i, i] >= MASS_TOTAL - 1e-10, (
                f"M[{i},{i}] = {M[i,i]:.6f} < total_mass = {MASS_TOTAL:.6f}")

    def test_mass_matrix_at_home_position(self):
        """At q=(0,0) with identity quaternion, M[0:3,0:3] must equal m_total * I_3."""
        sys_obj = _make_system()
        state = _hover_state(0.0, 0.0)
        M = np.array(sys_obj.compute_mass_matrix(state))

        # Top-left 3x3 block: translational inertia = m_total * I
        np.testing.assert_allclose(
            M[0:3, 0:3], MASS_TOTAL * np.eye(3), atol=1e-8,
            err_msg="Translational mass block M[0:3,0:3] != m_total * I at home")

    def test_mass_matrix_configuration_dependence(self):
        """M(q1, q2) must change when arm configuration changes."""
        sys_obj = _make_system()
        M_home = np.array(sys_obj.compute_mass_matrix(_hover_state(0.0, 0.0)))
        M_away = np.array(sys_obj.compute_mass_matrix(_hover_state(0.5, 1.0)))

        # Mass matrices must differ when arm is in different configuration
        assert not np.allclose(M_home, M_away, atol=1e-10), (
            "Mass matrix unchanged between home and extended arm — "
            "configuration dependence may be broken")


# ---------------------------------------------------------------------------
# Coriolis / centripetal tests
# ---------------------------------------------------------------------------

class TestManipulatorCoriolisMatrix:
    """Unit tests for the Coriolis and centripetal vector C(q, qdot) * qdot."""

    def test_coriolis_at_zero_velocity(self):
        """C(q, 0) * 0 = 0: Coriolis vector must vanish when all velocities are zero."""
        sys_obj = _make_system()
        configs = [(0, 0), (0.5, 0.3), (1.0, 0.8)]

        for q1, q2 in configs:
            state = _hover_state(q1, q2)
            # All velocity entries remain zero
            C_qdot = np.array(sys_obj.compute_coriolis_vector(state))
            np.testing.assert_allclose(
                C_qdot, 0.0, atol=1e-14,
                err_msg=f"C*qdot nonzero at zero velocity for q=[{q1},{q2}]")

    def test_coriolis_sign_consistency(self):
        """Reversing velocity sign must keep C*qdot sign the same (quadratic)."""
        sys_obj = _make_system()

        state_pos = _hover_state(0.5, 0.3)
        state_pos[10] = 0.5; state_pos[11] = 0.3; state_pos[12] = 0.1
        state_pos[15] = 1.0; state_pos[16] = -0.5

        state_neg = state_pos.copy()
        state_neg[3:6]   *= -1.0
        state_neg[10:13] *= -1.0
        state_neg[15:17] *= -1.0

        C_qdot_pos = np.array(sys_obj.compute_coriolis_vector(state_pos))
        C_qdot_neg = np.array(sys_obj.compute_coriolis_vector(state_neg))

        # C(q, -qdot) * (-qdot) = C(q, qdot) * qdot (quadratic in velocity)
        np.testing.assert_allclose(
            C_qdot_neg, C_qdot_pos, atol=1e-10,
            err_msg="C*qdot not equal under velocity sign reversal (quadratic property)")


# ---------------------------------------------------------------------------
# Gravity vector tests
# ---------------------------------------------------------------------------

class TestManipulatorGravity:
    """Unit tests for the gravity/generalized gravity vector G(q)."""

    def test_gravity_vector_translational_component(self):
        """G vector translational part must equal m_total * g in vertical direction."""
        sys_obj = _make_system()
        state = _hover_state(0.0, 0.0)
        G = np.array(sys_obj.compute_gravity_vector(state))

        # With identity rotation and arm at home, the vertical force component
        # (index 2, body frame z = world frame z) must balance total weight
        np.testing.assert_allclose(
            G[2], MASS_TOTAL * GRAVITY, atol=1e-8,
            err_msg=f"G[2]={G[2]:.6f} != m_total*g={MASS_TOTAL*GRAVITY:.6f}")

    def test_gravity_vector_zero_gravity(self):
        """With g=0, gravity vector must be zero everywhere."""
        import os, sys
        os.add_dll_directory(r"C:\msys64\mingw64\bin")
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
        import _core

        qp = _core.QuadrotorParams()
        qp.mass = MASS_QUAD
        qp.inertia = np.diag([0.0347, 0.0458, 0.0977])
        qp.arm_length = 0.25; qp.thrust_coeff = 8.54858e-6
        qp.torque_coeff = 1.36e-7; qp.drag_coeff = 0.0
        qp.motor_time_constant = 0.02; qp.max_motor_speed = 1200.0

        mp = _core.ManipulatorParams()
        mp.attachment_offset = np.array([0.0, 0.0, ATT_Z])
        lp1 = _core.LinkParams()
        lp1.mass = MASS_L1; lp1.length = L1; lp1.com_distance = LC1
        lp1.inertia = np.diag([0.001, 0.0027, 0.0027])
        lp2 = _core.LinkParams()
        lp2.mass = MASS_L2; lp2.length = L2; lp2.com_distance = LC2
        lp2.inertia = np.diag([0.0005, 0.0012, 0.0012])
        mp.link1 = lp1; mp.link2 = lp2
        mp.joint_lower_limit = np.array([-3.14, -1.57])
        mp.joint_upper_limit = np.array([3.14, 2.36])
        mp.max_joint_torque = 5.0

        ep = _core.EnvironmentParams()
        ep.gravity = 0.0   # zero gravity

        sys_obj_zero_g = _core.AerialManipulatorSystem(qp, mp, ep)
        state = _hover_state(0.5, 0.8)
        G = np.array(sys_obj_zero_g.compute_gravity_vector(state))
        np.testing.assert_allclose(G, 0.0, atol=1e-14,
            err_msg="Gravity vector nonzero when g=0")

    def test_gravity_vector_configuration_dependence(self):
        """G(q) must change with arm configuration (joint-space gravity terms)."""
        sys_obj = _make_system()
        G_home = np.array(sys_obj.compute_gravity_vector(_hover_state(0.0, 0.0)))
        G_away = np.array(sys_obj.compute_gravity_vector(_hover_state(0.5, 1.0)))

        # Joint-space gravity terms (indices 6,7) must differ
        assert not np.allclose(G_home[6:8], G_away[6:8], atol=1e-8), (
            "Joint-space gravity terms unchanged between configurations — "
            "gravity projection onto joint DOFs may be broken")
