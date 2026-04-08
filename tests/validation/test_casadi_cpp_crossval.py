"""Cross-validation: CasADi symbolic dynamics vs C++ engine.

Verifies that the CasADi-based dynamics model (used inside the NMPC
optimizer) produces identical state derivatives and mass matrices as
the C++ simulation engine at multiple operating points.

If CasADi is not installed, or if the CasADi dynamics builder is not
available, all tests in this module are skipped automatically.
"""

import numpy as np
import pytest

casadi = pytest.importorskip("casadi", reason="CasADi not installed")


def _make_system():
    """Build the C++ aerial manipulator system."""
    import os, sys
    os.add_dll_directory(r"C:\msys64\mingw64\bin")
    sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    import _core

    qp = _core.QuadrotorParams()
    qp.mass = 1.5; qp.inertia = np.diag([0.0347, 0.0458, 0.0977])
    qp.arm_length = 0.25; qp.thrust_coeff = 8.54858e-6
    qp.torque_coeff = 1.36e-7; qp.drag_coeff = 0.0
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


def _build_casadi_dynamics():
    """Import and instantiate the CasADi dynamics builder.

    Returns a callable f(state, u) -> xdot, or skips the test if the
    builder module is not present.
    """
    try:
        import sys, os
        sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "control"))
        from casadi_dynamics import build_dynamics_function
        return build_dynamics_function()
    except ImportError:
        pytest.skip("CasADi dynamics builder (control/casadi_dynamics.py) not found")


# ---------------------------------------------------------------------------
# Helper state constructors
# ---------------------------------------------------------------------------

def _hover_state():
    """Hovering at z=1m, identity quaternion, arm at zero."""
    state = np.zeros(17)
    state[2] = 1.0
    state[6] = 1.0
    return state


def _tilted_state(roll=0.1, pitch=0.2):
    """Quadrotor tilted by roll and pitch angles [rad]."""
    state = np.zeros(17)
    state[2] = 1.0
    # Quaternion for roll then pitch (small-angle approximation via exact formula)
    cr, sr = np.cos(roll / 2), np.sin(roll / 2)
    cp, sp = np.cos(pitch / 2), np.sin(pitch / 2)
    # q_roll * q_pitch
    qw = cr * cp
    qx = sr * cp
    qy = cr * sp
    qz = -sr * sp
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    state[6:10] = [qw / norm, qx / norm, qy / norm, qz / norm]
    return state


def _arm_extended_state(q1=0.5, q2=0.8):
    """Hover with arm at specified joint angles [rad]."""
    state = np.zeros(17)
    state[2] = 1.0; state[6] = 1.0
    state[13] = q1; state[14] = q2
    return state


def _moving_state(vx=1.0, omega_z=0.5, qd1=2.0):
    """Hover with non-zero translational velocity, yaw rate, and joint velocity."""
    state = np.zeros(17)
    state[2] = 1.0; state[6] = 1.0
    state[3] = vx           # vx [m/s]
    state[12] = omega_z     # omega_z [rad/s]
    state[15] = qd1         # joint 1 velocity [rad/s]
    return state


def _hover_input(sys_obj):
    """Thrust allocation for hover equilibrium."""
    m_total = sys_obj.total_mass()
    f_each = m_total * 9.81 / 4.0
    return np.array([f_each, f_each, f_each, f_each, 0.0, 0.0])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCasAdiCppConsistency:
    """Verify that CasADi symbolic dynamics match the C++ engine."""

    def test_state_derivative_match(self):
        """f_casadi(x, u) == f_cpp(x, u) at multiple operating points."""
        sys_obj = _make_system()
        f_casadi = _build_casadi_dynamics()

        test_cases = [
            ("hover",        _hover_state()),
            ("tilted",       _tilted_state(roll=0.1, pitch=0.2)),
            ("arm_extended", _arm_extended_state(q1=0.5, q2=0.8)),
            ("moving",       _moving_state(vx=1.0, omega_z=0.5, qd1=2.0)),
        ]

        u = _hover_input(sys_obj)

        for label, state in test_cases:
            xdot_cpp = np.array(sys_obj.compute_state_derivative(state, u))
            xdot_casadi = np.array(f_casadi(state, u)).ravel()

            np.testing.assert_allclose(
                xdot_cpp, xdot_casadi, atol=1e-6,
                err_msg=f"CasADi/C++ xdot mismatch at state '{label}': "
                        f"max diff = {np.max(np.abs(xdot_cpp - xdot_casadi)):.2e}")

    def test_mass_matrix_match(self):
        """M_casadi(q) == M_cpp(q) at multiple configurations."""
        sys_obj = _make_system()
        f_casadi = _build_casadi_dynamics()

        configs = [
            ("home",     _hover_state()),
            ("q1=0.5",   _arm_extended_state(q1=0.5, q2=0.0)),
            ("q2=0.8",   _arm_extended_state(q1=0.0, q2=0.8)),
            ("full_arm", _arm_extended_state(q1=0.5, q2=0.8)),
        ]

        for label, state in configs:
            M_cpp = np.array(sys_obj.compute_mass_matrix(state))

            # CasADi mass matrix via the builder's dedicated method (if available)
            try:
                M_casadi = np.array(f_casadi.mass_matrix(state))
            except AttributeError:
                pytest.skip("CasADi dynamics builder does not expose mass_matrix()")

            np.testing.assert_allclose(
                M_cpp, M_casadi, atol=1e-8,
                err_msg=f"CasADi/C++ mass matrix mismatch at '{label}'")

    def test_gravity_vector_match(self):
        """G_casadi(q) == G_cpp(q) at multiple configurations."""
        sys_obj = _make_system()
        f_casadi = _build_casadi_dynamics()

        configs = [
            ("home",     _hover_state()),
            ("tilted",   _tilted_state(roll=0.1, pitch=0.1)),
            ("arm_0.5",  _arm_extended_state(q1=0.5, q2=0.8)),
        ]

        for label, state in configs:
            G_cpp = np.array(sys_obj.compute_gravity_vector(state))

            try:
                G_casadi = np.array(f_casadi.gravity_vector(state)).ravel()
            except AttributeError:
                pytest.skip("CasADi dynamics builder does not expose gravity_vector()")

            np.testing.assert_allclose(
                G_cpp, G_casadi, atol=1e-8,
                err_msg=f"CasADi/C++ gravity vector mismatch at '{label}'")
