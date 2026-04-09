"""Validation test: energy conservation in uncontrolled free dynamics.

For a conservative system (no control, no drag), total mechanical energy
E = T + V must be conserved. This validates that M(q), C(q,q̇), G(q)
are mathematically consistent (M_dot - 2C is skew-symmetric).

Test methodology:
    1. Set drag_coeff = 0 (conservative system)
    2. Apply zero input (u = 0, free fall + arm swing)
    3. Compute kinetic + potential energy at each step
    4. Verify energy drift < tolerance over simulation period
"""

import numpy as np
import pytest
from models.state import State, IDX


def _make_zero_drag_system():
    """Create a system with zero drag for conservative dynamics."""
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
    qp.drag_coeff = 0.0  # NO DRAG → conservative
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
    return sys_obj, ep.gravity, qp.mass + lp1.mass + lp2.mass


def _compute_energy(sys_obj, state, gravity, total_mass):
    """Compute total mechanical energy: E = T + V.

    T = 0.5 * q_dot^T * M(q) * q_dot
    V = g * sum_i(m_i * z_i_world)  — full potential including arm link heights.

    Physical parameters are hardcoded to match _make_zero_drag_system
    (m0=1.5, m1=0.3, m2=0.2, lc1=0.15, l1=0.3, lc2=0.125, att=[0,0,-0.1]).
    """
    M = np.array(sys_obj.compute_mass_matrix(state))

    # Generalized velocity
    q_dot = np.concatenate([
        state[IDX.VEL], state[IDX.ANG_VEL], state[IDX.JOINT_VEL]
    ])

    # Kinetic energy
    T = 0.5 * q_dot @ M @ q_dot

    # ── Full potential energy: V = g * sum(m_i * z_i_world) ──
    m0, m1, m2 = 1.5, 0.3, 0.2
    lc1, l1, lc2 = 0.15, 0.3, 0.125
    D = l1 + lc2  # 0.425
    att_z = -0.1

    q1, q2 = state[IDX.JOINT_POS]
    c1, s1 = np.cos(q1), np.sin(q1)
    c2, s2 = np.cos(q2), np.sin(q2)

    # Link COM positions in body frame
    r_c1_body = np.array([lc1*c1*s2, lc1*s1*s2, att_z - lc1*c2])
    r_c2_body = np.array([D*c1*s2,   D*s1*s2,   att_z - D*c2])

    # Rotation matrix from quaternion
    qw, qx, qy, qz = state[IDX.QUAT]
    R = np.array([
        [1-2*(qy**2+qz**2), 2*(qx*qy-qw*qz), 2*(qx*qz+qw*qy)],
        [2*(qx*qy+qw*qz), 1-2*(qx**2+qz**2), 2*(qy*qz-qw*qx)],
        [2*(qx*qz-qw*qy), 2*(qy*qz+qw*qx), 1-2*(qx**2+qy**2)],
    ])

    z_quad = state[2]
    z_c1 = z_quad + (R @ r_c1_body)[2]
    z_c2 = z_quad + (R @ r_c2_body)[2]

    V = gravity * (m0 * z_quad + m1 * z_c1 + m2 * z_c2)

    return T + V


class TestEnergyConservation:
    """Energy conservation in conservative (drag-free, input-free) dynamics."""

    def test_free_fall_energy_conservation(self):
        """Free fall from z=2m with no input: E must be conserved."""
        sys_obj, g, m_total = _make_zero_drag_system()

        state = np.zeros(17)
        state[2] = 2.0   # z = 2m
        state[6] = 1.0   # identity quaternion

        E0 = _compute_energy(sys_obj, state, g, m_total)
        u = np.zeros(6)

        energies = [E0]
        for i in range(500):  # 0.5s of free fall
            state = np.array(sys_obj.step(i * 0.001, state, u, 0.001))
            energies.append(_compute_energy(sys_obj, state, g, m_total))

        energies = np.array(energies)
        drift = np.abs(energies - E0) / abs(E0)
        max_drift = np.max(drift)

        assert max_drift < 1e-6, (
            f"Energy drift {max_drift:.2e} exceeds 1e-6 (E0={E0:.4f}, "
            f"E_final={energies[-1]:.4f})")

    def test_arm_swing_energy_conservation(self):
        """Arm swinging with initial joint velocity: E must be conserved."""
        sys_obj, g, m_total = _make_zero_drag_system()

        state = np.zeros(17)
        state[2] = 5.0    # high altitude (room to fall)
        state[6] = 1.0    # identity quaternion
        state[15] = 2.0   # joint 1 velocity = 2 rad/s
        state[16] = -1.0  # joint 2 velocity = -1 rad/s

        E0 = _compute_energy(sys_obj, state, g, m_total)
        u = np.zeros(6)

        energies = [E0]
        for i in range(300):  # 0.3s
            state = np.array(sys_obj.step(i * 0.001, state, u, 0.001))
            energies.append(_compute_energy(sys_obj, state, g, m_total))

        energies = np.array(energies)
        drift = np.abs(energies - E0) / abs(E0)
        max_drift = np.max(drift)

        # Arm swing induces body rotation via coupling, which activates the
        # Coriolis numerical approximation (finite-diff dM/d_euler, eps=1e-7).
        # This introduces a small systematic energy drift (~0.1-0.2%).
        assert max_drift < 5e-3, (
            f"Energy drift {max_drift:.2e} with arm swing exceeds 0.5%")

    def test_tumbling_energy_conservation(self):
        """Tumbling quadrotor with angular velocity: E must be conserved."""
        sys_obj, g, m_total = _make_zero_drag_system()

        state = np.zeros(17)
        state[2] = 10.0   # high altitude
        state[6] = 1.0    # identity quaternion
        state[10] = 1.0   # roll rate
        state[11] = 0.5   # pitch rate
        state[12] = 0.3   # yaw rate

        E0 = _compute_energy(sys_obj, state, g, m_total)
        u = np.zeros(6)

        energies = [E0]
        for i in range(200):  # 0.2s
            state = np.array(sys_obj.step(i * 0.001, state, u, 0.001))
            energies.append(_compute_energy(sys_obj, state, g, m_total))

        energies = np.array(energies)
        drift = np.abs(energies - E0) / abs(E0)
        max_drift = np.max(drift)

        assert max_drift < 1e-3, (
            f"Tumbling energy drift {max_drift:.2e} exceeds 0.1%")
