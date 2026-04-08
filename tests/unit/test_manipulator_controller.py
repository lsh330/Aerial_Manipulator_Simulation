"""Unit tests for ManipulatorController with gravity compensation."""

import numpy as np
import pytest
from control.manipulator_controller import ManipulatorController
from models.state import State


@pytest.fixture
def controller():
    return ManipulatorController(
        Kp=[20.0, 20.0], Kd=[1.0, 1.0],
        gravity_compensation=True,
        reaction_compensation=True,
        max_torque=5.0,
        link_masses=(0.3, 0.2),
        link_com_distances=(0.15, 0.125),
        link1_length=0.3,
        gravity=9.81,
    )


@pytest.fixture
def controller_no_gravity():
    return ManipulatorController(
        Kp=[20.0, 20.0], Kd=[1.0, 1.0],
        gravity_compensation=False,
        max_torque=5.0,
        link_masses=(0.3, 0.2),
        link_com_distances=(0.15, 0.125),
        link1_length=0.3,
        gravity=9.81,
    )


class TestManipulatorController:
    def test_output_shape(self, controller):
        state = State()
        ref = {"joint_positions": np.array([0.0, 0.0])}
        tau = controller.update(state.data, ref, dt=0.001)
        assert tau.shape == (2,)

    def test_zero_error_zero_torque_no_gravity(self, controller_no_gravity):
        """Without gravity comp, zero error → zero torque."""
        state = State()
        ref = {"joint_positions": np.array([0.0, 0.0])}
        tau = controller_no_gravity.update(state.data, ref, dt=0.001)
        np.testing.assert_allclose(tau, [0, 0], atol=1e-10)

    def test_gravity_comp_at_vertical(self, controller):
        """At q=[0,0] (arm vertical down), gravity torque should be ~0 for level body."""
        state = State()  # identity quaternion → level body
        ref = {"joint_positions": np.array([0.0, 0.0])}
        tau = controller.update(state.data, ref, dt=0.001)
        # At q2=0, sin(q2)=0 → gravity torque ≈ 0
        np.testing.assert_allclose(tau, [0, 0], atol=0.01)

    def test_gravity_comp_at_horizontal(self, controller):
        """At q=[0, π/2] (arm horizontal), elevation gravity torque should be nonzero."""
        state = State()
        state.joint_positions = [0.0, np.pi/2]
        ref = {"joint_positions": np.array([0.0, np.pi/2])}
        tau = controller.update(state.data, ref, dt=0.001)
        # PD error = 0, only gravity comp remains
        # tau_q2 should compensate gravity pulling arm down
        # G_q2 ≈ -(m1*lc1 + m2*D) * (-g) * sin(pi/2) > 0
        assert abs(tau[1]) > 0.5  # should be ~1.275 N·m

    def test_position_error_produces_torque(self, controller_no_gravity):
        """Joint position error → restoring torque."""
        state = State()
        state.joint_positions = [0.0, 0.0]
        ref = {"joint_positions": np.array([0.5, -0.3])}
        tau = controller_no_gravity.update(state.data, ref, dt=0.001)
        assert tau[0] > 0  # positive error → positive torque
        assert tau[1] < 0  # negative error → negative torque

    def test_saturation(self, controller):
        """Large error should be clamped to max_torque."""
        state = State()
        ref = {"joint_positions": np.array([3.0, 3.0])}  # huge error
        tau = controller.update(state.data, ref, dt=0.001)
        assert np.all(np.abs(tau) <= 5.0)

    def test_gravity_comp_differs_with_body_tilt(self, controller):
        """Tilted body should change gravity compensation."""
        ref = {"joint_positions": np.array([0.0, np.pi/4])}

        # Level body
        state_level = State()
        state_level.joint_positions = [0.0, np.pi/4]
        tau_level = controller.update(state_level.data, ref, dt=0.001)

        # 30° pitch
        state_tilted = State()
        state_tilted.joint_positions = [0.0, np.pi/4]
        angle = np.radians(30)
        state_tilted.quaternion = [np.cos(angle/2), 0, np.sin(angle/2), 0]
        tau_tilted = controller.update(state_tilted.data, ref, dt=0.001)

        # Gravity comp should differ because g_body changes with body tilt
        assert not np.allclose(tau_level, tau_tilted, atol=0.01)
