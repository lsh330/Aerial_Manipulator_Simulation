"""Unit tests for PositionController."""

import numpy as np
import pytest
from control.position_controller import PositionController
from models.state import State


@pytest.fixture
def controller():
    return PositionController(
        Kp=[6.0, 6.0, 8.0],
        Kd=[4.5, 4.5, 5.6],
        Ki=[0.1, 0.1, 0.2],
        total_mass=2.0,
        gravity=9.81,
        integrator_limit=1.0,
    )


class TestPositionController:
    def test_hover_output_shape(self, controller):
        """Output should be 10 elements: [thrust, R_des(9)]."""
        state = State()
        state.position = [0, 0, 1]
        state.quaternion = [1, 0, 0, 0]
        ref = {
            "position": np.array([0, 0, 1]),
            "velocity": np.zeros(3),
        }
        out = controller.update(state.data, ref, dt=0.001)
        assert out.shape == (10,)

    def test_hover_thrust_magnitude(self, controller):
        """At hover equilibrium, thrust ≈ m*g."""
        state = State()
        state.position = [0, 0, 1]
        ref = {"position": np.array([0, 0, 1])}
        out = controller.update(state.data, ref, dt=0.001)
        thrust = out[0]
        np.testing.assert_allclose(thrust, 2.0 * 9.81, atol=0.5)

    def test_hover_R_des_near_identity(self, controller):
        """At hover, desired rotation should be near identity."""
        state = State()
        state.position = [0, 0, 1]
        ref = {"position": np.array([0, 0, 1])}
        out = controller.update(state.data, ref, dt=0.001)
        R_des = out[1:10].reshape(3, 3)
        np.testing.assert_allclose(R_des, np.eye(3), atol=0.1)

    def test_position_error_increases_thrust(self, controller):
        """Being below target should increase thrust."""
        state = State()
        state.position = [0, 0, 0.5]  # below target
        ref = {"position": np.array([0, 0, 1.0])}
        out = controller.update(state.data, ref, dt=0.001)
        thrust = out[0]
        assert thrust > 2.0 * 9.81  # more than hover

    def test_reset_clears_integrator(self, controller):
        state = State()
        state.position = [0, 0, 0.5]
        ref = {"position": np.array([0, 0, 1.0])}
        controller.update(state.data, ref, dt=0.001)
        controller.reset()
        # After reset, integral should be zero
        assert np.allclose(controller._integral, 0)
