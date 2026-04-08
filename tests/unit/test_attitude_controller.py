"""Unit tests for AttitudeController."""

import numpy as np
import pytest
from control.attitude_controller import AttitudeController
from models.state import State


@pytest.fixture
def controller():
    return AttitudeController(
        Kr=[8.0, 8.0, 4.0],
        Kw=[0.8, 0.8, 0.5],
        inertia=np.diag([0.0347, 0.0458, 0.0977]),
    )


class TestAttitudeController:
    def test_zero_error_zero_torque(self, controller):
        """At identity with zero angular velocity → zero torque."""
        state = State()
        ref = {"R_des": np.eye(3)}
        tau = controller.update(state.data, ref, dt=0.001)
        np.testing.assert_allclose(tau, np.zeros(3), atol=1e-10)

    def test_roll_error_produces_roll_torque(self, controller):
        """Small roll error → negative roll torque (restoring)."""
        state = State()
        # Small roll of 5°
        angle = np.radians(5)
        state.quaternion = [np.cos(angle/2), np.sin(angle/2), 0, 0]
        ref = {"R_des": np.eye(3)}
        tau = controller.update(state.data, ref, dt=0.001)
        # Roll torque should be restoring (negative for positive roll error)
        assert tau[0] < 0

    def test_output_shape(self, controller):
        state = State()
        ref = {"R_des": np.eye(3)}
        tau = controller.update(state.data, ref, dt=0.001)
        assert tau.shape == (3,)

    def test_vee_map(self):
        """Vee map extracts vector from skew-symmetric matrix."""
        S = np.array([[0, -3, 2], [3, 0, -1], [-2, 1, 0]])
        v = AttitudeController._vee(S)
        np.testing.assert_array_equal(v, [1, 2, 3])
