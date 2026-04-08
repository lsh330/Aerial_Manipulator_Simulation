"""Unit tests for ControlAllocator."""

import numpy as np
import pytest
from control.control_allocator import ControlAllocator


@pytest.fixture
def allocator():
    return ControlAllocator(
        arm_length=0.25,
        thrust_coeff=8.54858e-6,
        torque_coeff=1.36e-7,
    )


class TestMixingMatrix:
    def test_hover_allocation(self, allocator):
        """Equal thrust on all motors → zero torque."""
        result = allocator.allocate(
            thrust_total=19.62,
            tau_body=np.zeros(3),
            tau_joint=np.zeros(2),
        )
        # Motor thrusts should be roughly equal
        motors = result[:4]
        np.testing.assert_allclose(motors, 19.62/4, atol=0.01)

    def test_pure_roll_torque(self, allocator):
        """Roll torque should be produced by differential motor 2 vs 4."""
        result = allocator.allocate(
            thrust_total=19.62,
            tau_body=np.array([0.5, 0.0, 0.0]),
            tau_joint=np.zeros(2),
        )
        motors = result[:4]
        # Motor 4 should be higher, motor 2 lower (or vice versa)
        assert motors[3] != motors[1]

    def test_joint_torque_passthrough(self, allocator):
        """Joint torques pass through directly."""
        result = allocator.allocate(
            thrust_total=19.62,
            tau_body=np.zeros(3),
            tau_joint=np.array([1.5, -0.8]),
        )
        np.testing.assert_allclose(result[4:], [1.5, -0.8])

    def test_negative_thrust_clipping(self, allocator):
        """Motor thrusts should never be negative."""
        result = allocator.allocate(
            thrust_total=1.0,  # very low total thrust
            tau_body=np.array([5.0, 0.0, 0.0]),  # large torque demand
            tau_joint=np.zeros(2),
        )
        assert np.all(result[:4] >= 0.0)

    def test_mixing_matrix_shape(self, allocator):
        assert allocator.mixing_matrix.shape == (4, 4)
        assert allocator.mixing_matrix_inv.shape == (4, 4)

    def test_mixing_inverse_identity(self, allocator):
        """A * A^{-1} ≈ I."""
        product = allocator.mixing_matrix @ allocator.mixing_matrix_inv
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)
