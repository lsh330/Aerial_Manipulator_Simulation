"""Unit tests for State wrapper class."""

import numpy as np
import pytest
from models.state import State, STATE_DIM, IDX


class TestStateInit:
    def test_default_state(self):
        s = State()
        assert len(s.data) == STATE_DIM
        assert s.quaternion[0] == 1.0  # identity w=1
        np.testing.assert_array_equal(s.position, [0, 0, 0])

    def test_custom_state(self):
        data = np.zeros(STATE_DIM)
        data[2] = 5.0
        data[6] = 1.0
        s = State(data)
        assert s.position[2] == 5.0

    def test_invalid_size_raises(self):
        with pytest.raises(AssertionError):
            State(np.zeros(10))


class TestStateProperties:
    def test_position_setter(self):
        s = State()
        s.position = [1, 2, 3]
        np.testing.assert_array_equal(s.position, [1, 2, 3])

    def test_joint_positions(self):
        s = State()
        s.joint_positions = [0.5, -0.3]
        np.testing.assert_allclose(s.joint_positions, [0.5, -0.3])

    def test_euler_at_identity(self):
        s = State()
        rpy = s.euler_angles()
        np.testing.assert_allclose(rpy, [0, 0, 0], atol=1e-10)

    def test_rotation_matrix_identity(self):
        s = State()
        R = s.rotation_matrix()
        np.testing.assert_allclose(R, np.eye(3), atol=1e-10)

    def test_euler_90deg_roll(self):
        s = State()
        # Quaternion for 90° roll: q = [cos(45°), sin(45°), 0, 0]
        s.quaternion = [np.cos(np.pi/4), np.sin(np.pi/4), 0, 0]
        rpy = s.euler_angles()
        np.testing.assert_allclose(rpy[0], np.pi/2, atol=1e-10)

    def test_copy_independence(self):
        s = State()
        s.position = [1, 2, 3]
        s2 = s.copy()
        s2.position = [4, 5, 6]
        assert s.position[0] == 1.0  # original unchanged


class TestStateRepr:
    def test_repr_contains_pos(self):
        s = State()
        s.position = [1.5, 0, 0]
        assert "1.5" in repr(s)
