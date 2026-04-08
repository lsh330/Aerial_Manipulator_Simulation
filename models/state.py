"""State vector management with named indexing and unit conversions."""

import numpy as np
from dataclasses import dataclass


# State vector layout: [pos(3), vel(3), quat(4), ang_vel(3), joint_pos(2), joint_vel(2)] = 17
STATE_DIM = 17

@dataclass(frozen=True)
class StateIndex:
    """Named indices into the 17-element state vector."""
    POS = slice(0, 3)
    VEL = slice(3, 6)
    QUAT = slice(6, 10)
    ANG_VEL = slice(10, 13)
    JOINT_POS = slice(13, 15)
    JOINT_VEL = slice(15, 17)


IDX = StateIndex()


class State:
    """Wrapper around a numpy state vector with named access.

    Provides property-based access to state components:
        state.position, state.velocity, state.quaternion, etc.
    """

    def __init__(self, data: np.ndarray | None = None) -> None:
        if data is None:
            self._data = np.zeros(STATE_DIM)
            self._data[6] = 1.0  # identity quaternion w=1
        else:
            assert len(data) == STATE_DIM, f"Expected {STATE_DIM} elements, got {len(data)}"
            self._data = np.array(data, dtype=np.float64)

    @property
    def data(self) -> np.ndarray:
        return self._data

    @property
    def position(self) -> np.ndarray:
        """World-frame position [m], shape (3,)."""
        return self._data[IDX.POS]

    @position.setter
    def position(self, val: np.ndarray) -> None:
        self._data[IDX.POS] = val

    @property
    def velocity(self) -> np.ndarray:
        """World-frame linear velocity [m/s], shape (3,)."""
        return self._data[IDX.VEL]

    @velocity.setter
    def velocity(self, val: np.ndarray) -> None:
        self._data[IDX.VEL] = val

    @property
    def quaternion(self) -> np.ndarray:
        """Attitude quaternion [w, x, y, z] (Hamilton convention), shape (4,)."""
        return self._data[IDX.QUAT]

    @quaternion.setter
    def quaternion(self, val: np.ndarray) -> None:
        self._data[IDX.QUAT] = val

    @property
    def angular_velocity(self) -> np.ndarray:
        """Body-frame angular velocity [rad/s], shape (3,)."""
        return self._data[IDX.ANG_VEL]

    @angular_velocity.setter
    def angular_velocity(self, val: np.ndarray) -> None:
        self._data[IDX.ANG_VEL] = val

    @property
    def joint_positions(self) -> np.ndarray:
        """[q1_azimuth, q2_elevation] in [rad], shape (2,)."""
        return self._data[IDX.JOINT_POS]

    @joint_positions.setter
    def joint_positions(self, val: np.ndarray) -> None:
        self._data[IDX.JOINT_POS] = val

    @property
    def joint_velocities(self) -> np.ndarray:
        """[dq1/dt, dq2/dt] in [rad/s], shape (2,)."""
        return self._data[IDX.JOINT_VEL]

    @joint_velocities.setter
    def joint_velocities(self, val: np.ndarray) -> None:
        self._data[IDX.JOINT_VEL] = val

    def euler_angles(self) -> np.ndarray:
        """Convert quaternion to ZYX Euler angles [roll, pitch, yaw] in [rad].

        Returns:
            np.ndarray: shape (3,) — [roll, pitch, yaw] in radians.
        """
        w, x, y, z = self.quaternion
        roll = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
        pitch = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
        yaw = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))
        return np.array([roll, pitch, yaw])

    def rotation_matrix(self) -> np.ndarray:
        """Convert quaternion to 3x3 rotation matrix (body-to-world).

        Returns:
            np.ndarray: shape (3, 3) rotation matrix R such that v_world = R @ v_body.
        """
        w, x, y, z = self.quaternion
        return np.array([
            [1-2*(y*y+z*z), 2*(x*y-w*z),   2*(x*z+w*y)],
            [2*(x*y+w*z),   1-2*(x*x+z*z), 2*(y*z-w*x)],
            [2*(x*z-w*y),   2*(y*z+w*x),   1-2*(x*x+y*y)],
        ])

    def copy(self) -> "State":
        """Return a deep copy of this state."""
        return State(self._data.copy())

    def __repr__(self) -> str:
        rpy = np.degrees(self.euler_angles())
        jd = np.degrees(self.joint_positions)
        return (f"State(pos={self.position}, "
                f"rpy_deg=[{rpy[0]:.1f}, {rpy[1]:.1f}, {rpy[2]:.1f}], "
                f"joints_deg=[{jd[0]:.1f}, {jd[1]:.1f}])")
