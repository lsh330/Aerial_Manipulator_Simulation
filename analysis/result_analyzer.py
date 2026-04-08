"""Performance metrics computation from simulation results."""

import numpy as np
from analysis.data_logger import DataLogger
from models.state import IDX


class ResultAnalyzer:
    """Computes performance metrics from logged simulation data."""

    def __init__(self, logger: DataLogger):
        self._logger = logger
        self._time = logger.get_time()
        self._states = logger.get_states()
        self._inputs = logger.get_inputs()

    def position_rmse(self, reference: np.ndarray) -> np.ndarray:
        """RMSE of position tracking error per axis.

        Args:
            reference: (N, 3) reference positions or (3,) constant reference

        Returns:
            [rmse_x, rmse_y, rmse_z]
        """
        pos = self._states[:, IDX.POS]
        if reference.ndim == 1:
            reference = np.tile(reference, (pos.shape[0], 1))
        error = pos - reference
        return np.sqrt(np.mean(error**2, axis=0))

    def attitude_error_norm(self) -> np.ndarray:
        """Time series of attitude error magnitude (angle from identity) [rad]."""
        quats = self._states[:, IDX.QUAT]
        # Angle from identity: angle = 2 * arccos(|qw|)
        qw = np.abs(quats[:, 0])
        qw = np.clip(qw, 0, 1)
        return 2.0 * np.arccos(qw)

    def joint_tracking_rmse(self, reference: np.ndarray) -> np.ndarray:
        """RMSE of joint angle tracking per joint.

        Args:
            reference: (N, 2) or (2,) reference joint angles

        Returns:
            [rmse_q1, rmse_q2]
        """
        joints = self._states[:, IDX.JOINT_POS]
        if reference.ndim == 1:
            reference = np.tile(reference, (joints.shape[0], 1))
        error = joints - reference
        return np.sqrt(np.mean(error**2, axis=0))

    def settling_time(self, signal: np.ndarray, target: float,
                      threshold: float = 0.02) -> float | None:
        """Time when signal permanently enters ±threshold band around target.

        Returns:
            Settling time [s], or None if never settles.
        """
        band = np.abs(signal - target) <= threshold * np.abs(target + 1e-10)
        # Find last time it exits the band
        if not np.any(band):
            return None
        # Check from end
        for i in range(len(band) - 1, -1, -1):
            if not band[i]:
                if i + 1 < len(self._time):
                    return self._time[i + 1]
                return None
        return self._time[0]

    def total_energy(self, total_mass: float, gravity: float) -> np.ndarray:
        """Compute total mechanical energy (kinetic + potential) time series."""
        vel = self._states[:, IDX.VEL]
        pos_z = self._states[:, 2]
        omega = self._states[:, IDX.ANG_VEL]

        KE_trans = 0.5 * total_mass * np.sum(vel**2, axis=1)
        PE = total_mass * gravity * pos_z

        # Rotational KE approximation (using total mass, not exact inertia)
        KE_rot = 0.5 * np.sum(omega**2, axis=1)  # simplified

        return KE_trans + KE_rot + PE

    def control_effort(self) -> dict:
        """Compute control effort metrics."""
        inputs = self._inputs
        dt = np.diff(self._time, prepend=self._time[0])

        return {
            "motor_integral": np.sum(inputs[:, :4]**2 * dt[:, None], axis=0),
            "joint_integral": np.sum(inputs[:, 4:]**2 * dt[:, None], axis=0),
            "max_motor_thrust": np.max(inputs[:, :4], axis=0),
            "max_joint_torque": np.max(np.abs(inputs[:, 4:]), axis=0),
        }

    def summary(self, pos_ref: np.ndarray = None, joint_ref: np.ndarray = None) -> dict:
        """Generate complete performance summary."""
        result = {
            "duration": self._time[-1] - self._time[0],
            "num_steps": len(self._time),
            "control_effort": self.control_effort(),
        }
        if pos_ref is not None:
            result["position_rmse"] = self.position_rmse(pos_ref)
        if joint_ref is not None:
            result["joint_rmse"] = self.joint_tracking_rmse(joint_ref)
        return result
