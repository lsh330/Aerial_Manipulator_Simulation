"""Control allocation: total thrust/torques → individual motor commands + joint torques."""

import numpy as np


class ControlAllocator:
    """Maps desired total thrust + body torques + joint torques to actuator commands.

    Input: F_total (scalar), tau_body (3), tau_joint (2)
    Output: [f1, f2, f3, f4, tau_q1, tau_q2] (InputVector, 6 elements)

    Uses the pseudoinverse of the mixing matrix for rotor allocation.
    """

    def __init__(self, arm_length: float, thrust_coeff: float, torque_coeff: float,
                 max_motor_thrust: float = None):
        self._L = arm_length
        self._k_f = thrust_coeff
        self._k_tau = torque_coeff
        self._max_thrust = max_motor_thrust

        # Build mixing matrix: [F; tau_x; tau_y; tau_z] = A * [f1; f2; f3; f4]
        k = torque_coeff / thrust_coeff
        L = arm_length
        self._mixing = np.array([
            [ 1.0,  1.0,  1.0,  1.0],
            [ 0.0, -L,    0.0,  L  ],
            [ L,    0.0, -L,    0.0],
            [ k,   -k,    k,   -k  ],
        ])
        self._mixing_inv = np.linalg.pinv(self._mixing)

    def allocate(self, thrust_total: float, tau_body: np.ndarray,
                 tau_joint: np.ndarray) -> np.ndarray:
        """Compute actuator commands from desired wrench.

        Args:
            thrust_total: Total desired thrust [N]
            tau_body: Desired body torques [tau_roll, tau_pitch, tau_yaw] [N·m]
            tau_joint: Desired joint torques [tau_q1, tau_q2] [N·m]

        Returns:
            6-element input vector: [f1, f2, f3, f4, tau_q1, tau_q2]
        """
        wrench = np.array([thrust_total, tau_body[0], tau_body[1], tau_body[2]])
        motor_thrusts = self._mixing_inv @ wrench

        # Clamp motor thrusts to non-negative (motors can only push)
        motor_thrusts = np.maximum(motor_thrusts, 0.0)

        # Optional maximum thrust saturation
        if self._max_thrust is not None:
            motor_thrusts = np.minimum(motor_thrusts, self._max_thrust)

        return np.concatenate([motor_thrusts, tau_joint])

    @property
    def mixing_matrix(self) -> np.ndarray:
        return self._mixing

    @property
    def mixing_matrix_inv(self) -> np.ndarray:
        return self._mixing_inv
