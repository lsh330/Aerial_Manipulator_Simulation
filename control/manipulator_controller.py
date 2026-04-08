"""Manipulator joint controller: computed torque with gravity compensation."""

import numpy as np
from control.base_controller import BaseController
from models.state import State


class ManipulatorController(BaseController):
    """Computed Torque / PD + gravity compensation for 2-DOF manipulator.

    Reference dict:
        joint_positions: [q1_des, q2_des]
        joint_velocities: [q1_dot_des, q2_dot_des] (optional)
        end_effector_pos: [x, y, z] (optional, uses IK if provided)
    """

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray,
                 gravity_compensation: bool = True,
                 reaction_compensation: bool = True,
                 max_torque: float = 5.0):
        super().__init__(saturation_limits=np.array([max_torque, max_torque]))
        self._Kp = np.diag(np.asarray(Kp, dtype=float))
        self._Kd = np.diag(np.asarray(Kd, dtype=float))
        self._gravity_comp = gravity_compensation
        self._reaction_comp = reaction_compensation

    def _compute_error(self, state: np.ndarray, reference: dict) -> dict:
        s = State(state)

        q_des = np.asarray(reference["joint_positions"])
        q_dot_des = np.asarray(reference.get("joint_velocities", np.zeros(2)))

        return {
            "e_q": q_des - s.joint_positions,
            "e_q_dot": q_dot_des - s.joint_velocities,
            "q": s.joint_positions,
            "q_dot": s.joint_velocities,
            "R_body": s.rotation_matrix(),
        }

    def _compute_control(self, error: dict, dt: float) -> np.ndarray:
        e_q = error["e_q"]
        e_q_dot = error["e_q_dot"]

        # PD control
        tau = self._Kp @ e_q + self._Kd @ e_q_dot

        return tau
