"""Manipulator joint controller: PD + gravity compensation + reaction compensation."""

import numpy as np
from control.base_controller import BaseController
from models.state import State


class ManipulatorController(BaseController):
    """PD + gravity compensation for 2-DOF azimuth-elevation manipulator.

    Reference dict:
        joint_positions: [q1_des, q2_des]
        joint_velocities: [q1_dot_des, q2_dot_des] (optional)
    """

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray,
                 gravity_compensation: bool = True,
                 reaction_compensation: bool = True,
                 max_torque: float = 5.0,
                 link_masses: tuple[float, float] = (0.3, 0.2),
                 link_com_distances: tuple[float, float] = (0.15, 0.125),
                 link1_length: float = 0.3,
                 gravity: float = 9.81):
        super().__init__(saturation_limits=np.array([max_torque, max_torque]))
        self._Kp = np.diag(np.asarray(Kp, dtype=float))
        self._Kd = np.diag(np.asarray(Kd, dtype=float))
        self._gravity_comp = gravity_compensation
        self._reaction_comp = reaction_compensation
        self._m1, self._m2 = link_masses
        self._lc1, self._lc2 = link_com_distances
        self._l1 = link1_length
        self._g = gravity

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
            "omega_body": s.angular_velocity,
        }

    def _compute_control(self, error: dict, dt: float) -> np.ndarray:
        e_q = error["e_q"]
        e_q_dot = error["e_q_dot"]
        q = error["q"]
        R_body = error["R_body"]

        # PD control
        tau = self._Kp @ e_q + self._Kd @ e_q_dot

        # Gravity compensation
        if self._gravity_comp:
            tau += self._compute_gravity_compensation(q, R_body)

        return tau

    def _compute_gravity_compensation(self, q: np.ndarray, R_body: np.ndarray) -> np.ndarray:
        """Compute gravity compensation torque for azimuth-elevation joints.

        G_j = dV/dq_j where V = -g_body^T * (m1*r_c1 + m2*r_c2)
        """
        # Gravity in body frame
        g_world = np.array([0.0, 0.0, -self._g])
        g_body = R_body.T @ g_world

        c1, s1 = np.cos(q[0]), np.sin(q[0])
        c2, s2 = np.cos(q[1]), np.sin(q[1])
        D = self._l1 + self._lc2  # combined distance for link2

        # dV/dq1 (azimuth): only nonzero if gravity has horizontal component in body frame
        alpha = self._m1 * self._lc1 + self._m2 * D
        dV_dq1 = -alpha * s2 * (-g_body[0] * s1 + g_body[1] * c1)

        # dV/dq2 (elevation)
        dV_dq2 = -alpha * (g_body[0] * c1 * c2 + g_body[1] * s1 * c2 + g_body[2] * s2)

        return np.array([dV_dq1, dV_dq2])
