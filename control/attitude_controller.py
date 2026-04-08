"""Inner-loop attitude controller: SO(3) geometric control on rotation matrices."""

import numpy as np
from control.base_controller import BaseController
from models.state import State


class AttitudeController(BaseController):
    """Geometric attitude controller on SO(3).

    Avoids Euler angle singularities by working directly with rotation matrices.
    Based on: Lee, Leok, McClamroch (2010) "Geometric Tracking Control of a
    Quadrotor UAV on SE(3)"

    Reference dict:
        R_des: 3x3 desired rotation matrix (from position controller)
    """

    def __init__(self, Kr: np.ndarray, Kw: np.ndarray, inertia: np.ndarray):
        super().__init__()
        self._Kr = np.diag(np.asarray(Kr, dtype=float))
        self._Kw = np.diag(np.asarray(Kw, dtype=float))
        self._J = np.asarray(inertia, dtype=float)

    def _compute_error(self, state: np.ndarray, reference: dict) -> dict:
        s = State(state)
        R = s.rotation_matrix()
        R_des = reference["R_des"]
        omega = s.angular_velocity

        # SO(3) attitude error (vee map of skew-symmetric error)
        e_R_mat = 0.5 * (R_des.T @ R - R.T @ R_des)
        e_R = self._vee(e_R_mat)

        # Angular velocity error (body frame)
        omega_des = reference.get("omega_des", np.zeros(3))
        e_omega = omega - R.T @ R_des @ omega_des

        return {
            "e_R": e_R,
            "e_omega": e_omega,
            "omega": omega,
        }

    def _compute_control(self, error: dict, dt: float) -> np.ndarray:
        e_R = error["e_R"]
        e_omega = error["e_omega"]
        omega = error["omega"]

        # Geometric control law:
        # tau = -Kr * e_R - Kw * e_omega + omega × (J * omega)
        tau = (-self._Kr @ e_R
               - self._Kw @ e_omega
               + np.cross(omega, self._J @ omega))

        return tau

    @staticmethod
    def _vee(S: np.ndarray) -> np.ndarray:
        """Vee map: extract vector from skew-symmetric matrix."""
        return np.array([S[2, 1], S[0, 2], S[1, 0]])
