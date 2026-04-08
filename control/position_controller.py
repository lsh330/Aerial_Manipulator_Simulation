"""Outer-loop position controller: desired position → thrust vector + desired attitude."""

import numpy as np
from control.base_controller import BaseController
from models.state import State


class PositionController(BaseController):
    """PID position controller with feedforward.

    Computes desired thrust vector F_des in world frame and extracts
    desired attitude (R_des) for the inner attitude loop.

    Reference dict:
        position: [x, y, z]
        velocity: [vx, vy, vz] (optional)
        acceleration: [ax, ay, az] (optional, feedforward)
    """

    def __init__(self, Kp: np.ndarray, Kd: np.ndarray, Ki: np.ndarray,
                 total_mass: float, gravity: float,
                 integrator_limit: float = 2.0):
        super().__init__()
        self._Kp = np.asarray(Kp, dtype=float)
        self._Kd = np.asarray(Kd, dtype=float)
        self._Ki = np.asarray(Ki, dtype=float)
        self._mass = total_mass
        self._gravity = gravity
        self._int_limit = integrator_limit
        self._integral = np.zeros(3)

    def _compute_error(self, state: np.ndarray, reference: dict) -> dict:
        s = State(state)
        pos_ref = np.asarray(reference["position"])
        vel_ref = np.asarray(reference.get("velocity", np.zeros(3)))

        return {
            "pos": pos_ref - s.position,
            "vel": vel_ref - s.velocity,
            "acc_ff": np.asarray(reference.get("acceleration", np.zeros(3))),
            "yaw_des": reference.get("yaw", 0.0),
            "state": s,
        }

    def _compute_control(self, error: dict, dt: float) -> np.ndarray:
        e_pos = error["pos"]
        e_vel = error["vel"]
        acc_ff = error["acc_ff"]
        yaw_des = error["yaw_des"]
        state = error["state"]

        # Integrate position error with anti-windup
        self._integral += e_pos * dt
        self._integral = np.clip(self._integral, -self._int_limit, self._int_limit)

        # Desired acceleration in world frame
        a_des = (self._Kp * e_pos
                 + self._Kd * e_vel
                 + self._Ki * self._integral
                 + acc_ff)

        # Desired thrust vector (world frame)
        # F_des = m * (a_des + g*e3)
        g_vec = np.array([0.0, 0.0, self._gravity])
        F_des = self._mass * (a_des + g_vec)

        # Total thrust magnitude
        thrust_magnitude = np.linalg.norm(F_des)
        if thrust_magnitude < 1e-6:
            thrust_magnitude = self._mass * self._gravity

        # Extract desired attitude from thrust direction
        z_des = F_des / np.linalg.norm(F_des)

        # Desired yaw direction
        x_c = np.array([np.cos(yaw_des), np.sin(yaw_des), 0.0])

        # Construct desired rotation matrix
        y_des = np.cross(z_des, x_c)
        y_norm = np.linalg.norm(y_des)
        if y_norm < 1e-6:
            y_des = np.array([0.0, 1.0, 0.0])
        else:
            y_des /= y_norm
        x_des = np.cross(y_des, z_des)

        R_des = np.column_stack([x_des, y_des, z_des])

        # Pack output: [thrust_magnitude, R_des(9)] = 10 elements
        output = np.zeros(10)
        output[0] = thrust_magnitude
        output[1:10] = R_des.flatten()
        return output

    def reset(self):
        super().reset()
        self._integral = np.zeros(3)
