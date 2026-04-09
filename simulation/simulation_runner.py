"""Main simulation loop orchestrating all components (Mediator pattern)."""

import numpy as np
from typing import Callable, Optional
from simulation.simulation_config import SimulationConfig
from simulation.time_manager import TimeManager
from models.state import State
from models.system_wrapper import SystemWrapper
from models.output_manager import OutputManager
from analysis.data_logger import DataLogger


class SimulationRunner:
    """Mediator that orchestrates engine, controllers, logger, and time management."""

    def __init__(self, config: SimulationConfig, nmpc_kwargs: dict | None = None) -> None:
        self.config = config
        self.time_mgr = TimeManager(config.dt, config.duration, config.log_interval)
        self.logger = DataLogger()
        self.output = OutputManager()

        # Build dynamics engine
        self.system = SystemWrapper(
            config.quadrotor, config.manipulator, config.environment,
            integrator=config.integrator,
        )

        # Build controllers
        from control.nmpc_controller import NMPCController
        self.nmpc_ctrl = NMPCController(
            config.quadrotor, config.manipulator, config.environment,
            **(nmpc_kwargs or {}))

    def run(
        self,
        reference_trajectory: Callable[[float], dict],
        progress_callback: Optional[Callable[[float], None]] = None,
    ) -> DataLogger:
        """Execute the full simulation.

        Args:
            reference_trajectory: Callable ``t -> dict`` with keys:

                - ``position`` (np.ndarray, shape (3,)): desired position [m]
                - ``velocity`` (np.ndarray, shape (3,)): desired velocity [m/s]
                - ``acceleration`` (np.ndarray, shape (3,)): desired acceleration [m/s²]
                - ``yaw`` (float): desired yaw angle [rad]
                - ``joint_positions`` (np.ndarray, shape (2,)): desired joint angles [rad]
                - ``joint_velocities`` (np.ndarray, shape (2,)): desired joint rates [rad/s]

            progress_callback: Optional callable invoked with progress fraction in [0, 1]
                every 1000 integration steps.

        Returns:
            DataLogger: Logger object containing the full recorded simulation data.
        """
        state = State(self.config.initial_state_vector())
        _nmpc_input_cache: Optional[np.ndarray] = None
        _nmpc_counter = 0

        self.nmpc_ctrl.set_reference_trajectory(reference_trajectory)

        # Pre-compute constant quantities to reduce per-step overhead
        dt = self.time_mgr.dt
        nmpc_interval = max(1, int(self.nmpc_ctrl.dt_mpc / dt))
        should_log = self.time_mgr.should_log
        log_on_step = self.logger.on_step
        system_step = self.system.step
        advance = self.time_mgr.advance
        compute_control = self.nmpc_ctrl.compute_control

        # ── Inner PD attitude correction (multi-rate: 1kHz between NMPC 50Hz) ──
        # Compensates for inter-sample attitude disturbances from arm coupling.
        # Extracts desired quaternion from NMPC reference and applies proportional
        # correction to body torques via mixing matrix inverse.
        L = self.config.quadrotor.arm_length
        k_ratio = self.config.quadrotor.torque_coeff / self.config.quadrotor.thrust_coeff
        # Mixing matrix inverse (maps [F, τ_r, τ_p, τ_y] → [f1, f2, f3, f4])
        A = np.array([
            [1.0,  1.0,  1.0,  1.0],
            [0.0, -L,    0.0,  L  ],
            [L,    0.0, -L,    0.0],
            [k_ratio, -k_ratio, k_ratio, -k_ratio],
        ])
        A_inv = np.linalg.inv(A)
        # PD gains (tuned conservatively to avoid NMPC interference)
        Kp_att = np.array([3.0, 3.0, 1.5])   # [roll, pitch, yaw] proportional
        Kd_att = np.array([1.0, 1.0, 0.3])   # damping

        # _ref_cache holds the most recently evaluated reference dict so that the
        # logging branch can reuse it when it coincides with an NMPC solve step,
        # avoiding a redundant reference_trajectory() call.
        _ref_cache: Optional[dict] = None
        _ref_cache_t: float = -1.0
        _quat_des = np.array([1.0, 0.0, 0.0, 0.0])  # desired quaternion

        while not self.time_mgr.is_finished():
            t = self.time_mgr.t

            # ── NMPC Control (called every nmpc_interval steps) ──
            if _nmpc_input_cache is None or _nmpc_counter % nmpc_interval == 0:
                ref = reference_trajectory(t)
                ref["_t"] = t
                _ref_cache = ref
                _ref_cache_t = t
                _nmpc_input_cache = compute_control(state.data, ref, dt)
                # Extract desired quaternion from NMPC predicted next state
                sol_x = self.nmpc_ctrl._sol_prev
                if sol_x is not None:
                    x1 = np.asarray(sol_x[17:34]).ravel()
                    _quat_des = x1[6:10]
                    _quat_des /= np.linalg.norm(_quat_des)

            # ── Inner PD correction at 1kHz ──
            quat = state.data[6:10]
            omega = state.data[10:13]
            # Quaternion error: q_err = q_des^* ⊗ q
            qw, qx, qy, qz = _quat_des
            pw, px, py, pz = quat
            q_err = np.array([
                qw*pw + qx*px + qy*py + qz*pz,    # w
                qw*px - qx*pw - qy*pz + qz*py,     # x
                qw*py + qx*pz - qy*pw - qz*px,     # y
                qw*pz - qx*py + qy*px - qz*pw,     # z
            ])
            # Sign correction for shortest-path rotation
            if q_err[0] < 0:
                q_err = -q_err
            # PD torque correction (body frame)
            tau_corr = Kp_att * q_err[1:4] + Kd_att * (-omega)
            # Map correction to motor thrusts: Δf = A_inv @ [0, τ_r, τ_p, τ_y]
            delta_f = A_inv @ np.array([0.0, tau_corr[0], tau_corr[1], tau_corr[2]])
            # Apply correction
            input_vec = _nmpc_input_cache.copy()
            input_vec[:4] += delta_f
            input_vec[:4] = np.clip(input_vec[:4], 0.0, 12.3)
            input_vec[4:] = np.clip(input_vec[4:], -5.0, 5.0)
            _nmpc_counter += 1

            # ── Log data (reuse NMPC ref when on same timestep) ──
            if should_log():
                if _ref_cache_t == t:
                    ref_log = _ref_cache
                else:
                    ref_log = reference_trajectory(t)
                    _ref_cache = ref_log
                    _ref_cache_t = t
                log_on_step(t, state.data, input_vec, ref_log)

            # ── Dynamics integration ──
            state = system_step(t, state, input_vec, dt)

            # ── Advance time ──
            advance()

            if progress_callback and self.time_mgr.step_count % 1000 == 0:
                progress_callback(self.time_mgr.progress())

        return self.logger
