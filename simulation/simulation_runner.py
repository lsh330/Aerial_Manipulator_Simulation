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
        # Pre-allocated buffers for inner PD loop (avoid per-step allocation)
        _q_err = np.zeros(4)
        _tau_wrench = np.zeros(4)
        _input_buf = np.zeros(6)

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

            # ── Inner PD correction at 1kHz (pre-allocated buffers) ──
            sd = state.data
            qw, qx, qy, qz = _quat_des[0], _quat_des[1], _quat_des[2], _quat_des[3]
            pw, px, py, pz = sd[6], sd[7], sd[8], sd[9]
            _q_err[0] = qw*pw + qx*px + qy*py + qz*pz
            _q_err[1] = qw*px - qx*pw - qy*pz + qz*py
            _q_err[2] = qw*py + qx*pz - qy*pw - qz*px
            _q_err[3] = qw*pz - qx*py + qy*px - qz*pw
            if _q_err[0] < 0:
                _q_err *= -1
            # PD torque → motor thrust correction (in-place)
            _tau_wrench[0] = 0.0
            _tau_wrench[1] = Kp_att[0] * _q_err[1] - Kd_att[0] * sd[10]
            _tau_wrench[2] = Kp_att[1] * _q_err[2] - Kd_att[1] * sd[11]
            _tau_wrench[3] = Kp_att[2] * _q_err[3] - Kd_att[2] * sd[12]
            _input_buf[:] = _nmpc_input_cache
            _input_buf[:4] += A_inv @ _tau_wrench
            np.clip(_input_buf[:4], 0.0, 12.3, out=_input_buf[:4])
            np.clip(_input_buf[4:], -5.0, 5.0, out=_input_buf[4:])
            input_vec = _input_buf
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
