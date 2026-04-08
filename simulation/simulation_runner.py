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

    def __init__(self, config: SimulationConfig) -> None:
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
            config.quadrotor, config.manipulator, config.environment)

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

        while not self.time_mgr.is_finished():
            t = self.time_mgr.t

            # ── NMPC Control (called every nmpc_interval steps) ──
            if _nmpc_input_cache is None or _nmpc_counter % nmpc_interval == 0:
                ref = reference_trajectory(t)
                ref["_t"] = t
                _nmpc_input_cache = compute_control(state.data, ref, dt)
            _nmpc_counter += 1
            input_vec = _nmpc_input_cache

            # ── Log data (reference always fresh for current time) ──
            if should_log():
                ref = reference_trajectory(t)
                log_on_step(t, state.data, input_vec, ref)

            # ── Dynamics integration ──
            state = system_step(t, state, input_vec, dt)

            # ── Advance time ──
            advance()

            if progress_callback and self.time_mgr.step_count % 1000 == 0:
                progress_callback(self.time_mgr.progress())

        return self.logger
