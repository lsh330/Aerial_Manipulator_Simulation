"""Main simulation loop orchestrating all components (Mediator pattern)."""

import numpy as np
from typing import Callable
from simulation.simulation_config import SimulationConfig
from simulation.time_manager import TimeManager
from models.state import State
from models.system_wrapper import SystemWrapper
from models.output_manager import OutputManager
from control.position_controller import PositionController
from control.attitude_controller import AttitudeController
from control.manipulator_controller import ManipulatorController
from control.control_allocator import ControlAllocator
from analysis.data_logger import DataLogger


class SimulationRunner:
    """Mediator that orchestrates engine, controllers, logger, and time management."""

    def __init__(self, config: SimulationConfig):
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
        self._build_controllers(config)

    def _build_controllers(self, cfg: SimulationConfig):
        import yaml
        from pathlib import Path
        ctrl_path = Path(__file__).parent.parent / "config" / "controller_params.yaml"
        with open(ctrl_path) as f:
            cp = yaml.safe_load(f)

        pc = cp["position_controller"]
        self.position_ctrl = PositionController(
            Kp=pc["gains"]["Kp"], Kd=pc["gains"]["Kd"], Ki=pc["gains"]["Ki"],
            total_mass=self.system.total_mass(),
            gravity=cfg.environment.gravity,
            integrator_limit=pc["integrator_limit"],
        )

        ac = cp["attitude_controller"]
        self.attitude_ctrl = AttitudeController(
            Kr=ac["gains"]["Kr"], Kw=ac["gains"]["Kw"],
            inertia=cfg.quadrotor.inertia,
        )

        mc = cp["manipulator_controller"]
        self.manip_ctrl = ManipulatorController(
            Kp=mc["gains"]["Kp"], Kd=mc["gains"]["Kd"],
            gravity_compensation=mc["gravity_compensation"],
            reaction_compensation=mc["reaction_compensation"],
            max_torque=cfg.manipulator.max_joint_torque,
            link_masses=(cfg.manipulator.link1.mass, cfg.manipulator.link2.mass),
            link_com_distances=(cfg.manipulator.link1.com_distance, cfg.manipulator.link2.com_distance),
            link1_length=cfg.manipulator.link1.length,
            gravity=cfg.environment.gravity,
        )

        self.allocator = ControlAllocator(
            arm_length=cfg.quadrotor.arm_length,
            thrust_coeff=cfg.quadrotor.thrust_coeff,
            torque_coeff=cfg.quadrotor.torque_coeff,
        )

    def run(self, reference_trajectory: Callable[[float], dict],
            progress_callback: Callable[[float], None] | None = None) -> DataLogger:
        """Execute the full simulation.

        Args:
            reference_trajectory: function t → dict with keys:
                position, velocity, acceleration, yaw, joint_positions, joint_velocities
            progress_callback: optional function called with progress fraction [0,1]

        Returns:
            DataLogger with recorded data
        """
        state = State(self.config.initial_state_vector())

        while not self.time_mgr.is_finished():
            t = self.time_mgr.t
            dt = self.time_mgr.dt
            ref = reference_trajectory(t)

            # ── Hierarchical control ──
            # 1. Position → desired thrust + attitude
            pos_out = self.position_ctrl.update(state.data, ref, dt)
            thrust_total = pos_out[0]
            R_des = pos_out[1:10].reshape(3, 3)

            # 2. Attitude → body torques
            att_ref = {"R_des": R_des}
            tau_body = self.attitude_ctrl.update(state.data, att_ref, dt)

            # 3. Manipulator → joint torques
            tau_joint = self.manip_ctrl.update(state.data, ref, dt)

            # 4. Control allocation → actuator commands
            input_vec = self.allocator.allocate(thrust_total, tau_body, tau_joint)

            # ── Log data ──
            if self.time_mgr.should_log():
                self.logger.on_step(t, state.data, input_vec, ref)

            # ── Dynamics integration ──
            state = self.system.step(t, state, input_vec, dt)

            # ── Advance time ──
            self.time_mgr.advance()

            if progress_callback and self.time_mgr.step_count % 1000 == 0:
                progress_callback(self.time_mgr.progress())

        return self.logger
