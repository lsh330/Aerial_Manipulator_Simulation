"""Simulation configuration: load and validate all parameters."""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from models.parameter_manager import (
    ParameterManager, QuadrotorConfig, ManipulatorConfig, EnvironmentConfig,
)


@dataclass
class SimulationConfig:
    """Complete simulation configuration."""

    # Physical parameters
    quadrotor: QuadrotorConfig = field(default_factory=QuadrotorConfig)
    manipulator: ManipulatorConfig = field(default_factory=ManipulatorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    # Simulation settings
    duration: float = 10.0
    dt: float = 0.001
    integrator: str = "rk4"

    # Initial conditions
    initial_position: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, 1.0]))
    initial_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_quaternion: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0, 0.0, 0.0]))
    initial_angular_velocity: np.ndarray = field(default_factory=lambda: np.zeros(3))
    initial_joint_angles: np.ndarray = field(default_factory=lambda: np.zeros(2))
    initial_joint_velocities: np.ndarray = field(default_factory=lambda: np.zeros(2))

    # Output settings
    log_interval: int = 10
    save_format: str = "hdf5"
    image_format: str = "png"
    image_dpi: int = 300
    animation_format: str = "gif"
    animation_fps: int = 30

    @classmethod
    def from_yaml(cls, config_dir: str | Path = "config") -> "SimulationConfig":
        config_dir = Path(config_dir)
        pm = ParameterManager(config_dir)
        quad, manip, env = pm.load_default_params()

        with open(config_dir / "simulation_params.yaml", encoding="utf-8") as f:
            sim_raw = yaml.safe_load(f)

        sim = sim_raw["simulation"]
        ic = sim_raw["initial_conditions"]
        out = sim_raw["output"]

        return cls(
            quadrotor=quad,
            manipulator=manip,
            environment=env,
            duration=sim["duration"],
            dt=sim["dt"],
            integrator=sim["integrator"],
            initial_position=np.array(ic["position"]),
            initial_velocity=np.array(ic["velocity"]),
            initial_quaternion=np.array(ic["orientation"]),
            initial_angular_velocity=np.array(ic["angular_velocity"]),
            initial_joint_angles=np.array(ic["joint_angles"]),
            initial_joint_velocities=np.array(ic["joint_velocities"]),
            log_interval=out["log_interval"],
            save_format=out["save_format"],
            image_format=out["image_format"],
            image_dpi=out["image_dpi"],
            animation_format=out["animation_format"],
            animation_fps=out["animation_fps"],
        )

    def initial_state_vector(self) -> np.ndarray:
        """Build the 17-element initial state vector."""
        return np.concatenate([
            self.initial_position,
            self.initial_velocity,
            self.initial_quaternion,
            self.initial_angular_velocity,
            self.initial_joint_angles,
            self.initial_joint_velocities,
        ])
