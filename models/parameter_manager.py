"""Load and validate simulation parameters from YAML config files."""

import yaml
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field


CONFIG_DIR = Path(__file__).parent.parent / "config"


@dataclass
class QuadrotorConfig:
    mass: float = 1.5
    arm_length: float = 0.25
    inertia: np.ndarray = field(default_factory=lambda: np.diag([0.0347, 0.0458, 0.0977]))
    thrust_coeff: float = 8.54858e-6
    torque_coeff: float = 1.36e-7
    drag_coeff: float = 1.6e-2
    motor_time_constant: float = 0.02
    max_motor_speed: float = 1200.0


@dataclass
class LinkConfig:
    mass: float = 0.0
    length: float = 0.0
    com_distance: float = 0.0
    inertia: np.ndarray = field(default_factory=lambda: np.zeros((3, 3)))


@dataclass
class ManipulatorConfig:
    attachment_offset: np.ndarray = field(default_factory=lambda: np.array([0.0, 0.0, -0.1]))
    link1: LinkConfig = field(default_factory=LinkConfig)
    link2: LinkConfig = field(default_factory=LinkConfig)
    joint_lower_limit: np.ndarray = field(default_factory=lambda: np.array([-np.pi, -np.pi/2]))
    joint_upper_limit: np.ndarray = field(default_factory=lambda: np.array([np.pi, 3*np.pi/4]))
    max_joint_torque: float = 5.0


@dataclass
class EnvironmentConfig:
    gravity: float = 9.81
    air_density: float = 1.225


class ParameterManager:
    """Loads YAML config files and converts to structured parameter objects."""

    def __init__(self, config_dir: str | Path = CONFIG_DIR):
        self.config_dir = Path(config_dir)

    def load_default_params(self) -> tuple[QuadrotorConfig, ManipulatorConfig, EnvironmentConfig]:
        with open(self.config_dir / "default_params.yaml") as f:
            raw = yaml.safe_load(f)

        quad = self._parse_quadrotor(raw["quadrotor"])
        manip = self._parse_manipulator(raw["manipulator"])
        env = EnvironmentConfig(
            gravity=raw["environment"]["gravity"],
            air_density=raw["environment"]["air_density"],
        )
        return quad, manip, env

    def _parse_quadrotor(self, d: dict) -> QuadrotorConfig:
        I = d["inertia"]
        return QuadrotorConfig(
            mass=d["mass"],
            arm_length=d["arm_length"],
            inertia=np.diag([I["Ixx"], I["Iyy"], I["Izz"]]),
            thrust_coeff=d["thrust_coeff"],
            torque_coeff=d["torque_coeff"],
            drag_coeff=d["drag_coeff"],
            motor_time_constant=d["motor_time_constant"],
            max_motor_speed=d["max_motor_speed"],
        )

    def _parse_manipulator(self, d: dict) -> ManipulatorConfig:
        jl = d["joint_limits"]
        return ManipulatorConfig(
            attachment_offset=np.array(d["attachment_offset"]),
            link1=self._parse_link(d["link1"]),
            link2=self._parse_link(d["link2"]),
            joint_lower_limit=np.array([jl["q1_min"], jl["q2_min"]]),
            joint_upper_limit=np.array([jl["q1_max"], jl["q2_max"]]),
            max_joint_torque=d["max_joint_torque"],
        )

    def _parse_link(self, d: dict) -> LinkConfig:
        I = d["inertia"]
        return LinkConfig(
            mass=d["mass"],
            length=d["length"],
            com_distance=d["com_distance"],
            inertia=np.diag([I["Ixx"], I["Iyy"], I["Izz"]]),
        )

    def to_cpp_quadrotor_params(self, cfg: QuadrotorConfig):
        """Convert to C++ QuadrotorParams (requires _core module)."""
        from models._core_stub import make_quadrotor_params
        return make_quadrotor_params(cfg)

    def to_cpp_manipulator_params(self, cfg: ManipulatorConfig):
        """Convert to C++ ManipulatorParams (requires _core module)."""
        from models._core_stub import make_manipulator_params
        return make_manipulator_params(cfg)
