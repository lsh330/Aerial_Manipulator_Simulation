"""Facade wrapping the C++ dynamics engine with a Pythonic API.

When the C++ module is not built, falls back to a pure-Python implementation
for development and testing.
"""

import numpy as np
from models.state import State, STATE_DIM
from models.parameter_manager import (
    ParameterManager, QuadrotorConfig, ManipulatorConfig, EnvironmentConfig,
)

# Try importing the compiled C++ module
try:
    import os, sys
    # MinGW runtime DLLs needed on Windows
    _mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.isdir(_mingw_bin):
        os.add_dll_directory(_mingw_bin)
    # Ensure models/ is on sys.path for _core.pyd
    _models_dir = os.path.dirname(os.path.abspath(__file__))
    if _models_dir not in sys.path:
        sys.path.insert(0, _models_dir)
    import _core
    HAS_CPP_ENGINE = True
except ImportError:
    HAS_CPP_ENGINE = False


class SystemWrapper:
    """Pythonic facade for the aerial manipulator dynamics engine.

    Wraps either the C++ engine (_core) or a pure-Python fallback,
    presenting a uniform interface.
    """

    def __init__(
        self,
        quad_cfg: QuadrotorConfig,
        manip_cfg: ManipulatorConfig,
        env_cfg: EnvironmentConfig,
        integrator: str = "rk4",
    ):
        self._quad_cfg = quad_cfg
        self._manip_cfg = manip_cfg
        self._env_cfg = env_cfg

        if HAS_CPP_ENGINE:
            self._init_cpp_engine(integrator)
        else:
            self._init_python_fallback(integrator)

    def _init_cpp_engine(self, integrator: str):
        # Build C++ parameter structs
        qp = _core.QuadrotorParams()
        qp.mass = self._quad_cfg.mass
        qp.inertia = self._quad_cfg.inertia
        qp.arm_length = self._quad_cfg.arm_length
        qp.thrust_coeff = self._quad_cfg.thrust_coeff
        qp.torque_coeff = self._quad_cfg.torque_coeff
        qp.drag_coeff = self._quad_cfg.drag_coeff
        qp.motor_time_constant = self._quad_cfg.motor_time_constant
        qp.max_motor_speed = self._quad_cfg.max_motor_speed

        mp = _core.ManipulatorParams()
        mp.attachment_offset = self._manip_cfg.attachment_offset
        mp.link1 = _core.LinkParams()
        mp.link1.mass = self._manip_cfg.link1.mass
        mp.link1.length = self._manip_cfg.link1.length
        mp.link1.com_distance = self._manip_cfg.link1.com_distance
        mp.link1.inertia = self._manip_cfg.link1.inertia
        mp.link2 = _core.LinkParams()
        mp.link2.mass = self._manip_cfg.link2.mass
        mp.link2.length = self._manip_cfg.link2.length
        mp.link2.com_distance = self._manip_cfg.link2.com_distance
        mp.link2.inertia = self._manip_cfg.link2.inertia
        mp.joint_lower_limit = self._manip_cfg.joint_lower_limit
        mp.joint_upper_limit = self._manip_cfg.joint_upper_limit
        mp.max_joint_torque = self._manip_cfg.max_joint_torque

        ep = _core.EnvironmentParams()
        ep.gravity = self._env_cfg.gravity
        ep.air_density = self._env_cfg.air_density

        self._system = _core.AerialManipulatorSystem(qp, mp, ep)

        if integrator == "rk4":
            self._system.set_integrator(_core.RK4Integrator())
        elif integrator == "rkf45":
            self._system.set_integrator(_core.RKF45Integrator())
        self._use_cpp = True

    def _init_python_fallback(self, integrator: str):
        """Pure-Python fallback when C++ engine is not available."""
        self._use_cpp = False
        self._integrator_name = integrator

    def step(self, t: float, state: State, input_vec: np.ndarray, dt: float) -> State:
        """Advance the system by one time step."""
        if self._use_cpp:
            new_data = self._system.step(t, state.data, input_vec, dt)
            return State(new_data)
        else:
            return self._python_step(t, state, input_vec, dt)

    def compute_state_derivative(self, state: State, input_vec: np.ndarray) -> np.ndarray:
        if self._use_cpp:
            return self._system.compute_state_derivative(state.data, input_vec)
        else:
            raise NotImplementedError("Python fallback derivative not yet implemented")

    def total_mass(self) -> float:
        if self._use_cpp:
            return self._system.total_mass()
        return (self._quad_cfg.mass
                + self._manip_cfg.link1.mass
                + self._manip_cfg.link2.mass)

    def hover_input(self) -> np.ndarray:
        """Compute input vector for steady hover (arm down, level)."""
        m_total = self.total_mass()
        g = self._env_cfg.gravity
        thrust_per_motor = m_total * g / 4.0
        return np.array([
            thrust_per_motor, thrust_per_motor,
            thrust_per_motor, thrust_per_motor,
            0.0, 0.0,  # zero joint torques
        ])

    @property
    def has_cpp_engine(self) -> bool:
        return self._use_cpp

    def _python_step(self, t, state, input_vec, dt):
        """Minimal RK4 fallback for testing without C++ build."""
        raise NotImplementedError(
            "C++ engine not available. Build with: scripts/build.sh"
        )
