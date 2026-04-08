"""Integration test: NMPC coupled dynamics — arm motion."""

import numpy as np
import pytest
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from analysis.result_analyzer import ResultAnalyzer

def smooth_step(t, a, b):
    if t <= a: return 0.0
    if t >= b: return 1.0
    s = (t - a) / (b - a)
    return 0.5 * (1 - np.cos(np.pi * s))

def arm_motion_reference(t):
    q2 = np.radians(45) * smooth_step(t, 0.5, 2.5)
    return {"position": np.array([0, 0, 1]), "velocity": np.zeros(3),
            "yaw": 0.0, "joint_positions": np.array([0.0, q2]),
            "joint_velocities": np.zeros(2)}

@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 4.0
    cfg.log_interval = 50
    return cfg

class TestCoupledDynamicsNMPC:
    def test_arm_causes_disturbance(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(arm_motion_reference)
        max_att = np.degrees(ResultAnalyzer(logger).attitude_error_norm().max())
        assert max_att > 0.01, "No attitude disturbance — coupling may be broken"

    def test_disturbance_bounded(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(arm_motion_reference)
        max_att = np.degrees(ResultAnalyzer(logger).attitude_error_norm().max())
        assert max_att < 3.0, f"Attitude error {max_att:.2f}° exceeds 3° bound"

    def test_position_maintained(self, config):
        """NMPC should maintain position within 2cm during arm sweep."""
        runner = SimulationRunner(config)
        logger = runner.run(arm_motion_reference)
        states = logger.get_states()
        max_err = np.max(np.abs(states[:, :3] - [0, 0, 1])) * 100
        assert max_err < 2.0, f"Position error {max_err:.2f}cm exceeds 2cm"
