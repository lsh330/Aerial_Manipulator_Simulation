"""Integration test: NMPC hover stabilization."""

import numpy as np
import pytest
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from analysis.result_analyzer import ResultAnalyzer


def hover_reference(t):
    return {"position": np.array([0.0, 0.0, 1.0]), "velocity": np.zeros(3),
            "yaw": 0.0, "joint_positions": np.zeros(2), "joint_velocities": np.zeros(2)}


@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 2.0
    cfg.log_interval = 200
    return cfg


class TestHoverStabilityNMPC:
    def test_position_holds(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(hover_reference)
        states = logger.get_states()
        assert not np.any(np.isnan(states)), "NMPC diverged"
        max_err = np.max(np.abs(states[:, :3] - [0, 0, 1]))
        assert max_err < 0.001, f"Position drift {max_err:.6f}m exceeds 1mm"

    def test_attitude_holds(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(hover_reference)
        analyzer = ResultAnalyzer(logger)
        max_att = np.degrees(analyzer.attitude_error_norm().max())
        assert max_att < 0.1, f"Attitude error {max_att:.4f}° exceeds 0.1°"

    def test_joints_hold(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(hover_reference)
        states = logger.get_states()
        max_j = np.degrees(np.max(np.abs(states[:, 13:15])))
        assert max_j < 0.01, f"Joint drift {max_j:.4f}° exceeds 0.01°"
