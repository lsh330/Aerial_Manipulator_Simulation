"""Integration test: NMPC circular trajectory tracking."""

import numpy as np
import pytest
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from analysis.result_analyzer import ResultAnalyzer

RADIUS, OMEGA, ALT = 0.3, 0.3 * np.pi, 1.0

def circular_reference(t):
    return {"position": np.array([RADIUS*np.cos(OMEGA*t), RADIUS*np.sin(OMEGA*t), ALT]),
            "velocity": np.array([-RADIUS*OMEGA*np.sin(OMEGA*t), RADIUS*OMEGA*np.cos(OMEGA*t), 0]),
            "acceleration": np.array([-RADIUS*OMEGA**2*np.cos(OMEGA*t), -RADIUS*OMEGA**2*np.sin(OMEGA*t), 0]),
            "yaw": 0.0, "joint_positions": np.zeros(2), "joint_velocities": np.zeros(2)}

@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 10.0
    cfg.initial_position = np.array([RADIUS, 0, ALT])
    cfg.initial_velocity = np.array([0, RADIUS * OMEGA, 0])
    cfg.log_interval = 100
    return cfg

class TestCircularTrackingNMPC:
    def test_no_divergence(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(circular_reference)
        assert not np.any(np.isnan(logger.get_states()))

    def test_position_rmse_below_1cm(self, config):
        """NMPC should achieve sub-centimeter tracking."""
        runner = SimulationRunner(config)
        logger = runner.run(circular_reference)
        times = logger.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        rmse = ResultAnalyzer(logger).position_rmse(ref)
        total = np.sqrt(np.sum(rmse**2)) * 100
        assert total < 1.0, f"Total RMSE {total:.3f}cm exceeds 1cm"

    def test_altitude_precise(self, config):
        runner = SimulationRunner(config)
        logger = runner.run(circular_reference)
        times = logger.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        z_rmse = ResultAnalyzer(logger).position_rmse(ref)[2] * 100
        assert z_rmse < 0.05, f"z RMSE {z_rmse:.4f}cm exceeds 0.05cm"
