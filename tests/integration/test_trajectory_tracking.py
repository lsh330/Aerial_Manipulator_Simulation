"""Integration test: trajectory tracking performance.

Verifies position tracking RMSE for circular trajectory with both
hierarchical PID and SDRE controllers against acceptance thresholds.
"""

import numpy as np
import pytest
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from analysis.result_analyzer import ResultAnalyzer


RADIUS = 0.3
OMEGA = 0.3 * np.pi
ALTITUDE = 1.0


def circular_reference(t):
    return {
        "position": np.array([RADIUS * np.cos(OMEGA * t),
                               RADIUS * np.sin(OMEGA * t), ALTITUDE]),
        "velocity": np.array([-RADIUS * OMEGA * np.sin(OMEGA * t),
                                RADIUS * OMEGA * np.cos(OMEGA * t), 0]),
        "acceleration": np.array([-RADIUS * OMEGA**2 * np.cos(OMEGA * t),
                                   -RADIUS * OMEGA**2 * np.sin(OMEGA * t), 0]),
        "yaw": 0.0,
        "joint_positions": np.zeros(2),
        "joint_velocities": np.zeros(2),
    }


@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 10.0
    cfg.initial_position = np.array([RADIUS, 0, ALTITUDE])
    cfg.initial_velocity = np.array([0, RADIUS * OMEGA, 0])
    cfg.log_interval = 100
    return cfg


class TestCircularTrackingPID:
    """Circular trajectory tracking with hierarchical PID."""

    def test_no_divergence(self, config):
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(circular_reference)
        assert not np.any(np.isnan(logger.get_states())), "PID diverged"

    def test_position_rmse_below_threshold(self, config):
        """Position RMSE must be below 3cm per axis (PID)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(circular_reference)
        times = logger.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        rmse = ResultAnalyzer(logger).position_rmse(ref)
        assert np.all(rmse < 0.03), f"PID RMSE {rmse*100}cm exceeds 3cm"

    def test_altitude_maintained(self, config):
        """z-axis RMSE must be below 2mm (PID)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(circular_reference)
        times = logger.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        rmse_z = ResultAnalyzer(logger).position_rmse(ref)[2]
        assert rmse_z < 0.002, f"PID z RMSE {rmse_z*100:.3f}cm exceeds 0.2cm"


class TestCircularTrackingSDRE:
    """Circular trajectory tracking with SDRE controller."""

    def test_no_divergence(self, config):
        runner = SimulationRunner(config, controller_mode="sdre")
        logger = runner.run(circular_reference)
        assert not np.any(np.isnan(logger.get_states())), "SDRE diverged"

    def test_position_rmse_below_threshold(self, config):
        """SDRE should achieve below 1.5cm per axis (better than PID)."""
        runner = SimulationRunner(config, controller_mode="sdre")
        logger = runner.run(circular_reference)
        times = logger.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        rmse = ResultAnalyzer(logger).position_rmse(ref)
        assert np.all(rmse < 0.015), f"SDRE RMSE {rmse*100}cm exceeds 1.5cm"

    def test_sdre_outperforms_pid(self, config):
        """SDRE total RMSE must be lower than PID."""
        # PID run
        runner_pid = SimulationRunner(config, controller_mode="hierarchical")
        log_pid = runner_pid.run(circular_reference)
        times = log_pid.get_time()
        ref = np.array([circular_reference(t)["position"] for t in times])
        rmse_pid = np.sqrt(np.sum(ResultAnalyzer(log_pid).position_rmse(ref)**2))

        # SDRE run
        runner_sdre = SimulationRunner(config, controller_mode="sdre")
        log_sdre = runner_sdre.run(circular_reference)
        times_s = log_sdre.get_time()
        ref_s = np.array([circular_reference(t)["position"] for t in times_s])
        rmse_sdre = np.sqrt(np.sum(ResultAnalyzer(log_sdre).position_rmse(ref_s)**2))

        assert rmse_sdre < rmse_pid, (
            f"SDRE ({rmse_sdre*100:.3f}cm) should outperform PID ({rmse_pid*100:.3f}cm)")
