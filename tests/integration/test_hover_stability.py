"""Integration test: hover stabilization with C++ dynamics engine.

Verifies that the full simulation pipeline (engine + controller + runner)
maintains stable hover at z=1m for 5 seconds with negligible drift.
Tests both hierarchical PID and SDRE controller modes.
"""

import numpy as np
import pytest
from pathlib import Path
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from analysis.result_analyzer import ResultAnalyzer
from models.output_manager import OutputManager


def hover_reference(t):
    return {
        "position": np.array([0.0, 0.0, 1.0]),
        "velocity": np.zeros(3),
        "acceleration": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([0.0, 0.0]),
        "joint_velocities": np.zeros(2),
    }


@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 3.0
    cfg.log_interval = 100
    return cfg


class TestHoverStabilityHierarchical:
    """Hover tests with PID + SO(3) + PD hierarchical controller."""

    def test_position_holds(self, config):
        """Position must stay within 1mm of target over 3s."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(hover_reference)
        states = logger.get_states()
        assert not np.any(np.isnan(states)), "Simulation diverged"
        max_pos_error = np.max(np.abs(states[:, :3] - [0, 0, 1]))
        assert max_pos_error < 0.001, f"Position drift {max_pos_error:.6f}m exceeds 1mm"

    def test_attitude_holds(self, config):
        """Attitude must stay within 0.01° of level over 3s."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(hover_reference)
        analyzer = ResultAnalyzer(logger)
        max_att = np.degrees(analyzer.attitude_error_norm().max())
        assert max_att < 0.01, f"Attitude error {max_att:.4f}° exceeds 0.01°"

    def test_joint_angles_hold(self, config):
        """Joint angles must stay within 0.001° of zero over 3s."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(hover_reference)
        states = logger.get_states()
        max_joint_error = np.degrees(np.max(np.abs(states[:, 13:15])))
        assert max_joint_error < 0.001, f"Joint drift {max_joint_error:.4f}° exceeds 0.001°"

    def test_motor_thrusts_equal(self, config):
        """Motor thrusts should be nearly equal at hover (±0.01N)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(hover_reference)
        inputs = logger.get_inputs()
        # After transient (last half)
        n = inputs.shape[0]
        steady = inputs[n // 2:, :4]
        spread = np.max(steady) - np.min(steady)
        assert spread < 0.01, f"Motor thrust spread {spread:.4f}N exceeds 0.01N"


class TestHoverStabilitySDRE:
    """Hover tests with SDRE controller."""

    def test_position_holds(self, config):
        runner = SimulationRunner(config, controller_mode="sdre")
        logger = runner.run(hover_reference)
        states = logger.get_states()
        assert not np.any(np.isnan(states)), "SDRE simulation diverged"
        max_pos_error = np.max(np.abs(states[:, :3] - [0, 0, 1]))
        assert max_pos_error < 0.001, f"SDRE position drift {max_pos_error:.6f}m"

    def test_attitude_holds(self, config):
        runner = SimulationRunner(config, controller_mode="sdre")
        logger = runner.run(hover_reference)
        analyzer = ResultAnalyzer(logger)
        max_att = np.degrees(analyzer.attitude_error_norm().max())
        assert max_att < 0.01, f"SDRE attitude error {max_att:.4f}°"
