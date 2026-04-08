"""Integration test: coupled dynamics — arm motion disturbs quadrotor.

Verifies that manipulator motion causes measurable but bounded
attitude disturbance, confirming the coupled dynamics are active.
"""

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
    return {
        "position": np.array([0, 0, 1]),
        "velocity": np.zeros(3),
        "acceleration": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([0.0, q2]),
        "joint_velocities": np.zeros(2),
    }


@pytest.fixture
def config():
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 4.0
    cfg.log_interval = 50
    return cfg


class TestCoupledDynamics:
    """Tests confirming coupled dynamics between arm and quadrotor."""

    def test_arm_motion_causes_attitude_disturbance(self, config):
        """Arm motion must cause nonzero attitude error (coupling exists)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(arm_motion_reference)
        analyzer = ResultAnalyzer(logger)
        max_att = np.degrees(analyzer.attitude_error_norm().max())
        assert max_att > 0.1, (
            f"Max attitude error {max_att:.4f}° is too small — coupling may be broken")

    def test_attitude_disturbance_bounded(self, config):
        """Attitude error must stay below 5° (controller compensates)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(arm_motion_reference)
        analyzer = ResultAnalyzer(logger)
        max_att = np.degrees(analyzer.attitude_error_norm().max())
        assert max_att < 5.0, f"Attitude error {max_att:.2f}° exceeds 5° bound"

    def test_position_maintained_during_arm_motion(self, config):
        """Position must stay within 20cm during arm sweep (CoM shift ~4.6cm)."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(arm_motion_reference)
        states = logger.get_states()
        max_pos_error = np.max(np.abs(states[:, :3] - [0, 0, 1]))
        assert max_pos_error < 0.20, f"Position error {max_pos_error*100:.1f}cm exceeds 20cm"

    def test_motor_thrusts_asymmetric_during_arm_motion(self, config):
        """Motor thrusts should become asymmetric to compensate arm torques."""
        runner = SimulationRunner(config, controller_mode="hierarchical")
        logger = runner.run(arm_motion_reference)
        inputs = logger.get_inputs()
        # During arm motion (t=1~2s), motors should be unequal
        n = inputs.shape[0]
        mid_start, mid_end = n // 4, n // 2
        mid_motors = inputs[mid_start:mid_end, :4]
        max_spread = np.max(np.max(mid_motors, axis=1) - np.min(mid_motors, axis=1))
        assert max_spread > 0.5, (
            f"Motor spread {max_spread:.2f}N too small — coupling compensation may be missing")

    def test_sdre_less_disturbance_than_pid(self, config):
        """SDRE should produce less attitude disturbance than PID during arm motion."""
        runner_pid = SimulationRunner(config, controller_mode="hierarchical")
        log_pid = runner_pid.run(arm_motion_reference)
        att_pid = np.degrees(ResultAnalyzer(log_pid).attitude_error_norm().max())

        runner_sdre = SimulationRunner(config, controller_mode="sdre")
        log_sdre = runner_sdre.run(arm_motion_reference)
        att_sdre = np.degrees(ResultAnalyzer(log_sdre).attitude_error_norm().max())

        assert att_sdre < att_pid, (
            f"SDRE att error ({att_sdre:.2f}°) should be less than PID ({att_pid:.2f}°)")
