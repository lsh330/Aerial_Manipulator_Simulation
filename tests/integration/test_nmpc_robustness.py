"""Test NMPC controller robustness and edge cases.

Verifies that the NMPC controller handles:
    1. Infeasible / difficult initial conditions gracefully (fallback)
    2. Joint position constraint enforcement
    3. Thrust saturation without divergence
    4. Large initial attitude errors

The simulation infrastructure (SimulationRunner, SimulationConfig) is imported
and tests are skipped automatically if it is not available.
"""

import numpy as np
import pytest

# Skip entire module if simulation infrastructure is not present
SimulationRunner = pytest.importorskip(
    "simulation.simulation_runner", reason="simulation package not available"
).SimulationRunner

from simulation.simulation_config import SimulationConfig


# ---------------------------------------------------------------------------
# Reference trajectories
# ---------------------------------------------------------------------------

def _hover_reference(t):
    """Hover at origin, z=1m."""
    return {
        "position": np.array([0.0, 0.0, 1.0]),
        "velocity": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.zeros(2),
        "joint_velocities": np.zeros(2),
    }


def _joint_limit_reference(t):
    """Command joint positions near the upper limit of joint 2."""
    # joint 2 upper limit: 2.36 rad -> command 2.3 rad
    return {
        "position": np.array([0.0, 0.0, 1.0]),
        "velocity": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([0.0, 2.3]),
        "joint_velocities": np.zeros(2),
    }


def _extreme_position_reference(t):
    """Command a step to a far-away position (potential infeasibility)."""
    return {
        "position": np.array([10.0, 10.0, 5.0]),   # very far from start
        "velocity": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.zeros(2),
        "joint_velocities": np.zeros(2),
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def config_short():
    """Short-duration config for fast robustness checks."""
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 1.0
    cfg.log_interval = 200
    return cfg


@pytest.fixture
def config_medium():
    """Medium-duration config for constraint and convergence checks."""
    cfg = SimulationConfig.from_yaml()
    cfg.duration = 3.0
    cfg.log_interval = 100
    return cfg


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestNMPCFallback:
    """Verify that the NMPC does not crash under difficult operating conditions."""

    def test_solver_does_not_crash_at_hover(self, config_short):
        """NMPC must complete hover simulation without exception or NaN."""
        runner = SimulationRunner(config_short)
        logger = runner.run(_hover_reference)
        states = logger.get_states()
        assert not np.any(np.isnan(states)), "NMPC produced NaN at hover"
        assert not np.any(np.isinf(states)), "NMPC produced Inf at hover"

    def test_large_initial_attitude_error_no_divergence(self, config_short):
        """Large initial attitude error must not cause divergence.

        Initialises the quadrotor with a 30-degree roll tilt and verifies
        that the NMPC recovers without the state diverging.
        """
        cfg = config_short
        roll = np.radians(30.0)
        cr, sr = np.cos(roll / 2), np.sin(roll / 2)
        cfg.initial_quaternion = np.array([cr, sr, 0.0, 0.0])   # 30 deg roll

        runner = SimulationRunner(cfg)
        logger = runner.run(_hover_reference)
        states = logger.get_states()

        assert not np.any(np.isnan(states)), "State diverged (NaN) from large roll"
        # Position must remain within a reasonable bound
        max_pos_err = np.max(np.abs(states[:, :3]))
        assert max_pos_err < 5.0, (
            f"Position exceeded 5m bound under large roll perturbation: "
            f"max|pos| = {max_pos_err:.2f}m")

    def test_extreme_reference_no_crash(self, config_short):
        """Commanding an unreachable position must not crash the solver."""
        runner = SimulationRunner(config_short)
        try:
            logger = runner.run(_extreme_position_reference)
            states = logger.get_states()
            # Even if tracking fails, state must not contain NaN/Inf
            assert not np.any(np.isnan(states)), "NaN in state with extreme reference"
        except Exception as exc:
            pytest.fail(
                f"NMPC raised an exception with extreme reference: {type(exc).__name__}: {exc}")


class TestNMPCConstraints:
    """Verify that NMPC respects joint and thrust constraints."""

    def test_joint_limits_respected(self, config_medium):
        """Joint positions must never exceed hardware limits during trajectory.

        joint 1: [-3.14, 3.14] rad
        joint 2: [-1.57, 2.36] rad
        """
        runner = SimulationRunner(config_medium)
        logger = runner.run(_joint_limit_reference)
        states = logger.get_states()

        q1 = states[:, 13]
        q2 = states[:, 14]

        assert np.all(q1 >= -3.14 - 1e-3), (
            f"Joint 1 below lower limit: min = {q1.min():.4f}")
        assert np.all(q1 <=  3.14 + 1e-3), (
            f"Joint 1 above upper limit: max = {q1.max():.4f}")
        assert np.all(q2 >= -1.57 - 1e-3), (
            f"Joint 2 below lower limit: min = {q2.min():.4f}")
        assert np.all(q2 <=  2.36 + 1e-3), (
            f"Joint 2 above upper limit: max = {q2.max():.4f}")

    def test_thrust_inputs_non_negative(self, config_medium):
        """Motor thrusts must never be negative (physical constraint)."""
        runner = SimulationRunner(config_medium)
        logger = runner.run(_hover_reference)
        inputs = logger.get_inputs()

        thrusts = inputs[:, :4]   # columns 0-3: motor thrusts [N]
        assert np.all(thrusts >= -1e-6), (
            f"Negative motor thrust observed: min = {thrusts.min():.4e} N")

    def test_joint_torques_bounded(self, config_medium):
        """Joint torques must stay within max_joint_torque = 5.0 Nm."""
        MAX_TORQUE = 5.0   # [Nm]
        runner = SimulationRunner(config_medium)
        logger = runner.run(_joint_limit_reference)
        inputs = logger.get_inputs()

        joint_torques = inputs[:, 4:6]   # columns 4-5: joint torques [Nm]
        assert np.all(np.abs(joint_torques) <= MAX_TORQUE + 1e-6), (
            f"Joint torque exceeded {MAX_TORQUE} Nm: "
            f"max = {np.abs(joint_torques).max():.4f} Nm")


class TestNMPCHoverPerformance:
    """Regression tests: NMPC must meet minimum performance at hover."""

    def test_hover_position_within_5mm(self, config_medium):
        """At steady state (last 20% of sim), position error must be < 5mm."""
        runner = SimulationRunner(config_medium)
        logger = runner.run(_hover_reference)
        states = logger.get_states()

        n = len(states)
        ss_states = states[int(0.8 * n):]   # last 20%
        err = np.max(np.abs(ss_states[:, :3] - np.array([0, 0, 1])))

        assert err < 0.005, (
            f"Hover steady-state position error {err*1000:.2f}mm exceeds 5mm")

    def test_hover_no_oscillation(self, config_medium):
        """Position standard deviation at steady state must be < 1mm."""
        runner = SimulationRunner(config_medium)
        logger = runner.run(_hover_reference)
        states = logger.get_states()

        n = len(states)
        ss_states = states[int(0.8 * n):]
        std = np.std(ss_states[:, :3], axis=0)
        max_std = np.max(std)

        assert max_std < 0.001, (
            f"Hover oscillation: max std = {max_std*1000:.3f}mm, expected < 1mm")
