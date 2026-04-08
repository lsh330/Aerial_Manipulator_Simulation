"""Example 02: Circular trajectory tracking while maintaining arm down.

The quadrotor follows a circular path in the x-y plane at z=1m.
Tests position controller tracking performance under continuous motion.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from visualization.plot_manager import PlotManager
from analysis.result_analyzer import ResultAnalyzer
from models.output_manager import OutputManager


RADIUS = 0.5       # [m] circle radius
OMEGA = 0.5 * np.pi  # [rad/s] angular rate → period = 4s
ALTITUDE = 1.0      # [m]


def circular_reference(t: float) -> dict:
    """Circular trajectory: x = R*cos(ωt), y = R*sin(ωt), z = 1m."""
    pos = np.array([
        RADIUS * np.cos(OMEGA * t),
        RADIUS * np.sin(OMEGA * t),
        ALTITUDE,
    ])
    vel = np.array([
        -RADIUS * OMEGA * np.sin(OMEGA * t),
         RADIUS * OMEGA * np.cos(OMEGA * t),
        0.0,
    ])
    acc = np.array([
        -RADIUS * OMEGA**2 * np.cos(OMEGA * t),
        -RADIUS * OMEGA**2 * np.sin(OMEGA * t),
        0.0,
    ])
    return {
        "position": pos,
        "velocity": vel,
        "acceleration": acc,
        "yaw": 0.0,
        "joint_positions": np.array([0.0, 0.0]),
        "joint_velocities": np.zeros(2),
    }


def main():
    config = SimulationConfig.from_yaml()
    config.duration = 10.0
    # Start at beginning of circle
    config.initial_position = np.array([RADIUS, 0.0, ALTITUDE])

    runner = SimulationRunner(config)
    print("Running circular trajectory tracking...")
    logger = runner.run(circular_reference, progress_callback=lambda p: print(f"  {p*100:.0f}%"))

    # Build reference trajectory for analysis
    times = logger.get_time()
    ref_pos = np.array([circular_reference(t)["position"] for t in times])

    analyzer = ResultAnalyzer(logger)
    print("\n=== Tracking Performance ===")
    print(f"Position RMSE [m]:  {analyzer.position_rmse(ref_pos)}")

    # Plots
    output = OutputManager()
    plotter = PlotManager(logger, dpi=config.image_dpi)
    plotter.plot_position(reference=ref_pos, save_path=output.simulation_image("circle_position"))
    plotter.plot_attitude(save_path=output.simulation_image("circle_attitude"))
    plotter.plot_control_inputs(save_path=output.simulation_image("circle_controls"))
    plotter.plot_3d_trajectory(save_path=output.simulation_image("circle_trajectory_3d"))
    print("\nPlots saved to output/simulations/images/")


if __name__ == "__main__":
    main()
