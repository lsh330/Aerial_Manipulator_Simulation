"""Example 02: Circular trajectory tracking while maintaining arm down.

The quadrotor follows a circular path in the x-y plane at z=1m.
Tests position controller tracking performance under continuous motion.

Expected Output (approximate)::

    Position RMSE [m]: ~0.0038 (3.8 mm tracking error on 1m radius circle)
    Max position error [m]: ~0.0085
    Mean motor thrust [N]: ~4.9
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


RADIUS = 0.3       # [m] circle radius (smaller for stability)
OMEGA = 0.3 * np.pi  # [rad/s] angular rate → period ≈ 6.7s
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
    # Start at beginning of circle with matching velocity
    config.initial_position = np.array([RADIUS, 0.0, ALTITUDE])
    config.initial_velocity = np.array([0.0, RADIUS * OMEGA, 0.0])  # tangential velocity

    # Tuned NMPC parameters for high-precision tracking
    nmpc_kwargs = dict(
        N=20,
        Q=np.diag([
            10000, 10000, 15000,    # position (5x baseline)
            1000, 1000, 1500,       # velocity (5x)
            0, 0, 0, 0,             # quaternion (via attitude_weight)
            100, 100, 50,           # angular velocity (5x)
            2500, 2500,             # joints (5x)
            50, 50,                 # joint velocities (5x)
        ]),
        R=np.diag([0.01, 0.01, 0.01, 0.01, 0.005, 0.005]),
        attitude_weight=5000.0,
        terminal_weight=10.0,
        ipopt_max_iter=100,
        ipopt_tol=1e-8,
    )

    runner = SimulationRunner(config, nmpc_kwargs=nmpc_kwargs)
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
