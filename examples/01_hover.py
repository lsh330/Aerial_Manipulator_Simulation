"""Example 01: Hover stabilization at z=1m with arm hanging down.

Verifies that the system maintains stable hover with zero tracking error.
Plots position, attitude, and control inputs.

Expected Output (approximate)::

    Position RMSE [m]: ~1e-24 (essentially zero -- system stays at hover)
    Attitude error [deg]: ~0.0
    Max motor thrust [N]: ~4.91 (= m*g/4 at hover)
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


def hover_reference(t: float) -> dict:
    """Constant hover reference: position [0,0,1], arm down [0,0]."""
    return {
        "position": np.array([0.0, 0.0, 1.0]),
        "velocity": np.zeros(3),
        "acceleration": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([0.0, 0.0]),
        "joint_velocities": np.zeros(2),
    }


def main():
    config = SimulationConfig.from_yaml()
    config.duration = 5.0  # 5-second hover test

    runner = SimulationRunner(config)
    print("Running hover simulation...")
    logger = runner.run(hover_reference, progress_callback=lambda p: print(f"  {p*100:.0f}%"))

    # Analysis
    analyzer = ResultAnalyzer(logger)
    summary = analyzer.summary(
        pos_ref=np.array([0.0, 0.0, 1.0]),
        joint_ref=np.array([0.0, 0.0]),
    )
    print("\n=== Hover Performance ===")
    print(f"Position RMSE [m]:  {summary['position_rmse']}")
    print(f"Joint RMSE [rad]:   {summary['joint_rmse']}")
    print(f"Max motor thrust:   {summary['control_effort']['max_motor_thrust']} N")

    # Plots
    output = OutputManager()
    plotter = PlotManager(logger, dpi=config.image_dpi)
    plotter.plot_position(
        reference=np.array([0.0, 0.0, 1.0]),
        save_path=output.simulation_image("hover_position"),
    )
    plotter.plot_attitude(save_path=output.simulation_image("hover_attitude"))
    plotter.plot_control_inputs(save_path=output.simulation_image("hover_controls"))
    plotter.plot_3d_trajectory(save_path=output.simulation_image("hover_trajectory_3d"))
    print("\nPlots saved to output/simulations/images/")


if __name__ == "__main__":
    main()
