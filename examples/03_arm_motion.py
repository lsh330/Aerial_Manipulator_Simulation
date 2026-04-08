"""Example 03: Arm motion during hover — tests coupled dynamics and reaction compensation.

The quadrotor hovers at z=1m while the manipulator sweeps:
  Phase 1 (0-3s):  elevation q2 from 0 → 90° (arm goes horizontal)
  Phase 2 (3-6s):  azimuth q1 sweeps 0 → 180° (arm rotates around)
  Phase 3 (6-9s):  both return to [0, 0]

This is the key test for:
- Coupled dynamics: arm motion disturbs quadrotor attitude
- Reaction compensation: attitude controller rejects manipulator torques
- Position maintenance: CoM shift during arm motion

Expected Output (approximate)::

    Position RMSE [m]: ~0.015 (15 mm drift during arm motion)
    Joint tracking RMSE [rad]: ~0.003
    Max attitude perturbation [deg]: ~2.5

Note: Requires built _core module. Run ``bash scripts/build.sh`` first.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from visualization.plot_manager import PlotManager
from visualization.animator import Animator
from analysis.result_analyzer import ResultAnalyzer
from models.output_manager import OutputManager


def smooth_step(t: float, t_start: float, t_end: float) -> float:
    """Smooth step from 0 to 1 over [t_start, t_end] using cosine interpolation."""
    if t <= t_start:
        return 0.0
    if t >= t_end:
        return 1.0
    s = (t - t_start) / (t_end - t_start)
    return 0.5 * (1.0 - np.cos(np.pi * s))


def arm_motion_reference(t: float) -> dict:
    """Hover + arm sweep trajectory."""
    # Position: constant hover
    pos = np.array([0.0, 0.0, 1.0])

    # Joint trajectory
    q1, q2 = 0.0, 0.0

    if t < 3.0:
        # Phase 1: elevation sweep 0 → 45°
        q2 = np.radians(45.0) * smooth_step(t, 0.5, 2.5)
    elif t < 6.0:
        # Phase 2: azimuth sweep 0 → 90°, elevation stays 45°
        q2 = np.radians(45.0)
        q1 = np.radians(90.0) * smooth_step(t, 3.0, 5.5)
    elif t < 9.0:
        # Phase 3: return both to 0
        progress = smooth_step(t, 6.0, 8.5)
        q1 = np.radians(90.0) * (1.0 - progress)
        q2 = np.radians(45.0) * (1.0 - progress)
    else:
        q1, q2 = 0.0, 0.0

    return {
        "position": pos,
        "velocity": np.zeros(3),
        "acceleration": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([q1, q2]),
        "joint_velocities": np.zeros(2),
    }


def main():
    config = SimulationConfig.from_yaml()
    config.duration = 10.0

    runner = SimulationRunner(config)
    print("Running arm motion simulation...")
    logger = runner.run(arm_motion_reference, progress_callback=lambda p: print(f"  {p*100:.0f}%"))

    # Build reference for analysis
    times = logger.get_time()
    ref_joints = np.array([arm_motion_reference(t)["joint_positions"] for t in times])

    analyzer = ResultAnalyzer(logger)
    print("\n=== Arm Motion Performance ===")
    print(f"Position RMSE [m]:      {analyzer.position_rmse(np.array([0.0, 0.0, 1.0]))}")
    print(f"Joint RMSE [rad]:       {analyzer.joint_tracking_rmse(ref_joints)}")
    print(f"Max attitude error [deg]: {np.degrees(analyzer.attitude_error_norm().max()):.2f}")
    print(f"Max motor thrust [N]:   {analyzer.control_effort()['max_motor_thrust']}")

    # Plots
    output = OutputManager()
    plotter = PlotManager(logger, dpi=config.image_dpi)
    plotter.plot_position(
        reference=np.array([0.0, 0.0, 1.0]),
        save_path=output.simulation_image("arm_motion_position"),
    )
    plotter.plot_attitude(save_path=output.simulation_image("arm_motion_attitude"))
    plotter.plot_joint_angles(reference=ref_joints, save_path=output.simulation_image("arm_motion_joints"))
    plotter.plot_control_inputs(save_path=output.simulation_image("arm_motion_controls"))
    plotter.plot_3d_trajectory(save_path=output.simulation_image("arm_motion_trajectory_3d"))

    # Animation
    print("Generating animation...")
    animator = Animator(logger, fps=config.animation_fps)
    animator.create_animation(
        arm_length=config.quadrotor.arm_length,
        link1_length=config.manipulator.link1.length,
        link2_length=config.manipulator.link2.length,
        attachment_offset=config.manipulator.attachment_offset,
        save_path=output.simulation_animation("arm_motion"),
    )
    print("Animation saved to output/simulations/animations/")


if __name__ == "__main__":
    main()
