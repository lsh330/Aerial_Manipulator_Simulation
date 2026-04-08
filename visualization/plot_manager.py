"""Static plot generation for simulation results."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from analysis.data_logger import DataLogger
from models.state import IDX
from visualization.plot_styles import apply_style, COLORS, LINE_STYLES


class PlotManager:
    """Generates publication-quality static plots from simulation data."""

    def __init__(self, logger: DataLogger, dpi: int = 300):
        self._logger = logger
        self._dpi = dpi
        self._time = logger.get_time()
        self._states = logger.get_states()
        self._inputs = logger.get_inputs()
        apply_style()

    def plot_position(self, reference: np.ndarray = None,
                      save_path: Path = None) -> plt.Figure:
        """Plot position (x, y, z) time history."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ["x", "y", "z"]

        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(self._time, self._states[:, i],
                    color=COLORS[label], label=f"${label}$")
            if reference is not None:
                ref = reference[:, i] if reference.ndim > 1 else np.full_like(self._time, reference[i])
                ax.plot(self._time, ref, color=COLORS["reference"],
                        linestyle=LINE_STYLES["reference"], label="ref")
            ax.set_ylabel(f"${label}$ [m]")
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("Position Tracking")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        return fig

    def plot_attitude(self, save_path: Path = None) -> plt.Figure:
        """Plot Euler angles (roll, pitch, yaw) from quaternion."""
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        labels = ["roll", "pitch", "yaw"]

        quats = self._states[:, IDX.QUAT]
        euler = np.zeros((len(self._time), 3))
        for i in range(len(self._time)):
            w, x, y, z = quats[i]
            euler[i, 0] = np.arctan2(2*(w*x + y*z), 1 - 2*(x*x + y*y))
            euler[i, 1] = np.arcsin(np.clip(2*(w*y - z*x), -1, 1))
            euler[i, 2] = np.arctan2(2*(w*z + x*y), 1 - 2*(y*y + z*z))

        euler_deg = np.degrees(euler)

        for i, (ax, label) in enumerate(zip(axes, labels)):
            ax.plot(self._time, euler_deg[:, i], color=COLORS[label], label=f"${label}$")
            ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
            ax.set_ylabel(f"{label} [deg]")
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("Attitude (Euler Angles)")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        return fig

    def plot_joint_angles(self, reference: np.ndarray = None,
                          save_path: Path = None) -> plt.Figure:
        """Plot manipulator joint angles."""
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        labels = ["q1", "q2"]
        names = ["Azimuth $q_1$", "Elevation $q_2$"]

        joints = self._states[:, IDX.JOINT_POS]
        joints_deg = np.degrees(joints)

        for i, (ax, label, name) in enumerate(zip(axes, labels, names)):
            ax.plot(self._time, joints_deg[:, i], color=COLORS[label], label=name)
            if reference is not None:
                ref = reference[:, i] if reference.ndim > 1 else np.full_like(self._time, reference[i])
                ax.plot(self._time, np.degrees(ref), color=COLORS["reference"],
                        linestyle=LINE_STYLES["reference"], label="ref")
            ax.set_ylabel(f"{name} [deg]")
            ax.legend(loc="upper right")

        axes[-1].set_xlabel("Time [s]")
        axes[0].set_title("Manipulator Joint Angles")
        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        return fig

    def plot_control_inputs(self, save_path: Path = None) -> plt.Figure:
        """Plot motor thrusts and joint torques."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

        # Motor thrusts
        for i in range(4):
            ax1.plot(self._time, self._inputs[:, i],
                     color=COLORS[f"motor{i+1}"], label=f"$f_{i+1}$")
        ax1.set_ylabel("Motor Thrust [N]")
        ax1.set_title("Control Inputs")
        ax1.legend(loc="upper right", ncol=4)

        # Joint torques
        ax2.plot(self._time, self._inputs[:, 4], color=COLORS["q1"], label=r"$\tau_{q1}$")
        ax2.plot(self._time, self._inputs[:, 5], color=COLORS["q2"], label=r"$\tau_{q2}$")
        ax2.set_ylabel("Joint Torque [N·m]")
        ax2.set_xlabel("Time [s]")
        ax2.legend(loc="upper right")

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        return fig

    def plot_3d_trajectory(self, save_path: Path = None) -> plt.Figure:
        """Plot 3D flight trajectory."""
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        pos = self._states[:, IDX.POS]
        ax.plot(pos[:, 0], pos[:, 1], pos[:, 2], color=COLORS["x"], linewidth=2)
        ax.scatter(*pos[0], color="green", s=100, label="Start", zorder=5)
        ax.scatter(*pos[-1], color="red", s=100, label="End", zorder=5)

        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("3D Flight Trajectory")
        ax.legend()

        if save_path:
            fig.savefig(save_path, dpi=self._dpi, bbox_inches="tight")
        return fig

    def save_all(self, output_dir: Path, prefix: str = "sim"):
        """Generate and save all standard plots."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.plot_position(save_path=output_dir / f"{prefix}_position.png")
        self.plot_attitude(save_path=output_dir / f"{prefix}_attitude.png")
        self.plot_joint_angles(save_path=output_dir / f"{prefix}_joints.png")
        self.plot_control_inputs(save_path=output_dir / f"{prefix}_controls.png")
        self.plot_3d_trajectory(save_path=output_dir / f"{prefix}_trajectory_3d.png")
        plt.close("all")
