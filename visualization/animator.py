"""3D animation of aerial manipulator simulation."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d.art3d import Line3D
from pathlib import Path
from analysis.data_logger import DataLogger
from models.state import State, IDX
from visualization.plot_styles import apply_style


class Animator:
    """3D real-time animation of quadrotor + manipulator."""

    def __init__(self, logger: DataLogger, fps: int = 30):
        self._logger = logger
        self._fps = fps
        self._time = logger.get_time()
        self._states = logger.get_states()
        apply_style()

    def create_animation(self, arm_length: float = 0.25,
                         link1_length: float = 0.3,
                         link2_length: float = 0.25,
                         attachment_offset: np.ndarray = None,
                         save_path: Path = None,
                         speed: float = 1.0) -> FuncAnimation:
        """Create 3D animation of the aerial manipulator.

        Args:
            arm_length: Quadrotor arm length [m]
            link1_length, link2_length: Manipulator link lengths [m]
            attachment_offset: Joint1 position in body frame [m]
            save_path: Path to save animation (gif/mp4)
            speed: Playback speed multiplier
        """
        if attachment_offset is None:
            attachment_offset = np.array([0.0, 0.0, -0.1])

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection="3d")

        # Determine plot limits
        pos = self._states[:, IDX.POS]
        margin = 1.0
        x_lim = [pos[:, 0].min() - margin, pos[:, 0].max() + margin]
        y_lim = [pos[:, 1].min() - margin, pos[:, 1].max() + margin]
        z_lim = [max(0, pos[:, 2].min() - margin), pos[:, 2].max() + margin]

        # Initialize plot elements
        quad_lines = [ax.plot([], [], [], 'b-', linewidth=2)[0] for _ in range(2)]
        link1_line, = ax.plot([], [], [], 'r-', linewidth=3)
        link2_line, = ax.plot([], [], [], 'm-', linewidth=3)
        trail_line, = ax.plot([], [], [], 'gray', linewidth=0.5, alpha=0.5)
        time_text = ax.text2D(0.05, 0.95, '', transform=ax.transAxes, fontsize=12)

        # Subsample for target fps
        dt_sim = self._time[1] - self._time[0] if len(self._time) > 1 else 0.001
        step = max(1, int(1.0 / (self._fps * dt_sim * speed)))
        frames = range(0, len(self._time), step)

        def _rotation_matrix(state_vec):
            s = State(state_vec)
            return s.rotation_matrix()

        def update(frame_idx):
            i = list(frames)[frame_idx]
            state = self._states[i]
            s = State(state)
            R = s.rotation_matrix()
            p = s.position

            # Quadrotor arms (X configuration)
            L = arm_length
            motor_pos_body = [
                np.array([L, 0, 0]), np.array([0, -L, 0]),
                np.array([-L, 0, 0]), np.array([0, L, 0]),
            ]
            motor_pos_world = [p + R @ mp for mp in motor_pos_body]

            # Draw X arms
            quad_lines[0].set_data_3d(
                [motor_pos_world[0][0], p[0], motor_pos_world[2][0]],
                [motor_pos_world[0][1], p[1], motor_pos_world[2][1]],
                [motor_pos_world[0][2], p[2], motor_pos_world[2][2]],
            )
            quad_lines[1].set_data_3d(
                [motor_pos_world[1][0], p[0], motor_pos_world[3][0]],
                [motor_pos_world[1][1], p[1], motor_pos_world[3][1]],
                [motor_pos_world[1][2], p[2], motor_pos_world[3][2]],
            )

            # Manipulator
            q1, q2 = s.joint_positions
            c1, s1 = np.cos(q1), np.sin(q1)
            c2, s2 = np.cos(q2), np.sin(q2)

            joint1_world = p + R @ attachment_offset
            link1_end_body = attachment_offset + link1_length * np.array([c1*s2, s1*s2, -c2])
            joint2_world = p + R @ link1_end_body
            link2_end_body = link1_end_body + link2_length * np.array([c1*s2, s1*s2, -c2])
            ee_world = p + R @ link2_end_body

            link1_line.set_data_3d(
                [joint1_world[0], joint2_world[0]],
                [joint1_world[1], joint2_world[1]],
                [joint1_world[2], joint2_world[2]],
            )
            link2_line.set_data_3d(
                [joint2_world[0], ee_world[0]],
                [joint2_world[1], ee_world[1]],
                [joint2_world[2], ee_world[2]],
            )

            # Trail
            trail_end = min(i + 1, len(self._states))
            trail = self._states[:trail_end, IDX.POS]
            trail_line.set_data_3d(trail[:, 0], trail[:, 1], trail[:, 2])

            time_text.set_text(f"t = {self._time[i]:.2f} s")

            return quad_lines + [link1_line, link2_line, trail_line, time_text]

        ax.set_xlim(x_lim)
        ax.set_ylim(y_lim)
        ax.set_zlim(z_lim)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_zlabel("Z [m]")
        ax.set_title("Aerial Manipulator Simulation")

        anim = FuncAnimation(fig, update, frames=len(list(frames)),
                             interval=1000/self._fps, blit=False)

        if save_path:
            save_path = Path(save_path)
            if save_path.suffix == ".gif":
                anim.save(save_path, writer="pillow", fps=self._fps)
            else:
                anim.save(save_path, writer="ffmpeg", fps=self._fps)

        return anim
