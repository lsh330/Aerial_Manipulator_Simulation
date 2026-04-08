"""Generate a control engineering block diagram for the aerial manipulator."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


def draw_block(ax, xy, w, h, text, color="#D6EAF8", fontsize=8, bold_title=None):
    """Draw a rounded-rectangle block with text."""
    x, y = xy
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.05", linewidth=1.2,
                          edgecolor="#2C3E50", facecolor=color)
    ax.add_patch(box)
    if bold_title:
        ax.text(x, y + h*0.18, bold_title, ha="center", va="center",
                fontsize=fontsize, fontweight="bold", color="#2C3E50")
        ax.text(x, y - h*0.15, text, ha="center", va="center",
                fontsize=fontsize-1, color="#555555", style="italic")
    else:
        ax.text(x, y, text, ha="center", va="center",
                fontsize=fontsize, color="#2C3E50")


def draw_sum(ax, xy, radius=0.12, signs=None):
    """Draw a summing junction circle."""
    x, y = xy
    circle = plt.Circle((x, y), radius, fill=True, facecolor="white",
                         edgecolor="#2C3E50", linewidth=1.5, zorder=5)
    ax.add_patch(circle)
    # Draw + and - inside
    ax.plot([x - radius*0.5, x + radius*0.5], [y, y],
            color="#2C3E50", linewidth=0.8, zorder=6)
    ax.plot([x, x], [y - radius*0.5, y + radius*0.5],
            color="#2C3E50", linewidth=0.8, zorder=6)
    if signs:
        for pos, sign in signs.items():
            dx, dy = {"left": (-radius*1.3, 0), "right": (radius*1.3, 0),
                      "top": (0, radius*1.3), "bottom": (0, -radius*1.3)}[pos]
            ax.text(x + dx, y + dy, sign, ha="center", va="center",
                    fontsize=7, color="#C0392B", fontweight="bold", zorder=6)


def draw_arrow(ax, start, end, text="", text_offset=(0, 0.08), color="#2C3E50"):
    """Draw an arrow between two points with optional label."""
    ax.annotate("", xy=end, xytext=start,
                arrowprops=dict(arrowstyle="-|>", color=color, lw=1.3))
    if text:
        mid_x = (start[0] + end[0]) / 2 + text_offset[0]
        mid_y = (start[1] + end[1]) / 2 + text_offset[1]
        ax.text(mid_x, mid_y, text, ha="center", va="center",
                fontsize=6.5, color="#8E44AD", style="italic")


def main():
    fig, ax = plt.subplots(1, 1, figsize=(16, 7))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Aerial Manipulator — Hierarchical Control Block Diagram",
                 fontsize=13, fontweight="bold", pad=15)

    # ── Row 1: Position Loop (y=4) ──
    # Reference
    draw_block(ax, (0.5, 4), 1.2, 0.7, "$p_d, v_d, a_d$", color="#FADBD8", fontsize=7,
               bold_title="Reference")

    # Sum junction (position error)
    draw_sum(ax, (2.2, 4), signs={"left": "+", "bottom": "−"})

    # Position PID
    draw_block(ax, (4.0, 4), 2.2, 0.9,
               "$K_p e + K_d \\dot{e} + K_i \\int e$", color="#D5F5E3",
               bold_title="Position PID")

    # Desired force / attitude extraction
    draw_block(ax, (7.0, 4), 2.0, 0.9,
               "$R_d = f(F_{des}, \\psi_d)$", color="#FCF3CF",
               bold_title="Attitude Extract")

    # ── Row 2: Attitude Loop (y=2.2) ──
    # Sum junction (attitude error)
    draw_sum(ax, (2.2, 2.2), signs={"left": "+", "bottom": "−"})

    # SO(3) Controller
    draw_block(ax, (4.0, 2.2), 2.2, 0.9,
               "$-K_R e_R - K_\\omega e_\\omega$\n$+ \\omega \\times J\\omega$",
               color="#D5F5E3", bold_title="SO(3) Attitude")

    # ── Row 3: Joint Loop (y=0.5) ──
    # Reference joints
    draw_block(ax, (0.5, 0.5), 1.2, 0.7, "$q_{d}, \\dot{q}_{d}$",
               color="#FADBD8", fontsize=7, bold_title="Joint Ref")

    # Sum junction (joint error)
    draw_sum(ax, (2.2, 0.5), signs={"left": "+", "bottom": "−"})

    # Joint PD + Gravity Comp
    draw_block(ax, (4.0, 0.5), 2.2, 0.9,
               "$K_p^j e_q + K_d^j \\dot{e}_q$\n$+ G_q(q, R)$",
               color="#D5F5E3", bold_title="Joint PD+Grav")

    # ── Control Allocator (y=1.5 between attitude and joint) ──
    draw_block(ax, (7.5, 1.3), 2.0, 1.5,
               "$A_{mix}^{-1}$\n$[F,\\tau] \\to [f_i]$",
               color="#E8DAEF", bold_title="Control\nAllocator")

    # ── Plant (C++ Engine) ──
    draw_block(ax, (11.0, 2.2), 2.4, 2.2,
               "$M(q)\\ddot{q} + C\\dot{q} + G = Bu$\n\nRK4 / RKF45\nQuaternion",
               color="#D6EAF8", bold_title="C++ Dynamics")

    # ── State output ──
    draw_block(ax, (14.0, 2.2), 1.4, 0.7, "$x(t)$", color="#F5CBA7",
               bold_title="State")

    # ── Arrows ──
    # Position path
    draw_arrow(ax, (1.1, 4), (2.08, 4), "$p_d$")
    draw_arrow(ax, (2.32, 4), (2.9, 4), "$e_p$")
    draw_arrow(ax, (5.1, 4), (6.0, 4), "$F_{des}$")
    draw_arrow(ax, (8.0, 4), (8.2, 4))
    # F_des down to allocator
    ax.annotate("", xy=(8.2, 2.05), xytext=(8.2, 3.55),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.3))
    ax.text(8.45, 2.8, "$F_{total}$", fontsize=6.5, color="#8E44AD", style="italic")

    # R_des down to attitude sum
    draw_arrow(ax, (7.0, 3.55), (7.0, 2.9))
    ax.annotate("", xy=(2.2, 2.32), xytext=(7.0, 2.9),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.0,
                                connectionstyle="arc3,rad=-0.2"))
    ax.text(4.5, 3.1, "$R_d$", fontsize=6.5, color="#8E44AD", style="italic")

    # Attitude path
    draw_arrow(ax, (2.32, 2.2), (2.9, 2.2), "$e_R, e_\\omega$")
    draw_arrow(ax, (5.1, 2.2), (6.5, 2.2), "$\\tau_{body}$")
    ax.annotate("", xy=(6.5, 1.8), xytext=(6.5, 2.2),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.3))

    # Joint path
    draw_arrow(ax, (1.1, 0.5), (2.08, 0.5), "$q_d$")
    draw_arrow(ax, (2.32, 0.5), (2.9, 0.5), "$e_q$")
    draw_arrow(ax, (5.1, 0.5), (6.5, 0.5), "$\\tau_{joint}$")
    ax.annotate("", xy=(6.5, 0.8), xytext=(6.5, 0.5),
                arrowprops=dict(arrowstyle="-|>", color="#2C3E50", lw=1.3))

    # Allocator → Plant
    draw_arrow(ax, (8.5, 1.3), (9.8, 1.8), "$u = [f_i, \\tau_j]$", text_offset=(0, 0.15))

    # Plant → State
    draw_arrow(ax, (12.2, 2.2), (13.3, 2.2), "$x(t+dt)$")

    # ── Feedback paths ──
    # State → Position feedback (long bottom loop)
    ax.annotate("", xy=(14, 1.55), xytext=(14, 1.85),
                arrowprops=dict(arrowstyle="-", color="#E74C3C", lw=1.5))
    ax.plot([14, 14, 2.2, 2.2], [1.55, -0.8, -0.8, 3.88],
            color="#E74C3C", linewidth=1.5, linestyle="--")
    ax.annotate("", xy=(2.2, 3.88), xytext=(2.2, 3.5),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.5))
    ax.text(7, -0.6, "State Feedback: $p, v, R, \\omega, q, \\dot{q}$",
            fontsize=7, color="#E74C3C", ha="center", style="italic")

    # State → Attitude feedback
    ax.plot([13.5, 13.5, 2.2], [-0.8, -0.3, -0.3],
            color="#E74C3C", linewidth=1.0, linestyle=":")
    ax.annotate("", xy=(2.2, 2.08), xytext=(2.2, -0.3),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.0))

    # State → Joint feedback
    ax.plot([13, -0.3, 13], [-0.3, 0.0, 0.0],
            color="#E74C3C", linewidth=1.0, linestyle=":")
    ax.annotate("", xy=(2.2, 0.38), xytext=(2.2, 0.0),
                arrowprops=dict(arrowstyle="-|>", color="#E74C3C", lw=1.0))

    # ── Legend ──
    legend_elements = [
        mpatches.Patch(facecolor="#D5F5E3", edgecolor="#2C3E50", label="Controller"),
        mpatches.Patch(facecolor="#D6EAF8", edgecolor="#2C3E50", label="Plant (C++)"),
        mpatches.Patch(facecolor="#FADBD8", edgecolor="#2C3E50", label="Reference"),
        mpatches.Patch(facecolor="#E8DAEF", edgecolor="#2C3E50", label="Allocator"),
        plt.Line2D([0], [0], color="#E74C3C", linestyle="--", label="Feedback"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", fontsize=8,
              framealpha=0.9, edgecolor="#2C3E50")

    plt.tight_layout()
    out_path = os.path.join(os.path.dirname(__file__), "..", "docs", "images", "control_block_diagram.png")
    fig.savefig(out_path, dpi=200, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out_path}")
    plt.close()


if __name__ == "__main__":
    main()
