"""Generate a proper control-engineering-style NMPC block diagram.

Follows standard conventions from control textbooks:
  - Rectangular blocks for transfer functions / dynamic systems
  - Circular summing junctions with +/- signs
  - Proper signal flow arrows with labels
  - Dashed feedback paths
  - Plant, Controller, Observer clearly distinguished
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def draw_block(ax, xy, w, h, text_lines, facecolor='white',
               edgecolor='black', linewidth=1.5, fontsize=9, text_color='black',
               bold_title=True):
    """Draw a rectangular transfer-function-style block."""
    x, y = xy
    rect = FancyBboxPatch((x, y), w, h, boxstyle='square,pad=0',
                           facecolor=facecolor, edgecolor=edgecolor,
                           linewidth=linewidth, zorder=2)
    ax.add_patch(rect)
    n = len(text_lines)
    for i, line in enumerate(text_lines):
        yy = y + h - (i + 1) * h / (n + 1)
        weight = 'bold' if (i == 0 and bold_title) else 'normal'
        ax.text(x + w / 2, yy, line, ha='center', va='center',
                fontsize=fontsize, fontweight=weight, color=text_color, zorder=3)


def draw_summing_junction(ax, xy, radius=0.18, signs=None):
    """Draw a circular summing junction with +/- signs."""
    x, y = xy
    circle = Circle((x, y), radius, facecolor='white', edgecolor='black',
                     linewidth=1.5, zorder=3)
    ax.add_patch(circle)
    # Draw cross inside
    ax.plot([x - radius * 0.5, x + radius * 0.5], [y, y],
            color='black', lw=0.7, zorder=4)
    ax.plot([x, x], [y - radius * 0.5, y + radius * 0.5],
            color='black', lw=0.7, zorder=4)
    # Place +/- signs
    if signs:
        offsets = {
            'left': (-radius - 0.12, 0.06),
            'bottom': (0.06, -radius - 0.12),
            'right': (radius + 0.12, 0.06),
            'top': (0.06, radius + 0.12),
        }
        for pos, sign in signs.items():
            if pos in offsets:
                dx, dy = offsets[pos]
                ax.text(x + dx, y + dy, sign, ha='center', va='center',
                        fontsize=9, fontweight='bold', color='black', zorder=4)


def arrow(ax, start, end, color='black', lw=1.3, style='-'):
    """Draw a signal-flow arrow."""
    ax.annotate('', xy=end, xytext=start,
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=style, shrinkA=0, shrinkB=0),
                zorder=2)


def corner_arrow(ax, start, corners, end, color='black', lw=1.3, style='-'):
    """Draw a signal-flow arrow with right-angle corners."""
    points = [start] + corners + [end]
    for i in range(len(points) - 2):
        ax.plot([points[i][0], points[i + 1][0]],
                [points[i][1], points[i + 1][1]],
                color=color, lw=lw, linestyle=style, zorder=1)
    # Arrow at the end
    ax.annotate('', xy=end, xytext=points[-2],
                arrowprops=dict(arrowstyle='->', color=color, lw=lw,
                                linestyle=style, shrinkA=0, shrinkB=0),
                zorder=2)


def signal_label(ax, xy, text, fontsize=8, color='black', ha='center', va='bottom'):
    """Place a signal label near an arrow."""
    ax.text(xy[0], xy[1], text, ha=ha, va=va, fontsize=fontsize,
            color=color, fontstyle='italic', zorder=5)


def main():
    fig, ax = plt.subplots(1, 1, figsize=(18, 8.5), dpi=300)
    ax.set_xlim(-0.5, 17.5)
    ax.set_ylim(-1.0, 7.5)
    ax.set_aspect('equal')
    ax.axis('off')

    # ================================================================
    # Color scheme (muted academic style)
    # ================================================================
    C_REF   = '#F5F0E8'   # warm cream
    C_CTRL  = '#E8F0E8'   # light green
    C_PLANT = '#E8ECF5'   # light blue
    C_EST   = '#FFF5E0'   # warm yellow
    C_MEAS  = '#F5E8F0'   # light mauve

    # ================================================================
    # Row positions
    # ================================================================
    Y_MAIN = 3.5     # main signal path y-center
    Y_FB   = 0.6     # feedback path y-center

    # ================================================================
    # 1. Reference Generator  r(t)
    # ================================================================
    draw_block(ax, (-0.3, Y_MAIN - 0.6), 2.2, 1.2,
               ['Reference Generator',
                r'$r(t) = [p_d,\, v_d,\, \psi_d,\, q_{j,d}]$'],
               facecolor=C_REF, edgecolor='#8B7355', fontsize=8)

    # ================================================================
    # 2. Summing Junction (error computation)
    # ================================================================
    SJ_X = 3.0
    draw_summing_junction(ax, (SJ_X, Y_MAIN), signs={'left': '+', 'bottom': '$-$'})

    # Reference → Summing Junction
    arrow(ax, (1.9, Y_MAIN), (SJ_X - 0.18, Y_MAIN))
    signal_label(ax, (2.45, Y_MAIN + 0.12),
                 r'$x_{\mathrm{ref}}(t)$', fontsize=8, color='#8B4513')

    # ================================================================
    # 3. NMPC Optimizer (Controller)
    # ================================================================
    NMPC_X = 4.0
    NMPC_W = 3.8
    NMPC_H = 2.8
    NMPC_Y = Y_MAIN - NMPC_H / 2

    # Outer NMPC block
    draw_block(ax, (NMPC_X, NMPC_Y), NMPC_W, NMPC_H,
               [''],  # title handled separately
               facecolor=C_CTRL, edgecolor='#2E8B57', linewidth=2.0)

    ax.text(NMPC_X + NMPC_W / 2, NMPC_Y + NMPC_H - 0.25,
            'NMPC Controller', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#1B5E20', zorder=5)

    # Inner sub-block: Cost Function
    draw_block(ax, (NMPC_X + 0.15, NMPC_Y + 1.4), 1.65, 0.95,
               [r'$\min_{U}\; J(x,u)$',
                r'$\sum \|e\|_Q^2 + \|u\|_R^2$',
                r'$+ \, w_a \|q_{\mathrm{err}}\|^2$'],
               facecolor='white', edgecolor='#4CAF50', fontsize=7,
               linewidth=1.0, bold_title=True)

    # Inner sub-block: Prediction Model
    draw_block(ax, (NMPC_X + 2.0, NMPC_Y + 1.4), 1.65, 0.95,
               ['Prediction Model',
                r'$x_{k+1} = f_{\mathrm{RK4}}(x_k, u_k)$',
                r'CasADi AD $\to$ DLL'],
               facecolor='white', edgecolor='#4CAF50', fontsize=7,
               linewidth=1.0)

    # Inner sub-block: IPOPT NLP Solver
    draw_block(ax, (NMPC_X + 0.6, NMPC_Y + 0.2), 2.6, 0.85,
               ['IPOPT Interior-Point Solver',
                r'Analytic $\nabla f,\, \nabla^2\! f$ (CasADi AD)',
                r'Warm start $\cdot$ $N=20$ $\cdot$ $\Delta t=20$ ms'],
               facecolor='white', edgecolor='#4CAF50', fontsize=7,
               linewidth=1.0)

    # Arrow: internal cost → solver
    arrow(ax, (NMPC_X + 0.975, NMPC_Y + 1.4), (NMPC_X + 0.975, NMPC_Y + 1.05),
          color='#4CAF50', lw=0.9)
    # Arrow: internal model → solver
    arrow(ax, (NMPC_X + 2.825, NMPC_Y + 1.4), (NMPC_X + 2.825, NMPC_Y + 1.05),
          color='#4CAF50', lw=0.9)

    # Summing Junction → NMPC
    arrow(ax, (SJ_X + 0.18, Y_MAIN), (NMPC_X, Y_MAIN))
    signal_label(ax, (3.6, Y_MAIN + 0.12), r'$e(t)$', fontsize=8, color='#B22222')

    # ================================================================
    # 4. ZOH (Zero-Order Hold)
    # ================================================================
    ZOH_X = 8.5
    draw_block(ax, (ZOH_X, Y_MAIN - 0.35), 1.0, 0.7,
               ['ZOH', r'$T_s = 20$ ms'],
               facecolor='#F5F5F5', edgecolor='black', fontsize=7)

    # NMPC → ZOH
    arrow(ax, (NMPC_X + NMPC_W, Y_MAIN), (ZOH_X, Y_MAIN))
    signal_label(ax, (8.15, Y_MAIN + 0.12),
                 r'$u^*_0$', fontsize=8, color='#1B5E20')

    # ================================================================
    # 5. Plant (C++ Dynamics Engine)
    # ================================================================
    PLANT_X = 10.2
    PLANT_W = 3.5
    PLANT_H = 2.8
    PLANT_Y = Y_MAIN - PLANT_H / 2

    draw_block(ax, (PLANT_X, PLANT_Y), PLANT_W, PLANT_H,
               [''],
               facecolor=C_PLANT, edgecolor='#1565C0', linewidth=2.0)

    ax.text(PLANT_X + PLANT_W / 2, PLANT_Y + PLANT_H - 0.25,
            'Plant: Aerial Manipulator', ha='center', va='center',
            fontsize=11, fontweight='bold', color='#0D47A1', zorder=5)

    # Inner: Coupled Dynamics
    draw_block(ax, (PLANT_X + 0.15, PLANT_Y + 1.35), 3.2, 1.0,
               ['Coupled Multibody Dynamics',
                r'$M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) = Bu$',
                r'$q \in \mathbb{R}^8:\; [p,\, \phi,\, q_1,\, q_2]$'],
               facecolor='white', edgecolor='#1976D2', fontsize=7.5,
               linewidth=1.0)

    # Inner: Integrator
    draw_block(ax, (PLANT_X + 0.15, PLANT_Y + 0.2), 1.5, 0.85,
               ['RK4 Integrator',
                r'$x(t+\Delta t)$',
                r'$\Delta t = 1$ ms'],
               facecolor='white', edgecolor='#1976D2', fontsize=7,
               linewidth=1.0)

    # Inner: Quaternion Norm
    draw_block(ax, (PLANT_X + 1.85, PLANT_Y + 0.2), 1.5, 0.85,
               ['Quaternion',
                'Normalization',
                r'$q \leftarrow q / \|q\|$'],
               facecolor='white', edgecolor='#1976D2', fontsize=7,
               linewidth=1.0)

    # Arrow: dynamics → integrator
    arrow(ax, (PLANT_X + 1.75, PLANT_Y + 1.35),
          (PLANT_X + 1.75, PLANT_Y + 1.05),
          color='#1976D2', lw=0.9)
    signal_label(ax, (PLANT_X + 1.85, PLANT_Y + 1.15),
                 r'$\dot{x}$', fontsize=7, color='#1565C0', ha='left')

    # Arrow: integrator → quat norm
    arrow(ax, (PLANT_X + 1.65, PLANT_Y + 0.625),
          (PLANT_X + 1.85, PLANT_Y + 0.625),
          color='#1976D2', lw=0.9)

    # ZOH → Plant
    arrow(ax, (ZOH_X + 1.0, Y_MAIN), (PLANT_X, Y_MAIN))
    signal_label(ax, (9.85, Y_MAIN + 0.15),
                 r'$u = [f_{1..4},\, \tau_{q_{1,2}}]$', fontsize=7, color='#0D47A1')

    # ================================================================
    # 6. Output (State)
    # ================================================================
    OUT_X = 14.5
    OUT_Y = Y_MAIN

    # Plant → Output
    arrow(ax, (PLANT_X + PLANT_W, Y_MAIN), (15.8, Y_MAIN))

    signal_label(ax, (14.6, Y_MAIN + 0.15),
                 r'$x(t) \in \mathbb{R}^{17}$', fontsize=8, color='#0D47A1')
    signal_label(ax, (14.6, Y_MAIN - 0.25),
                 r'$[p,\, v,\, q,\, \omega,\, q_j,\, \dot{q}_j]$',
                 fontsize=7, color='#555555')

    # Output node (branch point)
    ax.plot(15.8, Y_MAIN, 'ko', markersize=4, zorder=5)

    # Arrow continuing right (to external/logging)
    arrow(ax, (15.8, Y_MAIN), (17.0, Y_MAIN))
    signal_label(ax, (16.4, Y_MAIN + 0.15), r'$y(t)$', fontsize=8)

    # ================================================================
    # 7. Feedback Path (state feedback)
    # ================================================================
    FB_COLOR = '#C62828'

    # Down from branch point
    corner_arrow(ax, (15.8, Y_MAIN), [(15.8, Y_FB), (SJ_X, Y_FB)],
                 (SJ_X, Y_MAIN - 0.18),
                 color=FB_COLOR, lw=1.3, style='--')

    signal_label(ax, (9.5, Y_FB + 0.15),
                 r'Full-State Feedback: $\hat{x}(t) = x(t)$  (perfect measurement assumed)',
                 fontsize=7.5, color=FB_COLOR)

    # ================================================================
    # 8. Disturbance injection point
    # ================================================================
    DIST_X = 9.6
    DIST_Y = Y_MAIN + 2.2

    # Disturbance summing junction (before plant)
    SJ2_X = 9.85
    SJ2_Y = Y_MAIN
    # (Disturbance arrow from top)
    ax.annotate('', xy=(DIST_X, Y_MAIN + 0.7), xytext=(DIST_X, DIST_Y),
                arrowprops=dict(arrowstyle='->', color='#757575', lw=1.0,
                                linestyle=':', shrinkA=0, shrinkB=0), zorder=1)
    signal_label(ax, (DIST_X + 0.1, DIST_Y - 0.1),
                 r'$d(t)$: unmodeled disturbances', fontsize=7,
                 color='#757575', ha='left', va='top')

    # ================================================================
    # 9. Title and annotations
    # ================================================================
    ax.text(8.5, 7.1,
            'Aerial Manipulator — Unified NMPC Control System Block Diagram',
            ha='center', va='center', fontsize=14, fontweight='bold',
            color='#212121')

    ax.text(8.5, 6.55,
            r'Underactuated system: 8 DOF, 6 inputs ($n_u < n_q$) '
            r'— NMPC implicitly resolves attitude $\leftrightarrow$ translation coupling',
            ha='center', va='center', fontsize=8, color='#616161',
            fontstyle='italic')

    # ================================================================
    # 10. Legend
    # ================================================================
    legend_x, legend_y = 0.0, -0.6
    items = [
        (C_REF,   '#8B7355', 'Reference trajectory'),
        (C_CTRL,  '#2E8B57', 'NMPC controller'),
        (C_PLANT, '#1565C0', 'Plant (C++ dynamics engine)'),
    ]
    for i, (fc, ec, label) in enumerate(items):
        rx = legend_x + i * 4.5
        rect = FancyBboxPatch((rx, legend_y - 0.15), 0.35, 0.3,
                               boxstyle='square,pad=0',
                               facecolor=fc, edgecolor=ec, linewidth=1.2)
        ax.add_patch(rect)
        ax.text(rx + 0.5, legend_y, label, ha='left', va='center',
                fontsize=8, color='#424242')

    # Feedback legend
    rx_fb = legend_x + 3 * 4.5
    ax.plot([rx_fb, rx_fb + 0.35], [legend_y, legend_y],
            color=FB_COLOR, lw=1.3, linestyle='--')
    ax.text(rx_fb + 0.5, legend_y, 'Full-state feedback',
            ha='left', va='center', fontsize=8, color='#424242')

    # ================================================================
    # Save
    # ================================================================
    plt.tight_layout(pad=0.5)
    out = Path(__file__).parent.parent / 'docs' / 'images' / 'control_block_diagram.png'
    plt.savefig(str(out), dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    plt.close()
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
