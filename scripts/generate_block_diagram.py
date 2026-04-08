"""Generate NMPC control block diagram for README."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

fig, ax = plt.subplots(1, 1, figsize=(16, 9), dpi=300)
ax.set_xlim(0, 16)
ax.set_ylim(0, 9)
ax.axis('off')

# Title
ax.text(8, 8.5, 'Aerial Manipulator \u2014 Unified NMPC Control Block Diagram',
        ha='center', va='center', fontsize=16, fontweight='bold')

# Colors
ref_c = '#FFE0E0'
ctrl_c = '#E0FFE0'
plant_c = '#E0E8FF'
state_c = '#FFE8D0'
int_c = '#F0F8E0'
fb_c = '#CC3333'

# ── Reference ──
r = FancyBboxPatch((0.3, 5.5), 2.8, 2.2, boxstyle='round,pad=0.15',
                   facecolor=ref_c, edgecolor='#CC6666', linewidth=1.5)
ax.add_patch(r)
ax.text(1.7, 7.2, 'Reference', ha='center', fontsize=11, fontweight='bold', color='#993333')
ax.text(1.7, 6.7, '$p_d, v_d, \\psi_d$', ha='center', fontsize=9, fontstyle='italic')
ax.text(1.7, 6.3, '$q_{j,d},\\ \\dot{q}_{j,d}$', ha='center', fontsize=9, fontstyle='italic')
ax.text(1.7, 5.85, '(trajectory function)', ha='center', fontsize=7, color='gray')

# ── NMPC Controller (large) ──
n = FancyBboxPatch((4.0, 3.8), 5.5, 4.2, boxstyle='round,pad=0.15',
                   facecolor=ctrl_c, edgecolor='#66AA66', linewidth=2)
ax.add_patch(n)
ax.text(6.75, 7.7, 'NMPC Controller', ha='center', fontsize=13, fontweight='bold', color='#336633')

# Internal: CasADi
c = FancyBboxPatch((4.3, 5.5), 2.2, 1.8, boxstyle='round,pad=0.1',
                   facecolor=int_c, edgecolor='#88AA88', linewidth=1)
ax.add_patch(c)
ax.text(5.4, 7.0, 'CasADi Symbolic', ha='center', fontsize=8, fontweight='bold', color='#446644')
ax.text(5.4, 6.6, 'Dynamics', ha='center', fontsize=8, fontweight='bold', color='#446644')
ax.text(5.4, 6.15, '(exact C++ replica)', ha='center', fontsize=7, color='#668866')
ax.text(5.4, 5.8, 'Compiled DLL', ha='center', fontsize=7, fontweight='bold', color='#886644')

# Internal: IPOPT
ip = FancyBboxPatch((6.8, 5.5), 2.4, 1.8, boxstyle='round,pad=0.1',
                    facecolor=int_c, edgecolor='#88AA88', linewidth=1)
ax.add_patch(ip)
ax.text(8.0, 7.0, 'IPOPT Optimizer', ha='center', fontsize=8, fontweight='bold', color='#446644')
ax.text(8.0, 6.55, '$\\min \\sum \\|x-x_{ref}\\|^2_Q$', ha='center', fontsize=7, color='#446644')
ax.text(8.0, 6.2, '$+ \\|u-u_{ref}\\|^2_R + att$', ha='center', fontsize=7, color='#446644')
ax.text(8.0, 5.8, 'N=20, ~3 iterations', ha='center', fontsize=7, fontweight='bold', color='#886644')

# Internal arrow
ax.annotate('', xy=(6.8, 6.4), xytext=(6.5, 6.4),
            arrowprops=dict(arrowstyle='->', color='#668866', lw=1.2))

# NMPC info
ax.text(6.75, 4.8, '$N=20,\\ \\Delta t_{mpc}=0.02$ s', ha='center', fontsize=8, color='#446644')
ax.text(6.75, 4.3, 'Position + Attitude + Joint\nunified optimization',
        ha='center', fontsize=8, color='#336633', fontstyle='italic')

# ── C++ Dynamics Engine ──
e = FancyBboxPatch((10.5, 4.5), 3.0, 2.8, boxstyle='round,pad=0.15',
                   facecolor=plant_c, edgecolor='#6666AA', linewidth=2)
ax.add_patch(e)
ax.text(12.0, 6.9, 'C++ Dynamics Engine', ha='center', fontsize=11, fontweight='bold', color='#334488')
ax.text(12.0, 6.3, '$M(q)\\ddot{q} + C\\dot{q} + G = Bu$', ha='center', fontsize=9, color='#334488')
ax.text(12.0, 5.7, 'RK4 Integration', ha='center', fontsize=9, color='#5566AA')
ax.text(12.0, 5.2, 'Quaternion Normalization', ha='center', fontsize=8, color='#5566AA')
ax.text(12.0, 4.7, '(Eigen3 + pybind11)', ha='center', fontsize=7, color='gray')

# ── State ──
s = FancyBboxPatch((13.0, 1.5), 2.5, 2.0, boxstyle='round,pad=0.15',
                   facecolor=state_c, edgecolor='#CC8844', linewidth=1.5)
ax.add_patch(s)
ax.text(14.25, 3.1, 'State', ha='center', fontsize=12, fontweight='bold', color='#885522')
ax.text(14.25, 2.6, '$x(t) \\in \\mathbb{R}^{17}$', ha='center', fontsize=9, color='#885522')
ax.text(14.25, 2.1, '$[p, v, q, \\omega, q_j, \\dot{q}_j]$', ha='center', fontsize=8, color='#AA7744')
ax.text(14.25, 1.7, '(17-dim)', ha='center', fontsize=7, color='gray')

# ── Arrows ──
# Reference -> NMPC
ax.annotate('', xy=(4.0, 6.6), xytext=(3.1, 6.6),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
ax.text(3.55, 6.95, '$x_{ref}(t)$', ha='center', fontsize=8, color='#993333')

# NMPC -> Engine
ax.annotate('', xy=(10.5, 5.9), xytext=(9.5, 5.9),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
ax.text(10.0, 6.35, '$u = [f_1..f_4, \\tau_{q_1}, \\tau_{q_2}]$', ha='center', fontsize=8, color='#334488')

# Engine -> State
ax.annotate('', xy=(14.25, 3.5), xytext=(12.0, 4.5),
            arrowprops=dict(arrowstyle='->', color='black', lw=1.8))
ax.text(13.5, 4.2, '$x(t+dt)$', ha='center', fontsize=8, color='#334488')

# State Feedback (red dashed)
ax.annotate('', xy=(4.0, 4.3), xytext=(13.0, 1.8),
            arrowprops=dict(arrowstyle='->', color=fb_c, lw=1.5,
                          linestyle='dashed', connectionstyle='arc3,rad=-0.15'))
ax.text(8.0, 1.3, 'State Feedback: $x = [p, v, q, \\omega, q_j, \\dot{q}_j]$',
        ha='center', fontsize=9, color=fb_c, fontstyle='italic')

# ── Legend ──
items = [(ctrl_c, '#66AA66', 'Controller (NMPC)'),
         (plant_c, '#6666AA', 'Plant (C++)'),
         (ref_c, '#CC6666', 'Reference'),
         (state_c, '#CC8844', 'State')]
for i, (fc, ec, lab) in enumerate(items):
    y = 8.3 - i * 0.35
    rect = FancyBboxPatch((14.2, y - 0.12), 0.3, 0.24, boxstyle='round,pad=0.02',
                          facecolor=fc, edgecolor=ec, linewidth=1)
    ax.add_patch(rect)
    ax.text(14.7, y, lab, ha='left', va='center', fontsize=7)

ax.plot([14.2, 14.5], [6.8, 6.8], color=fb_c, linestyle='dashed', lw=1.2)
ax.text(14.7, 6.8, 'Feedback', ha='left', va='center', fontsize=7)

plt.tight_layout()
out = Path(__file__).parent.parent / 'docs' / 'images' / 'control_block_diagram.png'
plt.savefig(str(out), dpi=300, bbox_inches='tight', facecolor='white', edgecolor='none')
print(f'Saved: {out}')
