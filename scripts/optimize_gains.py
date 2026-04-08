"""Optimize PID gains for circular trajectory tracking via Nelder-Mead."""
import sys, os, warnings
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

import numpy as np
import time
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from control.position_controller import PositionController
from analysis.result_analyzer import ResultAnalyzer
from scipy.optimize import minimize

R_c, W, ALT = 0.3, 0.3 * np.pi, 1.0

def circ_ref(t):
    return {
        "position": np.array([R_c*np.cos(W*t), R_c*np.sin(W*t), ALT]),
        "velocity": np.array([-R_c*W*np.sin(W*t), R_c*W*np.cos(W*t), 0]),
        "acceleration": np.array([-R_c*W**2*np.cos(W*t), -R_c*W**2*np.sin(W*t), 0]),
        "yaw": 0.0, "joint_positions": np.zeros(2), "joint_velocities": np.zeros(2),
    }

config = SimulationConfig.from_yaml()
config.initial_position = np.array([R_c, 0, ALT])
config.initial_velocity = np.array([0, R_c * W, 0])
config.duration = 1.0
config.log_interval = 500

best = [np.inf]
n_eval = [0]

def cost(v):
    n_eval[0] += 1
    g = np.exp(v)
    try:
        runner = SimulationRunner(config)
        runner.position_ctrl = PositionController(
            Kp=[g[0], g[0], g[1]], Kd=[g[2], g[2], g[3]],
            Ki=[g[4], g[4], g[5]],
            total_mass=2.0, gravity=9.81, integrator_limit=2.0,
        )
        logger = runner.run(circ_ref)
        s = logger.get_states()
        if np.any(np.isnan(s)):
            return 1e6
        t = logger.get_time()
        r = np.array([circ_ref(ti)["position"] for ti in t])
        rmse = ResultAnalyzer(logger).position_rmse(r)
        c = np.sum(rmse ** 2) * 100
        if c < best[0]:
            best[0] = c
            total_rmse_cm = np.sqrt(np.sum(rmse ** 2)) * 100
            print(f"[{n_eval[0]:3d}] {total_rmse_cm:.3f}cm "
                  f"Kp=[{g[0]:.1f},{g[1]:.1f}] Kd=[{g[2]:.1f},{g[3]:.1f}] "
                  f"Ki=[{g[4]:.3f},{g[5]:.3f}]", flush=True)
        return c
    except Exception:
        return 1e6

x0 = np.log([6.0, 8.0, 4.5, 5.6, 0.1, 0.2])
t0 = time.time()
print("Optimizing position PID gains (6 params, 1s sim)...")
res = minimize(cost, x0, method="Nelder-Mead",
               options={"maxiter": 50, "adaptive": True})

elapsed = time.time() - t0
g = np.exp(res.x)
print(f"\nDone: {elapsed:.0f}s, {n_eval[0]} evaluations")
print(f"Optimal gains:")
print(f"  Kp: [{g[0]:.3f}, {g[0]:.3f}, {g[1]:.3f}]")
print(f"  Kd: [{g[2]:.3f}, {g[2]:.3f}, {g[3]:.3f}]")
print(f"  Ki: [{g[4]:.4f}, {g[4]:.4f}, {g[5]:.4f}]")
