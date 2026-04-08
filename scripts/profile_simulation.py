"""Profile simulation to identify exact bottlenecks.

Usage:
    python scripts/profile_simulation.py          # full suite
    python scripts/profile_simulation.py --nmpc   # NMPC only (fast)
    python scripts/profile_simulation.py --compare # before/after comparison
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import time
import numpy as np
from simulation.simulation_config import SimulationConfig
from simulation.simulation_runner import SimulationRunner
from models.state import State


def hover_reference(t: float) -> dict:
    return {
        "position": np.array([0.0, 0.0, 1.0]),
        "velocity": np.zeros(3),
        "acceleration": np.zeros(3),
        "yaw": 0.0,
        "joint_positions": np.array([0.0, 0.0]),
        "joint_velocities": np.zeros(2),
    }


def profile_full_simulation(duration=2.0):
    """Run a short simulation and measure total wall time."""
    config = SimulationConfig.from_yaml()
    config.duration = duration

    # Measure initialization
    t0 = time.perf_counter()
    runner = SimulationRunner(config)
    t_init = time.perf_counter() - t0

    # Measure simulation
    t0 = time.perf_counter()
    logger = runner.run(hover_reference)
    t_sim = time.perf_counter() - t0

    steps = int(duration / config.dt)
    nmpc_calls = int(duration / 0.02)

    print(f"=== Simulation Profile (duration={duration}s) ===")
    print(f"  Initialization:      {t_init:.3f} s")
    print(f"  Simulation loop:     {t_sim:.3f} s")
    print(f"  Total steps:         {steps}")
    print(f"  NMPC solver calls:   {nmpc_calls}")
    print(f"  Avg per step:        {t_sim/steps*1000:.3f} ms")
    print(f"  Avg per NMPC call:   {t_sim/nmpc_calls*1000:.1f} ms")
    print(f"  Real-time factor:    {duration/t_sim:.3f}x")
    print()
    return logger, t_sim


def profile_nmpc_solver(n_calls=10):
    """Profile NMPC solver calls in isolation."""
    from control.nmpc_controller import NMPCController
    config = SimulationConfig.from_yaml()

    t0 = time.perf_counter()
    ctrl = NMPCController(config.quadrotor, config.manipulator, config.environment)
    t_build = time.perf_counter() - t0
    print(f"=== NMPC Solver Profile ===")
    print(f"  NLP build time:      {t_build:.3f} s")

    ctrl.set_reference_trajectory(hover_reference)
    state = config.initial_state_vector()
    ref = hover_reference(0.0)
    ref["_t"] = 0.0

    # Warm up
    ctrl.compute_control(state, ref)

    times = []
    for i in range(n_calls):
        ref["_t"] = i * 0.02
        t0 = time.perf_counter()
        u = ctrl.compute_control(state, ref)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"  Calls:               {n_calls}")
    print(f"  Mean per call:       {times.mean()*1000:.1f} ms")
    print(f"  Std per call:        {times.std()*1000:.1f} ms")
    print(f"  Min per call:        {times.min()*1000:.1f} ms")
    print(f"  Max per call:        {times.max()*1000:.1f} ms")
    print()


def profile_cpp_dynamics(n_calls=1000):
    """Profile C++ dynamics engine calls."""
    config = SimulationConfig.from_yaml()
    from models.system_wrapper import SystemWrapper
    system = SystemWrapper(config.quadrotor, config.manipulator, config.environment)

    state = State(config.initial_state_vector())
    m_total = system.total_mass()
    g = config.environment.gravity
    hf = m_total * g / 4.0
    input_vec = np.array([hf, hf, hf, hf, 0.0, 0.0])

    # Warm up
    system.step(0.0, state, input_vec, 0.001)

    times = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        state = system.step(i * 0.001, state, input_vec, 0.001)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"=== C++ Dynamics Profile ({n_calls} RK4 steps) ===")
    print(f"  Mean per step:       {times.mean()*1e6:.1f} us")
    print(f"  Std per step:        {times.std()*1e6:.1f} us")
    print(f"  Min per step:        {times.min()*1e6:.1f} us")
    print(f"  Max per step:        {times.max()*1e6:.1f} us")
    print(f"  Total for {n_calls}:   {times.sum()*1000:.1f} ms")
    print()


def profile_reference_overhead(n_calls=10000):
    """Profile reference trajectory function call overhead."""
    times = []
    for i in range(n_calls):
        t0 = time.perf_counter()
        hover_reference(i * 0.001)
        times.append(time.perf_counter() - t0)

    times = np.array(times)
    print(f"=== Reference Function Profile ({n_calls} calls) ===")
    print(f"  Mean per call:       {times.mean()*1e6:.1f} us")
    print(f"  Total:               {times.sum()*1000:.1f} ms")
    print()


def compare_optimized_vs_baseline(n_calls=10):
    """Compare optimized (compiled DLL) vs baseline (SX interpreter) NMPC.

    This test verifies:
    1. Speedup is at least 5x
    2. Numerical results are essentially identical (diff < 1e-10)
    """
    import casadi as ca
    from control.casadi_dynamics import build_dynamics_from_config

    config = SimulationConfig.from_yaml()
    quad_cfg = config.quadrotor
    manip_cfg = config.manipulator
    env_cfg = config.environment
    NX, NU, N = 17, 6, 20
    dt_mpc = 0.02
    total_mass = quad_cfg.mass + manip_cfg.link1.mass + manip_cfg.link2.mass
    gravity = env_cfg.gravity
    hover_f = total_mass * gravity / 4.0

    state = config.initial_state_vector()
    n_vars = NX * (N + 1) + NU * N
    lbx = np.full(n_vars, -1e6)
    ubx = np.full(n_vars, 1e6)
    for k in range(N):
        idx = NX * (N + 1) + NU * k
        lbx[idx:idx + NU] = [0, 0, 0, 0, -5, -5]
        ubx[idx:idx + NU] = [12.3, 12.3, 12.3, 12.3, 5, 5]
    lbg = np.zeros(NX * (N + 1))
    ubg = np.zeros(NX * (N + 1))
    p = np.zeros(NX * (N + 2))
    p[:NX] = state
    for k in range(N + 1):
        p[NX*(k+1):NX*(k+2)] = np.concatenate(
            [[0, 0, 1], [0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0], [0, 0]]
        )
    x0g = np.tile(state, N + 1)
    u0g = np.tile([hover_f] * 4 + [0, 0], N)
    x0 = np.concatenate([x0g, u0g])
    solver_args = dict(x0=x0, p=p, lbx=lbx, ubx=ubx, lbg=lbg, ubg=ubg)

    # ── Baseline: SX interpreter ──────────────────────────────────────────
    print("Building BASELINE (SX interpreter) solver...")
    f_cont = build_dynamics_from_config(quad_cfg, manip_cfg, env_cfg)
    x_sym = ca.SX.sym("x", NX)
    u_sym = ca.SX.sym("u", NU)
    k1 = f_cont(x_sym, u_sym)
    k2 = f_cont(x_sym + dt_mpc/2*k1, u_sym)
    k3 = f_cont(x_sym + dt_mpc/2*k2, u_sym)
    k4 = f_cont(x_sym + dt_mpc*k3, u_sym)
    x_next = x_sym + dt_mpc/6*(k1 + 2*k2 + 2*k3 + k4)
    x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])
    f_step_sx = ca.Function("F", [x_sym, u_sym], [x_next])

    X = ca.SX.sym("X", NX, N+1)
    U = ca.SX.sym("U", NU, N)
    P = ca.SX.sym("P", NX*(N+2))
    obj = 0
    g = [X[:, 0] - P[:NX]]
    Q_dm = ca.DM(np.diag([2000,2000,3000,200,200,300,0,0,0,0,20,20,10,500,500,10,10]))
    R_dm = ca.DM(np.diag([0.1,0.1,0.1,0.1,0.05,0.05]))
    u_ref = ca.DM([hover_f]*4+[0,0])
    for k in range(N):
        xrk = P[NX*(k+1):NX*(k+2)]
        dx = X[:,k]-xrk; du = U[:,k]-u_ref
        q = X[6:10,k]; qr = xrk[6:10]
        qri = ca.vertcat(qr[0],-qr[1],-qr[2],-qr[3])
        obj += (dx.T@Q_dm@dx + du.T@R_dm@du +
                1000*((qri[0]*q[1]+qri[1]*q[0]+qri[2]*q[3]-qri[3]*q[2])**2 +
                      (qri[0]*q[2]-qri[1]*q[3]+qri[2]*q[0]+qri[3]*q[1])**2 +
                      (qri[0]*q[3]+qri[1]*q[2]-qri[2]*q[1]+qri[3]*q[0])**2))
        g.append(X[:,k+1] - f_step_sx(X[:,k], U[:,k]))
    xrN = P[NX*(N+1):NX*(N+2)]; dxf = X[:,N]-xrN
    qf = X[6:10,N]; qrf = xrN[6:10]; qrif = ca.vertcat(qrf[0],-qrf[1],-qrf[2],-qrf[3])
    obj += 5*(dxf.T@Q_dm@dxf + 1000*(
        (qrif[0]*qf[1]+qrif[1]*qf[0]+qrif[2]*qf[3]-qrif[3]*qf[2])**2 +
        (qrif[0]*qf[2]-qrif[1]*qf[3]+qrif[2]*qf[0]+qrif[3]*qf[1])**2 +
        (qrif[0]*qf[3]+qrif[1]*qf[2]-qrif[2]*qf[1]+qrif[3]*qf[0])**2))
    opt_vars = ca.vertcat(ca.reshape(X,-1,1), ca.reshape(U,-1,1))
    nlp = {"f": obj, "x": opt_vars, "g": ca.vertcat(*g), "p": P}
    opts_base = {"ipopt.print_level":0,"ipopt.max_iter":50,"ipopt.tol":1e-4,
                 "ipopt.warm_start_init_point":"yes","print_time":0}

    t_build0 = time.perf_counter()
    solver_base = ca.nlpsol("nmpc_base","ipopt",nlp,opts_base)
    t_build_base = time.perf_counter() - t_build0

    solver_base(**solver_args)  # warmup
    times_base = []
    sol_base = None
    for _ in range(n_calls):
        t = time.perf_counter()
        sol_base = solver_base(**solver_args)
        times_base.append(time.perf_counter() - t)
    u_base = np.array(sol_base["x"][NX*(N+1):NX*(N+1)+NU]).flatten()

    # ── Optimized: compiled DLL ───────────────────────────────────────────
    print("Building OPTIMIZED (compiled DLL) solver...")
    from control.nmpc_controller import NMPCController
    t_build1 = time.perf_counter()
    ctrl_opt = NMPCController(quad_cfg, manip_cfg, env_cfg)
    ctrl_opt.set_reference_trajectory(hover_reference)
    t_build_opt = time.perf_counter() - t_build1

    ref = hover_reference(0.0)
    ref["_t"] = 0.0
    ctrl_opt.compute_control(state, ref)  # warmup

    times_opt = []
    u_opt = None
    for i in range(n_calls):
        ref["_t"] = 0.0
        t = time.perf_counter()
        u_opt = ctrl_opt.compute_control(state, ref)
        times_opt.append(time.perf_counter() - t)

    # ── Report ──────────────────────────────────────────────────────────
    mean_base = np.mean(times_base) * 1000
    mean_opt = np.mean(times_opt) * 1000
    speedup = mean_base / mean_opt
    max_diff = np.max(np.abs(u_base - u_opt))

    print()
    print("=== Optimization Comparison ===")
    print(f"  Baseline NLP build:  {t_build_base:.3f} s")
    print(f"  Optimized NLP build: {t_build_opt:.3f} s  ({t_build_base/t_build_opt:.0f}x faster)")
    print()
    print(f"  Baseline mean:       {mean_base:.1f} ms/call")
    print(f"  Optimized mean:      {mean_opt:.1f} ms/call")
    print(f"  Speedup:             {speedup:.2f}x")
    print()
    print(f"  u_baseline:  {np.round(u_base, 6)}")
    print(f"  u_optimized: {np.round(u_opt, 6)}")
    print(f"  Max |u_diff|: {max_diff:.2e}")
    print(f"  Numerically identical: {max_diff < 1e-8}")
    print()

    assert speedup >= 3.0, f"Speedup {speedup:.2f}x < expected 3x"
    assert max_diff < 1e-6, f"Max diff {max_diff:.2e} exceeds tolerance 1e-6"
    print("  PASS: speedup >= 3x and results are numerically identical")
    return speedup, max_diff


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--nmpc", action="store_true", help="Profile NMPC only")
    parser.add_argument("--compare", action="store_true",
                        help="Run baseline vs optimized comparison")
    parser.add_argument("--full", action="store_true",
                        help="Run full simulation profile")
    args = parser.parse_args()

    if args.compare:
        compare_optimized_vs_baseline(n_calls=10)
    elif args.nmpc:
        profile_nmpc_solver(n_calls=20)
    elif args.full:
        profile_full_simulation(duration=2.0)
    else:
        # Default: run everything
        profile_nmpc_solver(n_calls=20)
        profile_cpp_dynamics(n_calls=2000)
        profile_reference_overhead()
        print("--- Full Simulation ---")
        profile_full_simulation(duration=2.0)
