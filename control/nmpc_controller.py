"""NMPC controller using CasADi symbolic dynamics with analytic Jacobians.

The dynamics are built symbolically (casadi_dynamics.py) to exactly replicate
the C++ engine's M(q), C(q,qdot), G(q), B(q). CasADi's automatic differentiation
provides exact analytic Jacobians for IPOPT -- no finite differences needed.

State (17): identical to C++ engine [pos, vel, quat, omega, q_j, qd_j]
Input (6):  [f1, f2, f3, f4, tau_q1, tau_q2]

Optimization approach:
    1. CasADi CodeGenerator: f_step + fwd/rev/fwd-rev derivatives -> compiled DLL
       (one-time cost ~70-90s, then persistent across sessions via cache)
    2. MX-based NLP using ca.external() -> IPOPT solves with compiled code
    3. nlpsol build time: ~0.17s (vs 7.1s for SX interpreter) = 42x faster
    4. Per-call solve: ~46ms (vs 415ms) = ~9x speedup (with trajectory shift warm-start)
    Result is numerically identical to the SX interpreter (diff < 1e-13).
"""

import os
import subprocess
import tempfile
import hashlib
import shutil
from pathlib import Path

import numpy as np
import casadi as ca
from control.casadi_dynamics import build_dynamics_from_config


# ── GCC path: prefer MSYS2 MinGW, fall back to PATH ─────────────────────────
_MINGW_GCC = r"C:\msys64\mingw64\bin\gcc.exe"
_GCC = _MINGW_GCC if os.path.exists(_MINGW_GCC) else "gcc"

# ── Persistent DLL cache directory (must be ASCII-only path for CasADi) ──────
# CasADi's DllLibrary cannot handle non-ASCII (e.g. Korean) paths on Windows.
# Use LOCALAPPDATA which is always an ASCII-safe path.
_LOCALAPPDATA = os.environ.get("LOCALAPPDATA", tempfile.gettempdir())
_CACHE_DIR = Path(_LOCALAPPDATA) / "aerial_manipulator_nmpc"


def _ensure_gcc_in_path():
    """Add MinGW bin to PATH so gcc is discoverable by subprocesses."""
    mingw_bin = r"C:\msys64\mingw64\bin"
    if os.path.exists(mingw_bin) and mingw_bin not in os.environ.get("PATH", ""):
        os.environ["PATH"] = mingw_bin + ";" + os.environ.get("PATH", "")


def _compile_dll(c_file: Path, dll_file: Path, second_order: bool = True) -> bool:
    """Compile a CasADi-generated C file into a shared library with MinGW.

    Args:
        c_file: Path to the generated .c file.
        dll_file: Output DLL path.
        second_order: If True use -O2 (avoids gcc memory issues for large files),
                      otherwise -O3.

    Returns:
        True on success, False on failure.
    """
    _ensure_gcc_in_path()
    opt_flag = "-O2" if second_order else "-O3"
    cmd = [
        _GCC, opt_flag, "-march=native",
        "-shared", "-fPIC",
        str(c_file), "-o", str(dll_file), "-lm",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"[nmpc] DLL compile failed:\n{result.stderr[:500]}")
        return False
    return True


def _build_f_step_dll(
    quad_cfg, manip_cfg, env_cfg,
    N: int, dt_mpc: float,
    cache_dir: Path,
    n_substeps: int = 1,
) -> tuple[ca.Function, Path]:
    """Build RK4 step function, generate C code with full 2nd-order derivatives,
    compile to DLL, and cache the result.

    The DLL is keyed by a hash of the physical parameters so it is reused
    across Python sessions whenever the model is unchanged.

    Args:
        n_substeps: Number of RK4 sub-steps within each MPC step.
            With n_substeps=K, each dt_mpc interval is integrated using K
            RK4 steps of size dt_mpc/K. This eliminates discretization
            mismatch between prediction model and simulation engine.

    Returns:
        (f_step_SX, dll_path)  — f_step_SX is used for correctness checks;
        dll_path is used to create ca.external().
    """
    from control.casadi_dynamics import build_dynamics_from_config

    NX, NU = 17, 6

    # ── Build symbolic step function ──────────────────────────────────────
    f_cont = build_dynamics_from_config(quad_cfg, manip_cfg, env_cfg)
    x_sym = ca.SX.sym("x", NX)
    u_sym = ca.SX.sym("u", NU)

    if n_substeps == 1:
        # Direct single-step RK4: avoids CasADi loop overhead in SX graph,
        # which would otherwise double the size of the generated C code and
        # slow IPOPT function evaluations by ~40%.
        k1 = f_cont(x_sym, u_sym)
        k2 = f_cont(x_sym + dt_mpc / 2 * k1, u_sym)
        k3 = f_cont(x_sym + dt_mpc / 2 * k2, u_sym)
        k4 = f_cont(x_sym + dt_mpc * k3, u_sym)
        x_next = x_sym + dt_mpc / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    else:
        h_sub = dt_mpc / n_substeps
        x_k = x_sym
        for _ in range(n_substeps):
            k1 = f_cont(x_k, u_sym)
            k2 = f_cont(x_k + h_sub / 2 * k1, u_sym)
            k3 = f_cont(x_k + h_sub / 2 * k2, u_sym)
            k4 = f_cont(x_k + h_sub * k3, u_sym)
            x_k = x_k + h_sub / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        x_next = x_k
    # Quaternion normalization
    x_next[6:10] = x_next[6:10] / ca.norm_2(x_next[6:10])

    # CSE (Common Subexpression Elimination): reduces SX graph instructions
    # by ~8%, shrinking generated C code and speeding up IPOPT function evals.
    # Numerical results are identical (diff < machine epsilon).
    x_next_cse = ca.cse(x_next)
    f_step = ca.Function("F", [x_sym, u_sym], [x_next_cse])

    # ── Compute a hash of ALL parameters that affect dynamics ────────────
    # Every physical parameter used by build_dynamics_from_config must be
    # included so that a parameter change invalidates the cached DLL.
    param_str = (
        f"{quad_cfg.mass:.6f}_{quad_cfg.arm_length:.6f}_{quad_cfg.thrust_coeff:.6e}_"
        f"{quad_cfg.torque_coeff:.6e}_{quad_cfg.drag_coeff:.6e}_"
        f"{np.array2string(np.asarray(quad_cfg.inertia).ravel(), precision=8)}_"
        f"{manip_cfg.link1.mass:.6f}_{manip_cfg.link1.length:.6f}_"
        f"{manip_cfg.link1.com_distance:.6f}_"
        f"{np.array2string(np.asarray(manip_cfg.link1.inertia).ravel(), precision=8)}_"
        f"{manip_cfg.link2.mass:.6f}_{manip_cfg.link2.length:.6f}_"
        f"{manip_cfg.link2.com_distance:.6f}_"
        f"{np.array2string(np.asarray(manip_cfg.link2.inertia).ravel(), precision=8)}_"
        f"{np.array2string(np.asarray(manip_cfg.attachment_offset), precision=6)}_"
        f"{env_cfg.gravity:.6f}_{dt_mpc:.6f}_{ca.__version__}_cse1"
        + (f"_sub{n_substeps}" if n_substeps > 1 else "")
    )
    # N is excluded from the hash because the DLL contains only the single-step
    # function F(x,u)->x_next which is independent of the horizon length.
    # _cse1 suffix marks that CSE is applied to the symbolic graph before codegen.
    param_hash = hashlib.sha256(param_str.encode()).hexdigest()[:16]
    dll_name = f"f_step_{param_hash}.dll"
    c_name = f"f_step_{param_hash}.c"

    cache_dir.mkdir(parents=True, exist_ok=True)
    dll_path = cache_dir / dll_name
    c_path = cache_dir / c_name

    if dll_path.exists():
        # Validate cached DLL: must be non-empty and loadable
        if dll_path.stat().st_size > 0:
            try:
                ca.external("F", str(dll_path))
                print(f"[nmpc] Using cached DLL: {dll_path.name}")
                return f_step, dll_path
            except Exception:
                print(f"[nmpc] Cached DLL corrupt, recompiling...")
                dll_path.unlink(missing_ok=True)
        else:
            dll_path.unlink(missing_ok=True)

    # ── Verify the cache_dir path is ASCII-safe ───────────────────────────
    try:
        str(cache_dir).encode("ascii")
    except UnicodeEncodeError:
        raise RuntimeError(
            f"[nmpc] Cache directory '{cache_dir}' contains non-ASCII characters. "
            "CasADi cannot load DLLs from non-ASCII paths on Windows. "
            "Set a different cache path by modifying _CACHE_DIR in nmpc_controller.py."
        )

    # ── Generate C code with 1st and 2nd order AD ─────────────────────────
    print("[nmpc] Generating C code for f_step (with 2nd-order AD)...")
    f_fwd = f_step.forward(1)
    f_rev = f_step.reverse(1)
    f_fwd_rev = f_rev.forward(1)   # 2nd order: needed for exact Hessian

    cg = ca.CodeGenerator(c_name.replace(".c", ""))
    cg.add(f_step)
    cg.add(f_fwd)
    cg.add(f_rev)
    cg.add(f_fwd_rev)
    cg.generate(str(cache_dir) + "/")

    # ── Compile ──────────────────────────────────────────────────────────
    print(f"[nmpc] Compiling DLL (this happens once per model configuration)...")
    import time as _time
    t0 = _time.perf_counter()
    ok = _compile_dll(c_path, dll_path, second_order=True)
    elapsed = _time.perf_counter() - t0
    if ok:
        print(f"[nmpc] DLL compiled in {elapsed:.1f}s -> {dll_path.name}")
    else:
        raise RuntimeError(
            "[nmpc] DLL compilation failed. "
            "Check that gcc is at C:/msys64/mingw64/bin/gcc.exe"
        )

    return f_step, dll_path


class NMPCController:
    """Nonlinear MPC with analytic Jacobians via CasADi AD.

    Uses pre-compiled C code (DLL) for the RK4 dynamics integrator step,
    enabling IPOPT to call O2-optimized native code instead of the CasADi
    SX interpreter for every function evaluation. Combined with trajectory
    shift warm-starting, this gives a ~9x speedup with numerically identical
    results (difference < 1e-13).
    """

    NX = 17
    NU = 6

    def __init__(self, quad_cfg, manip_cfg, env_cfg,
                 N: int = 20, dt_mpc: float = 0.02,
                 Q: np.ndarray = None, R: np.ndarray = None,
                 attitude_weight: float = 1000.0,
                 terminal_weight: float = 5.0,
                 ipopt_max_iter: int = 50,
                 ipopt_tol: float = 1e-4,
                 n_substeps: int = 1,
                 hessian_approximation: str = "limited-memory"):
        self._N = N
        self._dt_mpc = dt_mpc
        self._total_mass = quad_cfg.mass + manip_cfg.link1.mass + manip_cfg.link2.mass
        self._gravity = env_cfg.gravity
        self._attitude_weight = attitude_weight
        self._terminal_weight = terminal_weight
        self._ipopt_max_iter = ipopt_max_iter
        self._ipopt_tol = ipopt_tol
        self._hessian_approximation = hessian_approximation

        if Q is None:
            Q = self._default_Q()
        if R is None:
            R = self._default_R()
        self._Q = Q
        self._R = R

        self._ref_func = None
        self._u_prev = None
        self._sol_prev = None  # Full previous solution for trajectory shift

        # Build and compile f_step DLL (cached; only compiled once per model)
        self._f_step_sx, dll_path = _build_f_step_dll(
            quad_cfg, manip_cfg, env_cfg, N, dt_mpc, _CACHE_DIR,
            n_substeps=n_substeps,
        )
        self._f_step_ext = ca.external("F", str(dll_path))

        self._build_nlp()

    def _default_Q(self):
        return np.diag([
            2000, 2000, 3000,  # position
            200, 200, 300,     # velocity
            0, 0, 0, 0,       # quaternion (attitude via quaternion error cost)
            20, 20, 10,        # angular velocity
            500, 500,          # joint angles
            10, 10,            # joint velocities
        ])

    def _default_R(self):
        return np.diag([0.1, 0.1, 0.1, 0.1, 0.05, 0.05])

    def _build_nlp(self):
        N = self._N
        nx, nu = self.NX, self.NU

        # Use MX (not SX) when building with external function
        X = ca.MX.sym("X", nx, N + 1)
        U = ca.MX.sym("U", nu, N)
        # P: x0(nx) + ref states (N+1)*nx
        P = ca.MX.sym("P", nx * (N + 2))

        x0 = P[:nx]
        obj = 0
        g = []

        # Initial constraint
        g.append(X[:, 0] - x0)

        Q = ca.DM(self._Q)
        R = ca.DM(self._R)

        hover_f = self._total_mass * self._gravity / 4.0
        u_ref = ca.DM([hover_f] * 4 + [0, 0])

        for k in range(N):
            x_ref_k = P[nx * (k + 1): nx * (k + 2)]

            # State cost -- position, velocity, omega, joints
            dx = X[:, k] - x_ref_k
            du = U[:, k] - u_ref

            # Attitude cost via quaternion error
            q = X[6:10, k]
            q_ref = x_ref_k[6:10]
            q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
            q_err_x = q_ref_inv[0]*q[1] + q_ref_inv[1]*q[0] + q_ref_inv[2]*q[3] - q_ref_inv[3]*q[2]
            q_err_y = q_ref_inv[0]*q[2] - q_ref_inv[1]*q[3] + q_ref_inv[2]*q[0] + q_ref_inv[3]*q[1]
            q_err_z = q_ref_inv[0]*q[3] + q_ref_inv[1]*q[2] - q_ref_inv[2]*q[1] + q_ref_inv[3]*q[0]
            att_cost = self._attitude_weight * (q_err_x**2 + q_err_y**2 + q_err_z**2)

            obj += dx.T @ Q @ dx + du.T @ R @ du + att_cost

            # Dynamics constraint using compiled external function
            x_next = self._f_step_ext(X[:, k], U[:, k])
            g.append(X[:, k + 1] - x_next)

        # Terminal cost
        x_ref_N = P[nx * (N + 1): nx * (N + 2)]
        dx_f = X[:, N] - x_ref_N
        q_f = X[6:10, N]
        qr_f = x_ref_N[6:10]
        qri_f = ca.vertcat(qr_f[0], -qr_f[1], -qr_f[2], -qr_f[3])
        qe_x = qri_f[0]*q_f[1] + qri_f[1]*q_f[0] + qri_f[2]*q_f[3] - qri_f[3]*q_f[2]
        qe_y = qri_f[0]*q_f[2] - qri_f[1]*q_f[3] + qri_f[2]*q_f[0] + qri_f[3]*q_f[1]
        qe_z = qri_f[0]*q_f[3] + qri_f[1]*q_f[2] - qri_f[2]*q_f[1] + qri_f[3]*q_f[0]
        obj += self._terminal_weight * (dx_f.T @ Q @ dx_f + self._attitude_weight * (qe_x**2 + qe_y**2 + qe_z**2))

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))
        lbg_list = [0.0] * (nx * (N + 1))
        ubg_list = [0.0] * (nx * (N + 1))

        nlp = {"f": obj, "x": opt_vars, "g": ca.vertcat(*g), "p": P}
        opts = {
            "ipopt.print_level": 0,
            "ipopt.max_iter": self._ipopt_max_iter,
            "ipopt.tol": self._ipopt_tol,
            "ipopt.warm_start_init_point": "yes",
            "ipopt.warm_start_bound_push": 1e-6,
            "ipopt.warm_start_mult_bound_push": 1e-6,
            "ipopt.warm_start_slack_bound_push": 1e-6,
            "print_time": 0,
            # hessian_approximation: "limited-memory" (L-BFGS) is ~1.6x faster
            # than "exact" for this NLP structure (3 IPOPT iterations typical).
            # Benchmark: exact=56ms/call vs limited-memory=34ms/call (warm-start).
            # Trajectory tracking quality is numerically equivalent at hover and
            # during smooth trajectories. Switch to "exact" if solver diverges on
            # aggressive manoeuvres.
            "ipopt.hessian_approximation": self._hessian_approximation,
        }
        self._solver = ca.nlpsol("nmpc", "ipopt", nlp, opts)

        n_vars = nx * (N + 1) + nu * N
        self._lbx = np.full(n_vars, -1e6)
        self._ubx = np.full(n_vars, 1e6)
        u_min = [0, 0, 0, 0, -5, -5]
        u_max = [12.3, 12.3, 12.3, 12.3, 5, 5]
        for k in range(N):
            idx = nx * (N + 1) + nu * k
            self._lbx[idx:idx + nu] = u_min
            self._ubx[idx:idx + nu] = u_max

        self._lbg = np.array(lbg_list, dtype=float)
        self._ubg = np.array(ubg_list, dtype=float)
        self._n_params = nx * (N + 2)

        # Pre-allocate reusable buffers to avoid per-call allocation
        self._p_buf = np.zeros(self._n_params)
        self._x0_buf = np.zeros(n_vars)
        # Scratch buffer for inlined _build_ref_state inside compute_control
        self._ref_row = np.zeros(nx)
        self._ref_row[6] = 1.0  # identity quaternion w=1 (default)

    def set_reference_trajectory(self, ref_func):
        self._ref_func = ref_func

    def compute_control(self, state_17: np.ndarray, reference: dict,
                        dt: float = 0.001) -> np.ndarray:
        N, nx, nu = self._N, self.NX, self.NU
        t_now = reference.get("_t", 0.0)
        dt_mpc = self._dt_mpc

        # Build parameter vector (pre-allocated buffer reuse)
        p = self._p_buf
        p[:nx] = state_17
        ref_func = self._ref_func
        if ref_func is not None:
            # Inline _build_ref_state to avoid repeated function-call overhead
            # across N+1 = 21 iterations.  The pre-allocated _ref_row scratch
            # buffer avoids temporary allocations inside the loop.
            ref_row = self._ref_row  # shape (nx,), pre-allocated
            for k in range(N + 1):
                ref = ref_func(t_now + k * dt_mpc)
                pos = ref["position"]
                vel = ref["velocity"]
                yaw = ref.get("yaw", 0.0)
                cy = np.cos(yaw * 0.5)
                sy = np.sin(yaw * 0.5)
                ref_row[0:3] = pos
                ref_row[3:6] = vel
                ref_row[6] = cy
                ref_row[7] = 0.0
                ref_row[8] = 0.0
                ref_row[9] = sy
                ref_row[10:13] = 0.0
                ref_row[13:15] = ref["joint_positions"]
                ref_row[15:17] = ref["joint_velocities"]
                p[nx * (k + 1): nx * (k + 2)] = ref_row
        else:
            ref_state = self._build_ref_state(reference)
            for k in range(N + 1):
                p[nx * (k + 1): nx * (k + 2)] = ref_state

        # Trajectory-shift warm-starting: shift previous full solution forward
        x0_vec = self._x0_buf
        if self._sol_prev is not None:
            sol_flat = np.asarray(self._sol_prev).ravel()
            x_end = nx * (N + 1)
            # Shift states: [1:N+1] → [0:N], repeat last
            x0_vec[:x_end - nx] = sol_flat[nx:x_end]
            x0_vec[x_end - nx:x_end] = sol_flat[x_end - nx:x_end]
            # Override initial state
            x0_vec[:nx] = state_17
            # Shift inputs: [1:N] → [0:N-1], repeat last
            u_start = x_end
            x0_vec[u_start:u_start + nu * (N - 1)] = sol_flat[u_start + nu:u_start + nu * N]
            x0_vec[u_start + nu * (N - 1):] = sol_flat[u_start + nu * (N - 1):u_start + nu * N]
        else:
            x0_vec[:nx * (N + 1)] = np.tile(state_17, N + 1)
            hf = self._total_mass * self._gravity / 4.0
            x0_vec[nx * (N + 1):] = np.tile([hf] * 4 + [0, 0], N)

        sol = self._solver(
            x0=x0_vec, p=p,
            lbx=self._lbx, ubx=self._ubx,
            lbg=self._lbg, ubg=self._ubg,
        )
        sol_x = sol["x"]
        self._sol_prev = sol_x
        # Use DM slice indexing to extract only the u block before converting
        # to numpy — avoids converting the full (477,) vector just to slice.
        _u_start = nx * (N + 1)
        u0 = np.asarray(sol_x[_u_start: _u_start + nu]).ravel()
        u1 = np.asarray(sol_x[_u_start + nu: _u_start + 2 * nu]).ravel()
        self._u_prev = u0
        self._u_next = u1
        return u0

    def _build_ref_state(self, ref: dict) -> np.ndarray:
        pos = np.asarray(ref.get("position", np.zeros(3)))
        vel = np.asarray(ref.get("velocity", np.zeros(3)))
        yaw = ref.get("yaw", 0.0)
        q_j = np.asarray(ref.get("joint_positions", np.zeros(2)))
        qd_j = np.asarray(ref.get("joint_velocities", np.zeros(2)))
        cy, sy = np.cos(yaw / 2), np.sin(yaw / 2)
        return np.concatenate([pos, vel, [cy, 0, 0, sy], np.zeros(3), q_j, qd_j])

    @property
    def dt_mpc(self):
        return self._dt_mpc

    @property
    def horizon(self):
        return self._N
