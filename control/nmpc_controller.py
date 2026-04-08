"""NMPC controller using CasADi symbolic dynamics with analytic Jacobians.

The dynamics are built symbolically (casadi_dynamics.py) to exactly replicate
the C++ engine's M(q), C(q,qdot), G(q), B(q). CasADi's automatic differentiation
provides exact analytic Jacobians for IPOPT — no finite differences needed.

State (17): identical to C++ engine [pos, vel, quat, omega, q_j, qd_j]
Input (6):  [f1, f2, f3, f4, tau_q1, tau_q2]
"""

import numpy as np
import casadi as ca
from control.casadi_dynamics import build_dynamics_from_config


class NMPCController:
    """Nonlinear MPC with analytic Jacobians via CasADi AD."""

    NX = 17
    NU = 6

    def __init__(self, quad_cfg, manip_cfg, env_cfg,
                 N: int = 20, dt_mpc: float = 0.02,
                 Q: np.ndarray = None, R: np.ndarray = None):
        self._N = N
        self._dt_mpc = dt_mpc
        self._total_mass = quad_cfg.mass + manip_cfg.link1.mass + manip_cfg.link2.mass
        self._gravity = env_cfg.gravity

        if Q is None:
            Q = self._default_Q()
        if R is None:
            R = self._default_R()
        self._Q = Q
        self._R = R

        self._ref_func = None
        self._u_prev = None

        # Build symbolic dynamics (exact match to C++ engine)
        f_cont = build_dynamics_from_config(quad_cfg, manip_cfg, env_cfg)

        # RK4 integrator
        x_sym = ca.SX.sym('x', self.NX)
        u_sym = ca.SX.sym('u', self.NU)
        k1 = f_cont(x_sym, u_sym)
        k2 = f_cont(x_sym + dt_mpc/2 * k1, u_sym)
        k3 = f_cont(x_sym + dt_mpc/2 * k2, u_sym)
        k4 = f_cont(x_sym + dt_mpc * k3, u_sym)
        x_next = x_sym + dt_mpc/6 * (k1 + 2*k2 + 2*k3 + k4)

        # Quaternion normalization after RK4 step
        quat_next = x_next[6:10]
        quat_norm = ca.norm_2(quat_next)
        x_next[6:10] = quat_next / quat_norm

        self._f_step = ca.Function('F', [x_sym, u_sym], [x_next])

        self._build_nlp()

    def _default_Q(self):
        return np.diag([
            2000, 2000, 3000,  # position — high for precise tracking
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

        X = ca.SX.sym('X', nx, N + 1)
        U = ca.SX.sym('U', nu, N)
        # P: x0(nx) + ref states (N+1)*nx
        P = ca.SX.sym('P', nx * (N + 2))

        x0 = P[:nx]
        obj = 0
        g = []
        lbg, ubg = [], []

        # Initial constraint
        g.append(X[:, 0] - x0)
        lbg += [0] * nx
        ubg += [0] * nx

        Q = ca.DM(self._Q)
        R = ca.DM(self._R)

        hover_f = self._total_mass * self._gravity / 4.0
        u_ref = ca.DM([hover_f] * 4 + [0, 0])

        for k in range(N):
            x_ref_k = P[nx*(k+1): nx*(k+2)]

            # State cost — position, velocity, omega, joints
            dx = X[:, k] - x_ref_k
            du = U[:, k] - u_ref

            # Attitude cost via quaternion error (more robust than direct diff)
            # q_err = q_ref^{-1} ⊗ q — for small errors, imaginary part ≈ 0.5*angle_error
            q = X[6:10, k]
            q_ref = x_ref_k[6:10]
            # Quaternion conjugate of ref: [w, -x, -y, -z]
            q_ref_inv = ca.vertcat(q_ref[0], -q_ref[1], -q_ref[2], -q_ref[3])
            # Quaternion multiplication: q_err = q_ref_inv ⊗ q
            q_err_w = q_ref_inv[0]*q[0] - q_ref_inv[1]*q[1] - q_ref_inv[2]*q[2] - q_ref_inv[3]*q[3]
            q_err_x = q_ref_inv[0]*q[1] + q_ref_inv[1]*q[0] + q_ref_inv[2]*q[3] - q_ref_inv[3]*q[2]
            q_err_y = q_ref_inv[0]*q[2] - q_ref_inv[1]*q[3] + q_ref_inv[2]*q[0] + q_ref_inv[3]*q[1]
            q_err_z = q_ref_inv[0]*q[3] + q_ref_inv[1]*q[2] - q_ref_inv[2]*q[1] + q_ref_inv[3]*q[0]
            # Attitude error cost: penalize imaginary parts (= sin(angle/2) * axis)
            att_cost = 1000 * (q_err_x**2 + q_err_y**2 + q_err_z**2)

            obj += dx.T @ Q @ dx + du.T @ R @ du + att_cost

            # Dynamics constraint
            x_next = self._f_step(X[:, k], U[:, k])
            g.append(X[:, k+1] - x_next)
            lbg += [0] * nx
            ubg += [0] * nx

        # Terminal cost
        x_ref_N = P[nx*(N+1): nx*(N+2)]
        dx_f = X[:, N] - x_ref_N
        q_f = X[6:10, N]; qr_f = x_ref_N[6:10]
        qri_f = ca.vertcat(qr_f[0], -qr_f[1], -qr_f[2], -qr_f[3])
        qe_x = qri_f[0]*q_f[1]+qri_f[1]*q_f[0]+qri_f[2]*q_f[3]-qri_f[3]*q_f[2]
        qe_y = qri_f[0]*q_f[2]-qri_f[1]*q_f[3]+qri_f[2]*q_f[0]+qri_f[3]*q_f[1]
        qe_z = qri_f[0]*q_f[3]+qri_f[1]*q_f[2]-qri_f[2]*q_f[1]+qri_f[3]*q_f[0]
        obj += 5 * (dx_f.T @ Q @ dx_f + 1000*(qe_x**2+qe_y**2+qe_z**2))

        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        nlp = {'f': obj, 'x': opt_vars, 'g': ca.vertcat(*g), 'p': P}
        opts = {
            'ipopt.print_level': 0, 'ipopt.max_iter': 50,
            'ipopt.tol': 1e-4, 'ipopt.warm_start_init_point': 'yes',
            'print_time': 0,
        }
        self._solver = ca.nlpsol('nmpc', 'ipopt', nlp, opts)

        n_vars = nx*(N+1) + nu*N
        self._lbx = np.full(n_vars, -1e6)
        self._ubx = np.full(n_vars, 1e6)
        u_min = [0, 0, 0, 0, -5, -5]
        u_max = [12.3, 12.3, 12.3, 12.3, 5, 5]
        for k in range(N):
            idx = nx*(N+1) + nu*k
            self._lbx[idx:idx+nu] = u_min
            self._ubx[idx:idx+nu] = u_max

        self._lbg = np.array(lbg, float)
        self._ubg = np.array(ubg, float)
        self._n_params = nx * (N + 2)

    def set_reference_trajectory(self, ref_func):
        self._ref_func = ref_func

    def compute_control(self, state_17: np.ndarray, reference: dict,
                        dt: float = 0.001) -> np.ndarray:
        N, nx, nu = self._N, self.NX, self.NU
        t_now = reference.get("_t", 0.0)

        p = np.zeros(self._n_params)
        p[:nx] = state_17
        for k in range(N + 1):
            t_k = t_now + k * self._dt_mpc
            ref_k = self._ref_func(t_k) if self._ref_func else reference
            p[nx*(k+1): nx*(k+2)] = self._build_ref_state(ref_k)

        if self._u_prev is not None:
            x0g = np.tile(state_17, N+1)
            u0g = np.tile(self._u_prev, N)
        else:
            x0g = np.tile(state_17, N+1)
            hf = self._total_mass * self._gravity / 4.0
            u0g = np.tile([hf]*4 + [0,0], N)

        sol = self._solver(x0=np.concatenate([x0g, u0g]), p=p,
                           lbx=self._lbx, ubx=self._ubx,
                           lbg=self._lbg, ubg=self._ubg)
        u_opt = np.array(sol['x'][nx*(N+1): nx*(N+1)+nu]).flatten()
        self._u_prev = u_opt
        return u_opt

    def _build_ref_state(self, ref: dict) -> np.ndarray:
        pos = np.asarray(ref.get("position", np.zeros(3)))
        vel = np.asarray(ref.get("velocity", np.zeros(3)))
        yaw = ref.get("yaw", 0.0)
        q_j = np.asarray(ref.get("joint_positions", np.zeros(2)))
        qd_j = np.asarray(ref.get("joint_velocities", np.zeros(2)))
        cy, sy = np.cos(yaw/2), np.sin(yaw/2)
        return np.concatenate([pos, vel, [cy, 0, 0, sy], np.zeros(3), q_j, qd_j])

    @property
    def dt_mpc(self): return self._dt_mpc
    @property
    def horizon(self): return self._N
