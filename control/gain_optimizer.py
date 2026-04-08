"""Automatic PID gain optimization for the aerial manipulator.

Two approaches:
1. LQR-based: Linearize at operating point → CARE → optimal gains
2. Simulation-based: Run short sim → minimize RMSE via Nelder-Mead

The simulation-based approach uses the actual nonlinear C++ dynamics
and is more reliable for the coupled system. It starts from LQR gains
as initial guess and refines via optimization.
"""

import numpy as np
from scipy.linalg import solve_continuous_are
from scipy.optimize import minimize
from dataclasses import dataclass


@dataclass
class OptimizedGains:
    """Container for optimized controller gains."""
    # Position controller
    pos_Kp: np.ndarray  # (3,)
    pos_Kd: np.ndarray  # (3,)
    pos_Ki: np.ndarray  # (3,)

    # Attitude controller
    att_Kr: np.ndarray   # (3,)
    att_Kw: np.ndarray   # (3,)

    # Manipulator controller
    joint_Kp: np.ndarray  # (2,)
    joint_Kd: np.ndarray  # (2,)


class GainOptimizer:
    """Computes optimal PID gains via LQR for the aerial manipulator system.

    The optimizer linearizes each control subsystem independently:
    1. Position loop: augmented double integrator (3-axis)
    2. Attitude loop: rotational dynamics (3-axis)
    3. Joint loop: augmented joint dynamics (2-axis)

    Usage:
        optimizer = GainOptimizer(total_mass, inertia, joint_inertia, gravity)
        gains = optimizer.optimize()
        # or with custom weights:
        gains = optimizer.optimize(pos_bandwidth=2.0, att_bandwidth=10.0)
    """

    def __init__(self, total_mass: float, inertia: np.ndarray,
                 joint_inertia: np.ndarray, gravity: float = 9.81,
                 max_thrust: float = 50.0, max_torque: float = 5.0,
                 max_angle: float = 0.3,
                 effective_rot_inertia: np.ndarray = None):
        """
        Args:
            total_mass: Total system mass [kg]
            inertia: 3x3 body inertia tensor [kg·m²]
            joint_inertia: (2,) diagonal of joint inertia [kg·m²]
            gravity: Gravitational acceleration [m/s²]
            max_thrust: Maximum total thrust deviation from hover [N]
            max_torque: Maximum joint torque [N·m]
            max_angle: Maximum acceptable attitude deviation [rad]
            effective_rot_inertia: (3,) effective rotational inertia from M_rr
                                  (includes manipulator coupling). If None, uses
                                  quadrotor inertia only.
        """
        self._m = total_mass
        self._J = np.diag(inertia) if inertia.ndim == 2 else inertia
        self._J_joint = np.asarray(joint_inertia)
        self._g = gravity
        self._max_thrust = max_thrust
        self._max_torque = max_torque
        self._max_angle = max_angle
        # Use effective (coupled) inertia if provided, else bare quadrotor
        self._J_eff = np.asarray(effective_rot_inertia) if effective_rot_inertia is not None else self._J

    def optimize(self, pos_bandwidth: float = 1.5,
                 att_bandwidth: float = 8.0,
                 joint_bandwidth: float = 10.0) -> OptimizedGains:
        """Compute optimal gains for all control loops.

        Args:
            pos_bandwidth: Desired position loop bandwidth [rad/s]
            att_bandwidth: Desired attitude loop bandwidth [rad/s]
            joint_bandwidth: Desired joint loop bandwidth [rad/s]

        Returns:
            OptimizedGains with Kp, Kd, Ki for each loop.
        """
        pos_Kp, pos_Kd, pos_Ki = self._optimize_position(pos_bandwidth)
        att_Kr, att_Kw = self._optimize_attitude(att_bandwidth)
        joint_Kp, joint_Kd = self._optimize_joint(joint_bandwidth)

        return OptimizedGains(
            pos_Kp=pos_Kp, pos_Kd=pos_Kd, pos_Ki=pos_Ki,
            att_Kr=att_Kr, att_Kw=att_Kw,
            joint_Kp=joint_Kp, joint_Kd=joint_Kd,
        )

    def _optimize_position(self, bandwidth: float):
        """LQR for position: augmented double integrator with integral action.

        Augmented state per axis: xi = [integral(e), e, e_dot]
        Dynamics: xi_dot = [[0,1,0],[0,0,1],[0,0,0]] xi + [[0],[0],[1/m]] u

        Bryson's rule for Q, R:
            Q_ii = 1 / (max_acceptable_deviation_i)^2
            R = 1 / (max_control_effort)^2
        """
        Kp = np.zeros(3)
        Kd = np.zeros(3)
        Ki = np.zeros(3)

        for axis in range(3):
            m_eff = self._m

            # Augmented state-space: [integral(e), e, e_dot]
            # Note: u here is acceleration (a = F/m), so B = [0,0,1]^T
            # This way K directly gives gains for the PID law:
            #   a_des = Ki*integral(e) + Kp*e + Kd*e_dot
            #   F_des = m * a_des (applied in position controller)
            A = np.array([
                [0, 1, 0],
                [0, 0, 1],
                [0, 0, 0],
            ])
            B = np.array([[0], [0], [1.0]])  # acceleration input

            # Frequency-domain approach: desired closed-loop bandwidth → gains
            # For augmented double integrator, target ω_n and ζ directly
            wn = bandwidth  # natural frequency [rad/s]
            zeta = 0.75     # damping ratio (slightly underdamped for fast response)

            # Q, R chosen to achieve target wn and zeta via LQR
            # Empirical scaling: Q penalizes state deviation, R penalizes effort
            Q = np.diag([
                0.1 * wn**2,   # integral state: mild penalty
                wn**4,          # position: primary objective
                wn**2,          # velocity: moderate penalty
            ])

            # R: higher = less aggressive (smaller gains)
            R = np.array([[1.0]])

            # Solve CARE
            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P  # K = [Ki, Kp, Kd] in accel units

            # Gains map to: F = m * (Kp*e + Kd*e_dot + Ki*int(e))
            # PositionController internally multiplies by mass, so store as-is
            Ki[axis] = K[0, 0]
            Kp[axis] = K[0, 1]
            Kd[axis] = K[0, 2]

        return Kp, Kd, Ki

    def _optimize_attitude(self, bandwidth: float):
        """LQR for attitude: rotational dynamics (no integral action).

        Per axis: e_ddot = (1/J) * tau
        State: [e_angle, e_omega]
        """
        Kr = np.zeros(3)
        Kw = np.zeros(3)

        for axis in range(3):
            J_ax = self._J_eff[axis]  # use effective (coupled) inertia

            A = np.array([[0, 1], [0, 0]])
            B = np.array([[0], [1.0 / J_ax]])

            # Attitude: tighter error requirements
            max_angle_err = self._max_angle  # ~17°
            max_rate_err = 1.0  # [rad/s]

            scale = (bandwidth / 8.0) ** 2
            Q = np.diag([
                scale / max_angle_err**2,
                1.0 / max_rate_err**2,
            ])

            # Torque budget
            max_torque_ax = 2.0  # [N·m] per axis
            R = np.array([[1.0 / max_torque_ax**2]])

            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P

            Kr[axis] = K[0, 0] * J_ax
            Kw[axis] = K[0, 1] * J_ax

        return Kr, Kw

    def _optimize_joint(self, bandwidth: float):
        """LQR for joint control: second-order dynamics with inertia.

        Per joint: q_ddot = (1/J_joint) * tau
        State: [e_q, e_q_dot]
        """
        Kp = np.zeros(2)
        Kd = np.zeros(2)

        for j in range(2):
            J_j = self._J_joint[j]

            A = np.array([[0, 1], [0, 0]])
            B = np.array([[0], [1.0 / J_j]])

            max_angle_err = 0.02  # [rad] ~1°
            max_rate_err = 0.5    # [rad/s]

            scale = (bandwidth / 10.0) ** 2
            Q = np.diag([
                scale / max_angle_err**2,
                1.0 / max_rate_err**2,
            ])

            R = np.array([[1.0 / self._max_torque**2]])

            P = solve_continuous_are(A, B, Q, R)
            K = np.linalg.inv(R) @ B.T @ P

            Kp[j] = K[0, 0] * J_j
            Kd[j] = K[0, 1] * J_j

        return Kp, Kd

    def optimize_for_trajectory(self, config, reference_func,
                                duration: float = 3.0,
                                initial_gains: OptimizedGains = None,
                                verbose: bool = True) -> OptimizedGains:
        """Simulation-based gain optimization using Nelder-Mead on the actual nonlinear system.

        Runs short simulations with different gain sets and minimizes the
        position + attitude RMSE. Uses actual C++ dynamics engine.

        Args:
            config: SimulationConfig
            reference_func: t → reference dict
            duration: Optimization sim duration [s] (shorter = faster)
            initial_gains: Starting gains (default: from manual tuning)
            verbose: Print optimization progress

        Returns:
            OptimizedGains minimizing tracking RMSE.
        """
        from simulation.simulation_runner import SimulationRunner
        from control.position_controller import PositionController
        from control.attitude_controller import AttitudeController
        from control.manipulator_controller import ManipulatorController
        from analysis.result_analyzer import ResultAnalyzer

        config.duration = duration

        if initial_gains is None:
            # Start from known-good manual gains
            initial_gains = OptimizedGains(
                pos_Kp=np.array([6.0, 6.0, 8.0]),
                pos_Kd=np.array([4.5, 4.5, 5.6]),
                pos_Ki=np.array([0.1, 0.1, 0.2]),
                att_Kr=np.array([8.0, 8.0, 4.0]),
                att_Kw=np.array([0.8, 0.8, 0.5]),
                joint_Kp=np.array([20.0, 20.0]),
                joint_Kd=np.array([1.0, 1.0]),
            )

        def _gains_to_vec(g: OptimizedGains) -> np.ndarray:
            """Pack gains into optimization vector (log-scale for positivity)."""
            return np.log(np.concatenate([
                g.pos_Kp, g.pos_Kd, g.pos_Ki,
                g.att_Kr, g.att_Kw,
                g.joint_Kp, g.joint_Kd,
            ]))

        def _vec_to_gains(v: np.ndarray) -> OptimizedGains:
            """Unpack optimization vector to gains."""
            g = np.exp(v)
            return OptimizedGains(
                pos_Kp=g[0:3], pos_Kd=g[3:6], pos_Ki=g[6:9],
                att_Kr=g[9:12], att_Kw=g[12:15],
                joint_Kp=g[15:17], joint_Kd=g[17:19],
            )

        best_cost = [np.inf]
        eval_count = [0]

        def _cost(v):
            eval_count[0] += 1
            g = _vec_to_gains(v)

            try:
                runner = SimulationRunner(config)
                runner.position_ctrl = PositionController(
                    Kp=g.pos_Kp, Kd=g.pos_Kd, Ki=g.pos_Ki,
                    total_mass=self._m, gravity=self._g, integrator_limit=2.0,
                )
                runner.attitude_ctrl = AttitudeController(
                    Kr=g.att_Kr, Kw=g.att_Kw,
                    inertia=np.diag(self._J) if self._J.ndim == 1 else self._J,
                )
                runner.manip_ctrl = ManipulatorController(
                    Kp=g.joint_Kp, Kd=g.joint_Kd,
                    gravity_compensation=True, max_torque=self._max_torque,
                )

                logger = runner.run(reference_func)
                states = logger.get_states()

                if np.any(np.isnan(states)) or np.any(np.isinf(states)):
                    return 1e6  # diverged

                times = logger.get_time()
                ref_pos = np.array([reference_func(t)["position"] for t in times])
                ref_joints = np.array([reference_func(t)["joint_positions"] for t in times])

                analyzer = ResultAnalyzer(logger)
                pos_rmse = analyzer.position_rmse(ref_pos)
                joint_rmse = analyzer.joint_tracking_rmse(ref_joints)

                # Weighted cost: position + joints + control effort penalty
                effort = analyzer.control_effort()
                motor_effort = np.sum(effort["motor_integral"])

                cost = (np.sum(pos_rmse**2) * 100   # position tracking
                        + np.sum(joint_rmse**2) * 10  # joint tracking
                        + motor_effort * 1e-6)         # mild effort penalty

                if cost < best_cost[0] and verbose:
                    best_cost[0] = cost
                    print(f"  [{eval_count[0]:3d}] cost={cost:.6f} pos_rmse={np.sqrt(np.sum(pos_rmse**2))*100:.2f}cm")

                return cost

            except Exception:
                return 1e6

        x0 = _gains_to_vec(initial_gains)

        if verbose:
            print(f"Optimizing gains over {duration}s trajectory ({len(x0)} parameters)...")

        result = minimize(_cost, x0, method="Nelder-Mead",
                          options={"maxiter": 200, "xatol": 0.01, "fatol": 1e-6,
                                   "adaptive": True})

        optimal = _vec_to_gains(result.x)
        if verbose:
            print(f"Optimization complete: {result.nit} iterations, {eval_count[0]} evals")
            self.print_gains(optimal)

        return optimal

    @staticmethod
    def print_gains(gains: OptimizedGains):
        """Pretty-print optimized gains."""
        print("=== Optimized Controller Gains ===")
        print(f"Position Kp: [{gains.pos_Kp[0]:.3f}, {gains.pos_Kp[1]:.3f}, {gains.pos_Kp[2]:.3f}]")
        print(f"Position Kd: [{gains.pos_Kd[0]:.3f}, {gains.pos_Kd[1]:.3f}, {gains.pos_Kd[2]:.3f}]")
        print(f"Position Ki: [{gains.pos_Ki[0]:.3f}, {gains.pos_Ki[1]:.3f}, {gains.pos_Ki[2]:.3f}]")
        print(f"Attitude Kr: [{gains.att_Kr[0]:.3f}, {gains.att_Kr[1]:.3f}, {gains.att_Kr[2]:.3f}]")
        print(f"Attitude Kw: [{gains.att_Kw[0]:.3f}, {gains.att_Kw[1]:.3f}, {gains.att_Kw[2]:.3f}]")
        print(f"Joint   Kp: [{gains.joint_Kp[0]:.3f}, {gains.joint_Kp[1]:.3f}]")
        print(f"Joint   Kd: [{gains.joint_Kd[0]:.3f}, {gains.joint_Kd[1]:.3f}]")
