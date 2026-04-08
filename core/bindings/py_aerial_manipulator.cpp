#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include "aerial_manipulator/types.hpp"
#include "aerial_manipulator/rigid_body.hpp"
#include "aerial_manipulator/quadrotor.hpp"
#include "aerial_manipulator/manipulator.hpp"
#include "aerial_manipulator/aerial_manipulator_system.hpp"
#include "aerial_manipulator/integrator.hpp"
#include "aerial_manipulator/rk4_integrator.hpp"
#include "aerial_manipulator/rkf45_integrator.hpp"

namespace py = pybind11;
using namespace aerial_manipulator;

PYBIND11_MODULE(_core, m) {
    m.doc() = "Aerial Manipulator C++ dynamics engine";

    // ── Parameter structs ──
    py::class_<QuadrotorParams>(m, "QuadrotorParams",
        "Physical parameters for the quadrotor platform.\n\n"
        "All values in SI units. See types.hpp for detailed documentation.")
        .def(py::init<>())
        .def_readwrite("mass", &QuadrotorParams::mass,
            "[kg] Quadrotor body mass (excluding manipulator)")
        .def_readwrite("inertia", &QuadrotorParams::inertia,
            "[kg*m^2] 3x3 body-frame inertia tensor (diagonal)")
        .def_readwrite("arm_length", &QuadrotorParams::arm_length,
            "[m] Motor-to-center distance")
        .def_readwrite("thrust_coeff", &QuadrotorParams::thrust_coeff,
            "[N/(rad/s)^2] Rotor thrust coefficient k_f: f_i = k_f * omega_i^2")
        .def_readwrite("torque_coeff", &QuadrotorParams::torque_coeff,
            "[N*m/(rad/s)^2] Rotor torque coefficient k_tau: tau_i = k_tau * omega_i^2")
        .def_readwrite("drag_coeff", &QuadrotorParams::drag_coeff,
            "[N*s/m] Isotropic translational drag coefficient")
        .def_readwrite("motor_time_constant", &QuadrotorParams::motor_time_constant,
            "[s] First-order motor response time constant tau_m")
        .def_readwrite("max_motor_speed", &QuadrotorParams::max_motor_speed,
            "[rad/s] Motor angular speed saturation omega_max");

    py::class_<LinkParams>(m, "LinkParams",
        "Physical parameters for a single manipulator link.\n\n"
        "All values in SI units.")
        .def(py::init<>())
        .def_readwrite("mass", &LinkParams::mass,
            "[kg] Link mass")
        .def_readwrite("length", &LinkParams::length,
            "[m] Total link length")
        .def_readwrite("com_distance", &LinkParams::com_distance,
            "[m] Distance from proximal joint to center of mass")
        .def_readwrite("inertia", &LinkParams::inertia,
            "[kg*m^2] Link-frame inertia tensor (3x3 diagonal, principal axes)");

    py::class_<ManipulatorParams>(m, "ManipulatorParams",
        "Physical parameters for the 2-DOF 3D manipulator.\n\n"
        "Joint 1 (azimuth) rotates about the body z-axis.\n"
        "Joint 2 (elevation) rotates about the rotated y-axis.")
        .def(py::init<>())
        .def_readwrite("attachment_offset", &ManipulatorParams::attachment_offset,
            "[m] Joint-1 position in body frame relative to quadrotor COM")
        .def_readwrite("link1", &ManipulatorParams::link1,
            "First link parameters (azimuth-elevation gimbal)")
        .def_readwrite("link2", &ManipulatorParams::link2,
            "Second link parameters (extends along link-1 direction)")
        .def_readwrite("joint_lower_limit", &ManipulatorParams::joint_lower_limit,
            "[rad] Lower joint limits [q1_min, q2_min]")
        .def_readwrite("joint_upper_limit", &ManipulatorParams::joint_upper_limit,
            "[rad] Upper joint limits [q1_max, q2_max]")
        .def_readwrite("max_joint_torque", &ManipulatorParams::max_joint_torque,
            "[N*m] Maximum torque per joint actuator");

    py::class_<EnvironmentParams>(m, "EnvironmentParams",
        "Environmental constants for the simulation.\n\n"
        "Defaults: gravity=9.81 m/s^2, air_density=1.225 kg/m^3, wind=zero.")
        .def(py::init<>())
        .def_readwrite("gravity", &EnvironmentParams::gravity,
            "[m/s^2] Gravitational acceleration magnitude")
        .def_readwrite("air_density", &EnvironmentParams::air_density,
            "[kg/m^3] Air density at operating altitude")
        .def_readwrite("wind_velocity", &EnvironmentParams::wind_velocity,
            "[m/s] Constant wind velocity in world frame (ENU)");

    // ── RigidBody ──
    py::class_<RigidBody>(m, "RigidBody",
        "Rigid body with mass, inertia, and optional geometric properties.")
        .def(py::init<>())
        .def(py::init<double, const Mat3&, double, double>(),
             py::arg("mass"), py::arg("inertia"),
             py::arg("length") = 0.0, py::arg("com_distance") = 0.0,
             "Construct with mass [kg], inertia [kg*m^2], optional length [m] and com_distance [m].")
        .def("mass", &RigidBody::mass, "[kg] Body mass")
        .def("inertia", &RigidBody::inertia, "[kg*m^2] 3x3 inertia tensor")
        .def("length", &RigidBody::length, "[m] Body length")
        .def("com_distance", &RigidBody::com_distance,
             "[m] Distance from reference point to center of mass");

    // ── Quadrotor ──
    py::class_<Quadrotor>(m, "Quadrotor",
        "Quadrotor aerodynamics and motor mixing model.\n\n"
        "Computes rotor forces, body torques, and translational drag "
        "from individual rotor thrusts (4-vector).")
        .def(py::init<const QuadrotorParams&>(), py::arg("params"),
             "Construct from QuadrotorParams.")
        .def("compute_forces", &Quadrotor::compute_forces,
             "Compute net thrust vector [N] in body frame from rotor thrust inputs.")
        .def("compute_torques", &Quadrotor::compute_torques,
             "Compute body torques [N*m] from rotor thrust inputs via mixing matrix.")
        .def("compute_drag", &Quadrotor::compute_drag,
             "Compute translational drag force [N] in body frame given body-frame velocity [m/s].")
        .def("mixing_matrix", &Quadrotor::mixing_matrix,
             "4x4 mixing matrix mapping rotor thrusts to [F_total, tau_roll, tau_pitch, tau_yaw].")
        .def("mixing_matrix_inv", &Quadrotor::mixing_matrix_inv,
             "Pseudo-inverse of mixing matrix for control allocation.")
        .def("hover_thrust", &Quadrotor::hover_thrust,
             "[N] Thrust per rotor required to hover (total_mass * g / 4).")
        .def("body", &Quadrotor::body, py::return_value_policy::reference_internal,
             "RigidBody representing the quadrotor frame.")
        .def("params", &Quadrotor::params, py::return_value_policy::reference_internal,
             "QuadrotorParams used to construct this object.");

    // ── Manipulator ──
    py::class_<Manipulator>(m, "Manipulator",
        "2-DOF 3D manipulator kinematics and dynamics.\n\n"
        "Joint 1 (azimuth, q1): rotation about body z-axis.\n"
        "Joint 2 (elevation, q2): rotation about rotated y-axis.\n"
        "All positions expressed in the quadrotor body frame.")
        .def(py::init<const ManipulatorParams&>(), py::arg("params"),
             "Construct from ManipulatorParams.")
        .def("forward_kinematics", &Manipulator::forward_kinematics,
             "End-effector position [m] in body frame for joint angles q (2-vector) [rad].")
        .def("link1_com_position", &Manipulator::link1_com_position,
             "Link-1 center-of-mass position [m] in body frame for joint angles q.")
        .def("link2_com_position", &Manipulator::link2_com_position,
             "Link-2 center-of-mass position [m] in body frame for joint angles q.")
        .def("jacobian", &Manipulator::jacobian,
             "3x2 geometric Jacobian of end-effector w.r.t. joint angles q.")
        .def("link1_com_jacobian", &Manipulator::link1_com_jacobian,
             "3x2 Jacobian of link-1 COM position w.r.t. joint angles q.")
        .def("link2_com_jacobian", &Manipulator::link2_com_jacobian,
             "3x2 Jacobian of link-2 COM position w.r.t. joint angles q.")
        .def("mass_matrix", &Manipulator::mass_matrix,
             "2x2 joint-space mass matrix M(q) [kg*m^2].")
        .def("coriolis_matrix", &Manipulator::coriolis_matrix,
             "2x2 Coriolis/centrifugal matrix C(q, qdot) [kg*m^2/s].")
        .def("gravity_vector", &Manipulator::gravity_vector,
             "2-vector joint gravity torques G(q) [N*m], given q, body-to-world rotation R, and g.")
        .def("joint_torque_to_body_wrench", &Manipulator::joint_torque_to_body_wrench,
             "Convert joint torques [N*m] to equivalent body wrench (force+torque) [N, N*m].")
        .def("params", &Manipulator::params, py::return_value_policy::reference_internal,
             "ManipulatorParams used to construct this object.")
        .def("attachment_offset", &Manipulator::attachment_offset,
             "[m] Joint-1 attachment point in body frame.");

    // ── Integrators ──
    py::class_<IntegratorBase, std::shared_ptr<IntegratorBase>>(m, "IntegratorBase",
        "Abstract base class for numerical integrators (Strategy pattern).")
        .def("name", &IntegratorBase::name,
             "Human-readable name identifying the integration method.");

    py::class_<RK4Integrator, IntegratorBase, std::shared_ptr<RK4Integrator>>(m, "RK4Integrator",
        "Fixed-step 4th-order Runge-Kutta integrator.\n\n"
        "Suitable for real-time simulation with a known, constant time step.\n"
        "Use RKF45Integrator for automatic step-size control.")
        .def(py::init<>(), "Construct an RK4 integrator.");

    py::class_<RKF45Integrator::Config>(m, "RKF45Config",
        "Configuration for the adaptive Runge-Kutta-Fehlberg 4(5) integrator.")
        .def(py::init<>())
        .def_readwrite("atol", &RKF45Integrator::Config::atol,
            "Absolute error tolerance (default 1e-6).")
        .def_readwrite("rtol", &RKF45Integrator::Config::rtol,
            "Relative error tolerance (default 1e-6).")
        .def_readwrite("dt_min", &RKF45Integrator::Config::dt_min,
            "[s] Minimum allowed step size (default 1e-8).")
        .def_readwrite("dt_max", &RKF45Integrator::Config::dt_max,
            "[s] Maximum allowed step size (default 0.1).")
        .def_readwrite("safety", &RKF45Integrator::Config::safety,
            "Safety factor for step-size control (default 0.9).");

    py::class_<RKF45Integrator, IntegratorBase, std::shared_ptr<RKF45Integrator>>(m, "RKF45Integrator",
        "Adaptive-step Runge-Kutta-Fehlberg 4(5) integrator.\n\n"
        "Automatically adjusts step size to meet atol/rtol error targets.\n"
        "More accurate than RK4 for stiff or fast-changing dynamics.")
        .def(py::init<>(), "Construct with default RKF45Config.")
        .def(py::init<const RKF45Integrator::Config&>(), py::arg("config"),
             "Construct with custom RKF45Config.")
        .def("last_step_size", &RKF45Integrator::last_step_size,
             "[s] Actual step size used in the most recent integration step.");

    // ── AerialManipulatorSystem ──
    py::class_<AerialManipulatorSystem>(m, "AerialManipulatorSystem",
        "Coupled multibody dynamics: quadrotor + 2-DOF 3D manipulator.\n\n"
        "Implements the full coupled equation of motion:\n"
        "  M(q) * q_ddot + C(q, q_dot) * q_dot + G(q) = B(q) * u\n\n"
        "State vector x (17-D):\n"
        "  [pos(3), vel(3), quat(4), ang_vel(3), joint_pos(2), joint_vel(2)]\n\n"
        "Input vector u (6-D):\n"
        "  [motor_thrusts(4) [N], joint_torques(2) [N*m]]\n\n"
        "Quaternion is automatically normalised at each derivative evaluation.\n"
        "Call set_integrator() before step().")
        .def(py::init<const QuadrotorParams&, const ManipulatorParams&, const EnvironmentParams&>(),
             py::arg("quad_params"), py::arg("manip_params"), py::arg("env_params"),
             "Construct the coupled system from component parameter structs.")
        .def("compute_state_derivative", &AerialManipulatorSystem::compute_state_derivative,
             py::arg("state"), py::arg("input"),
             "Compute dx/dt = f(x, u). Returns 17-D state derivative vector.\n"
             "Quaternion in state is normalised internally before use.")
        .def("compute_mass_matrix", &AerialManipulatorSystem::compute_mass_matrix,
             py::arg("state"),
             "Compute coupled 8x8 mass matrix M(q) in generalised coordinates.")
        .def("compute_coriolis_vector", &AerialManipulatorSystem::compute_coriolis_vector,
             py::arg("state"),
             "Compute Coriolis/centrifugal term C(q, q_dot)*q_dot (8-D vector).\n"
             "Uses hybrid numerical+analytical Christoffel symbol computation.")
        .def("compute_gravity_vector", &AerialManipulatorSystem::compute_gravity_vector,
             py::arg("state"),
             "Compute generalised gravity vector G(q) (8-D) [N or N*m].")
        .def("compute_input_matrix", &AerialManipulatorSystem::compute_input_matrix,
             py::arg("state"),
             "Compute configuration-dependent input matrix B(q) (8x6).")
        .def("step", &AerialManipulatorSystem::step,
             py::arg("t"), py::arg("state"), py::arg("input"), py::arg("dt"),
             "Advance state by dt [s] using the current integrator.\n"
             "Returns new 17-D state with quaternion normalised.\n"
             "Raises RuntimeError if no integrator has been set.")
        .def("set_integrator", &AerialManipulatorSystem::set_integrator,
             py::arg("integrator"),
             "Set the numerical integrator (RK4Integrator or RKF45Integrator).")
        .def("total_mass", &AerialManipulatorSystem::total_mass,
             "[kg] Total system mass: quadrotor body + link1 + link2.")
        .def("quadrotor", &AerialManipulatorSystem::quadrotor,
             py::return_value_policy::reference_internal,
             "Read-only reference to the internal Quadrotor object.")
        .def("manipulator", &AerialManipulatorSystem::manipulator,
             py::return_value_policy::reference_internal,
             "Read-only reference to the internal Manipulator object.");

    // ── Constants ──
    m.attr("STATE_DIM") = STATE_DIM;
    m.attr("INPUT_DIM") = INPUT_DIM;
    m.attr("NUM_ROTORS") = NUM_ROTORS;
    m.attr("NUM_JOINTS") = NUM_JOINTS;
    m.attr("DOF_TOTAL") = DOF_TOTAL;

    // State index namespace
    auto idx_mod = m.def_submodule("idx", "State vector index constants");
    idx_mod.attr("POS") = idx::POS;
    idx_mod.attr("VEL") = idx::VEL;
    idx_mod.attr("QUAT") = idx::QUAT;
    idx_mod.attr("ANG_VEL") = idx::ANG_VEL;
    idx_mod.attr("JOINT_POS") = idx::JOINT_POS;
    idx_mod.attr("JOINT_VEL") = idx::JOINT_VEL;
}
