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
    py::class_<QuadrotorParams>(m, "QuadrotorParams")
        .def(py::init<>())
        .def_readwrite("mass", &QuadrotorParams::mass)
        .def_readwrite("inertia", &QuadrotorParams::inertia)
        .def_readwrite("arm_length", &QuadrotorParams::arm_length)
        .def_readwrite("thrust_coeff", &QuadrotorParams::thrust_coeff)
        .def_readwrite("torque_coeff", &QuadrotorParams::torque_coeff)
        .def_readwrite("drag_coeff", &QuadrotorParams::drag_coeff)
        .def_readwrite("motor_time_constant", &QuadrotorParams::motor_time_constant)
        .def_readwrite("max_motor_speed", &QuadrotorParams::max_motor_speed);

    py::class_<LinkParams>(m, "LinkParams")
        .def(py::init<>())
        .def_readwrite("mass", &LinkParams::mass)
        .def_readwrite("length", &LinkParams::length)
        .def_readwrite("com_distance", &LinkParams::com_distance)
        .def_readwrite("inertia", &LinkParams::inertia);

    py::class_<ManipulatorParams>(m, "ManipulatorParams")
        .def(py::init<>())
        .def_readwrite("attachment_offset", &ManipulatorParams::attachment_offset)
        .def_readwrite("link1", &ManipulatorParams::link1)
        .def_readwrite("link2", &ManipulatorParams::link2)
        .def_readwrite("joint_lower_limit", &ManipulatorParams::joint_lower_limit)
        .def_readwrite("joint_upper_limit", &ManipulatorParams::joint_upper_limit)
        .def_readwrite("max_joint_torque", &ManipulatorParams::max_joint_torque);

    py::class_<EnvironmentParams>(m, "EnvironmentParams")
        .def(py::init<>())
        .def_readwrite("gravity", &EnvironmentParams::gravity)
        .def_readwrite("air_density", &EnvironmentParams::air_density);

    // ── RigidBody ──
    py::class_<RigidBody>(m, "RigidBody")
        .def(py::init<>())
        .def(py::init<double, const Mat3&, double, double>(),
             py::arg("mass"), py::arg("inertia"),
             py::arg("length") = 0.0, py::arg("com_distance") = 0.0)
        .def("mass", &RigidBody::mass)
        .def("inertia", &RigidBody::inertia)
        .def("length", &RigidBody::length)
        .def("com_distance", &RigidBody::com_distance);

    // ── Quadrotor ──
    py::class_<Quadrotor>(m, "Quadrotor")
        .def(py::init<const QuadrotorParams&>())
        .def("compute_forces", &Quadrotor::compute_forces)
        .def("compute_torques", &Quadrotor::compute_torques)
        .def("compute_drag", &Quadrotor::compute_drag)
        .def("mixing_matrix", &Quadrotor::mixing_matrix)
        .def("mixing_matrix_inv", &Quadrotor::mixing_matrix_inv)
        .def("hover_thrust", &Quadrotor::hover_thrust)
        .def("body", &Quadrotor::body, py::return_value_policy::reference_internal)
        .def("params", &Quadrotor::params, py::return_value_policy::reference_internal);

    // ── Manipulator ──
    py::class_<Manipulator>(m, "Manipulator")
        .def(py::init<const ManipulatorParams&>())
        .def("forward_kinematics", &Manipulator::forward_kinematics)
        .def("link1_com_position", &Manipulator::link1_com_position)
        .def("link2_com_position", &Manipulator::link2_com_position)
        .def("jacobian", &Manipulator::jacobian)
        .def("link1_com_jacobian", &Manipulator::link1_com_jacobian)
        .def("link2_com_jacobian", &Manipulator::link2_com_jacobian)
        .def("mass_matrix", &Manipulator::mass_matrix)
        .def("coriolis_matrix", &Manipulator::coriolis_matrix)
        .def("gravity_vector", &Manipulator::gravity_vector)
        .def("joint_torque_to_body_wrench", &Manipulator::joint_torque_to_body_wrench)
        .def("params", &Manipulator::params, py::return_value_policy::reference_internal)
        .def("attachment_offset", &Manipulator::attachment_offset);

    // ── Integrators ──
    py::class_<IntegratorBase, std::shared_ptr<IntegratorBase>>(m, "IntegratorBase")
        .def("name", &IntegratorBase::name);

    py::class_<RK4Integrator, IntegratorBase, std::shared_ptr<RK4Integrator>>(m, "RK4Integrator")
        .def(py::init<>());

    py::class_<RKF45Integrator::Config>(m, "RKF45Config")
        .def(py::init<>())
        .def_readwrite("atol", &RKF45Integrator::Config::atol)
        .def_readwrite("rtol", &RKF45Integrator::Config::rtol)
        .def_readwrite("dt_min", &RKF45Integrator::Config::dt_min)
        .def_readwrite("dt_max", &RKF45Integrator::Config::dt_max)
        .def_readwrite("safety", &RKF45Integrator::Config::safety);

    py::class_<RKF45Integrator, IntegratorBase, std::shared_ptr<RKF45Integrator>>(m, "RKF45Integrator")
        .def(py::init<>())
        .def(py::init<const RKF45Integrator::Config&>())
        .def("last_step_size", &RKF45Integrator::last_step_size);

    // ── AerialManipulatorSystem ──
    py::class_<AerialManipulatorSystem>(m, "AerialManipulatorSystem")
        .def(py::init<const QuadrotorParams&, const ManipulatorParams&, const EnvironmentParams&>())
        .def("compute_state_derivative", &AerialManipulatorSystem::compute_state_derivative)
        .def("compute_mass_matrix", &AerialManipulatorSystem::compute_mass_matrix)
        .def("compute_coriolis_vector", &AerialManipulatorSystem::compute_coriolis_vector)
        .def("compute_gravity_vector", &AerialManipulatorSystem::compute_gravity_vector)
        .def("compute_input_matrix", &AerialManipulatorSystem::compute_input_matrix)
        .def("step", &AerialManipulatorSystem::step)
        .def("set_integrator", &AerialManipulatorSystem::set_integrator)
        .def("total_mass", &AerialManipulatorSystem::total_mass)
        .def("quadrotor", &AerialManipulatorSystem::quadrotor, py::return_value_policy::reference_internal)
        .def("manipulator", &AerialManipulatorSystem::manipulator, py::return_value_policy::reference_internal);

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
