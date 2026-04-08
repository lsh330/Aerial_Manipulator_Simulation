add_library(aerial_manipulator_core STATIC
    src/rigid_body.cpp
    src/quadrotor.cpp
    src/manipulator.cpp
    src/aerial_manipulator_system.cpp
    src/rk4_integrator.cpp
    src/rkf45_integrator.cpp
)

target_include_directories(aerial_manipulator_core PUBLIC include)
target_link_libraries(aerial_manipulator_core PUBLIC Eigen3::Eigen)

# pybind11 Python module
add_subdirectory(bindings)
