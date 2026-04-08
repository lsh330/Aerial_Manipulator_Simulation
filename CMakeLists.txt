cmake_minimum_required(VERSION 3.16)
project(aerial_manipulator VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ── Dependencies ──
find_package(Eigen3 3.4 REQUIRED)
find_package(pybind11 REQUIRED)

# ── C++ core library ──
add_subdirectory(core)
