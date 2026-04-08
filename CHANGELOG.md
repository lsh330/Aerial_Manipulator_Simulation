# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [0.3.0] - 2026-04-09

### Changed
- **NMPC**: Exact Christoffel-based Coriolis in CasADi dynamics (eliminates plant-model mismatch)
- **NMPC**: Reduced default horizon N=20→10 for real-time feasibility (~20ms solve)
- **NMPC**: DARE-based terminal cost replaces heuristic 5Q

### Added
- **Control**: Solver fallback mechanism (returns previous/hover input on IPOPT failure)
- **Control**: Joint position constraints enforced in NLP
- **Control**: Input slew rate constraints (motor: 8 N/step, joint: 3 N·m/step)
- **Control**: Integral action via disturbance estimator for offset-free tracking
- **Dynamics**: Wind disturbance model (constant wind velocity)
- **Dynamics**: End-effector contact force model (spring-damper)
- **Safety**: Zero quaternion protection (returns identity rotation)
- **Safety**: Robust LDLT fallback in dynamics solve
- **Safety**: Thread-safe step() function (value capture instead of mutable member)
- **Tests**: Coriolis skew-symmetry validation (Ṁ - 2C)
- **Tests**: Momentum conservation test (zero-gravity)
- **Tests**: CasADi-C++ cross-validation test
- **Tests**: Manipulator M/C/G unit tests
- **Tests**: Mixing matrix invertibility test
- **Tests**: Step size sensitivity test
- **Tests**: NMPC robustness test
- **Docs**: Assumptions & Limitations section in README
- **Docs**: CHANGELOG.md, CONTRIBUTING.md

### Fixed
- Energy conservation test: potential energy now includes manipulator link COM heights
- Free-fall test: uses drag_coeff=0 for exact analytical comparison
- conftest.py: corrected fixture file reference (controller_params.yaml → nmpc_params.yaml)
- types.hpp: added SI unit annotations to all parameter struct fields
- pybind11 bindings: added Python help() docstrings

## [0.2.0] - 2026-04-01

### Changed
- Replaced PID/SDRE hierarchical controller with unified NMPC
- CasADi CodeGenerator DLL compilation for 9x solver speedup

## [0.1.0] - 2026-03-15

### Added
- C++ core: coupled Newton-Euler dynamics for quadrotor + 2-DOF manipulator
- RK4 and RKF45 numerical integrators
- pybind11 Python bindings
- Simulation loop with data logging and visualization
- 3 example scenarios (hover, circle tracking, arm motion)
- Unit and validation test suites (72/72 pass)
