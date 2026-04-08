# Contributing

Thank you for your interest in contributing to the Aerial Manipulator Simulation!

## Development Setup

1. Clone the repository
2. Install dependencies: `pip install -r requirements.txt`
3. Build C++ core: `bash scripts/build.sh`
4. Run tests: `pytest tests/ -v`

## Code Style

- **C++**: Follow the existing style (Eigen types, namespace `aerial_manipulator`)
- **Python**: PEP 8, type hints for public APIs, numpy-style docstrings
- **Tests**: pytest, one test class per concept, descriptive assertion messages

## Pull Request Process

1. Create a feature branch from `main`
2. Ensure all tests pass (`pytest tests/ -v`)
3. Update documentation if changing public APIs
4. Update CHANGELOG.md under `[Unreleased]`
