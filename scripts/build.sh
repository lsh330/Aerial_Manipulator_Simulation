#!/usr/bin/env bash
# Build the C++ core engine with pybind11 bindings
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${PROJECT_ROOT}/build"

echo "=== Building Aerial Manipulator C++ Engine ==="

mkdir -p "${BUILD_DIR}"
cd "${BUILD_DIR}"

cmake "${PROJECT_ROOT}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DPYTHON_EXECUTABLE="$(which python3)"

cmake --build . --config Release -j "$(nproc 2>/dev/null || echo 4)"

echo "=== Build complete ==="
