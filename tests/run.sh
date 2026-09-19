#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="${OPWORKS_BUILD_DIR:-$ROOT/build}"
ARCH="${CUDA_ARCHITECTURES:-}"
if [[ -z "$ARCH" ]]; then
    ARCH="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .' || true)"
    ARCH="${ARCH:-80}"
fi

cmake -S "$ROOT" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON -DCMAKE_CUDA_ARCHITECTURES="$ARCH"
cmake --build "$BUILD_DIR" --parallel "${OPWORKS_BUILD_JOBS:-2}"
ctest --test-dir "$BUILD_DIR" --output-on-failure
python "$ROOT/tests/run.py" --build-dir "$BUILD_DIR" "$@"
