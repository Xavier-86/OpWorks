#!/usr/bin/env bash
# Run every operator in solutions/ against the official test cases:
# correctness (example + functional) and performance (median / p20 latency).
set -euo pipefail

# The judge compiles with bare `nvcc` (no -I); expose include/ via the
# environment so solutions can write #include <opworks>.
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export CPLUS_INCLUDE_PATH="$ROOT/include${CPLUS_INCLUDE_PATH:+:$CPLUS_INCLUDE_PATH}"

cd "$ROOT/tests/runner"

run() {
    echo "=== $1 ==="
    python scripts/local_test.py "$1" --language cuda --mode all --solution "../solutions/$2"
}

run 1_vector_add vector_add.cu
run 21_relu relu.cu
run 4_reduction reduction.cu
run 5_softmax softmax.cu
run 113_layer_normalization layer_normalization.cu
run 2_matrix_multiplication matrix_multiplication.cu
