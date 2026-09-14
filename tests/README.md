# OpWorks operator test bench

Correctness + performance testing for CUDA operators built on OpWorks, driven
by the official LeetGPU challenge cases.

## Layout

- `runner/` — test runner (git submodule: local LeetGPU judge + challenge pack)
- `solutions/` — one self-contained `.cu` per operator under test
  (local scratch, gitignored)
- `run.sh` — run every solution: example + functional + performance

## Usage

```bash
git submodule update --init --recursive   # once after cloning
tests/run.sh
```

Each solution is compiled with bare `nvcc` and exercised against the official
test cases; `run.sh` exports `CPLUS_INCLUDE_PATH` so solutions can write
`#include <opworks>`. Built `.so` files are cached under `runner/build/` and
rebuilt automatically when a source file changes.

Coverage:

| Challenge | Exercises |
|---|---|
| `1_vector_add` | elementwise kernel + `OPWORKS_GRID_LOOP` |
| `21_relu` | elementwise kernel, unary |
| `4_reduction` | two-pass reduction + `block_reduce` |
| `5_softmax` | `block_reduce_all` statistics + `launch` |
| `113_layer_normalization` | per-row mean/variance |
| `2_matrix_multiplication` | `ops::mat_mul` skeleton (tiled GEMM) |

## Adding an operator

1. Create `solutions/<name>.cu` defining `extern "C" void solve(...)`; the
   signature must match the challenge's `get_solve_signature()`. Write a
   `__global__` kernel that writes the output pointer directly, then fire it
   with `launch(kernel, grid, block, args...)` — framework primitives
   (`OPWORKS_GRID_LOOP` / `OPWORKS_BLOCK_LOOP`, `block_reduce_all`,
   `DeviceBuffer`, `num_blocks_for`) cover the boilerplate. See any existing
   solution for the pattern.
2. Add `run <challenge_id> <name>.cu` to `run.sh`

## Environment

- any Python with a CUDA-enabled PyTorch (the base conda env works)
- `nvcc` on `PATH` (CUDA 12.8)
