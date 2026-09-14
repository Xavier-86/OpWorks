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
| `1_vector_add` | `ops::map`, binary user functor |
| `21_relu` | `ops::map`, unary user functor |
| `4_reduction` | `ops::reduce` |
| `5_softmax` | `ops::softmax` |
| `113_layer_normalization` | `ops::layer_norm` |
| `2_matrix_multiplication` | `ops::mat_mul` |

## Adding an operator

1. Create `solutions/<name>.cu` defining `extern "C" void solve(...)`; the
   signature must match the challenge's `get_solve_signature()`. If an `ops/`
   skeleton fits (map / reduce / softmax / layer_norm / mat_mul), the solution
   is a one-liner — see any existing solution. Otherwise write your own
   `__global__` kernel that writes the output pointer directly and fire it
   with `launch(kernel, grid, block, args...)`; framework primitives
   (`OPWORKS_GRID_LOOP` / `OPWORKS_BLOCK_LOOP`, `block_reduce_all`,
   `DeviceBuffer`, `num_blocks_for`) cover the boilerplate.
2. Add `run <challenge_id> <name>.cu` to `run.sh`

## Environment

- any Python with a CUDA-enabled PyTorch (the base conda env works)
- `nvcc` on `PATH` (CUDA 12.8)
