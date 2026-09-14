# OpWorks operator test bench

Correctness + performance testing for CUDA operators built on OpWorks, driven
by the official LeetGPU challenge cases.

## Layout

- `runner/` — test runner (git submodule: local LeetGPU judge + challenge pack)
- `solutions/` — one self-contained `.cu` per operator under test
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
| `1_vector_add` | ElementwiseBuilder, binary functor |
| `21_relu` | ElementwiseBuilder, unary functor |
| `4_reduction` | ReductionBuilder |
| `5_softmax` | custom kernel: `block_reduce_all` + `launch` |
| `113_layer_normalization` | custom kernel: per-row mean/variance |
| `2_matrix_multiplication` | custom kernel: tiled GEMM |

## Adding an operator

1. Create `solutions/<name>.cu` defining `extern "C" void solve(...)`; the
   signature must match the challenge's `get_solve_signature()`. Two styles:
   - map/reduce-shaped op: define a functor, wrap inputs with
     `DeviceBuffer::wrap`, call the builder, `cudaMemcpy` the result back
     (see `vector_add.cu`)
   - anything else: write your own `__global__` kernel that writes the output
     directly and fire it with `launch(kernel, grid, block, args...)` (see
     `matrix_multiplication.cu`); block statistics via `block_reduce_all`
     (see `softmax.cu`)
2. Add `run <challenge_id> <name>.cu` to `run.sh`

## Environment

- any Python with a CUDA-enabled PyTorch (the base conda env works)
- `nvcc` on `PATH` (CUDA 12.8)
