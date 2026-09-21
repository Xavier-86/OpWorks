# OpWorks operator test bench

Correctness + performance testing for CUDA operators built on OpWorks, driven
by the official challenge cases.

## Layout

- `runner/` — test runner (git submodule: local judge + challenge pack)
- `integration/` — six version-controlled CUDA adapters
- `unit/` — framework contracts, ownership, boundaries, streams and graph replay
- `model/` — Llama inference operator and model tests (need `tests/fixtures/llama/`,
  skipped with code 77 when fixtures are absent; built with `-DOPWORKS_BUILD_LLAMA=ON`).
  `ops_test.cu` covers each operator against `ops_fixtures.pack` plus the
  chunked-vs-explicit attention self-check at length 2500 and the bf16/WMMA
  GEMM paths; `llama_test.cu` / `llama_bf16_test.cu` share
  `model_test_body.cuh` (per-layer hidden alignment, logits, greedy
  regression over 20 prompts, KV-cache consistency, long-context prefill,
  session isolation) with FP32/BF16 thresholds; `test_tokenizer.py` covers
  the chat-template pipeline (determinism, single BOS, Unicode/newlines,
  special-token text).
- `run.sh` — configure/build with CMake, run CTest, then upstream challenge cases
- `run.py` — load CMake-built shared libraries and use the runner's case utilities
- `solutions/` — optional personal scratch, ignored and never required by tests

## Usage

```bash
git submodule update --init --recursive   # once after cloning
tests/run.sh
```

CMake builds the adapters under `build/integration/` with explicit C++17 and
CUDA architecture flags. It tracks included headers and compiler options, so
editing a `.cuh` triggers recompilation. The runner's standalone timestamp-only
`.cu` cache is not used; the submodule is unmodified.

```bash
CUDA_ARCHITECTURES=89 tests/run.sh --mode functional
tests/run.sh --warmup 5 --repeat 10
cmake --build build --clean-first --parallel 2  # optional full rebuild
```

Override the build directory with `OPWORKS_BUILD_DIR` and parallelism with
`OPWORKS_BUILD_JOBS`. `run.py` alone uses existing binaries; use `run.sh` after
source changes. The shell entry point requires a single-configuration generator
(Unix Makefiles or Ninja).

Framework tests do not need Python, PyTorch or submodules:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode 1 ./build/opworks_tests
compute-sanitizer --tool racecheck --error-exitcode 1 ./build/opworks_tests
```

Framework coverage includes empty/invalid sizes, pointer and scalar contracts,
builder arity and lengths, move ownership, const views, aligned/tail/unaligned
map, exact in-place map, both reduction overloads, non-default streams, reusable
workspace, graph capture/replay, row operations, matmul edge tiles and epilogues,
and inclusion from multiple translation units. Tests return skip code 77 if no
CUDA device/driver is available. Device memory-pool support is needed for the
convenience reduction tests.

Adapters synchronize at their foreign-function boundary, so correctness and
timing include a completed operation. The library itself does not synchronize
each launch. Benchmark latency includes adapter/allocation overhead and is not
a pure kernel-throughput measurement.
The reduction adapter caches workspace per host thread/device for repeated
measurements. Framework tests separately exercise the convenience allocator.

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

1. Create `integration/<name>.cu` defining `extern "C" void solve(...)`; the
   signature must match the challenge's `get_solve_signature()`. If an `ops/`
   skeleton fits (map / reduce / softmax / layer_norm / mat_mul), the solution
   is a one-liner — see any existing solution. Otherwise write your own
   `__global__` kernel that writes the output pointer directly and fire it
   with `launch(kernel, grid, block, args...)`, then `synchronize()` at the
   adapter boundary; framework primitives
   (`OPWORKS_GRID_LOOP` / `OPWORKS_BLOCK_LOOP`, `block_reduce_all`,
   `DeviceBuffer`, `num_blocks_for`) cover the boilerplate.
2. Add the adapter target to `CMakeLists.txt` and its challenge mapping to `run.py`.

## Environment

- CMake 3.24+, a host C++ compiler, and `nvcc` on `PATH` (CUDA 11.2+; tested on 12.8)
- Python with CUDA-enabled PyTorch for upstream integration tests only
- a compatible NVIDIA GPU for runtime tests
