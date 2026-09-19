# OpWorks

A minimal header-only CUDA operator **framework**. C++17, float32, CUDA 11.2+.
Direct `nvcc` use needs no build system; CMake is provided for consumers and tests.

It provides memory management, launch helpers, block-reduction primitives,
two generic builders (map / reduce), and an `ops/` layer with generic map/reduce,
row softmax/layer normalization, and tiled matrix multiplication.

```cpp
#include <opworks>
using namespace opworks;

struct Add {
  static constexpr int kArity = 2;
  __device__ float operator()(float a, float b) const { return a + b; }
};
struct Sum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};

std::vector<float> host_a{1, 2, 3}, host_b{4, 5, 6};
DeviceBuffer a = DeviceBuffer::from_host(host_a);
DeviceBuffer b = DeviceBuffer::from_host(host_b);

// elementwise: c = a + b
DeviceBuffer c = ElementwiseBuilder(a, b).apply<Add>();

// reduction: s = sum(a)
float s = ReductionBuilder(a).apply<Sum>().to_host_scalar();
```

Every builder call allocates the output and enqueues the kernel. Host copies
such as `to_host_scalar()` wait for completion. Inputs and outputs must remain
alive until queued work finishes.

## Build

```bash
nvcc -std=c++17 -arch=sm_89 -Iinclude main.cu -o main
```

Choose an architecture supported by your toolkit and target GPU. With CMake:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2
ctest --test-dir build --output-on-failure
```

Consumers can use `add_subdirectory(OpWorks)` and link `OpWorks::opworks`.
Set `BUILD_TESTING=OFF` for library-only configuration. CMake defaults to
architecture 80; `tests/run.sh` detects the local GPU or accepts
`CUDA_ARCHITECTURES=89`. Set `CUDACXX` on the initial configure to select a toolkit.

## Custom functors

Builders are templates over user-defined functors. Stateless or stateful:

```cpp
// elementwise functor: kArity declares the arity (1 = unary, 2 = binary)
struct Add {
  static constexpr int kArity = 2;
  __device__ float operator()(float a, float b) const { return a + b; }
};

// reduction functor: identity element + associative, commutative combine
struct Sum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};

// stateful functor: runtime parameters ride into the kernel by value
struct ScaleAdd {
  static constexpr int kArity = 2;
  float alpha;
  __device__ float operator()(float a, float b) const { return alpha * a + b; }
};

ElementwiseBuilder(a, b).apply<Add>();            // stateless
ElementwiseBuilder(a, b).apply(ScaleAdd{2.5f});   // deduced, carries alpha
```

## Custom operators

For operations beyond the supplied skeletons, write a CUDA kernel using
framework primitives — e.g. row statistics via `block_reduce_all`:

```cpp
// one block per row of a rows x cols matrix
__global__ void row_mean(const float* in, float* out, int cols) {
  float local = Sum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) { local = Sum::combine(local, in[blockIdx.x * cols + j]); }
  float mean = block_reduce_all<Sum>(local) / cols;  // valid on all threads
  if (threadIdx.x == 0) out[blockIdx.x] = mean;
}

launch(row_mean, rows, kThreads, in, out, cols);  // enqueue + launch-error check
synchronize();                                // wait when the result is needed
```

## API

| Builder | Call | Result |
|---|---|---|
| `ElementwiseBuilder(a, b)` / `ElementwiseBuilder(a)` | `.apply<Op>()` / `.apply(op)` | binary / unary elementwise op |
| `ReductionBuilder(a)` | `.apply<Op>()` | 1-element `DeviceBuffer` |

Operator skeletons (`ops/` layer, raw pointers):

| Skeleton | Call | Result |
|---|---|---|
| `map` | `ops::map(a, b, out, n, op)` / `ops::map(in, out, n, op)` | binary / unary elementwise, 4-wide packed when 16B-aligned |
| `reduce` | `ops::reduce<Op>(in, out, n)` | two-pass reduction to `out[0]` |
| `softmax` | `ops::softmax(in, out, rows, cols)` | row-wise softmax |
| `layer_norm` | `ops::layer_norm(in, out, gamma, beta, rows, cols, eps)` | row-wise layer normalization |
| `mat_mul` | `ops::mat_mul(A, B, C, M, N, K[, epilogue])` | `C(MxK) = epilogue(A(MxN) @ B(NxK))` |

`ElementwiseBuilder` vectorizes 4-wide automatically when inputs are 16-byte
aligned (scalar tail + scalar fallback otherwise) — you always write a plain
scalar functor.

Framework primitives for custom kernels:

- `launch(kernel, grid, block, args...)` — asynchronous default-stream launch;
  grid/block take `dim3` or plain ints
- `launch_async(stream, kernel, grid, block, args...)` — explicit-stream launch
- `launch_sync([stream,] kernel, grid, block, args...)` — launch and wait for that stream
- `synchronize(stream = nullptr)` — wait and check asynchronous execution errors
- `OPWORKS_GRID_LOOP(i, n)` / `OPWORKS_BLOCK_LOOP(i, n)` — grid/block-stride
  loops for custom kernels
- `block_reduce<Op>(val)` — block-wide reduce, result on thread 0
- `block_reduce_all<Op>(val)` — same, broadcast to all threads
- `num_blocks_for(n)` — adaptive grid size, capped at `kNumWaves` waves per SM
- `kThreads` — default block size
- `OPWORKS_CUDA_CHECK(expr)` — error checking

All tuning knobs (`kThreads`, `kNumWaves`, pack size, matmul tile shapes) live
in `core/config.cuh` and can be overridden at compile time, e.g.
`nvcc -DOPWORKS_THREADS=512`.

`DeviceBuffer` is an RAII wrapper around `cudaMalloc`:

- `DeviceBuffer::from_host(vec, stream)` / `to_host(stream)` / `to_host_scalar(stream)` —
  blocking host copies; stream defaults to the default stream
- `buffer.view()` — a mutable or const borrowed `DeviceSpan`, matching the owner
- `DeviceSpan<float>(ptr, n)` / `DeviceSpan<const float>(ptr, n)` — external memory
- `data()` / `size()`

## Execution, ownership, and contracts

`launch` no longer calls `cudaDeviceSynchronize`. All raw-pointer `ops` calls
accept a final `cudaStream_t` argument (default `nullptr`) and enqueue work.
Launch errors are checked immediately; execution errors are checked at
`synchronize`, `launch_sync`, or a blocking host copy. Use `launch_sync` while
debugging an individual kernel. A device-wide barrier is only appropriate when
the caller explicitly needs to wait for work across all streams.

For example, queue a chain on one stream and wait once:

```cpp
cudaStream_t stream;
OPWORKS_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
DeviceBuffer out(a.size()), scalar(1);
DeviceBuffer workspace(ops::reduction_workspace_size(a.size()));
ops::map(a.data(), b.data(), out.data(), a.size(), Add{}, stream);
ops::reduce<Sum>(out.data(), scalar.data(), out.size(), workspace.view(), stream);
float result = scalar.to_host_scalar(stream);
OPWORKS_CUDA_CHECK(cudaStreamDestroy(stream));
```

The workspace overload of `reduce` allocates nothing and supports graph capture.
Keep the workspace alive until completion; reuse it on the same stream, or wait
before using it on another stream. It must not overlap the input or output.
The convenience overload `reduce<Op>(in, out, n, stream)` uses
`cudaMallocAsync` / `cudaFreeAsync` for multi-block inputs, so its temporary memory
is released after the last kernel. That overload requires device memory-pool
support; the explicit workspace overload does not. Empty reductions write
`Op::identity()` and small reductions use one kernel without scratch allocation.
For repeated calls, prefer explicit workspace: synchronizing after each
convenience call can cause the default CUDA memory pool to release cached memory,
adding allocation overhead to subsequent calls.

Builders accept buffers or spans and borrow their inputs. They reject temporary
buffer owners, but callers must still preserve the underlying storage through
`apply` and GPU completion. `ElementwiseBuilder::apply(op, stream)` and
`ReductionBuilder::apply<Op>([workspace,] stream)` accept an explicit stream.
Builders and `DeviceBuffer` still use `cudaMalloc`/`cudaFree`, which can introduce
synchronization; preallocate buffers and use raw-pointer ops for a hot path or
graph capture. Cross-stream dependencies are the caller's responsibility
(use CUDA events, or synchronize the producer).

Sizes are nonnegative `int`, with at most `INT_MAX` elements per tensor. Invalid
sizes, pointers, mismatched builder lengths/arity and invalid epsilon throw
`std::invalid_argument`. CUDA runtime failures retain the fail-fast
`OPWORKS_CUDA_CHECK` policy. Raw-pointer callers must supply sufficient storage.

- Empty map and zero-row normalization are no-ops. Normalization with nonzero
  rows requires positive columns and finite positive layer-norm epsilon.
- Matmul with zero output rows/columns is a no-op; a zero reduction dimension
  writes `epilogue(0)`. Output must not overlap either input.
- Map supports exact in-place operation, not partially overlapping ranges.
- `to_host_scalar()` requires exactly one element. Empty buffers allocate nothing.
- Reduction functors require an identity and an associative, commutative combine;
  floating-point rounding depends on reduction order.
- `block_reduce` / `block_reduce_all` require a one-dimensional block with a
  positive multiple of 32 threads, and participation by the entire block.

Migration: replace `DeviceBuffer::wrap` with `DeviceSpan<float>` or
`DeviceSpan<const float>`; replace old synchronous `launch` calls with
`launch_sync`, or synchronize once at the end of a chain.

## Layout

```
include/
├── opworks                  # extensionless umbrella — #include <opworks>
├── core/
│   ├── config.cuh           # tuning knobs (kThreads, kNumWaves, matmul tiles; -DOPWORKS_* overridable)
│   ├── cuda_utils.cuh       # CUDA_CHECK + launch + loop macros + grid sizing
│   ├── block_reduce.cuh     # warp-shuffle block reduction primitives
│   ├── device_span.cuh      # const-correct, non-owning device views
│   └── device_buffer.cuh    # DeviceBuffer — RAII device memory
├── builders/
│   ├── elementwise.cuh      # DeviceBuffer sugar over ops::map
│   └── reduction.cuh        # DeviceBuffer sugar over ops::reduce
└── ops/                     # raw-pointer operator skeletons
    ├── elementwise.cuh      # ops::map, variadic pack kernel
    ├── reduction.cuh        # ops::reduce, two-pass
    ├── softmax.cuh          # ops::softmax, row-wise
    ├── layer_norm.cuh       # ops::layer_norm, row-wise
    └── matmul.cuh           # ops::mat_mul, tiled GEMM with epilogue hook
```

Layering: `core` ← `ops` ← `builders`. Kernels and internal
functors live in `opworks::detail`; the public surface is `opworks::` plus
`opworks::ops::`.

## Design notes

- grid/block-stride loop macros (`OPWORKS_GRID_LOOP` / `OPWORKS_BLOCK_LOOP`)
  let one kernel cover any problem size without a grid-size calculation per
  launch
- one generic variadic elementwise kernel instantiated at pack size 4 / 1,
  with the scalar tail folded into the same kernel
- adaptive grid sizing capped at `kNumWaves` resident blocks per SM, so huge
  inputs do not flood the scheduler
- stateful functors carried into the kernel by value (kernel arguments are
  already device-side constants, so no factory indirection is needed)

## Editor setup (clangd)

clangd cannot handle nvcc commands ("Unable to handle compilation, expected
exactly one compiler job"). Generate a local `.clangd` that parses CUDA with
clang++ instead — paths are auto-detected (nvcc location, GPU arch):

```bash
scripts/gen_clangd.sh   # then restart the clangd server in your editor
```

The generated `.clangd` contains machine-specific absolute paths and is
gitignored; re-run the script on a new machine.

## Checks

Run `tests/run.sh` for framework tests and the six upstream challenge adapters.
See [tests/README.md](tests/README.md) for setup, coverage and sanitizer commands.
Formatting uses `.clang-format` with clang-format 18. CI checks formatting and
compiles all test consumers without requiring a GPU; runtime checks run on a GPU host.
