# OpWorks

A minimal header-only CUDA operator **framework**. C++17, float32, no
dependencies beyond the CUDA toolkit — no build system required.

The library ships no concrete operators. It provides memory management, launch
helpers, block-reduction primitives, two generic builders (map / reduce), and
an `ops/` layer of reusable operator skeletons with user-supplied hooks.

```cpp
#include <opworks>
using namespace opworks;

DeviceBuffer a = DeviceBuffer::from_host(host_a);
DeviceBuffer b = DeviceBuffer::from_host(host_b);

// elementwise: c = a + b
DeviceBuffer c = ElementwiseBuilder(a, b).apply<Add>();

// reduction: s = sum(a)
float s = ReductionBuilder(a).apply<Sum>().to_host_scalar();
```

Every builder call allocates the output, computes the launch config, launches
the kernel, and checks errors internally.

## Build

```bash
nvcc -std=c++17 -arch=sm_89 -Iinclude main.cu -o main
```

## Custom functors

Builders are templates over user-defined functors. Stateless or stateful:

```cpp
// elementwise functor: kArity declares the arity (1 = unary, 2 = binary)
struct Add {
  static constexpr int kArity = 2;
  __device__ float operator()(float a, float b) const { return a + b; }
};

// reduction functor: identity element + associative combine
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

Anything beyond map/reduce (softmax, layernorm, matmul, ...) is a plain CUDA
kernel written by you, built on the framework primitives — e.g. row statistics
via `block_reduce_all`:

```cpp
// one block per row of a rows x cols matrix
__global__ void row_mean(const float* in, float* out, int cols) {
  float local = Sum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) { local = Sum::combine(local, in[blockIdx.x * cols + j]); }
  float mean = block_reduce_all<Sum>(local) / cols;  // valid on all threads
  if (threadIdx.x == 0) out[blockIdx.x] = mean;
}

launch(row_mean, rows, kThreads, in, out, cols);  // launch + check + sync
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

- `launch(kernel, grid, block, args...)` — kernel launch + error check + sync;
  grid/block take `dim3` or plain ints
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

- `DeviceBuffer::from_host(vec)` / `to_host()` / `to_host_scalar()` — host copies
- `DeviceBuffer::wrap(ptr, n)` — non-owning view of external device memory
- `data()` / `size()`

## Layout

```
include/
├── opworks                  # extensionless umbrella — #include <opworks>
├── core/
│   ├── config.cuh           # tuning knobs (kThreads, kNumWaves, matmul tiles; -DOPWORKS_* overridable)
│   ├── cuda_utils.cuh       # CUDA_CHECK + launch + loop macros + grid sizing
│   ├── block_reduce.cuh     # warp-shuffle block reduction primitives
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
