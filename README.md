# OpWorks

A minimal header-only CUDA operator **framework**. C++17, float32, CUDA 11.2+.
Direct `nvcc` use needs no build system; CMake is provided for consumers and tests.

It provides memory management, launch helpers, block-reduction primitives,
two generic builders (map / reduce), and an `ops/` layer with generic map/reduce,
row softmax/layer normalization, and tiled matrix multiplication.

On top of the framework, an opt-in Llama-3.2-1B-Instruct inference runtime
(`-DOPWORKS_BUILD_LLAMA=ON`) runs the full model forward pass, KV cache and
greedy / temperature-top-p sampling on OpWorks' own kernels — see
[Llama inference runtime](#llama-inference-runtime).

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

Builder configuration scaffolding is also available for operations that will
grow specialized kernels:

```cpp
auto row = RowwiseBuilder::rms_norm(input.view(), output.view(), weight.view(), rows, cols);
row.apply([](const auto &config) { /* launch the RMSNorm kernel */ });

MatmulBuilder gemm(a.view(), b.view(), c.view(), M, N, K);
gemm.transpose_b().bias(bias.view()).activation(MatmulActivation::relu);
gemm.apply([](const auto &config) { /* launch the fused GEMM kernel */ });
```

`ScanBuilder`, `TransformBuilder`, and `GatherScatterBuilder` use the same
callback boundary. They validate buffer sizes and preserve operation metadata;
their callbacks are intentionally the extension point for future kernels.

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

## Llama inference runtime

Opt-in (`-DOPWORKS_BUILD_LLAMA=ON`) implementation of Llama-3.2-1B-Instruct
inference on OpWorks' own CUDA kernels. FP32 and BF16 weight packs, batch=1,
up to 8192 total sequence length, greedy and temperature/top-p sampling.
Python drives tokenization (HF chat template) and the CLI; the model forward,
KV cache and decode loop run entirely in C++/CUDA behind a small C ABI
(`build/libopworks_llama.so`, called via ctypes — no LibTorch dependency).

### Quickstart

```bash
# one-time: convert an HF snapshot into an OpWorks pack (FP32 or BF16)
python scripts/llama/export_weights.py --model-dir /path/to/snapshot \
    --output data/llama-fp32.pack --dtype float32    # or bfloat16

cmake -S . -B build -DOPWORKS_BUILD_LLAMA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_CUDA_ARCHITECTURES=89
cmake --build build --parallel 2

python examples/llama/generate.py --model data/llama-bf16.pack \
    --tokenizer /path/to/snapshot --prompt "Explain what a GPU does." \
    --max-seq-len 8192 --max-new-tokens 128 \
    [--temperature 0.6 --top-p 0.9 --seed 42]   # default temperature=0 = greedy

python scripts/llama/benchmark.py --model data/llama-bf16.pack \
    --tokenizer /path/to/snapshot [--hf-baseline]
```

Weight packs, snapshots and fixtures live under the gitignored `data/` and
`tests/fixtures/llama/`; `data/manifest.json` records the locked versions
(source revision, pack sha256, toolchain). The weights came from the ungated
`unsloth/Llama-3.2-1B-Instruct` mirror at commit `5a8abab4` (huggingface.co
is unreachable from this host and the official repo is gated; the safetensors
sha256 is recorded in every pack's manifest). `--hf-baseline` needs Python
headers for torch's triton path (e.g. `python3-dev`, or point `CPATH` at an
existing `include/python3.12`).

### Design

- Layering stays `core` ← `ops` ← `builders`; the model code lives in
  `include/runtime/` (pack reader, config, weights, KV cache, workspace) and
  `include/models/llama.cuh`, so generic ops never depend on the model.
- Model ops under `include/ops/`: `embedding`, `rms_norm`, `linear` (fp32 and
  bf16 weights, fp32 accumulation), `linear_mma` (WMMA tensor-core bf16 for
  prefill GEMMs), `rope` (llama3 frequency scaling), `swiglu`, `argmax`,
  `split_heads`, `causal_scale`, `attention_prefill` (chunked online-softmax,
  used past 2048 rows — no quadratic score matrix), `attention_decode`
  (single-query GQA over the KV cache).
- Weights keep the HF `[out, in]` layout and are read transposed on the fly;
  the embedding doubles as the LM head (tied). Activations and the KV cache
  are fp32; weights may be fp32 or bf16.
- Sessions own their KV cache (`[8, max_seq, 64]` per layer per K/V), scratch
  buffers and CUDA stream; nothing is allocated per token on the hot path.
- The C ABI captures all C++ exceptions; token sampling happens host-side
  from the last-position logits, so the FFI stays small.

### Correctness

`ctest -R llama` covers: every operator against exported fixtures
(`ops_fixtures.pack`), per-layer hidden-state and final-logits alignment with
a fixed HF reference (`transformers` 5.17, fp32, eager attention, TF32 off),
greedy generation regression over 20 fixed prompts (0 token divergences,
fp32 and bf16 engines), KV-cache consistency (decode vs full-prefix recompute
at lengths 1/2/31/32/33/127/128/129 plus capacity boundaries), long-context
(2500-token) prefill through the chunked path, tokenizer pipeline edge cases
(empty/Unicode/newlines/multi-turn/special-token text), session isolation and
reset, and CLI edge cases (out-of-vocab ids, over-capacity, empty prompt,
`max_new_tokens` 0/1). `compute-sanitizer` memcheck reports 0 errors over
the operator and full bf16 model tests (use the CUDA 12.8 sanitizer at
`/usr/local/cuda-12.8/bin/compute-sanitizer`; the PATH default 12.0 build
lacks its injection library). Tests skip with code 77 when the packs or
fixtures are absent; regenerate them with `scripts/llama/export_reference.py`
and `scripts/llama/export_regression.py`.

### Performance

Measured on this repo's RTX 4080 SUPER (batch=1, greedy, 128 generated
tokens, median of repetitions after warmup; reproduce with
`scripts/llama/benchmark.py` and `scripts/llama/benchmark_vllm.py`).

Decode (tokens/s) and TTFT (prefill, ms) versus vLLM 0.29 (bf16, CUDA
graphs + paged attention, prefix caching disabled so every rep recomputes
the prompt):

| prompt tokens | OpWorks TTFT | vLLM TTFT | OpWorks decode | vLLM decode | HF bf16 e2e |
| --- | --- | --- | --- | --- | --- |
| 128 | 48.6 ms | 6.9 ms | 232 tok/s | 250 tok/s | 183 tok/s |
| 512 | 103.5 ms | 15.1 ms | 215 tok/s | 250 tok/s | 180 tok/s |
| 1024 | 183.9 ms | 27.0 ms | 198 tok/s | 250 tok/s | 177 tok/s |
| 4096 | 711.3 ms | 103.4 ms | 134 tok/s | 240 tok/s | 150 tok/s |
| 8000 | 1874.3 ms | 226.9 ms | 95 tok/s | 230 tok/s | 122 tok/s |

Reading the gap: at short contexts decode is bandwidth-bound (~2.3 GiB of
bf16 weights per token) and OpWorks sits within ~7% of vLLM; the gap grows
to ~2.4x at 8K contexts where vLLM's flash-decoding wins. TTFT is 7-8x
behind vLLM across the board — vLLM prefill runs fused bf16 kernels under
CUDA graphs while our prefill is fp32-activation SIMT/WMMA kernels with
per-op launches. Known headroom, in order: kernel-launch overhead
(~190 launches/token, CUDA Graphs would remove it), split-KV decode
attention at long contexts, bf16 activations end-to-end.

Peak GPU memory: 4.6 GiB (BF16, 8K session) and 5.9 GiB (FP32, 2K session) —
within the 6/8 GiB design budgets. FP32 decode runs at 127 tok/s (128-token
prompt) for reference. Model load 1.0 s (BF16) / 3.6 s (FP32)
vs vLLM 9.4 s; tokenizer ~9 ms.

The vLLM baseline runs in its own venv (`uv pip install vllm ninja`);
flashinfer's JIT needs a CUDA >= 12.5 toolkit, so run with
`CUDA_HOME=/usr/local/cuda-12.8` (and `CPATH` pointing at Python headers)
on this host.

## Layout

```
include/
├── opworks                  # extensionless umbrella — #include <opworks>
├── core/
│   ├── config.cuh           # tuning knobs (kThreads, kNumWaves, matmul tiles; -DOPWORKS_* overridable)
│   ├── cuda_utils.cuh       # CUDA_CHECK + launch + loop macros + grid sizing
│   ├── block_reduce.cuh     # warp-shuffle block reduction primitives
│   ├── device_span.cuh      # const-correct, non-owning device views
│   ├── device_buffer.cuh    # DeviceBuffer — RAII device memory (fp32)
│   └── typed_device_buffer.cuh  # TypedDeviceBuffer<T> — fp32 / int32 / bf16
├── builders/
│   ├── elementwise.cuh      # DeviceBuffer sugar over ops::map
│   ├── reduction.cuh        # DeviceBuffer sugar over ops::reduce
│   ├── rowwise.cuh          # softmax/layernorm/RMSNorm configuration
│   ├── matmul.cuh           # GEMM transpose/bias/activation configuration
│   ├── scan.cuh             # prefix scan configuration
│   ├── transform.cuh        # layout/shape transform configuration
│   └── gather_scatter.cuh   # indexed access configuration
├── ops/                     # raw-pointer operator skeletons
│   ├── elementwise.cuh      # ops::map, variadic pack kernel
│   ├── reduction.cuh        # ops::reduce, two-pass
│   ├── softmax.cuh          # ops::softmax, row-wise
│   ├── layer_norm.cuh       # ops::layer_norm, row-wise
│   ├── matmul.cuh           # ops::mat_mul, tiled GEMM with epilogue hook
│   └── ...                  # model ops: embedding/rms_norm/linear(+mma)/rope/
│                            #   swiglu/argmax/split_heads/causal_scale/attention_*
├── runtime/                 # Llama runtime: json, pack_reader, model_config,
│                            #   weights, kv_cache, workspace
├── models/llama.cuh         # Llama decoder forward + sessions
src/llama_c_api.cu           # C ABI shared library (OPWORKS_BUILD_LLAMA)
examples/llama/generate.py   # Python ctypes CLI
scripts/llama/               # weight export, reference fixtures, benchmark
tests/model/                 # operator/model/tokenizer/edge tests for the runtime
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

Run `tests/run.sh` for framework tests and the six upstream challenge adapters
(needs a `python` on PATH, e.g. the local `.venv`). The Llama runtime tests
are separate: configure with `-DOPWORKS_BUILD_LLAMA=ON` and run
`ctest -R llama`. See [tests/README.md](tests/README.md) for setup, coverage
and sanitizer commands.
Formatting uses `.clang-format` with clang-format 18. CI checks formatting and
compiles all test consumers without requiring a GPU; runtime checks run on a GPU host.
