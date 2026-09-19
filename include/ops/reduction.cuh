#pragma once

#include "../core/device_span.cuh"
#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

template <typename Op> __global__ void reduce_partial_kernel(const float *in, float *partial, int n) {
    float acc = Op::identity();
    OPWORKS_GRID_LOOP(i, n) {
        acc = Op::combine(acc, in[i]);
    }
    acc = block_reduce<Op>(acc);
    if (threadIdx.x == 0)
        partial[blockIdx.x] = acc;
}

template <typename Op> __global__ void reduce_final_kernel(const float *partial, float *out, int num_blocks) {
    float acc = Op::identity();
    OPWORKS_BLOCK_LOOP(i, num_blocks) {
        acc = Op::combine(acc, partial[i]);
    }
    acc = block_reduce<Op>(acc);
    if (threadIdx.x == 0)
        out[0] = acc;
}

} // namespace opworks::detail

namespace opworks::ops {

// Workspace size in floats. Small/empty inputs use a single kernel and no scratch.
inline int reduction_workspace_size(int n) {
    int blocks = detail::blocks_for(n);
    if (blocks <= 1)
        return 0;
    return blocks < kMaxThreadsPerBlock ? blocks : kMaxThreadsPerBlock;
}

// Allocation-free asynchronous path. Workspace must remain alive until stream
// completion and must not overlap input/output or concurrent reductions.
template <typename Op>
inline void reduce(const float *in, float *out, int n, DeviceSpan<float> workspace, cudaStream_t stream = nullptr) {
    const int blocks = reduction_workspace_size(n);
    detail::require(out && (n == 0 || in), "reduction requires valid input/output pointers");
    detail::require(workspace.size() >= blocks, "reduction workspace is too small");
    if (blocks == 0) {
        launch_async(stream, detail::reduce_partial_kernel<Op>, 1, kThreads, in, out, n);
        return;
    }
    launch_async(stream, detail::reduce_partial_kernel<Op>, blocks, kThreads, in, workspace.data(), n);
    launch_async(stream, detail::reduce_final_kernel<Op>, 1, kThreads, workspace.data(), out, blocks);
}

// Convenience path: stream-ordered allocation/free keeps scratch alive without
// synchronizing the host. Requires CUDA 11.2+ and memory-pool support.
template <typename Op> inline void reduce(const float *in, float *out, int n, cudaStream_t stream = nullptr) {
    const int count = reduction_workspace_size(n);
    detail::require(out && (n == 0 || in), "reduction requires valid input/output pointers");
    float *scratch = nullptr;
    if (count != 0)
        OPWORKS_CUDA_CHECK(cudaMallocAsync(&scratch, sizeof(float) * count, stream));
    reduce<Op>(in, out, n, DeviceSpan<float>(scratch, count), stream);
    if (scratch)
        OPWORKS_CUDA_CHECK(cudaFreeAsync(scratch, stream));
}

} // namespace opworks::ops
