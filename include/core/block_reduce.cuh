#pragma once

#include <cuda_runtime.h>

#include "config.cuh"

namespace opworks {

// Block-wide reduce; the result is only valid on thread 0. Every thread must
// participate in a 1D block whose size is a positive multiple of kWarpSize.
template <typename Op> __device__ float block_reduce(float val) {
    constexpr int kMaxWarps = kMaxThreadsPerBlock / kWarpSize;
    __shared__ float shared[kMaxWarps];
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;

    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
        val = Op::combine(val, __shfl_down_sync(kFullWarpMask, val, offset));

    if (lane == 0)
        shared[warp] = val;
    __syncthreads();

    int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
    val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : Op::identity();
    if (warp == 0) {
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1)
            val = Op::combine(val, __shfl_down_sync(kFullWarpMask, val, offset));
    }
    return val;
}

// Block-wide reduce + broadcast; the result is valid on all threads.
template <typename Op> __device__ float block_reduce_all(float val) {
    __shared__ float result;
    float r = block_reduce<Op>(val);
    if (threadIdx.x == 0)
        result = r;
    __syncthreads();
    return result;
}

} // namespace opworks

namespace opworks::detail {

// Internal reduction ops shared by the ops/ skeletons.
struct ReduceSum {
    static __device__ float identity() { return 0.f; }
    static __device__ float combine(float a, float b) { return a + b; }
};
struct ReduceMax {
    static __device__ float identity() { return kNegInf; }
    static __device__ float combine(float a, float b) { return fmaxf(a, b); }
};

} // namespace opworks::detail
