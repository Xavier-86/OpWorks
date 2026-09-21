#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// (value, index) block reduction; ties resolve to the lowest index.
__device__ __forceinline__ void argmax_combine(float &va, int &ia, float vb, int ib) {
    if (vb > va || (vb == va && ib < ia)) {
        va = vb;
        ia = ib;
    }
}

// Single-block argmax over n floats. Result written to out[0].
static __global__ void argmax_kernel(const float *x, int n, int *out) {
    float best = kNegInf;
    int best_i = 0x7fffffff;
    OPWORKS_BLOCK_LOOP(i, n) {
        argmax_combine(best, best_i, x[i], i);
    }

    // warp reduce
    for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_down_sync(kFullWarpMask, best, offset);
        int oi = __shfl_down_sync(kFullWarpMask, best_i, offset);
        argmax_combine(best, best_i, ov, oi);
    }
    __shared__ float warp_val[kMaxThreadsPerBlock / kWarpSize];
    __shared__ int warp_idx[kMaxThreadsPerBlock / kWarpSize];
    int lane = threadIdx.x % kWarpSize;
    int warp = threadIdx.x / kWarpSize;
    if (lane == 0) {
        warp_val[warp] = best;
        warp_idx[warp] = best_i;
    }
    __syncthreads();
    if (warp == 0) {
        int nwarps = (blockDim.x + kWarpSize - 1) / kWarpSize;
        best = lane < nwarps ? warp_val[lane] : kNegInf;
        best_i = lane < nwarps ? warp_idx[lane] : 0x7fffffff;
        for (int offset = kWarpSize / 2; offset > 0; offset >>= 1) {
            float ov = __shfl_down_sync(kFullWarpMask, best, offset);
            int oi = __shfl_down_sync(kFullWarpMask, best_i, offset);
            argmax_combine(best, best_i, ov, oi);
        }
        if (lane == 0)
            out[0] = best_i;
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Index of the maximum of x[0..n); ties resolve to the lowest index.
// The result lands in out[0] on device (copy it back explicitly).
inline void argmax(const float *x, int n, int *out, cudaStream_t stream = nullptr) {
    detail::require(n > 0, "argmax requires a nonempty input");
    detail::require(x && out, "argmax requires non-null pointers");
    launch_async(stream, detail::argmax_kernel, 1, kMaxThreadsPerBlock, x, n, out);
}

} // namespace opworks::ops
