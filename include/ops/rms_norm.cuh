#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// One block per row: out = x * rsqrt(mean(x^2) + eps) * weight.
static __global__ void rms_norm_kernel(const float *x, const float *weight, float *out, int cols, float eps) {
    const float *row_in = x + static_cast<int64_t>(blockIdx.x) * cols;
    float *row_out = out + static_cast<int64_t>(blockIdx.x) * cols;

    float local = ReduceSum::identity();
    OPWORKS_BLOCK_LOOP(j, cols) {
        float v = row_in[j];
        local += v * v;
    }
    float sum_sq = block_reduce_all<ReduceSum>(local);
    float inv_rms = rsqrtf(sum_sq / cols + eps);

    OPWORKS_BLOCK_LOOP(j, cols) {
        row_out[j] = row_in[j] * inv_rms * weight[j];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// RMSNorm over the last dim of a rows x cols matrix (no mean subtraction).
inline void rms_norm(const float *x, const float *weight, float *out, int rows, int cols, float eps,
                     cudaStream_t stream = nullptr) {
    detail::validate_matrix(rows, cols);
    if (rows == 0)
        return;
    detail::require(cols > 0, "rms_norm requires positive columns for nonempty rows");
    detail::require(x && weight && out, "rms_norm requires non-null pointers");
    detail::require(eps > 0.f, "rms_norm epsilon must be positive");
    launch_async(stream, detail::rms_norm_kernel, rows, kThreads, x, weight, out, cols, eps);
}

} // namespace opworks::ops
