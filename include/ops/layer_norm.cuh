#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// static: header-defined non-template kernel needs internal linkage so the
// header can be included from multiple translation units.
static __global__ void layer_norm_kernel(const float *in, float *out, const float *gamma, const float *beta, int cols,
                                         float eps) {
    const float *row_in = in + blockIdx.x * cols;
    float *row_out = out + blockIdx.x * cols;

    float local_sum = ReduceSum::identity();
    OPWORKS_BLOCK_LOOP(j, cols) {
        local_sum += row_in[j];
    }
    float mean = block_reduce_all<ReduceSum>(local_sum) / cols;

    float local_var = ReduceSum::identity();
    OPWORKS_BLOCK_LOOP(j, cols) {
        float d = row_in[j] - mean;
        local_var += d * d;
    }
    float rstd = rsqrtf(block_reduce_all<ReduceSum>(local_var) / cols + eps);

    OPWORKS_BLOCK_LOOP(j, cols) {
        row_out[j] = (row_in[j] - mean) * rstd * gamma[j] + beta[j];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// layer normalization over the last dim of a rows x cols matrix
inline void layer_norm(const float *in, float *out, const float *gamma, const float *beta, int rows, int cols,
                       float eps = kLayerNormEps, cudaStream_t stream = nullptr) {
    detail::validate_matrix(rows, cols);
    detail::require(std::isfinite(eps) && eps > 0, "layer norm epsilon must be finite and positive");
    if (rows == 0)
        return;
    detail::require(cols > 0, "layer norm requires positive columns for nonempty rows");
    detail::require(in && out && gamma && beta, "layer norm requires non-null pointers");
    launch_async(stream, detail::layer_norm_kernel, rows, kThreads, in, out, gamma, beta, cols, eps);
}

} // namespace opworks::ops
