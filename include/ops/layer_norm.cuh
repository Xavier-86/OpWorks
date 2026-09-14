#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::ops {

__global__ void layer_norm_kernel(const float* in, float* out,
                                  const float* gamma, const float* beta,
                                  int cols, float eps) {
  const float* row_in = in + blockIdx.x * cols;
  float* row_out = out + blockIdx.x * cols;

  float local_sum = detail::ReduceSum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) { local_sum += row_in[j]; }
  float mean = block_reduce_all<detail::ReduceSum>(local_sum) / cols;

  float local_var = detail::ReduceSum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) {
    float d = row_in[j] - mean;
    local_var += d * d;
  }
  float rstd = rsqrtf(block_reduce_all<detail::ReduceSum>(local_var) / cols + eps);

  OPWORKS_BLOCK_LOOP(j, cols) {
    row_out[j] = (row_in[j] - mean) * rstd * gamma[j] + beta[j];
  }
}

// layer normalization over the last dim of a rows x cols matrix
inline void layer_norm(const float* in, float* out, const float* gamma,
                       const float* beta, int rows, int cols,
                       float eps = 1e-5f) {
  launch(layer_norm_kernel, rows, kThreads, in, out, gamma, beta, cols, eps);
}

}  // namespace opworks::ops
