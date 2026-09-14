#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::ops {

__global__ void softmax_rows_kernel(const float* in, float* out, int cols) {
  const float* row_in = in + blockIdx.x * cols;
  float* row_out = out + blockIdx.x * cols;

  float local_max = detail::ReduceMax::identity();
  OPWORKS_BLOCK_LOOP(j, cols) {
    local_max = detail::ReduceMax::combine(local_max, row_in[j]);
  }
  float max_val = block_reduce_all<detail::ReduceMax>(local_max);

  float local_sum = detail::ReduceSum::identity();
  OPWORKS_BLOCK_LOOP(j, cols) { local_sum += expf(row_in[j] - max_val); }
  float sum = block_reduce_all<detail::ReduceSum>(local_sum);

  OPWORKS_BLOCK_LOOP(j, cols) { row_out[j] = expf(row_in[j] - max_val) / sum; }
}

// softmax over the last dim of a rows x cols matrix, one block per row
inline void softmax(const float* in, float* out, int rows, int cols) {
  launch(softmax_rows_kernel, rows, kThreads, in, out, cols);
}

}  // namespace opworks::ops
