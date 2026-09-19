#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// static: header-defined non-template kernel needs internal linkage so the
// header can be included from multiple translation units.
static __global__ void softmax_rows_kernel(const float* in, float* out, int cols) {
    const float* row_in = in + blockIdx.x * cols;
    float* row_out = out + blockIdx.x * cols;

    float local_max = ReduceMax::identity();
    OPWORKS_BLOCK_LOOP(j, cols) {
        local_max = ReduceMax::combine(local_max, row_in[j]);
    }
    float max_val = block_reduce_all<ReduceMax>(local_max);

    float local_sum = ReduceSum::identity();
    OPWORKS_BLOCK_LOOP(j, cols) { local_sum += expf(row_in[j] - max_val); }
    float sum = block_reduce_all<ReduceSum>(local_sum);

    OPWORKS_BLOCK_LOOP(j, cols) { row_out[j] = expf(row_in[j] - max_val) / sum; }
}

}  // namespace opworks::detail

namespace opworks::ops {

// softmax over the last dim of a rows x cols matrix, one block per row
inline void softmax(const float* in, float* out, int rows, int cols) {
    launch(detail::softmax_rows_kernel, rows, kThreads, in, out, cols);
}

}  // namespace opworks::ops
