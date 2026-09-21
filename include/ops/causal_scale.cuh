#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// scores[rows, cols] holds QK^T for queries at absolute positions
// [cols - rows, cols). Visible entries are scaled; masked ones become -inf.
static __global__ void causal_scale_kernel(float *scores, int rows, int cols, float scale) {
    OPWORKS_GRID_LOOP(i, static_cast<int64_t>(rows) * cols) {
        int row = static_cast<int>(i / cols);
        int col = static_cast<int>(i % cols);
        int qpos = cols - rows + row;
        scores[i] = col <= qpos ? scores[i] * scale : kNegInf;
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Applies 1/sqrt(D)-style scaling and the causal mask in place on a
// [rows, cols] score matrix. Query row r attends to keys 0..(cols-rows+r).
inline void causal_scale(float *scores, int rows, int cols, float scale, cudaStream_t stream = nullptr) {
    detail::validate_matrix(rows, cols);
    if (rows == 0)
        return;
    detail::require(cols >= rows, "causal scores need at least as many keys as queries");
    detail::require(scores && scale > 0.f, "causal_scale requires a buffer and a positive scale");
    int64_t n = static_cast<int64_t>(rows) * cols;
    launch_async(stream, detail::causal_scale_kernel, num_blocks_for(static_cast<int>(n)), kThreads, scores, rows,
                 cols, scale);
}

} // namespace opworks::ops
