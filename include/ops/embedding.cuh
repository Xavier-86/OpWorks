#pragma once

#include <cuda_bf16.h>

#include "../core/cuda_utils.cuh"
#include "linear.cuh" // detail::to_f32

namespace opworks::detail {

template <typename WT>
static __global__ void embedding_kernel(const WT *table, const int *ids, float *out, int hidden, int n) {
    OPWORKS_GRID_LOOP(i, static_cast<int64_t>(n) * hidden) {
        int row = static_cast<int>(i / hidden);
        out[i] = to_f32(table[static_cast<int64_t>(ids[row]) * hidden + (i % hidden)]);
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// out[t, :] = table[ids[t], :] for a [vocab, hidden] table; fp32 output,
// table may be fp32 or bf16. ids must be in range; callers validate.
template <typename WT>
inline void embedding(const WT *table, const int *ids, float *out, int n, int hidden,
                      cudaStream_t stream = nullptr) {
    detail::validate_size(n);
    detail::validate_size(hidden);
    if (n == 0)
        return;
    detail::require(hidden > 0 && table && ids && out, "embedding requires positive width and non-null pointers");
    launch_async(stream, detail::embedding_kernel<WT>, num_blocks_for(n * hidden), kThreads, table, ids, out,
                 hidden, n);
}

} // namespace opworks::ops
