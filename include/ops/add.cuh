#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

static __global__ void add_kernel(const float *a, const float *b, float *out, int64_t n) {
    OPWORKS_GRID_LOOP(i, n) {
        out[i] = a[i] + b[i];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// out = a + b elementwise; out may alias a or b (same-index elementwise).
inline void add(const float *a, const float *b, float *out, int64_t n, cudaStream_t stream = nullptr) {
    detail::require(n >= 0, "add size must be nonnegative");
    if (n == 0)
        return;
    detail::require(a && b && out, "add requires non-null pointers");
    detail::require(n <= std::numeric_limits<int>::max(), "add input exceeds the supported INT_MAX elements");
    launch_async(stream, detail::add_kernel, num_blocks_for(static_cast<int>(n)), kThreads, a, b, out, n);
}

} // namespace opworks::ops
