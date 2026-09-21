#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

static __global__ void swiglu_kernel(const float *gate, const float *up, float *out, int64_t n) {
    OPWORKS_GRID_LOOP(i, n) {
        float g = gate[i];
        out[i] = g / (1.f + expf(-g)) * up[i];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// out = silu(gate) * up, elementwise over n values.
inline void swiglu(const float *gate, const float *up, float *out, int64_t n, cudaStream_t stream = nullptr) {
    detail::require(n >= 0, "swiglu size must be nonnegative");
    if (n == 0)
        return;
    detail::require(gate && up && out, "swiglu requires non-null pointers");
    detail::require(n <= std::numeric_limits<int>::max(), "swiglu input exceeds the supported INT_MAX elements");
    launch_async(stream, detail::swiglu_kernel, num_blocks_for(static_cast<int>(n)), kThreads, gate, up, out, n);
}

} // namespace opworks::ops
