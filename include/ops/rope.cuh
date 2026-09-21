#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// Rotary position embedding, HF rotate_half convention:
// pair (i, i + D/2), out_lo = a*cos - b*sin, out_hi = b*cos + a*sin.
// x is [rows, heads * head_dim]; absolute position of row t is pos_start + t.
static __global__ void rope_kernel(float *x, const float *inv_freq, int rows, int heads, int head_dim,
                                   int pos_start) {
    const int half = head_dim / 2;
    const int row_width = heads * head_dim;
    OPWORKS_GRID_LOOP(i, static_cast<int64_t>(rows) * heads * half) {
        int pair = static_cast<int>(i % half);
        int head = static_cast<int>(i / half) % heads;
        int row = static_cast<int>(i / (static_cast<int64_t>(half) * heads));
        float freq = (pos_start + row) * inv_freq[pair];
        float c = cosf(freq), s = sinf(freq);
        float *p = x + static_cast<int64_t>(row) * row_width + head * head_dim + pair;
        float a = p[0], b = p[half];
        p[0] = a * c - b * s;
        p[half] = b * c + a * s;
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// In-place RoPE on x[rows, heads * head_dim] at absolute positions
// [pos_start, pos_start + rows). inv_freq holds head_dim/2 precomputed
// frequencies (llama3 scaling applied on the host).
inline void rope(float *x, const float *inv_freq, int rows, int heads, int head_dim, int pos_start,
                 cudaStream_t stream = nullptr) {
    detail::validate_size(rows);
    if (rows == 0)
        return;
    detail::require(heads > 0 && head_dim > 0 && head_dim % 2 == 0, "rope requires even positive head_dim");
    detail::require(pos_start >= 0, "rope position must be nonnegative");
    detail::require(x && inv_freq, "rope requires non-null pointers");
    int64_t n = static_cast<int64_t>(rows) * heads * (head_dim / 2);
    detail::require(n <= std::numeric_limits<int>::max(), "rope input exceeds the supported INT_MAX pairs");
    launch_async(stream, detail::rope_kernel, num_blocks_for(static_cast<int>(n)), kThreads, x, inv_freq, rows,
                 heads, head_dim, pos_start);
}

} // namespace opworks::ops
