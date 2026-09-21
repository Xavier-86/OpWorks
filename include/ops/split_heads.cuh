#pragma once

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// out[h, out_offset + t, d] = in[t, h * head_dim + d]; out rows are strided
// (KV cache capacity stride), in rows are packed heads * head_dim.
static __global__ void split_heads_kernel(const float *in, float *out, int rows, int heads, int head_dim,
                                          int out_row_stride, int out_row_offset) {
    OPWORKS_GRID_LOOP(i, static_cast<int64_t>(rows) * heads * head_dim) {
        int d = static_cast<int>(i % head_dim);
        int h = static_cast<int>(i / head_dim) % heads;
        int t = static_cast<int>(i / (static_cast<int64_t>(head_dim) * heads));
        out[(static_cast<int64_t>(h) * out_row_stride + out_row_offset + t) * head_dim + d] =
            in[(static_cast<int64_t>(t) * heads + h) * head_dim + d];
    }
}

// out[t, h * head_dim + d] = in[h, t, d] with strided input rows.
static __global__ void merge_heads_kernel(const float *in, float *out, int rows, int heads, int head_dim,
                                          int in_row_stride, int in_row_offset) {
    OPWORKS_GRID_LOOP(i, static_cast<int64_t>(rows) * heads * head_dim) {
        int d = static_cast<int>(i % head_dim);
        int h = static_cast<int>(i / head_dim) % heads;
        int t = static_cast<int>(i / (static_cast<int64_t>(head_dim) * heads));
        out[(static_cast<int64_t>(t) * heads + h) * head_dim + d] =
            in[(static_cast<int64_t>(h) * in_row_stride + in_row_offset + t) * head_dim + d];
    }
}

} // namespace opworks::detail

namespace opworks::ops {

// Split packed rows [rows, heads * head_dim] into per-head slabs
// [heads, out_row_stride, head_dim], writing at row offset out_row_offset
// (used to append into KV cache; out_row_stride is the cache capacity).
inline void split_heads(const float *in, float *out, int rows, int heads, int head_dim, int out_row_stride,
                        int out_row_offset = 0, cudaStream_t stream = nullptr) {
    detail::validate_size(rows);
    if (rows == 0)
        return;
    detail::require(heads > 0 && head_dim > 0, "split_heads requires positive head shape");
    detail::require(out_row_stride >= out_row_offset + rows, "split_heads exceeds the output row stride");
    detail::require(in && out, "split_heads requires non-null pointers");
    int64_t n = static_cast<int64_t>(rows) * heads * head_dim;
    detail::require(n <= std::numeric_limits<int>::max(), "split_heads input exceeds INT_MAX elements");
    launch_async(stream, detail::split_heads_kernel, num_blocks_for(static_cast<int>(n)), kThreads, in, out, rows,
                 heads, head_dim, out_row_stride, out_row_offset);
}

// Inverse of split_heads: gather per-head slabs back into packed rows.
inline void merge_heads(const float *in, float *out, int rows, int heads, int head_dim, int in_row_stride,
                        int in_row_offset = 0, cudaStream_t stream = nullptr) {
    detail::validate_size(rows);
    if (rows == 0)
        return;
    detail::require(heads > 0 && head_dim > 0, "merge_heads requires positive head shape");
    detail::require(in_row_stride >= in_row_offset + rows, "merge_heads exceeds the input row stride");
    detail::require(in && out, "merge_heads requires non-null pointers");
    int64_t n = static_cast<int64_t>(rows) * heads * head_dim;
    detail::require(n <= std::numeric_limits<int>::max(), "merge_heads input exceeds INT_MAX elements");
    launch_async(stream, detail::merge_heads_kernel, num_blocks_for(static_cast<int>(n)), kThreads, in, out, rows,
                 heads, head_dim, in_row_stride, in_row_offset);
}

} // namespace opworks::ops
