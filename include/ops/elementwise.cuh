#pragma once

#include <cstdint>

#include "../core/cuda_utils.cuh"

namespace opworks::detail {

// Packed variadic elementwise: a single variadic kernel covers any arity;
// variadic kernel covers any arity; instantiating pack_size = 4 / 1 gives the
// vectorized path and the scalar fallback. The n % pack_size scalar tail is
// handled by the first threads of the same kernel.
template <int pack_size>
struct alignas(sizeof(float) * pack_size) FloatPack {
    float elem[pack_size];
};

// Same pack type for every input; mentioning T makes the kernel signature
// depend on In... so the variadic expansion has a pack to expand over.
template <typename T, int pack_size>
using PackOf = FloatPack<pack_size>;

template <int pack_size, typename Op, typename... In>
__global__ void map_kernel(Op op, int n_pack, FloatPack<pack_size>* pack_out, const PackOf<In, pack_size>*... pack_in, int n_tail, float* tail_out, const In*... tail_in) {
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    OPWORKS_GRID_LOOP(i, n_pack) {
        FloatPack<pack_size> r;
#pragma unroll
        for (int j = 0; j < pack_size; ++j) {
            r.elem[j] = op(pack_in[i].elem[j]...);
        }
        pack_out[i] = r;
    }
    if (tid < n_tail) {
        tail_out[tid] = op(tail_in[tid]...);
    }
}

template <int pack_size>
inline bool aligned_for_pack(const float* p) {
    return reinterpret_cast<uintptr_t>(p) % alignof(FloatPack<pack_size>) == 0;
}

template <typename Op, typename... In>
void map_launch(const Op& op, int n, float* out, const In*... in) {
    const int n_pack = n / kPackSize;
    const int tail_offset = n_pack * kPackSize;
    if ((aligned_for_pack<kPackSize>(out) && ... && aligned_for_pack<kPackSize>(in)) && n_pack > 0) {
        launch(map_kernel<kPackSize, Op, In...>, num_blocks_for(n_pack), kThreads, op, n_pack, reinterpret_cast<FloatPack<kPackSize>*>(out), reinterpret_cast<const FloatPack<kPackSize>*>(in)..., n - tail_offset, out + tail_offset, (in + tail_offset)...);
    } else {
        launch(map_kernel<1, Op, In...>, num_blocks_for(n), kThreads, op, n, reinterpret_cast<FloatPack<1>*>(out), reinterpret_cast<const FloatPack<1>*>(in)..., 0, out, in...);
    }
}

}  // namespace opworks::detail

namespace opworks::ops {

// binary map: out[i] = op(a[i], b[i]); kPackSize-wide packed when aligned
template <typename Op>
inline void map(const float* a, const float* b, float* out, int n, const Op& op = {}) {
    detail::map_launch(op, n, out, a, b);
}

// unary map: out[i] = op(in[i])
template <typename Op>
inline void map(const float* in, float* out, int n, const Op& op = {}) {
    detail::map_launch(op, n, out, in);
}

}  // namespace opworks::ops
