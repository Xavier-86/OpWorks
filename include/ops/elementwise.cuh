#pragma once

#include <cstdint>

#include "../core/cuda_utils.cuh"

namespace opworks {

// Scalar kernels; also the fallback when inputs are not 16B-aligned.
template <typename Op>
__global__ void elementwise_binary_kernel(Op op, const float* a, const float* b,
                                          float* out, int n) {
  OPWORKS_GRID_LOOP(i, n) { out[i] = op(a[i], b[i]); }
}

template <typename Op>
__global__ void elementwise_unary_kernel(Op op, const float* in, float* out,
                                         int n) {
  OPWORKS_GRID_LOOP(i, n) { out[i] = op(in[i]); }
}

// float4-packed kernels (oneflow-style vectorization): one thread handles a
// 4-element pack; the n % 4 scalar tail is handled by the first threads.
template <typename Op>
__global__ void elementwise_binary_kernel4(Op op, const float4* a,
                                           const float4* b, float4* out,
                                           int n4, const float* at,
                                           const float* bt, float* outt,
                                           int n) {
  OPWORKS_GRID_LOOP(i, n4) {
    float4 x = a[i], y = b[i], r;
    r.x = op(x.x, y.x);
    r.y = op(x.y, y.y);
    r.z = op(x.z, y.z);
    r.w = op(x.w, y.w);
    out[i] = r;
  }
  int t = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) outt[t] = op(at[t], bt[t]);
}

template <typename Op>
__global__ void elementwise_unary_kernel4(Op op, const float4* in, float4* out,
                                          int n4, const float* intail,
                                          float* outtail, int n) {
  OPWORKS_GRID_LOOP(i, n4) {
    float4 x = in[i], r;
    r.x = op(x.x);
    r.y = op(x.y);
    r.z = op(x.z);
    r.w = op(x.w);
    out[i] = r;
  }
  int t = n4 * 4 + blockIdx.x * blockDim.x + threadIdx.x;
  if (t < n) outtail[t] = op(intail[t]);
}

namespace ops {

// binary map: out[i] = op(a[i], b[i]); float4-vectorized when 16B-aligned
template <typename Op>
inline void map(const float* a, const float* b, float* out, int n,
                const Op& op = {}) {
  bool packable = reinterpret_cast<uintptr_t>(a) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(b) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out) % 16 == 0;
  int n4 = n / 4;
  if (packable && n4 > 0) {
    launch(elementwise_binary_kernel4<Op>, num_blocks_for(n4), kThreads, op,
           reinterpret_cast<const float4*>(a),
           reinterpret_cast<const float4*>(b),
           reinterpret_cast<float4*>(out), n4, a, b, out, n);
  } else {
    launch(elementwise_binary_kernel<Op>, num_blocks_for(n), kThreads, op, a,
           b, out, n);
  }
}

// unary map: out[i] = op(in[i])
template <typename Op>
inline void map(const float* in, float* out, int n, const Op& op = {}) {
  bool packable = reinterpret_cast<uintptr_t>(in) % 16 == 0 &&
                  reinterpret_cast<uintptr_t>(out) % 16 == 0;
  int n4 = n / 4;
  if (packable && n4 > 0) {
    launch(elementwise_unary_kernel4<Op>, num_blocks_for(n4), kThreads, op,
           reinterpret_cast<const float4*>(in), reinterpret_cast<float4*>(out),
           n4, in, out, n);
  } else {
    launch(elementwise_unary_kernel<Op>, num_blocks_for(n), kThreads, op, in,
           out, n);
  }
}

}  // namespace ops

}  // namespace opworks
