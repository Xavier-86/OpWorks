#pragma once

#include <cstdint>

#include "../core/cuda_utils.cuh"
#include "../core/device_buffer.cuh"

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

class ElementwiseBuilder {
 public:
  // binary: out[i] = op(a[i], b[i])
  ElementwiseBuilder(const DeviceBuffer& a, const DeviceBuffer& b)
      : a_(a.data()), b_(b.data()), n_(a.size()) {}
  // unary: out[i] = op(in[i])
  explicit ElementwiseBuilder(const DeviceBuffer& a)
      : a_(a.data()), n_(a.size()) {}

  // op may carry runtime state (e.g. ScaleAdd{alpha}); it is copied to the
  // kernel by value. Stateless functors can keep calling apply<Op>().
  template <typename Op>
  DeviceBuffer apply(const Op& op = Op{}) {
    DeviceBuffer out(n_);
    bool packable = reinterpret_cast<uintptr_t>(a_) % 16 == 0 &&
                    (!b_ || reinterpret_cast<uintptr_t>(b_) % 16 == 0);
    int n4 = n_ / 4;
    if constexpr (Op::kArity == 2) {
      if (packable && n4 > 0) {
        launch(elementwise_binary_kernel4<Op>, num_blocks_for(n4), kThreads,
               op, reinterpret_cast<const float4*>(a_),
               reinterpret_cast<const float4*>(b_),
               reinterpret_cast<float4*>(out.data()), n4, a_, b_, out.data(),
               n_);
      } else {
        launch(elementwise_binary_kernel<Op>, num_blocks_for(n_), kThreads, op,
               a_, b_, out.data(), n_);
      }
    } else {
      if (packable && n4 > 0) {
        launch(elementwise_unary_kernel4<Op>, num_blocks_for(n4), kThreads, op,
               reinterpret_cast<const float4*>(a_),
               reinterpret_cast<float4*>(out.data()), n4, a_, out.data(), n_);
      } else {
        launch(elementwise_unary_kernel<Op>, num_blocks_for(n_), kThreads, op,
               a_, out.data(), n_);
      }
    }
    return out;
  }

 private:
  const float* a_ = nullptr;
  const float* b_ = nullptr;
  int n_ = 0;
};

}  // namespace opworks
