#pragma once

#include "../container/device_buffer.cuh"
#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"

namespace opworks::detail {

template <typename Op>
__global__ void reduce_partial_kernel(const float* in, float* partial, int n) {
  float acc = Op::identity();
  OPWORKS_GRID_LOOP(i, n) { acc = Op::combine(acc, in[i]); }
  acc = block_reduce<Op>(acc);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

template <typename Op>
__global__ void reduce_final_kernel(const float* partial, float* out, int num_blocks) {
  float acc = Op::identity();
  OPWORKS_BLOCK_LOOP(i, num_blocks) { acc = Op::combine(acc, partial[i]); }
  acc = block_reduce<Op>(acc);
  if (threadIdx.x == 0) out[0] = acc;
}

}  // namespace opworks::detail

namespace opworks::ops {

// two-pass reduction to a single scalar: out[0] = reduce(in[0..n))
template <typename Op>
inline void reduce(const float* in, float* out, int n) {
  int blocks = blocks_for(n);
  if (blocks > 1024) blocks = 1024;  // final pass is a single block
  DeviceBuffer partial(blocks);
  launch(detail::reduce_partial_kernel<Op>, blocks, kThreads, in, partial.data(), n);
  launch(detail::reduce_final_kernel<Op>, 1, kThreads, partial.data(), out, blocks);
}

}  // namespace opworks::ops
