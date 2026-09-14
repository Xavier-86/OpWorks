#pragma once

#include "../core/block_reduce.cuh"
#include "../core/cuda_utils.cuh"
#include "../core/device_buffer.cuh"

namespace opworks {

template <typename Op>
__global__ void reduce_partial_kernel(const float* in, float* partial, int n) {
  float acc = Op::identity();
  OPWORKS_GRID_LOOP(i, n) { acc = Op::combine(acc, in[i]); }
  acc = block_reduce<Op>(acc);
  if (threadIdx.x == 0) partial[blockIdx.x] = acc;
}

template <typename Op>
__global__ void reduce_final_kernel(const float* partial, float* out,
                                    int num_blocks) {
  float acc = Op::identity();
  OPWORKS_BLOCK_LOOP(i, num_blocks) { acc = Op::combine(acc, partial[i]); }
  acc = block_reduce<Op>(acc);
  if (threadIdx.x == 0) out[0] = acc;
}

class ReductionBuilder {
 public:
  explicit ReductionBuilder(const DeviceBuffer& in)
      : in_(in.data()), n_(in.size()) {}

  template <typename Op>
  DeviceBuffer apply() {
    int blocks = blocks_for(n_);
    if (blocks > kMaxBlocks) blocks = kMaxBlocks;
    DeviceBuffer partial(blocks);
    DeviceBuffer out(1);
    launch(reduce_partial_kernel<Op>, blocks, kThreads, in_, partial.data(),
           n_);
    launch(reduce_final_kernel<Op>, 1, kThreads, partial.data(), out.data(),
           blocks);
    return out;
  }

 private:
  static constexpr int kMaxBlocks = 1024;
  const float* in_ = nullptr;
  int n_ = 0;
};

}  // namespace opworks
