#pragma once

#include <cuda_runtime.h>

namespace opworks {

// Block-wide reduce; the result is only valid on thread 0.
template <typename Op>
__device__ float block_reduce(float val) {
  __shared__ float shared[32];
  int lane = threadIdx.x & 31;
  int warp = threadIdx.x >> 5;

  for (int offset = 16; offset > 0; offset >>= 1)
    val = Op::combine(val, __shfl_down_sync(0xffffffffu, val, offset));

  if (lane == 0) shared[warp] = val;
  __syncthreads();

  int nwarps = (blockDim.x + 31) >> 5;
  val = (threadIdx.x < nwarps) ? shared[threadIdx.x] : Op::identity();
  if (warp == 0) {
    for (int offset = 16; offset > 0; offset >>= 1)
      val = Op::combine(val, __shfl_down_sync(0xffffffffu, val, offset));
  }
  return val;
}

// Block-wide reduce + broadcast; the result is valid on all threads.
template <typename Op>
__device__ float block_reduce_all(float val) {
  __shared__ float result;
  float r = block_reduce<Op>(val);
  if (threadIdx.x == 0) result = r;
  __syncthreads();
  return result;
}

namespace detail {

// Internal reduction ops shared by the ops/ skeletons.
struct ReduceSum {
  static __device__ float identity() { return 0.f; }
  static __device__ float combine(float a, float b) { return a + b; }
};
struct ReduceMax {
  static __device__ float identity() { return -INFINITY; }
  static __device__ float combine(float a, float b) { return fmaxf(a, b); }
};

}  // namespace detail

}  // namespace opworks
