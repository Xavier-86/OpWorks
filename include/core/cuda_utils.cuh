#pragma once

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

#define OPWORKS_CUDA_CHECK(expr)                                             \
  do {                                                                       \
    cudaError_t err_ = (expr);                                               \
    if (err_ != cudaSuccess) {                                               \
      std::fprintf(stderr, "[OpWorks] CUDA error %s at %s:%d: %s\n", #expr,  \
                   __FILE__, __LINE__, cudaGetErrorString(err_));            \
      std::abort();                                                          \
    }                                                                        \
  } while (0)

namespace opworks {

constexpr int kThreads = 256;
constexpr int kNumWaves = 32;  // resident blocks per SM for latency hiding

inline int blocks_for(int n, int threads = kThreads) {
  return (n + threads - 1) / threads;
}

// Adaptive grid size (oneflow-style): enough blocks to cover n, capped at
// kNumWaves waves of resident blocks so huge inputs do not flood the
// scheduler. Device attributes are queried once and cached.
inline int num_blocks_for(int n) {
  static const int max_blocks = [] {
    int dev = 0, sm_count = 0, tpm = 0;
    OPWORKS_CUDA_CHECK(cudaGetDevice(&dev));
    OPWORKS_CUDA_CHECK(
        cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
    OPWORKS_CUDA_CHECK(
        cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
    int per_sm = tpm / kThreads;
    return sm_count * per_sm * kNumWaves;
  }();
  int need = blocks_for(n);
  return need < max_blocks ? need : max_blocks;
}

// Stride loops for custom kernels (oneflow CUDA_1D_KERNEL_LOOP style):
//   OPWORKS_GRID_LOOP(i, n)  { acc += in[i]; }   // whole grid covers [0, n)
//   OPWORKS_BLOCK_LOOP(j, n) { ... }             // one block covers [0, n)
#define OPWORKS_GRID_LOOP(i, n)                                \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < (n); \
       i += gridDim.x * blockDim.x)

#define OPWORKS_BLOCK_LOOP(i, n) \
  for (int i = threadIdx.x; i < (n); i += blockDim.x)

// Launch any kernel with error checking and a sync afterwards; grid/block
// accept dim3 or plain ints. The same boilerplate the builders use internally.
template <typename... Params, typename... Args>
inline void launch(void (*kernel)(Params...), dim3 grid, dim3 block,
                   Args... args) {
  kernel<<<grid, block>>>(args...);
  OPWORKS_CUDA_CHECK(cudaGetLastError());
  OPWORKS_CUDA_CHECK(cudaDeviceSynchronize());
}

}  // namespace opworks
