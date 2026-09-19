#pragma once

#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <unordered_map>
#include <cuda_runtime.h>

#include "config.cuh"

#define OPWORKS_CUDA_CHECK(expr)                                                                                       \
    do {                                                                                                               \
        cudaError_t err_ = (expr);                                                                                     \
        if (err_ != cudaSuccess) {                                                                                     \
            std::fprintf(stderr, "[OpWorks] CUDA error %s at %s:%d: %s\n", #expr, __FILE__, __LINE__,                  \
                         cudaGetErrorString(err_));                                                                    \
            std::abort();                                                                                              \
        }                                                                                                              \
    } while (0)

namespace opworks {

// kThreads / kNumWaves and the other tuning knobs live in core/config.cuh.

namespace detail {

inline void require(bool condition, const char *message) {
    if (!condition)
        throw std::invalid_argument(message);
}

inline void validate_size(int n) {
    require(n >= 0, "size must be nonnegative");
}

inline void validate_matrix(int rows, int cols) {
    validate_size(rows);
    validate_size(cols);
    require(static_cast<int64_t>(rows) * cols <= std::numeric_limits<int>::max(),
            "matrix exceeds the supported INT_MAX elements");
}

inline int blocks_for(int n, int threads = kThreads) {
    validate_size(n);
    require(threads > 0, "block size must be positive");
    return n / threads + (n % threads != 0);
}

} // namespace detail

// Adaptive grid size: enough blocks to cover n, capped at
// kNumWaves waves of resident blocks so huge inputs do not flood the
// scheduler. Cache separately for each current device and host thread.
inline int num_blocks_for(int n) {
    detail::validate_size(n);
    if (n == 0)
        return 0;
    int dev = 0;
    OPWORKS_CUDA_CHECK(cudaGetDevice(&dev));
    static thread_local std::unordered_map<int, int> cache;
    auto found = cache.find(dev);
    if (found == cache.end()) {
        int sm_count = 0, tpm = 0;
        OPWORKS_CUDA_CHECK(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, dev));
        OPWORKS_CUDA_CHECK(cudaDeviceGetAttribute(&tpm, cudaDevAttrMaxThreadsPerMultiProcessor, dev));
        int per_sm = tpm / kThreads;
        found = cache.emplace(dev, sm_count * per_sm * kNumWaves).first;
    }
    const int max_blocks = found->second;
    int need = detail::blocks_for(n);
    return need < max_blocks ? need : max_blocks;
}

// Stride loops for custom kernels (CUDA_1D_KERNEL_LOOP style):
//   OPWORKS_GRID_LOOP(i, n)  { acc += in[i]; }   // whole grid covers [0, n)
//   OPWORKS_BLOCK_LOOP(j, n) { ... }             // one block covers [0, n)
#define OPWORKS_GRID_LOOP(i, n)                                                                                        \
    for (int64_t i = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < (n);                             \
         i += static_cast<int64_t>(gridDim.x) * blockDim.x)

#define OPWORKS_BLOCK_LOOP(i, n) for (int64_t i = threadIdx.x; i < (n); i += blockDim.x)

// Same stride loop over a flattened 2D/3D block: thread linear id and stride
// cover the whole block, not just the x dimension.
#define OPWORKS_BLOCK_LOOP_FLAT(i, n)                                                                                  \
    for (int i = threadIdx.x + threadIdx.y * blockDim.x + threadIdx.z * blockDim.x * blockDim.y; i < (n);              \
         i += blockDim.x * blockDim.y * blockDim.z)

// Enqueue work and check launch errors only. Input/output memory must remain
// alive until stream completion. Execution errors surface at synchronization.
template <typename... Params, typename... Args>
inline void launch_async(cudaStream_t stream, void (*kernel)(Params...), dim3 grid, dim3 block, Args... args) {
    detail::require(grid.x && grid.y && grid.z && block.x && block.y && block.z, "launch dimensions must be positive");
    kernel<<<grid, block, 0, stream>>>(args...);
    OPWORKS_CUDA_CHECK(cudaGetLastError());
}

inline void synchronize(cudaStream_t stream = nullptr) {
    OPWORKS_CUDA_CHECK(cudaStreamSynchronize(stream));
}

template <typename... Params, typename... Args>
inline void launch(void (*kernel)(Params...), dim3 grid, dim3 block, Args... args) {
    launch_async(nullptr, kernel, grid, block, args...);
}

template <typename... Params, typename... Args>
inline void launch_sync(cudaStream_t stream, void (*kernel)(Params...), dim3 grid, dim3 block, Args... args) {
    launch_async(stream, kernel, grid, block, args...);
    synchronize(stream);
}

template <typename... Params, typename... Args>
inline void launch_sync(void (*kernel)(Params...), dim3 grid, dim3 block, Args... args) {
    launch_sync(nullptr, kernel, grid, block, args...);
}

} // namespace opworks
